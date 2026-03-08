# Observability

cvpipe gives you three complementary ways to look inside a running pipeline: **probes**
for per-frame inspection, **events** for pipeline health, and `AsyncQueueBridge` for
wiring the result stream to an async application.

---

## Probes

A probe is a non-breaking observation hook. The Scheduler calls `observe()` after a
nominated component finishes processing a frame. The probe sees the full `Frame` state at
that point — all slots and meta written so far — but must not modify it.

Probes run in the **streaming thread**. Keep `observe()` fast — under 0.5 ms. For
anything heavier, push data to a queue and process it on a background thread.

Probes are attached to `pipeline._scheduler` **after `pipeline.start()`**:

```python
pipeline.start()
scheduler = pipeline._scheduler

scheduler.add_probe(my_probe, after="detector")  # fires after the detector component
scheduler.add_probe(my_probe, after=None)         # fires after every component
```

### Basic probe

```python
from cvpipe import Probe, Frame

class DetectionCountProbe(Probe):
    def observe(self, frame: Frame, after_component: str) -> None:
        count = frame.meta.get("detection_count", 0)
        if count > 0:
            print(f"[frame {frame.idx}] {count} detections after '{after_component}'")
```

### Production-grade latency probe

In a real deployment you track rolling latency per component and expose it via a metrics
endpoint. The probe pushes data to an internal dict that can be safely read from any
thread:

```python
from collections import deque
from threading import Lock
from cvpipe import Probe, Frame

class LatencyProbe(Probe):
    """
    Tracks rolling p50/p95/p99 latency per component.
    Safe to read from any thread (HTTP handler, Prometheus exporter, etc.)
    """

    def __init__(self, window: int = 200):
        self._window  = window
        self._lock    = Lock()
        self._samples: dict[str, deque] = {}
        self._last_ts: dict[str, float] = {}

    def observe(self, frame: Frame, after_component: str) -> None:
        import time
        now = time.monotonic()
        if after_component in self._last_ts:
            latency_ms = (now - self._last_ts[after_component]) * 1000
            with self._lock:
                buf = self._samples.setdefault(
                    after_component, deque(maxlen=self._window)
                )
                buf.append(latency_ms)
        self._last_ts[after_component] = now

    def percentiles(self, component_id: str) -> dict:
        """Call from any thread — e.g. a /metrics HTTP handler."""
        with self._lock:
            samples = sorted(self._samples.get(component_id, []))
        if not samples:
            return {}
        n = len(samples)
        return {
            "p50": samples[int(n * 0.50)],
            "p95": samples[int(n * 0.95)],
            "p99": samples[int(n * 0.99)],
            "count": n,
        }
```

Attach and expose via FastAPI:

```python
from fastapi import FastAPI

app       = FastAPI()
lat_probe = LatencyProbe(window=300)

pipeline.start()
pipeline._scheduler.add_probe(lat_probe, after=None)   # profile all components

@app.get("/metrics")
def metrics():
    return {
        cid: lat_probe.percentiles(cid)
        for cid in ["preprocessor", "detector", "tracker"]
    }
```

---

## DiagnosticsProbe

The built-in `DiagnosticsProbe` is the fastest way to find a bottleneck. It writes a
complete timing breakdown to `frame.meta["diagnostics"]` after every component:

```python
from cvpipe import DiagnosticsProbe

pipeline.start()
diag_probe = DiagnosticsProbe()
pipeline._scheduler.add_probe(diag_probe, after=None)
```

Read the results in a `ResultBus` subscriber or any other per-frame handler:

```python
def on_result(result):
    diag = result.meta.get("diagnostics")
    if diag is None:
        return

    # One-line summary
    print(diag.summary())
    # → "Frame 150 | preprocessor:1.2ms → detector:18.4ms → tracker:0.9ms | total:20.5ms"

    # Per-component detail — flag anything over 15 ms
    for comp in diag.components:
        if comp.latency_ms > 15:
            print(f"  SLOW: {comp.component_id} took {comp.latency_ms:.1f}ms")

pipeline.result_bus.subscribe(on_result)
```

Note: subscribe to `result_bus` **before** `pipeline.start()`. Attach probes **after**
`pipeline.start()` (the scheduler does not exist until then).

---

## Pipeline events

Subscribe on `pipeline.event_bus` **before calling `start()`**:

```python
from cvpipe import PipelineStateEvent, ComponentErrorEvent, FrameDroppedEvent

pipeline.event_bus.subscribe(PipelineStateEvent,
    lambda e: logger.info("pipeline: %s", e.state))

pipeline.event_bus.subscribe(ComponentErrorEvent,
    lambda e: logger.error("[frame %d] %s: %s", e.frame_idx, e.component_id, e.message))

pipeline.event_bus.subscribe(FrameDroppedEvent,
    lambda e: drop_counts.update([e.reason]))

pipeline.start()
```

### PipelineStateEvent

Lifecycle transitions: `starting` → `running` → `stopping` → `stopped`. Also `reset`
after `pipeline.reset()` completes, and `error` if any component's `setup()` raises.

### ComponentErrorEvent

Emitted when a component raises during `process()`. The frame is dropped; the pipeline
continues. Fields: `component_id`, `message`, `traceback`, `frame_idx`.

### ComponentMetricEvent

Emitted after every component on every frame. Use it for live latency dashboards:

```python
from cvpipe import ComponentMetricEvent
from collections import deque

latencies: dict[str, deque] = {}

pipeline.event_bus.subscribe(ComponentMetricEvent, lambda e: (
    latencies.setdefault(e.component_id, deque(maxlen=100)).append(e.latency_ms)
))
```

### FrameDroppedEvent

Emitted when the Scheduler drops a frame. Drop reasons:
- `source_stall` — `source.next()` returned `None`
- `backpressure` — previous frame still in flight when the next arrived
- `component_error` — a component raised an exception

High `backpressure` counts mean the pipeline can't keep up with the camera. Check
`ComponentMetricEvent` latencies to find the slow component.

---

## AsyncQueueBridge

The `AsyncQueueBridge` bridges the synchronous `ResultBus` (streaming thread) into an
asyncio event loop (FastAPI, aiohttp). Wire it before `pipeline.start()`; attach probes
after `pipeline.start()`:

```python
import asyncio
from cvpipe import AsyncQueueBridge, FrameResult

loop   = asyncio.get_event_loop()
bridge = AsyncQueueBridge(loop=loop, maxsize=8)

# Wire before start() — subscribe is registration, not consumption
pipeline.result_bus.subscribe(bridge.put)

pipeline.start()

# Attach probes after start()
pipeline._scheduler.add_probe(DiagnosticsProbe(), after=None)

# Start async consumer — must be called from inside the event loop
await bridge.start_consumer(handle_result)

async def handle_result(result: FrameResult) -> None:
    await websocket.send_bytes(result.jpeg_bytes)
```

When the queue fills up (consumer too slow), the **oldest** item is dropped — not the
newest. Your client always gets the most recent frame.

### FastAPI wiring

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

bridge: AsyncQueueBridge | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global bridge
    loop = asyncio.get_event_loop()

    bridge = AsyncQueueBridge(loop=loop, maxsize=8)
    pipeline.result_bus.subscribe(bridge.put)          # before start()

    pipeline.event_bus.subscribe(ComponentErrorEvent,
        lambda e: logger.error("%s: %s", e.component_id, e.message))

    pipeline.start()
    pipeline._scheduler.add_probe(DiagnosticsProbe(), after=None)  # after start()

    await bridge.start_consumer(broadcast_result)

    yield

    await bridge.stop()
    loop.run_in_executor(None, pipeline.stop)

app = FastAPI(lifespan=lifespan)
```

---

## Recommendations

**Subscribe to events before `start()`, attach probes after `start()`.** These two APIs
have opposite ordering requirements — events before, probes after.

**Keep `observe()` under 0.5 ms.** It runs in the streaming thread. If you need to write
to a database or send an HTTP request, push data to a `queue.Queue` and consume it on a
background thread.

**Use `DiagnosticsProbe` during development, `LatencyProbe` in production.** The
`DiagnosticsProbe` writes to `frame.meta` on every frame — useful for debugging, not
suitable for a high-throughput deployment. A `LatencyProbe` backed by thread-safe deques
is more appropriate for a running server.

**`backpressure` drops mean the pipeline is slower than the camera.** Check
`ComponentMetricEvent` p99 latencies. The slow component will stand out immediately.

---

## Next Steps

- [API Reference](./api_reference.md) — Complete reference for all classes and methods
- [End-to-End Example](./example_webcam_api.md) — Full annotated detection server with probes
