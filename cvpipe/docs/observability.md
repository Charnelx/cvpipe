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

## Dashboard

cvpipe includes a built-in metrics dashboard for real-time pipeline monitoring. Enable it with a single call before `pipeline.start()`:

```python
from cvpipe import build
from cvpipe.dashboard import enable_dashboard

pipeline = build(config_path, components_dir)
pipeline.validate()
enable_dashboard(pipeline)
pipeline.start()
```

The dashboard provides:

- **Real-time latency charts** (p50, p95, p99 per component)
- **Frame drop tracking** (by reason)
- **Error logs** with full tracebacks
- **FPS monitoring** using exponential moving average, tracking only the terminal component
- **Prometheus endpoint** for integration with monitoring systems
- **WebSocket streaming** for live updates

Access the dashboard at `http://localhost:8881`.

### Configuration

```python
from cvpipe.dashboard import enable_dashboard, DashboardConfig

enable_dashboard(pipeline, DashboardConfig(
    port=8881,
    host="127.0.0.1",
    prometheus=True,
    websocket=True,
    latency_window=500,
    history_duration_minutes=10.0,
    fps_alpha=0.05,  # Smoother FPS
))

### FPS Calculation

The FPS calculator uses **consecutive frame indices** from `ComponentMetricEvent` to determine the frame rate, rather than relying solely on timestamps. This provides more accurate measurements.

**How it works:**
1. The dashboard tracks the **last component in the pipeline** (auto-detected from `pipeline._components[-1]._component_id`)
2. FPS is calculated using the delta between consecutive `frame_idx` values
3. Uses an exponential moving average (EMA) with configurable `alpha` (default: 0.1)

**Staleness Detection:**
- **Warning** (5 seconds): Logs `"Frame stall detected, FPS may be inaccurate"` when no new frames arrive for 5 seconds
- **Reset** (10 seconds): Logs `"FPS reset due to stall"` and resets FPS to 0 when no new frames arrive for 10 seconds

**Frame Skip Handling:**
- If `frame_idx` skips forward (non-consecutive) or goes backwards, FPS resets to 0
- This handles camera stream resets, stream restarts, or other discontinuities

**Recovery:**
- When a new consecutive frame arrives after staleness, FPS tracking automatically resumes

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | HTML dashboard |
| `GET /api/v1/metrics` | Full metrics JSON |
| `GET /api/v1/metrics/latency` | Latency percentiles |
| `GET /api/v1/metrics/drops` | Frame drop counts |
| `GET /api/v1/metrics/errors` | Error statistics |
| `GET /api/v1/metrics/state` | Pipeline state |
| `GET /api/v1/metrics/fps` | Current FPS |
| `GET /api/v1/metrics/history` | Latency time series |
| `GET /metrics` | Prometheus format |
| `WS /ws/metrics` | WebSocket streaming |
| `POST /api/v1/metrics/export` | Export metrics to file |

### Custom Telemetry

Register custom event handlers to track application-specific metrics:

```python
from dataclasses import dataclass
from cvpipe import Event, Component
from cvpipe.dashboard import enable_dashboard

@dataclass(frozen=True)
class GPUMemoryEvent(Event):
    component_id: str
    memory_mb: float
    utilization_pct: float

# In your component
class YOLODetector(Component):
    def process(self, frame):
        # ... inference ...
        self.emit(GPUMemoryEvent(
            component_id=self._component_id,
            memory_mb=torch.cuda.memory_allocated() / 1e6,
            utilization_pct=torch.cuda.utilization(),
        ))

# Register handler
def handle_gpu_event(event):
    return {
        event.component_id: {
            "memory_mb": event.memory_mb,
            "utilization_pct": event.utilization_pct,
        }
    }

pipeline = build(config_path, components_dir)
server = enable_dashboard(pipeline)
pipeline._dashboard_collector.register_custom_event(GPUMemoryEvent, handle_gpu_event)
pipeline.start()
```

### Dependencies

The dashboard requires `fastapi` and `uvicorn`. Install with:

```bash
pip install cvpipe[dashboard]
```

If dependencies are not installed, `enable_dashboard()` logs a warning and returns `None`.

---

## Next Steps

- [API Reference](./api_reference.md) — Complete reference for all classes and methods
- [End-to-End Example](./example_webcam_api.md) — Full annotated detection server with probes
