# Observability

cvpipe provides several mechanisms for monitoring pipeline health and performance.

## Probes

Probes are non-breaking observation hooks called after each component:

```python
from cvpipe import Probe, Frame

class MyProbe(Probe):
    def observe(self, frame: Frame, after_component: str) -> None:
        count = len(frame.meta.get("detections", []))
        print(f"Frame {frame.idx}: {count} detections after {after_component}")
```

Attach a probe to the scheduler:

```python
scheduler.add_probe(MyProbe(), after="tracker")  # after specific component
scheduler.add_probe(MyProbe(), after=None)       # after every component
```

### Best Practices

- Probes run in the streaming thread — keep callbacks under 1ms
- For heavy work (logging to disk, UI updates), enqueue data and process asynchronously

## DiagnosticsProbe

The built-in `DiagnosticsProbe` collects per-frame timing:

```python
from cvpipe import DiagnosticsProbe

probe = DiagnosticsProbe()
scheduler.add_probe(probe, after=None)  # observe after every component
```

After each frame, `frame.meta["diagnostics"]` contains a `FrameDiagnostics` dataclass:

```python
# Access via attribute:
diag = frame.meta["diagnostics"]
print(f"Frame {diag.frame_idx}: {diag.total_ms:.2f}ms total")
for comp in diag.components:
    print(f"  {comp.component_id}: {comp.latency_ms:.2f}ms")

# Or use the summary():
print(diag.summary())
# Example output: "Frame 42 | proposer:5.2ms → embedder:12.8ms | total:45.3ms"
```

## ComponentMetricEvent

Emitted after each component processes a frame:

```python
from cvpipe import ComponentMetricEvent

def handle_metric(event: ComponentMetricEvent) -> None:
    print(f"{event.component_id}: {event.latency_ms:.2f}ms on frame {event.frame_idx}")

event_bus.subscribe(ComponentMetricEvent, handle_metric)
```

## ComponentErrorEvent

Emitted when a component raises an exception:

```python
from cvpipe import ComponentErrorEvent

def handle_error(event: ComponentErrorEvent) -> None:
    print(f"Component {event.component_id} failed on frame {event.frame_idx}")
    print(f"Error: {event.message}")
    print(f"Traceback: {event.traceback}")

event_bus.subscribe(ComponentErrorEvent, handle_error)
```

## PipelineStateEvent

Emitted on pipeline lifecycle transitions:

```python
from cvpipe import PipelineStateEvent

def handle_state(event: PipelineStateEvent) -> None:
    print(f"Pipeline state: {event.state}")
    if event.detail:
        print(f"Detail: {event.detail}")

event_bus.subscribe(PipelineStateEvent, handle_state)
```

States: `starting` → `running` → `stopping` → `stopped` (or `error`). The `reset` state is emitted by `pipeline.reset()`.

## FrameDroppedEvent

Emitted when a frame is dropped:

```python
from cvpipe import FrameDroppedEvent

def handle_drop(event: FrameDroppedEvent) -> None:
    print(f"Frame {event.frame_idx} dropped: {event.reason}")

event_bus.subscribe(FrameDroppedEvent, handle_drop)
```

Reasons:
- `source_stall` — FrameSource returned None
- `backpressure` — Previous frame still processing
- `component_error` — Component raised an exception

## AsyncQueueBridge

`AsyncQueueBridge` bridges the non-async `ResultBus` to an asyncio event loop. Use it when your application uses async frameworks (FastAPI, aiohttp):

```python
import asyncio
from cvpipe import AsyncQueueBridge

loop = asyncio.get_event_loop()
bridge = AsyncQueueBridge(loop=loop, maxsize=8)

# Subscribe bridge to ResultBus
result_bus.subscribe(bridge.put)
result_bus.start()

# Start async consumer
await bridge.start_consumer(my_async_handler)

async def my_async_handler(result: FrameResult) -> None:
    await websocket.send_bytes(result.jpeg_bytes)

# At shutdown:
await bridge.stop()
```

### Key Features

- **Thread-safe**: `put()` can be called from any thread
- **Lossy**: When the queue is full, the oldest item is dropped (matches ResultBus semantics)
- **Async consumer**: Handler runs in the event loop, enabling `await` calls

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `loop` | asyncio.AbstractEventLoop | Event loop for the consumer |
| `maxsize` | int | Queue depth before dropping oldest (default: 8) |

### Methods

| Method | Description |
|--------|-------------|
| `put(item)` | Enqueue item from any thread |
| `start_consumer(handler)` | Start async consumer coroutine |
| `stop()` | Cancel consumer task |
| `qsize` | Current queue depth (property) |

## → Next Steps

- [API Reference](./api_reference.md) — Complete API documentation
