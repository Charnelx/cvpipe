# API Reference

Complete reference for all public classes, functions, and data structures in cvpipe.

---

## build()

Top-level factory function. The standard way to assemble a pipeline from YAML.

```python
from cvpipe import build
from pathlib import Path

pipeline = build(
    config_path=Path("pipeline.yaml"),
    components_dir=Path("myapp/"),
)
```

**Signature:** `build(config_path: Path, components_dir: Path) -> Pipeline`

| Parameter | Type | Description |
|-----------|------|-------------|
| `config_path` | Path | Path to the `pipeline.yaml` file |
| `components_dir` | Path | Root directory for component and source discovery. All subdirs with `__init__.py` are scanned |

Discovers all `Component` and `FrameSource` subclasses from `components_dir`, reads the
YAML topology, instantiates each component with the `config` dict from YAML, constructs
and returns an unvalidated `Pipeline`. Call `pipeline.validate()` then `pipeline.start()`
after.

---

## Frame

Mutable per-frame workspace. One instance travels through every component in the pipeline.

```python
from cvpipe import Frame

frame = Frame(idx=42, ts=time.monotonic())
frame.slots["boxes_xyxy"] = tensor
frame.meta["detection_count"] = 3
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `idx` | `int` | Monotonically increasing frame counter. Dropped frames are not counted |
| `ts` | `float` | `time.monotonic()` timestamp at capture |
| `slots` | `dict[str, Any]` | Named tensor slots — `torch.Tensor`, GPU or CPU |
| `meta` | `dict[str, Any]` | CPU-side metadata — scalars, strings, lists, dicts |

**Special `frame.meta` keys (read by the Scheduler after all components run):**

| Key | Type | Written by | Description |
|-----|------|-----------|-------------|
| `"jpeg_bytes"` | bytes | Terminal component | JPEG-encoded frame for streaming |
| `"detections"` | list | Terminal component | List of detection dicts |
| `"result_meta"` | dict | Terminal component | Merged into `FrameResult.meta` |
| `"diagnostics"` | FrameDiagnostics | DiagnosticsProbe | Per-component timing (if probe attached) |

---

## SlotSchema

Descriptor for one named data slot.

```python
from cvpipe import SlotSchema
import torch

SlotSchema(
    name="boxes_xyxy",
    dtype=torch.float32,
    shape=(None, 4),          # None = variable-length dimension
    device="gpu",
    coord_system="xyxy",
    description="Detected bounding boxes, absolute pixels, xyxy format",
)
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | str | ✓ | Slot identifier. Use `noun_coordsystem` format. Must be a valid Python identifier |
| `dtype` | torch.dtype / type / None | ✓ | `torch.float32` etc. for tensor slots; Python type or None for meta slots |
| `shape` | tuple | | Expected shape. `None` for variable-length dimensions |
| `device` | str | | `"gpu"`, `"cpu"`, or `"any"` |
| `coord_system` | str | | Coordinate tag — validated against consumers |
| `description` | str | | Human-readable description |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `is_tensor_slot()` | bool | True if dtype is a `torch.dtype` |
| `is_meta_slot()` | bool | True if dtype is a Python type or None |
| `compatible_with(other)` | list[str] | Returns list of compatibility errors (empty = compatible) |

---

## Component

Abstract base class for all pipeline components.

```python
from cvpipe import Component, Frame, SlotSchema

class MyDetector(Component):
    INPUTS     = [SlotSchema("frame_bgr", torch.uint8, (None, None, 3), "cpu")]
    OUTPUTS    = [SlotSchema("boxes_xyxy", torch.float32, (None, 4), "cpu",
                             coord_system="xyxy")]
    SUBSCRIBES = [ConfidenceChangedEvent]

    def __init__(self, weights: str):
        super().__init__()   # required — creates self._lock
        self._weights = weights

    def process(self, frame: Frame) -> None:
        ...
```

**Class attributes — declare on your subclass:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `INPUTS` | `list[SlotSchema]` | Slots this component reads from `frame.slots` |
| `OUTPUTS` | `list[SlotSchema]` | Slots this component writes to `frame.slots` |
| `SUBSCRIBES` | `list[type[Event]]` | Event types handled in `on_event()` |

**Framework-injected instance attributes:**

| Attribute | Type | Available | Description |
|-----------|------|-----------|-------------|
| `_component_id` | str | After `Pipeline(...)` | YAML-assigned component ID |
| `_event_bus` | EventBus | After `pipeline.start()` | For emitting events |
| `_lock` | threading.Lock | Always | Protect state shared between `process()` and `on_event()` |

**Methods to implement:**

| Method | Thread | Called | Description |
|--------|--------|--------|-------------|
| `process(frame)` | Streaming | Every frame | Main work — abstract, must implement |
| `setup()` | Main | Once before first frame | Load models, open devices |
| `teardown()` | Main | Once after last frame | Release GPU memory, close handles |
| `reset()` | Main (Scheduler paused) | Via `pipeline.reset()` | Clear per-session state |
| `on_event(event)` | Event dispatch | When subscribed events fire | Handle runtime changes |

**Utility method:**

| Method | Description |
|--------|-------------|
| `emit(event)` | Publish an event on the EventBus. Safe from `process()` or `on_event()` |

---

## FrameSource

Abstract interface for frame sources.

```python
from cvpipe import FrameSource
import cv2, time

class WebcamSource(FrameSource):
    def setup(self) -> None:
        self._cap = cv2.VideoCapture(0)

    def next(self):
        ok, frame = self._cap.read()
        return (frame, time.monotonic()) if ok else None

    def teardown(self) -> None:
        self._cap.release()
```

**Methods:**

| Method | Description |
|--------|-------------|
| `setup()` | Called once before the frame loop |
| `teardown()` | Called once after the frame loop stops |
| `next()` | Returns `(payload, timestamp)` or `None`. Must be non-blocking |

When `next()` returns a non-dict payload, the Scheduler places it in
`frame.slots["frame_raw"]`. When it returns a dict, it is merged into `frame.meta`.

Do **not** declare `frame_raw` in a component's `INPUTS` — it is injected by the
Scheduler and has no upstream component producer.

---

## Pipeline

The assembled, validated, runnable pipeline.

```python
from cvpipe import Pipeline

pipeline = Pipeline(
    source=my_source,
    components=[prep, detector, tracker, assembler],
    branches=[branch_spec],
    branch_components={"fast_mode": [fast_det]},
)
```

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | FrameSource | required | The frame source |
| `components` | list[Component] | required | Main-path components in execution order |
| `branches` | list[BranchSpec] | `[]` | Branch specifications (from `PipelineConfig`) |
| `branch_components` | dict[str, list[Component]] | `{}` | Instantiated branch components |
| `connections` | dict / None | None | Explicit slot connections (auto-inferred if omitted) |
| `result_bus` | ResultBus / None | auto | Created with `capacity=4` if not provided |
| `event_bus` | EventBus / None | auto | Created if not provided |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `result_bus` | ResultBus | The result bus instance |
| `event_bus` | EventBus | The event bus instance |
| `is_running` | bool | True while the Scheduler thread is alive |

**Methods:**

| Method | Description |
|--------|-------------|
| `validate()` | Check all slot contracts. Raises `PipelineConfigError` with all errors |
| `start()` | Setup all components and start the frame loop |
| `stop()` | Drain current frame, teardown all components, stop threads |
| `reset()` | Pause Scheduler, call `reset()` on all components in order, resume |
| `component(id)` | Return component by YAML ID. Searches main path and all branches. Raises `KeyError` if not found |

---

## Scheduler

Runs the frame loop. Managed by `Pipeline`. Access it via `pipeline._scheduler` after
`pipeline.start()` to attach probes.

**Probe attachment (after `pipeline.start()`):**

| Method | Description |
|--------|-------------|
| `add_probe(probe, after)` | Attach a probe. `after=None` fires after every component. `after="id"` fires only after that component |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `is_running` | bool | True if the streaming thread is alive |
| `frame_count` | int | Total successfully processed frames |

---

## ComponentRegistry

Maps module names to `Component` and `FrameSource` classes.

```python
from cvpipe import ComponentRegistry
from pathlib import Path

registry = ComponentRegistry()
registry.discover(Path("myapp/components/"))
registry.discover(Path("myapp/sources/"))
cls = registry.get("yolo_detector")
```

**Component methods:**

| Method | Description |
|--------|-------------|
| `register(name, cls)` | Explicitly register a Component class |
| `unregister(name)` | Remove a registered component |
| `get(name)` | Return Component class. Raises `ComponentNotFoundError` if not found |
| `discover(path)` | Auto-discover Component and FrameSource subclasses from a directory |

**Source methods:**

| Method | Description |
|--------|-------------|
| `register_source(name, cls)` | Explicitly register a FrameSource class |
| `unregister_source(name)` | Remove a registered source |
| `get_source(name)` | Return FrameSource class. Raises `KeyError` if not found |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `registered_names` | list[str] | Sorted list of registered component names |
| `registered_source_names` | list[str] | Sorted list of registered source names |

---

## EventBus

Pub/sub channel for management events.

```python
from cvpipe import EventBus

bus = EventBus(maxsize=256)
bus.subscribe(MyEvent, my_handler)
bus.start()
bus.publish(MyEvent(...))
bus.stop(timeout=2.0)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `subscribe(event_type, handler)` | Register a handler. Call before `start()` for guaranteed delivery |
| `publish(event)` | Enqueue an event. Non-blocking, safe from any thread |
| `start()` | Start the dispatch thread |
| `stop(timeout=2.0)` | Signal stop and wait for queue to drain |
| `handler_count(event_type)` | Number of handlers registered for a given type |

---

## ResultBus

Lossy ring buffer for high-frequency per-frame results.

```python
from cvpipe import ResultBus

bus = ResultBus(capacity=4)
bus.subscribe(my_callback)
bus.start()
bus.stop()
```

**Methods:**

| Method | Description |
|--------|-------------|
| `subscribe(callback)` | Register a result callback (one thread per subscriber) |
| `push(result)` | Push a result from the streaming thread. Drops oldest if at capacity |
| `start()` | Start subscriber threads |
| `stop(timeout=2.0)` | Stop subscriber threads |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `capacity` | int | Maximum items before oldest is dropped |
| `occupancy` | int | Current number of results in the buffer |

---

## AsyncQueueBridge

Thread-safe bridge from `ResultBus` (sync) to an asyncio event loop.

```python
from cvpipe import AsyncQueueBridge

bridge = AsyncQueueBridge(loop=loop, maxsize=8)
pipeline.result_bus.subscribe(bridge.put)   # before pipeline.start()
pipeline.start()
await bridge.start_consumer(handler)        # from inside the event loop
# ...
await bridge.stop()
```

**Constructor:** `AsyncQueueBridge(loop, maxsize=8)`

**Methods:**

| Method | Description |
|--------|-------------|
| `put(item)` | Enqueue from any thread. Drops oldest if full. Thread-safe |
| `start_consumer(handler)` | Start an async consumer Task. Raises `RuntimeError` if called twice |
| `stop()` | Cancel the consumer Task. Safe to call even if never started |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `qsize` | int | Current queue depth |

---

## Probe

Abstract base for observation hooks.

```python
from cvpipe import Probe, Frame

class MyProbe(Probe):
    def observe(self, frame: Frame, after_component: str) -> None:
        ...

pipeline.start()
pipeline._scheduler.add_probe(MyProbe(), after="detector")
```

Probes must be attached **after** `pipeline.start()`. They run in the streaming thread —
keep `observe()` under 0.5 ms.

## DiagnosticsProbe

Built-in probe that writes `frame.meta["diagnostics"]` after every component.
Attach with `after=None` to observe all components:

```python
from cvpipe import DiagnosticsProbe

pipeline.start()
pipeline._scheduler.add_probe(DiagnosticsProbe(), after=None)
```

`frame.meta["diagnostics"]` is a `FrameDiagnostics` object:

| Attribute | Type | Description |
|-----------|------|-------------|
| `frame_idx` | int | Frame counter |
| `ts` | float | Capture timestamp |
| `components` | list[ComponentTrace] | Per-component trace |
| `total_ms` | float | Total wall-clock time for the frame |
| `summary()` | str | One-line human-readable summary |

Each `ComponentTrace` has `component_id`, `latency_ms`, `output_slots`, `output_meta`.

---

## Data classes

### FrameResult

Produced by the Scheduler after every completed frame. Available in `ResultBus`
subscribers and `AsyncQueueBridge` consumers.

| Field | Type | Description |
|-------|------|-------------|
| `frame_idx` | int | Frame counter |
| `ts` | float | Capture timestamp |
| `jpeg_bytes` | bytes | JPEG-encoded frame (from `frame.meta["jpeg_bytes"]`) |
| `detections` | list[dict] | Detection records (from `frame.meta["detections"]`) |
| `meta` | dict | Arbitrary per-frame metadata (from `frame.meta["result_meta"]`) |

---

## Configuration classes

### PipelineConfig

```python
from cvpipe.config import PipelineConfig

config = PipelineConfig.from_yaml(Path("pipeline.yaml"))
config.source          # str
config.source_config   # dict
config.components      # list[ComponentSpec]
config.branches        # list[BranchSpec]
config.connections     # dict | None
```

### ComponentSpec

| Attribute | Type | Description |
|-----------|------|-------------|
| `module` | str | Directory name for discovery |
| `id` | str | Unique pipeline ID |
| `config` | dict | Constructor kwargs |

### BranchSpec

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | str | Unique branch ID |
| `trigger` | str | Python expression evaluated against `frame.meta` |
| `inject_after` | str | Component ID after which the branch starts |
| `merge_before` | str | Component ID before which the branch merges back |
| `components` | list[ComponentSpec] | Components in this branch |
| `exclusive` | bool | If True, main-path components in range are skipped when trigger fires |

---

## Event classes

### Event (base)

```python
from cvpipe import Event
from dataclasses import dataclass

@dataclass(frozen=True)
class MyEvent(Event):
    value: float
    # ts: float  — inherited, auto-set to time.monotonic() on construction
```

All custom events must be `@dataclass(frozen=True)` subclasses of `Event`.

### Built-in events

| Class | Key attributes | Emitted by |
|-------|---------------|-----------|
| `PipelineStateEvent` | `state: str`, `detail: str` | Pipeline — states: `starting`, `running`, `stopping`, `stopped`, `reset`, `error` |
| `ComponentErrorEvent` | `component_id`, `message`, `traceback`, `frame_idx` | Scheduler |
| `ComponentMetricEvent` | `component_id`, `latency_ms`, `frame_idx` | Scheduler — after every component on every frame |
| `FrameDroppedEvent` | `reason: str`, `frame_idx` | Scheduler — reasons: `source_stall`, `backpressure`, `component_error` |

---

## Exceptions

| Exception | Raised when |
|-----------|-------------|
| `PipelineConfigError` | `validate()` finds one or more contract violations. Lists all errors |
| `ComponentNotFoundError` | `registry.get(name)` — module name not registered |
| `AmbiguousComponentError` | `discover()` — a directory exports multiple Component subclasses |
| `ContractError` | Slot contract violation during validation |
| `SlotNotFoundError` | A component requires a slot that no upstream component produces |
| `CoordinateSystemError` | Writer and reader declare different coordinate systems |
| `DuplicateSlotWriterError` | Two components declare the same slot in `OUTPUTS` |
| `CvPipeError` | Base class for all framework exceptions |
