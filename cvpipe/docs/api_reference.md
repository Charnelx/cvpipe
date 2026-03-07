# API Reference

## Core Classes

### Frame

Mutable per-frame workspace passed through the pipeline.

```python
frame: Frame = Frame(idx=0, ts=1.0)
```

**Attributes:**
- `idx: int` — Monotonically increasing frame counter
- `ts: float` — Wall-clock capture timestamp
- `slots: dict[str, Any]` — Tensor slots
- `meta: dict[str, Any]` — CPU metadata

**Methods:**
- `__repr__() -> str`

### SlotSchema

Descriptor for a named data slot.

```python
SlotSchema(
    name="proposals_xyxy",
    dtype=torch.float32,
    shape=(None, 4),
    device="gpu",
    coord_system="xyxy",
    description="Detected boxes",
)
```

**Parameters:**
- `name: str` — Slot identifier (must be valid Python identifier)
- `dtype: Any` — torch.dtype, Python type, or None
- `shape: tuple` — Expected tensor shape (None = variable)
- `device: str` — "gpu", "cpu", or "any"
- `coord_system: str | None` — Coordinate system tag
- `description: str` — Human-readable description

**Methods:**
- `is_tensor_slot() -> bool` — True if dtype is torch.dtype
- `is_meta_slot() -> bool` — True if dtype is Python type or None
- `compatible_with(other: SlotSchema) -> list[str]` — Check compatibility

### Component

Abstract base class for pipeline components.

```python
class MyComponent(Component):
    INPUTS = [SlotSchema("input", int)]
    OUTPUTS = [SlotSchema("output", int)]
    SUBSCRIBES = []

    def process(self, frame: Frame) -> None:
        frame.slots["output"] = frame.slots["input"] * 2
```

**Class Attributes:**
- `INPUTS: ClassVar[list[SlotSchema]]` — Input slots
- `OUTPUTS: ClassVar[list[SlotSchema]]` — Output slots
- `SUBSCRIBES: ClassVar[list[type[Event]]]` — Event types to handle

**Methods:**
- `process(frame: Frame) -> None` — Process one frame (abstract)
- `setup() -> None` — Called once before first frame
- `teardown() -> None` — Called once after last frame
- `reset() -> None` — Reset per-session state
- `on_event(event: Event) -> None` — Handle events
- `emit(event: Event) -> None` — Publish an event
- `component_id -> str` — YAML-assigned identifier
- `input_slot_names() -> set[str]` — Set of input slot names
- `output_slot_names() -> set[str]` — Set of output slot names
- `get_input_schema(name: str) -> SlotSchema | None`
- `get_output_schema(name: str) -> SlotSchema | None`

### FrameSource

Abstract interface for frame sources.

```python
class MySource(FrameSource):
    def next(self) -> tuple[Any, float] | None:
        return (frame_data, timestamp)
```

**Methods:**
- `setup() -> None` — Called once before frame loop
- `teardown() -> None` — Called once after frame loop
- `next() -> tuple[Any, float] | None` — Get next frame

## Pipeline Classes

### Pipeline

Assembled, validated, runnable pipeline.

```python
pipeline = Pipeline(
    source=my_source,
    components=[comp_a, comp_b],
    result_bus=ResultBus(),
    event_bus=EventBus(),
)
```

**Properties:**
- `result_bus -> ResultBus`
- `event_bus -> EventBus`
- `is_running -> bool`

**Methods:**
- `validate() -> None` — Validate component contracts
- `start() -> None` — Start the pipeline
- `stop() -> None` — Stop the pipeline
- `reset() -> None` — Reset component state without stopping
- `component(id: str) -> Component` — Get component by ID

### Scheduler

Runs the frame processing loop.

**Import:** `from cvpipe.scheduler import Scheduler`

**Properties:**
- `is_running -> bool`
- `frame_count -> int`

**Methods:**
- `start() -> None` — Start streaming thread
- `stop(timeout: float = 5.0) -> None` — Stop streaming thread
- `pause(timeout: float = 2.0) -> None` — Pause between frames
- `resume() -> None` — Resume after pause
- `add_probe(probe: Probe, after: str | None) -> None` — Attach probe

### ComponentRegistry

Maps module names to Component classes.

```python
registry = ComponentRegistry()
registry.discover(Path("detector/components/"))
cls = registry.get("frcnn_proposer")
```

**Methods:**
- `register(name: str, cls: type[Component]) -> None`
- `unregister(name: str) -> None`
- `get(name: str) -> type[Component]`
- `discover(components_dir: Path) -> None`

**Properties:**
- `registered_names -> list[str]`

## Bus Classes

### EventBus

Pub/sub event channel.

```python
bus = EventBus()
bus.subscribe(MyEvent, handler)
bus.start()
bus.publish(MyEvent(...))
bus.stop()
```

**Methods:**
- `subscribe(event_type: type[Event], handler: Callable) -> None`
- `publish(event: Event) -> None`
- `start() -> None`
- `stop(timeout: float = 2.0) -> None`
- `handler_count(event_type: type) -> int`

### ResultBus

Lossy ring buffer for frame results.

```python
bus = ResultBus(capacity=4)
bus.subscribe(callback)
bus.start()
bus.push(FrameResult(...))
bus.stop()
```

**Methods:**
- `subscribe(callback: Callable[[FrameResult], None]) -> None`
- `push(result: FrameResult) -> None`
- `start() -> None`
- `stop(timeout: float = 2.0) -> None`

**Properties:**
- `capacity -> int`
- `occupancy -> int`

### AsyncQueueBridge

Thread-safe bridge from ResultBus to asyncio event loop.

```python
bridge = AsyncQueueBridge(loop=loop, maxsize=8)
result_bus.subscribe(bridge.put)
await bridge.start_consumer(my_handler)
await bridge.stop()
```

**Constructor:**
- `__init__(loop: asyncio.AbstractEventLoop, maxsize: int = 8)`

**Methods:**
- `put(item: Any) -> None` — Enqueue from any thread
- `start_consumer(handler: Callable) -> None` — Start async consumer
- `stop() -> None` — Cancel consumer

**Properties:**
- `qsize -> int` — Current queue depth

## Data Classes

### FrameResult

Summary of a completed frame.

```python
FrameResult(
    frame_idx=42,
    ts=1234.567,
    jpeg_bytes=b"...",
    detections=[...],
    meta={},
)
```

### FrameDiagnostics

Per-frame diagnostic trace.

```python
FrameDiagnostics(
    frame_idx=42,
    ts=1234.567,
    components=[...],
    total_ms=45.3,
)
```

**Methods:**
- `summary() -> str` — One-line summary

### ComponentTrace

Timing for one component.

```python
ComponentTrace(
    component_id="proposer",
    latency_ms=5.2,
    output_slots=["proposals_xyxy"],
    output_meta=[],
    notes="",
)
```

## Event Classes

### Event

Base class for all events.

```python
@dataclass(frozen=True)
class MyEvent(Event):
    value: int
```

**Attributes:**
- `ts: float` — Monotonic timestamp

### ComponentErrorEvent

Emitted when a component raises.

```python
ComponentErrorEvent(
    component_id="proposer",
    message="...",
    traceback="...",
    frame_idx=42,
)
```

### ComponentMetricEvent

Emitted after each component processes.

```python
ComponentMetricEvent(
    component_id="proposer",
    latency_ms=5.2,
    frame_idx=42,
)
```

### FrameDroppedEvent

Emitted when a frame is dropped.

```python
FrameDroppedEvent(
    reason="component_error",
    frame_idx=42,
)
```

### PipelineStateEvent

Emitted on lifecycle transitions.

```python
PipelineStateEvent(
    state="running",
    detail="",
)
```

## Configuration Classes

### PipelineConfig

Parsed YAML pipeline configuration.

**Attributes:**
- `source: str`
- `source_config: dict[str, Any]`
- `components: list[ComponentSpec]`
- `connections: dict[str, list[str]] | None`
- `branches: list[BranchSpec]`

**Methods:**
- `from_yaml(path: Path | str) -> PipelineConfig`

### ComponentSpec

One component from YAML.

**Attributes:**
- `module: str`
- `id: str`
- `config: dict[str, Any]`

### BranchSpec

Conditional branch from YAML.

**Attributes:**
- `id: str`
- `components: list[ComponentSpec]`
- `trigger: str`
- `inject_after: str`
- `merge_before: str`
- `exclusive: bool` — If true, branch is exclusive (skip main path)

## Exception Classes

| Exception | Description |
|-----------|-------------|
| `CvPipeError` | Base class for all framework exceptions |
| `PipelineConfigError` | Pipeline validation failed |
| `ContractError` | Slot contract violation |
| `SlotNotFoundError` | Required input slot not produced |
| `CoordinateSystemError` | Coord system mismatch |
| `DuplicateSlotWriterError` | Multiple writers for same slot |
| `ComponentError` | Runtime error in process() |
| `ComponentNotFoundError` | Module not found in registry |
| `AmbiguousComponentError` | Module exports multiple Components |
