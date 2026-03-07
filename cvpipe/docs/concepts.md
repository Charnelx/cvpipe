# Core Concepts

This document explains the fundamental concepts behind cvpipe.

## The Pipeline Model

A cvpipe pipeline is a directed acyclic graph (DAG) of components. Each component represents one processing stage:

```
FrameSource → Proposer → Embedder → Scorer → ResultBus
```

Components are executed in topological order. The same `Frame` instance is passed sequentially through all components — no data is copied between stages.

## Frame

`Frame` is a mutable per-frame workspace. It holds:

- `idx`: Monotonically increasing frame counter
- `ts`: Wall-clock capture timestamp
- `slots`: Dict of tensor data (GPU/CPU tensors)
- `meta`: Dict of CPU-side metadata (scalars, strings, dicts)

Components read from `frame.slots` and `frame.meta`, then write their outputs to the same dicts.

### Why __slots__?

`Frame` uses `__slots__` to reduce per-instance memory overhead. Frames are created at camera frame-rate (25–60 Hz); avoiding `__dict__` saves ~200 bytes per instance.

## SlotSchema

`SlotSchema` describes a named data slot:

```python
SlotSchema(
    name="proposals_xyxy",      # noun_coordsystem convention
    dtype=torch.float32,        # torch.dtype for slots, Python type for meta
    shape=(None, 4),            # None for variable dimensions
    device="gpu",               # "gpu", "cpu", or "any"
    coord_system="xyxy",        # for validation
    description="...",
)
```

### Slot Naming Convention

Use `noun_coordsystem` format:
- `proposals_xyxy` — region proposals in absolute xyxy coords
- `embeddings_cls` — CLS token embeddings
- `frame_bgr` — raw BGR frame

### Slots vs Meta

| Property | slots | meta |
|----------|-------|------|
| Data type | torch.Tensor | Python types (int, str, dict) |
| Device | GPU or CPU | CPU only |
| Use for | Large tensors | Small scalars, strings, routing decisions |

## Component Lifecycle

1. `__init__()` — called at pipeline assembly time
2. `setup()` — called once before the frame loop starts
3. `process(frame)` — called for each frame (hot path)
4. `teardown()` — called once after the pipeline stops

### Thread Guarantees

- `process()` runs only in the streaming thread
- `on_event()` runs only in the event dispatch thread
- These two threads share component state — use `self._lock` to protect shared mutable state

## EventBus vs ResultBus

| Bus | Purpose | Frequency | Blocking |
|-----|---------|-----------|----------|
| EventBus | Management signals, errors, state changes | Low (seconds) | Non-blocking |
| ResultBus | Per-frame inference results | High (25–60 Hz) | Lossy ring buffer |

## The Two-Thread Model

| Thread | Owner | Responsibility |
|--------|-------|----------------|
| Streaming | Scheduler | Frame loop: pull frame → run components → push result |
| Event dispatch | EventBus | Dequeue events → call subscriber handlers |
| Main | Application | FastAPI server, HTTP/WebSocket |

## Error Containment

When a component raises an exception in `process()`:

1. The Scheduler catches the exception
2. Emits `ComponentErrorEvent` with details
3. Emits `FrameDroppedEvent(reason="component_error")`
4. Continues the frame loop with the next frame

The pipeline never crashes due to a bad component.

## → Next Steps

- [Building Components](./building_components.md) — Implement your first component
- [Building Pipelines](./building_pipelines.md) — Configure pipeline YAML
- [Observability](./observability.md) — Monitor pipeline health
