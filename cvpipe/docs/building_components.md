# Building Components

This guide walks through implementing a cvpipe component.

## Minimal Component

```python
from cvpipe import Component, Frame

class Identity(Component):
    INPUTS = []
    OUTPUTS = []

    def process(self, frame: Frame) -> None:
        pass
```

## Declaring INPUTS and OUTPUTS

Use `SlotSchema` to declare what your component reads and writes:

```python
from cvpipe import Component, Frame, SlotSchema

class Proposer(Component):
    INPUTS = [
        SlotSchema("frame_bgr", torch.uint8, (None, None, 3), "gpu"),
    ]
    OUTPUTS = [
        SlotSchema(
            name="proposals_xyxy",
            dtype=torch.float32,
            shape=(None, 4),
            device="gpu",
            coord_system="xyxy",
            description="Detected bounding boxes",
        ),
    ]
```

### SlotSchema Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| name | str | Slot identifier (noun_coordsystem) |
| dtype | torch.dtype / type / None | Expected data type |
| shape | tuple | Expected tensor shape (None = variable) |
| device | str | "gpu", "cpu", or "any" |
| coord_system | str | Coordinate system tag for validation |
| description | str | Human-readable description |

## Declaring SUBSCRIBES

If your component reacts to events, declare them:

```python
from cvpipe import Component, Frame, SlotSchema, Event

@dataclass(frozen=True)
class ClassRegisteredEvent(Event):
    class_id: int
    label: str
    fingerprint: torch.Tensor
    threshold: float

class Scorer(Component):
    INPUTS = [SlotSchema("embeddings_cls", torch.float32, (None, 512))]
    OUTPUTS = [SlotSchema("scores", torch.float32, (None,))]
    SUBSCRIBES = [ClassRegisteredEvent]

    def __init__(self):
        super().__init__()  # Required! Creates self._lock
        self._fingerprints: dict[int, torch.Tensor] = {}
        self._thresholds: dict[int, float] = {}

    def process(self, frame: Frame) -> None:
        # Snapshot state inside lock, then release immediately
        with self._lock:
            fingerprints = dict(self._fingerprints)
            thresholds = dict(self._thresholds)
        # All inference happens OUTSIDE the lock
        scores = compute_similarity(frame.slots["embeddings_cls"], fingerprints, thresholds)
        frame.slots["scores"] = scores
```

## Implementing on_event()

`on_event()` runs in the event dispatch thread — not the streaming thread. Use `self._lock` to protect any state also accessed in `process()`.

**Key principle: minimal critical section.** Only hold the lock long enough to snapshot the data, then release it before doing any work:

```python
def on_event(self, event: Event) -> None:
    if isinstance(event, ClassRegisteredEvent):
        # Write side: acquire lock briefly, update state, release
        with self._lock:
            self._fingerprints[event.class_id] = event.fingerprint
            self._thresholds[event.class_id] = event.threshold
```

The matching read side in `process()` follows the same pattern:

```python
def process(self, frame: Frame) -> None:
    # Read side: acquire lock briefly, copy state, release
    with self._lock:
        fingerprints = dict(self._fingerprints)
        thresholds = dict(self._thresholds)
    # All heavy work (inference, computation) happens OUTSIDE the lock
    scores = compute_scores(fingerprints, thresholds, frame.slots["embeddings_cls"])
    frame.slots["scores"] = scores
```

This pattern ensures:
1. **Consistency** — both write and read are synchronized
2. **No blocking** — lock is held for <100µs (just dict copies)
3. **Hot path safety** — inference runs without holding any locks

## Implementing setup() and teardown()

Use `setup()` for heavy initialization:

```python
def setup(self) -> None:
    self._model = load_model(self._model_path)
    self._model.to(self._device)
```

Use `teardown()` for cleanup:

```python
def teardown(self) -> None:
    del self._model
```

## Implementing reset()

Use `reset()` to clear per-session application state without reloading GPU models:

```python
def reset(self) -> None:
    self._session_cache.clear()
    self._tracker_history = []
    self._frame_count = 0
```

When to use reset():
- Clearing cached results
- Resetting tracker histories
- Clearing rolling averages
- Any state that should not persist across sessions

When NOT to use reset():
- Releasing GPU memory (use `teardown()` instead)
- Closing file handles (use `teardown()` instead)

The framework calls `reset()` in topological order (upstream first). Called via `pipeline.reset()`.

## Using self.emit()

Emit events from your component:

```python
from cvpipe import ComponentErrorEvent

def process(self, frame: Frame) -> None:
    try:
        # ... detection logic ...
    except Exception as exc:
        self.emit(ComponentErrorEvent(
            component_id=self.component_id,
            message=str(exc),
            traceback=traceback.format_exc(),
            frame_idx=frame.idx,
        ))
```

## Module Package Pattern

Structure your component as a package:

```
detector/components/
└── frcnn_proposer/
    ├── __init__.py    # exports exactly one Component subclass
    └── _model.py      # private implementation
```

In `__init__.py`:

```python
from ._proposer import FRCNNProposer

__all__ = ["FRCNNProposer"]
```

## Testing a Component

Test in isolation without a full pipeline:

```python
from cvpipe import Frame

# Create a test frame
frame = Frame(idx=0, ts=0.0)
frame.slots["frame_bgr"] = torch.zeros((480, 640, 3), dtype=torch.uint8)

# Run your component
component = MyComponent()
component.process(frame)

# Check outputs
assert "proposals_xyxy" in frame.slots
```

## Complete Example: FakeProposer

A fully annotated example component that generates random bounding boxes:

```python
import torch
from cvpipe import Component, Frame, SlotSchema

class FakeProposer(Component):
    """
    Generates random bounding box proposals for testing.
    
    This component demonstrates:
    - Declaring INPUTS (none required)
    - Declaring OUTPUTS (proposals_xyxy)
    - Using setup() for initialization
    - Writing to frame.slots (tensor output)
    """
    
    # No inputs required - generates proposals from scratch
    INPUTS = []
    
    # Outputs proposals as tensor in xyxy format
    OUTPUTS = [
        SlotSchema(
            name="proposals_xyxy",
            dtype=torch.float32,
            shape=(None, 4),  # (N, 4) for N boxes
            device="cpu",
            coord_system="xyxy",
            description="Random bounding boxes [x1, y1, x2, y2]",
        ),
    ]
    
    # No event subscriptions
    SUBSCRIBES = []

    def __init__(self, num_proposals: int = 10, seed: int = 42):
        """
        Initialize the proposer.
        
        Parameters
        ----------
        num_proposals : int
            Number of random boxes to generate per frame.
        seed : int
            Random seed for reproducible boxes.
        """
        super().__init__()
        self._num_proposals = num_proposals
        self._seed = seed
        self._rng = None  # Created in setup()

    def setup(self) -> None:
        """Called once before first frame. Initialize RNG here."""
        self._rng = torch.Generator()
        self._rng.manual_seed(self._seed)

    def process(self, frame: Frame) -> None:
        """
        Generate random proposals and write to frame.slots.
        
        This runs on every frame in the streaming thread.
        """
        # Generate random boxes in [0, 1] normalized coords
        boxes = torch.rand(
            (self._num_proposals, 4), 
            dtype=torch.float32,
            generator=self._rng
        )
        
        # Convert to xyxy format (expand to image size)
        # Assume 640x480 for this example
        boxes[:, 2] *= 640  # x2
        boxes[:, 3] *= 480  # y2
        
        # Write to frame.slots (not frame.meta - this is a tensor)
        frame.slots["proposals_xyxy"] = boxes

    def teardown(self) -> None:
        """Called once after pipeline stops. Cleanup here."""
        self._rng = None
```

## → Next Steps

- [Building Pipelines](./building_pipelines.md) — Configure pipeline YAML
- [Observability](./observability.md) — Add diagnostics
- [API Reference](./api_reference.md) — Complete API docs
