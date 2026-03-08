# Building Components

A component is one processing stage in the pipeline. It reads from a `Frame`, does its
work, and writes results back to the same `Frame`. Every component follows the same
lifecycle and the same contract system.

---

## The minimal component

```python
from cvpipe import Component, Frame

class PassThrough(Component):
    INPUTS     = []
    OUTPUTS    = []
    SUBSCRIBES = []

    def process(self, frame: Frame) -> None:
        pass
```

Three class attributes, one method. Everything else is optional.

---

## Declaring INPUTS and OUTPUTS

Use `SlotSchema` to declare every tensor slot your component reads and writes.
`pipeline.validate()` checks the full graph before any model loads — missing slots,
duplicate writers, and coordinate system mismatches all surface as clear errors.

```python
from cvpipe import Component, Frame, SlotSchema
import torch

class FasterRCNNDetector(Component):
    INPUTS = [
        SlotSchema("frame_bgr", torch.uint8, (None, None, 3), "cpu",
                   description="Raw BGR frame"),
    ]
    OUTPUTS = [
        SlotSchema("boxes_xyxy", torch.float32, (None, 4), "gpu",
                   coord_system="xyxy",
                   description="Detected bounding boxes in absolute pixel coords"),
        SlotSchema("class_scores", torch.float32, (None,), "gpu",
                   description="Confidence score per detection"),
    ]
    SUBSCRIBES = []
```

### SlotSchema parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | str | ✓ | Slot identifier. Use `noun_coordsystem` format |
| `dtype` | torch.dtype / Python type / None | ✓ | `torch.float32` etc. for tensor slots; Python type for meta slots |
| `shape` | tuple | ✓ | Expected shape. `None` for variable-length dimensions |
| `device` | str | ✓ | `"gpu"`, `"cpu"`, or `"any"` |
| `coord_system` | str | | `"xyxy"`, `"xywh"` etc. Validated against readers |
| `description` | str | | Human-readable description |

**Naming convention:** `noun_coordsystem` — `boxes_xyxy`, `frame_bgr`, `features_pool`.
The coordinate suffix makes mismatches obvious before any model runs.

### slots vs meta

Not everything belongs in `frame.slots`. Tensors go there; everything else goes in
`frame.meta`. `SlotSchema` with a `torch.dtype` declares a tensor slot. For meta-only
outputs — plain Python objects like detection lists, counts, and flags — you do not need
to declare a `SlotSchema`:

```python
def process(self, frame: Frame) -> None:
    # tensors → slots (declared in OUTPUTS)
    frame.slots["boxes_xyxy"]   = boxes_tensor
    frame.slots["class_scores"] = scores_tensor

    # scalars, lists, dicts → meta (no SlotSchema needed)
    frame.meta["detection_count"] = len(boxes_tensor)
    frame.meta["detections"] = [
        {"label": "car", "score": 0.91, "box": [10, 20, 80, 60], "track_id": 3},
    ]
```

### What about `frame_raw`?

`frame.slots["frame_raw"]` is injected by the Scheduler directly from `source.next()`.
Do **not** declare it in `INPUTS` — there is no upstream component that produces it, and
the validator would raise a `SlotNotFoundError`. Just read it directly:

```python
def process(self, frame: Frame) -> None:
    raw = frame.slots["frame_raw"]   # available without declaring in INPUTS
```

---

## setup() and teardown()

Load models and open devices in `setup()`. Keep `__init__` fast.

```python
class FasterRCNNDetector(Component):
    INPUTS = [
        SlotSchema("frame_bgr", torch.uint8, (None, None, 3), "cpu"),
    ]
    OUTPUTS = [
        SlotSchema("boxes_xyxy", torch.float32, (None, 4), "gpu",
                   coord_system="xyxy"),
        SlotSchema("class_scores", torch.float32, (None,), "gpu"),
    ]
    SUBSCRIBES = []

    def __init__(self, weights: str, device: str = "cuda"):
        super().__init__()          # always call — creates self._lock
        self._weights = weights
        self._device  = device
        self._model   = None        # loaded in setup()

    def setup(self) -> None:
        import torchvision
        self._model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=None
        )
        self._model.load_state_dict(torch.load(self._weights))
        self._model.to(self._device).eval()

    def teardown(self) -> None:
        del self._model
        self._model = None

    def process(self, frame: Frame) -> None:
        bgr = frame.slots["frame_bgr"].float() / 255.0
        rgb = bgr[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).to(self._device)
        with torch.no_grad():
            output = self._model(rgb)[0]
        frame.slots["boxes_xyxy"]   = output["boxes"]
        frame.slots["class_scores"] = output["scores"]
        frame.meta["detection_count"] = len(output["boxes"])
```

---

## Reacting to events with on_event()

Declare event types in `SUBSCRIBES` and implement `on_event()` to update runtime state.
`on_event()` runs in the event dispatch thread; `process()` runs in the streaming thread.
Use `self._lock` to protect shared state. Always snapshot quickly, then release before
doing the work:

```python
from dataclasses import dataclass
from cvpipe import Component, Frame, SlotSchema, Event

@dataclass(frozen=True)
class ConfidenceChangedEvent(Event):
    value: float

class YOLODetector(Component):
    INPUTS = [
        SlotSchema("frame_bgr", torch.uint8, (None, None, 3), "cpu"),
    ]
    OUTPUTS = [
        SlotSchema("boxes_xyxy", torch.float32, (None, 4), "cpu",
                   coord_system="xyxy"),
    ]
    SUBSCRIBES = [ConfidenceChangedEvent]

    def __init__(self, weights: str = "yolov8n.pt", confidence: float = 0.45):
        super().__init__()
        self._weights    = weights
        self._confidence = confidence   # shared — protected by self._lock

    def setup(self) -> None:
        from ultralytics import YOLO
        self._model = YOLO(self._weights)

    def on_event(self, event) -> None:
        if isinstance(event, ConfidenceChangedEvent):
            with self._lock:
                self._confidence = event.value   # fast write, release immediately

    def process(self, frame: Frame) -> None:
        with self._lock:
            conf = self._confidence              # snapshot, release immediately
        results = self._model(
            frame.slots["frame_bgr"].numpy(), conf=conf, verbose=False
        )
        frame.slots["boxes_xyxy"] = results[0].boxes.xyxy
        frame.meta["detection_count"] = len(results[0].boxes)
```

Holding the lock across model inference would block the event dispatch thread for the
entire forward pass. Always: snapshot → release → work.

---

## reset()

`reset()` clears per-session state without stopping the pipeline or reloading models.
The Scheduler is paused before `reset()` is called, so `process()` is guaranteed not to
run concurrently. Only acquire `self._lock` if `on_event()` also touches the state:

```python
def reset(self) -> None:
    with self._lock:         # needed because on_event() also touches _track_ids
        self._track_ids   = {}
        self._lost_frames = {}
    self._frame_count = 0    # only touched by process() — no lock needed after pause
```

Use `reset()` for: clearing tracker histories, resetting FPS counters, reloading a class
list for a new detection job. Do not use it for unloading GPU models — that is
`teardown()`'s job.

---

## Writing FrameResult keys (terminal component)

The terminal component in a pipeline is responsible for writing three keys to `frame.meta`
that the Scheduler uses to build the `FrameResult` pushed to the `ResultBus`:

```python
frame.meta["jpeg_bytes"]   # bytes — JPEG-encoded frame
frame.meta["detections"]   # list of detection dicts
frame.meta["result_meta"]  # dict — FPS, mode flags, anything else for the client
```

A `ResultAssembler` component is the standard pattern:

```python
import cv2, time, threading
from collections import deque
from cvpipe import Component, Frame

class ResultAssembler(Component):
    INPUTS = [
        SlotSchema("frame_bgr", torch.uint8, (None, None, 3), "cpu"),
    ]
    OUTPUTS    = []
    SUBSCRIBES = []

    def __init__(self, jpeg_quality: int = 85):
        super().__init__()
        self._quality   = jpeg_quality
        self._timestamps: deque = deque(maxlen=30)

    def reset(self) -> None:
        self._timestamps.clear()

    def process(self, frame: Frame) -> None:
        bgr = frame.slots["frame_bgr"].numpy()
        _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self._quality])

        now = time.monotonic()
        self._timestamps.append(now)
        fps = 0.0
        if len(self._timestamps) >= 2:
            elapsed = self._timestamps[-1] - self._timestamps[0]
            fps = (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0

        frame.meta["jpeg_bytes"]  = bytes(buf)
        frame.meta["detections"]  = frame.meta.get("tracks", frame.meta.get("detections", []))
        frame.meta["result_meta"] = {"fps": round(fps, 1)}
```

---

## Emitting events

Call `self.emit()` from `process()` or `on_event()` to publish an event on the EventBus:

```python
from dataclasses import dataclass
from cvpipe import Event

@dataclass(frozen=True)
class ObjectEnteredZoneEvent(Event):
    track_id: int
    zone: str

class ZoneMonitor(Component):
    INPUTS = OUTPUTS = SUBSCRIBES = []

    def process(self, frame: Frame) -> None:
        for track in frame.meta.get("tracks", []):
            if self._inside_zone(track["box"]):
                self.emit(ObjectEnteredZoneEvent(track_id=track["track_id"], zone="entry"))
```

---

## Component package layout

Each component lives in its own directory. The directory name is what you reference in
`pipeline.yaml`. Export exactly one `Component` subclass from `__init__.py`:

```
components/
└── faster_rcnn_detector/
    ├── __init__.py      # exports exactly ONE Component subclass
    └── _backbone.py     # private helpers — prefixed with _ so discovery ignores them
```

```python
# faster_rcnn_detector/__init__.py
from ._detector import FasterRCNNDetector

__all__ = ["FasterRCNNDetector"]
```

In `pipeline.yaml`:
```yaml
- module: faster_rcnn_detector   # directory name
  id: detector
  config:
    weights: checkpoints/frcnn.pth
    device: cuda
```

---

## Testing a component in isolation

No pipeline needed. Construct, call `setup()`, build a `Frame`, call `process()`:

```python
import torch
from cvpipe import Frame

def test_detector_produces_boxes():
    comp = FasterRCNNDetector(weights="checkpoints/frcnn.pth", device="cpu")
    comp._component_id = "detector"
    comp.setup()

    frame = Frame(idx=0, ts=0.0)
    frame.slots["frame_bgr"] = torch.zeros((480, 640, 3), dtype=torch.uint8)

    comp.process(frame)

    assert "boxes_xyxy" in frame.slots
    assert frame.slots["boxes_xyxy"].shape[1] == 4
    comp.teardown()
```

Testing `on_event()` is equally direct — no threading required:

```python
def test_confidence_update():
    comp = YOLODetector(confidence=0.45)
    comp._component_id = "detector"
    comp.setup()
    comp.on_event(ConfidenceChangedEvent(value=0.8))
    with comp._lock:
        assert comp._confidence == 0.8
```

---

## Complete annotated example: frame preprocessor

```python
import torch
import cv2
from cvpipe import Component, Frame, SlotSchema

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


class FramePreprocessor(Component):
    """
    Convert a raw uint8 BGR frame to a normalised float32 CHW tensor on GPU.

    Reads:  frame.slots["frame_raw"]   — raw numpy array from source
    Writes: frame.slots["frame_bgr"]   — uint8  (H, W, 3) on CPU  (for JPEG encoding)
            frame.slots["frame_norm"]  — float32 (3, H, W) on GPU (for model input)
    Also writes frame.meta["inference_enabled"] for branch trigger.
    """

    INPUTS = []   # frame_raw is injected by the Scheduler; do not declare it here
    OUTPUTS = [
        SlotSchema("frame_bgr",  torch.uint8,   (None, None, 3), "cpu",
                   description="Resized BGR frame for JPEG encoding"),
        SlotSchema("frame_norm", torch.float32, (3, None, None), "gpu",
                   description="Normalised CHW tensor, ImageNet mean/std"),
    ]
    SUBSCRIBES = []

    def __init__(self, target_width: int = 640, target_height: int = 480,
                 device: str = "cuda"):
        super().__init__()
        self._w      = target_width
        self._h      = target_height
        self._device = device
        self._inference_enabled = True

    def setup(self) -> None:
        self._mean = torch.tensor(_IMAGENET_MEAN, device=self._device).view(3, 1, 1)
        self._std  = torch.tensor(_IMAGENET_STD,  device=self._device).view(3, 1, 1)

    def process(self, frame: Frame) -> None:
        raw = frame.slots["frame_raw"]                         # numpy (H, W, 3) BGR
        if raw.shape[:2] != (self._h, self._w):
            raw = cv2.resize(raw, (self._w, self._h))

        bgr_t = torch.from_numpy(raw.copy())                   # uint8 CPU
        frame.slots["frame_bgr"] = bgr_t                       # for ResultAssembler

        rgb  = bgr_t[..., [2, 1, 0]]                           # BGR -> RGB
        chw  = rgb.permute(2, 0, 1).to(self._device).float() / 255.0
        frame.slots["frame_norm"] = (chw - self._mean) / self._std

        frame.meta["inference_enabled"] = self._inference_enabled

    def teardown(self) -> None:
        del self._mean, self._std
```

---

## Runtime Slot Validation

cvpipe can validate tensor slot writes at runtime to catch shape, dtype, and device
mismatches early. This is especially useful during development when debugging
component contracts.

### Validation modes

| Mode | Behavior | Use case |
|------|----------|----------|
| `off` | No validation (default in production) | Production deployment |
| `warn` | Log warnings on mismatch | Development (default) |
| `strict` | Raise `SlotValidationError` immediately | Testing, CI |

### Enabling validation

**In YAML:**

```yaml
pipeline:
  source: webcam_source
  components:
    - module: detector
      id: detector

  validation:
    mode: warn  # "off" | "warn" | "strict"
```

**In Python:**

```python
from cvpipe import build, Pipeline

# Via build()
pipeline = build(config_path, components_dir)

# Or explicit mode change before start()
pipeline.set_validation_mode("strict")
pipeline.start()
```

### What gets validated

When a component writes to `frame.slots`, the validation checks:

1. **Type**: Is it a `torch.Tensor` for tensor slots?
2. **dtype**: Does it match the declared dtype (e.g., `torch.float32`)?
3. **Shape**: Does each dimension match the schema (with `None` allowing variable length)?
4. **Device**: Is it on GPU or CPU as declared?

Example error message:

```
[detector] Slot 'boxes_xyxy': dtype torch.int64 != expected torch.float32; shape[1]: 3 != expected 4
```

### Performance impact

Validation adds approximately **0.5 µs overhead per slot write**. For a typical
pipeline with 10 slot writes per frame at 30 fps, this is ~150 µs per frame
(~0.5% overhead).

When `mode="warn"`, each slot is logged at most once per frame to avoid log spam.

**Recommendation:** Use `warn` mode during development, `strict` in CI/testing,
and `off` in production.

---

## Common mistakes

**Forgetting `super().__init__()`** — `self._lock` is created there. Any `on_event()`
that uses it will raise `AttributeError`.

**Loading models in `__init__()`** — runs at construction time. The GPU may not be ready.
Use `setup()`.

**Declaring `frame_raw` in `INPUTS`** — it is injected by the Scheduler, not produced by
any component. Declaring it causes a `SlotNotFoundError` during validation.

**Holding `self._lock` during inference** — blocks the event dispatch thread for the
entire forward pass duration. Snapshot → release → work.

**Writing `jpeg_bytes`/`detections`/`result_meta` in a non-terminal component** — these
are `FrameResult` assembly keys read by the Scheduler after all components run. Only the
terminal `ResultAssembler` should write them.

**Wrong tensor dtype/shape at runtime** — the detector writes `torch.int64` but
the tracker expects `torch.float32`. Enable validation mode `warn` during development
to catch these mismatches early.

---

## Next Steps

- [Building Pipelines](./building_pipelines.md) — Wire components into a pipeline
- [Observability](./observability.md) — Add probes and diagnostics
- [API Reference](./api_reference.md) — Complete API reference
