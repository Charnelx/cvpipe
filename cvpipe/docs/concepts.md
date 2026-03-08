# Core Concepts

The mental model behind cvpipe. Read this before anything else.

---

## The pipeline as a DAG

A cvpipe pipeline is a **directed acyclic graph** of components. Each node is one
processing stage. A `Frame` object flows through every node in topological order — each
component reads what upstream stages wrote and writes its own outputs.

The simplest pipeline is a straight line:

```
WebcamSource --> [Preprocessor] --> [Detector] --> [Tracker] --> [ResultAssembler]
```

Execution within a single frame is **strictly sequential and single-threaded**. The same
`Frame` instance is passed to each component in order. No data is copied between stages.

---

## Non-linear execution — branches

An **exclusive branch** is a structural if/else in the pipeline. When a condition holds,
a segment of the main path is bypassed and alternative components run instead.

```
                         true
[Preprocessor] --(mode='fast')?---------> [LightweightDetector] --------+
                         |                                               |
                         | false                                         v
                         +-----------> [FasterRCNN] --> [Classifier] ---+
                                                                        |
                                                                   [Tracker]
```

`mode` is written to `frame.meta` by `Preprocessor`. The branch is declared in YAML —
no component contains any `if` checks for routing.

### Nested branches

Branches can nest as long as one range is fully contained within the other. The typical
pattern is an outer branch that disables all inference, and inner branches that select
between detector variants:

```
[Preprocessor]
      |
      +-- inference_enabled=False --> [PassthroughMarker] -----------------+
      |                                                                     |
      +-- inference_enabled=True                                           |
                  |                                                        |
                  +-- mode='fast' --> [LightweightDetector] -------+       |
                  |                                                |       |
                  +-- mode='full' --> [FasterRCNN] --> [Classifier]+       |
                                                           |               |
                                                      [Tracker]            |
                                                           |               |
                                                [ResultAssembler] <--------+
```

Both branch conditions are YAML declarations. `Preprocessor` writes both
`frame.meta["inference_enabled"]` and `frame.meta["mode"]`.

---

## Frame

`Frame` is the mutable per-frame workspace. One instance travels through the entire
pipeline. Every component shares it.

```python
frame.idx    # int   — monotonically increasing counter (dropped frames not counted)
frame.ts     # float — time.monotonic() timestamp at capture
frame.slots  # dict  — named tensor data (torch.Tensor, GPU or CPU)
frame.meta   # dict  — CPU-side metadata: scalars, strings, lists, dicts
```

### slots vs meta

| | `frame.slots` | `frame.meta` |
|---|---|---|
| Holds | `torch.Tensor` | Any Python object |
| Lives on | GPU or CPU | CPU only |
| Typical use | Images, feature maps, bounding box tensors | Flags, counts, detection lists, routing strings |
| Examples | `frame_bgr` (H×W×3), `boxes_xyxy` (N×4) | `detection_count`, `mode`, `fps` |

Use `slots` for tensors. Use `meta` for everything else.

### Source payload injection

The Scheduler calls `source.next()`, takes the returned payload, and injects it into
`frame.slots["frame_raw"]` (unless the payload is a dict, in which case it is merged
into `frame.meta`). The first component in the pipeline reads `frame.slots["frame_raw"]`
and converts it to whatever format downstream components expect.

### FrameResult extraction

After all components run, the Scheduler calls `_extract_result(frame)` and pushes a
`FrameResult` to the `ResultBus`. It reads three keys from `frame.meta`:

```python
frame.meta["jpeg_bytes"]   # bytes  — JPEG-encoded frame for streaming
frame.meta["detections"]   # list   — per-detection dicts
frame.meta["result_meta"]  # dict   — FPS, mode, flags — goes into FrameResult.meta
```

The terminal component (typically a `ResultAssembler`) is responsible for writing these
keys. If they are absent, `FrameResult` is returned with empty defaults.

---

## SlotSchema

`SlotSchema` is how components advertise their data contracts. Each schema describes one
named slot:

```python
from cvpipe import SlotSchema
import torch

SlotSchema(
    name="boxes_xyxy",         # noun_coordsystem naming convention
    dtype=torch.float32,
    shape=(None, 4),           # None = variable-length dimension
    device="gpu",
    coord_system="xyxy",       # validated against any consumer that reads this slot
    description="Detected bounding boxes in absolute pixel coords, xyxy format",
)
```

Components declare `INPUTS` and `OUTPUTS` as lists of `SlotSchema`. `pipeline.validate()`
checks the full graph before any model loads:

- Every `INPUTS` slot must have exactly one upstream `OUTPUTS` writer
- No two components can write to the same slot name
- Coordinate systems must match between writer and reader

**Important:** `frame.slots["frame_raw"]` is injected by the Scheduler from the source
payload. It is not produced by any component. Do **not** declare it in `INPUTS` — the
validator will look for an upstream component that produces it and find none.

---

## Component lifecycle

```
pipeline.start()
    |
    +-- setup()        <- load models, open devices, allocate buffers
    |                     called once on every component, in pipeline order
    |
    |   +-- frame loop -----------------------------------------------+
    |   |   process(frame)    <- streaming thread, every frame        |
    |   |   on_event(event)   <- event thread, when events arrive     |
    |   +-------------------------------------------------------------+
    |
    +-- teardown()     <- release GPU memory, close files and devices
                          called in reverse pipeline order

pipeline.reset()
    +-- reset()        <- clear per-session state, keep models loaded
                          Scheduler is paused before any reset() call
```

**`setup()` vs `__init__()`:** store config and set attributes to `None` in `__init__`.
Load models and open devices in `setup()`. This keeps pipeline construction fast and
lets the framework sequence expensive initialisation correctly.

---

## Thread model

```
Main thread              Event dispatch thread        Streaming thread
----------------         ---------------------        ----------------
Your application         EventBus loop                Scheduler frame loop
HTTP/WebSocket           calls on_event()             calls source.next()
pipeline.reset()                                      calls process()
```

**`process()` and `on_event()` run on different threads.** Protect shared mutable state
with `self._lock`. The right pattern is snapshot → release → work:

```python
def on_event(self, event):
    if isinstance(event, ConfidenceChangedEvent):
        with self._lock:
            self._confidence = event.value   # fast write, release immediately

def process(self, frame: Frame) -> None:
    with self._lock:
        confidence = self._confidence        # snapshot — fast, release immediately
    # model inference happens outside the lock
    results = self._model(frame.slots["frame_bgr"], conf=confidence)
```

Holding the lock during model inference would block event dispatch for the entire forward
pass duration. Always snapshot, then release, then do the work.

`reset()` is different: the Scheduler is paused before any `reset()` call, so `process()`
is guaranteed not to run concurrently. Acquire `self._lock` in `reset()` only if the
state is also touched by `on_event()`.

---

## EventBus vs ResultBus

| | EventBus | ResultBus |
|---|---|---|
| **Purpose** | Management signals | Per-frame results |
| **Typical content** | "Set confidence to 0.6", "Reload model" | JPEG bytes, detection list, FPS |
| **Frequency** | Seconds between events | Every frame (25–60 Hz) |
| **Delivery** | Ordered, best-effort | Lossy ring buffer — drops oldest under backpressure |

The ResultBus is lossy by design. A live video stream doesn't need every frame to reach
the WebSocket client — it needs the most recent one. Dropping frames under load is a
feature.

---

## Error containment

When a component raises during `process()`, the Scheduler catches the exception,
emits `ComponentErrorEvent` and `FrameDroppedEvent(reason="component_error")`, discards
the frame, and continues the frame loop. One bad frame never stops the pipeline.

---

## Next Steps

- [Building Components](./building_components.md) — Implement your first component
- [Building Pipelines](./building_pipelines.md) — Configure topology in YAML
- [Observability](./observability.md) — Probes, metrics, event monitoring
