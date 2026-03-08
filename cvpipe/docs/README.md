# cvpipe

A lightweight Python framework for building computer vision inference pipelines as
directed acyclic graphs (DAGs) of swappable components.

```bash
pip install git+https://github.com/Charnelx/cvpipe.git
```

---

## Why cvpipe?

Building a CV inference system from scratch means solving the same scaffolding problems
every time: a frame capture loop that doesn't block the inference thread, passing tensors
between model stages without copying them, error handling so one bad frame doesn't crash
the whole system, getting results out to a WebSocket without async boilerplate.

cvpipe handles all of that so you can focus on the vision work.

**What you get:**

- **Typed slot contracts.** Components declare what data they read and write. The full
  graph is validated before any model loads — missing slots, duplicate writers, and
  coordinate system mismatches all surface as clear errors at startup.

- **Zero-copy frame passing.** A single `Frame` object travels through every stage. No
  serialisation between steps, no inter-stage queues, no redundant tensor copies.

- **Event-driven state.** A pub/sub `EventBus` lets components react to runtime changes —
  confidence threshold updates, enable/disable signals, model hot-swaps — without polling
  or shared globals.

- **Graceful error containment.** When a component raises, the frame is dropped and a
  `ComponentErrorEvent` is emitted. The pipeline keeps running.

- **Conditional branches.** Declare if/else routing in YAML — skip heavy inference when
  the camera is in preview mode, swap between a fast and a slow model at runtime.

- **Auto-discovery.** Drop a component in a directory, export one class — cvpipe finds it.

- **Async-ready.** `AsyncQueueBridge` connects your result stream to FastAPI or aiohttp
  WebSockets without managing threads yourself.

---

## Quickstart — manual construction

The fastest way to get something running is to construct a pipeline directly in Python.
This is useful for prototyping, but **the recommended approach for production is to
declare the pipeline topology in a YAML file** and use `cvpipe.build()` to assemble it.
YAML-based pipelines separate topology from code, are easier to review, and allow
component swap-outs without touching Python. See [Building Pipelines](./docs/building_pipelines.md).

```python
import cv2, time
from ultralytics import YOLO
from cvpipe import Pipeline, FrameSource, Component, Frame, SlotSchema

# ── Source: pull frames from a webcam ─────────────────────────────────────────

class WebcamSource(FrameSource):
    def setup(self):
        self._cap = cv2.VideoCapture(0)

    def next(self):
        ok, frame = self._cap.read()
        return (frame, time.monotonic()) if ok else None

    def teardown(self):
        self._cap.release()


# ── Component: run YOLOv8 and write detections ────────────────────────────────
# Reads frame_raw from slots (injected by the Scheduler from WebcamSource.next()).
# Writes detections, jpeg_bytes, and result_meta to frame.meta so the Scheduler
# can assemble a FrameResult for the ResultBus.

class YOLODetector(Component):
    INPUTS     = []   # frame_raw is injected by the framework, not declared as INPUTS
    OUTPUTS    = []   # this component writes only to frame.meta, not to typed tensor slots
    SUBSCRIBES = []

    def __init__(self, confidence: float = 0.45):
        super().__init__()
        self._confidence = confidence

    def setup(self):
        self._model = YOLO("yolov8n.pt")

    def process(self, frame: Frame) -> None:
        import cv2
        bgr = frame.slots["frame_raw"]
        results = self._model(bgr, conf=self._confidence, verbose=False)

        detections = [
            {
                "label":      self._model.names[int(b.cls)],
                "confidence": float(b.conf),
                "box":        [round(v) for v in b.xyxy[0].tolist()],
                "track_id":   -1,
                "state":      "TRACKED",
                "score":      float(b.conf),
            }
            for b in results[0].boxes
        ]

        # Encode the annotated frame as JPEG so the ResultBus can stream it
        _, buf = cv2.imencode(".jpg", bgr)
        frame.meta["jpeg_bytes"]  = bytes(buf)
        frame.meta["detections"]  = detections
        frame.meta["result_meta"] = {"fps": 0.0}


# ── Wire and run ──────────────────────────────────────────────────────────────

pipeline = Pipeline(
    source=WebcamSource(),
    components=[YOLODetector()],
)
pipeline.validate()
pipeline.start()

# Subscribe to the result stream after start()
pipeline.result_bus.subscribe(
    lambda r: print(f"Frame {r.frame_idx}: {len(r.detections)} detections")
)

input("Press Enter to stop...\n")
pipeline.stop()
```

---

## YAML-based pipeline (recommended)

Declare the topology in `pipeline.yaml` and let `cvpipe.build()` handle discovery,
construction, and wiring. No manual component instantiation required:

```yaml
# pipeline.yaml
pipeline:
  source: webcam_source
  source_config:
    device_index: 0

  components:
    - module: preprocessor
      id: prep
      config: {device: cuda}

    - module: yolo_detector
      id: detector
      config: {weights: yolov8n.pt, confidence: 0.45}

    - module: tracker
      id: tracker

    - module: result_assembler
      id: assembler

  branches:
    - id: fast_mode
      trigger: "mode == 'fast'"
      inject_after: prep
      merge_before: tracker
      exclusive: true
      components:
        - module: lightweight_detector
          id: fast_det
          config: {confidence: 0.3}
```

```python
from cvpipe import build
from pathlib import Path

pipeline = build(
    config_path=Path("pipeline.yaml"),
    components_dir=Path("myapp/"),   # discovers both components/ and sources/ subdirs
)
pipeline.validate()
pipeline.start()
```

`build()` discovers all `Component` and `FrameSource` subclasses from subdirectories
under `components_dir`, instantiates them with the config from the YAML, and returns a
ready-to-validate `Pipeline`.

---

## What a production pipeline looks like

A detection server with two runtime modes selected by a mode-decision component:

```
[Preprocessor]
     |
     +-- mode='fast' -----> [LightweightDetector] ---------------+
     |                                                           |
     +-- mode='full' -----> [FasterRCNN] --> [FeatClassifier] ---+
                                                                 |
                                                            [Tracker]
                                                                 |
                                                         [ResultAssembler]
                                                                 |
                                                            [ResultBus] -> WebSocket
```

Both branch points live in `pipeline.yaml`. `Preprocessor` writes `frame.meta["mode"]`.
No component contains guard clauses.

---

## Project layout

```
myproject/
├── components/
│   ├── preprocessor/
│   │   └── __init__.py      # exports one Component subclass
│   │   └── preprocessor.py  # pre-processor main code
│   ├── yolo_detector/
│   │   ├── __init__.py
│   │   └── model.py         # private helpers — not exported, discovery skips them
│   └── tracker/
│   │   └── __init__.py
│   │   └── tracker.py       # tracking code
│   └── webcam_source/
│       └── __init__.py
│       └── web_cam_src.py   # web-cam code 
├── pipeline.yaml            # pipeline graph
└── server.py                # web-app backend
```

---

## Documentation

|                                                     | |
|-----------------------------------------------------|---|
| [Concepts](./concepts.md)                           | Frame, slots, buses, threads, DAGs, branches |
| [Building Components](./building_components.md) | Implement and test components |
| [Building Pipelines](./building_pipelines.md)   | YAML topology, `build()`, lifecycle |
| [Observability](./observability.md)            | Probes, events, FastAPI streaming |
| [End-to-End Example](./example_webcam_api.md)  | Full annotated detection server |
| [API Reference](./api_reference.md)            | Complete class and method reference |

---

## Requirements

- Python 3.11+
- PyTorch (optional — required only for GPU tensor slots)

## License

MIT
