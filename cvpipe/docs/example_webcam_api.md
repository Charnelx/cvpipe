# End-to-End Example: Webcam Detection Server

A complete, working object detection server built on cvpipe. This example shows:

- Webcam frame capture via a `FrameSource`
- YOLOv8 detection and IoU tracking as separate components with typed slot contracts
- A `ResultAssembler` that writes the `FrameResult` keys the framework expects
- A `DiagnosticsProbe` and a `LatencyProbe` for per-component timing
- Result streaming over WebSocket via `AsyncQueueBridge`
- A REST endpoint to change the confidence threshold at runtime via `EventBus`
- The `build()` function for zero-boilerplate pipeline assembly

---

## Project layout

```
detection_server/
├── components/
│   ├── preprocessor/
│   │   └── __init__.py
│   ├── yolo_detector/
│   │   └── __init__.py
│   ├── iou_tracker/
│   │   └── __init__.py
│   └── result_assembler/
│   │   └── __init__.py
│   └── webcam_source/
│       └── __init__.py
├── events.py
├── pipeline.yaml
├── server.py
└── static/
    └── index.html
```

---

## events.py

Shared event definitions. Import from here everywhere — no circular dependencies.

```python
# events.py
from dataclasses import dataclass
from cvpipe import Event

@dataclass(frozen=True)
class ConfidenceChangedEvent(Event):
    """Published by the REST API when the user adjusts the detection threshold."""
    value: float
```

---

## sources/webcam_source/\_\_init\_\_.py

```python
import time
import cv2
from cvpipe import FrameSource


class WebcamSource(FrameSource):
    """
    Reads BGR frames from a local webcam.

    Returns each frame as a raw numpy array. The Scheduler places it in
    frame.slots["frame_raw"] — the first component (Preprocessor) consumes
    it from there.

    Returns None when the camera stalls, signalling the Scheduler to retry.
    """

    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480):
        self._device_index = device_index
        self._width        = width
        self._height       = height
        self._cap          = None

    def setup(self) -> None:
        self._cap = cv2.VideoCapture(self._device_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open webcam at index {self._device_index}")

    def next(self):
        ok, frame = self._cap.read()
        return (frame, time.monotonic()) if ok else None

    def teardown(self) -> None:
        if self._cap:
            self._cap.release()
```

---

## components/preprocessor/\_\_init\_\_.py

```python
import cv2
import torch
from cvpipe import Component, Frame, SlotSchema


class Preprocessor(Component):
    """
    Resize the raw source frame and convert it to a uint8 BGR tensor.

    Reads:  frame.slots["frame_raw"]  — raw numpy array from WebcamSource
    Writes: frame.slots["frame_bgr"]  — uint8 (H, W, 3) CPU tensor
            frame.meta["mode"]        — routing key for the fast/full branch

    frame_raw is injected by the Scheduler and is NOT declared in INPUTS.
    """

    INPUTS = []   # frame_raw is injected by the Scheduler, not produced by any component
    OUTPUTS = [
        SlotSchema("frame_bgr", torch.uint8, (None, None, 3), "cpu",
                   description="Resized BGR frame for detection and JPEG encoding"),
    ]
    SUBSCRIBES = []

    def __init__(self, width: int = 640, height: int = 480, fast_mode: bool = False):
        super().__init__()
        self._w         = width
        self._h         = height
        self._fast_mode = fast_mode

    def process(self, frame: Frame) -> None:
        raw = frame.slots["frame_raw"]
        if raw.shape[:2] != (self._h, self._w):
            raw = cv2.resize(raw, (self._w, self._h))
        frame.slots["frame_bgr"] = torch.from_numpy(raw.copy())
        frame.meta["mode"] = "fast" if self._fast_mode else "full"
```

---

## components/yolo_detector/\_\_init\_\_.py

```python
import torch
from cvpipe import Component, Frame, SlotSchema

# Import from events.py — both main process and branch components share it
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from events import ConfidenceChangedEvent


class YOLODetector(Component):
    """
    Run YOLOv8 on frame.slots["frame_bgr"] and write bounding boxes + a detection list.

    Subscribes to ConfidenceChangedEvent so the confidence threshold can be updated at
    runtime via the REST API without restarting the pipeline.

    Swapping to a different detector (e.g. Faster R-CNN) means:
      1. Create a new component directory (e.g. faster_rcnn_detector/)
      2. Declare the same INPUTS/OUTPUTS schema
      3. Change pipeline.yaml: module: faster_rcnn_detector
    No other file needs to change.
    """

    INPUTS = [
        SlotSchema("frame_bgr", torch.uint8, (None, None, 3), "cpu",
                   description="BGR frame from Preprocessor"),
    ]
    OUTPUTS = [
        SlotSchema("boxes_xyxy", torch.float32, (None, 4), "cpu",
                   coord_system="xyxy",
                   description="Raw detection boxes, absolute pixel coords"),
        SlotSchema("class_ids",  torch.int64,   (None,),   "cpu",
                   description="Class index per detection"),
        SlotSchema("scores",     torch.float32, (None,),   "cpu",
                   description="Confidence score per detection"),
    ]
    SUBSCRIBES = [ConfidenceChangedEvent]

    def __init__(self, weights: str = "yolov8n.pt", confidence: float = 0.45):
        super().__init__()
        self._weights    = weights
        self._confidence = confidence   # protected by self._lock
        self._model      = None

    def setup(self) -> None:
        from ultralytics import YOLO
        self._model = YOLO(self._weights)

    def on_event(self, event) -> None:
        if isinstance(event, ConfidenceChangedEvent):
            with self._lock:
                self._confidence = event.value

    def process(self, frame: Frame) -> None:
        with self._lock:
            conf = self._confidence      # snapshot — release before inference

        results = self._model(
            frame.slots["frame_bgr"].numpy(), conf=conf, verbose=False
        )
        boxes = results[0].boxes

        frame.slots["boxes_xyxy"] = boxes.xyxy.cpu()
        frame.slots["class_ids"]  = boxes.cls.long().cpu()
        frame.slots["scores"]     = boxes.conf.cpu()
        frame.meta["class_names"] = [self._model.names[int(c)] for c in boxes.cls]
```

---

## components/iou_tracker/\_\_init\_\_.py

```python
import torch
from cvpipe import Component, Frame, SlotSchema


class IoUTracker(Component):
    """
    Assign consistent track IDs to detections across frames using IoU matching.

    Reads:  frame.slots["boxes_xyxy"]  — (N, 4) detection boxes from YOLODetector
            frame.slots["class_ids"]   — (N,)   class indices
            frame.slots["scores"]      — (N,)   confidence scores
            frame.meta["class_names"]  — list of class name strings
    Writes: frame.meta["tracks"]       — list of detection dicts with track_id
    """

    INPUTS = [
        SlotSchema("boxes_xyxy", torch.float32, (None, 4), "cpu", coord_system="xyxy"),
        SlotSchema("class_ids",  torch.int64,   (None,),   "cpu"),
        SlotSchema("scores",     torch.float32, (None,),   "cpu"),
    ]
    OUTPUTS    = []   # writes only to frame.meta["tracks"]
    SUBSCRIBES = []

    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 15):
        super().__init__()
        self._iou_threshold   = iou_threshold
        self._max_lost_frames = max_lost_frames
        self._next_id  = 0
        self._tracks: dict[int, dict] = {}

    def reset(self) -> None:
        """Clear all track state between sessions."""
        self._next_id = 0
        self._tracks  = {}

    def process(self, frame: Frame) -> None:
        boxes      = frame.slots["boxes_xyxy"]
        class_ids  = frame.slots["class_ids"]
        scores     = frame.slots["scores"]
        names      = frame.meta.get("class_names", [])

        if len(boxes) == 0:
            frame.meta["tracks"] = []
            return

        matched_ids = self._match(boxes.tolist(), frame.idx)
        frame.meta["tracks"] = [
            {
                "track_id": matched_ids[i],
                "label":    names[i] if i < len(names) else str(int(class_ids[i])),
                "score":    float(scores[i]),
                "box":      [round(v) for v in boxes[i].tolist()],
                "state":    "TRACKED",
            }
            for i in range(len(boxes))
        ]

    def _iou(self, a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / union if union > 0 else 0.0

    def _match(self, boxes, frame_idx) -> list[int]:
        ids, used = [], set()
        for box in boxes:
            best_id, best_iou = None, self._iou_threshold
            for tid, track in self._tracks.items():
                if tid in used:
                    continue
                iou = self._iou(box, track["box"])
                if iou > best_iou:
                    best_iou, best_id = iou, tid
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
            used.add(best_id)
            self._tracks[best_id] = {"box": box, "last_seen": frame_idx}
            ids.append(best_id)
        # Expire stale tracks
        self._tracks = {
            tid: t for tid, t in self._tracks.items()
            if frame_idx - t["last_seen"] <= self._max_lost_frames
        }
        return ids
```

---

## components/result\_assembler/\_\_init\_\_.py

```python
import cv2
import time
import torch
from collections import deque
from cvpipe import Component, Frame, SlotSchema


class ResultAssembler(Component):
    """
    Terminal component. Encodes the frame as JPEG and writes the three keys that
    the Scheduler reads to build a FrameResult for the ResultBus:

        frame.meta["jpeg_bytes"]   — bytes
        frame.meta["detections"]   — list[dict]
        frame.meta["result_meta"]  — dict

    Must run last in the pipeline. No component should write these keys before it.
    """

    INPUTS = [
        SlotSchema("frame_bgr", torch.uint8, (None, None, 3), "cpu"),
    ]
    OUTPUTS    = []
    SUBSCRIBES = []

    def __init__(self, jpeg_quality: int = 80):
        super().__init__()
        self._quality     = jpeg_quality
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
        frame.meta["detections"]  = frame.meta.get("tracks", [])
        frame.meta["result_meta"] = {
            "fps":             round(fps, 1),
            "detection_count": len(frame.meta.get("tracks", [])),
        }
```

---

## pipeline.yaml

```yaml
pipeline:
  source: webcam_source
  source_config:
    device_index: 0
    width: 640
    height: 480

  components:
    - module: preprocessor
      id: preprocessor
      config:
        width: 640
        height: 480

    - module: yolo_detector
      id: detector
      config:
        weights: yolov8n.pt
        confidence: 0.45

    - module: iou_tracker
      id: tracker
      config:
        iou_threshold: 0.3
        max_lost_frames: 15

    - module: result_assembler
      id: assembler
      config:
        jpeg_quality: 80

  branches:
    # When inference is disabled, skip detector and tracker.
    # The assembler still runs and produces a result with empty detections.
    - id: inference_off
      trigger: "inference_enabled == False"
      inject_after: preprocessor
      merge_before: assembler
      exclusive: true
      components: []
```

---

## server.py

```python
# server.py
import asyncio
import json
import logging
from collections import deque
from contextlib import asynccontextmanager
from threading import Lock

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from cvpipe import (
    build,
    AsyncQueueBridge,
    DiagnosticsProbe,
    FrameResult,
    ComponentErrorEvent,
    FrameDroppedEvent,
    PipelineStateEvent,
)
from cvpipe import Probe, Frame
from pathlib import Path

from events import ConfidenceChangedEvent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ── Shared state ──────────────────────────────────────────────────────────────

pipeline = None
bridge: AsyncQueueBridge | None = None
ws_clients: set[WebSocket] = set()


# ── Custom probe: per-component rolling latency ───────────────────────────────

class LatencyProbe(Probe):
    """Thread-safe rolling p50/p95/p99 latency tracker."""

    def __init__(self, window: int = 200):
        self._lock    = Lock()
        self._samples: dict[str, deque] = {}
        self._last_ts: dict[str, float] = {}

    def observe(self, frame: Frame, after_component: str) -> None:
        import time
        now = time.monotonic()
        if after_component in self._last_ts:
            ms = (now - self._last_ts[after_component]) * 1000
            with self._lock:
                self._samples.setdefault(after_component, deque(maxlen=200)).append(ms)
        self._last_ts[after_component] = now

    def snapshot(self) -> dict:
        with self._lock:
            result = {}
            for cid, buf in self._samples.items():
                s = sorted(buf)
                n = len(s)
                if n:
                    result[cid] = {
                        "p50": round(s[int(n * 0.50)], 2),
                        "p95": round(s[int(n * 0.95)], 2),
                        "p99": round(s[int(n * 0.99)], 2),
                        "count": n,
                    }
            return result

lat_probe = LatencyProbe()


# ── Application lifespan ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, bridge

    loop = asyncio.get_event_loop()

    # Assemble the pipeline from YAML — discovery, construction, wiring in one call
    pipeline = build(
        config_path=Path("pipeline.yaml"),
        components_dir=Path("detection_server/"),
    )

    # Subscribe to events BEFORE start()
    pipeline.event_bus.subscribe(PipelineStateEvent,
        lambda e: logger.info("pipeline: %s", e.state))
    pipeline.event_bus.subscribe(ComponentErrorEvent,
        lambda e: logger.error("[frame %d] %s: %s", e.frame_idx, e.component_id, e.message))
    pipeline.event_bus.subscribe(FrameDroppedEvent,
        lambda e: logger.debug("frame dropped: %s", e.reason))

    # Wire ResultBus → AsyncQueueBridge BEFORE start()
    bridge = AsyncQueueBridge(loop=loop, maxsize=8)
    pipeline.result_bus.subscribe(bridge.put)

    pipeline.start()

    # Attach probes AFTER start() — _scheduler does not exist before then
    pipeline._scheduler.add_probe(DiagnosticsProbe(), after=None)
    pipeline._scheduler.add_probe(lat_probe,          after=None)

    await bridge.start_consumer(_broadcast)

    yield   # server handles requests here

    await bridge.stop()
    loop.run_in_executor(None, pipeline.stop)


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Result broadcast ──────────────────────────────────────────────────────────

async def _broadcast(result: FrameResult) -> None:
    if not ws_clients:
        return

    payload = json.dumps({
        "frame_idx":       result.frame_idx,
        "fps":             result.meta.get("fps", 0.0),
        "detection_count": result.meta.get("detection_count", 0),
        "detections":      result.detections,
    })

    dead: set[WebSocket] = set()
    for ws in list(ws_clients):
        try:
            await ws.send_bytes(result.jpeg_bytes)
            await ws.send_text(payload)
        except Exception:
            dead.add(ws)
    ws_clients.difference_update(dead)


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    logger.info("client connected — %d total", len(ws_clients))
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        ws_clients.discard(ws)


# ── REST endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
def index():
    with open("static/index.html") as f:
        return HTMLResponse(f.read())

@app.get("/status")
def status():
    return {
        "running":     pipeline.is_running if pipeline else False,
        "clients":     len(ws_clients),
        "queue_depth": bridge.qsize if bridge else 0,
    }

@app.get("/metrics")
def metrics():
    """Per-component rolling latency percentiles from LatencyProbe."""
    return lat_probe.snapshot()

@app.post("/confidence/{value}")
def set_confidence(value: float):
    """Change detection confidence at runtime — no pipeline restart."""
    if not 0.0 < value < 1.0:
        return {"error": "must be between 0 and 1"}
    pipeline.event_bus.publish(ConfidenceChangedEvent(value=value))
    return {"confidence": value}
```

---

## static/index.html

```html
<!DOCTYPE html>
<html>
<head>
  <title>cvpipe — Detection</title>
  <style>
    body { background: #111; color: #eee; font-family: monospace; text-align: center; }
    canvas { border: 1px solid #333; margin-top: 10px; display: block; margin: 10px auto; }
    #stats { font-size: 13px; color: #aaa; margin: 6px; }
    button { margin: 3px; padding: 5px 12px; cursor: pointer; }
  </style>
</head>
<body>
  <h2>cvpipe detection</h2>
  <canvas id="view" width="640" height="480"></canvas>
  <div id="stats">connecting...</div>
  <div>
    <button onclick="post('/confidence/0.3')">conf 0.3</button>
    <button onclick="post('/confidence/0.5')">conf 0.5</button>
    <button onclick="post('/confidence/0.7')">conf 0.7</button>
  </div>
  <script>
    const canvas = document.getElementById("view");
    const ctx    = canvas.getContext("2d");
    const stats  = document.getElementById("stats");
    let pendingBlob = null;

    const ws = new WebSocket(`ws://${location.host}/stream`);
    ws.binaryType = "arraybuffer";

    ws.onmessage = ev => {
      if (ev.data instanceof ArrayBuffer) {
        pendingBlob = URL.createObjectURL(new Blob([ev.data], {type:"image/jpeg"}));
      } else {
        const meta = JSON.parse(ev.data);
        if (!pendingBlob) return;
        const img = new Image();
        img.onload = () => {
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
          URL.revokeObjectURL(pendingBlob);
          pendingBlob = null;
          ctx.strokeStyle = "#00dd44"; ctx.lineWidth = 2;
          ctx.font = "12px monospace"; ctx.fillStyle = "#00dd44";
          for (const d of meta.detections) {
            const [x1,y1,x2,y2] = d.box;
            ctx.strokeRect(x1, y1, x2-x1, y2-y1);
            ctx.fillText(`${d.label} ${(d.score*100).toFixed(0)}%`, x1+2, y1>14?y1-3:y1+13);
          }
          stats.textContent = `frame ${meta.frame_idx}  |  ${meta.fps} fps  |  ${meta.detection_count} objects`;
        };
        img.src = pendingBlob;
      }
    };

    function post(path) { fetch(path, {method:"POST"}); }
  </script>
</body>
</html>
```

---

## Running it

```bash
pip install git+https://github.com/Charnelx/cvpipe.git
pip install ultralytics opencv-python fastapi uvicorn

uvicorn detection_server.server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. The confidence buttons publish `ConfidenceChangedEvent` to
the EventBus — the detector updates its threshold on the next frame without any restart.

The `/metrics` endpoint returns rolling latency percentiles from `LatencyProbe`:

```bash
curl http://localhost:8000/metrics
# {"preprocessor":{"p50":0.8,"p95":1.1,"p99":1.4,"count":300},
#  "detector":{"p50":18.2,"p95":22.1,"p99":24.3,"count":300}, ...}
```

---

## Swapping the detector

To replace YOLOv8 with Faster R-CNN (or any other detector):

1. Create `components/faster_rcnn_detector/__init__.py` with the same `INPUTS`/`OUTPUTS`
   schema as `YOLODetector` — same slot names, same coordinate systems.
2. In `pipeline.yaml`, change `module: yolo_detector` to `module: faster_rcnn_detector`.
3. Restart the server.

The tracker, result assembler, and server are completely unchanged. This is the payoff
of declaring slot contracts — `IoUTracker` reads `boxes_xyxy`, `class_ids`, and `scores`
regardless of which component writes them.

---

## What's next

**Add `pipeline.reset()`.** If your use case has multiple sessions (switching what is
being detected), call `pipeline.reset()` between them. `IoUTracker.reset()` clears all
track history; `ResultAssembler.reset()` resets the FPS counter. Models stay loaded.

**Add an exclusive branch for inference disable.** The `pipeline.yaml` above already
includes the `inference_off` branch with an empty `components` list. To activate it,
write `frame.meta["inference_enabled"] = False` in `Preprocessor` when needed.

**Expose `DiagnosticsProbe` data.** The `DiagnosticsProbe` writes
`frame.meta["diagnostics"]` on every frame. The `result_meta` dict (and thus
`FrameResult.meta`) can include the diagnostics summary string — add it in
`ResultAssembler.process()` if you want it to appear in the WebSocket metadata stream.
