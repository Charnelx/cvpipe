import asyncio
import logging
from pathlib import Path
from typing import Set

from cvpipe import build
from cvpipe.dashboard import enable_dashboard
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
# Suppress the noisy DRM warning in onnxruntime
logging.getLogger("onnxruntime").setLevel(logging.ERROR)

from detector.events import InferenceToggleEvent

app = FastAPI()

CONFIG_PATH = Path("detector/pipelines/default.yaml")
COMPONENTS_DIR = Path("detector/components")

pipeline = None


class StreamManager:
    def __init__(self):
        self.queues: Set[asyncio.Queue] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    def push(self, result):
        if not self.loop:
            return
        jpeg_bytes = result.jpeg_bytes
        if not jpeg_bytes:
            return

        for q in list(self.queues):
            try:
                self.loop.call_soon_threadsafe(q.put_nowait, jpeg_bytes)
            except asyncio.QueueFull:
                pass
            except Exception:
                pass


stream_manager = StreamManager()

pipeline = None


@app.on_event("startup")
def startup_event():
    global pipeline
    try:
        stream_manager.loop = asyncio.get_running_loop()
    except RuntimeError:
        pass

    pipeline = build(CONFIG_PATH, COMPONENTS_DIR)
    pipeline.result_bus.subscribe(lambda result: stream_manager.push(result))
    pipeline.validate()
    enable_dashboard(pipeline)
    pipeline.start()


@app.on_event("shutdown")
def shutdown_event():
    if pipeline is not None:
        pipeline.stop()


class ToggleRequest(BaseModel):
    enabled: bool


@app.post("/api/inference")
def toggle_inference(req: ToggleRequest):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not started")
    pipeline.event_bus.publish(InferenceToggleEvent(enabled=req.enabled))
    return {"status": "success", "inference_enabled": req.enabled}


async def mjpeg_streamer():
    q = asyncio.Queue(maxsize=5)
    stream_manager.queues.add(q)
    try:
        while pipeline is not None and pipeline.is_running:
            try:
                jpeg_bytes = await asyncio.wait_for(q.get(), timeout=1.0)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n")
            except asyncio.TimeoutError:
                continue
    finally:
        stream_manager.queues.discard(q)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        mjpeg_streamer(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


# Mount frontend files under /
app.mount("/", StaticFiles(directory="detector/static", html=True), name="static")
