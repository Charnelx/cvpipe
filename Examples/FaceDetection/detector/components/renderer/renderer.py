import cv2
from cvpipe import Component, Frame, SlotSchema
import numpy as np


class Renderer(Component):
    """
    Draws bounding boxes onto the frame and encodes it as JPEG.
    Places the raw bytes into frame.meta["jpeg_bytes"] so the Scheduler's
    default _extract_result() puts them in FrameResult.jpeg_bytes.
    """

    INPUTS = [
        SlotSchema("frame_bgr", np.ndarray, description="Original BGR frame")
        # Notice we don't strictly *require* proposals_xyxy. If inference is off,
        # it won't be there, so we handle its absence gracefully.
    ]
    OUTPUTS = [SlotSchema("jpeg_bytes", bytes, description="Encoded JPEG for MJPEG stream")]
    SUBSCRIBES = []

    def __init__(self, quality: int = 80):
        super().__init__()
        self._quality = quality

    def process(self, frame: Frame) -> None:
        bgr = frame.slots.get("frame_bgr")
        if bgr is None:
            return

        # Optional: proposals_xyxy (only exists if inference ran)
        boxes = frame.meta.get("proposals_xyxy")

        # Draw boxes if present
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Fast encoding
        ret, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        if ret:
            frame.meta["jpeg_bytes"] = buf.tobytes()
