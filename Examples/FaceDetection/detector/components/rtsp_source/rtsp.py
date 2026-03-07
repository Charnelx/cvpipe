import logging
import time
from typing import Any

import cv2
from cvpipe import FrameSource

logger = logging.getLogger(__name__)


class RtspSource(FrameSource):
    """
    OpenCV-based RTSP FrameSource.
    Yields (bgr_array, ts) tuples.
    """

    def __init__(self, rtsp_url: str):
        self._rtsp_url = rtsp_url
        self._cap = None

    def setup(self) -> None:
        logger.info(f"Waiting for RTSP stream: {self._rtsp_url}")
        self._last_reconnect = 0.0

    def teardown(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def next(self) -> tuple[Any, float] | None:
        now = time.monotonic()
        
        # Reconnection logic with 2 second backoff
        if self._cap is None or not self._cap.isOpened():
            if now - getattr(self, "_last_reconnect", 0.0) < 2.0:
                return None
            self._last_reconnect = now
            logger.debug(f"Attempting to connect to {self._rtsp_url}...")
            self._cap = cv2.VideoCapture(self._rtsp_url)
            if not self._cap.isOpened():
                return None
            logger.info("Successfully connected to RTSP stream.")

        ret, frame = self._cap.read()

        ts = time.monotonic()
        if not ret:
            logger.warning("Failed to read frame from RTSP stream, releasing connection.")
            self._cap.release()
            self._cap = None
            return None

        return frame, ts
