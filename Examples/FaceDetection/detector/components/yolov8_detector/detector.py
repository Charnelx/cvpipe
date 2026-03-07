import logging
import os

import cv2
from cvpipe import Component, Frame, SlotSchema
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class Yolov8Detector(Component):
    """
    Runs YOLOv8 face detection using onnxruntime.
    Expects input shape 640x640 by default. Outputs bounding boxes.
    """

    INPUTS = [SlotSchema("frame_bgr", np.ndarray, description="Original BGR frame")]
    OUTPUTS = [
        SlotSchema("proposals_xyxy", np.ndarray, description="Bounding boxes in logical coords")
    ]
    SUBSCRIBES = []

    def __init__(
        self,
        model_path: str = "models/yolov8n-face.onnx",
        conf_thres: float = 0.5,
        iou_thres: float = 0.45,
    ):
        super().__init__()
        self._model_path = model_path
        self._conf_thres = conf_thres
        self._iou_thres = iou_thres
        self._session = None
        self._input_name = None

    def setup(self) -> None:
        if not os.path.exists(self._model_path):
            raise FileNotFoundError(f"ONNX model not found at {self._model_path}")

        # Suppress DRM device discovery C++ warnings
        ort.set_default_logger_severity(3)

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(self._model_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        logger.info(
            f"Loaded ONNX model {self._model_path} with providers: {self._session.get_providers()}"
        )

    def teardown(self) -> None:
        self._session = None

    def process(self, frame: Frame) -> None:
        # Hot path: No I/O or big allocations
        bgr = frame.slots.get("frame_bgr")
        if bgr is None:
            return

        h, w = bgr.shape[:2]

        # Basic letterbox/resize to 640x640 (simplified for sample app)
        img = cv2.resize(bgr, (640, 640))
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (640, 640), swapRB=True)

        outputs = self._session.run(None, {self._input_name: blob})[0]
        # outputs typically shape (1, 5, 8400) for YOLOv8n single-class
        outputs = outputs[0].T  # (8400, C)

        # Standard YOLOv8 ONNX: columns are [cx, cy, w, h, conf]
        boxes = outputs[:, :4]
        scores = outputs[:, 4]  # We assume single class (faces)

        # Filter by confidence
        mask = scores > self._conf_thres
        boxes = boxes[mask]
        scores = scores[mask]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"[Yolov8Detector] Raw output shape: {outputs.shape}, "
                f"Conf threshold: {self._conf_thres}"
            )
            logger.debug(f"[Yolov8Detector] Boxes above threshold: {len(boxes)}")

        if len(boxes) == 0:
            frame.meta["proposals_xyxy"] = np.empty((0, 4))
            return

        # cxcywh -> x1y1x2y2 in 640x640 space
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        # Scale back to original image size
        x1 = x1 * (w / 640.0)
        y1 = y1 * (h / 640.0)
        x2 = x2 * (w / 640.0)
        y2 = y2 * (h / 640.0)

        xyxy = np.stack((x1, y1, x2, y2), axis=1)

        # NMS
        indices = cv2.dnn.NMSBoxes(
            xyxy.tolist(), scores.tolist(), self._conf_thres, self._iou_thres
        )
        if len(indices) > 0:
            valid_xyxy = xyxy[indices.flatten()]
        else:
            valid_xyxy = np.empty((0, 4))

        frame.meta["proposals_xyxy"] = valid_xyxy
