from cvpipe import Component, Frame, SlotSchema
import numpy as np

from detector.events import InferenceToggleEvent


class Router(Component):
    """
    Responsible for emitting the routing decision to frame.meta
    so the exclusive branch can correctly evaluate whether to run inference.
    """

    INPUTS = []
    OUTPUTS = [
        SlotSchema("inference_enabled", bool, description="Whether to run face detection"),
        SlotSchema("frame_bgr", np.ndarray, description="Promotes frame_raw dict to bgr slot"),
    ]
    SUBSCRIBES = [InferenceToggleEvent]

    def __init__(self):
        super().__init__()
        self._inference_enabled = False

    def on_event(self, event) -> None:
        if isinstance(event, InferenceToggleEvent):
            with self._lock:
                self._inference_enabled = event.enabled

    def process(self, frame: Frame) -> None:
        with self._lock:
            enabled = self._inference_enabled

        frame.meta["inference_enabled"] = enabled

        # Safely extract raw array from scheduler and write it to a slot if available
        # The Scheduler places the payload into frame.meta["frame_raw"]
        raw_bgr = frame.slots.get("frame_raw")
        if raw_bgr is not None:
            frame.slots["frame_bgr"] = raw_bgr
