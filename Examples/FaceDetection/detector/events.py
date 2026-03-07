from dataclasses import dataclass

from cvpipe import Event


@dataclass(frozen=True)
class InferenceToggleEvent(Event):
    """
    Emitted when the user toggles inference on or off via the web interface.
    """

    enabled: bool
