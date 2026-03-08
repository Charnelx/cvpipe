# cvpipe/__init__.py
# Everything a component author needs is importable from cvpipe directly.

from .bus import ResultBus, FrameResult
from .builder import build
from .bridge import AsyncQueueBridge
from .config import PipelineConfig, ComponentSpec, BranchSpec
from .component import Component
from .event import (
    Event,
    EventBus,
    ComponentErrorEvent,
    ComponentMetricEvent,
    FrameDroppedEvent,
    PipelineStateEvent,
)
from .errors import (
    CvPipeError,
    PipelineConfigError,
    ContractError,
    SlotNotFoundError,
    CoordinateSystemError,
    DuplicateSlotWriterError,
    ComponentError,
    ComponentNotFoundError,
    AmbiguousComponentError,
    SlotValidationError,
)
from .frame import Frame, SlotSchema
from .probe import Probe, DiagnosticsProbe, FrameDiagnostics, ComponentTrace
from .pipeline import Pipeline
from .scheduler import FrameSource
from .registry import ComponentRegistry

__version__ = "0.1.0"
__all__ = [
    "Frame",
    "SlotSchema",
    "Component",
    "Event",
    "EventBus",
    "ComponentErrorEvent",
    "ComponentMetricEvent",
    "FrameDroppedEvent",
    "PipelineStateEvent",
    "ResultBus",
    "FrameResult",
    "Probe",
    "DiagnosticsProbe",
    "FrameDiagnostics",
    "ComponentTrace",
    "Pipeline",
    "FrameSource",
    "PipelineConfig",
    "ComponentSpec",
    "BranchSpec",
    "ComponentRegistry",
    "AsyncQueueBridge",
    "CvPipeError",
    "PipelineConfigError",
    "ContractError",
    "SlotNotFoundError",
    "CoordinateSystemError",
    "DuplicateSlotWriterError",
    "ComponentError",
    "ComponentNotFoundError",
    "AmbiguousComponentError",
    "SlotValidationError",
]
