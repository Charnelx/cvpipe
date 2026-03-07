# cvpipe/__init__.py
# Everything a component author needs is importable from cvpipe directly.

from .frame import Frame, SlotSchema
from .component import Component
from .event import (
    Event,
    EventBus,
    ComponentErrorEvent,
    ComponentMetricEvent,
    FrameDroppedEvent,
    PipelineStateEvent,
)
from .bus import ResultBus, FrameResult
from .probe import Probe, DiagnosticsProbe, FrameDiagnostics, ComponentTrace
from .pipeline import Pipeline
from .scheduler import FrameSource
from .config import PipelineConfig, ComponentSpec, BranchSpec
from .registry import ComponentRegistry
from .bridge import AsyncQueueBridge
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
)

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
]
