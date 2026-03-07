# cvpipe/component.py
from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from .frame import Frame, SlotSchema
from .event import Event, EventBus

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class Component(ABC):
    """
    Abstract base class for all cvpipe pipeline components.

    A component represents one processing stage in the pipeline DAG.
    It declares what data it reads (INPUTS), what data it produces
    (OUTPUTS), and what events it reacts to (SUBSCRIBES).

    The framework guarantees:
    - setup() is called exactly once before the first process() call
    - process() is called only from the streaming thread
    - on_event() is called only from the event dispatch thread
    - teardown() is called exactly once after the last process() call

    Subclass requirements:
    - Declare INPUTS, OUTPUTS (may be empty lists)
    - Implement process()
    - Optionally override on_event(), setup(), teardown()
    - If holding mutable state accessed by both process() and on_event(),
      protect it with self._lock (a threading.Lock)

    Class attributes (set by framework, do not set manually):
    - _component_id: str — the "id" from the YAML spec
    - _event_bus: EventBus — set at pipeline.start() time

    Example::

        class MyProposer(Component):
            INPUTS = [
                SlotSchema("frame_bgr", torch.uint8, (None, None, 3), "gpu"),
            ]
            OUTPUTS = [
                SlotSchema("proposals_xyxy", torch.float32, (None, 4), "gpu",
                           coord_system="xyxy"),
            ]
            SUBSCRIBES = []

            def __init__(self, confidence: float = 0.5) -> None:
                super().__init__()
                self._confidence = confidence
                self._model = None  # loaded in setup()

            def setup(self) -> None:
                self._model = load_model(self._confidence)

            def process(self, frame: Frame) -> None:
                bgr = frame.slots["frame_bgr"]
                boxes = self._model.run(bgr)
                frame.slots["proposals_xyxy"] = boxes

            def teardown(self) -> None:
                self._model = None
    """

    # ── Class-level contract declarations ──────────────────────────────────────
    # These are read by the framework at assembly time.
    # Subclasses override these as class attributes (not instance attributes).

    INPUTS: ClassVar[list[SlotSchema]] = []
    """Slots this component reads from Frame.slots or Frame.meta."""

    OUTPUTS: ClassVar[list[SlotSchema]] = []
    """Slots this component writes to Frame.slots or Frame.meta."""

    SUBSCRIBES: ClassVar[list[type[Event]]] = []
    """Event types this component handles in on_event()."""

    def __init__(self) -> None:
        """
        Base __init__. Subclasses must call super().__init__().

        Do not load models or open resources here — use setup() instead.
        __init__ is called during pipeline assembly (before validation),
        so failures here prevent the pipeline from starting clearly.
        """
        self._component_id: str = ""
        """The YAML-assigned identifier for this component instance."""
        self._event_bus: EventBus | None = None
        """EventBus instance, set by Pipeline at start() time."""
        self._lock = threading.Lock()
        """
        threading.Lock for protecting shared state between process()
        (streaming thread) and on_event() (event dispatch thread).

        Only acquire this lock around the minimal critical section —
        do not hold it during model inference or other long operations.
        """

    # ── Required interface ─────────────────────────────────────────────────────

    @abstractmethod
    def process(self, frame: Frame) -> None:
        """
        Process one frame. Called from the streaming thread.

        Read input slots from frame.slots / frame.meta.
        Write output slots to frame.slots / frame.meta.
        All operations must be in-place on the frame — do not return data.

        Constraints (HOT PATH):
        - No I/O (no file reads, no DB queries, no network calls)
        - No logging at INFO level or above (use DEBUG)
        - No large memory allocations — reuse buffers
        - No blocking lock acquisitions (>100µs)
        - Must complete within the frame budget (~33ms at 30fps)

        Parameters
        ----------
        frame : Frame
            The current frame workspace. Mutable. Not thread-safe.
        """

    # ── Optional interface ─────────────────────────────────────────────────────

    def setup(self) -> None:
        """
        Called once by the framework before the first process() call.

        Use for: loading models, opening file handles, allocating GPU
        memory buffers, connecting to external services.

        Called in the streaming thread before the frame loop starts.
        Blocking here is acceptable — setup is not time-critical.

        The framework calls setup() on all components in topological
        order (upstream first).
        """

    def teardown(self) -> None:
        """
        Called once by the framework after the pipeline stops.

        Use for: releasing GPU memory, closing file handles,
        disconnecting from services.

        The framework calls teardown() in reverse topological order
        (downstream first).
        """

    def reset(self) -> None:
        """
        Called by Pipeline.reset() to clear per-session application state.

        Implement to reset any state that should not persist across sessions:
        cached results, per-session registries, tracker histories, rolling
        averages, cross-frame stashes.

        Do NOT release GPU resources (models, pre-allocated tensors) — that is
        teardown()'s responsibility. reset() is specifically for application-layer
        state, not infrastructure-layer resources.

        Thread safety:
            reset() is called from the main/API thread, NOT the streaming thread.
            The Scheduler is paused before reset() is called and resumes after.
            process() is guaranteed not to execute concurrently with reset().
            If your component holds a lock that process() acquires, you do NOT
            need to acquire it in reset() — there is no concurrent process() call.
            If your component holds state also accessed by on_event() (event
            dispatch thread), you MUST acquire self._lock in reset().

        Ordering:
            Pipeline.reset() calls reset() on components in the same topological
            order as setup() (upstream first). If component B depends on state
            from component A's reset, declare A before B in the pipeline.

        Default implementation:
            No-op. Components with no resettable state do not need to override this.
        """

    def on_event(self, event: Event) -> None:
        """
        Called when an event matching a type in SUBSCRIBES is published.
        Called from the event dispatch thread (not the streaming thread).

        Override this method to handle incoming events. Use self._lock
        to protect any state also accessed in process().

        The default implementation is a no-op.

        Parameters
        ----------
        event : Event
            The published event. Its concrete type will be one of the
            types listed in self.SUBSCRIBES.
        """

    # ── Framework-provided utilities ───────────────────────────────────────────

    def emit(self, event: Event) -> None:
        """
        Publish an event to the EventBus.

        Safe to call from process() (streaming thread) or on_event()
        (event dispatch thread). Never blocks.

        Typically used to emit ComponentErrorEvent or custom
        application-specific health events.

        Parameters
        ----------
        event : Event
            Event to publish. Must be a frozen dataclass subclass of Event.
        """
        if self._event_bus is not None:
            self._event_bus.publish(event)
        else:
            logger.debug(
                "[%s] emit() called before EventBus was injected — dropped: %s",
                self._component_id or type(self).__name__,
                type(event).__name__,
            )

    # ── Introspection helpers (used by framework, not by subclasses) ───────────

    @property
    def component_id(self) -> str:
        """The YAML-assigned identifier for this component instance."""
        return self._component_id

    def input_slot_names(self) -> set[str]:
        """Set of slot names declared in INPUTS."""
        return {s.name for s in self.INPUTS}

    def output_slot_names(self) -> set[str]:
        """Set of slot names declared in OUTPUTS."""
        return {s.name for s in self.OUTPUTS}

    def get_input_schema(self, name: str) -> SlotSchema | None:
        """Return the SlotSchema for the named input slot, or None."""
        return next((s for s in self.INPUTS if s.name == name), None)

    def get_output_schema(self, name: str) -> SlotSchema | None:
        """Return the SlotSchema for the named output slot, or None."""
        return next((s for s in self.OUTPUTS if s.name == name), None)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"id={self._component_id!r}, "
            f"inputs={[s.name for s in self.INPUTS]}, "
            f"outputs={[s.name for s in self.OUTPUTS]})"
        )
