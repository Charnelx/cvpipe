# cvpipe/event.py
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Type

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Event:
    """
    Base class for all events in the cvpipe event system.

    All concrete event types must be frozen dataclasses that inherit
    from Event. The ``ts`` field is auto-populated with the current
    monotonic timestamp at instantiation.

    Rules for subclasses:
    - Must be decorated with @dataclass(frozen=True)
    - Must call super().__init__() implicitly via dataclass inheritance
    - All fields must be immutable (no lists/dicts — use tuples)
    - Must be importable without side effects (used in SUBSCRIBES declarations)

    Example::

        @dataclass(frozen=True)
        class ClassRegisteredEvent(Event):
            class_id:  int
            label:     str
            threshold: float
    """

    ts: float = field(default_factory=time.monotonic, compare=False, init=False)


@dataclass(frozen=True)
class ComponentErrorEvent(Event):
    """
    Emitted by Scheduler when a component raises an exception in process().
    The frame that caused the error is dropped; the pipeline continues.

    Fields
    ------
    component_id : str
        The ``id`` field from the YAML pipeline spec for the failing component.
    message : str
        str(exception)
    traceback : str
        Full formatted traceback as a single string.
    frame_idx : int
        Index of the frame that triggered the error.
    """

    component_id: str
    message: str
    traceback: str
    frame_idx: int


@dataclass(frozen=True)
class ComponentMetricEvent(Event):
    """
    Emitted by Scheduler after every component processes a frame.
    Used for latency profiling and performance dashboards.

    Fields
    ------
    component_id : str
        Component identifier from YAML.
    latency_ms : float
        Wall-clock time taken by component.process() in milliseconds.
    frame_idx : int
        Frame this measurement belongs to.
    """

    component_id: str
    latency_ms: float
    frame_idx: int


@dataclass(frozen=True)
class FrameDroppedEvent(Event):
    """
    Emitted by Scheduler when a frame is dropped before processing completes.

    Reasons
    -------
    backpressure
        The previous frame is still being processed when a new frame
        arrives (inference is slower than capture rate). The new frame
        is skipped.
    component_error
        A component raised an exception; the frame is abandoned.
    source_stall
        The FrameSource returned None (camera disconnect, end of file).
    """

    reason: str  # Literal["backpressure", "component_error", "source_stall"]
    frame_idx: int


@dataclass(frozen=True)
class PipelineStateEvent(Event):
    """
    Emitted by Pipeline on lifecycle transitions.

    State machine:
        (none) → starting → running → stopping → stopped
                                    ↘ error

    The "reset" state is emitted by pipeline.reset() to signal that
    all components have been reset without stopping the pipeline.
    """

    state: str  # Literal["starting", "running", "stopping", "stopped", "error", "reset"]
    detail: str = ""


class EventBus:
    """
    Pub/sub event channel.

    Low-frequency management and health signals flow through this bus.
    It is NOT for per-frame inference data (use ResultBus for that).

    Threading model:
        - publish() → non-blocking, safe from any thread
        - subscribe() → call during setup, before start()
        - Handlers called from _dispatch_thread (not streaming thread,
          not main thread)
        - Handlers must be thread-safe w.r.t. any state they touch

    Usage::

        bus = EventBus()
        bus.subscribe(ClassRegisteredEvent, scorer.on_event)
        bus.start()

        # From any thread:
        bus.publish(ClassRegisteredEvent(class_id=1, label="duck", ...))

        # On shutdown:
        bus.stop()
    """

    _SENTINEL = object()

    def __init__(self, maxsize: int = 256) -> None:
        """
        Parameters
        ----------
        maxsize : int
            Maximum number of events queued before publish() blocks.
            Default 256 is generous for management events; increase if
            publishing many rapid events during batch registration.
        """
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=maxsize)
        self._handlers: dict[type, list[Callable[[Event], None]]] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False

    def subscribe(
        self,
        event_type: Type[Event],
        handler: Callable[[Event], None],
    ) -> None:
        """
        Register ``handler`` to be called when an event of ``event_type``
        (or a subclass) is published.

        Parameters
        ----------
        event_type : type
            The event class to subscribe to. Exact match only —
            subscribing to Event base does NOT receive all events.
            To receive all events of a family, subscribe to each type.
        handler : callable
            Called with the event as its sole argument.
            Must not block for more than ~5ms.
            Must be thread-safe.

        Notes
        -----
        Calling subscribe() after start() is safe but creates a brief
        window where events of this type are not delivered. For
        guaranteed delivery, subscribe before start().
        """
        with self._lock:
            self._handlers.setdefault(event_type, []).append(handler)

    def publish(self, event: Event) -> None:
        """
        Enqueue ``event`` for delivery to all matching subscribers.

        Non-blocking. If the queue is full, the oldest event is
        discarded and a warning is logged. Never raises.
        Safe to call from any thread.
        """
        if not self._running:
            return
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                pass
            logger.warning(
                "[EventBus] Queue full (%d), dropped oldest event to admit %s",
                self._queue.maxsize,
                type(event).__name__,
            )

    def start(self) -> None:
        """Start the dispatch thread. Idempotent — safe to call multiple times."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._dispatch_loop,
            name="EventBus-dispatch",
            daemon=True,
        )
        self._thread.start()
        logger.debug("[EventBus] Started")

    def stop(self, timeout: float = 2.0) -> None:
        """
        Signal the dispatch thread to stop and wait for it to finish.

        Parameters
        ----------
        timeout : float
            Seconds to wait for the dispatch thread to drain remaining
            events and exit. Events still in the queue at timeout are lost.
        """
        if not self._running:
            return
        self._running = False
        try:
            self._queue.put_nowait(self._SENTINEL)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        logger.debug("[EventBus] Stopped")

    def _dispatch_loop(self) -> None:
        while True:
            try:
                event = self._queue.get(timeout=0.1)
            except queue.Empty:
                if not self._running:
                    break
                continue

            if event is self._SENTINEL:
                break

            with self._lock:
                handlers = list(self._handlers.get(type(event), []))

            for handler in handlers:
                try:
                    handler(event)
                except Exception:
                    logger.exception(
                        "[EventBus] Handler %r raised for event %s — skipping",
                        handler,
                        type(event).__name__,
                    )

    def handler_count(self, event_type: type) -> int:
        """Return number of handlers registered for event_type. For testing."""
        with self._lock:
            return len(self._handlers.get(event_type, []))
