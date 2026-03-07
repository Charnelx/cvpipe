# cvpipe/bus.py
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """
    Serialisable summary of one completed pipeline frame.

    Produced by the final component in the pipeline (or a dedicated
    ResultAssembler component) and pushed to ResultBus. This is the
    only object that crosses the streaming-thread → transport boundary.

    Fields
    ------
    frame_idx : int
        Matches Frame.idx for correlation with diagnostics.
    ts : float
        Capture timestamp (matches Frame.ts).
    jpeg_bytes : bytes
        JPEG-encoded annotated frame. Empty bytes if encoding failed
        or inference is disabled.
    detections : list[dict]
        List of detection records. Each dict contains at minimum:
            {
                "label":    str,
                "track_id": int,
                "box":      [x1, y1, x2, y2],   # int pixels
                "score":    float,
                "state":    "TRACKED" | "LOST",
            }
        Serialisation is the application's responsibility — the framework
        does not dictate detection schema beyond this minimum.
    meta : dict[str, Any]
        Arbitrary per-frame metadata for transport: FPS, scan mode,
        proposal count, diagnostics. Application-defined.
    """

    frame_idx: int
    ts: float
    jpeg_bytes: bytes = b""
    detections: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


class ResultBus:
    """
    High-frequency per-frame result delivery channel.

    Lossy ring buffer: when the buffer is full, the oldest result is
    discarded. This ensures the streaming thread never blocks waiting
    for consumers.

    Each subscriber receives results in its own dedicated thread.
    A slow subscriber will miss frames (the buffer drops old frames)
    but will never slow down the pipeline.

    Usage::

        bus = ResultBus(capacity=4)
        bus.subscribe(ws_handler.on_result)
        bus.start()

        # From streaming thread:
        bus.push(FrameResult(frame_idx=42, ...))

        # On shutdown:
        bus.stop()
    """

    def __init__(self, capacity: int = 4) -> None:
        """
        Parameters
        ----------
        capacity : int
            Maximum number of FrameResults held before oldest is dropped.
            4 is sufficient for a single WebSocket consumer at 30fps —
            results are consumed faster than produced under normal load.
            Increase if multiple slow consumers are attached.
        """
        if capacity < 1:
            raise ValueError(f"ResultBus capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._buffer: list[FrameResult | None] = [None] * capacity
        self._buffer_seq: list[int] = [0] * capacity
        self._head = 0
        self._count = 0
        self._write_seq = 0
        self._lock = threading.Lock()
        self._not_empty: threading.Condition = threading.Condition(self._lock)
        self._subscribers: list[_SubscriberThread] = []
        self._running = False

    def subscribe(self, callback: Callable[[FrameResult], None]) -> None:
        """
        Register a callback to receive FrameResults.

        A dedicated thread is created for each subscriber.
        Call before start() for guaranteed delivery from the first frame.

        Parameters
        ----------
        callback : callable
            Called with each FrameResult. Must not block indefinitely.
            Exceptions are logged and do not stop the subscriber thread.
        """
        sub = _SubscriberThread(
            callback,
            self._not_empty,
            self._buffer,
            self._buffer_seq,
            self._capacity,
            self._lock,
        )
        self._subscribers.append(sub)
        if self._running:
            sub.start()

    def push(self, result: FrameResult) -> None:
        """
        Enqueue a result. Non-blocking. Drops oldest if full.

        Must be called from the streaming thread only.
        """
        with self._lock:
            self._write_seq += 1
            if self._count == self._capacity:
                logger.debug(
                    "[ResultBus] Buffer full — dropping oldest frame %d",
                    self._buffer[self._head].frame_idx if self._buffer[self._head] else -1,
                )
            else:
                self._count += 1
            self._buffer[self._head] = result
            self._buffer_seq[self._head] = self._write_seq
            self._head = (self._head + 1) % self._capacity
            self._not_empty.notify_all()

    def start(self) -> None:
        """Start subscriber threads. Idempotent."""
        if self._running:
            return
        self._running = True
        for sub in self._subscribers:
            sub.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop all subscriber threads."""
        self._running = False
        with self._not_empty:
            self._not_empty.notify_all()
        for sub in self._subscribers:
            sub.stop(timeout)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def occupancy(self) -> int:
        """Current number of results in the buffer. For diagnostics."""
        with self._lock:
            return self._count


class _SubscriberThread:
    """Internal: one thread per ResultBus subscriber."""

    def __init__(
        self,
        callback: Callable[[FrameResult], None],
        condition: threading.Condition,
        buffer: list[FrameResult | None],
        buffer_seq: list[int],
        capacity: int,
        lock: threading.Lock,
    ) -> None:
        self._callback = callback
        self._condition = condition
        self._buffer = buffer
        self._buffer_seq = buffer_seq
        self._capacity = capacity
        self._lock = lock
        self._read_pos = 0
        self._last_delivered_seq = 0
        self._thread = threading.Thread(
            target=self._run,
            name=f"ResultBus-{callback.__qualname__}",
            daemon=True,
        )
        self._running = False

    def start(self) -> None:
        self._running = True
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._running = False
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while self._running:
            result: FrameResult | None = None
            with self._condition:
                newest_seq = self._buffer_seq[self._read_pos]
                while self._running and newest_seq <= self._last_delivered_seq:
                    self._condition.wait(timeout=0.1)
                    newest_seq = self._buffer_seq[self._read_pos]
                if not self._running:
                    break
                result = self._buffer[self._read_pos]
                self._last_delivered_seq = self._buffer_seq[self._read_pos]
                self._read_pos = (self._read_pos + 1) % self._capacity

            if result is not None:
                try:
                    self._callback(result)
                except Exception:
                    logger.exception(
                        "[ResultBus] Subscriber %r raised — continuing",
                        self._callback,
                    )
