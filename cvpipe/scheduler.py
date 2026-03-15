# cvpipe/scheduler.py
from __future__ import annotations

import logging
import threading
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

from .frame import Frame
from .bus import ResultBus, FrameResult
from .event import (
    EventBus,
    ComponentMetricEvent,
    ComponentErrorEvent,
    FrameDroppedEvent,
)
from .component import Component
from .probe import Probe

logger = logging.getLogger(__name__)


@dataclass
class ExclusiveBranch:
    """
    An exclusive branch attached to an ExecutionSegment.
    Evaluated once per frame, before the segment's components run.
    """

    branch_id: str
    trigger_src: str
    components: list[Component]


@dataclass
class ExecutionSegment:
    """
    A group of main-path components, optionally replaceable by an exclusive branch.

    If exclusive_branch is None: components always run (standard segment).
    If exclusive_branch is set: trigger is evaluated each frame.
        truthy  → exclusive_branch.components run; self.components are skipped
        falsy   → self.components run; exclusive_branch.components are skipped

    Nested exclusive branches: when exclusive_branch fires, nested_segments (if any)
    are processed first, then exclusive_branch.components run. This allows hierarchical
    exclusive branching (e.g., outer branch for mode selection, inner branches for
    mode-specific processing).
    """

    components: list[Component]
    exclusive_branch: ExclusiveBranch | None = None
    nested_segments: list[ExecutionSegment] | None = None


class FrameSource(ABC):
    """
    Abstract interface for pipeline frame sources.

    A FrameSource provides frames to the Scheduler. It is not a Component
    — it has no INPUTS/OUTPUTS/SUBSCRIBES declarations. It runs in the
    streaming thread, called by the Scheduler.

    Subclasses implement: camera capture, file reading, test frame generation.

    The framework does NOT wrap FrameSource.next() in error recovery —
    a FrameSource that consistently returns None signals end-of-stream
    and the Scheduler will emit FrameDroppedEvent(reason="source_stall").
    FrameSources should handle their own reconnection logic internally.
    """

    def setup(self) -> None:
        """Called once before the frame loop starts. Open devices here."""

    def teardown(self) -> None:
        """Called once after the frame loop stops. Release devices here."""

    def wait_ready(self, timeout: float = 10.0) -> bool:
        """
        Block until the source is ready to provide frames.

        Called by Pipeline.start() after setup() and before the Scheduler
        starts. Default implementation returns True immediately (assumes
        synchronous setup).

        Sources with asynchronous initialization (e.g., grab threads,
        RTSP handshakes) should override this to block until the first
        frame is available or timeout elapses.

        Parameters
        ----------
        timeout : float
            Maximum seconds to wait. Implementations should honor this
            to avoid indefinite blocking during startup.

        Returns
        -------
        bool
            True if the source is ready to provide frames.
            False if timeout elapsed before ready.

        Notes
        -----
        - Called from the main/API thread, not the streaming thread
        - Should not be called after teardown()
        - May be called multiple times (idempotent if already ready)
        - Implementations should log appropriately on timeout
        """
        return True

    @abstractmethod
    def next(self) -> tuple[Any, float] | None:
        """
        Return the next frame payload and its capture timestamp.

        Returns
        -------
        tuple[payload, timestamp] | None
            payload: raw frame data in whatever format the source provides
                     (e.g. np.ndarray BGR image). The first Component in
                     the pipeline is responsible for placing this into
                     frame.slots.
            timestamp: float, time.monotonic() at capture
            None: no frame available (stall, end of stream, reconnecting)

        Must be non-blocking. If no frame is ready, return None immediately.
        The Scheduler will retry on the next tick.
        """


class Scheduler:
    """
    Runs the pipeline frame loop.

    Lifecycle:
        scheduler = Scheduler(
            source=my_source,
            components=[comp_a, comp_b, comp_c],  # topological order
            result_bus=result_bus,
            event_bus=event_bus,
        )
        scheduler.start()
        # ... pipeline runs ...
        scheduler.stop()

    Threading:
        - start() creates and starts the streaming thread, then returns
        - stop() signals the streaming thread to stop and waits for it

    Frame loop (runs in streaming thread):
        1. source.next() → payload or None
        2. If None: emit FrameDroppedEvent("source_stall"), sleep 10ms, retry
        3. If previous frame still processing: emit FrameDroppedEvent("backpressure"), skip
        4. Construct Frame(idx, ts)
        5. Place payload into frame using _inject_source_frame()
        6. For each component in order:
           a. Record start time
           b. component.process(frame)
           c. Record end time
           d. Emit ComponentMetricEvent(component_id, latency_ms, frame_idx)
           e. If exception: emit ComponentErrorEvent, emit FrameDroppedEvent,
              break inner loop, discard frame, continue outer loop
        7. _extract_result(frame) → FrameResult
        8. result_bus.push(result)
        9. Increment frame counter

    Error containment:
        Step 6e means a single bad component never crashes the loop.
        The bad frame is dropped; the next frame starts fresh.
        Consecutive errors from the same component are logged with
        increasing severity (WARNING for first 3, ERROR thereafter).
    """

    def __init__(
        self,
        source: FrameSource,
        segments: list[ExecutionSegment] | None = None,
        result_bus: ResultBus | None = None,
        event_bus: EventBus | None = None,
        source_frame_slot: str = "frame_raw",
        components: list[Component] | None = None,
        validation_mode: Literal["off", "warn", "strict"] = "warn",
    ) -> None:
        self._source = source
        if segments is not None:
            self._segments = segments
        elif components is not None:
            self._segments = [ExecutionSegment(components=[c]) for c in components]
        else:
            raise ValueError("Either 'segments' or 'components' must be provided")
        self._result_bus = result_bus
        self._event_bus = event_bus
        self._source_slot = source_frame_slot
        self._validation_mode = validation_mode

        self._thread: threading.Thread | None = None
        self._stop_flag: threading.Event = threading.Event()
        self._frame_idx = 0
        self._source_frame_idx = 0

        self._pause_requested: threading.Event = threading.Event()
        self._pause_acked: threading.Event = threading.Event()

        self._error_counts: dict[str, int] = {}
        self._probes: dict[str | None, list[Probe]] = {}

    def start(self) -> None:
        """Start the streaming thread. Non-blocking."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Scheduler is already running")
        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="Scheduler-streaming",
            daemon=True,
        )
        self._thread.start()
        logger.info("[Scheduler] Streaming thread started")

    def stop(self, timeout: float = 5.0) -> None:
        """
        Signal the streaming thread to stop and wait for it.

        Parameters
        ----------
        timeout : float
            Seconds to wait. If the thread does not stop within timeout,
            a warning is logged but stop() returns anyway.
        """
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(
                    "[Scheduler] Streaming thread did not stop within %.1fs", timeout
                )
        logger.info("[Scheduler] Stopped")

    def pause(self, timeout: float = 2.0) -> None:
        """
        Signal the streaming thread to pause between frames.

        Blocks until the streaming thread acknowledges the pause or timeout elapses.
        The streaming thread completes its current frame before pausing — no frame
        is interrupted mid-processing.

        Must be called from outside the streaming thread (e.g. from the main/API thread).
        Must be paired with a subsequent resume() call. Do not call pause() twice
        without an intervening resume().

        Parameters
        ----------
        timeout : float
            Maximum seconds to wait for the streaming thread to acknowledge.
            If the streaming thread does not pause within timeout, a warning is
            logged but pause() returns anyway. The caller should handle this case.
        """
        self._pause_requested.set()
        if not self._pause_acked.wait(timeout=timeout):
            logger.warning(
                "[Scheduler] pause() did not complete within %.1fs — proceeding",
                timeout,
            )

    def resume(self) -> None:
        """
        Resume the streaming thread after a pause() call.

        Safe to call even if the thread did not fully acknowledge pause()
        (it will resume immediately when it next checks the pause flag).
        """
        self._pause_requested.clear()
        self._pause_acked.clear()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def frame_count(self) -> int:
        """Total frames processed (not dropped). For diagnostics."""
        return self._frame_idx

    def add_probe(self, probe: Probe, after: str | None = None) -> None:
        """
        Attach a probe to run after a specific component.

        Parameters
        ----------
        probe : Probe
            The probe to attach.
        after : str | None
            Component ID to probe after. None = call after every component.
            Must be called before start().
        """
        self._probes.setdefault(after, []).append(probe)

    def _eval_trigger(self, src: str, frame: Frame) -> bool:
        """
        Evaluate a branch trigger expression against frame.meta.

        The expression is evaluated in a restricted namespace containing only the
        contents of frame.meta as local variables. No builtins are exposed.
        Any exception during evaluation is caught, logged at DEBUG, and treated
        as False (main path runs).

        Parameters
        ----------
        src : str
            Python expression. Must be a single expression, not a statement.
            Example: "routing_decision == 'scan'"
        frame : Frame
            Current frame. frame.meta is used as the local variable namespace.

        Returns
        -------
        bool
            True if trigger fires; False otherwise (including on eval error).
        """
        try:
            return bool(eval(src, {"__builtins__": {}}, dict(frame.meta)))
        except Exception as e:
            logger.error(
                f"[Scheduler] Branch trigger '{src}' eval failed on frame {frame.idx}: {e!r}"
            )
            return False

    def _run(self) -> None:
        """Main frame loop. Runs entirely in the streaming thread."""
        logger.debug("[Scheduler] Frame loop starting")
        while not self._stop_flag.is_set():
            if self._pause_requested.is_set():
                self._pause_acked.set()
                while self._pause_requested.is_set() and not self._stop_flag.is_set():
                    time.sleep(0.005)
                self._pause_acked.clear()

            result = self._source.next()

            if result is None:
                self._event_bus.publish(
                    FrameDroppedEvent(
                        reason="source_stall", frame_idx=self._source_frame_idx
                    )
                )
                time.sleep(0.01)
                continue

            payload, ts = result
            source_idx = self._source_frame_idx
            self._source_frame_idx += 1

            frame = Frame(idx=self._frame_idx, ts=ts)
            # TODO: check this condition
            if isinstance(payload, dict):
                frame.meta.update(payload)
            else:
                frame.slots[self._source_slot] = payload

            ok = self._run_components(frame, source_idx)
            if not ok:
                continue

            frame_result = self._extract_result(frame)
            self._result_bus.push(frame_result)
            self._frame_idx += 1

        logger.debug("[Scheduler] Frame loop exited")

    def _run_components(self, frame: Frame, source_idx: int) -> bool:
        """
        Run all components sequentially on frame.

        Returns True if all components ran successfully.
        Returns False if any component raised — the frame is discarded.
        """
        for segment in self._segments:
            branch = segment.exclusive_branch
            if branch is not None and self._eval_trigger(branch.trigger_src, frame):
                if not self._run_segment_components(
                    segment.nested_segments or [], frame, source_idx
                ):
                    return False
                components_to_run = branch.components
            else:
                components_to_run = segment.components

            for component in components_to_run:
                if not self._run_single_component(component, frame, source_idx):
                    return False

        return True

    def _run_segment_components(
        self,
        segments: list[ExecutionSegment],
        frame: Frame,
        source_idx: int,
    ) -> bool:
        """Run components from a list of segments (used for nested segments)."""
        for segment in segments:
            branch = segment.exclusive_branch
            if branch is not None and self._eval_trigger(branch.trigger_src, frame):
                if not self._run_segment_components(
                    segment.nested_segments or [], frame, source_idx
                ):
                    return False
                components_to_run = branch.components
            else:
                components_to_run = segment.components

            for component in components_to_run:
                if not self._run_single_component(component, frame, source_idx):
                    return False

        return True

    def _run_single_component(
        self, component: Component, frame: Frame, source_idx: int
    ) -> bool:
        """Run a single component and handle errors/metrics."""
        cid = component.component_id
        t_start = time.perf_counter()

        frame._set_validation(component, self._validation_mode)
        try:
            component.process(frame)
        except Exception as exc:
            t_elapsed = (time.perf_counter() - t_start) * 1000
            tb_str = traceback.format_exc()

            count = self._error_counts.get(cid, 0) + 1
            self._error_counts[cid] = count
            log_fn = logger.error if count > 3 else logger.warning
            log_fn(
                "[Scheduler] Component '%s' raised %s on frame %d (error #%d): %s",
                cid,
                type(exc).__name__,
                source_idx,
                count,
                exc,
            )

            self._event_bus.publish(
                ComponentErrorEvent(
                    component_id=cid,
                    message=str(exc),
                    traceback=tb_str,
                    frame_idx=source_idx,
                )
            )
            self._event_bus.publish(
                FrameDroppedEvent(reason="component_error", frame_idx=source_idx)
            )
            return False
        finally:
            frame._clear_validation()

        t_elapsed = (time.perf_counter() - t_start) * 1000
        self._error_counts[cid] = 0
        self._event_bus.publish(
            ComponentMetricEvent(
                component_id=cid, latency_ms=t_elapsed, frame_idx=frame.idx
            )
        )

        for probe in self._probes.get(cid, []):
            try:
                probe.observe(frame, cid)
            except Exception:
                logger.debug("[Scheduler] Probe %r raised — ignored", probe)
        for probe in self._probes.get(None, []):
            try:
                probe.observe(frame, cid)
            except Exception:
                logger.debug("[Scheduler] Probe %r raised — ignored", probe)

        return True

    def _extract_result(self, frame: Frame) -> FrameResult:
        """
        Extract a FrameResult from the completed frame.

        Looks for these well-known keys in frame.meta (all optional):
        - "jpeg_bytes"  → bytes
        - "detections"  → list[dict]
        - "result_meta" → dict  (merged into FrameResult.meta)

        If none are present, returns a minimal FrameResult with empty
        jpeg_bytes and detections. The application can override this
        by having a terminal component write these keys.
        """
        return FrameResult(
            frame_idx=frame.idx,
            ts=frame.ts,
            jpeg_bytes=frame.meta.get("jpeg_bytes", b""),
            detections=frame.meta.get("detections", []),
            meta=frame.meta.get("result_meta", {}),
        )
