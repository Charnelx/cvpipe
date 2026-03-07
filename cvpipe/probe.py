# cvpipe/probe.py
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .frame import Frame

logger = logging.getLogger(__name__)


class Probe(ABC):
    """
    Non-breaking observation hook attached to the pipeline.

    A Probe is called by the Scheduler after a specific component
    (or after every component) processes a frame. It receives a
    read-only snapshot of the frame's current state.

    Probes run in the streaming thread. They must complete in < 1ms.
    For heavier work (logging, UI updates, metric recording), enqueue
    the data and process it in a separate thread.

    Usage::

        class MyProbe(Probe):
            def observe(self, frame: Frame, after_component: str) -> None:
                count = len(frame.meta.get("detections", []))
                logger.debug("Frame %d: %d detections after %s",
                             frame.idx, count, after_component)

        pipeline.add_probe(MyProbe(), after="tracker")

    Implementation note:
        The Scheduler calls probe.observe() with the Frame. The Probe
        receives the live Frame (not a copy) — it MUST NOT write to
        frame.slots or frame.meta. This is a convention contract, not
        enforced by the framework (enforcing it would require copying
        every frame for every probe, which is unacceptable overhead).
    """

    @abstractmethod
    def observe(self, frame: Frame, after_component: str) -> None:
        """
        Observe the frame after ``after_component`` has processed it.

        Parameters
        ----------
        frame : Frame
            The live frame. READ ONLY — do not modify slots or meta.
        after_component : str
            The component_id of the component that just ran.
        """


@dataclass
class ComponentTrace:
    """Timing and output summary for one component on one frame."""

    component_id: str
    latency_ms: float
    output_slots: list[str]
    output_meta: list[str]
    notes: str = ""


@dataclass
class FrameDiagnostics:
    """Complete per-frame diagnostic trace."""

    frame_idx: int
    ts: float
    components: list[ComponentTrace] = field(default_factory=list)
    total_ms: float = 0.0

    def summary(self) -> str:
        """One-line human-readable summary."""
        parts = [f"{c.component_id}:{c.latency_ms:.1f}ms" for c in self.components]
        return f"Frame {self.frame_idx} | {' → '.join(parts)} | total:{self.total_ms:.1f}ms"


class DiagnosticsProbe(Probe):
    """
    Collects per-component timing and writes a FrameDiagnostics record
    to frame.meta["diagnostics"] after all components run.

    NOTE: This probe writes to frame.meta, which breaks the general
    Probe contract that probes must not modify the frame. This is
    the only exception to that rule.

    Attach with after=None to observe after every component.

    The DiagnosticsProbe tracks which slots/meta keys exist before and
    after each component to determine what each component wrote.

    Usage::

        probe = DiagnosticsProbe()
        scheduler.add_probe(probe, after=None)  # observe after every component
    """

    def __init__(self) -> None:
        self._frame_start: dict[int, float] = {}
        self._frame_traces: dict[int, list[ComponentTrace]] = {}
        self._slots_before: dict[int, set[str]] = {}
        self._meta_before: dict[int, set[str]] = {}

    def observe(self, frame: Frame, after_component: str) -> None:
        idx = frame.idx

        if idx not in self._frame_start:
            self._frame_start[idx] = time.perf_counter()
            self._frame_traces[idx] = []
            self._slots_before[idx] = set()
            self._meta_before[idx] = set()

        current_slots = set(frame.slots.keys())
        current_meta = set(frame.meta.keys())

        new_slots = current_slots - self._slots_before.get(idx, set())
        new_meta = current_meta - self._meta_before.get(idx, set())

        elapsed = (time.perf_counter() - self._frame_start[idx]) * 1000

        trace = ComponentTrace(
            component_id=after_component,
            latency_ms=elapsed,
            output_slots=sorted(new_slots),
            output_meta=sorted(new_meta - {"diagnostics"}),
        )
        self._frame_traces[idx].append(trace)

        self._slots_before[idx] = current_slots
        self._meta_before[idx] = current_meta

        total_ms = (time.perf_counter() - self._frame_start[idx]) * 1000
        frame.meta["diagnostics"] = FrameDiagnostics(
            frame_idx=idx,
            ts=frame.ts,
            components=list(self._frame_traces[idx]),
            total_ms=total_ms,
        )

        for old_idx in list(self._frame_start.keys()):
            if old_idx < idx - 10:
                del self._frame_start[old_idx]
                self._frame_traces.pop(old_idx, None)
                self._slots_before.pop(old_idx, None)
                self._meta_before.pop(old_idx, None)
