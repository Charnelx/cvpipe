from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime


logger = logging.getLogger(__name__)


@dataclass
class ErrorRecord:
    """Stored error for dashboard display."""

    component_id: str
    message: str
    traceback: str
    frame_idx: int
    ts: float

    def to_dict(self) -> dict:
        return {
            "component_id": self.component_id,
            "message": self.message,
            "traceback": self.traceback,
            "frame_idx": self.frame_idx,
            "ts": self.ts,
            "ts_iso": datetime.fromtimestamp(self.ts).isoformat(),
        }


def compute_percentiles(samples: list[float]) -> dict[str, float | int]:
    """Compute p50, p95, p99 from sorted samples."""
    if not samples:
        return {}
    n = len(samples)
    return {
        "p50_ms": samples[int(n * 0.50)],
        "p95_ms": samples[int(n * 0.95)],
        "p99_ms": samples[int(n * 0.99)],
        "min_ms": samples[0],
        "max_ms": samples[-1],
        "current_ms": samples[-1],
        "samples": n,
    }


class FPSCalculator:
    """
    Thread-safe FPS calculator using exponential moving average.

    EMA formula: fps = alpha * instant_fps + (1 - alpha) * previous_fps

    Lower alpha = smoother, slower to respond.
    Higher alpha = more responsive, noisier.

    Uses frame_idx to track consecutive frames for accurate FPS calculation.
    Resets FPS on frame skips or stalls (>10s without frames).
    """

    _STALENESS_WARNING_THRESHOLD = 5.0
    _STALENESS_RESET_THRESHOLD = 10.0

    def __init__(self, alpha: float = 0.1, target_component_id: str | None = None):
        """
        Parameters
        ----------
        alpha : float
            Smoothing factor (0 < alpha < 1).
            - 0.1: responds to changes over ~10 frames
            - 0.05: responds over ~20 frames
        target_component_id : str | None
            If set, only update FPS for events from this component.
            If None, track all events (backward compatible).
        """
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        self._alpha = alpha
        self._target_component_id = target_component_id
        self._lock = threading.Lock()
        self._ema: float | None = None
        self._last_ts: float | None = None
        self._last_frame_idx: int | None = None
        self._last_update_ts: float | None = None
        self._is_stale_warning_logged = False

    def update(
        self, ts: float, frame_idx: int, component_id: str | None = None
    ) -> None:
        """
        Update FPS based on frame_idx and timestamp.

        Parameters
        ----------
        ts : float
            Monotonic timestamp of the event.
        frame_idx : int
            Frame index from ComponentMetricEvent.
        component_id : str | None
            Component ID for filtering (if target_component_id is set).
        """
        if (
            self._target_component_id is not None
            and component_id != self._target_component_id
        ):
            return

        with self._lock:
            self._handle_staleness(ts)
            self._update_fps(ts, frame_idx)

    def _handle_staleness(self, ts: float) -> None:
        """Check for staleness and reset FPS if needed."""
        if self._last_update_ts is None:
            return

        time_since_update = ts - self._last_update_ts

        if time_since_update > self._STALENESS_RESET_THRESHOLD:
            logger.warning(
                "[FPSCalculator] FPS reset due to stall (%.1fs without frames)",
                time_since_update,
            )
            self._ema = None
            self._last_ts = None
            self._last_frame_idx = None
            self._last_update_ts = ts
            self._is_stale_warning_logged = False
        elif (
            time_since_update > self._STALENESS_WARNING_THRESHOLD
            and not self._is_stale_warning_logged
        ):
            logger.warning(
                "[FPSCalculator] Frame stall detected, FPS may be inaccurate (%.1fs without new frame)",
                time_since_update,
            )
            self._is_stale_warning_logged = True

    def _update_fps(self, ts: float, frame_idx: int) -> None:
        """Update FPS based on consecutive frame_idx."""
        if self._last_frame_idx is None:
            self._last_frame_idx = frame_idx
            self._last_update_ts = ts
            self._is_stale_warning_logged = False
            return

        if frame_idx == self._last_frame_idx + 1:
            if self._last_ts is not None:
                delta = ts - self._last_ts
                if delta > 0:
                    instant_fps = 1.0 / delta
                    if self._ema is None:
                        self._ema = instant_fps
                    else:
                        self._ema = (
                            self._alpha * instant_fps + (1 - self._alpha) * self._ema
                        )
            self._last_ts = ts
            self._last_frame_idx = frame_idx
            self._last_update_ts = ts
            if self._is_stale_warning_logged:
                self._is_stale_warning_logged = False
        else:
            self._ema = None
            self._last_ts = None
            self._last_frame_idx = frame_idx
            self._last_update_ts = ts

    def get(self) -> float:
        """Return current EMA FPS."""
        with self._lock:
            return self._ema or 0.0


class LatencyHistory:
    """
    Ring buffer for latency time series (last N minutes).

    Aggregates samples into buckets for efficient storage.
    """

    def __init__(self, duration_minutes: float = 5.0, resolution_seconds: float = 1.0):
        """
        Parameters
        ----------
        duration_minutes : float
            How much history to keep (default: 5 minutes)
        resolution_seconds : float
            Bucket size for aggregation (default: 1 second)
        """
        self._max_buckets = int(duration_minutes * 60 / resolution_seconds)
        self._resolution = resolution_seconds
        self._lock = threading.Lock()
        self._history: dict[str, deque[tuple[float, float, int]]] = {}

    def add(self, component_id: str, ts: float, latency_ms: float) -> None:
        """Add a latency sample."""
        bucket_ts = int(ts / self._resolution) * self._resolution

        with self._lock:
            hist = self._history.setdefault(
                component_id, deque(maxlen=self._max_buckets)
            )

            if hist and hist[-1][0] == bucket_ts:
                old_avg, old_count = hist[-1][1], hist[-1][2]
                new_count = old_count + 1
                new_avg = (old_avg * old_count + latency_ms) / new_count
                hist[-1] = (bucket_ts, new_avg, new_count)
            else:
                hist.append((bucket_ts, latency_ms, 1))

    def get_series(self, component_id: str) -> list[dict]:
        """Get time series for charting."""
        with self._lock:
            hist = self._history.get(component_id, [])
            return [
                {"ts": ts, "latency_ms": avg, "samples": count}
                for ts, avg, count in hist
            ]

    def get_all_series(self) -> dict[str, list[dict]]:
        """Get all component time series."""
        with self._lock:
            return {
                comp: [
                    {"ts": ts, "latency_ms": avg, "samples": count}
                    for ts, avg, count in hist
                ]
                for comp, hist in self._history.items()
            }
