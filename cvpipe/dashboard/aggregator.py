from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime


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
    """

    def __init__(self, alpha: float = 0.1):
        """
        Parameters
        ----------
        alpha : float
            Smoothing factor (0 < alpha < 1).
            - 0.1: responds to changes over ~10 frames
            - 0.05: responds over ~20 frames
        """
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        self._alpha = alpha
        self._lock = threading.Lock()
        self._ema: float | None = None
        self._last_ts: float | None = None

    def update(self, ts: float) -> None:
        """Call on each frame with its timestamp."""
        with self._lock:
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
