from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .aggregator import ErrorRecord, FPSCalculator, LatencyHistory, compute_percentiles

if TYPE_CHECKING:
    from cvpipe import (
        ComponentErrorEvent,
        ComponentMetricEvent,
        Event,
        FrameDroppedEvent,
        PipelineStateEvent,
    )


class MetricsCollector:
    """
    Subscribes to EventBus and collects metrics events.
    Thread-safe, minimal overhead on event handlers.
    """

    def __init__(
        self,
        latency_window: int = 300,
        history_duration_minutes: float = 5.0,
        fps_alpha: float = 0.1,
        max_errors_per_component: int = 10,
    ) -> None:
        self._latency_window = latency_window
        self._lock = threading.Lock()

        self._latency_rolling: dict[str, deque[tuple[float, float]]] = {}

        self._latency_history = LatencyHistory(
            duration_minutes=history_duration_minutes
        )

        self._fps_calculator = FPSCalculator(alpha=fps_alpha)

        self._drop_counts: dict[str, int] = {}

        self._errors: dict[str, list[ErrorRecord]] = {}
        self._total_errors = 0
        self._max_errors = max_errors_per_component

        self._pipeline_state: str = "unknown"
        self._state_ts: float = 0
        self._frame_count = 0
        self._start_time: float | None = None

        self._custom_handlers: dict[type, Callable[[Event], dict]] = {}
        self._custom_metrics: dict[str, Any] = {}

    def on_component_metric(self, event: "ComponentMetricEvent") -> None:
        """Called from event dispatch thread. O(1) amortized."""
        with self._lock:
            samples = self._latency_rolling.setdefault(
                event.component_id, deque(maxlen=self._latency_window)
            )
            samples.append((event.ts, event.latency_ms))

            self._latency_history.add(event.component_id, event.ts, event.latency_ms)

            self._fps_calculator.update(event.ts)
            self._frame_count += 1

    def on_frame_dropped(self, event: "FrameDroppedEvent") -> None:
        """Called from event dispatch thread."""
        with self._lock:
            self._drop_counts[event.reason] = self._drop_counts.get(event.reason, 0) + 1

    def on_component_error(self, event: "ComponentErrorEvent") -> None:
        """Called from event dispatch thread."""
        with self._lock:
            self._total_errors += 1
            errors = self._errors.setdefault(event.component_id, [])
            errors.append(
                ErrorRecord(
                    component_id=event.component_id,
                    message=event.message,
                    traceback=event.traceback,
                    frame_idx=event.frame_idx,
                    ts=event.ts,
                )
            )
            if len(errors) > self._max_errors:
                errors.pop(0)

    def on_pipeline_state(self, event: "PipelineStateEvent") -> None:
        """Called from event dispatch thread."""
        with self._lock:
            self._pipeline_state = event.state
            self._state_ts = event.ts
            if event.state == "running" and self._start_time is None:
                self._start_time = event.ts

    def register_custom_event(
        self,
        event_type: type,
        handler: Callable[[Event], dict],
    ) -> None:
        """
        Register a handler for custom telemetry events.

        Handler returns a dict that gets merged into metrics["custom"].

        Example::

            def handle_gpu(event: GPUMemoryEvent) -> dict:
                return {
                    event.component_id: {
                        "memory_mb": event.memory_mb,
                        "utilization_pct": event.utilization_pct,
                    }
                }

            collector.register_custom_event(GPUMemoryEvent, handle_gpu)
        """
        self._custom_handlers[event_type] = handler

    def on_custom_event(self, event: "Event") -> None:
        """Process custom telemetry events."""
        handler = self._custom_handlers.get(type(event))
        if handler:
            with self._lock:
                result = handler(event)
                self._deep_merge(self._custom_metrics.setdefault("custom", {}), result)

    def _deep_merge(self, target: dict, source: dict) -> None:
        """Deep merge source into target."""
        for key, value in source.items():
            if (
                key in target
                and isinstance(target[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_merge(target[key], value)
            else:
                target[key] = value

    def _compute_latency_stats(self) -> dict[str, dict[str, float | int]]:
        """Compute latency percentiles for each component."""
        stats: dict[str, dict[str, float | int]] = {}
        for comp_id, samples in self._latency_rolling.items():
            sorted_latencies = sorted(lat for _, lat in samples)
            stats[comp_id] = compute_percentiles(sorted_latencies)
        return stats

    def _get_latency_history(self) -> dict[str, list[dict]]:
        """Get latency history for time series charts."""
        return self._latency_history.get_all_series()

    def _compute_uptime(self) -> float:
        """Compute uptime in seconds."""
        if self._start_time is None:
            return 0.0
        return time.monotonic() - self._start_time

    def snapshot(self) -> dict[str, Any]:
        """Thread-safe snapshot for HTTP handlers."""
        with self._lock:
            all_errors = [e for errors in self._errors.values() for e in errors]
            recent_errors = sorted(all_errors, key=lambda e: e.ts, reverse=True)[:10]

            return {
                "latency": self._compute_latency_stats(),
                "latency_history": self._get_latency_history(),
                "drops": {
                    "total": sum(self._drop_counts.values()),
                    "by_reason": dict(self._drop_counts),
                },
                "errors": {
                    "total": self._total_errors,
                    "by_component": {k: len(v) for k, v in self._errors.items()},
                    "recent": [e.to_dict() for e in recent_errors],
                },
                "state": {
                    "status": self._pipeline_state,
                    "uptime_seconds": self._compute_uptime(),
                    "frame_count": self._frame_count,
                },
                "fps": {
                    "current": self._fps_calculator.get(),
                    "target": 30,
                },
                "custom": dict(self._custom_metrics.get("custom", {})),
            }
