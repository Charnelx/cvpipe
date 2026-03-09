from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .aggregator import ErrorRecord, FPSCalculator, LatencyHistory
from .collector import MetricsCollector

if TYPE_CHECKING:
    from cvpipe import Pipeline
    from .server import DashboardServer

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for the metrics dashboard."""

    enabled: bool = True
    port: int = 8881
    host: str = "0.0.0.0"
    prometheus: bool = True
    websocket: bool = True
    update_interval_ms: int = 1000
    latency_window: int = 300
    history_duration_minutes: float = 5.0
    fps_alpha: float = 0.1
    max_errors_per_component: int = 10


def enable_dashboard(
    pipeline: "Pipeline",
    config: DashboardConfig | None = None,
) -> "DashboardServer | None":
    """
    Enable the metrics dashboard for a pipeline.

    Must be called BEFORE pipeline.start().

    Parameters
    ----------
    pipeline : Pipeline
        The pipeline to monitor.
    config : DashboardConfig | None
        Dashboard configuration. Uses defaults if not provided.

    Returns
    -------
    DashboardServer | None
        The dashboard server instance, or None if disabled or dependencies missing.

    Example
    -------
    ::

        from cvpipe import build
        from cvpipe.dashboard import enable_dashboard, DashboardConfig

        pipeline = build(config_path, components_dir)
        pipeline.validate()

        # With default config
        enable_dashboard(pipeline)

        # Or with custom config
        enable_dashboard(pipeline, DashboardConfig(
            port=8881,
            prometheus=True,
            history_duration_minutes=10.0,
        ))

        pipeline.start()
    """
    if config is None:
        config = DashboardConfig()

    if not config.enabled:
        return None

    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        logger.warning(
            "[Dashboard] Dependencies not installed. "
            "Install with: pip install cvpipe[dashboard]"
        )
        return None

    from .server import DashboardServer

    collector = MetricsCollector(
        latency_window=config.latency_window,
        history_duration_minutes=config.history_duration_minutes,
        fps_alpha=config.fps_alpha,
        max_errors_per_component=config.max_errors_per_component,
    )

    from cvpipe import (
        ComponentErrorEvent,
        ComponentMetricEvent,
        FrameDroppedEvent,
        PipelineStateEvent,
    )

    pipeline.event_bus.subscribe(ComponentMetricEvent, collector.on_component_metric)
    pipeline.event_bus.subscribe(FrameDroppedEvent, collector.on_frame_dropped)
    pipeline.event_bus.subscribe(ComponentErrorEvent, collector.on_component_error)
    pipeline.event_bus.subscribe(PipelineStateEvent, collector.on_pipeline_state)

    server = DashboardServer(
        collector=collector,
        pipeline=pipeline,
        port=config.port,
        host=config.host,
        prometheus=config.prometheus,
        websocket=config.websocket,
        update_interval_ms=config.update_interval_ms,
    )

    server.start()
    pipeline._dashboard_collector = collector  # type: ignore[attr-defined]

    return server


__all__ = [
    "DashboardConfig",
    "DashboardServer",
    "MetricsCollector",
    "FPSCalculator",
    "LatencyHistory",
    "ErrorRecord",
    "enable_dashboard",
]
