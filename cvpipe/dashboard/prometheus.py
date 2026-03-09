from __future__ import annotations

from typing import Any


def render_prometheus(metrics: dict[str, Any]) -> str:
    """Convert metrics dict to Prometheus format."""
    lines: list[str] = []

    # Latency
    lines.append("# HELP cvpipe_component_latency_ms Component processing latency")
    lines.append("# TYPE cvpipe_component_latency_ms summary")
    for comp, stats in metrics.get("latency", {}).items():
        for quantile, key in [
            ("0.5", "p50_ms"),
            ("0.95", "p95_ms"),
            ("0.99", "p99_ms"),
        ]:
            if key in stats:
                lines.append(
                    f'cvpipe_component_latency_ms{{component="{comp}",quantile="{quantile}"}} {stats[key]:.3f}'
                )
        if "samples" in stats:
            lines.append(
                f'cvpipe_component_latency_ms_count{{component="{comp}"}} {stats["samples"]}'
            )

    # Drops
    lines.append("# HELP cvpipe_frame_drops_total Total frames dropped")
    lines.append("# TYPE cvpipe_frame_drops_total counter")
    drops_data = metrics.get("drops", {})
    if drops_data:
        for reason, count in drops_data.get("by_reason", {}).items():
            lines.append(f'cvpipe_frame_drops_total{{reason="{reason}"}} {count}')

    # Errors
    lines.append("# HELP cvpipe_errors_total Total component errors")
    lines.append("# TYPE cvpipe_errors_total counter")
    lines.append(f"cvpipe_errors_total {metrics.get('errors', {}).get('total', 0)}")

    # State
    lines.append("# HELP cvpipe_pipeline_state Pipeline state (1=running, 0=stopped)")
    lines.append("# TYPE cvpipe_pipeline_state gauge")
    state_value = 1 if metrics.get("state", {}).get("status") == "running" else 0
    lines.append(f"cvpipe_pipeline_state {state_value}")

    # FPS
    lines.append("# HELP cvpipe_fps Current frames per second")
    lines.append("# TYPE cvpipe_fps gauge")
    lines.append(f"cvpipe_fps {metrics.get('fps', {}).get('current', 0):.1f}")

    # Frame count
    lines.append("# HELP cvpipe_frames_total Total frames processed")
    lines.append("# TYPE cvpipe_frames_total counter")
    lines.append(
        f"cvpipe_frames_total {metrics.get('state', {}).get('frame_count', 0)}"
    )

    # Uptime
    lines.append("# HELP cvpipe_uptime_seconds Pipeline uptime in seconds")
    lines.append("# TYPE cvpipe_uptime_seconds gauge")
    lines.append(
        f"cvpipe_uptime_seconds {metrics.get('state', {}).get('uptime_seconds', 0):.1f}"
    )

    # Custom metrics
    custom = metrics.get("custom", {})
    if custom:
        lines.append("# HELP cvpipe_custom_metric Custom application metrics")
        lines.append("# TYPE cvpipe_custom_metric gauge")
        for category, components in custom.items():
            if isinstance(components, dict):
                for comp, values in components.items():
                    if isinstance(values, dict):
                        for metric_name, value in values.items():
                            lines.append(
                                f'cvpipe_custom_metric{{category="{category}",component="{comp}",metric="{metric_name}"}} {value}'
                            )

    return "\n".join(lines)
