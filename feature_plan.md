# Pipeline Health Dashboard - Feature Plan

## Overview

**Goal:** Provide a zero-configuration HTTP dashboard for real-time pipeline health monitoring.

**Current State:** Users manually build `/metrics` endpoints (as shown in the example_webcam_api.md). The framework emits `ComponentMetricEvent` for latency data but requires custom code to aggregate and expose it.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Pipeline Process                                                │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌────────────────────────┐    │
│  │ Scheduler│───▶│ EventBus │───▶│ MetricsCollector       │    │
│  │          │    │          │    │ (event handler)        │    │
│  └──────────┘    └──────────┘    └────────────────────────┘    │
│                                         │                       │
│                                         ▼                       │
│                           ┌────────────────────────┐           │
│                           │ AggregatedMetrics      │           │
│                           │ (thread-safe store)    │           │
│                           └────────────────────────┘           │
│                                         │                       │
│                                         ▼                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ DashboardServer (separate thread)                        │  │
│  │ Port: 8881 (configurable)                               │  │
│  │                                                          │  │
│  │   ├── GET /api/v1/metrics (JSON)                         │  │
│  │   ├── GET /api/v1/metrics/latency                        │  │
│  │   ├── GET /api/v1/metrics/drops                          │  │
│  │   ├── GET /api/v1/metrics/errors                         │  │
│  │   ├── GET /api/v1/metrics/state                          │  │
│  │   ├── GET /api/v1/metrics/fps                            │  │
│  │   ├── GET /metrics (Prometheus)                          │  │
│  │   ├── WS /ws/metrics (WebSocket)                         │  │
│  │   ├── POST /api/v1/metrics/export (save to file)         │  │
│  │   └── GET / (HTML dashboard)                             │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Metrics Exposed

### Latency Metrics (from ComponentMetricEvent)

```json
{
  "latency": {
    "preprocessor": {
      "p50_ms": 1.2,
      "p95_ms": 1.8,
      "p99_ms": 2.1,
      "min_ms": 0.9,
      "max_ms": 2.5,
      "current_ms": 1.3,
      "samples": 300
    },
    "detector": {
      "p50_ms": 18.4,
      "p95_ms": 22.1,
      "p99_ms": 24.3,
      "min_ms": 15.2,
      "max_ms": 28.1,
      "current_ms": 19.2,
      "samples": 300
    }
  }
}
```

### Frame Drop Metrics (from FrameDroppedEvent)

```json
{
  "drops": {
    "total": 42,
    "by_reason": {
      "backpressure": 30,
      "component_error": 10,
      "source_stall": 2
    },
    "rate_per_minute": 1.5
  }
}
```

### Pipeline State (from PipelineStateEvent)

```json
{
  "state": {
    "status": "running",
    "started_at": "2026-03-08T20:00:00Z",
    "uptime_seconds": 3600,
    "frame_count": 108000
  }
}
```

### Component Errors (from ComponentErrorEvent)

```json
{
  "errors": {
    "total": 3,
    "by_component": {
      "detector": 2,
      "tracker": 1
    },
    "recent": [
      {
        "component_id": "detector",
        "message": "CUDA out of memory",
        "traceback": "...",
        "frame_idx": 1234,
        "ts": 1709925900.123,
        "ts_iso": "2026-03-08T20:05:00Z"
      }
    ]
  }
}
```

### Frame Rate (EMA)

```json
{
  "fps": {
    "current": 28.5,
    "target": 30
  }
}
```

### Queue Depths

```json
{
  "queues": {
    "result_bus": {
      "capacity": 4,
      "occupancy": 2
    },
    "event_bus": {
      "capacity": 256,
      "occupancy": 5
    }
  }
}
```

### Custom Metrics (from user events)

```json
{
  "custom": {
    "gpu": {
      "detector": {
        "memory_mb": 2450.5,
        "utilization_pct": 85.2
      }
    },
    "model": {
      "detector": {
        "confidence": 0.45,
        "detections_count": 12
      }
    }
  }
}
```

---

## File Structure

```
cvpipe/
├── __init__.py              # Add dashboard exports
├── dashboard/
│   ├── __init__.py          # enable_dashboard(), DashboardConfig
│   ├── collector.py         # MetricsCollector
│   ├── aggregator.py        # FPSCalculator, LatencyHistory, percentiles
│   ├── server.py            # DashboardServer (FastAPI)
│   ├── prometheus.py        # render_prometheus()
│   ├── custom.py            # CustomMetricHandler
│   ├── templates/
│   │   └── index.html       # Full-featured dashboard
│   └── static/
│       ├── dashboard.js     # Chart.js integration, WebSocket
│       └── dashboard.css    # Dark minimalistic theme
tests/cvpipe/
└── test_dashboard.py        # Unit tests
```

---

## Component Details

### 1. collector.py - MetricsCollector

```python
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
    ):
        self._latency_window = latency_window
        self._lock = threading.Lock()

        # Latency: component_id -> deque of (ts, latency_ms)
        self._latency_rolling: dict[str, deque] = {}

        # History for time series chart
        self._latency_history = LatencyHistory(duration_minutes=history_duration_minutes)

        # FPS using EMA
        self._fps_calculator = FPSCalculator(alpha=fps_alpha)

        # Drops: reason -> count
        self._drop_counts: dict[str, int] = {}

        # Errors
        self._errors: dict[str, list[ErrorRecord]] = {}
        self._total_errors = 0
        self._max_errors = max_errors_per_component

        # State
        self._pipeline_state: str = "unknown"
        self._state_ts: float = 0
        self._frame_count = 0
        self._start_time: float | None = None

        # Custom metrics
        self._custom_handlers: dict[type, Callable[[Event], dict]] = {}
        self._custom_metrics: dict = {}

    def on_component_metric(self, event: ComponentMetricEvent) -> None:
        """Called from event dispatch thread. O(1) amortized."""
        with self._lock:
            # Rolling window for percentiles
            samples = self._latency_rolling.setdefault(
                event.component_id, deque(maxlen=self._latency_window)
            )
            samples.append((event.ts, event.latency_ms))

            # History for time series
            self._latency_history.add(event.component_id, event.ts, event.latency_ms)

            # FPS
            self._fps_calculator.update(event.ts)
            self._frame_count += 1

    def on_frame_dropped(self, event: FrameDroppedEvent) -> None:
        """Called from event dispatch thread."""
        with self._lock:
            self._drop_counts[event.reason] = self._drop_counts.get(event.reason, 0) + 1

    def on_component_error(self, event: ComponentErrorEvent) -> None:
        """Called from event dispatch thread."""
        with self._lock:
            self._total_errors += 1
            errors = self._errors.setdefault(event.component_id, [])
            errors.append(ErrorRecord(
                component_id=event.component_id,
                message=event.message,
                traceback=event.traceback,
                frame_idx=event.frame_idx,
                ts=event.ts,
            ))
            # Keep only last N errors per component
            if len(errors) > self._max_errors:
                errors.pop(0)

    def on_pipeline_state(self, event: PipelineStateEvent) -> None:
        """Called from event dispatch thread."""
        with self._lock:
            self._pipeline_state = event.state
            self._state_ts = event.ts
            if event.state == "running" and self._start_time is None:
                self._start_time = event.ts

    def register_custom_event(
        self,
        event_type: type[Event],
        handler: Callable[[Event], dict],
    ) -> None:
        """
        Register a handler for custom telemetry events.

        Handler returns a dict that gets merged into metrics["custom"].

        Example:
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

    def on_custom_event(self, event: Event) -> None:
        """Process custom telemetry events."""
        handler = self._custom_handlers.get(type(event))
        if handler:
            with self._lock:
                result = handler(event)
                self._deep_merge(self._custom_metrics.setdefault("custom", {}), result)

    def snapshot(self) -> dict:
        """Thread-safe snapshot for HTTP handlers."""
        with self._lock:
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
                    "recent": [e.to_dict() for errors in self._errors.values() for e in errors][-10:],
                },
                "state": {
                    "status": self._pipeline_state,
                    "uptime_seconds": self._compute_uptime(),
                    "frame_count": self._frame_count,
                },
                "fps": {
                    "current": self._fps_calculator.get(),
                    "target": 30,  # Could be configured
                },
                "custom": dict(self._custom_metrics.get("custom", {})),
            }
```

### 2. aggregator.py - Statistics Functions

```python
import threading
from collections import deque
from dataclasses import dataclass
import time


def compute_percentiles(samples: list[float]) -> dict:
    """Compute p50, p95, p99 from sorted samples."""
    if not samples:
        return {}
    n = len(samples)
    return {
        "p50": samples[int(n * 0.50)],
        "p95": samples[int(n * 0.95)],
        "p99": samples[int(n * 0.99)],
        "min": samples[0],
        "max": samples[-1],
        "count": n,
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
                        self._ema = self._alpha * instant_fps + (1 - self._alpha) * self._ema
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
        # component_id -> deque of (bucket_ts, avg_latency, count)
        self._history: dict[str, deque] = {}

    def add(self, component_id: str, ts: float, latency_ms: float) -> None:
        """Add a latency sample."""
        bucket_ts = int(ts / self._resolution) * self._resolution

        with self._lock:
            hist = self._history.setdefault(
                component_id, deque(maxlen=self._max_buckets)
            )

            # Aggregate into bucket
            if hist and hist[-1][0] == bucket_ts:
                # Update existing bucket
                old_avg, old_count = hist[-1][1], hist[-1][2]
                new_count = old_count + 1
                new_avg = (old_avg * old_count + latency_ms) / new_count
                hist[-1] = (bucket_ts, new_avg, new_count)
            else:
                # New bucket
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


@dataclass
class ErrorRecord:
    """Stored error for dashboard display."""

    component_id: str
    message: str
    traceback: str
    frame_idx: int
    ts: float

    def to_dict(self) -> dict:
        from datetime import datetime
        return {
            "component_id": self.component_id,
            "message": self.message,
            "traceback": self.traceback,
            "frame_idx": self.frame_idx,
            "ts": self.ts,
            "ts_iso": datetime.fromtimestamp(self.ts).isoformat(),
        }
```

### 3. server.py - DashboardServer

```python
import asyncio
import json
import logging
import threading
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, PlainTextResponse

logger = logging.getLogger(__name__)


class DashboardServer:
    """
    HTTP server for metrics dashboard.

    Runs in separate thread, does not block pipeline.
    """

    def __init__(
        self,
        collector,
        pipeline,
        port: int = 8080,
        host: str = "0.0.0.0",
        prometheus: bool = True,
        websocket: bool = True,
        update_interval_ms: int = 1000,
    ):
        self._collector = collector
        self._pipeline = pipeline
        self._port = port
        self._host = host
        self._prometheus = prometheus
        self._websocket = websocket
        self._update_interval = update_interval_ms / 1000.0

        self._thread: threading.Thread | None = None
        self._running = False

        self._app = FastAPI(title="cvpipe Dashboard")
        self._setup_routes()

    def _setup_routes(self) -> None:
        @self._app.get("/", response_class=HTMLResponse)
        async def dashboard():
            return self._render_html()

        @self._app.get("/api/v1/metrics")
        async def get_metrics():
            return self._collector.snapshot()

        @self._app.get("/api/v1/metrics/latency")
        async def get_latency():
            return self._collector.snapshot().get("latency", {})

        @self._app.get("/api/v1/metrics/drops")
        async def get_drops():
            return self._collector.snapshot().get("drops", {})

        @self._app.get("/api/v1/metrics/errors")
        async def get_errors():
            return self._collector.snapshot().get("errors", {})

        @self._app.get("/api/v1/metrics/state")
        async def get_state():
            return self._collector.snapshot().get("state", {})

        @self._app.get("/api/v1/metrics/fps")
        async def get_fps():
            return self._collector.snapshot().get("fps", {})

        if self._prometheus:
            from .prometheus import render_prometheus

            @self._app.get("/metrics", response_class=PlainTextResponse)
            async def prometheus_metrics():
                return render_prometheus(self._collector.snapshot())

        @self._app.post("/api/v1/metrics/export")
        async def export_metrics(path: str = "metrics_export.json"):
            """Export current metrics to file."""
            data = self._collector.snapshot()
            data["export_ts"] = time.time()
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            return {"status": "ok", "path": path}

        if self._websocket:

            @self._app.websocket("/ws/metrics")
            async def websocket_metrics(websocket: WebSocket):
                await websocket.accept()
                try:
                    while self._running:
                        data = self._collector.snapshot()
                        await websocket.send_json(data)
                        await asyncio.sleep(self._update_interval)
                except Exception:
                    pass

    def _render_html(self) -> str:
        """Return the dashboard HTML."""
        # Read from templates directory
        template_path = Path(__file__).parent / "templates" / "index.html"
        if template_path.exists():
            return template_path.read_text()
        return "<html><body><h1>Dashboard template not found</h1></body></html>"

    def start(self) -> None:
        """Start the server in a background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_server,
            name="Dashboard-server",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "[Dashboard] Server started at http://%s:%d",
            self._host,
            self._port,
        )

    def _run_server(self) -> None:
        import uvicorn

        config = uvicorn.Config(
            self._app,
            host=self._host,
            port=self._port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        server.run()

    def stop(self) -> None:
        """Stop the server."""
        self._running = False
        logger.info("[Dashboard] Server stopped")
```

### 4. prometheus.py - Prometheus Format

```python
def render_prometheus(metrics: dict) -> str:
    """Convert metrics dict to Prometheus format."""
    lines = []

    # Latency
    lines.append("# HELP cvpipe_component_latency_ms Component processing latency")
    lines.append("# TYPE cvpipe_component_latency_ms summary")
    for comp, stats in metrics.get("latency", {}).items():
        for quantile, key in [("0.5", "p50"), ("0.95", "p95"), ("0.99", "p99")]:
            if key in stats:
                lines.append(
                    f'cvpipe_component_latency_ms{{component="{comp}",quantile="{quantile}"}} {stats[key]:.3f}'
                )
        if "count" in stats:
            lines.append(f'cvpipe_component_latency_ms_count{{component="{comp}"}} {stats["count"]}')

    # Drops
    lines.append("# HELP cvpipe_frame_drops_total Total frames dropped")
    lines.append("# TYPE cvpipe_frame_drops_total counter")
    for reason, count in metrics.get("drops", {}).get("by_reason", {}).items():
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
    lines.append(f"cvpipe_frames_total {metrics.get('state', {}).get('frame_count', 0)}")

    # Custom metrics
    custom = metrics.get("custom", {})
    if custom:
        lines.append("# HELP cvpipe_custom_metric Custom application metrics")
        lines.append("# TYPE cvpipe_custom_metric gauge")
        for category, components in custom.items():
            for comp, values in components.items():
                for metric_name, value in values.items():
                    lines.append(
                        f'cvpipe_custom_metric{{category="{category}",component="{comp}",metric="{metric_name}"}} {value}'
                    )

    return "\n".join(lines)
```

### 5. __init__.py - Public API

```python
# cvpipe/dashboard/__init__.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .collector import MetricsCollector
from .server import DashboardServer

if TYPE_CHECKING:
    from cvpipe import Pipeline

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for the metrics dashboard."""

    enabled: bool = True
    port: int = 8080
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
) -> DashboardServer | None:
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
            port=9000,
            prometheus=True,
            history_duration_minutes=10.0,
        ))

        pipeline.start()
    """
    if config is None:
        config = DashboardConfig()

    if not config.enabled:
        return None

    # Check dependencies
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        logger.warning(
            "[Dashboard] Dependencies not installed. "
            "Install with: pip install cvpipe[dashboard]"
        )
        return None

    # Create collector
    collector = MetricsCollector(
        latency_window=config.latency_window,
        history_duration_minutes=config.history_duration_minutes,
        fps_alpha=config.fps_alpha,
        max_errors_per_component=config.max_errors_per_component,
    )

    # Subscribe to events
    from cvpipe import (
        ComponentMetricEvent,
        ComponentErrorEvent,
        FrameDroppedEvent,
        PipelineStateEvent,
    )

    pipeline.event_bus.subscribe(ComponentMetricEvent, collector.on_component_metric)
    pipeline.event_bus.subscribe(FrameDroppedEvent, collector.on_frame_dropped)
    pipeline.event_bus.subscribe(ComponentErrorEvent, collector.on_component_error)
    pipeline.event_bus.subscribe(PipelineStateEvent, collector.on_pipeline_state)

    # Create and start server
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

    # Attach collector for custom metrics registration
    pipeline._dashboard_collector = collector

    return server
```

---

## YAML Configuration

```yaml
pipeline:
  source: webcam_source
  components:
    - module: preprocessor
      id: prep
    - module: detector
      id: detector
    - module: tracker
      id: tracker
    - module: result_assembler
      id: assembler

  dashboard:
    enabled: true
    port: 8881
    host: "0.0.0.0"

    # Features
    prometheus: true      # Enable /metrics endpoint
    websocket: true       # Enable /ws/metrics endpoint

    # Timing
    update_interval_ms: 1000     # WebSocket update frequency
    fps_alpha: 0.1               # EMA smoothing factor (lower = smoother)

    # History
    latency_window: 300          # Rolling window for percentiles (samples)
    history_duration_minutes: 5.0  # Time series history (minutes)

    # Errors
    max_errors_per_component: 10 # Recent errors to keep per component

    # Export
    export_on_shutdown: false    # Auto-export on pipeline.stop()
    export_path: "metrics_export.json"
```

---

## Usage Examples

### Basic Usage

```python
from cvpipe import build
from cvpipe.dashboard import enable_dashboard

pipeline = build(config_path, components_dir)
pipeline.validate()

# Enable dashboard with defaults
enable_dashboard(pipeline)

pipeline.start()
# Dashboard available at http://localhost:8080
```

### Custom Configuration

```python
from cvpipe.dashboard import enable_dashboard, DashboardConfig

pipeline = build(config_path, components_dir)
pipeline.validate()

enable_dashboard(pipeline, DashboardConfig(
    port=9000,
    host="127.0.0.1",
    prometheus=True,
    websocket=True,
    latency_window=500,
    history_duration_minutes=10.0,
    fps_alpha=0.05,  # Smoother FPS
))

pipeline.start()
```

### Custom Telemetry Events

```python
from dataclasses import dataclass
from cvpipe import Event, Component
from cvpipe.dashboard import enable_dashboard

@dataclass(frozen=True)
class GPUMemoryEvent(Event):
    component_id: str
    memory_mb: float
    utilization_pct: float

class YOLODetector(Component):
    INPUTS = [...]
    OUTPUTS = [...]

    def process(self, frame: Frame) -> None:
        # ... inference ...

        # Emit custom telemetry
        self.emit(GPUMemoryEvent(
            component_id=self._component_id,
            memory_mb=torch.cuda.memory_allocated() / 1e6,
            utilization_pct=torch.cuda.utilization(),
        ))

# Register custom event handler
pipeline = build(config_path, components_dir)
server = enable_dashboard(pipeline)

# Register custom metric handler
def handle_gpu_event(event: GPUMemoryEvent) -> dict:
    return {
        event.component_id: {
            "memory_mb": event.memory_mb,
            "utilization_pct": event.utilization_pct,
        }
    }

pipeline._dashboard_collector.register_custom_event(GPUMemoryEvent, handle_gpu_event)

pipeline.start()
```

---

## Dependencies

Add to `pyproject.toml` as **optional** dependencies:

```toml
[project.optional-dependencies]
dashboard = [
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
]
```

If not installed, `enable_dashboard()` logs a warning and returns `None`.

---

## Testing Strategy

```python
# tests/cvpipe/test_dashboard.py

import pytest
import threading
import time

from cvpipe.dashboard.collector import MetricsCollector
from cvpipe.dashboard.aggregator import FPSCalculator, LatencyHistory
from cvpipe import ComponentMetricEvent, FrameDroppedEvent, ComponentErrorEvent


class TestMetricsCollector:
    def test_latency_collection(self) -> None:
        collector = MetricsCollector(latency_window=100)
        event = ComponentMetricEvent(component_id="test", latency_ms=5.0, frame_idx=1)
        collector.on_component_metric(event)

        snapshot = collector.snapshot()
        assert "test" in snapshot["latency"]

    def test_drop_counting(self) -> None:
        collector = MetricsCollector()
        collector.on_frame_dropped(FrameDroppedEvent(reason="backpressure", frame_idx=1))
        collector.on_frame_dropped(FrameDroppedEvent(reason="backpressure", frame_idx=2))

        snapshot = collector.snapshot()
        assert snapshot["drops"]["by_reason"]["backpressure"] == 2

    def test_error_recording(self) -> None:
        collector = MetricsCollector(max_errors_per_component=5)
        collector.on_component_error(ComponentErrorEvent(
            component_id="detector",
            message="test error",
            traceback="test traceback",
            frame_idx=1,
        ))

        snapshot = collector.snapshot()
        assert snapshot["errors"]["total"] == 1
        assert len(snapshot["errors"]["recent"]) == 1

    def test_thread_safety(self) -> None:
        collector = MetricsCollector()
        errors = []

        def writer():
            for i in range(100):
                collector.on_component_metric(ComponentMetricEvent(
                    component_id="test",
                    latency_ms=float(i),
                    frame_idx=i,
                ))

        def reader():
            for i in range(100):
                try:
                    collector.snapshot()
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestFPSCalculator:
    def test_ema_calculation(self) -> None:
        fps = FPSCalculator(alpha=0.5)
        fps.update(0.0)
        fps.update(0.033)  # ~30 fps
        fps.update(0.066)

        assert 20 < fps.get() < 40

    def test_empty_returns_zero(self) -> None:
        fps = FPSCalculator()
        assert fps.get() == 0.0


class TestLatencyHistory:
    def test_bucket_aggregation(self) -> None:
        history = LatencyHistory(duration_minutes=1.0, resolution_seconds=1.0)

        # Add multiple samples in same bucket
        history.add("test", 1.0, 10.0)
        history.add("test", 1.5, 20.0)

        series = history.get_series("test")
        assert len(series) == 1
        assert series[0]["latency_ms"] == 15.0  # Average
        assert series[0]["samples"] == 2

    def test_ring_buffer(self) -> None:
        history = LatencyHistory(duration_minutes=0.1, resolution_seconds=0.1)

        for i in range(100):
            history.add("test", float(i) * 0.1, float(i))

        series = history.get_series("test")
        assert len(series) <= 60  # Max buckets
```

---

## Implementation Order

| Phase | Task | Estimated Time |
|-------|------|----------------|
| **1** | Create `dashboard/` package structure | 30 min |
| **2** | Implement `collector.py` | 2 hours |
| **3** | Implement `aggregator.py` | 1 hour |
| **4** | Implement `server.py` (FastAPI routes) | 2 hours |
| **5** | Implement `prometheus.py` | 1 hour |
| **6** | Create HTML template (`index.html`) | 2 hours |
| **7** | Create JavaScript (`dashboard.js`) | 1 hour |
| **8** | Create CSS (`dashboard.css`) | 30 min |
| **9** | Add `enable_dashboard()` to `__init__.py` | 1 hour |
| **10** | Add YAML config parsing to `config.py` | 1 hour |
| **11** | Update `Pipeline` to support dashboard | 1 hour |
| **12** | Write unit tests | 3 hours |
| **13** | Update documentation | 2 hours |

**Total: ~18 hours**

---

## Documentation Updates

### Files to Update

1. **`cvpipe/docs/index.md`** - Add dashboard to feature list
2. **`cvpipe/docs/observability.md`** - Add dashboard section
3. **`cvpipe/docs/api_reference.md`** - Document `DashboardConfig`, `enable_dashboard()`
4. **New: `cvpipe/docs/dashboard.md`** - Full dashboard documentation

### Example Addition to observability.md

```markdown
## Dashboard

cvpipe includes a built-in metrics dashboard for real-time pipeline monitoring.
Enable it with a single call before `pipeline.start()`:

\`\`\`python
from cvpipe import build
from cvpipe.dashboard import enable_dashboard

pipeline = build(config_path, components_dir)
enable_dashboard(pipeline)
pipeline.start()
\`\`\`

The dashboard provides:

- **Real-time latency charts** (p50, p95, p99 per component)
- **Frame drop tracking** (by reason)
- **Error logs** with full tracebacks
- **FPS monitoring** using exponential moving average
- **Prometheus endpoint** for integration with monitoring systems
- **WebSocket streaming** for live updates

Access the dashboard at `http://localhost:8080`.
```

---

## Acceptance Criteria

1. ✅ Dashboard starts with `enable_dashboard(pipeline)`
2. ✅ JSON endpoint `/api/v1/metrics` returns all metrics
3. ✅ Prometheus endpoint `/metrics` returns Prometheus format
4. ✅ WebSocket `/ws/metrics` streams updates every second
5. ✅ HTML dashboard displays real-time charts
6. ✅ FPS uses EMA (configurable alpha)
7. ✅ Latency history stored for configurable duration
8. ✅ Errors show component, message, traceback, frame_idx, timestamp
9. ✅ Custom telemetry events can be registered
10. ✅ All tests pass with 80%+ coverage
11. ✅ Documentation updated
