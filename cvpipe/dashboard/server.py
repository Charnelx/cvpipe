from __future__ import annotations

import asyncio
import json
import logging
import socket
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse


if TYPE_CHECKING:
    from cvpipe import Pipeline
    from cvpipe.dashboard.collector import MetricsCollector


class DashboardServer:
    """
    HTTP server for metrics dashboard.

    Runs in separate thread, does not block pipeline.
    """

    def __init__(
        self,
        collector: "MetricsCollector",
        pipeline: "Pipeline",
        port: int = 8080,
        host: str = "0.0.0.0",
        prometheus: bool = True,
        websocket: bool = True,
        update_interval_ms: int = 1000,
    ) -> None:
        self._collector = collector
        self._pipeline = pipeline
        self._port = port
        self._host = host
        self._prometheus = prometheus
        self._websocket = websocket
        self._update_interval = update_interval_ms / 1000.0

        self._thread: threading.Thread | None = None
        self._running = False

        self._app: Any = None
        self._setup_routes()

    def _setup_routes(self) -> None:
        self._app = FastAPI(title="cvpipe Dashboard")

        # --- CORS middleware ---
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:8881", "http://127.0.0.1:8881"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # -----------------------

        @self._app.get("/", response_class=HTMLResponse)
        async def dashboard() -> str:
            return self._render_html()

        @self._app.get("/api/v1/metrics")
        async def get_metrics() -> dict[str, Any]:
            return self._collector.snapshot()

        @self._app.get("/api/v1/metrics/latency")
        async def get_latency() -> dict[str, Any]:
            return self._collector.snapshot().get("latency", {})

        @self._app.get("/api/v1/metrics/drops")
        async def get_drops() -> dict[str, Any]:
            return self._collector.snapshot().get("drops", {})

        @self._app.get("/api/v1/metrics/errors")
        async def get_errors() -> dict[str, Any]:
            return self._collector.snapshot().get("errors", {})

        @self._app.get("/api/v1/metrics/state")
        async def get_state() -> dict[str, Any]:
            return self._collector.snapshot().get("state", {})

        @self._app.get("/api/v1/metrics/fps")
        async def get_fps() -> dict[str, Any]:
            return self._collector.snapshot().get("fps", {})

        @self._app.get("/api/v1/metrics/history")
        async def get_history() -> dict[str, Any]:
            return self._collector.snapshot().get("latency_history", {})

        if self._prometheus:
            from .prometheus import render_prometheus

            @self._app.get("/metrics", response_class=PlainTextResponse)
            async def prometheus_metrics() -> str:
                return render_prometheus(self._collector.snapshot())

        @self._app.post("/api/v1/metrics/export")
        async def export_metrics(path: str = "metrics_export.json") -> dict[str, str]:
            data = self._collector.snapshot()
            data["export_ts"] = time.time()
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            return {"status": "ok", "path": path}

        if self._websocket:
            from starlette.websockets import WebSocketDisconnect

            @self._app.websocket("/ws/metrics")
            async def websocket_metrics(websocket: WebSocket) -> None:
                await websocket.accept()
                try:
                    while self._running:
                        data = self._collector.snapshot()
                        await websocket.send_json(data)
                        await asyncio.sleep(self._update_interval)
                except WebSocketDisconnect:
                    logger.info("[Dashboard] WebSocket client disconnected")
                except Exception:
                    logger.exception("[Dashboard] WebSocket error")

    def _render_html(self) -> str:
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
        self._wait_until_ready(timeout=5.0)
        logger.info(
            "[Dashboard] Server started at http://%s:%d",
            self._host,
            self._port,
        )

    def _wait_until_ready(self, timeout: float) -> None:
        """Wait for uvicorn server to be ready to accept connections."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                sock.connect((self._host, self._port))
                sock.close()
                return
            except (ConnectionRefusedError, socket.timeout, OSError):
                time.sleep(0.1)
        logger.warning("[Dashboard] Server ready wait timed out")

    def _run_server(self) -> None:
        import uvicorn

        config = uvicorn.Config(
            self._app,
            host=self._host,
            port=self._port,
            log_level="info",
            ws="auto",
        )
        server = uvicorn.Server(config)
        server.run()

    def stop(self) -> None:
        """Stop the server."""
        self._running = False
        logger.info("[Dashboard] Server stopped")
