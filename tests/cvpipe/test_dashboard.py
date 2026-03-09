from __future__ import annotations

import threading
import time

import pytest

from cvpipe.dashboard.aggregator import (
    ErrorRecord,
    FPSCalculator,
    LatencyHistory,
    compute_percentiles,
)
from cvpipe.dashboard.collector import MetricsCollector
from cvpipe.dashboard.prometheus import render_prometheus
from cvpipe import (
    ComponentErrorEvent,
    ComponentMetricEvent,
    FrameDroppedEvent,
    PipelineStateEvent,
)


class TestComputePercentiles:
    def test_empty_samples(self) -> None:
        result = compute_percentiles([])
        assert result == {}

    def test_single_sample(self) -> None:
        result = compute_percentiles([5.0])
        assert result["p50_ms"] == 5.0
        assert result["p95_ms"] == 5.0
        assert result["p99_ms"] == 5.0
        assert result["min_ms"] == 5.0
        assert result["max_ms"] == 5.0
        assert result["samples"] == 1

    def test_multiple_samples(self) -> None:
        samples = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        result = compute_percentiles(samples)
        assert result["min_ms"] == 1.0
        assert result["max_ms"] == 10.0
        assert result["samples"] == 10


class TestFPSCalculator:
    def test_empty_returns_zero(self) -> None:
        fps = FPSCalculator()
        assert fps.get() == 0.0

    def test_single_update_returns_zero(self) -> None:
        fps = FPSCalculator(alpha=0.5)
        fps.update(0.0)
        assert fps.get() == 0.0

    def test_ema_calculation(self) -> None:
        fps = FPSCalculator(alpha=0.5)
        fps.update(0.0)
        fps.update(0.033)
        fps.update(0.066)

        assert 20 < fps.get() < 40

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError):
            FPSCalculator(alpha=0.0)
        with pytest.raises(ValueError):
            FPSCalculator(alpha=1.0)
        with pytest.raises(ValueError):
            FPSCalculator(alpha=-0.1)

    def test_thread_safety(self) -> None:
        fps = FPSCalculator(alpha=0.1)
        errors: list[Exception] = []

        def writer() -> None:
            for i in range(100):
                fps.update(float(i) * 0.033)

        def reader() -> None:
            for _ in range(100):
                try:
                    val = fps.get()
                    assert val >= 0
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


class TestLatencyHistory:
    def test_empty_history(self) -> None:
        history = LatencyHistory()
        assert history.get_series("nonexistent") == []

    def test_bucket_aggregation(self) -> None:
        history = LatencyHistory(duration_minutes=1.0, resolution_seconds=1.0)

        history.add("test", 1.0, 10.0)
        history.add("test", 1.5, 20.0)

        series = history.get_series("test")
        assert len(series) == 1
        assert series[0]["latency_ms"] == 15.0
        assert series[0]["samples"] == 2

    def test_separate_buckets(self) -> None:
        history = LatencyHistory(duration_minutes=1.0, resolution_seconds=1.0)

        history.add("test", 1.0, 10.0)
        history.add("test", 2.0, 20.0)

        series = history.get_series("test")
        assert len(series) == 2

    def test_ring_buffer(self) -> None:
        history = LatencyHistory(duration_minutes=0.1, resolution_seconds=0.1)

        for i in range(100):
            history.add("test", float(i) * 0.1, float(i))

        series = history.get_series("test")
        assert len(series) <= 60

    def test_get_all_series(self) -> None:
        history = LatencyHistory()

        history.add("comp1", 1.0, 10.0)
        history.add("comp2", 1.0, 20.0)

        all_series = history.get_all_series()
        assert "comp1" in all_series
        assert "comp2" in all_series


class TestErrorRecord:
    def test_to_dict(self) -> None:
        record = ErrorRecord(
            component_id="detector",
            message="CUDA out of memory",
            traceback="Traceback...",
            frame_idx=1234,
            ts=1709925900.123,
        )

        result = record.to_dict()
        assert result["component_id"] == "detector"
        assert result["message"] == "CUDA out of memory"
        assert result["traceback"] == "Traceback..."
        assert result["frame_idx"] == 1234
        assert result["ts"] == 1709925900.123
        assert "ts_iso" in result


class TestMetricsCollector:
    def test_latency_collection(self) -> None:
        collector = MetricsCollector(latency_window=100)
        event = ComponentMetricEvent(
            component_id="test",
            latency_ms=5.0,
            frame_idx=1,
        )
        collector.on_component_metric(event)

        snapshot = collector.snapshot()
        assert "test" in snapshot["latency"]
        assert snapshot["latency"]["test"]["samples"] == 1

    def test_latency_percentiles(self) -> None:
        collector = MetricsCollector(latency_window=100)

        for i in range(10):
            collector.on_component_metric(
                ComponentMetricEvent(
                    component_id="test",
                    latency_ms=float(i + 1),
                    frame_idx=i,
                )
            )

        snapshot = collector.snapshot()
        stats = snapshot["latency"]["test"]
        assert stats["min_ms"] == 1.0
        assert stats["max_ms"] == 10.0
        assert stats["p50_ms"] == 6.0

    def test_drop_counting(self) -> None:
        collector = MetricsCollector()
        collector.on_frame_dropped(
            FrameDroppedEvent(reason="backpressure", frame_idx=1)
        )
        collector.on_frame_dropped(
            FrameDroppedEvent(reason="backpressure", frame_idx=2)
        )
        collector.on_frame_dropped(
            FrameDroppedEvent(reason="component_error", frame_idx=3)
        )

        snapshot = collector.snapshot()
        assert snapshot["drops"]["total"] == 3
        assert snapshot["drops"]["by_reason"]["backpressure"] == 2
        assert snapshot["drops"]["by_reason"]["component_error"] == 1

    def test_error_recording(self) -> None:
        collector = MetricsCollector(max_errors_per_component=5)
        collector.on_component_error(
            ComponentErrorEvent(
                component_id="detector",
                message="test error",
                traceback="test traceback",
                frame_idx=1,
            )
        )

        snapshot = collector.snapshot()
        assert snapshot["errors"]["total"] == 1
        assert len(snapshot["errors"]["recent"]) == 1
        assert snapshot["errors"]["by_component"]["detector"] == 1

    def test_error_limit_per_component(self) -> None:
        collector = MetricsCollector(max_errors_per_component=3)

        for i in range(5):
            collector.on_component_error(
                ComponentErrorEvent(
                    component_id="detector",
                    message=f"error {i}",
                    traceback=f"traceback {i}",
                    frame_idx=i,
                )
            )

        snapshot = collector.snapshot()
        assert snapshot["errors"]["total"] == 5
        assert snapshot["errors"]["by_component"]["detector"] == 3
        assert len(snapshot["errors"]["recent"]) == 3

    def test_pipeline_state(self) -> None:
        collector = MetricsCollector()
        collector.on_pipeline_state(PipelineStateEvent(state="running"))

        snapshot = collector.snapshot()
        assert snapshot["state"]["status"] == "running"

    def test_fps_updates(self) -> None:
        collector = MetricsCollector(fps_alpha=0.5)

        for i in range(5):
            collector.on_component_metric(
                ComponentMetricEvent(
                    component_id="test",
                    latency_ms=1.0,
                    frame_idx=i,
                )
            )

        snapshot = collector.snapshot()
        assert snapshot["fps"]["current"] > 0

    def test_frame_count(self) -> None:
        collector = MetricsCollector()

        for i in range(10):
            collector.on_component_metric(
                ComponentMetricEvent(
                    component_id="test",
                    latency_ms=1.0,
                    frame_idx=i,
                )
            )

        snapshot = collector.snapshot()
        assert snapshot["state"]["frame_count"] == 10

    def test_thread_safety(self) -> None:
        collector = MetricsCollector()
        errors: list[Exception] = []

        def writer() -> None:
            for i in range(100):
                collector.on_component_metric(
                    ComponentMetricEvent(
                        component_id="test",
                        latency_ms=float(i),
                        frame_idx=i,
                    )
                )

        def reader() -> None:
            for _ in range(100):
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

    def test_custom_event_handler(self) -> None:
        collector = MetricsCollector()

        class CustomEvent:
            def __init__(self, value: int) -> None:
                self.value = value
                self.ts = time.monotonic()

        def handler(event: CustomEvent) -> dict:
            return {"custom_value": event.value}

        collector.register_custom_event(CustomEvent, handler)  # type: ignore[arg-type]
        collector.on_custom_event(CustomEvent(42))  # type: ignore[arg-type]

        snapshot = collector.snapshot()
        assert "custom_value" in snapshot["custom"]


class TestRenderPrometheus:
    def test_empty_metrics(self) -> None:
        result = render_prometheus({})
        assert "cvpipe_fps" in result
        assert "cvpipe_errors_total" in result

    def test_latency_metrics(self) -> None:
        metrics = {
            "latency": {
                "detector": {
                    "p50_ms": 10.0,
                    "p95_ms": 20.0,
                    "p99_ms": 30.0,
                    "samples": 100,
                }
            }
        }
        result = render_prometheus(metrics)
        assert 'component="detector"' in result
        assert 'quantile="0.5"' in result
        assert "10.000" in result

    def test_drop_metrics(self) -> None:
        metrics = {
            "drops": {
                "total": 5,
                "by_reason": {"backpressure": 3, "component_error": 2},
            }
        }
        result = render_prometheus(metrics)
        assert 'cvpipe_frame_drops_total{reason="backpressure"} 3' in result
        assert 'cvpipe_frame_drops_total{reason="component_error"} 2' in result

    def test_error_metrics(self) -> None:
        metrics = {"errors": {"total": 10}}
        result = render_prometheus(metrics)
        assert "cvpipe_errors_total 10" in result

    def test_state_metrics_running(self) -> None:
        metrics = {"state": {"status": "running"}}
        result = render_prometheus(metrics)
        assert "cvpipe_pipeline_state 1" in result

    def test_state_metrics_stopped(self) -> None:
        metrics = {"state": {"status": "stopped"}}
        result = render_prometheus(metrics)
        assert "cvpipe_pipeline_state 0" in result

    def test_fps_metrics(self) -> None:
        metrics = {"fps": {"current": 28.5}}
        result = render_prometheus(metrics)
        assert "cvpipe_fps 28.5" in result

    def test_frame_count_metrics(self) -> None:
        metrics = {"state": {"frame_count": 1000}}
        result = render_prometheus(metrics)
        assert "cvpipe_frames_total 1000" in result

    def test_custom_metrics(self) -> None:
        metrics = {
            "custom": {
                "gpu": {
                    "detector": {
                        "memory_mb": 2450.5,
                        "utilization_pct": 85.2,
                    }
                }
            }
        }
        result = render_prometheus(metrics)
        assert 'cvpipe_custom_metric{category="gpu"' in result
        assert "memory_mb" in result
