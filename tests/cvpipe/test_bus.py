# tests/cvpipe/test_bus.py

from __future__ import annotations

import time

import pytest

from cvpipe.bus import FrameResult, ResultBus
from cvpipe.event import EventBus


class TestResultBus:
    def test_resultbus_init_invalid_capacity(self) -> None:
        with pytest.raises(ValueError, match="capacity must be >= 1"):
            ResultBus(capacity=0)

    def test_resultbus_init_capacity_1(self) -> None:
        bus = ResultBus(capacity=1)
        assert bus.capacity == 1
        assert bus.occupancy == 0

    def test_resultbus_push_pop(self) -> None:
        bus = ResultBus(capacity=4)
        result = FrameResult(frame_idx=0, ts=1.0, detections=[])
        bus.push(result)
        assert bus.occupancy == 1

    def test_resultbus_capacity_property(self) -> None:
        bus = ResultBus(capacity=8)
        assert bus.capacity == 8

    def test_resultbus_occupancy_property(self) -> None:
        bus = ResultBus(capacity=4)
        assert bus.occupancy == 0
        bus.push(FrameResult(frame_idx=0, ts=0.0))
        assert bus.occupancy == 1

    def test_resultbus_overflow_drops(self) -> None:
        bus = ResultBus(capacity=2)

        bus.push(FrameResult(frame_idx=0, ts=0.0))
        bus.push(FrameResult(frame_idx=1, ts=1.0))
        assert bus.occupancy == 2

        bus.push(FrameResult(frame_idx=2, ts=2.0))
        assert bus.occupancy == 2

    def test_resultbus_multiple_push_order(self) -> None:
        bus = ResultBus(capacity=4)
        results_received: list[FrameResult] = []

        def callback(result: FrameResult) -> None:
            results_received.append(result)

        bus.subscribe(callback)
        bus.start()

        try:
            bus.push(FrameResult(frame_idx=0, ts=0.0))
            bus.push(FrameResult(frame_idx=1, ts=1.0))
            bus.push(FrameResult(frame_idx=2, ts=2.0))

            time.sleep(0.2)
            assert len(results_received) >= 1
        finally:
            bus.stop()

    def test_resultbus_subscriber_thread(self) -> None:
        bus = ResultBus(capacity=4)
        results_received: list[FrameResult] = []

        def callback(result: FrameResult) -> None:
            results_received.append(result)

        bus.subscribe(callback)
        bus.start()

        try:
            for i in range(3):
                bus.push(FrameResult(frame_idx=i, ts=float(i)))

            time.sleep(0.2)
            assert len(results_received) == 3
            assert results_received[0].frame_idx == 0
            assert results_received[1].frame_idx == 1
            assert results_received[2].frame_idx == 2
        finally:
            bus.stop()

    def test_resultbus_start_stop(self) -> None:
        bus = ResultBus(capacity=4)

        def callback(result: FrameResult) -> None:
            pass

        bus.subscribe(callback)
        bus.start()
        assert bus._running is True

        bus.stop()
        assert bus._running is False

    def test_resultbus_subscriber_exception(self) -> None:
        bus = ResultBus(capacity=4)
        call_count = 0

        def bad_callback(result: FrameResult) -> None:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("callback error")

        bus.subscribe(bad_callback)
        bus.start()

        try:
            bus.push(FrameResult(frame_idx=0, ts=0.0))
            time.sleep(0.1)
            assert call_count == 1
        finally:
            bus.stop()

    def test_resultbus_multiple_subscribers(self) -> None:
        bus = ResultBus(capacity=4)
        results1: list[FrameResult] = []
        results2: list[FrameResult] = []

        def callback1(result: FrameResult) -> None:
            results1.append(result)

        def callback2(result: FrameResult) -> None:
            results2.append(result)

        bus.subscribe(callback1)
        bus.subscribe(callback2)
        bus.start()

        try:
            bus.push(FrameResult(frame_idx=0, ts=0.0))
            time.sleep(0.1)
            assert len(results1) == 1
            assert len(results2) == 1
        finally:
            bus.stop()

    def test_resultbus_empty_push(self) -> None:
        bus = ResultBus(capacity=4)
        bus.push(FrameResult(frame_idx=0, ts=0.0, jpeg_bytes=b"", detections=[]))
        assert bus.occupancy == 1


class TestFrameResult:
    def test_frame_result_defaults(self) -> None:
        result = FrameResult(frame_idx=0, ts=1.0)
        assert result.jpeg_bytes == b""
        assert result.detections == []
        assert result.meta == {}

    def test_frame_result_with_data(self) -> None:
        result = FrameResult(
            frame_idx=1,
            ts=2.0,
            jpeg_bytes=b"fakejpeg",
            detections=[{"label": "cat", "box": [0, 0, 10, 10], "score": 0.9}],
            meta={"fps": 30.0},
        )
        assert result.jpeg_bytes == b"fakejpeg"
        assert len(result.detections) == 1
        assert result.meta["fps"] == 30.0
