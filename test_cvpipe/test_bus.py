# tests/cvpipe/test_bus.py
import time
import pytest
from cvpipe import ResultBus, FrameResult


def test_ring_buffer_drops_oldest():
    bus = ResultBus(capacity=4)
    received = []
    bus.subscribe(lambda r: received.append(r.frame_idx))
    bus.start()
    for i in range(6):
        bus.push(FrameResult(frame_idx=i, ts=float(i)))
    time.sleep(0.2)
    bus.stop()
    assert 5 in received
    assert 0 not in received or received.index(0) < received.index(5)


def test_two_subscribers_independent():
    bus = ResultBus(capacity=4)
    a, b = [], []
    bus.subscribe(lambda r: a.append(r.frame_idx))
    bus.subscribe(lambda r: b.append(r.frame_idx))
    bus.start()
    time.sleep(0.05)  # wait for subscriber threads to initialize
    bus.push(FrameResult(frame_idx=1, ts=1.0))
    time.sleep(0.3)
    bus.stop()
    assert 1 in a and 1 in b


def test_occupancy():
    bus = ResultBus(capacity=4)
    assert bus.occupancy == 0
    bus.start()
    for i in range(3):
        bus.push(FrameResult(frame_idx=i, ts=float(i)))
    assert bus.occupancy == 3


def test_capacity():
    bus = ResultBus(capacity=4)
    assert bus.capacity == 4


def test_capacity_zero_raises():
    with pytest.raises(ValueError):
        ResultBus(capacity=0)


def test_frame_result_defaults():
    r = FrameResult(frame_idx=1, ts=1.0)
    assert r.jpeg_bytes == b""
    assert r.detections == []
    assert r.meta == {}
