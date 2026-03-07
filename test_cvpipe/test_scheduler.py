# tests/cvpipe/test_scheduler.py
import time
import pytest
from cvpipe import Component
from cvpipe.event import ComponentErrorEvent, FrameDroppedEvent
from cvpipe.scheduler import Scheduler, FrameSource


class CounterSource(FrameSource):
    def __init__(self, n: int = 5):
        self.n = n
        self._count = 0

    def next(self):
        if self._count >= self.n:
            return None
        self._count += 1
        return (f"frame_{self._count}", time.monotonic())


class PassthroughComponent(Component):
    INPUTS = []
    OUTPUTS = []

    def process(self, frame):
        pass


def test_frames_processed(event_bus, result_bus):
    source = CounterSource(n=5)
    comp = PassthroughComponent()
    comp._component_id = "pass"

    sched = Scheduler(source=source, components=[comp], result_bus=result_bus, event_bus=event_bus)
    sched.start()
    time.sleep(0.5)
    sched.stop()
    assert sched.frame_count == 5


def test_component_error_continues(event_bus, result_bus):
    source = CounterSource(n=10)
    errors = []

    class BoomComp(Component):
        INPUTS = []
        OUTPUTS = []

        def process(self, frame):
            raise ValueError("boom")

    comp = BoomComp()
    comp._component_id = "boom"
    event_bus.subscribe(ComponentErrorEvent, lambda e: errors.append(e))

    sched = Scheduler(source=source, components=[comp], result_bus=result_bus, event_bus=event_bus)
    sched.start()
    time.sleep(0.5)
    sched.stop()
    assert len(errors) == 10


def test_source_stall_emits_event(event_bus, result_bus):
    drops = []
    event_bus.subscribe(FrameDroppedEvent, lambda e: drops.append(e))

    source = CounterSource(n=0)
    sched = Scheduler(source=source, components=[], result_bus=result_bus, event_bus=event_bus)
    sched.start()
    time.sleep(0.2)
    sched.stop()
    stall_events = [d for d in drops if d.reason == "source_stall"]
    assert len(stall_events) > 0


def test_start_twice_raises(event_bus, result_bus):
    source = CounterSource(n=1)
    comp = PassthroughComponent()
    comp._component_id = "test"
    sched = Scheduler(source=source, components=[comp], result_bus=result_bus, event_bus=event_bus)
    sched.start()
    with pytest.raises(RuntimeError):
        sched.start()
    sched.stop()


def test_frame_count_only_success(event_bus, result_bus):
    source = CounterSource(n=5)

    class FailOnSecond(Component):
        INPUTS = []
        OUTPUTS = []

        def process(self, frame):
            if frame.idx == 1:
                raise ValueError("fail")

    comp = FailOnSecond()
    comp._component_id = "failer"
    event_bus.subscribe(ComponentErrorEvent, lambda e: None)

    sched = Scheduler(source=source, components=[comp], result_bus=result_bus, event_bus=event_bus)
    sched.start()
    time.sleep(0.5)
    sched.stop()
    assert sched.frame_count < 5
