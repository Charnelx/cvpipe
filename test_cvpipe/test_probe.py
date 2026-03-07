# test_cvpipe/test_probe.py
import time
import pytest
from cvpipe import Probe
from cvpipe.scheduler import Scheduler, FrameSource


class CounterSource(FrameSource):
    def __init__(self, n: int = 3):
        self.n = n
        self._count = 0

    def next(self):
        if self._count >= self.n:
            return None
        self._count += 1
        return (f"frame_{self._count}", time.monotonic())


def test_probe_called_after_component(event_bus, result_bus):
    called_after = []

    class RecordProbe(Probe):
        def observe(self, frame, after_component):
            called_after.append(after_component)

    from test_cvpipe.conftest import make_slot_writer

    writer = make_slot_writer("x", 1)
    writer._component_id = "writer"
    source = CounterSource(n=3)

    sched = Scheduler(
        source=source, components=[writer], result_bus=result_bus, event_bus=event_bus
    )
    sched.add_probe(RecordProbe(), after="writer")
    sched.start()
    time.sleep(0.3)
    sched.stop()
    assert all(c == "writer" for c in called_after)
    assert len(called_after) == 3


def test_probe_exception_does_not_crash(event_bus, result_bus):

    class BadProbe(Probe):
        def observe(self, frame, after_component):
            raise RuntimeError("probe boom")

    from test_cvpipe.conftest import make_slot_writer

    writer = make_slot_writer("x", 1)
    writer._component_id = "w"
    source = CounterSource(n=3)

    sched = Scheduler(
        source=source, components=[writer], result_bus=result_bus, event_bus=event_bus
    )
    sched.add_probe(BadProbe(), after=None)
    sched.start()
    time.sleep(0.3)
    sched.stop()
    assert sched.frame_count == 3


def test_probe_abstract():
    with pytest.raises(TypeError):
        Probe()


def test_probe_after_none(event_bus, result_bus):
    calls = []

    class AllProbe(Probe):
        def observe(self, frame, after_component):
            calls.append(after_component)

    from test_cvpipe.conftest import make_slot_writer

    a = make_slot_writer("x", 1)
    a._component_id = "comp_a"
    b = make_slot_writer("y", 2)
    b._component_id = "comp_b"
    source = CounterSource(n=2)

    sched = Scheduler(source=source, components=[a, b], result_bus=result_bus, event_bus=event_bus)
    sched.add_probe(AllProbe(), after=None)
    sched.start()
    time.sleep(0.3)
    sched.stop()
    assert len(calls) == 4
