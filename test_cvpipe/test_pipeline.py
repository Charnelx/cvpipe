# test_cvpipe/test_pipeline.py
import time
import pytest
from dataclasses import dataclass
from cvpipe import (
    Pipeline,
    SlotSchema,
    Component,
    Event,
    PipelineConfigError,
)
from cvpipe.scheduler import FrameSource


class CounterSource(FrameSource):
    def __init__(self, n: int = 5):
        self.n = n
        self._count = 0

    def next(self):
        if self._count >= self.n:
            return None
        self._count += 1
        return (f"frame_{self._count}", time.monotonic())


def test_validation_passes():
    from test_cvpipe.conftest import make_slot_writer, make_slot_reader

    writer = make_slot_writer("data", 42)
    writer._component_id = "writer"
    reader = make_slot_reader("data")
    reader._component_id = "reader"

    pipeline = Pipeline(
        source=CounterSource(n=0),
        components=[writer, reader],
    )
    pipeline.validate()


def test_validation_missing_slot():
    from test_cvpipe.conftest import make_slot_reader

    reader = make_slot_reader("data")
    reader._component_id = "reader"

    pipeline = Pipeline(
        source=CounterSource(n=0),
        components=[reader],
    )
    with pytest.raises(PipelineConfigError) as exc_info:
        pipeline.validate()
    assert "data" in str(exc_info.value)


def test_validation_all_errors_collected():

    class ReadsAB(Component):
        INPUTS = [SlotSchema("a", None), SlotSchema("b", None)]

        def process(self, frame):
            pass

    comp = ReadsAB()
    comp._component_id = "reads_ab"
    pipeline = Pipeline(source=CounterSource(n=0), components=[comp])
    with pytest.raises(PipelineConfigError) as exc_info:
        pipeline.validate()
    assert len(exc_info.value.errors) == 2


def test_start_stop():
    from test_cvpipe.conftest import make_slot_writer

    writer = make_slot_writer("x", 1)
    writer._component_id = "w"
    source = CounterSource(n=3)
    pipeline = Pipeline(source=source, components=[writer])
    pipeline.validate()
    pipeline.start()
    time.sleep(0.3)
    pipeline.stop()
    assert not pipeline.is_running


def test_event_subscription_wired(event_bus):

    @dataclass(frozen=True)
    class MyEvent(Event):
        value: int

    received = []

    class SubscribingComp(Component):
        SUBSCRIBES = [MyEvent]
        INPUTS = []
        OUTPUTS = []

        def process(self, frame):
            pass

        def on_event(self, event):
            received.append(event.value)

    comp = SubscribingComp()
    pipeline = Pipeline(
        source=CounterSource(n=0),
        components=[comp],
        event_bus=event_bus,
    )
    pipeline.start()
    event_bus.publish(MyEvent(value=99))
    time.sleep(0.1)
    pipeline.stop()
    assert 99 in received


def test_is_running(event_bus, result_bus):
    pipeline = Pipeline(
        source=CounterSource(n=1),
        components=[],
        event_bus=event_bus,
        result_bus=result_bus,
    )
    assert not pipeline.is_running
    pipeline.start()
    assert pipeline.is_running
    pipeline.stop()
    assert not pipeline.is_running


def test_result_bus_accessible():
    pipeline = Pipeline(source=CounterSource(n=0), components=[])
    assert pipeline.result_bus is not None


def test_event_bus_accessible():
    pipeline = Pipeline(source=CounterSource(n=0), components=[])
    assert pipeline.event_bus is not None


def test_component_accessor_returns_correct_component():
    from test_cvpipe.conftest import make_slot_writer

    writer = make_slot_writer("data", 42)
    writer._component_id = "my_writer"

    pipeline = Pipeline(
        source=CounterSource(n=0),
        components=[writer],
    )

    retrieved = pipeline.component("my_writer")
    assert retrieved is writer


def test_component_accessor_case_classname():
    class MyComponent(Component):
        INPUTS = []
        OUTPUTS = []

        def process(self, frame):
            pass

    comp = MyComponent()
    pipeline = Pipeline(
        source=CounterSource(n=0),
        components=[comp],
    )

    retrieved = pipeline.component("MyComponent")
    assert retrieved is comp


def test_component_accessor_raises_keyerror():
    pipeline = Pipeline(source=CounterSource(n=0), components=[])

    with pytest.raises(KeyError) as exc_info:
        pipeline.component("nonexistent")

    assert "nonexistent" in str(exc_info.value)
    assert "Available" in str(exc_info.value)


def test_component_accessor_before_start():
    from test_cvpipe.conftest import make_slot_writer

    writer = make_slot_writer("data", 42)
    writer._component_id = "writer"

    pipeline = Pipeline(
        source=CounterSource(n=0),
        components=[writer],
    )

    retrieved = pipeline.component("writer")
    assert retrieved is writer


def test_component_accessor_after_stop():
    from test_cvpipe.conftest import make_slot_writer

    writer = make_slot_writer("data", 42)
    writer._component_id = "writer"

    pipeline = Pipeline(source=CounterSource(n=1), components=[writer])
    pipeline.start()
    time.sleep(0.2)
    pipeline.stop()

    retrieved = pipeline.component("writer")
    assert retrieved is writer
