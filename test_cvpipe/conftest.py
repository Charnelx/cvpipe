# tests/cvpipe/conftest.py
import pytest
from cvpipe import Frame, SlotSchema, Component, EventBus, ResultBus
from cvpipe.scheduler import FrameSource


class PassthroughComponent(Component):
    """Component that does nothing. Used to test pipeline wiring."""

    INPUTS = []
    OUTPUTS = []

    def process(self, frame: Frame) -> None:
        pass


class CounterSource(FrameSource):
    """FrameSource that emits N synthetic frames then stops."""

    def __init__(self, n: int = 5) -> None:
        self.n = n
        self._count = 0
        self.stopped = False

    def next(self):
        import time

        if self._count >= self.n:
            self.stopped = True
            return None
        self._count += 1
        return (f"frame_{self._count}", time.monotonic())


def make_slot_writer(slot_name: str, value):
    """Factory to create a SlotWriterComponent with a specific slot name."""

    class SlotWriterComponent(Component):
        OUTPUTS = [SlotSchema(slot_name, None, (), "any")]

        def __init__(self):
            super().__init__()

        def process(self, frame: Frame) -> None:
            frame.slots[slot_name] = value

    return SlotWriterComponent()


def make_slot_reader(slot_name: str):
    """Factory to create a SlotReaderComponent with a specific slot name."""

    class SlotReaderComponent(Component):
        INPUTS = [SlotSchema(slot_name, None, (), "any")]

        def __init__(self):
            super().__init__()
            self.seen = []

        def process(self, frame: Frame) -> None:
            self.seen.append(frame.slots.get(slot_name))

    return SlotReaderComponent()


@pytest.fixture
def event_bus():
    bus = EventBus()
    bus.start()
    yield bus
    bus.stop()


@pytest.fixture
def result_bus():
    bus = ResultBus(capacity=4)
    bus.start()
    yield bus
    bus.stop()
