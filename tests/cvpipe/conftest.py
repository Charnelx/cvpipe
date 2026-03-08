# tests/cvpipe/conftest.py
# Shared test fixtures and test doubles for cvpipe unit tests.

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import pytest

from cvpipe.bus import FrameResult, ResultBus
from cvpipe.component import Component
from cvpipe.event import Event, EventBus
from cvpipe.frame import Frame, SlotSchema
from cvpipe.scheduler import FrameSource


def wait_for(
    condition: Callable[[], bool], timeout: float = 1.0, interval: float = 0.01
) -> None:
    """Poll until condition is true or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(interval)
    raise TimeoutError("Condition not met within timeout")


class CounterSource(FrameSource):
    """FrameSource that yields N frames, then returns None."""

    def __init__(self, n: int, payload: Any = None) -> None:
        self._n = n
        self._payload = payload
        self._count = 0

    def setup(self) -> None:
        pass

    def teardown(self) -> None:
        pass

    def next(self) -> tuple[Any, float] | None:
        if self._count >= self._n:
            return None
        ts = time.monotonic()
        self._count += 1
        return (self._payload, ts)


class PassthroughComponent(Component):
    """Component with no inputs or outputs, process() is a no-op."""

    INPUTS: list[SlotSchema] = []
    OUTPUTS: list[SlotSchema] = []

    def process(self, frame: Frame) -> None:
        pass


class SlotWriterComponent(Component):
    """Component that writes a fixed value to a named slot."""

    INPUTS: list[SlotSchema] = []
    OUTPUTS: list[SlotSchema] = [
        SlotSchema("dummy", type(None)),
    ]

    def __init__(self, slot_name: str, value: Any, into_meta: bool = False) -> None:
        super().__init__()
        self._slot_name = slot_name
        self._value = value
        self._into_meta = into_meta

    def process(self, frame: Frame) -> None:
        if self._into_meta:
            frame.meta[self._slot_name] = self._value
        else:
            frame.slots[self._slot_name] = self._value


class SlotReaderComponent(Component):
    """Component that reads a slot and stores values in a list."""

    INPUTS: list[SlotSchema] = []
    OUTPUTS: list[SlotSchema] = []

    def __init__(self, slot_name: str, from_meta: bool = False) -> None:
        super().__init__()
        self._slot_name = slot_name
        self._from_meta = from_meta
        self._values: list[Any] = []

    def process(self, frame: Frame) -> None:
        if self._from_meta:
            self._values.append(frame.meta.get(self._slot_name))
        else:
            self._values.append(frame.slots.get(self._slot_name))


class ErrorComponent(Component):
    """Component that raises a configured exception on process()."""

    INPUTS: list[SlotSchema] = []
    OUTPUTS: list[SlotSchema] = []

    def __init__(self, exception: Exception) -> None:
        super().__init__()
        self._exception = exception

    def process(self, frame: Frame) -> None:
        raise self._exception


class EventCollector:
    """Context manager that collects events published to an EventBus."""

    def __init__(self, event_bus: EventBus, event_type: type[Event]) -> None:
        self._event_bus = event_bus
        self._event_type = event_type
        self._events: list[Event] = []
        self._handler: Callable[[Event], None] | None = None

    def __enter__(self) -> "EventCollector":
        self._handler = self._collect
        self._event_bus.subscribe(self._event_type, self._handler)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._handler is not None:
            self._event_bus._handlers[self._event_type].remove(self._handler)

    def _collect(self, event: Event) -> None:
        self._events.append(event)

    @property
    def events(self) -> list[Event]:
        return self._events


class ResultCollector:
    """Context manager that collects results from a ResultBus."""

    def __init__(self, result_bus: ResultBus) -> None:
        self._result_bus = result_bus
        self._results: list[FrameResult] = []
        self._callback: Callable[[FrameResult], None] | None = None

    def __enter__(self) -> "ResultCollector":
        self._callback = self._collect
        self._result_bus.subscribe(self._callback)
        self._result_bus.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._result_bus.stop()

    def _collect(self, result: FrameResult) -> None:
        self._results.append(result)

    @property
    def results(self) -> list[FrameResult]:
        return self._results


@pytest.fixture
def empty_frame() -> Frame:
    """A Frame with no slots or meta."""
    return Frame(idx=0, ts=0.0)


@pytest.fixture
def frame_with_slots() -> Frame:
    """A Frame pre-populated with some slots and meta."""
    frame = Frame(idx=1, ts=1.0)
    frame.slots["tensor_slot"] = "fake_tensor"
    frame.meta["meta_slot"] = "fake_meta"
    return frame


@pytest.fixture
def event_bus() -> EventBus:
    """A fresh EventBus (not started)."""
    return EventBus()


@pytest.fixture
def started_event_bus() -> EventBus:
    """A started EventBus (remember to stop in teardown)."""
    bus = EventBus()
    bus.start()
    yield bus
    bus.stop()


@pytest.fixture
def result_bus() -> ResultBus:
    """A fresh ResultBus."""
    return ResultBus(capacity=4)
