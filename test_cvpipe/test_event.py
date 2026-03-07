# tests/cvpipe/test_event.py
import time
from dataclasses import dataclass
from cvpipe import Event


@dataclass(frozen=True)
class TestEvent(Event):
    value: int


@dataclass(frozen=True)
class EventA(Event):
    pass


@dataclass(frozen=True)
class EventB(Event):
    pass


def test_handler_called(event_bus):
    received = []
    event_bus.subscribe(TestEvent, lambda e: received.append(e.value))
    event_bus.publish(TestEvent(value=42))
    time.sleep(0.1)
    assert received == [42]


def test_exact_type_matching(event_bus):
    received_a = []
    event_bus.subscribe(EventA, lambda e: received_a.append(True))
    event_bus.publish(EventB())
    time.sleep(0.1)
    assert received_a == []


def test_handler_exception_does_not_stop_bus(event_bus):
    received = []

    def bad_handler(e):
        raise RuntimeError("boom")

    def good_handler(e):
        received.append(True)

    event_bus.subscribe(TestEvent, bad_handler)
    event_bus.subscribe(TestEvent, good_handler)
    event_bus.publish(TestEvent(value=1))
    time.sleep(0.1)
    assert received == [True]


def test_publish_before_start(event_bus):
    event_bus.publish(TestEvent(value=1))
    time.sleep(0.1)


def test_handler_count(event_bus):
    assert event_bus.handler_count(TestEvent) == 0
    event_bus.subscribe(TestEvent, lambda e: None)
    assert event_bus.handler_count(TestEvent) == 1


def test_multiple_handlers_order(event_bus):
    order = []

    def first(e):
        order.append(1)

    def second(e):
        order.append(2)

    event_bus.subscribe(TestEvent, first)
    event_bus.subscribe(TestEvent, second)
    event_bus.publish(TestEvent(value=1))
    time.sleep(0.1)
    assert order == [1, 2]
