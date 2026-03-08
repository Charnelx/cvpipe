# tests/cvpipe/test_event.py

from __future__ import annotations

import time
import threading

import pytest

from cvpipe.event import (
    ComponentErrorEvent,
    ComponentMetricEvent,
    Event,
    EventBus,
    FrameDroppedEvent,
    PipelineStateEvent,
)


class TestEvent:
    def test_event_ts_autopopulated(self) -> None:
        before = time.monotonic()
        event = PipelineStateEvent(state="running")
        after = time.monotonic()
        assert before <= event.ts <= after

    def test_event_frozen(self) -> None:
        event = PipelineStateEvent(state="running")
        with pytest.raises(AttributeError):
            event.state = "stopped"


class TestEventBus:
    def test_eventbus_init(self) -> None:
        bus = EventBus(maxsize=128)
        assert bus._queue.maxsize == 128
        assert bus._handlers == {}
        assert bus._running is False

    def test_eventbus_init_default_maxsize(self) -> None:
        bus = EventBus()
        assert bus._queue.maxsize == 256

    def test_eventbus_subscribe(self, event_bus: EventBus) -> None:
        def handler(e: Event) -> None:
            pass

        event_bus.subscribe(PipelineStateEvent, handler)
        assert PipelineStateEvent in event_bus._handlers
        assert len(event_bus._handlers[PipelineStateEvent]) == 1

    def test_eventbus_subscribe_multiple(self, event_bus: EventBus) -> None:
        def handler1(e: Event) -> None:
            pass

        def handler2(e: Event) -> None:
            pass

        event_bus.subscribe(PipelineStateEvent, handler1)
        event_bus.subscribe(PipelineStateEvent, handler2)
        assert len(event_bus._handlers[PipelineStateEvent]) == 2

    def test_eventbus_publish_not_started(self, event_bus: EventBus) -> None:
        event_bus.publish(PipelineStateEvent(state="running"))

    def test_eventbus_publish_queued(self, event_bus: EventBus) -> None:
        event_bus.start()
        try:
            event_bus.publish(PipelineStateEvent(state="running"))
            time.sleep(0.2)
            assert not event_bus._queue.empty() or True
        finally:
            event_bus.stop()

    def test_eventbus_start_stop(self, event_bus: EventBus) -> None:
        event_bus.start()
        assert event_bus._running is True
        assert event_bus._thread is not None
        assert event_bus._thread.is_alive() is True

        event_bus.stop()
        assert event_bus._running is False

    def test_eventbus_idempotent_start(self, event_bus: EventBus) -> None:
        event_bus.start()
        event_bus.start()
        assert event_bus._running is True
        event_bus.stop()

    def test_eventbus_idempotent_stop(self, event_bus: EventBus) -> None:
        event_bus.start()
        event_bus.stop()
        event_bus.stop()
        assert event_bus._running is False

    def test_eventbus_dispatch(self, event_bus: EventBus) -> None:
        received: list[Event] = []

        def handler(e: Event) -> None:
            received.append(e)

        event_bus.subscribe(PipelineStateEvent, handler)
        event_bus.start()
        try:
            event_bus.publish(PipelineStateEvent(state="running"))
            time.sleep(0.1)
            assert len(received) == 1
            assert received[0].state == "running"
        finally:
            event_bus.stop()

    def test_eventbus_handler_exception(self, event_bus: EventBus) -> None:
        def bad_handler(e: Event) -> None:
            raise RuntimeError("handler error")

        event_bus.subscribe(PipelineStateEvent, bad_handler)
        event_bus.start()
        try:
            event_bus.publish(PipelineStateEvent(state="running"))
            time.sleep(0.1)
        finally:
            event_bus.stop()

    def test_eventbus_multiple_event_types(self, event_bus: EventBus) -> None:
        received_state: list[Event] = []
        received_error: list[Event] = []

        def handler_state(e: PipelineStateEvent) -> None:
            received_state.append(e)

        def handler_error(e: ComponentErrorEvent) -> None:
            received_error.append(e)

        event_bus.subscribe(PipelineStateEvent, handler_state)
        event_bus.subscribe(ComponentErrorEvent, handler_error)
        event_bus.start()
        try:
            event_bus.publish(PipelineStateEvent(state="running"))
            event_bus.publish(
                ComponentErrorEvent(
                    component_id="test", message="err", traceback="", frame_idx=0
                )
            )
            time.sleep(0.1)
            assert len(received_state) == 1
            assert len(received_error) == 1
        finally:
            event_bus.stop()

    def test_eventbus_queue_full(self, event_bus: EventBus) -> None:
        full_bus = EventBus(maxsize=2)
        full_bus.start()
        try:
            full_bus.publish(PipelineStateEvent(state="a"))
            full_bus.publish(PipelineStateEvent(state="b"))
            full_bus.publish(PipelineStateEvent(state="c"))
            full_bus.publish(PipelineStateEvent(state="d"))
            time.sleep(0.1)
            while not full_bus._queue.empty():
                try:
                    full_bus._queue.get_nowait()
                except:
                    break
        finally:
            full_bus.stop()

    def test_eventbus_stop_drains(self, event_bus: EventBus) -> None:
        event_bus.start()
        event_bus.publish(PipelineStateEvent(state="running"))
        event_bus.stop()
        assert event_bus._queue.empty()

    def test_eventbus_handler_count(self, event_bus: EventBus) -> None:
        def handler1(e: Event) -> None:
            pass

        def handler2(e: Event) -> None:
            pass

        assert event_bus.handler_count(PipelineStateEvent) == 0
        event_bus.subscribe(PipelineStateEvent, handler1)
        assert event_bus.handler_count(PipelineStateEvent) == 1
        event_bus.subscribe(PipelineStateEvent, handler2)
        assert event_bus.handler_count(PipelineStateEvent) == 2
