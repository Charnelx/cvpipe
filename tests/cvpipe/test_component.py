# tests/cvpipe/test_component.py

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from cvpipe.component import Component
from cvpipe.event import ComponentErrorEvent, EventBus
from cvpipe.frame import Frame, SlotSchema


class TestComponent:
    def test_component_init(self) -> None:
        comp = _ConcreteComponent()

        assert isinstance(comp._lock, type(threading.Lock()))
        assert comp._component_id == ""
        assert comp._event_bus is None

    def test_component_process_abstract(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            Component()

    def test_component_setup_optional(self) -> None:
        comp = _ConcreteComponent()
        comp.setup()

    def test_component_teardown_optional(self) -> None:
        comp = _ConcreteComponent()
        comp.teardown()

    def test_component_reset_optional(self) -> None:
        comp = _ConcreteComponent()
        comp.reset()

    def test_component_on_event_optional(self) -> None:
        comp = _ConcreteComponent()
        event = ComponentErrorEvent(
            component_id="test",
            message="error",
            traceback="",
            frame_idx=0,
        )
        comp.on_event(event)

    def test_component_emit_no_bus(self) -> None:
        comp = _ConcreteComponent()
        event = ComponentErrorEvent(
            component_id="test",
            message="error",
            traceback="",
            frame_idx=0,
        )
        comp.emit(event)

    def test_component_emit_with_bus(self) -> None:
        bus = EventBus()
        bus.start()
        try:
            comp = _ConcreteComponent()
            comp._event_bus = bus

            received: list[ComponentErrorEvent] = []

            def handler(e: ComponentErrorEvent) -> None:
                received.append(e)

            bus.subscribe(ComponentErrorEvent, handler)

            event = ComponentErrorEvent(
                component_id="test",
                message="error",
                traceback="",
                frame_idx=0,
            )
            comp.emit(event)

            import time

            time.sleep(0.05)
            assert len(received) == 1
            assert received[0].message == "error"
        finally:
            bus.stop()

    def test_component_input_slot_names(self) -> None:
        comp = _InputComponent()
        names = comp.input_slot_names()
        assert names == {"input_a", "input_b"}

    def test_component_output_slot_names(self) -> None:
        comp = _OutputComponent()
        names = comp.output_slot_names()
        assert names == {"output_a", "output_b"}

    def test_component_get_input_schema(self) -> None:
        comp = _InputComponent()
        schema = comp.get_input_schema("input_a")
        assert schema is not None
        assert schema.name == "input_a"

    def test_component_get_input_schema_missing(self) -> None:
        comp = _InputComponent()
        schema = comp.get_input_schema("nonexistent")
        assert schema is None

    def test_component_get_output_schema(self) -> None:
        comp = _OutputComponent()
        schema = comp.get_output_schema("output_a")
        assert schema is not None
        assert schema.name == "output_a"

    def test_component_get_output_schema_missing(self) -> None:
        comp = _OutputComponent()
        schema = comp.get_output_schema("nonexistent")
        assert schema is None

    def test_component_repr(self) -> None:
        comp = _ConcreteComponent()
        comp._component_id = "test_id"
        r = repr(comp)
        assert "ConcreteComponent" in r
        assert "test_id" in r

    def test_component_id_property(self) -> None:
        comp = _ConcreteComponent()
        assert comp.component_id == ""
        comp._component_id = "my_id"
        assert comp.component_id == "my_id"


class _ConcreteComponent(Component):
    INPUTS: list[SlotSchema] = []
    OUTPUTS: list[SlotSchema] = []

    def process(self, frame: Frame) -> None:
        pass


class _InputComponent(Component):
    INPUTS: list[SlotSchema] = [
        SlotSchema("input_a", int),
        SlotSchema("input_b", str),
    ]
    OUTPUTS: list[SlotSchema] = []

    def process(self, frame: Frame) -> None:
        pass


class _OutputComponent(Component):
    INPUTS: list[SlotSchema] = []
    OUTPUTS: list[SlotSchema] = [
        SlotSchema("output_a", int),
        SlotSchema("output_b", str),
    ]

    def process(self, frame: Frame) -> None:
        pass
