# tests/cvpipe/test_component.py
import pytest
from cvpipe import Component, SlotSchema, Event


def test_component_abstract():
    with pytest.raises(TypeError):
        Component()


def test_component_defaults():
    class C(Component):
        def process(self, frame):
            pass

    c = C()
    assert c.INPUTS == []
    assert c.OUTPUTS == []
    assert c.SUBSCRIBES == []


def test_emit_without_bus():
    class C(Component):
        def process(self, frame):
            pass

    c = C()
    c.emit(Event())


def test_input_slot_names():
    class C(Component):
        INPUTS = [SlotSchema("a", None), SlotSchema("b", None)]

        def process(self, frame):
            pass

    assert C().input_slot_names() == {"a", "b"}


def test_output_slot_names():
    class C(Component):
        OUTPUTS = [SlotSchema("x", None), SlotSchema("y", None)]

        def process(self, frame):
            pass

    assert C().output_slot_names() == {"x", "y"}


def test_get_input_schema():
    class C(Component):
        INPUTS = [SlotSchema("test", str)]

        def process(self, frame):
            pass

    c = C()
    assert c.get_input_schema("test") is not None
    assert c.get_input_schema("missing") is None


def test_get_output_schema():
    class C(Component):
        OUTPUTS = [SlotSchema("out", int)]

        def process(self, frame):
            pass

    c = C()
    assert c.get_output_schema("out") is not None
    assert c.get_output_schema("missing") is None


def test_repr():
    class C(Component):
        def process(self, frame):
            pass

    c = C()
    c._component_id = "test_id"
    assert "test_id" in repr(c)
