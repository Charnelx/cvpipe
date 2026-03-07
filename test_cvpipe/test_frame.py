# tests/cvpipe/test_frame.py
import pytest
from cvpipe import Frame, SlotSchema


def test_slot_schema_invalid_name():
    with pytest.raises(ValueError):
        SlotSchema(name="not valid!", dtype=None)


def test_slot_schema_hashable():
    s = SlotSchema("x", None)
    assert s in {s}


def test_compatible_with_coord_mismatch():
    upstream = SlotSchema("boxes", None, (None, 4), "any", coord_system="xyxy")
    downstream = SlotSchema("boxes", None, (None, 4), "any", coord_system="xywh")
    errors = upstream.compatible_with(downstream)
    assert any("coordinate system" in e for e in errors)


def test_compatible_with_shape_rank():
    a = SlotSchema("x", None, (4,), "any")
    b = SlotSchema("x", None, (4, 4), "any")
    errors = a.compatible_with(b)
    assert any("rank" in e for e in errors)


def test_compatible_with_variable_dim():
    upstream = SlotSchema("x", None, (None, 4), "any")
    downstream = SlotSchema("x", None, (10, 4), "any")
    assert upstream.compatible_with(downstream) == []


def test_frame_mutable():
    frame = Frame(idx=0, ts=1.0)
    frame.slots["test"] = 42
    assert frame.slots["test"] == 42


def test_frame_repr():
    frame = Frame(idx=5, ts=1.23)
    frame.slots["a"] = 1
    frame.meta["b"] = 2
    r = repr(frame)
    assert "5" in r and "a" in r and "b" in r


def test_slot_schema_valid_identifier():
    s = SlotSchema("frame_bgr", None)
    assert s.name == "frame_bgr"


def test_slot_schema_is_meta_slot():
    s = SlotSchema("test", str)
    assert s.is_meta_slot() is True


def test_slot_schema_compatible_empty_lists():
    upstream = SlotSchema("x", None)
    downstream = SlotSchema("x", None)
    assert upstream.compatible_with(downstream) == []


def test_compatible_with_device_any_to_gpu():
    upstream = SlotSchema("x", None, (), "any")
    downstream = SlotSchema("x", None, (), "gpu")
    errors = upstream.compatible_with(downstream)
    assert any("device" in e for e in errors)
