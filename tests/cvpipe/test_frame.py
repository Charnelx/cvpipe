# tests/cvpipe/test_frame.py

from __future__ import annotations

import pytest

from cvpipe.frame import Frame, SlotSchema


class TestSlotSchema:
    def test_slot_schema_valid(self) -> None:
        schema = SlotSchema(
            name="test_slot",
            dtype=int,
            shape=(None, 4),
            device="cpu",
            coord_system="xyxy",
            description="Test slot",
        )
        assert schema.name == "test_slot"
        assert schema.dtype == int
        assert schema.shape == (None, 4)
        assert schema.device == "cpu"
        assert schema.coord_system == "xyxy"
        assert schema.description == "Test slot"

    def test_slot_schema_invalid_name(self) -> None:
        with pytest.raises(ValueError, match="must be a valid Python identifier"):
            SlotSchema(name="123-invalid", dtype=int)

    def test_slot_schema_invalid_name_space(self) -> None:
        with pytest.raises(ValueError, match="must be a valid Python identifier"):
            SlotSchema(name="has space", dtype=int)

    def test_slot_schema_default_values(self) -> None:
        schema = SlotSchema(name="default_slot", dtype=str)
        assert schema.shape == ()
        assert schema.device == "any"
        assert schema.coord_system is None
        assert schema.description == ""

    def test_slot_schema_is_tensor_slot_no_torch(self) -> None:
        schema = SlotSchema(name="test", dtype=int)
        assert schema.is_tensor_slot() is False

    def test_slot_schema_is_meta_slot(self) -> None:
        schema = SlotSchema(name="test", dtype=str)
        assert schema.is_meta_slot() is True

    def test_slot_schema_compatible_with_coords_mismatch(self) -> None:
        upstream = SlotSchema(name="box", dtype=int, coord_system="xyxy")
        downstream = SlotSchema(name="box", dtype=int, coord_system="xywh")
        errors = upstream.compatible_with(downstream)
        assert len(errors) == 1
        assert "coordinate system mismatch" in errors[0]

    def test_slot_schema_compatible_with_coords_same(self) -> None:
        upstream = SlotSchema(name="box", dtype=int, coord_system="xyxy")
        downstream = SlotSchema(name="box", dtype=int, coord_system="xyxy")
        errors = upstream.compatible_with(downstream)
        assert errors == []

    def test_slot_schema_compatible_with_coords_one_none(self) -> None:
        upstream = SlotSchema(name="box", dtype=int, coord_system=None)
        downstream = SlotSchema(name="box", dtype=int, coord_system="xyxy")
        errors = upstream.compatible_with(downstream)
        assert errors == []

    def test_slot_schema_compatible_with_shape_rank_mismatch(self) -> None:
        upstream = SlotSchema(name="box", dtype=int, shape=(4,))
        downstream = SlotSchema(name="box", dtype=int, shape=(None, 4))
        errors = upstream.compatible_with(downstream)
        assert len(errors) == 1
        assert "shape rank mismatch" in errors[0]

    def test_slot_schema_compatible_with_shape_dim_mismatch(self) -> None:
        upstream = SlotSchema(name="box", dtype=int, shape=(3,))
        downstream = SlotSchema(name="box", dtype=int, shape=(4,))
        errors = upstream.compatible_with(downstream)
        assert len(errors) == 1
        assert "shape mismatch at dim 0" in errors[0]

    def test_slot_schema_compatible_with_shape_broadcastable_none(self) -> None:
        upstream = SlotSchema(name="box", dtype=int, shape=(None, 4))
        downstream = SlotSchema(name="box", dtype=int, shape=(None, 4))
        errors = upstream.compatible_with(downstream)
        assert errors == []

    def test_slot_schema_compatible_with_shape_broadcastable_fixed(self) -> None:
        upstream = SlotSchema(name="box", dtype=int, shape=(10, 4))
        downstream = SlotSchema(name="box", dtype=int, shape=(None, 4))
        errors = upstream.compatible_with(downstream)
        assert errors == []

    def test_slot_schema_compatible_with_device_mismatch(self) -> None:
        upstream = SlotSchema(name="box", dtype=int, device="cpu")
        downstream = SlotSchema(name="box", dtype=int, device="gpu")
        errors = upstream.compatible_with(downstream)
        assert len(errors) == 1
        assert "device mismatch" in errors[0]

    def test_slot_schema_compatible_with_device_gpu_to_any(self) -> None:
        upstream = SlotSchema(name="box", dtype=int, device="gpu")
        downstream = SlotSchema(name="box", dtype=int, device="any")
        errors = upstream.compatible_with(downstream)
        assert errors == []

    def test_slot_schema_compatible_all_ok(self) -> None:
        upstream = SlotSchema(
            name="box",
            dtype=int,
            shape=(None, 4),
            device="gpu",
            coord_system="xyxy",
        )
        downstream = SlotSchema(
            name="box",
            dtype=int,
            shape=(10, 4),
            device="gpu",
            coord_system="xyxy",
        )
        errors = upstream.compatible_with(downstream)
        assert errors == []


class TestFrame:
    def test_frame_creation(self) -> None:
        frame = Frame(idx=42, ts=1.5)
        assert frame.idx == 42
        assert frame.ts == 1.5
        assert frame.slots == {}
        assert frame.meta == {}

    def test_frame_slots_access(self) -> None:
        frame = Frame(idx=0, ts=0.0)
        frame.slots["key1"] = "value1"
        frame.slots["key2"] = "value2"
        assert frame.slots["key1"] == "value1"
        assert frame.slots["key2"] == "value2"
        assert "key1" in frame.slots

    def test_frame_meta_access(self) -> None:
        frame = Frame(idx=0, ts=0.0)
        frame.meta["key1"] = "value1"
        frame.meta["key2"] = 123
        assert frame.meta["key1"] == "value1"
        assert frame.meta["key2"] == 123

    def test_frame_repr(self) -> None:
        frame = Frame(idx=5, ts=1.234)
        frame.slots["a"] = 1
        frame.meta["b"] = 2
        r = repr(frame)
        assert "Frame" in r
        assert "idx=5" in r
        assert "slots=" in r
        assert "meta=" in r

    def test_frame_slots_initialized_as_dict(self) -> None:
        frame = Frame(idx=0, ts=0.0)
        assert isinstance(frame.slots, dict)
        assert isinstance(frame.meta, dict)
