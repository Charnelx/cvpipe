# tests/cvpipe/test_validated_slots.py
"""Unit tests for runtime slot validation."""

from __future__ import annotations

import sys

import pytest

from cvpipe import Component, Frame, SlotSchema
from cvpipe.validated_slots import ValidatedSlots
from cvpipe.errors import SlotValidationError

# Check if torch is available for tensor tests
TORCH_AVAILABLE = "torch" in sys.modules
if not TORCH_AVAILABLE:
    try:
        import torch

        TORCH_AVAILABLE = True
    except ImportError:
        pass


def requires_torch(func):
    """Skip test if torch is not available."""
    return pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")(func)


class TestValidatedSlotsBasics:
    """Basic functionality tests."""

    def test_pass_through_non_declared_slot(self) -> None:
        """Undeclared slots pass through without validation."""
        schemas = {"output_a": SlotSchema("output_a", int, (), "any")}
        data: dict = {}
        vs = ValidatedSlots(data, schemas, "off", "test")

        vs["undeclared"] = 42

        assert "undeclared" in data
        assert data["undeclared"] == 42

    def test_getitem_returns_value(self) -> None:
        """__getitem__ returns the stored value."""
        schemas = {"x": SlotSchema("x", int, (), "any")}
        data: dict = {"x": 42}
        vs = ValidatedSlots(data, schemas, "off", "test")

        assert vs["x"] == 42

    def test_contains_works(self) -> None:
        """__contains__ checks the underlying dict."""
        schemas = {"x": SlotSchema("x", int, (), "any")}
        data: dict = {"x": 42}
        vs = ValidatedSlots(data, schemas, "off", "test")

        assert "x" in vs
        assert "y" not in vs

    def test_get_with_default(self) -> None:
        """get() returns default for missing keys."""
        schemas = {"x": SlotSchema("x", int, (), "any")}
        data: dict = {}
        vs = ValidatedSlots(data, schemas, "off", "test")

        assert vs.get("x", 99) == 99


class TestTensorValidation:
    """Tensor slot validation tests."""

    @requires_torch
    def test_valid_tensor_passes_strict(self) -> None:
        """Valid tensor passes strict mode validation."""
        import torch

        schemas = {"boxes": SlotSchema("boxes", torch.float32, (None, 4), "cpu")}
        data: dict = {}
        vs = ValidatedSlots(data, schemas, "strict", "test")

        vs["boxes"] = torch.randn(10, 4, dtype=torch.float32)

        assert data["boxes"].shape == (10, 4)

    @requires_torch
    def test_dtype_mismatch_strict_raises(self) -> None:
        """dtype mismatch raises in strict mode."""
        import torch

        schemas = {"boxes": SlotSchema("boxes", torch.float32, (None, 4), "cpu")}
        data: dict = {}
        vs = ValidatedSlots(data, schemas, "strict", "test")

        with pytest.raises(SlotValidationError) as exc:
            vs["boxes"] = torch.randn(10, 4, dtype=torch.float32).to(torch.int64)

        assert "dtype" in str(exc.value)
        assert exc.value.component_id == "test"
        assert exc.value.slot_name == "boxes"

    @requires_torch
    def test_shape_rank_mismatch_strict_raises(self) -> None:
        """Shape rank mismatch raises in strict mode."""
        import torch

        schemas = {"boxes": SlotSchema("boxes", torch.float32, (None, 4), "cpu")}
        data: dict = {}
        vs = ValidatedSlots(data, schemas, "strict", "test")

        with pytest.raises(SlotValidationError) as exc:
            vs["boxes"] = torch.randn(10, 4, 5, dtype=torch.float32)

        assert "shape rank" in str(exc.value)

    @requires_torch
    def test_shape_dim_mismatch_strict_raises(self) -> None:
        """Shape dimension mismatch raises in strict mode."""
        import torch

        schemas = {"boxes": SlotSchema("boxes", torch.float32, (None, 4), "cpu")}
        data: dict = {}
        vs = ValidatedSlots(data, schemas, "strict", "test")

        with pytest.raises(SlotValidationError) as exc:
            vs["boxes"] = torch.randn(10, 3, dtype=torch.float32)

        assert "shape[1]" in str(exc.value)

    @requires_torch
    def test_device_mismatch_gpu_expected_strict_raises(self) -> None:
        """Device mismatch (expected GPU, got CPU) raises in strict mode."""
        import torch

        schemas = {"boxes": SlotSchema("boxes", torch.float32, (None, 4), "gpu")}
        data: dict = {}
        vs = ValidatedSlots(data, schemas, "strict", "test")

        with pytest.raises(SlotValidationError) as exc:
            vs["boxes"] = torch.randn(10, 4, dtype=torch.float32)

        assert "device" in str(exc.value)
        assert "CPU" in str(exc.value)

    @requires_torch
    def test_non_tensor_for_tensor_slot_raises(self) -> None:
        """Writing non-tensor to tensor slot raises."""
        import torch

        schemas = {"boxes": SlotSchema("boxes", torch.float32, (None, 4), "cpu")}
        data: dict = {}
        vs = ValidatedSlots(data, schemas, "strict", "test")

        with pytest.raises(SlotValidationError) as exc:
            vs["boxes"] = [[1, 2, 3, 4]]

        assert "torch.Tensor" in str(exc.value)

    @requires_torch
    def test_variable_length_shape_passes(self) -> None:
        """None in shape allows variable length."""
        import torch

        schemas = {"boxes": SlotSchema("boxes", torch.float32, (None, 4), "cpu")}
        data: dict = {}
        vs = ValidatedSlots(data, schemas, "strict", "test")

        vs["boxes"] = torch.randn(1, 4, dtype=torch.float32)
        assert data["boxes"].shape == (1, 4)

        data.clear()
        vs["boxes"] = torch.randn(100, 4, dtype=torch.float32)
        assert data["boxes"].shape == (100, 4)


class TestMetaValidation:
    """Meta slot (non-tensor) validation tests."""

    def test_valid_meta_int_passes_strict(self) -> None:
        """Valid int passes strict mode."""
        schemas = {"count": SlotSchema("count", int, (), "any")}
        data: dict = {}
        vs = ValidatedSlots(data, schemas, "strict", "test")

        vs["count"] = 42

        assert data["count"] == 42

    def test_valid_meta_str_passes_strict(self) -> None:
        """Valid str passes strict mode."""
        schemas = {"label": SlotSchema("label", str, (), "any")}
        data: dict = {}
        vs = ValidatedSlots(data, schemas, "strict", "test")

        vs["label"] = "car"

        assert data["label"] == "car"

    def test_valid_meta_dict_passes_strict(self) -> None:
        """Valid dict passes strict mode."""
        schemas = {"meta": SlotSchema("meta", dict, (), "any")}
        data: dict = {}
        vs = ValidatedSlots(data, schemas, "strict", "test")

        vs["meta"] = {"key": "value"}

        assert data["meta"] == {"key": "value"}

    def test_type_mismatch_strict_raises(self) -> None:
        """Type mismatch raises in strict mode."""
        schemas = {"count": SlotSchema("count", int, (), "any")}
        data: dict = {}
        vs = ValidatedSlots(data, schemas, "strict", "test")

        with pytest.raises(SlotValidationError) as exc:
            vs["count"] = "not an int"

        assert "type" in str(exc.value)

    def test_none_dtype_allows_anything(self) -> None:
        """None dtype allows any type."""
        schemas = {"data": SlotSchema("data", None, (), "any")}
        data: dict = {}
        vs = ValidatedSlots(data, schemas, "strict", "test")

        vs["data"] = 42
        assert data["data"] == 42

        vs["data"] = "string"
        assert data["data"] == "string"

        vs["data"] = [1, 2, 3]
        assert data["data"] == [1, 2, 3]


class TestWarnMode:
    """Warn mode tests."""

    @requires_torch
    def test_warn_mode_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Warn mode logs warning instead of raising."""
        import torch

        schemas = {"boxes": SlotSchema("boxes", torch.float32, (None, 4), "cpu")}
        data: dict = {}
        vs = ValidatedSlots(data, schemas, "warn", "test_component")

        with caplog.at_level("WARNING"):
            vs["boxes"] = torch.randn(10, 3, dtype=torch.float32).to(torch.int64)

        assert any("test_component" in r.message for r in caplog.records)
        assert any("Slot 'boxes'" in r.message for r in caplog.records)

    @requires_torch
    def test_warn_mode_only_logs_once_per_slot(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warn mode logs only once per slot name."""
        import torch

        schemas = {"boxes": SlotSchema("boxes", torch.float32, (None, 4), "cpu")}
        data: dict = {}
        vs = ValidatedSlots(data, schemas, "warn", "test")

        with caplog.at_level("WARNING"):
            vs["boxes"] = torch.randn(10, 3, dtype=torch.float32).to(torch.int64)
            vs["boxes"] = torch.randn(10, 2, dtype=torch.float32)

        warnings = [r for r in caplog.records if "Slot 'boxes'" in r.message]
        assert len(warnings) == 1

    @requires_torch
    def test_warn_mode_different_slots_both_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Different slots each get one warning."""
        import torch

        schemas = {
            "boxes": SlotSchema("boxes", torch.float32, (None, 4), "cpu"),
            "scores": SlotSchema("scores", torch.float32, (None,), "cpu"),
        }
        data: dict = {}
        vs = ValidatedSlots(data, schemas, "warn", "test")

        with caplog.at_level("WARNING"):
            vs["boxes"] = torch.randn(10, 3, dtype=torch.float32).to(torch.int64)
            vs["scores"] = torch.randn(10, dtype=torch.float32).to(torch.int64)

        warnings = [r for r in caplog.records if "Slot" in r.message]
        assert len(warnings) == 2


class TestOffMode:
    """Off mode (no validation) tests."""

    @requires_torch
    def test_off_mode_no_validation(self) -> None:
        """Off mode skips all validation."""
        import torch

        schemas = {"boxes": SlotSchema("boxes", torch.float32, (None, 4), "cpu")}
        data: dict = {}
        vs = ValidatedSlots(data, schemas, "off", "test")

        vs["boxes"] = torch.randn(10, 3, dtype=torch.float32).to(torch.int64)

        assert data["boxes"].shape == (10, 3)
        assert data["boxes"].dtype == torch.int64


class TestFrameValidation:
    """Frame validation context tests."""

    def test_frame_slots_without_context(self) -> None:
        """Without validation context, slots is raw dict."""
        frame = Frame(idx=0, ts=0.0)

        assert frame.slots is frame._slots
        assert isinstance(frame.slots, dict)

    @requires_torch
    def test_frame_slots_with_context(self) -> None:
        """With validation context, slots returns ValidatedSlots."""
        import torch

        from tests.cvpipe.conftest import SlotWriterComponent

        frame = Frame(idx=0, ts=0.0)
        comp = SlotWriterComponent("test_slot", torch.randn(10, 4))
        comp._component_id = "writer"

        frame._set_validation(comp, "warn")

        assert hasattr(frame.slots, "_schemas")

    @requires_torch
    def test_frame_clear_validation(self) -> None:
        """Clearing validation context returns raw dict."""
        import torch

        from tests.cvpipe.conftest import SlotWriterComponent

        frame = Frame(idx=0, ts=0.0)
        comp = SlotWriterComponent("test_slot", torch.randn(10, 4))
        comp._component_id = "writer"

        frame._set_validation(comp, "warn")
        assert hasattr(frame.slots, "_schemas")

        frame._clear_validation()
        assert frame.slots is frame._slots

    @requires_torch
    def test_frame_meta_always_dict(self) -> None:
        """meta property always returns raw dict."""
        import torch

        from tests.cvpipe.conftest import SlotWriterComponent

        frame = Frame(idx=0, ts=0.0)
        comp = SlotWriterComponent("test_slot", torch.randn(10, 4))
        comp._component_id = "writer"

        frame._set_validation(comp, "warn")

        assert frame.meta is frame._meta
        assert isinstance(frame.meta, dict)


class TestMultipleErrors:
    """Tests for multiple validation errors in one write."""

    @requires_torch
    def test_multiple_errors_reported(self) -> None:
        """Multiple errors are all reported."""
        import torch

        schemas = {"boxes": SlotSchema("boxes", torch.float32, (None, 4), "gpu")}
        data: dict = {}
        vs = ValidatedSlots(data, schemas, "strict", "test")

        with pytest.raises(SlotValidationError) as exc:
            vs["boxes"] = torch.randn(10, 3, dtype=torch.float32).to(torch.int64)

        assert len(exc.value.errors) >= 2
        error_str = str(exc.value)
        assert "dtype" in error_str
        assert "shape" in error_str
