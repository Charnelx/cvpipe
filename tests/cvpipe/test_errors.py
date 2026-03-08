# tests/cvpipe/test_errors.py

from __future__ import annotations

import pytest

from cvpipe.errors import (
    AmbiguousComponentError,
    ComponentError,
    ComponentNotFoundError,
    CoordinateSystemError,
    CvPipeError,
    DuplicateSlotWriterError,
    PipelineConfigError,
    SlotNotFoundError,
)


class TestCvPipeError:
    def test_cvpipe_error_exists(self) -> None:
        err = CvPipeError("test")
        assert isinstance(err, Exception)


class TestPipelineConfigError:
    def test_pipeline_config_error(self) -> None:
        errors = ["error1", "error2"]
        err = PipelineConfigError(errors)
        assert err.errors == errors
        assert "error1" in str(err)
        assert "error2" in str(err)
        assert "2 configuration error" in str(err)

    def test_pipeline_config_error_single(self) -> None:
        err = PipelineConfigError(["single error"])
        assert "1 configuration error" in str(err)

    def test_pipeline_config_error_bullet_points(self) -> None:
        err = PipelineConfigError(["error1", "error2"])
        msg = str(err)
        assert "• error1" in msg
        assert "• error2" in msg


class TestSlotNotFoundError:
    def test_slot_not_found_error(self) -> None:
        err = SlotNotFoundError(component_id="comp1", slot_name="missing_slot")
        assert err.component_id == "comp1"
        assert err.slot_name == "missing_slot"
        assert "comp1" in str(err)
        assert "missing_slot" in str(err)


class TestCoordinateSystemError:
    def test_coordinate_system_error(self) -> None:
        err = CoordinateSystemError(
            slot_name="box",
            upstream="xyxy",
            downstream="xywh",
            upstream_id="producer",
            downstream_id="consumer",
        )
        assert "box" in str(err)
        assert "xyxy" in str(err)
        assert "xywh" in str(err)
        assert "producer" in str(err)
        assert "consumer" in str(err)


class TestDuplicateSlotWriterError:
    def test_duplicate_slot_writer_error(self) -> None:
        err = DuplicateSlotWriterError(
            slot_name="slot_x", writer_a="comp_a", writer_b="comp_b"
        )
        assert err.errors[0] == (
            "Slot 'slot_x' is declared as OUTPUTS by both 'comp_a' and 'comp_b' — "
            "each slot must have exactly one writer"
        )


class TestComponentError:
    def test_component_error(self) -> None:
        original = ValueError("original error")
        err = ComponentError(component_id="comp1", original=original, frame_idx=42)
        assert err.component_id == "comp1"
        assert err.original is original
        assert err.frame_idx == 42
        assert "comp1" in str(err)
        assert "ValueError" in str(err)
        assert "original error" in str(err)
        assert "frame 42" in str(err)


class TestComponentNotFoundError:
    def test_component_not_found_error(self) -> None:
        err = ComponentNotFoundError(module_name="missing_module")
        assert err.module_name == "missing_module"
        assert "missing_module" in str(err)


class TestAmbiguousComponentError:
    def test_ambiguous_component_error(self) -> None:
        err = AmbiguousComponentError(
            module_name="my_module", found=["ClassA", "ClassB"]
        )
        assert err.module_name == "my_module"
        assert err.found == ["ClassA", "ClassB"]
        assert "my_module" in str(err)
        assert "ClassA" in str(err)
        assert "ClassB" in str(err)
