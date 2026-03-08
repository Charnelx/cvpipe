# tests/cvpipe/test_probe.py

from __future__ import annotations

import pytest

from cvpipe.frame import Frame, SlotSchema
from cvpipe.probe import ComponentTrace, DiagnosticsProbe, FrameDiagnostics, Probe


class TestProbe:
    def test_probe_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            Probe()


class TestComponentTrace:
    def test_component_trace_creation(self) -> None:
        trace = ComponentTrace(
            component_id="test_comp",
            latency_ms=1.5,
            output_slots=["slot_a"],
            output_meta=["meta_b"],
            notes="test note",
        )
        assert trace.component_id == "test_comp"
        assert trace.latency_ms == 1.5
        assert trace.output_slots == ["slot_a"]
        assert trace.output_meta == ["meta_b"]
        assert trace.notes == "test note"


class TestFrameDiagnostics:
    def test_frame_diagnostics_creation(self) -> None:
        diag = FrameDiagnostics(
            frame_idx=1,
            ts=1.5,
            components=[],
            total_ms=10.0,
        )
        assert diag.frame_idx == 1
        assert diag.ts == 1.5
        assert diag.components == []
        assert diag.total_ms == 10.0

    def test_frame_diagnostics_summary(self) -> None:
        diag = FrameDiagnostics(
            frame_idx=5,
            ts=1.0,
            components=[
                ComponentTrace("comp_a", 1.0, ["out"], []),
                ComponentTrace("comp_b", 2.0, [], ["meta"]),
            ],
            total_ms=3.0,
        )
        summary = diag.summary()
        assert "Frame 5" in summary
        assert "comp_a:1.0ms" in summary
        assert "comp_b:2.0ms" in summary
        assert "total:3.0ms" in summary


class TestDiagnosticsProbe:
    def test_diagnostics_probe_collects(self) -> None:
        probe = DiagnosticsProbe()
        frame = Frame(idx=0, ts=0.0)
        frame.slots["output_a"] = "tensor"
        frame.meta["meta_b"] = "value"

        probe.observe(frame, "comp_a")
        probe.observe(frame, "comp_b")

        diag = frame.meta.get("diagnostics")
        assert diag is not None
        assert diag.frame_idx == 0
        assert len(diag.components) == 2

    def test_diagnostics_probe_tracks_slots(self) -> None:
        probe = DiagnosticsProbe()
        frame = Frame(idx=1, ts=1.0)

        frame.slots["slot_a"] = "tensor1"
        probe.observe(frame, "comp_a")

        frame.slots["slot_b"] = "tensor2"
        probe.observe(frame, "comp_b")

        diag = frame.meta["diagnostics"]
        assert "slot_a" in diag.components[0].output_slots
        assert "slot_b" in diag.components[1].output_slots

    def test_diagnostics_probe_tracks_meta(self) -> None:
        probe = DiagnosticsProbe()
        frame = Frame(idx=1, ts=1.0)

        frame.meta["key_a"] = "value1"
        probe.observe(frame, "comp_a")

        frame.meta["key_b"] = "value2"
        probe.observe(frame, "comp_b")

        diag = frame.meta["diagnostics"]
        assert "key_a" in diag.components[0].output_meta
        assert "key_b" in diag.components[1].output_meta

    def test_diagnostics_probe_excludes_self_from_meta(self) -> None:
        probe = DiagnosticsProbe()
        frame = Frame(idx=1, ts=1.0)
        frame.meta["diagnostics"] = "existing"
        frame.meta["other"] = "value"

        probe.observe(frame, "comp_a")

        diag = frame.meta["diagnostics"]
        assert "diagnostics" not in diag.components[0].output_meta
        assert "other" in diag.components[0].output_meta


class TestCustomProbe:
    def test_custom_probe_implementation(self) -> None:
        observed: list[tuple[Frame, str]] = []

        class CustomProbe(Probe):
            def observe(self, frame: Frame, after_component: str) -> None:
                observed.append((frame, after_component))

        probe = CustomProbe()
        frame = Frame(idx=0, ts=0.0)
        probe.observe(frame, "test_component")

        assert len(observed) == 1
        assert observed[0][0] is frame
        assert observed[0][1] == "test_component"
