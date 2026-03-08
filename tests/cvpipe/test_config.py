# tests/cvpipe/test_config.py

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from cvpipe.config import BranchSpec, ComponentSpec, PipelineConfig


class TestPipelineConfig:
    def test_pipeline_config_from_yaml_missing(self) -> None:
        with pytest.raises(FileNotFoundError):
            PipelineConfig.from_yaml("/nonexistent/path.yaml")

    def test_pipeline_config_from_yaml_invalid_yaml(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("{ invalid yaml content")
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(Exception):
                PipelineConfig.from_yaml(path)
        finally:
            path.unlink()

    def test_pipeline_config_from_yaml_missing_key(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("not_pipeline:\n  key: value\n")
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(Exception) as exc:
                PipelineConfig.from_yaml(path)
            assert "top-level" in str(exc.value).lower()
        finally:
            path.unlink()

    def test_pipeline_config_source_required(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("pipeline:\n  components: []\n")
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(Exception) as exc:
                PipelineConfig.from_yaml(path)
            assert "source" in str(exc.value).lower()
        finally:
            path.unlink()

    def test_pipeline_config_components_must_be_list(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("pipeline:\n  source: test\n  components: not_a_list\n")
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(Exception) as exc:
                PipelineConfig.from_yaml(path)
            assert "list" in str(exc.value).lower()
        finally:
            path.unlink()

    def test_pipeline_config_component_missing_module(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("pipeline:\n  source: test\n  components:\n    - id: comp1\n")
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(Exception) as exc:
                PipelineConfig.from_yaml(path)
            assert "module" in str(exc.value).lower()
        finally:
            path.unlink()

    def test_pipeline_config_component_missing_id(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("pipeline:\n  source: test\n  components:\n    - module: mymod\n")
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(Exception) as exc:
                PipelineConfig.from_yaml(path)
            assert "id" in str(exc.value).lower()
        finally:
            path.unlink()

    def test_pipeline_config_component_invalid_id(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                "pipeline:\n  source: test\n  components:\n    - module: mymod\n      id: 123-invalid\n"
            )
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(Exception) as exc:
                PipelineConfig.from_yaml(path)
            assert "identifier" in str(exc.value).lower()
        finally:
            path.unlink()

    def test_pipeline_config_duplicate_id(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                "pipeline:\n"
                "  source: test\n"
                "  components:\n"
                "    - module: mod1\n"
                "      id: comp1\n"
                "    - module: mod2\n"
                "      id: comp1\n"
            )
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(Exception) as exc:
                PipelineConfig.from_yaml(path)
            assert "already used" in str(exc.value).lower()
        finally:
            path.unlink()

    def test_pipeline_config_valid(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                "pipeline:\n"
                "  source: camera\n"
                "  source_config:\n"
                "    device: 0\n"
                "  components:\n"
                "    - module: proposer\n"
                "      id: proposer\n"
                "      config:\n"
                "        confidence: 0.5\n"
            )
            f.flush()
            path = Path(f.name)

        try:
            config = PipelineConfig.from_yaml(path)
            assert config.source == "camera"
            assert config.source_config == {"device": 0}
            assert len(config.components) == 1
            assert config.components[0].module == "proposer"
            assert config.components[0].id == "proposer"
            assert config.components[0].config == {"confidence": 0.5}
        finally:
            path.unlink()

    def test_pipeline_config_connections(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                "pipeline:\n"
                "  source: test\n"
                "  connections:\n"
                "    comp_a:\n"
                "      - comp_b\n"
            )
            f.flush()
            path = Path(f.name)

        try:
            config = PipelineConfig.from_yaml(path)
            assert config.connections == {"comp_a": ["comp_b"]}
        finally:
            path.unlink()

    def test_pipeline_config_branches(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                "pipeline:\n"
                "  source: test\n"
                "  branches:\n"
                "    - id: branch1\n"
                "      trigger: routing == 'scan'\n"
                "      inject_after: comp_a\n"
                "      merge_before: comp_b\n"
                "      components:\n"
                "        - module: scanner\n"
                "          id: scanner\n"
            )
            f.flush()
            path = Path(f.name)

        try:
            config = PipelineConfig.from_yaml(path)
            assert len(config.branches) == 1
            assert config.branches[0].id == "branch1"
            assert config.branches[0].trigger == "routing == 'scan'"
            assert config.branches[0].inject_after == "comp_a"
            assert config.branches[0].merge_before == "comp_b"
            assert len(config.branches[0].components) == 1
        finally:
            path.unlink()

    def test_pipeline_config_all_errors_collected(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("pipeline:\n  source: test\n  components:\n    - id: a\n")
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(Exception) as exc:
                PipelineConfig.from_yaml(path)
            errors_str = str(exc.value)
            assert "module" in errors_str
        finally:
            path.unlink()

    def test_pipeline_config_exclusive_branch(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                "pipeline:\n"
                "  source: test\n"
                "  branches:\n"
                "    - id: branch1\n"
                "      trigger: x\n"
                "      inject_after: comp_a\n"
                "      merge_before: comp_b\n"
                "      exclusive: true\n"
                "      components: []\n"
            )
            f.flush()
            path = Path(f.name)

        try:
            config = PipelineConfig.from_yaml(path)
            assert config.branches[0].exclusive is True
        finally:
            path.unlink()


class TestComponentSpec:
    def test_component_spec_validation(self) -> None:
        spec = ComponentSpec(module="test", id="comp1", config={"key": "value"})
        assert spec.module == "test"
        assert spec.id == "comp1"
        assert spec.config == {"key": "value"}

    def test_component_spec_empty_module(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ComponentSpec(module="", id="comp1")

    def test_component_spec_empty_id(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ComponentSpec(module="test", id="")

    def test_component_spec_invalid_id(self) -> None:
        with pytest.raises(ValueError, match="identifier"):
            ComponentSpec(module="test", id="123")

    def test_component_spec_default_config(self) -> None:
        spec = ComponentSpec(module="test", id="comp1")
        assert spec.config == {}


class TestBranchSpec:
    def test_branch_spec_defaults(self) -> None:
        spec = BranchSpec(
            id="branch1",
            components=[],
            trigger="x > 0",
            inject_after="comp_a",
            merge_before="comp_b",
        )
        assert spec.id == "branch1"
        assert spec.exclusive is False

    def test_branch_spec_exclusive_true(self) -> None:
        spec = BranchSpec(
            id="branch1",
            components=[],
            trigger="x > 0",
            inject_after="comp_a",
            merge_before="comp_b",
            exclusive=True,
        )
        assert spec.exclusive is True
