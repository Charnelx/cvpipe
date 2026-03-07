# tests/cvpipe/test_config.py
import pytest
from cvpipe.config import PipelineConfig
from cvpipe.errors import PipelineConfigError


def test_valid_yaml(tmp_path):
    yaml_content = """
pipeline:
  source: camera_source
  components:
    - module: frcnn_proposer
      id: proposer
      config:
        confidence: 0.5
"""
    path = tmp_path / "test.yaml"
    path.write_text(yaml_content)
    cfg = PipelineConfig.from_yaml(path)
    assert cfg.source == "camera_source"
    assert len(cfg.components) == 1
    assert cfg.components[0].id == "proposer"
    assert cfg.components[0].config == {"confidence": 0.5}


def test_missing_source(tmp_path):
    yaml_content = "pipeline:\n  components: []\n"
    path = tmp_path / "test.yaml"
    path.write_text(yaml_content)
    with pytest.raises(PipelineConfigError) as exc_info:
        PipelineConfig.from_yaml(path)
    assert "source" in str(exc_info.value)


def test_duplicate_ids_collected(tmp_path):
    yaml_content = """
pipeline:
  source: src
  components:
    - module: a
      id: same
    - module: b
      id: same
"""
    path = tmp_path / "test.yaml"
    path.write_text(yaml_content)
    with pytest.raises(PipelineConfigError) as exc_info:
        PipelineConfig.from_yaml(path)
    assert len(exc_info.value.errors) >= 1


def test_missing_module(tmp_path):
    yaml_content = """
pipeline:
  source: src
  components:
    - id: no_module
"""
    path = tmp_path / "test.yaml"
    path.write_text(yaml_content)
    with pytest.raises(PipelineConfigError) as exc_info:
        PipelineConfig.from_yaml(path)
    assert "module" in str(exc_info.value)


def test_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        PipelineConfig.from_yaml("/nonexistent/path.yaml")


def test_invalid_yaml_syntax(tmp_path):
    yaml_content = "pipeline:\n  source: test\n  components:\n    - {invalid"
    path = tmp_path / "test.yaml"
    path.write_text(yaml_content)
    with pytest.raises(PipelineConfigError):
        PipelineConfig.from_yaml(path)


def test_connections_optional(tmp_path):
    yaml_content = """
pipeline:
  source: src
  components: []
"""
    path = tmp_path / "test.yaml"
    path.write_text(yaml_content)
    cfg = PipelineConfig.from_yaml(path)
    assert cfg.connections is None


def test_branches_optional(tmp_path):
    yaml_content = """
pipeline:
  source: src
  components: []
"""
    path = tmp_path / "test.yaml"
    path.write_text(yaml_content)
    cfg = PipelineConfig.from_yaml(path)
    assert cfg.branches == []
