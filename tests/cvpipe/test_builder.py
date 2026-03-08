# tests/cvpipe/test_builder.py

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from cvpipe.builder import build
from cvpipe.pipeline import Pipeline


class TestBuilder:
    def test_build_creates_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            components_dir = Path(tmpdir) / "components"
            components_dir.mkdir()
            (components_dir / "test_source").mkdir()
            (components_dir / "test_source" / "__init__.py").write_text(
                "from cvpipe.scheduler import FrameSource\n"
                "class TestSource(FrameSource):\n"
                "    def setup(self): pass\n"
                "    def teardown(self): pass\n"
                "    def next(self): return None\n"
            )

            (components_dir / "test_comp").mkdir()
            (components_dir / "test_comp" / "__init__.py").write_text(
                "from cvpipe import Component\n"
                "from cvpipe.frame import Frame, SlotSchema\n"
                "class TestComp(Component):\n"
                "    INPUTS = []\n"
                "    OUTPUTS = [SlotSchema('out', int)]\n"
                "    def process(self, frame): pass\n"
            )

            config_file = Path(tmpdir) / "pipeline.yaml"
            config_file.write_text(
                "pipeline:\n"
                "  source: test_source\n"
                "  components:\n"
                "    - module: test_comp\n"
                "      id: comp1\n"
            )

            pipeline = build(config_file, components_dir)
            assert isinstance(pipeline, Pipeline)
            assert len(pipeline._components) == 1
            assert pipeline._source is not None
