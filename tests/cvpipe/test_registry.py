# tests/cvpipe/test_registry.py

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from cvpipe.component import Component
from cvpipe.errors import AmbiguousComponentError, ComponentNotFoundError
from cvpipe.frame import Frame, SlotSchema
from cvpipe.registry import ComponentRegistry
from cvpipe.scheduler import FrameSource


class _TestComponent(Component):
    INPUTS: list[SlotSchema] = []
    OUTPUTS: list[SlotSchema] = []

    def process(self, frame: Frame) -> None:
        pass


class _TestSource(FrameSource):
    def setup(self) -> None:
        pass

    def teardown(self) -> None:
        pass

    def next(self) -> tuple[None, float] | None:
        return None


class TestComponentRegistry:
    def test_registry_init(self) -> None:
        registry = ComponentRegistry()
        assert len(registry) == 0
        assert registry.registered_names == []
        assert registry.registered_source_names == []

    def test_registry_register(self) -> None:
        registry = ComponentRegistry()
        registry.register("test_comp", _TestComponent)
        assert "test_comp" in registry
        assert len(registry) == 1

    def test_registry_register_invalid_type(self) -> None:
        registry = ComponentRegistry()
        with pytest.raises(TypeError, match="Component subclass"):
            registry.register("not_a_component", object)

    def test_registry_register_duplicate(self) -> None:
        registry = ComponentRegistry()
        registry.register("test_comp", _TestComponent)
        with pytest.raises(ValueError, match="already registered"):
            registry.register("test_comp", _TestComponent)

    def test_registry_unregister(self) -> None:
        registry = ComponentRegistry()
        registry.register("test_comp", _TestComponent)
        registry.unregister("test_comp")
        assert "test_comp" not in registry
        assert len(registry) == 0

    def test_registry_unregister_not_found(self) -> None:
        registry = ComponentRegistry()
        registry.unregister("nonexistent")

    def test_registry_get(self) -> None:
        registry = ComponentRegistry()
        registry.register("test_comp", _TestComponent)
        cls = registry.get("test_comp")
        assert cls is _TestComponent

    def test_registry_get_not_found(self) -> None:
        registry = ComponentRegistry()
        with pytest.raises(ComponentNotFoundError) as exc:
            registry.get("nonexistent")
        assert "nonexistent" in str(exc.value)

    def test_registry_register_source(self) -> None:
        registry = ComponentRegistry()
        registry.register_source("test_source", _TestSource)
        assert "test_source" in registry.registered_source_names

    def test_registry_register_source_invalid_type(self) -> None:
        registry = ComponentRegistry()
        with pytest.raises(TypeError, match="FrameSource subclass"):
            registry.register_source("not_a_source", object)

    def test_registry_get_source(self) -> None:
        registry = ComponentRegistry()
        registry.register_source("test_source", _TestSource)
        cls = registry.get_source("test_source")
        assert cls is _TestSource

    def test_registry_get_source_not_found(self) -> None:
        registry = ComponentRegistry()
        with pytest.raises(KeyError) as exc:
            registry.get_source("nonexistent")
        assert "nonexistent" in str(exc.value)

    def test_registry_registered_names(self) -> None:
        registry = ComponentRegistry()
        registry.register("comp_a", _TestComponent)
        registry.register("comp_b", _TestComponent)
        assert registry.registered_names == ["comp_a", "comp_b"]

    def test_registry_contains(self) -> None:
        registry = ComponentRegistry()
        registry.register("test_comp", _TestComponent)
        assert "test_comp" in registry
        assert "nonexistent" not in registry

    def test_registry_len(self) -> None:
        registry = ComponentRegistry()
        assert len(registry) == 0
        registry.register("comp_a", _TestComponent)
        assert len(registry) == 1


class TestComponentRegistryDiscover:
    def test_registry_discover_missing_dir(self) -> None:
        registry = ComponentRegistry()
        with pytest.raises(FileNotFoundError):
            registry.discover(Path("/nonexistent/path"))

    def test_registry_discover_skips_no_init(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir) / "no_init_dir"
            subdir.mkdir()
            (subdir / "file.py").write_text("x = 1")

            registry = ComponentRegistry()
            registry.discover(Path(tmpdir))
            assert "no_init_dir" not in registry.registered_names

    def test_registry_discover_finds_component(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir) / "test_component"
            subdir.mkdir()
            (subdir / "__init__.py").write_text(
                "from cvpipe.component import Component\n"
                "from cvpipe.frame import SlotSchema\n"
                "class TestComp(Component):\n"
                "    INPUTS = []\n"
                "    OUTPUTS = []\n"
                "    def process(self, frame): pass\n"
            )

            registry = ComponentRegistry()
            registry.discover(Path(tmpdir))
            assert "test_component" in registry.registered_names

    def test_registry_discover_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir) / "ambiguous_module"
            subdir.mkdir()
            (subdir / "__init__.py").write_text(
                "from tests.cvpipe.test_registry import _TestComponent\n"
                "from tests.cvpipe.test_registry import _TestSource\n"
                "class Comp1(_TestComponent): pass\n"
                "class Comp2(_TestComponent): pass\n"
            )

            registry = ComponentRegistry()
            with pytest.raises(AmbiguousComponentError) as exc:
                registry.discover(Path(tmpdir))
            assert "ambiguous_module" in str(exc.value)

    def test_registry_discover_finds_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir) / "test_source"
            subdir.mkdir()
            (subdir / "__init__.py").write_text(
                "from tests.cvpipe.test_registry import _TestSource\n"
                "source = _TestSource\n"
            )

            registry = ComponentRegistry()
            registry.discover(Path(tmpdir))
            assert "test_source" in registry.registered_source_names

    def test_registry_discover_skips_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir) / "empty_module"
            subdir.mkdir()
            (subdir / "__init__.py").write_text("x = 1")

            registry = ComponentRegistry()
            registry.discover(Path(tmpdir))
            assert "empty_module" not in registry.registered_names

    def test_registry_discover_duplicate_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir) / "test_comp"
            subdir.mkdir()
            (subdir / "__init__.py").write_text(
                "from tests.cvpipe.test_registry import _TestComponent\n"
                "component = _TestComponent\n"
            )

            registry = ComponentRegistry()
            registry.register("test_comp", _TestComponent)
            registry.discover(Path(tmpdir))
            assert "test_comp" in registry.registered_names
