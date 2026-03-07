# tests/cvpipe/test_registry.py
import pytest
from cvpipe import Component, ComponentRegistry
from cvpipe.errors import ComponentNotFoundError, AmbiguousComponentError


def test_register_and_get():
    reg = ComponentRegistry()

    class MyComp(Component):
        def process(self, frame):
            pass

    reg.register("my_comp", MyComp)
    assert reg.get("my_comp") is MyComp


def test_get_not_found():
    reg = ComponentRegistry()
    with pytest.raises(ComponentNotFoundError):
        reg.get("nonexistent")


def test_register_not_component():
    reg = ComponentRegistry()
    with pytest.raises(TypeError):
        reg.register("not_comp", object)


def test_register_duplicate():
    reg = ComponentRegistry()

    class MyComp(Component):
        def process(self, frame):
            pass

    reg.register("dup", MyComp)
    with pytest.raises(ValueError):
        reg.register("dup", MyComp)


def test_unregister():
    reg = ComponentRegistry()

    class MyComp(Component):
        def process(self, frame):
            pass

    reg.register("test", MyComp)
    reg.unregister("test")
    assert "test" not in reg


def test_unregister_missing():
    reg = ComponentRegistry()
    reg.unregister("missing")


def test_contains():
    reg = ComponentRegistry()

    class MyComp(Component):
        def process(self, frame):
            pass

    reg.register("foo", MyComp)
    assert "foo" in reg
    assert "bar" not in reg


def test_len():
    reg = ComponentRegistry()

    class MyComp(Component):
        def process(self, frame):
            pass

    assert len(reg) == 0
    reg.register("a", MyComp)
    reg.register("b", MyComp)
    assert len(reg) == 2


def test_registered_names():
    reg = ComponentRegistry()

    class MyComp(Component):
        def process(self, frame):
            pass

    reg.register("z_comp", MyComp)
    reg.register("a_comp", MyComp)
    assert reg.registered_names == ["a_comp", "z_comp"]


def test_discover(tmp_path):
    comp_dir = tmp_path / "components" / "test_comp"
    comp_dir.mkdir(parents=True)
    (comp_dir / "__init__.py").write_text(
        """
from cvpipe import Component, Frame
class TestComp(Component):
    def process(self, frame: Frame) -> None:
        pass
"""
    )
    reg = ComponentRegistry()
    reg.discover(tmp_path / "components")
    assert "test_comp" in reg
    cls = reg.get("test_comp")
    assert issubclass(cls, Component)


def test_discover_ambiguous(tmp_path):
    comp_dir = tmp_path / "components" / "multi"
    comp_dir.mkdir(parents=True)
    (comp_dir / "__init__.py").write_text(
        """
from cvpipe import Component, Frame
class A(Component):
    def process(self, frame): pass
class B(Component):
    def process(self, frame): pass
"""
    )
    reg = ComponentRegistry()
    with pytest.raises(AmbiguousComponentError):
        reg.discover(tmp_path / "components")


def test_discover_no_init(tmp_path):
    comp_dir = tmp_path / "components" / "no_init"
    comp_dir.mkdir(parents=True)
    reg = ComponentRegistry()
    reg.discover(tmp_path / "components")
    assert "no_init" not in reg
