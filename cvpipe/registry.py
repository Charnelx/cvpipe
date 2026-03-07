# cvpipe/registry.py

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import sys
from pathlib import Path

from .component import Component
from .scheduler import FrameSource
from .errors import ComponentNotFoundError, AmbiguousComponentError

logger = logging.getLogger(__name__)


class ComponentRegistry:
    """
    Maps module names (strings from YAML) to Component and FrameSource classes.

    Two registration modes:
    1. Explicit: registry.register("my_component", MyComponent)
    2. Auto-discovery: registry.discover(Path("detector/components/"))

    Auto-discovery imports each subdirectory of the given path as a Python
    module and finds the single Component or FrameSource subclass exported
    from its __init__.py. The directory name becomes the registry key.

    Sources (FrameSource subclasses) are registered separately from components
    and retrieved via get_source(). This mirrors the YAML ``source:`` field,
    which references a source by its directory name just like components.

    Usage::

        registry = ComponentRegistry()
        registry.discover(Path("detector/components/"))
        registry.discover(Path("detector/sources/"))

        # From YAML: module: frcnn_proposer
        cls = registry.get("frcnn_proposer")
        instance = cls(**config_dict)

        # From YAML: source: rtsp_source
        source_cls = registry.get_source("rtsp_source")
        source = source_cls(**source_config)
    """

    def __init__(self) -> None:
        self._registry: dict[str, type[Component]] = {}
        self._sources: dict[str, type[FrameSource]] = {}

    def register(self, name: str, cls: type[Component]) -> None:
        """
        Explicitly register a Component class under ``name``.

        Parameters
        ----------
        name : str
            The key used in YAML ``module:`` fields.
        cls : type[Component]
            Must be a concrete subclass of Component (not Component itself).

        Raises
        ------
        TypeError
            If cls is not a Component subclass.
        ValueError
            If name is already registered (use force=True to override).
        """
        if not (isinstance(cls, type) and issubclass(cls, Component) and cls is not Component):
            raise TypeError(f"Expected a concrete Component subclass, got {cls!r}")
        if name in self._registry:
            raise ValueError(
                f"Component name '{name}' is already registered as "
                f"{self._registry[name].__name__}. "
                f"Use registry.unregister('{name}') first if intentional."
            )
        self._registry[name] = cls
        logger.debug("[ComponentRegistry] Registered '%s' → %s", name, cls.__name__)

    def unregister(self, name: str) -> None:
        """Remove a registered component. No-op if not registered."""
        self._registry.pop(name, None)

    def get(self, name: str) -> type[Component]:
        """
        Retrieve a Component class by name.

        Raises
        ------
        ComponentNotFoundError
            If name is not in the registry.
        """
        if name not in self._registry:
            raise ComponentNotFoundError(name)
        return self._registry[name]

    def register_source(self, name: str, cls: type[FrameSource]) -> None:
        """
        Explicitly register a FrameSource class under ``name``.

        Parameters
        ----------
        name : str
            The key used in YAML ``source:`` fields.
        cls : type[FrameSource]
            Must be a concrete subclass of FrameSource (not FrameSource itself).

        Raises
        ------
        TypeError
            If cls is not a FrameSource subclass.
        ValueError
            If name is already registered.
        """
        if not (isinstance(cls, type) and issubclass(cls, FrameSource) and cls is not FrameSource):
            raise TypeError(f"Expected a concrete FrameSource subclass, got {cls!r}")
        if name in self._sources:
            raise ValueError(
                f"Source name '{name}' is already registered as "
                f"{self._sources[name].__name__}. "
                f"Use registry.unregister_source('{name}') first if intentional."
            )
        self._sources[name] = cls
        logger.debug("[ComponentRegistry] Registered source '%s' → %s", name, cls.__name__)

    def unregister_source(self, name: str) -> None:
        """Remove a registered source. No-op if not registered."""
        self._sources.pop(name, None)

    def get_source(self, name: str) -> type[FrameSource]:
        """
        Retrieve a FrameSource class by name.

        Raises
        ------
        KeyError
            If name is not in the source registry.
        """
        if name not in self._sources:
            available = list(self._sources.keys())
            raise KeyError(
                f"No source named {name!r} found. "
                f"Available: {available}. "
                f"Is the source module in the discovered directory?"
            )
        return self._sources[name]

    def discover(self, components_dir: Path) -> None:
        """
        Auto-discover and register all Component and FrameSource subclasses
        found in ``components_dir``.

        Algorithm:
        1. Iterate all immediate subdirectories of components_dir
        2. Skip directories that do not have an __init__.py
        3. Import the directory as a Python module using its directory name
           as both the import name and the registry key
        4. Find all Component or FrameSource subclasses defined in that
           module's namespace (not abstract, not the base classes themselves)
        5. If exactly one → register it (Component → component registry,
           FrameSource → source registry)
        6. If zero → log warning, skip
        7. If more than one → raise AmbiguousComponentError

        May be called multiple times with different paths, e.g. once for
        components and once for sources.

        Parameters
        ----------
        components_dir : Path
            Absolute or relative path to the directory containing component
            or source module subdirectories.

        Raises
        ------
        AmbiguousComponentError
            If any module exports more than one Component subclass.
        FileNotFoundError
            If components_dir does not exist.
        """
        components_dir = Path(components_dir).resolve()
        if not components_dir.exists():
            raise FileNotFoundError(f"Components directory not found: {components_dir}")

        for subdir in sorted(components_dir.iterdir()):
            if not subdir.is_dir():
                continue
            init_file = subdir / "__init__.py"
            if not init_file.exists():
                logger.debug("[ComponentRegistry] Skipping %s — no __init__.py", subdir.name)
                continue

            module_name = subdir.name
            try:
                component_cls = self._import_component(subdir, module_name)
            except AmbiguousComponentError:
                raise
            except Exception as e:
                logger.warning(
                    "[ComponentRegistry] Failed to import '%s': %s — skipping",
                    module_name,
                    e,
                )
                continue

            if component_cls is None:
                logger.warning(
                    "[ComponentRegistry] No Component subclass found in '%s' — skipping",
                    module_name,
                )
                continue

            try:
                if isinstance(component_cls, type) and issubclass(component_cls, FrameSource):
                    self.register_source(module_name, component_cls)
                else:
                    self.register(module_name, component_cls)
            except (ValueError, KeyError):
                logger.warning(
                    "[ComponentRegistry] '%s' already registered — skipping duplicate",
                    module_name,
                )

    def _import_component(self, module_dir: Path, module_name: str) -> type[Component] | type[FrameSource] | None:
        """
        Import module_dir/__init__.py and return the single Component or
        FrameSource subclass. Uses importlib to load by file path, avoiding
        sys.path manipulation.
        """
        qualified_name = f"_cvpipe_components.{module_name}"

        if qualified_name in sys.modules:
            module = sys.modules[qualified_name]
        else:
            spec = importlib.util.spec_from_file_location(
                qualified_name,
                module_dir / "__init__.py",
                submodule_search_locations=[str(module_dir)],
            )
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            sys.modules[qualified_name] = module
            spec.loader.exec_module(module)

        found: list[type[Component] | type[FrameSource]] = [
            obj
            for obj in vars(module).values()
            if (
                isinstance(obj, type)
                and (
                    (issubclass(obj, Component) and obj is not Component)
                    or (issubclass(obj, FrameSource) and obj is not FrameSource)
                )
                and not inspect.isabstract(obj)
            )
        ]

        if len(found) > 1:
            raise AmbiguousComponentError(module_name, [c.__name__ for c in found])

        return found[0] if found else None

    @property
    def registered_names(self) -> list[str]:
        """Sorted list of all registered component names."""
        return sorted(self._registry.keys())

    @property
    def registered_source_names(self) -> list[str]:
        """Sorted list of all registered source names."""
        return sorted(self._sources.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)
