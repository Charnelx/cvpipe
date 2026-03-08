# cvpipe/builder.py

from pathlib import Path
from .pipeline import Pipeline
from .registry import ComponentRegistry
from .config import PipelineConfig


def build(config_path: Path, components_dir: Path) -> Pipeline:
    registry = ComponentRegistry()
    registry.discover(components_dir)

    config = PipelineConfig.from_yaml(config_path)

    source_cls = registry.get_source(config.source)
    source = source_cls(**config.source_config)

    components = []
    for spec in config.components:
        cls = registry.get(spec.module)
        comp = cls(**spec.config)
        comp._component_id = spec.id
        components.append(comp)

    branch_components = {}
    for branch_spec in config.branches:
        b_comps = []
        for spec in branch_spec.components:
            cls = registry.get(spec.module)
            comp = cls(**spec.config)
            comp._component_id = spec.id
            b_comps.append(comp)
        branch_components[branch_spec.id] = b_comps

    return Pipeline(
        source=source,
        components=components,
        connections=config.connections,
        branches=config.branches,
        branch_components=branch_components,
        validation_mode=config.validation.mode,
    )
