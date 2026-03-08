# cvpipe/config.py

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from .errors import PipelineConfigError

logger = logging.getLogger(__name__)


@dataclass
class ComponentSpec:
    """
    Parsed representation of one component entry in the YAML.

    Maps to one YAML block:
        - module: frcnn_proposer
          id: proposer
          config:
            confidence: 0.5
    """

    module: str
    id: str
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.module:
            raise ValueError("ComponentSpec.module must be non-empty")
        if not self.id:
            raise ValueError("ComponentSpec.id must be non-empty")
        if not self.id.isidentifier():
            raise ValueError(
                f"ComponentSpec.id must be a valid Python identifier, got: {self.id!r}"
            )


@dataclass
class BranchSpec:
    """
    A conditional branch — a sub-pipeline that runs only when ``trigger``
    evaluates to True against a frame's meta dict.

    Fields
    ------
    id : str
        Unique identifier for this branch.
    components : list[ComponentSpec]
        Components that run when the branch is active.
    trigger : str
        Python expression evaluated against frame.meta.
        Must be safe (no side effects). Example:
            "routing_decision == 'scan'"
        The expression is evaluated with frame.meta as the local namespace.
    inject_after : str
        ID of the main-pipeline component after which this branch begins.
        The branch receives the same Frame state as of that component's output.
    merge_before : str
        ID of the main-pipeline component before which this branch's output
        is merged back. The frame's slots/meta written by the branch
        are available to this component onward.
    exclusive : bool
        If True, this branch is exclusive: when trigger evaluates truthy,
        the main-path components between inject_after and merge_before
        are skipped and only branch components run. If False (default),
        branch components run in addition to the main path.
    """

    id: str
    components: list[ComponentSpec]
    trigger: str
    inject_after: str
    merge_before: str
    exclusive: bool = False


@dataclass
class ValidationConfig:
    """
    Runtime slot validation settings.

    Attributes
    ----------
    mode : Literal["off", "warn", "strict"]
        Validation mode:
        - "off": No validation (production)
        - "warn": Log warnings on mismatch (development, default)
        - "strict": Raise SlotValidationError (testing)
    """

    mode: Literal["off", "warn", "strict"] = "warn"


@dataclass
class PipelineConfig:
    """
    Complete parsed representation of a pipeline YAML file.

    This object is consumed by pipeline_factory to instantiate and wire
    the Pipeline graph. It contains no Component instances — only names
    and raw config dicts.
    """

    source: str
    source_config: dict[str, Any] = field(default_factory=dict)
    components: list[ComponentSpec] = field(default_factory=list)
    connections: dict[str, list[str]] | None = None
    branches: list[BranchSpec] = field(default_factory=list)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "PipelineConfig":
        """
        Load and parse a pipeline YAML file.

        Parameters
        ----------
        path : Path | str
            Path to the YAML file.

        Returns
        -------
        PipelineConfig
            Fully parsed configuration.

        Raises
        ------
        PipelineConfigError
            If the file cannot be read, is not valid YAML, or is missing
            required fields. All errors are collected before raising.
        FileNotFoundError
            If the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Pipeline config not found: {path}")

        errors: list[str] = []

        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            raise PipelineConfigError([f"YAML parse error in {path}: {e}"])

        if not isinstance(raw, dict) or "pipeline" not in raw:
            raise PipelineConfigError(
                [f"YAML file must have a top-level 'pipeline' key: {path}"]
            )

        pipeline_raw = raw["pipeline"]

        # ── source ────────────────────────────────────────────────────────────────
        source = pipeline_raw.get("source")
        if not source:
            errors.append("pipeline.source is required")
        source_config = pipeline_raw.get("source_config", {}) or {}

        # ── components ────────────────────────────────────────────────────────────
        components_raw = pipeline_raw.get("components", [])
        if not isinstance(components_raw, list):
            errors.append("pipeline.components must be a list")
            components_raw = []

        components: list[ComponentSpec] = []
        seen_ids: set[str] = set()
        for i, comp_raw in enumerate(components_raw):
            comp_errors, spec = _parse_component_spec(comp_raw, i, seen_ids)
            errors.extend(comp_errors)
            if spec:
                components.append(spec)
                seen_ids.add(spec.id)

        # ── connections (optional) ───────────────────────────────────────────────
        connections_raw = pipeline_raw.get("connections")
        connections: dict[str, list[str]] | None = None
        if connections_raw is not None:
            if not isinstance(connections_raw, dict):
                errors.append("pipeline.connections must be a mapping")
            else:
                connections = {k: list(v) for k, v in connections_raw.items()}

        # ── branches (optional) ─────────────────────────────────────────────────
        branches_raw = pipeline_raw.get("branches", []) or []
        branches: list[BranchSpec] = []
        for i, branch_raw in enumerate(branches_raw):
            branch_errors, branch = _parse_branch_spec(branch_raw, i, seen_ids)
            errors.extend(branch_errors)
            if branch:
                branches.append(branch)

        # ── validation (optional) ───────────────────────────────────────────────
        validation_raw = pipeline_raw.get("validation", {}) or {}
        valid_modes = {"off", "warn", "strict"}
        validation_mode = validation_raw.get("mode", "warn")
        if validation_mode not in valid_modes:
            errors.append(
                f"pipeline.validation.mode must be one of {valid_modes}, "
                f"got '{validation_mode}'"
            )
            validation_mode = "warn"
        validation = ValidationConfig(mode=validation_mode)

        if errors:
            raise PipelineConfigError(errors)

        return cls(
            source=source,
            source_config=source_config,
            components=components,
            connections=connections,
            branches=branches,
            validation=validation,
        )


def _parse_component_spec(
    raw: Any,
    index: int,
    seen_ids: set[str],
) -> tuple[list[str], ComponentSpec | None]:
    """Returns (errors, spec_or_None)."""
    errors: list[str] = []
    prefix = f"pipeline.components[{index}]"

    if not isinstance(raw, dict):
        return [f"{prefix} must be a mapping, got {type(raw).__name__}"], None

    module = raw.get("module", "").strip()
    id_ = raw.get("id", "").strip()
    config = raw.get("config") or {}

    if not module:
        errors.append(f"{prefix}.module is required")
    if not id_:
        errors.append(f"{prefix}.id is required")
    elif not id_.isidentifier():
        errors.append(f"{prefix}.id must be a valid Python identifier, got {id_!r}")
    elif id_ in seen_ids:
        errors.append(f"{prefix}.id '{id_}' is already used by another component")
    if not isinstance(config, dict):
        errors.append(f"{prefix}.config must be a mapping")
        config = {}

    if errors:
        return errors, None
    return [], ComponentSpec(module=module, id=id_, config=config)


def _parse_branch_spec(
    raw: Any,
    index: int,
    seen_main_ids: set[str],
) -> tuple[list[str], BranchSpec | None]:
    """Returns (errors, spec_or_None)."""
    errors: list[str] = []
    prefix = f"pipeline.branches[{index}]"

    if not isinstance(raw, dict):
        return [f"{prefix} must be a mapping, got {type(raw).__name__}"], None

    id_ = raw.get("id", "").strip()
    trigger = raw.get("trigger", "").strip()
    inject_after = raw.get("inject_after", "").strip()
    merge_before = raw.get("merge_before", "").strip()
    components_raw = raw.get("components", [])
    exclusive = bool(raw.get("exclusive", False))

    if not id_:
        errors.append(f"{prefix}.id is required")
    if not trigger:
        errors.append(f"{prefix}.trigger is required")
    if not inject_after:
        errors.append(f"{prefix}.inject_after is required")
    if not merge_before:
        errors.append(f"{prefix}.merge_before is required")

    if not isinstance(components_raw, list):
        errors.append(f"{prefix}.components must be a list")
        components_raw = []

    # Parse branch components
    branch_components: list[ComponentSpec] = []
    seen_branch_ids: set[str] = set()
    for i, comp_raw in enumerate(components_raw):
        comp_errors, spec = _parse_component_spec(comp_raw, i, seen_branch_ids)
        errors.extend(comp_errors)
        if spec:
            branch_components.append(spec)
            seen_branch_ids.add(spec.id)

    if errors:
        return errors, None
    return [], BranchSpec(
        id=id_,
        components=branch_components,
        trigger=trigger,
        inject_after=inject_after,
        merge_before=merge_before,
        exclusive=exclusive,
    )
