# cvpipe/errors.py

from __future__ import annotations


class CvPipeError(Exception):
    """Base class for all cvpipe framework exceptions."""


# ── Configuration and assembly errors ─────────────────────────────────────────


class PipelineConfigError(CvPipeError):
    """
    Raised when pipeline.validate() finds one or more errors.

    Contains a list of all errors found, not just the first.
    Always reports all errors to avoid frustrating "fix one error,
    discover another" iteration cycles.

    Attributes
    ----------
    errors : list[str]
        Human-readable error descriptions.

    Example::

        try:
            pipeline.validate()
        except PipelineConfigError as e:
            for err in e.errors:
                print(f"  • {err}")
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        bullet_list = "\n".join(f"  • {e}" for e in errors)
        super().__init__(f"Pipeline has {len(errors)} configuration error(s):\n{bullet_list}")


class ContractError(PipelineConfigError):
    """
    Raised when slot contracts between components are incompatible.

    Subclass of PipelineConfigError — always caught by the same handler.
    """


class SlotNotFoundError(ContractError):
    """
    A component declares an input slot that no upstream component produces.

    Attributes
    ----------
    component_id : str
        The component that declared the unsatisfied input.
    slot_name : str
        The slot name that could not be resolved.
    """

    def __init__(self, component_id: str, slot_name: str) -> None:
        self.component_id = component_id
        self.slot_name = slot_name
        super().__init__(
            [
                f"Component '{component_id}' requires slot '{slot_name}' "
                f"but no upstream component produces it"
            ]
        )


class CoordinateSystemError(ContractError):
    """
    An upstream component produces a slot with an incompatible
    coordinate system to what the downstream component expects.
    """

    def __init__(
        self,
        slot_name: str,
        upstream: str,
        downstream: str,
        upstream_id: str,
        downstream_id: str,
    ) -> None:
        super().__init__(
            [
                f"Slot '{slot_name}': coordinate system mismatch between "
                f"'{upstream_id}' (produces '{upstream}') and "
                f"'{downstream_id}' (expects '{downstream}')"
            ]
        )


class DuplicateSlotWriterError(ContractError):
    """Two components declare the same output slot."""

    def __init__(self, slot_name: str, writer_a: str, writer_b: str) -> None:
        super().__init__(
            [
                f"Slot '{slot_name}' is declared as OUTPUTS by both "
                f"'{writer_a}' and '{writer_b}' — each slot must have exactly one writer"
            ]
        )


# ── Runtime errors ─────────────────────────────────────────────────────────────


class ComponentError(CvPipeError):
    """
    Wraps an exception raised inside Component.process().

    The Scheduler catches all exceptions from process(), wraps them in
    ComponentError, emits a ComponentErrorEvent, and continues the loop.
    ComponentError is never propagated to the application — use
    ComponentErrorEvent subscriptions to observe runtime component failures.

    Attributes
    ----------
    component_id : str
    original : Exception
        The original exception raised by process().
    frame_idx : int
    """

    def __init__(self, component_id: str, original: Exception, frame_idx: int) -> None:
        self.component_id = component_id
        self.original = original
        self.frame_idx = frame_idx
        super().__init__(
            f"Component '{component_id}' raised {type(original).__name__} "
            f"on frame {frame_idx}: {original}"
        )


# ── Registry errors ────────────────────────────────────────────────────────────


class ComponentNotFoundError(CvPipeError):
    """
    A module name specified in the YAML config could not be resolved
    to a Component class by ComponentRegistry.
    """

    def __init__(self, module_name: str) -> None:
        self.module_name = module_name
        super().__init__(
            f"No component found for module name '{module_name}'. "
            f"Ensure the module directory exists in the components/ folder "
            f"and its __init__.py exports exactly one Component subclass."
        )


class AmbiguousComponentError(CvPipeError):
    """
    A module exports more than one Component subclass.
    Each module must export exactly one.
    """

    def __init__(self, module_name: str, found: list[str]) -> None:
        self.module_name = module_name
        self.found = found
        super().__init__(
            f"Module '{module_name}' exports {len(found)} Component subclasses: "
            f"{found}. Each module must export exactly one Component subclass."
        )
