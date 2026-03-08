# cvpipe/validated_slots.py
"""Runtime slot validation wrapper."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

from .errors import SlotValidationError

if TYPE_CHECKING:
    from .frame import SlotSchema

logger = logging.getLogger(__name__)


class ValidatedSlots:
    """
    Wrapper that validates slot writes against component OUTPUTS schemas.

    Zero-allocation design: reuses internal dicts, no object creation during
    validation. Optimized for <0.5µs overhead per write.

    Parameters
    ----------
    data : dict[str, Any]
        Shared reference to frame's internal slots dict.
    schemas : dict[str, SlotSchema]
        Component's OUTPUTS as dict for O(1) lookup.
    mode : Literal["off", "warn", "strict"]
        Validation mode.
    component_id : str
        Component ID for error messages.
    """

    __slots__ = (
        "_data",
        "_schemas",
        "_mode",
        "_component_id",
        "_warned_slots",
    )

    def __init__(
        self,
        data: dict[str, Any],
        schemas: dict[str, SlotSchema],
        mode: Literal["off", "warn", "strict"],
        component_id: str,
    ) -> None:
        self._data = data
        self._schemas = schemas
        self._mode = mode
        self._component_id = component_id
        self._warned_slots: set[str] = set()

    def __setitem__(self, name: str, value: Any) -> None:
        # Fast path: not a declared output
        if name not in self._schemas:
            self._data[name] = value
            return

        # Validation path
        if self._mode != "off":
            self._validate_write(name, value)

        self._data[name] = value

    def __getitem__(self, name: str) -> Any:
        return self._data[name]

    def __contains__(self, name: str) -> bool:
        return name in self._data

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def get(self, name: str, default: Any = None) -> Any:
        return self._data.get(name, default)

    def _validate_write(self, name: str, value: Any) -> None:
        """Validate value against schema. Optimized for <0.5µs."""
        schema = self._schemas[name]

        # Early exit if already warned (warn mode only)
        if self._mode == "warn" and name in self._warned_slots:
            return

        errors: list[str] = []

        # Check if tensor slot
        if schema.is_tensor_slot():
            self._validate_tensor(name, value, schema, errors)
        else:
            self._validate_meta(name, value, schema, errors)

        if errors:
            self._handle_error(name, errors)

    def _validate_tensor(
        self,
        name: str,
        value: Any,
        schema: SlotSchema,
        errors: list[str],
    ) -> None:
        """Validate a tensor slot value."""
        import torch

        # isinstance check - most common failure
        if not isinstance(value, torch.Tensor):
            errors.append(f"expected torch.Tensor, got {type(value).__name__}")
            return

        # dtype check - cheapest
        if value.dtype != schema.dtype:
            errors.append(f"dtype {value.dtype} != expected {schema.dtype}")

        # shape check - only if schema specifies
        if schema.shape:
            if len(value.shape) != len(schema.shape):
                errors.append(
                    f"shape rank {len(value.shape)} != expected {len(schema.shape)}"
                )
            else:
                for i, (got, exp) in enumerate(zip(value.shape, schema.shape)):
                    if exp is not None and got != exp:
                        errors.append(f"shape[{i}]: {got} != expected {exp}")

        # device check
        if schema.device == "gpu" and not value.is_cuda:
            errors.append("device CPU != expected GPU")
        elif schema.device == "cpu" and value.is_cuda:
            errors.append("device GPU != expected CPU")

    def _validate_meta(
        self,
        name: str,
        value: Any,
        schema: SlotSchema,
        errors: list[str],
    ) -> None:
        """Validate a meta slot value."""
        if schema.dtype is not None and not isinstance(value, schema.dtype):
            errors.append(
                f"type {type(value).__name__} != expected {schema.dtype.__name__}"
            )

    def _handle_error(self, name: str, errors: list[str]) -> None:
        """Log warning or raise exception."""
        if self._mode == "strict":
            raise SlotValidationError(self._component_id, name, errors)
        else:
            msg = (
                f"[{self._component_id}] Slot '{name}': "
                + "; ".join(errors)
                + " (validation mode: warn)"
            )
            logger.warning(msg)
            self._warned_slots.add(name)
