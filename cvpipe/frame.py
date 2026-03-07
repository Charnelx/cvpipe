# cvpipe/frame.py
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    pass


def _torch_available() -> bool:
    return "torch" in sys.modules


@dataclass(frozen=True)
class SlotSchema:
    """
    Descriptor for one named data slot in a Frame.

    Instances are class-level declarations on Component subclasses.
    They are never instantiated at runtime during inference.

    Parameters
    ----------
    name : str
        Slot identifier. Used as the key in Frame.slots or Frame.meta.
        Convention: ``noun_coordsystem`` e.g. ``proposals_xyxy``,
        ``embeddings_cls``, ``frame_bgr``.
        Must be a valid Python identifier (enforced in __post_init__).

    dtype : torch.dtype | type | None
        Expected data type.
        - ``torch.dtype`` (e.g. ``torch.float32``) → slot lives in Frame.slots
          as a GPU/CPU tensor.
        - ``type`` (e.g. ``int``, ``str``, ``dict``) → slot lives in Frame.meta.
        - ``None`` → slot lives in Frame.meta with no dtype constraint.

    shape : tuple[int | None, ...]
        Expected tensor shape. Only meaningful when dtype is a torch.dtype.
        Use ``None`` for variable dimensions: ``(None, 4)`` means
        "any number of rows, exactly 4 columns".
        Use ``()`` (empty tuple) for scalar tensors.
        Ignored when dtype is not a torch.dtype.

    device : Literal["gpu", "cpu", "any"]
        Where the tensor must reside.
        - ``"gpu"`` → must be on a CUDA device
        - ``"cpu"`` → must be on CPU
        - ``"any"`` → no constraint
        Ignored when dtype is not a torch.dtype.

    coord_system : str | None
        Optional coordinate system tag. Pure documentation that becomes
        machine-checkable: if component A outputs a slot with
        coord_system="xyxy" and component B declares the same slot name
        with coord_system="xywh", pipeline.validate() reports an error.
        Examples: ``"xyxy"``, ``"xywh"``, ``"normalized"``, ``None``.

    description : str
        Human-readable description. Appears in validation error messages
        and generated documentation.
    """

    name: str
    dtype: Any  # torch.dtype | type | None
    shape: tuple[int | None, ...] = field(default=())
    device: Literal["gpu", "cpu", "any"] = "any"
    coord_system: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(
                f"SlotSchema.name must be a valid Python identifier, got: {self.name!r}"
            )

    def is_tensor_slot(self) -> bool:
        """True when this slot holds a torch.Tensor (dtype is a torch.dtype)."""
        if not _torch_available():
            return False
        import torch

        return isinstance(self.dtype, torch.dtype)

    def is_meta_slot(self) -> bool:
        """True when this slot lives in Frame.meta (dtype is a Python type or None)."""
        return not self.is_tensor_slot()

    def compatible_with(self, other: "SlotSchema") -> list[str]:
        """
        Check compatibility between this schema (upstream output) and
        ``other`` (downstream input). Returns a list of error strings.
        Empty list means compatible.

        Checks performed:
        1. name must match (caller's responsibility to only call with same name)
        2. If both have coord_system set, they must be equal
        3. If both are tensor slots, shape must be broadcastable:
           - Same rank
           - Each dimension: both None, or both equal, or one is None
        4. device: "gpu" input requires "gpu" output (not "any")
        """
        errors: list[str] = []

        # Coord system
        if (
            self.coord_system is not None
            and other.coord_system is not None
            and self.coord_system != other.coord_system
        ):
            errors.append(
                f"Slot '{self.name}': coordinate system mismatch — "
                f"upstream produces '{self.coord_system}', "
                f"downstream expects '{other.coord_system}'"
            )

        # Shape (tensor slots only, or when both have non-empty shape tuples)
        if self.shape and other.shape:
            if len(self.shape) != len(other.shape):
                errors.append(
                    f"Slot '{self.name}': shape rank mismatch — "
                    f"upstream {self.shape}, downstream expects {other.shape}"
                )
            else:
                for i, (a, b) in enumerate(zip(self.shape, other.shape)):
                    if a is not None and b is not None and a != b:
                        errors.append(
                            f"Slot '{self.name}': shape mismatch at dim {i} — "
                            f"upstream {a}, downstream expects {b}"
                        )

        # Device
        if other.device == "gpu" and self.device != "gpu":
            errors.append(
                f"Slot '{self.name}': device mismatch — "
                f"upstream produces '{self.device}' tensor, downstream requires gpu"
            )

        return errors


class Frame:
    """
    Mutable per-frame workspace.

    A single Frame instance is created by the Scheduler for each captured
    video frame and passed sequentially through every Component in the
    pipeline. Components read their declared INPUTS slots and write their
    declared OUTPUTS slots — all in place on this object.

    Attributes
    ----------
    idx : int
        Monotonically increasing frame counter. Starts at 0, never resets
        while the pipeline is running. Gaps indicate dropped frames.

    ts : float
        Wall-clock capture timestamp in seconds (time.monotonic() at capture).
        Used for latency measurement and diagnostics.

    slots : dict[str, torch.Tensor]
        Named tensor slots. Keys are slot names (matching SlotSchema.name
        where is_tensor_slot() is True). Values are torch.Tensor on the
        declared device.

        Access pattern:
            bgr = frame.slots["frame_bgr"]           # read
            frame.slots["proposals_xyxy"] = tensor   # write

    meta : dict[str, Any]
        CPU-side metadata: small scalars, strings, dicts of scores,
        lists of labels, routing decisions. Not for large tensors.

        Access pattern:
            mode = frame.meta["proposal_mode"]        # read
            frame.meta["routing_decision"] = "scan"   # write

    Notes
    -----
    - Frame is NOT thread-safe. It is owned by the streaming thread for
      its entire lifetime and must not be accessed from other threads.
    - Do not hold references to Frame beyond the current pipeline run.
      The Scheduler may reuse Frame instances between runs (future
      optimisation); treat them as ephemeral.
    """

    __slots__ = ("idx", "ts", "slots", "meta")

    idx: int
    ts: float
    slots: dict[str, Any]
    meta: dict[str, Any]

    def __init__(self, idx: int, ts: float) -> None:
        self.idx = idx
        self.ts = ts
        self.slots: dict[str, Any] = {}
        self.meta: dict[str, Any] = {}

    def __repr__(self) -> str:
        slot_names = list(self.slots.keys())
        meta_keys = list(self.meta.keys())
        return f"Frame(idx={self.idx}, ts={self.ts:.3f}, slots={slot_names}, meta={meta_keys})"
