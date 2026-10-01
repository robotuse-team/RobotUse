"""Type-name registry: the string names graphs use ↔ gap_core.types definitions.

Subgraph ``inputs:`` / ``outputs:`` declarations (and the viz frontend's port
schemas) reference types by bare name — ``"OrientedBoundingBox"``,
``"PointCloud"`` — exactly as the proto-era graphs did. This registry is the
single lookup the validator and ``viz/graph_builder`` consult.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Any

from gap_core import types as _t

__all__ = ["TYPE_REGISTRY", "resolve_type", "type_fields", "FieldInfo"]


#: Bare type-name → TypedDict class (or alias) from gap_core.types.
TYPE_REGISTRY: dict[str, Any] = {
    "Vec3": _t.Vec3,
    "Quaternion": _t.Quaternion,
    "Se3Pose": _t.Se3Pose,
    "Pose": _t.Se3Pose,  # legacy graphs use the short name
    "OrientedBoundingBox": _t.OrientedBoundingBox,
    "BoundingBox2D": _t.BoundingBox2D,
    "CameraFrame": _t.CameraFrame,
    "CameraObservation": _t.CameraFrame,  # proto-era name
    "Mask": _t.Mask,
    "PointCloud": _t.PointCloud,
    "JointState": _t.JointState,
    "Trajectory": _t.Trajectory,
    "GripperState": _t.GripperState,
    "ArmState": _t.ArmState,
    "Observation": _t.Observation,
    "ObservationResponse": _t.Observation,  # proto-era name
    "CollisionMesh": _t.CollisionMesh,
    "WorldConfig": _t.WorldConfig,
    "GraspCandidates": _t.GraspCandidates,
    # Scalars usable in declarations
    "str": str,
    "string": str,
    "int": int,
    "float": float,
    "bool": bool,
}


@dataclass(frozen=True)
class FieldInfo:
    """One field of a registered type, for validation and viz port schemas."""

    name: str
    type_str: str
    required: bool
    is_message: bool  # nested TypedDict
    is_repeated: bool  # list[...]


def resolve_type(name: str) -> Any:
    """Look up a bare type name; raises KeyError with the known names listed."""
    try:
        return TYPE_REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(TYPE_REGISTRY))
        raise KeyError(
            f"unknown type name {name!r} in graph declaration; known types: {known}"
        ) from None


def _is_typeddict(tp: Any) -> bool:
    return isinstance(tp, type) and hasattr(tp, "__annotations__") and hasattr(
        tp, "__total__"
    )


def type_fields(name_or_type: str | Any) -> list[FieldInfo]:
    """Introspect a registered type (or TypedDict class) into FieldInfo rows.

    Non-TypedDict registrations (Mask = np.ndarray, scalars) return [].
    """
    tp = resolve_type(name_or_type) if isinstance(name_or_type, str) else name_or_type
    if not _is_typeddict(tp):
        return []
    required = getattr(tp, "__required_keys__", frozenset(tp.__annotations__))
    fields: list[FieldInfo] = []
    for fname, hint in typing.get_type_hints(tp).items():
        origin = typing.get_origin(hint)
        is_repeated = origin in (list, tuple)
        inner = typing.get_args(hint)[0] if is_repeated and typing.get_args(hint) else hint
        fields.append(
            FieldInfo(
                name=fname,
                type_str=getattr(inner, "__name__", str(inner)),
                required=fname in required,
                is_message=_is_typeddict(inner),
                is_repeated=is_repeated,
            )
        )
    return fields
