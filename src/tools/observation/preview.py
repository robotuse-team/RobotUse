"""Typed visual-point decisions for planner-owned robot motion.

The vision model may choose a registered RGB view, a normalized image point,
and optionally a bounded camera-relative gripper orientation for visualization.
It cannot emit metric positions, joints, trajectories, controller commands,
or executable code. Trusted perception and planning own every conversion
from the accepted image mark to robot motion; the orientation preview does
not authorize or constrain that motion.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import json
import math
import re
from typing import Any, Mapping


class VisualWaypointError(ValueError):
    """Raised when model-authored visual data violates the typed boundary."""


class TargetRole(str, Enum):
    PICK = "pick"
    PLACE = "place"
    WAYPOINT = "waypoint"


class PickPlacePhase(str, Enum):
    PICK_POINT = "pick_point"
    PICK_REVIEW = "pick_review"
    GRASP = "grasp"
    GRASP_TRAJECTORY = "grasp_trajectory"
    PLACE_POINT = "place_point"
    PLACE_REVIEW = "place_review"
    PLACE_TRAJECTORY = "place_trajectory"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class NormalizedPoint:
    """Provider-friendly image coordinate in the closed interval 0..1000."""

    u: float
    v: float

    def __post_init__(self) -> None:
        for name in ("u", "v"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise VisualWaypointError(f"{name} must be numeric")
            parsed = float(value)
            if not math.isfinite(parsed) or not 0.0 <= parsed <= 1000.0:
                raise VisualWaypointError(f"{name} must be finite and within 0..1000")
            object.__setattr__(self, name, parsed)

    def pixels(self, *, width: int, height: int) -> tuple[float, float]:
        if type(width) is not int or type(height) is not int or width < 2 or height < 2:
            raise VisualWaypointError("image dimensions must be integers >= 2")
        return (self.u * (width - 1) / 1000.0, self.v * (height - 1) / 1000.0)

    def to_dict(self) -> dict[str, float]:
        return {"u": self.u, "v": self.v}


@dataclass(frozen=True, slots=True)
class GripperOrientation:
    """Preview orientation: R_camera_from_gripper = Rz(yaw) Ry(pitch) Rx(roll).

    Selected optical camera: +X right, +Y down, +Z into scene. Gripper-local
    +Z is approach and +Y is jaw separation. This is not an executable pose.
    """

    frame: str = "selected_camera"
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.frame != "selected_camera":
            raise VisualWaypointError("orientation frame must be selected_camera")
        for name in ("roll_deg", "pitch_deg", "yaw_deg"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise VisualWaypointError(f"{name} must be numeric")
            try:
                parsed = float(value)
            except OverflowError as exc:
                raise VisualWaypointError(f"{name} must be finite and within -180..180 degrees") from exc
            if not math.isfinite(parsed) or not -180.0 <= parsed <= 180.0:
                raise VisualWaypointError(f"{name} must be finite and within -180..180 degrees")
            object.__setattr__(self, name, parsed)

    def to_dict(self) -> dict[str, str | float]:
        return {
            "frame": self.frame,
            "roll_deg": self.roll_deg,
            "pitch_deg": self.pitch_deg,
            "yaw_deg": self.yaw_deg,
        }


@dataclass(frozen=True, slots=True)
class VisualPointSelection:
    role: TargetRole
    view_id: str
    point: NormalizedPoint
    orientation: GripperOrientation | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, TargetRole):
            try:
                object.__setattr__(self, "role", TargetRole(self.role))
            except (TypeError, ValueError) as exc:
                raise VisualWaypointError("role must be pick, place, or waypoint") from exc
        if not isinstance(self.view_id, str) or self.view_id not in {"front", "wrist"}:
            raise VisualWaypointError("view_id must be front or wrist")
        if not isinstance(self.point, NormalizedPoint):
            raise VisualWaypointError("point must be a NormalizedPoint")
        if self.orientation is not None and not isinstance(self.orientation, GripperOrientation):
            raise VisualWaypointError("orientation must be a GripperOrientation")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "role": self.role.value,
            "view_id": self.view_id,
            "point": self.point.to_dict(),
        }
        if self.orientation is not None:
            result["orientation"] = self.orientation.to_dict()
        return result


@dataclass(frozen=True, slots=True)
class VisualPointReview:
    action: str
    revision: VisualPointSelection | None = None

    def __post_init__(self) -> None:
        if self.action not in {"keep", "revise"}:
            raise VisualWaypointError("review action must be keep or revise")
        if self.action == "keep" and self.revision is not None:
            raise VisualWaypointError("keep cannot carry a revision")
        if self.action == "revise" and self.revision is None:
            raise VisualWaypointError("revise requires a replacement point")


@dataclass(frozen=True, slots=True)
class SimplifiedPickPlaceState:
    """Minimal point -> review -> candidate -> execution protocol."""

    phase: PickPlacePhase = PickPlacePhase.PICK_POINT
    revision: int = 0
    pick_selection_id: str | None = None
    grasp_id: str | None = None
    grasp_trajectory_id: str | None = None
    place_selection_id: str | None = None
    place_trajectory_id: str | None = None

    def _expect(self, phase: PickPlacePhase) -> None:
        if self.phase is not phase:
            raise VisualWaypointError(f"expected phase {phase.value}, got {self.phase.value}")

    @staticmethod
    def _pick_or_place(role: TargetRole) -> TargetRole:
        try:
            parsed = TargetRole(role)
        except (TypeError, ValueError) as exc:
            raise VisualWaypointError("role must be pick or place") from exc
        if parsed not in {TargetRole.PICK, TargetRole.PLACE}:
            raise VisualWaypointError("role must be pick or place")
        return parsed

    def point_submitted(self, role: TargetRole) -> "SimplifiedPickPlaceState":
        role = self._pick_or_place(role)
        expected = PickPlacePhase.PICK_POINT if role is TargetRole.PICK else PickPlacePhase.PLACE_POINT
        self._expect(expected)
        next_phase = PickPlacePhase.PICK_REVIEW if role is TargetRole.PICK else PickPlacePhase.PLACE_REVIEW
        return replace(self, phase=next_phase, revision=self.revision + 1)

    def accept_point(self, role: TargetRole, selection_id: str) -> "SimplifiedPickPlaceState":
        role = self._pick_or_place(role)
        expected = PickPlacePhase.PICK_REVIEW if role is TargetRole.PICK else PickPlacePhase.PLACE_REVIEW
        self._expect(expected)
        if not selection_id:
            raise VisualWaypointError("selection_id must be non-empty")
        if role is TargetRole.PICK:
            return replace(self, phase=PickPlacePhase.GRASP, pick_selection_id=selection_id, revision=self.revision + 1)
        return replace(self, phase=PickPlacePhase.PLACE_TRAJECTORY, place_selection_id=selection_id, revision=self.revision + 1)

    def revise_point(self, role: TargetRole) -> "SimplifiedPickPlaceState":
        role = self._pick_or_place(role)
        expected = PickPlacePhase.PICK_REVIEW if role is TargetRole.PICK else PickPlacePhase.PLACE_REVIEW
        self._expect(expected)
        next_phase = PickPlacePhase.PICK_POINT if role is TargetRole.PICK else PickPlacePhase.PLACE_POINT
        return replace(self, phase=next_phase, revision=self.revision + 1)

    def select_grasp(self, grasp_id: str) -> "SimplifiedPickPlaceState":
        self._expect(PickPlacePhase.GRASP)
        if not grasp_id:
            raise VisualWaypointError("grasp_id must be non-empty")
        return replace(self, phase=PickPlacePhase.GRASP_TRAJECTORY, grasp_id=grasp_id, revision=self.revision + 1)

    def select_grasp_trajectory(self, trajectory_id: str) -> "SimplifiedPickPlaceState":
        self._expect(PickPlacePhase.GRASP_TRAJECTORY)
        if not trajectory_id:
            raise VisualWaypointError("trajectory_id must be non-empty")
        return replace(self, grasp_trajectory_id=trajectory_id, revision=self.revision + 1)

    def grasp_executed(self, verified: bool) -> "SimplifiedPickPlaceState":
        self._expect(PickPlacePhase.GRASP_TRAJECTORY)
        if self.grasp_trajectory_id is None:
            raise VisualWaypointError("grasp execution requires a selected trajectory")
        next_phase = PickPlacePhase.PLACE_POINT if verified else PickPlacePhase.FAILED
        return replace(self, phase=next_phase, revision=self.revision + 1)

    def select_place_trajectory(self, trajectory_id: str) -> "SimplifiedPickPlaceState":
        self._expect(PickPlacePhase.PLACE_TRAJECTORY)
        if not trajectory_id:
            raise VisualWaypointError("trajectory_id must be non-empty")
        return replace(self, place_trajectory_id=trajectory_id, revision=self.revision + 1)

    def finish(self, verified: bool) -> "SimplifiedPickPlaceState":
        self._expect(PickPlacePhase.PLACE_TRAJECTORY)
        if self.place_trajectory_id is None:
            raise VisualWaypointError("finish requires a selected trajectory")
        next_phase = PickPlacePhase.COMPLETE if verified else PickPlacePhase.FAILED
        return replace(self, phase=next_phase, revision=self.revision + 1)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VisualWaypointError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise VisualWaypointError(f"non-standard JSON constant is forbidden: {value}")


def _loads_json_object(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_json_constant,
    )


def _json_object(text: str) -> Mapping[str, Any]:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    try:
        value = _loads_json_object(stripped)
    except json.JSONDecodeError:
        objects = re.findall(r"\{[^{}]*\}", stripped)
        if not objects:
            raise VisualWaypointError("response contains no JSON object")
        value = _loads_json_object(objects[-1])
    if not isinstance(value, Mapping):
        raise VisualWaypointError("response must be a JSON object")
    return value


def _point_from_object(
    value: Mapping[str, Any], *, role: TargetRole, require_orientation: bool
) -> VisualPointSelection:
    keys = {"view_id", "u", "v"}
    if "orientation" in value or require_orientation:
        keys.add("orientation")
    if set(value) != keys:
        suffix = ", and orientation" if "orientation" in keys else ""
        raise VisualWaypointError(f"point response must contain exactly view_id, u, v{suffix}")
    orientation = None
    if "orientation" in value:
        raw = value["orientation"]
        if not isinstance(raw, Mapping) or set(raw) != {
            "frame", "roll_deg", "pitch_deg", "yaw_deg"
        }:
            raise VisualWaypointError(
                "orientation must contain exactly frame, roll_deg, pitch_deg, and yaw_deg"
            )
        orientation = GripperOrientation(**raw)
    return VisualPointSelection(
        role=role,
        view_id=value["view_id"],
        point=NormalizedPoint(u=value["u"], v=value["v"]),
        orientation=orientation,
    )


def parse_visual_point(
    text: str, *, role: TargetRole, require_orientation: bool = False
) -> VisualPointSelection:
    return _point_from_object(
        _json_object(text), role=role, require_orientation=require_orientation
    )


def parse_visual_review(
    text: str, *, role: TargetRole, require_orientation: bool = False
) -> VisualPointReview:
    value = _json_object(text)
    if value == {"action": "keep"}:
        return VisualPointReview(action="keep")
    if value.get("action") == "revise":
        return VisualPointReview(
            action="revise",
            revision=_point_from_object(
                {key: item for key, item in value.items() if key != "action"},
                role=role, require_orientation=require_orientation,
            ),
        )
    raise VisualWaypointError("review response must be exact keep or revise JSON")
