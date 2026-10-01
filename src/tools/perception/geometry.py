"""Finite RGB-D perception and bounded recovery for the visual robot actor.

The benchmark oracle is deliberately outside this module.  The actor may see
only rendered evidence and finite IDs; RGB/depth arrays, calibration, masks,
point clouds, oriented boxes, and any backend handles live in a private
registry.  No VLA or learned motor policy is used here.
"""


from __future__ import annotations


from dataclasses import dataclass, field, replace


from enum import Enum


import hashlib


import json


import math


from pathlib import Path


from types import MappingProxyType


from typing import Any, Iterable, Iterator, Mapping, Sequence


from src.tools.grasp.candidates import (
    Candidate,
    CandidateBatch,
    CandidateChoice,
    CandidateDecision,
    CandidateMetrics,
    CandidateSelectionError,
    CandidateStage,
    CandidateValidationError,
    EvidenceRef,
    FeasibilityStatus,
    SafetyStatus,
    StageSelection,
)


from src.core.models import JsonValue


METRIC_DEPTH_ARTIFACT_SCHEMA = "metric-depth-f32le.v1"


CALIBRATION_ARTIFACT_SCHEMA = "sensor-calibration.v1"


METRIC_DEPTH_MAGIC = b"RSS_METRIC_DEPTH_F32LE_V1\n"


class SensorPerceptionError(RuntimeError):
    """Raised when sensor evidence or the continual actor contract is invalid."""


class ObservationToolId(str, Enum):
    """Finite sensor wrappers; none accepts coordinates or arbitrary payloads."""

    CAPTURE_RGBD = "capture_rgbd"
    REOBSERVE = "reobserve"
    STOP_SAFELY = "stop_safely"


class PerceptionToolId(str, Enum):
    """Finite RGB-D perception wrappers selected through guarded IDs."""

    DINO_BOX_BALANCED = "dino_box_balanced"
    DINO_BOX_RECALL = "dino_box_recall"
    SAM_TEXT_FALLBACK = "sam_text_fallback"
    SAM_BOX_REFINE = "sam_box_refine"
    SAM_BOX_DEPTH_CONSISTENT = "sam_box_depth_consistent"
    MULTIVIEW_TRACK = "multiview_track"
    REOBSERVE = "reobserve"


class ObservationProfileId(str, Enum):
    """Fixed sensor-plus-proposal profiles selected by ID."""

    DINO_BOX_BALANCED = "dino_box_balanced"
    DINO_BOX_RECALL = "dino_box_recall"
    SAM_TEXT_FALLBACK = "sam_text_fallback"
    MULTIVIEW_TRACK = "multiview_track"


class PerceptionRole(str, Enum):
    TARGET = "target"
    DESTINATION = "destination"


class RecoveryActionId(str, Enum):
    RETAKE_SAME_PROFILE = "retake_same_profile"
    SWITCH_PROFILE = "switch_profile"
    RESEGMENT = "resegment"
    RESELECT_TARGET = "reselect_target"
    RESELECT_DESTINATION = "reselect_destination"
    STOP_SAFELY = "stop_safely"


class PerceptionFailureKind(str, Enum):
    CAPTURE_FAILED = "capture_failed"
    NO_HYPOTHESES = "no_hypotheses"
    MASK_INVALID = "mask_invalid"
    DEPTH_INSUFFICIENT = "depth_insufficient"
    GEOMETRY_INCONSISTENT = "geometry_inconsistent"
    TRACK_LOST = "track_lost"
    VERIFICATION_AMBIGUOUS = "verification_ambiguous"


class SensorActorPhase(str, Enum):
    CHOOSE_OBSERVATION_TOOL = "choose_observation_tool"
    CHOOSE_OBSERVATION_PROFILE = "choose_observation_profile"
    CAPTURE_PENDING = "capture_pending"
    SELECT_TARGET_HYPOTHESIS = "select_target_hypothesis"
    SELECT_TARGET_MASK = "select_target_mask"
    TARGET_GEOMETRY_PENDING = "target_geometry_pending"
    SELECT_DESTINATION_HYPOTHESIS = "select_destination_hypothesis"
    SELECT_DESTINATION_MASK = "select_destination_mask"
    DESTINATION_GEOMETRY_PENDING = "destination_geometry_pending"
    READY = "ready"
    RECOVERY = "recovery"
    STOPPED = "stopped"


class SensorTerminalStatus(str, Enum):
    RUNNING = "running"
    READY = "ready"
    STOPPED = "stopped"


def _require_id(value: Any, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or not value[0].isalnum()
        or any(not (character.isalnum() or character in "_.:-") for character in value)
    ):
        raise SensorPerceptionError(f"{context} must be a valid identifier")
    return value


def _require_finite(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SensorPerceptionError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise SensorPerceptionError(f"{context} must be a finite number")
    return result


def _require_sha256(value: Any, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SensorPerceptionError(f"{context} must be lowercase SHA-256")
    return value


def _sha256_path(path: str | Path, *, context: str) -> str:
    artifact = Path(path)
    if not artifact.is_file():
        raise SensorPerceptionError(f"{context} is missing: {artifact}")
    digest = hashlib.sha256()
    try:
        with artifact.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SensorPerceptionError(f"cannot read {context}: {artifact}") from exc
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RigidTransform:
    """A private camera-to-actor transform."""

    rotation: tuple[tuple[float, float, float], ...]
    translation: tuple[float, float, float]

    def __post_init__(self) -> None:
        rotation = tuple(tuple(_require_finite(v, context="rotation") for v in row) for row in self.rotation)
        translation = tuple(
            _require_finite(v, context="translation") for v in self.translation
        )
        if len(rotation) != 3 or any(len(row) != 3 for row in rotation):
            raise SensorPerceptionError("rotation must be 3x3")
        if len(translation) != 3:
            raise SensorPerceptionError("translation must contain three values")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    @classmethod
    def identity(cls) -> "RigidTransform":
        return cls(
            rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            translation=(0.0, 0.0, 0.0),
        )

    def apply(self, point: Sequence[float]) -> tuple[float, float, float]:
        if len(point) != 3:
            raise SensorPerceptionError("point must contain three values")
        checked = tuple(_require_finite(value, context="point") for value in point)
        return tuple(
            sum(self.rotation[row][column] * checked[column] for column in range(3))
            + self.translation[row]
            for row in range(3)
        )  # type: ignore[return-value]

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "rotation": [list(row) for row in self.rotation],
            "translation": list(self.translation),
        }


@dataclass(frozen=True, slots=True)
class PixelBox:
    left: int
    top: int
    right: int
    bottom: int

    def __post_init__(self) -> None:
        values = (self.left, self.top, self.right, self.bottom)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise SensorPerceptionError("pixel box coordinates must be integers")
        if self.left < 0 or self.top < 0 or self.right <= self.left or self.bottom <= self.top:
            raise SensorPerceptionError("pixel box must have positive in-frame area")

    @property
    def area(self) -> int:
        return (self.right - self.left) * (self.bottom - self.top)


@dataclass(frozen=True, slots=True)
class OrientedBoundingBox:
    """Depth-derived 3-D box in the actor's calibrated base frame."""

    center: tuple[float, float, float]
    axes: tuple[tuple[float, float, float], ...]
    half_extents: tuple[float, float, float]

    def __post_init__(self) -> None:
        center = tuple(_require_finite(value, context="OBB center") for value in self.center)
        axes = tuple(
            tuple(_require_finite(value, context="OBB axis") for value in axis)
            for axis in self.axes
        )
        half_extents = tuple(
            _require_finite(value, context="OBB extent") for value in self.half_extents
        )
        if len(center) != 3 or len(axes) != 3 or any(len(axis) != 3 for axis in axes):
            raise SensorPerceptionError("OBB center/axes must be three-dimensional")
        if len(half_extents) != 3 or any(value <= 0.0 for value in half_extents):
            raise SensorPerceptionError("OBB half extents must be positive")
        tolerance = 1e-6
        norms = tuple(math.sqrt(sum(value * value for value in axis)) for axis in axes)
        if any(abs(norm - 1.0) > tolerance for norm in norms):
            raise SensorPerceptionError("OBB axes must be unit length")
        if any(
            abs(sum(axes[first][index] * axes[second][index] for index in range(3)))
            > tolerance
            for first, second in ((0, 1), (0, 2), (1, 2))
        ):
            raise SensorPerceptionError("OBB axes must be orthogonal")
        cross_xy = (
            axes[0][1] * axes[1][2] - axes[0][2] * axes[1][1],
            axes[0][2] * axes[1][0] - axes[0][0] * axes[1][2],
            axes[0][0] * axes[1][1] - axes[0][1] * axes[1][0],
        )
        handedness = sum(cross_xy[index] * axes[2][index] for index in range(3))
        if handedness < 1.0 - tolerance:
            raise SensorPerceptionError("OBB axes must form a right-handed basis")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "half_extents", half_extents)

    def corners(self) -> tuple[tuple[float, float, float], ...]:
        corners: list[tuple[float, float, float]] = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    signs = (sx, sy, sz)
                    corners.append(
                        tuple(
                            self.center[axis]
                            + sum(
                                signs[basis]
                                * self.half_extents[basis]
                                * self.axes[basis][axis]
                                for basis in range(3)
                            )
                            for axis in range(3)
                        )
                    )
        return tuple(corners)

    def to_private_dict(self) -> dict[str, JsonValue]:
        return {
            "center": list(self.center),
            "axes": [list(axis) for axis in self.axes],
            "half_extents": list(self.half_extents),
        }

    @classmethod
    def from_private_dict(cls, value: Mapping[str, Any]) -> "OrientedBoundingBox":
        if not isinstance(value, Mapping) or set(value) != {
            "center",
            "axes",
            "half_extents",
        }:
            raise SensorPerceptionError("private OBB artifact has invalid keys")
        return cls(
            center=tuple(value["center"]),
            axes=tuple(tuple(axis) for axis in value["axes"]),
            half_extents=tuple(value["half_extents"]),
        )


@dataclass(frozen=True, slots=True)
class SensorFramePayload:
    """Private synchronized RGB-D frame and calibration."""

    frame_id: str
    view_id: str
    width: int
    height: int
    rgb: Any
    depth_m: Any
    intrinsics: tuple[tuple[float, float, float], ...]
    camera_to_base: RigidTransform
    rgb_path: str
    depth_evidence_path: str
    rgb_sha256: str
    depth_sha256: str
    depth_metric_path: str | None = None
    depth_evidence_sha256: str | None = None
    calibration_path: str | None = None
    calibration_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_id(self.frame_id, context="frame_id")
        _require_id(self.view_id, context="view_id")
        if (
            isinstance(self.width, bool)
            or isinstance(self.height, bool)
            or not isinstance(self.width, int)
            or not isinstance(self.height, int)
            or self.width <= 0
            or self.height <= 0
        ):
            raise SensorPerceptionError("frame dimensions must be positive integers")
        intrinsics = tuple(
            tuple(_require_finite(value, context="intrinsics") for value in row)
            for row in self.intrinsics
        )
        if len(intrinsics) != 3 or any(len(row) != 3 for row in intrinsics):
            raise SensorPerceptionError("intrinsics must be 3x3")
        if intrinsics[0][0] <= 0.0 or intrinsics[1][1] <= 0.0:
            raise SensorPerceptionError("camera focal lengths must be positive")
        object.__setattr__(self, "intrinsics", intrinsics)
        if not isinstance(self.camera_to_base, RigidTransform):
            raise SensorPerceptionError("camera_to_base must be RigidTransform")
        for name in ("rgb_path", "depth_evidence_path"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise SensorPerceptionError(f"{name} must be non-empty")
        for name in ("rgb_sha256", "depth_sha256"):
            _require_sha256(getattr(self, name), context=name)
        private_fields = (
            self.depth_metric_path,
            self.depth_evidence_sha256,
            self.calibration_path,
            self.calibration_sha256,
        )
        if any(value is not None for value in private_fields):
            if not all(value is not None for value in private_fields):
                raise SensorPerceptionError(
                    "private frame artifact paths and hashes must be provided together"
                )
            assert self.depth_metric_path is not None
            assert self.depth_evidence_sha256 is not None
            assert self.calibration_path is not None
            assert self.calibration_sha256 is not None
            if not self.depth_metric_path or not self.calibration_path:
                raise SensorPerceptionError("private frame artifact paths must be non-empty")
            _require_sha256(
                self.depth_evidence_sha256,
                context="depth_evidence_sha256",
            )
            _require_sha256(self.calibration_sha256, context="calibration_sha256")

    @property
    def has_private_artifacts(self) -> bool:
        return self.depth_metric_path is not None

    def _expected_calibration(self) -> dict[str, JsonValue]:
        return {
            "schema_version": CALIBRATION_ARTIFACT_SCHEMA,
            "view_id": self.view_id,
            "width": self.width,
            "height": self.height,
            "intrinsics": [list(row) for row in self.intrinsics],
            "camera_to_base": self.camera_to_base.to_dict(),
        }

    def private_evidence_record(self) -> dict[str, JsonValue]:
        """Verify and describe immutable private RGB-D/calibration artifacts."""

        if not self.has_private_artifacts:
            raise SensorPerceptionError("frame lacks persisted private depth/calibration artifacts")
        assert self.depth_metric_path is not None
        assert self.depth_evidence_sha256 is not None
        assert self.calibration_path is not None
        assert self.calibration_sha256 is not None
        checks = (
            (self.rgb_path, self.rgb_sha256, "RGB artifact"),
            (
                self.depth_evidence_path,
                self.depth_evidence_sha256,
                "depth preview artifact",
            ),
            (self.depth_metric_path, self.depth_sha256, "metric depth artifact"),
            (self.calibration_path, self.calibration_sha256, "calibration artifact"),
        )
        for path, expected, context in checks:
            if _sha256_path(path, context=context) != expected:
                raise SensorPerceptionError(f"{context} hash mismatches")
        try:
            with Path(self.depth_metric_path).open("rb") as stream:
                magic = stream.readline()
                raw_header = stream.readline()
                payload_size = len(stream.read())
            header = json.loads(raw_header)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SensorPerceptionError("metric depth artifact is malformed") from exc
        expected_header = {
            "schema_version": METRIC_DEPTH_ARTIFACT_SCHEMA,
            "dtype": "float32-le",
            "order": "C",
            "width": self.width,
            "height": self.height,
        }
        if magic != METRIC_DEPTH_MAGIC or header != expected_header:
            raise SensorPerceptionError("metric depth artifact metadata mismatches the frame")
        if payload_size != self.width * self.height * 4:
            raise SensorPerceptionError("metric depth artifact payload has the wrong size")
        try:
            calibration = json.loads(Path(self.calibration_path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SensorPerceptionError("calibration artifact is malformed") from exc
        if calibration != self._expected_calibration():
            raise SensorPerceptionError("calibration artifact does not match the frame")
        return {
            "frame_id": self.frame_id,
            "view_id": self.view_id,
            "width": self.width,
            "height": self.height,
            "rgb_path": self.rgb_path,
            "rgb_sha256": self.rgb_sha256,
            "depth_preview_path": self.depth_evidence_path,
            "depth_preview_sha256": self.depth_evidence_sha256,
            "depth_metric_path": self.depth_metric_path,
            "depth_metric_sha256": self.depth_sha256,
            "calibration_path": self.calibration_path,
            "calibration_sha256": self.calibration_sha256,
        }

    def public_metadata(self) -> dict[str, JsonValue]:
        metadata: dict[str, JsonValue] = {
            "frame_id": self.frame_id,
            "view_id": self.view_id,
            "width": self.width,
            "height": self.height,
            "rgb_path": self.rgb_path,
            "depth_evidence_path": self.depth_evidence_path,
            "rgb_sha256": self.rgb_sha256,
        }
        if self.depth_evidence_sha256 is not None:
            metadata["depth_evidence_sha256"] = self.depth_evidence_sha256
        return metadata


@dataclass(frozen=True, slots=True)
class VisualHypothesisPayload:
    candidate_id: str
    selector_id: str
    frame_id: str
    box: PixelBox
    seed_mask: Any
    confidence: float
    evidence_path: str
    source: str
    semantic_hint: str | None = None

    def __post_init__(self) -> None:
        _require_id(self.candidate_id, context="hypothesis candidate_id")
        _require_id(self.selector_id, context="hypothesis selector_id")
        _require_id(self.frame_id, context="hypothesis frame_id")
        if not isinstance(self.box, PixelBox):
            raise SensorPerceptionError("hypothesis box must be PixelBox")
        confidence = _require_finite(self.confidence, context="hypothesis confidence")
        if not 0.0 <= confidence <= 1.0:
            raise SensorPerceptionError("hypothesis confidence must be between 0 and 1")
        object.__setattr__(self, "confidence", confidence)
        if not isinstance(self.evidence_path, str) or not self.evidence_path:
            raise SensorPerceptionError("hypothesis evidence path must be non-empty")
        if not isinstance(self.source, str) or not self.source:
            raise SensorPerceptionError("hypothesis source must be non-empty")
        if self.semantic_hint is not None:
            if not isinstance(self.semantic_hint, str):
                raise SensorPerceptionError("hypothesis semantic hint must be text")
            if "\x00" in self.semantic_hint or any(
                ord(character) < 32 for character in self.semantic_hint
            ):
                raise SensorPerceptionError("hypothesis semantic hint is unsafe")
            hint = " ".join(self.semantic_hint.split())
            if (
                not hint
                or len(hint) > 160
                or any(
                    not (character.isalnum() or character in " _./()+-")
                    for character in hint
                )
            ):
                raise SensorPerceptionError("hypothesis semantic hint is unsafe")
            object.__setattr__(self, "semantic_hint", hint)


@dataclass(frozen=True, slots=True)
class MaskPayload:
    candidate_id: str
    selector_id: str
    hypothesis_id: str
    frame_id: str
    mask: Any
    confidence: float
    depth_coverage: float
    evidence_path: str
    method: str
    artifact_path: str | None = None
    artifact_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in ("candidate_id", "selector_id", "hypothesis_id", "frame_id"):
            _require_id(getattr(self, name), context=name)
        for name in ("confidence", "depth_coverage"):
            value = _require_finite(getattr(self, name), context=name)
            if not 0.0 <= value <= 1.0:
                raise SensorPerceptionError(f"{name} must be between 0 and 1")
            object.__setattr__(self, name, value)
        if not isinstance(self.evidence_path, str) or not self.evidence_path:
            raise SensorPerceptionError("mask evidence path must be non-empty")
        if not isinstance(self.method, str) or not self.method:
            raise SensorPerceptionError("mask method must be non-empty")
        if (self.artifact_path is None) != (self.artifact_sha256 is None):
            raise SensorPerceptionError("mask artifact path and hash must be provided together")
        if self.artifact_path is not None:
            if not isinstance(self.artifact_path, str) or not self.artifact_path:
                raise SensorPerceptionError("mask artifact path must be non-empty")
            _require_sha256(self.artifact_sha256, context="mask artifact_sha256")

    def private_evidence_record(self) -> dict[str, JsonValue]:
        if self.artifact_path is None or self.artifact_sha256 is None:
            raise SensorPerceptionError("mask lacks a persisted private artifact")
        if _sha256_path(self.artifact_path, context="mask artifact") != self.artifact_sha256:
            raise SensorPerceptionError("mask artifact hash mismatches")
        return {
            "mask_id": self.candidate_id,
            "hypothesis_id": self.hypothesis_id,
            "frame_id": self.frame_id,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass(frozen=True, slots=True)
class GeometryPayload:
    geometry_id: str
    mask_id: str
    role: PerceptionRole
    points: tuple[tuple[float, float, float], ...]
    obb: OrientedBoundingBox
    depth_coverage: float

    def __post_init__(self) -> None:
        _require_id(self.geometry_id, context="geometry_id")
        _require_id(self.mask_id, context="geometry mask_id")
        if not isinstance(self.role, PerceptionRole):
            try:
                object.__setattr__(self, "role", PerceptionRole(self.role))
            except (TypeError, ValueError) as exc:
                raise SensorPerceptionError("unsupported geometry role") from exc
        points = tuple(
            tuple(_require_finite(value, context="point cloud") for value in point)
            for point in self.points
        )
        if not points or any(len(point) != 3 for point in points):
            raise SensorPerceptionError("geometry requires a non-empty 3-D point cloud")
        object.__setattr__(self, "points", points)
        if not isinstance(self.obb, OrientedBoundingBox):
            raise SensorPerceptionError("geometry obb must be OrientedBoundingBox")
        coverage = _require_finite(self.depth_coverage, context="depth coverage")
        if not 0.0 <= coverage <= 1.0:
            raise SensorPerceptionError("depth coverage must be between 0 and 1")
        object.__setattr__(self, "depth_coverage", coverage)

    def to_private_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": "sensor-geometry.v1",
            "geometry_id": self.geometry_id,
            "mask_id": self.mask_id,
            "role": self.role.value,
            "points": [list(point) for point in self.points],
            "obb": self.obb.to_private_dict(),
            "depth_coverage": self.depth_coverage,
        }

    @classmethod
    def from_private_dict(cls, value: Mapping[str, Any]) -> "GeometryPayload":
        if not isinstance(value, Mapping) or set(value) != {
            "schema_version",
            "geometry_id",
            "mask_id",
            "role",
            "points",
            "obb",
            "depth_coverage",
        }:
            raise SensorPerceptionError("private geometry artifact has invalid keys")
        if value["schema_version"] != "sensor-geometry.v1":
            raise SensorPerceptionError("private geometry artifact schema is unsupported")
        return cls(
            geometry_id=value["geometry_id"],
            mask_id=value["mask_id"],
            role=value["role"],
            points=tuple(tuple(point) for point in value["points"]),
            obb=OrientedBoundingBox.from_private_dict(value["obb"]),
            depth_coverage=value["depth_coverage"],
        )


class PrivatePerceptionRegistry:
    """In-process payload registry with an intentionally lossy public manifest."""

    def __init__(self) -> None:
        self._frames: dict[str, SensorFramePayload] = {}
        self._hypotheses: dict[str, VisualHypothesisPayload] = {}
        self._masks: dict[str, MaskPayload] = {}
        self._geometries: dict[str, GeometryPayload] = {}

    @staticmethod
    def _insert(store: dict[str, Any], key: str, payload: Any) -> None:
        if key in store and store[key] is not payload:
            raise SensorPerceptionError(f"private registry ID collision: {key}")
        store[key] = payload

    def register_frame(self, payload: SensorFramePayload) -> None:
        if not isinstance(payload, SensorFramePayload):
            raise SensorPerceptionError("frame payload has the wrong type")
        self._insert(self._frames, payload.frame_id, payload)

    def register_hypothesis(self, payload: VisualHypothesisPayload) -> None:
        if not isinstance(payload, VisualHypothesisPayload):
            raise SensorPerceptionError("hypothesis payload has the wrong type")
        if payload.frame_id not in self._frames:
            raise SensorPerceptionError("hypothesis references an unknown frame")
        self._insert(self._hypotheses, payload.candidate_id, payload)

    def register_mask(self, payload: MaskPayload) -> None:
        if not isinstance(payload, MaskPayload):
            raise SensorPerceptionError("mask payload has the wrong type")
        hypothesis = self._hypotheses.get(payload.hypothesis_id)
        if hypothesis is None or hypothesis.frame_id != payload.frame_id:
            raise SensorPerceptionError("mask references a stale or unknown hypothesis")
        self._insert(self._masks, payload.candidate_id, payload)

    def register_geometry(
        self,
        payload: GeometryPayload,
        *,
        expected_role: PerceptionRole,
        expected_mask_id: str,
    ) -> None:
        if not isinstance(payload, GeometryPayload):
            raise SensorPerceptionError("geometry payload has the wrong type")
        checked_role = PerceptionRole(expected_role)
        _require_id(expected_mask_id, context="expected geometry mask_id")
        if payload.role is not checked_role or payload.mask_id != expected_mask_id:
            raise SensorPerceptionError("geometry role/mask binding is inconsistent")
        mask = self._masks.get(payload.mask_id)
        if mask is None:
            raise SensorPerceptionError("geometry references an unknown mask")
        if payload.depth_coverage != mask.depth_coverage:
            raise SensorPerceptionError("geometry depth coverage does not match its mask")
        existing = self._geometries.get(payload.geometry_id)
        if existing is not None and existing.to_private_dict() == payload.to_private_dict():
            # Retrying extraction for the same deterministic mask and point cloud
            # is an idempotent registration, not an identifier collision.
            return
        self._insert(self._geometries, payload.geometry_id, payload)

    def bind_geometry(
        self,
        *,
        state: "SensorActorState",
        payload: GeometryPayload,
    ) -> "SensorActorState":
        """Atomically validate the selected role/mask, register, and advance state."""

        if not isinstance(state, SensorActorState):
            raise SensorPerceptionError("geometry binding requires SensorActorState")
        if state.phase is SensorActorPhase.TARGET_GEOMETRY_PENDING:
            expected_role = PerceptionRole.TARGET
            expected_mask_id = state.selected_target_mask_id
        elif state.phase is SensorActorPhase.DESTINATION_GEOMETRY_PENDING:
            expected_role = PerceptionRole.DESTINATION
            expected_mask_id = state.selected_destination_mask_id
        else:
            raise CandidateSelectionError("geometry is not pending")
        if expected_mask_id is None or expected_mask_id != payload.mask_id:
            raise SensorPerceptionError("geometry does not match the selected role mask")
        advanced = state.register_geometry(role=expected_role, geometry=payload)
        self.register_geometry(
            payload,
            expected_role=expected_role,
            expected_mask_id=expected_mask_id,
        )
        return advanced

    def frame(self, frame_id: str) -> SensorFramePayload:
        try:
            return self._frames[frame_id]
        except KeyError as exc:
            raise SensorPerceptionError(f"unknown private frame: {frame_id}") from exc

    def hypothesis(self, candidate_id: str) -> VisualHypothesisPayload:
        try:
            return self._hypotheses[candidate_id]
        except KeyError as exc:
            raise SensorPerceptionError(
                f"unknown private hypothesis: {candidate_id}"
            ) from exc

    def mask(self, candidate_id: str) -> MaskPayload:
        try:
            return self._masks[candidate_id]
        except KeyError as exc:
            raise SensorPerceptionError(f"unknown private mask: {candidate_id}") from exc

    def geometry(self, geometry_id: str) -> GeometryPayload:
        try:
            return self._geometries[geometry_id]
        except KeyError as exc:
            raise SensorPerceptionError(f"unknown private geometry: {geometry_id}") from exc

    def geometries(self) -> tuple[GeometryPayload, ...]:
        """Every registered geometry, ordered by id so lookups are reproducible."""

        return tuple(self._geometries[key] for key in sorted(self._geometries))

    def masks(self) -> tuple[MaskPayload, ...]:
        """Every registered mask, ordered by id so lookups are reproducible."""

        return tuple(self._masks[key] for key in sorted(self._masks))

    def public_manifest(self) -> dict[str, JsonValue]:
        """Return evidence bookkeeping only, never arrays, calibration, clouds, or OBBs."""

        return {
            "frames": [
                self._frames[key].public_metadata() for key in sorted(self._frames)
            ],
            "hypotheses": [
                {
                    "candidate_id": item.candidate_id,
                    "selector_id": item.selector_id,
                    "frame_id": item.frame_id,
                    "confidence": item.confidence,
                    "evidence_path": item.evidence_path,
                    "source": item.source,
                }
                for item in (self._hypotheses[key] for key in sorted(self._hypotheses))
            ],
            "masks": [
                {
                    "candidate_id": item.candidate_id,
                    "selector_id": item.selector_id,
                    "hypothesis_id": item.hypothesis_id,
                    "frame_id": item.frame_id,
                    "confidence": item.confidence,
                    "depth_coverage": item.depth_coverage,
                    "evidence_path": item.evidence_path,
                    "method": item.method,
                }
                for item in (self._masks[key] for key in sorted(self._masks))
            ],
            "geometries": [
                {
                    "geometry_id": item.geometry_id,
                    "mask_id": item.mask_id,
                    "role": item.role.value,
                    "point_count": len(item.points),
                    "depth_coverage": item.depth_coverage,
                }
                for item in (self._geometries[key] for key in sorted(self._geometries))
            ],
            "privacy_contract": (
                "RGB/depth arrays, calibration, pixel masks, point clouds, OBB coordinates, "
                "and backend handles are private and omitted."
            ),
        }


def unproject_masked_depth(
    *,
    depth_m: Sequence[Sequence[float]],
    mask: Sequence[Sequence[bool]],
    intrinsics: Sequence[Sequence[float]],
    camera_to_base: RigidTransform,
    stride: int = 1,
    near_m: float = 0.02,
    far_m: float = 3.0,
) -> tuple[tuple[float, float, float], ...]:
    """Unproject selected metric-depth pixels into the calibrated base frame."""

    if isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0:
        raise SensorPerceptionError("stride must be a positive integer")
    near = _require_finite(near_m, context="near_m")
    far = _require_finite(far_m, context="far_m")
    if near < 0.0 or far <= near:
        raise SensorPerceptionError("depth bounds must satisfy 0 <= near < far")
    height = len(depth_m)
    if height == 0 or len(mask) != height:
        raise SensorPerceptionError("depth and mask must have the same non-empty shape")
    width = len(depth_m[0])
    if width == 0:
        raise SensorPerceptionError("depth rows cannot be empty")
    if any(len(row) != width for row in depth_m) or any(len(row) != width for row in mask):
        raise SensorPerceptionError("depth and mask rows must have a consistent shape")
    if len(intrinsics) != 3 or any(len(row) != 3 for row in intrinsics):
        raise SensorPerceptionError("intrinsics must be 3x3")
    fx = _require_finite(intrinsics[0][0], context="fx")
    fy = _require_finite(intrinsics[1][1], context="fy")
    cx = _require_finite(intrinsics[0][2], context="cx")
    cy = _require_finite(intrinsics[1][2], context="cy")
    if fx <= 0.0 or fy <= 0.0:
        raise SensorPerceptionError("focal lengths must be positive")

    points: list[tuple[float, float, float]] = []
    for row in range(0, height, stride):
        for column in range(0, width, stride):
            if not bool(mask[row][column]):
                continue
            try:
                z = float(depth_m[row][column])
            except (TypeError, ValueError) as exc:
                raise SensorPerceptionError("depth values must be numeric") from exc
            if not math.isfinite(z) or z < near or z > far:
                continue
            camera_point = ((column - cx) * z / fx, (row - cy) * z / fy, z)
            points.append(camera_to_base.apply(camera_point))
    return tuple(points)


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = tuple(sorted(values))
    if not ordered:
        raise SensorPerceptionError("cannot compute a quantile of no values")
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _robust_bounds(
    values: Sequence[float],
    *,
    trim_quantile: float,
    minimum_trim_points: int,
) -> tuple[float, float]:
    if len(values) < minimum_trim_points or trim_quantile == 0.0:
        return min(values), max(values)
    return _quantile(values, trim_quantile), _quantile(values, 1.0 - trim_quantile)


def fit_oriented_bounding_box(
    points: Sequence[Sequence[float]],
    *,
    trim_quantile: float = 0.02,
    minimum_trim_points: int = 20,
    minimum_extent_m: float = 0.004,
    maximum_extent_m: float = 1.5,
) -> OrientedBoundingBox:
    """Fit a deterministic, quantile-robust yaw OBB in the calibrated base frame."""

    checked = tuple(
        tuple(_require_finite(value, context="point cloud") for value in point)
        for point in points
    )
    if len(checked) < 3 or any(len(point) != 3 for point in checked):
        raise SensorPerceptionError("at least three 3-D points are required for an OBB")
    trim_quantile = _require_finite(trim_quantile, context="OBB trim_quantile")
    minimum_extent_m = _require_finite(minimum_extent_m, context="OBB minimum_extent_m")
    maximum_extent_m = _require_finite(maximum_extent_m, context="OBB maximum_extent_m")
    if (
        not 0.0 <= trim_quantile < 0.5
        or isinstance(minimum_trim_points, bool)
        or not isinstance(minimum_trim_points, int)
        or minimum_trim_points < 3
        or not 0.0 < minimum_extent_m <= maximum_extent_m
    ):
        raise SensorPerceptionError("OBB robustness settings are invalid")
    depth_values = tuple(point[2] for point in checked)
    depth_low, depth_high = _robust_bounds(
        depth_values,
        trim_quantile=trim_quantile,
        minimum_trim_points=minimum_trim_points,
    )
    depth_trimmed = tuple(point for point in checked if depth_low <= point[2] <= depth_high)
    if len(depth_trimmed) >= 3:
        checked = depth_trimmed
    count = float(len(checked))
    mean_x = sum(point[0] for point in checked) / count
    mean_y = sum(point[1] for point in checked) / count
    covariance_xx = sum((point[0] - mean_x) ** 2 for point in checked) / count
    covariance_xy = sum(
        (point[0] - mean_x) * (point[1] - mean_y) for point in checked
    ) / count
    covariance_yy = sum((point[1] - mean_y) ** 2 for point in checked) / count
    if covariance_xx + covariance_yy <= 1e-16:
        yaw = 0.0
    else:
        yaw = 0.5 * math.atan2(
            2.0 * covariance_xy,
            covariance_xx - covariance_yy,
        )
    axis_x = (math.cos(yaw), math.sin(yaw), 0.0)
    axis_y = (-math.sin(yaw), math.cos(yaw), 0.0)
    axis_z = (0.0, 0.0, 1.0)
    projected_x = tuple(point[0] * axis_x[0] + point[1] * axis_x[1] for point in checked)
    projected_y = tuple(point[0] * axis_y[0] + point[1] * axis_y[1] for point in checked)
    projected_z = tuple(point[2] for point in checked)
    bounds = tuple(
        _robust_bounds(
            values,
            trim_quantile=trim_quantile,
            minimum_trim_points=minimum_trim_points,
        )
        for values in (projected_x, projected_y, projected_z)
    )
    mid_x, mid_y, mid_z = tuple((low + high) / 2.0 for low, high in bounds)
    center = (
        mid_x * axis_x[0] + mid_y * axis_y[0],
        mid_x * axis_x[1] + mid_y * axis_y[1],
        mid_z,
    )
    extents = tuple(max(high - low, minimum_extent_m) for low, high in bounds)
    if any(extent > maximum_extent_m for extent in extents):
        raise SensorPerceptionError("OBB extent exceeds the configured workspace sanity bound")
    half_extents = tuple(extent / 2.0 for extent in extents)
    return OrientedBoundingBox(
        center=center,
        axes=(axis_x, axis_y, axis_z),
        half_extents=half_extents,  # type: ignore[arg-type]
    )


@dataclass(frozen=True, slots=True)
class PerceptionFailure:
    kind: PerceptionFailureKind
    detail: str
    failed_phase: SensorActorPhase

    def __post_init__(self) -> None:
        for name, enum_type in (
            ("kind", PerceptionFailureKind),
            ("failed_phase", SensorActorPhase),
        ):
            value = getattr(self, name)
            if not isinstance(value, enum_type):
                try:
                    object.__setattr__(self, name, enum_type(value))
                except (TypeError, ValueError) as exc:
                    raise SensorPerceptionError(f"unsupported {name}") from exc
        if not isinstance(self.detail, str) or not self.detail.strip() or "\x00" in self.detail:
            raise SensorPerceptionError("perception failure detail must be non-empty text")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PerceptionFailure":
        if set(value) != {"kind", "detail", "failed_phase"}:
            raise SensorPerceptionError("perception failure has invalid keys")
        return cls(
            kind=value["kind"],
            detail=value["detail"],
            failed_phase=value["failed_phase"],
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "kind": self.kind.value,
            "detail": self.detail,
            "failed_phase": self.failed_phase.value,
        }


@dataclass(frozen=True, slots=True)
class RecoveryBudget:
    retake_same_profile: int = 2
    switch_profile: int = 2
    resegment: int = 2
    reselect_target: int = 2
    reselect_destination: int = 2

    def __post_init__(self) -> None:
        for name in (
            "retake_same_profile",
            "switch_profile",
            "resegment",
            "reselect_target",
            "reselect_destination",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SensorPerceptionError("recovery budgets must be non-negative integers")

    def limit(self, action: RecoveryActionId) -> int | None:
        mapping = {
            RecoveryActionId.RETAKE_SAME_PROFILE: self.retake_same_profile,
            RecoveryActionId.SWITCH_PROFILE: self.switch_profile,
            RecoveryActionId.RESEGMENT: self.resegment,
            RecoveryActionId.RESELECT_TARGET: self.reselect_target,
            RecoveryActionId.RESELECT_DESTINATION: self.reselect_destination,
            RecoveryActionId.STOP_SAFELY: None,
        }
        return mapping[action]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecoveryBudget":
        fields = {
            "retake_same_profile",
            "switch_profile",
            "resegment",
            "reselect_target",
            "reselect_destination",
        }
        if set(value) != fields:
            raise SensorPerceptionError("recovery budget has invalid keys")
        return cls(**{name: value[name] for name in fields})

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "retake_same_profile": self.retake_same_profile,
            "switch_profile": self.switch_profile,
            "resegment": self.resegment,
            "reselect_target": self.reselect_target,
            "reselect_destination": self.reselect_destination,
        }


_PHASE_STAGE: Mapping[SensorActorPhase, CandidateStage] = MappingProxyType(
    {
        SensorActorPhase.CHOOSE_OBSERVATION_TOOL: CandidateStage.TOOL,
        SensorActorPhase.CHOOSE_OBSERVATION_PROFILE: CandidateStage.OBSERVATION_PROFILE,
        SensorActorPhase.SELECT_TARGET_HYPOTHESIS: CandidateStage.HYPOTHESIS,
        SensorActorPhase.SELECT_TARGET_MASK: CandidateStage.MASK,
        SensorActorPhase.SELECT_DESTINATION_HYPOTHESIS: CandidateStage.HYPOTHESIS,
        SensorActorPhase.SELECT_DESTINATION_MASK: CandidateStage.MASK,
        SensorActorPhase.RECOVERY: CandidateStage.RECOVERY,
    }
)


@dataclass(frozen=True, slots=True)
class SensorActorState:
    """Replayable perception state whose only untrusted input is one finite ID."""

    session_id: str
    revision: int = 0
    capture_revision: int = 0
    phase: SensorActorPhase = SensorActorPhase.CHOOSE_OBSERVATION_TOOL
    selected_tool_id: ObservationToolId | None = None
    selected_profile_id: ObservationProfileId | None = None
    selected_target_hypothesis_id: str | None = None
    selected_target_mask_id: str | None = None
    selected_destination_hypothesis_id: str | None = None
    selected_destination_mask_id: str | None = None
    target_geometry_id: str | None = None
    destination_geometry_id: str | None = None
    last_failure: PerceptionFailure | None = None
    recovery_counts: tuple[tuple[RecoveryActionId, int], ...] = ()
    recovery_budget: RecoveryBudget = field(default_factory=RecoveryBudget)
    terminal_status: SensorTerminalStatus = SensorTerminalStatus.RUNNING
    selections: tuple[StageSelection, ...] = ()

    def __post_init__(self) -> None:
        _require_id(self.session_id, context="sensor session_id")
        for name in ("revision", "capture_revision"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SensorPerceptionError(f"{name} must be a non-negative integer")
        for name, enum_type in (
            ("phase", SensorActorPhase),
            ("selected_tool_id", ObservationToolId),
            ("selected_profile_id", ObservationProfileId),
            ("terminal_status", SensorTerminalStatus),
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, enum_type):
                try:
                    object.__setattr__(self, name, enum_type(value))
                except (TypeError, ValueError) as exc:
                    raise SensorPerceptionError(f"unsupported {name}") from exc
        for name in (
            "selected_target_hypothesis_id",
            "selected_target_mask_id",
            "selected_destination_hypothesis_id",
            "selected_destination_mask_id",
            "target_geometry_id",
            "destination_geometry_id",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_id(value, context=name)
        if self.last_failure is not None and not isinstance(
            self.last_failure, PerceptionFailure
        ):
            raise SensorPerceptionError("last_failure must be PerceptionFailure")
        if not isinstance(self.recovery_budget, RecoveryBudget):
            raise SensorPerceptionError("recovery_budget must be RecoveryBudget")
        counts: list[tuple[RecoveryActionId, int]] = []
        seen: set[RecoveryActionId] = set()
        for action, count in self.recovery_counts:
            try:
                checked_action = RecoveryActionId(action)
            except (TypeError, ValueError) as exc:
                raise SensorPerceptionError("unsupported recovery action count") from exc
            if checked_action in seen or checked_action is RecoveryActionId.STOP_SAFELY:
                raise SensorPerceptionError("recovery counts must be unique bounded actions")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise SensorPerceptionError("recovery counts must be non-negative integers")
            seen.add(checked_action)
            counts.append((checked_action, count))
        object.__setattr__(self, "recovery_counts", tuple(sorted(counts, key=lambda item: item[0].value)))
        selections = tuple(self.selections)
        if not all(isinstance(selection, StageSelection) for selection in selections):
            raise SensorPerceptionError("sensor selections must be StageSelection values")
        object.__setattr__(self, "selections", selections)
        if self.phase is SensorActorPhase.READY:
            if not self.target_geometry_id or not self.destination_geometry_id:
                raise SensorPerceptionError("ready actor requires target and destination geometry")
            if self.terminal_status is not SensorTerminalStatus.READY:
                raise SensorPerceptionError("ready phase requires ready terminal status")
        elif self.phase is SensorActorPhase.STOPPED:
            if self.terminal_status is not SensorTerminalStatus.STOPPED:
                raise SensorPerceptionError("stopped phase requires stopped status")
        elif self.terminal_status is not SensorTerminalStatus.RUNNING:
            raise SensorPerceptionError("nonterminal phases require running status")
        if self.phase is SensorActorPhase.RECOVERY and self.last_failure is None:
            raise SensorPerceptionError("recovery phase requires a recorded failure")
        profile_required = {
            SensorActorPhase.CAPTURE_PENDING,
            SensorActorPhase.SELECT_TARGET_HYPOTHESIS,
            SensorActorPhase.SELECT_TARGET_MASK,
            SensorActorPhase.TARGET_GEOMETRY_PENDING,
            SensorActorPhase.SELECT_DESTINATION_HYPOTHESIS,
            SensorActorPhase.SELECT_DESTINATION_MASK,
            SensorActorPhase.DESTINATION_GEOMETRY_PENDING,
            SensorActorPhase.READY,
        }
        if self.phase in profile_required and (
            self.selected_tool_id not in {ObservationToolId.CAPTURE_RGBD, ObservationToolId.REOBSERVE}
            or self.selected_profile_id is None
        ):
            raise SensorPerceptionError(
                f"{self.phase.value} requires a selected capture tool and observation profile"
            )
        if self.phase is SensorActorPhase.CHOOSE_OBSERVATION_TOOL and any(
            value is not None
            for value in (
                self.selected_tool_id,
                self.selected_profile_id,
                self.selected_target_hypothesis_id,
                self.selected_target_mask_id,
                self.selected_destination_hypothesis_id,
                self.selected_destination_mask_id,
                self.target_geometry_id,
                self.destination_geometry_id,
            )
        ):
            raise SensorPerceptionError("observation-tool phase must not retain stale selections")
        if self.phase is SensorActorPhase.CHOOSE_OBSERVATION_PROFILE and (
            self.selected_tool_id not in {ObservationToolId.CAPTURE_RGBD, ObservationToolId.REOBSERVE}
            or self.selected_profile_id is not None
            or any(
                value is not None
                for value in (
                    self.selected_target_hypothesis_id,
                    self.selected_target_mask_id,
                    self.selected_destination_hypothesis_id,
                    self.selected_destination_mask_id,
                    self.target_geometry_id,
                    self.destination_geometry_id,
                )
            )
        ):
            raise SensorPerceptionError("observation-profile phase has inconsistent selections")
        if self.phase in {
            SensorActorPhase.SELECT_TARGET_MASK,
            SensorActorPhase.TARGET_GEOMETRY_PENDING,
            SensorActorPhase.SELECT_DESTINATION_HYPOTHESIS,
            SensorActorPhase.SELECT_DESTINATION_MASK,
            SensorActorPhase.DESTINATION_GEOMETRY_PENDING,
            SensorActorPhase.READY,
        } and self.selected_target_hypothesis_id is None:
            raise SensorPerceptionError(f"{self.phase.value} requires a target hypothesis")
        if self.phase in {
            SensorActorPhase.TARGET_GEOMETRY_PENDING,
            SensorActorPhase.SELECT_DESTINATION_HYPOTHESIS,
            SensorActorPhase.SELECT_DESTINATION_MASK,
            SensorActorPhase.DESTINATION_GEOMETRY_PENDING,
            SensorActorPhase.READY,
        } and self.selected_target_mask_id is None:
            raise SensorPerceptionError(f"{self.phase.value} requires a target mask")
        if self.phase in {
            SensorActorPhase.SELECT_DESTINATION_HYPOTHESIS,
            SensorActorPhase.SELECT_DESTINATION_MASK,
            SensorActorPhase.DESTINATION_GEOMETRY_PENDING,
            SensorActorPhase.READY,
        } and self.target_geometry_id is None:
            raise SensorPerceptionError(f"{self.phase.value} requires target geometry")
        if self.phase in {
            SensorActorPhase.SELECT_DESTINATION_MASK,
            SensorActorPhase.DESTINATION_GEOMETRY_PENDING,
            SensorActorPhase.READY,
        } and self.selected_destination_hypothesis_id is None:
            raise SensorPerceptionError(f"{self.phase.value} requires a destination hypothesis")
        if self.phase in {
            SensorActorPhase.DESTINATION_GEOMETRY_PENDING,
            SensorActorPhase.READY,
        } and self.selected_destination_mask_id is None:
            raise SensorPerceptionError(f"{self.phase.value} requires a destination mask")

    @property
    def expected_stage(self) -> CandidateStage | None:
        return _PHASE_STAGE.get(self.phase)

    @property
    def expected_parent_candidate_id(self) -> str | None:
        if self.phase in {
            SensorActorPhase.CHOOSE_OBSERVATION_TOOL,
            SensorActorPhase.CHOOSE_OBSERVATION_PROFILE,
            SensorActorPhase.RECOVERY,
        }:
            return None
        if self.phase is SensorActorPhase.SELECT_TARGET_HYPOTHESIS:
            return self.selected_profile_candidate_id
        if self.phase is SensorActorPhase.SELECT_TARGET_MASK:
            return self.selected_target_hypothesis_id
        if self.phase is SensorActorPhase.SELECT_DESTINATION_HYPOTHESIS:
            return self.target_geometry_id
        if self.phase is SensorActorPhase.SELECT_DESTINATION_MASK:
            return self.selected_destination_hypothesis_id
        return None

    @property
    def selected_profile_candidate_id(self) -> str | None:
        return next(
            (
                selection.candidate_id
                for selection in reversed(self.selections)
                if selection.stage is CandidateStage.OBSERVATION_PROFILE
            ),
            None,
        )

    def recovery_count(self, action: RecoveryActionId) -> int:
        return dict(self.recovery_counts).get(action, 0)

    def allowed_observation_tools(self) -> tuple[ObservationToolId, ...]:
        if self.phase is not SensorActorPhase.CHOOSE_OBSERVATION_TOOL:
            return ()
        tools = [ObservationToolId.CAPTURE_RGBD]
        if self.capture_revision > 0:
            tools.append(ObservationToolId.REOBSERVE)
        tools.append(ObservationToolId.STOP_SAFELY)
        return tuple(tools)

    def allowed_recovery_actions(self) -> tuple[RecoveryActionId, ...]:
        if self.phase is not SensorActorPhase.RECOVERY or self.last_failure is None:
            return ()
        kind = self.last_failure.kind
        actions: list[RecoveryActionId] = []
        if self.selected_profile_id is not None:
            actions.append(RecoveryActionId.RETAKE_SAME_PROFILE)
        actions.append(RecoveryActionId.SWITCH_PROFILE)
        if kind not in {PerceptionFailureKind.CAPTURE_FAILED, PerceptionFailureKind.NO_HYPOTHESES}:
            actions.append(RecoveryActionId.RESEGMENT)
        if self.selected_target_hypothesis_id is not None:
            actions.append(RecoveryActionId.RESELECT_TARGET)
        if self.selected_destination_hypothesis_id is not None:
            actions.append(RecoveryActionId.RESELECT_DESTINATION)
        bounded = []
        for action in actions:
            limit = self.recovery_budget.limit(action)
            if limit is None or self.recovery_count(action) < limit:
                bounded.append(action)
        bounded.append(RecoveryActionId.STOP_SAFELY)
        return tuple(dict.fromkeys(bounded))

    def validate_batch(self, batch: CandidateBatch) -> None:
        expected = self.expected_stage
        if expected is None:
            raise CandidateSelectionError(
                f"sensor actor phase {self.phase.value} does not accept a selector decision"
            )
        if batch.stage is not expected:
            raise CandidateSelectionError(
                f"expected {expected.value}, got {batch.stage.value}"
            )
        if batch.parent_candidate_id != self.expected_parent_candidate_id:
            raise CandidateSelectionError("sensor candidate batch has stale parent context")
        if batch.stage is CandidateStage.TOOL:
            allowed = {item.value for item in self.allowed_observation_tools()}
            if any(item.stable_key not in allowed for item in batch.eligible_candidates):
                raise CandidateSelectionError("observation tool precondition is not satisfied")
        if batch.stage is CandidateStage.RECOVERY:
            allowed = {item.value for item in self.allowed_recovery_actions()}
            if any(item.stable_key not in allowed for item in batch.eligible_candidates):
                raise CandidateSelectionError("recovery action is unavailable or exhausted")

    def choose(
        self,
        batch: CandidateBatch,
        choice: CandidateChoice,
    ) -> tuple[CandidateDecision, "SensorActorState"]:
        self.validate_batch(batch)
        decision = CandidateDecision.from_choice(batch, choice)
        candidate = batch.resolve(decision.candidate_id)
        history = self.selections + (
            StageSelection(stage=batch.stage, candidate_id=candidate.candidate_id),
        )
        common = {"revision": self.revision + 1, "selections": history}
        if batch.stage is CandidateStage.TOOL:
            tool = ObservationToolId(candidate.stable_key)
            if tool is ObservationToolId.STOP_SAFELY:
                return decision, replace(
                    self,
                    selected_tool_id=tool,
                    phase=SensorActorPhase.STOPPED,
                    terminal_status=SensorTerminalStatus.STOPPED,
                    **common,
                )
            return decision, replace(
                self,
                selected_tool_id=tool,
                phase=SensorActorPhase.CHOOSE_OBSERVATION_PROFILE,
                **common,
            )
        if batch.stage is CandidateStage.OBSERVATION_PROFILE:
            return decision, replace(
                self,
                selected_profile_id=ObservationProfileId(candidate.stable_key),
                phase=SensorActorPhase.CAPTURE_PENDING,
                last_failure=None,
                **common,
            )
        if batch.stage is CandidateStage.HYPOTHESIS:
            if self.phase is SensorActorPhase.SELECT_TARGET_HYPOTHESIS:
                return decision, replace(
                    self,
                    selected_target_hypothesis_id=candidate.candidate_id,
                    selected_target_mask_id=None,
                    target_geometry_id=None,
                    selected_destination_hypothesis_id=None,
                    selected_destination_mask_id=None,
                    destination_geometry_id=None,
                    phase=SensorActorPhase.SELECT_TARGET_MASK,
                    **common,
                )
            return decision, replace(
                self,
                selected_destination_hypothesis_id=candidate.candidate_id,
                selected_destination_mask_id=None,
                destination_geometry_id=None,
                phase=SensorActorPhase.SELECT_DESTINATION_MASK,
                **common,
            )
        if batch.stage is CandidateStage.MASK:
            if self.phase is SensorActorPhase.SELECT_TARGET_MASK:
                return decision, replace(
                    self,
                    selected_target_mask_id=candidate.candidate_id,
                    target_geometry_id=None,
                    phase=SensorActorPhase.TARGET_GEOMETRY_PENDING,
                    **common,
                )
            return decision, replace(
                self,
                selected_destination_mask_id=candidate.candidate_id,
                destination_geometry_id=None,
                phase=SensorActorPhase.DESTINATION_GEOMETRY_PENDING,
                **common,
            )
        if batch.stage is CandidateStage.RECOVERY:
            return decision, self._apply_recovery(
                RecoveryActionId(candidate.stable_key),
                revision=self.revision + 1,
                selections=history,
            )
        raise AssertionError(f"unsupported sensor selection stage: {batch.stage.value}")

    def complete_capture(self, *, succeeded: bool, hypothesis_count: int) -> "SensorActorState":
        if self.phase is not SensorActorPhase.CAPTURE_PENDING:
            raise CandidateSelectionError("RGB-D capture is not pending")
        if not isinstance(succeeded, bool):
            raise SensorPerceptionError("capture success must be boolean")
        if isinstance(hypothesis_count, bool) or not isinstance(hypothesis_count, int) or hypothesis_count < 0:
            raise SensorPerceptionError("hypothesis_count must be a non-negative integer")
        if not succeeded:
            return self.record_failure(
                PerceptionFailureKind.CAPTURE_FAILED,
                "RGB-D capture wrapper failed",
            )
        if hypothesis_count == 0:
            return self.record_failure(
                PerceptionFailureKind.NO_HYPOTHESES,
                "RGB-D proposal backend returned no visual hypotheses",
            )
        return replace(
            self,
            revision=self.revision + 1,
            capture_revision=self.capture_revision + 1,
            phase=SensorActorPhase.SELECT_TARGET_HYPOTHESIS,
            selected_target_hypothesis_id=None,
            selected_target_mask_id=None,
            selected_destination_hypothesis_id=None,
            selected_destination_mask_id=None,
            target_geometry_id=None,
            destination_geometry_id=None,
            last_failure=None,
        )

    def register_geometry(
        self,
        *,
        role: PerceptionRole,
        geometry: GeometryPayload,
    ) -> "SensorActorState":
        role = PerceptionRole(role)
        if not isinstance(geometry, GeometryPayload):
            raise SensorPerceptionError("geometry registration requires GeometryPayload")
        if geometry.role is not role:
            raise SensorPerceptionError("geometry role does not match the state transition")
        if role is PerceptionRole.TARGET:
            if self.phase is not SensorActorPhase.TARGET_GEOMETRY_PENDING:
                raise CandidateSelectionError("target geometry is not pending")
            if geometry.mask_id != self.selected_target_mask_id:
                raise SensorPerceptionError("target geometry does not match the selected mask")
            return replace(
                self,
                revision=self.revision + 1,
                target_geometry_id=geometry.geometry_id,
                phase=SensorActorPhase.SELECT_DESTINATION_HYPOTHESIS,
            )
        if self.phase is not SensorActorPhase.DESTINATION_GEOMETRY_PENDING:
            raise CandidateSelectionError("destination geometry is not pending")
        if geometry.mask_id != self.selected_destination_mask_id:
            raise SensorPerceptionError("destination geometry does not match the selected mask")
        return replace(
            self,
            revision=self.revision + 1,
            destination_geometry_id=geometry.geometry_id,
            phase=SensorActorPhase.READY,
            terminal_status=SensorTerminalStatus.READY,
            last_failure=None,
        )

    def record_failure(
        self,
        kind: PerceptionFailureKind,
        detail: str,
    ) -> "SensorActorState":
        if self.phase in {SensorActorPhase.READY, SensorActorPhase.STOPPED}:
            raise CandidateSelectionError("cannot fail a terminal sensor state")
        failure = PerceptionFailure(kind=kind, detail=detail, failed_phase=self.phase)
        return replace(
            self,
            revision=self.revision + 1,
            phase=SensorActorPhase.RECOVERY,
            last_failure=failure,
        )

    def request_refresh(self, *, detail: str) -> "SensorActorState":
        if self.phase is not SensorActorPhase.READY:
            raise CandidateSelectionError("only ready geometry can be marked stale")
        if not isinstance(detail, str) or not detail.strip():
            raise SensorPerceptionError("refresh detail must be non-empty")
        return replace(
            self,
            revision=self.revision + 1,
            phase=SensorActorPhase.CHOOSE_OBSERVATION_TOOL,
            terminal_status=SensorTerminalStatus.RUNNING,
            selected_tool_id=None,
            selected_profile_id=None,
            selected_target_hypothesis_id=None,
            selected_target_mask_id=None,
            selected_destination_hypothesis_id=None,
            selected_destination_mask_id=None,
            target_geometry_id=None,
            destination_geometry_id=None,
            last_failure=None,
        )

    def _apply_recovery(
        self,
        action: RecoveryActionId,
        *,
        revision: int,
        selections: tuple[StageSelection, ...],
    ) -> "SensorActorState":
        if action not in self.allowed_recovery_actions():
            raise CandidateSelectionError(f"recovery action unavailable: {action.value}")
        if action is RecoveryActionId.STOP_SAFELY:
            return replace(
                self,
                revision=revision,
                selections=selections,
                phase=SensorActorPhase.STOPPED,
                terminal_status=SensorTerminalStatus.STOPPED,
            )
        counts = dict(self.recovery_counts)
        counts[action] = counts.get(action, 0) + 1
        common = {
            "revision": revision,
            "selections": selections,
            "recovery_counts": tuple(counts.items()),
            "last_failure": None,
        }
        if action is RecoveryActionId.RETAKE_SAME_PROFILE:
            return replace(self, phase=SensorActorPhase.CAPTURE_PENDING, **common)
        if action is RecoveryActionId.SWITCH_PROFILE:
            return replace(
                self,
                selected_profile_id=None,
                selected_target_hypothesis_id=None,
                selected_target_mask_id=None,
                selected_destination_hypothesis_id=None,
                selected_destination_mask_id=None,
                target_geometry_id=None,
                destination_geometry_id=None,
                phase=SensorActorPhase.CHOOSE_OBSERVATION_PROFILE,
                **common,
            )
        if action is RecoveryActionId.RESELECT_DESTINATION:
            return replace(
                self,
                selected_destination_hypothesis_id=None,
                selected_destination_mask_id=None,
                destination_geometry_id=None,
                phase=SensorActorPhase.SELECT_DESTINATION_HYPOTHESIS,
                **common,
            )
        if action is RecoveryActionId.RESELECT_TARGET:
            return replace(
                self,
                selected_target_hypothesis_id=None,
                selected_target_mask_id=None,
                target_geometry_id=None,
                selected_destination_hypothesis_id=None,
                selected_destination_mask_id=None,
                destination_geometry_id=None,
                phase=SensorActorPhase.SELECT_TARGET_HYPOTHESIS,
                **common,
            )
        assert action is RecoveryActionId.RESEGMENT
        failed_phase = self.last_failure.failed_phase if self.last_failure else None
        destination_failure = failed_phase in {
            SensorActorPhase.SELECT_DESTINATION_HYPOTHESIS,
            SensorActorPhase.SELECT_DESTINATION_MASK,
            SensorActorPhase.DESTINATION_GEOMETRY_PENDING,
        }
        return replace(
            self,
            selected_destination_hypothesis_id=(
                None if destination_failure else self.selected_destination_hypothesis_id
            ),
            selected_destination_mask_id=(
                None if destination_failure else self.selected_destination_mask_id
            ),
            destination_geometry_id=(None if destination_failure else self.destination_geometry_id),
            selected_target_hypothesis_id=(
                self.selected_target_hypothesis_id if destination_failure else None
            ),
            selected_target_mask_id=(self.selected_target_mask_id if destination_failure else None),
            target_geometry_id=(self.target_geometry_id if destination_failure else None),
            phase=(
                SensorActorPhase.SELECT_DESTINATION_HYPOTHESIS
                if destination_failure
                else SensorActorPhase.SELECT_TARGET_HYPOTHESIS
            ),
            **common,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SensorActorState":
        fields = {
            "session_id",
            "revision",
            "capture_revision",
            "phase",
            "selected_tool_id",
            "selected_profile_id",
            "selected_target_hypothesis_id",
            "selected_target_mask_id",
            "selected_destination_hypothesis_id",
            "selected_destination_mask_id",
            "target_geometry_id",
            "destination_geometry_id",
            "last_failure",
            "recovery_counts",
            "recovery_budget",
            "terminal_status",
            "selections",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise SensorPerceptionError("sensor actor state has invalid keys")
        raw_failure = value["last_failure"]
        raw_counts = value["recovery_counts"]
        raw_selections = value["selections"]
        if not isinstance(raw_counts, list) or not isinstance(raw_selections, list):
            raise SensorPerceptionError("state recovery_counts/selections must be arrays")
        return cls(
            session_id=value["session_id"],
            revision=value["revision"],
            capture_revision=value["capture_revision"],
            phase=value["phase"],
            selected_tool_id=value["selected_tool_id"],
            selected_profile_id=value["selected_profile_id"],
            selected_target_hypothesis_id=value["selected_target_hypothesis_id"],
            selected_target_mask_id=value["selected_target_mask_id"],
            selected_destination_hypothesis_id=value["selected_destination_hypothesis_id"],
            selected_destination_mask_id=value["selected_destination_mask_id"],
            target_geometry_id=value["target_geometry_id"],
            destination_geometry_id=value["destination_geometry_id"],
            last_failure=(
                None if raw_failure is None else PerceptionFailure.from_dict(raw_failure)
            ),
            recovery_counts=tuple((RecoveryActionId(item[0]), item[1]) for item in raw_counts),
            recovery_budget=RecoveryBudget.from_dict(value["recovery_budget"]),
            terminal_status=value["terminal_status"],
            selections=tuple(StageSelection.from_dict(item) for item in raw_selections),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "session_id": self.session_id,
            "revision": self.revision,
            "capture_revision": self.capture_revision,
            "phase": self.phase.value,
            "selected_tool_id": self.selected_tool_id.value if self.selected_tool_id else None,
            "selected_profile_id": (
                self.selected_profile_id.value if self.selected_profile_id else None
            ),
            "selected_target_hypothesis_id": self.selected_target_hypothesis_id,
            "selected_target_mask_id": self.selected_target_mask_id,
            "selected_destination_hypothesis_id": self.selected_destination_hypothesis_id,
            "selected_destination_mask_id": self.selected_destination_mask_id,
            "target_geometry_id": self.target_geometry_id,
            "destination_geometry_id": self.destination_geometry_id,
            "last_failure": self.last_failure.to_dict() if self.last_failure else None,
            "recovery_counts": [
                [action.value, count] for action, count in self.recovery_counts
            ],
            "recovery_budget": self.recovery_budget.to_dict(),
            "terminal_status": self.terminal_status.value,
            "selections": [selection.to_dict() for selection in self.selections],
        }


def make_hypothesis_payload(
    *,
    state: SensorActorState,
    selector_id: str,
    frame_id: str,
    box: PixelBox,
    seed_mask: Any,
    confidence: float,
    evidence_path: str,
    source: str = "rgbd_depth_component",
    semantic_hint: str | None = None,
) -> VisualHypothesisPayload:
    if state.expected_stage is not CandidateStage.HYPOTHESIS:
        raise CandidateSelectionError("actor does not currently expect hypotheses")
    candidate = Candidate.create(
        stage=CandidateStage.HYPOTHESIS,
        stable_key=selector_id,
        parent_candidate_id=state.expected_parent_candidate_id,
        label=f"private hypothesis payload {selector_id}",
        metrics=CandidateMetrics(confidence=confidence),
        evidence=(EvidenceRef.image(evidence_path, view="hypothesis", label=selector_id),),
    )
    return VisualHypothesisPayload(
        candidate_id=candidate.candidate_id,
        selector_id=selector_id,
        frame_id=frame_id,
        box=box,
        seed_mask=seed_mask,
        confidence=confidence,
        evidence_path=evidence_path,
        source=source,
        semantic_hint=semantic_hint,
    )


def make_mask_payload(
    *,
    state: SensorActorState,
    selector_id: str,
    hypothesis: VisualHypothesisPayload,
    mask: Any,
    confidence: float,
    depth_coverage: float,
    evidence_path: str,
    method: str,
    artifact_path: str | None = None,
    artifact_sha256: str | None = None,
) -> MaskPayload:
    if state.expected_stage is not CandidateStage.MASK:
        raise CandidateSelectionError("actor does not currently expect masks")
    if state.expected_parent_candidate_id != hypothesis.candidate_id:
        raise CandidateSelectionError("mask hypothesis is not the current selection")
    candidate = Candidate.create(
        stage=CandidateStage.MASK,
        stable_key=selector_id,
        parent_candidate_id=hypothesis.candidate_id,
        label=f"private mask payload {selector_id}",
        metrics=CandidateMetrics(confidence=confidence),
        evidence=(EvidenceRef.image(evidence_path, view="mask", label=selector_id),),
    )
    return MaskPayload(
        candidate_id=candidate.candidate_id,
        selector_id=selector_id,
        hypothesis_id=hypothesis.candidate_id,
        frame_id=hypothesis.frame_id,
        mask=mask,
        confidence=confidence,
        depth_coverage=depth_coverage,
        evidence_path=evidence_path,
        method=method,
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256,
    )
