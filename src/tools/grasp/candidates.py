"""Strict finite candidate selection for visually grounded robot decisions.

The planner owns geometry, trajectories, and safety checks.  This module only
exposes immutable candidate metadata and permits an untrusted selector to
return one opaque ``candidate_id`` from a finite, pre-validated set.
"""


from __future__ import annotations


from dataclasses import dataclass, field, replace


from enum import Enum


import hashlib


import json


import math


import re


from typing import Any, Mapping


from src.core.models import JsonValue, ValidationError


class CandidateValidationError(ValidationError):
    """Raised when candidate data violates the finite-selection contract."""


class CandidateSelectionError(RuntimeError):
    """Raised when a proposed choice is absent, unsafe, or out of sequence."""


_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


_STABLE_KEY_RE = re.compile(r"^[^\x00-\x1f\x7f]{1,256}$")


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    context: str,
) -> None:
    missing = required - set(value)
    extra = set(value) - required
    if missing:
        raise CandidateValidationError(f"{context} is missing keys: {sorted(missing)}")
    if extra:
        raise CandidateValidationError(f"{context} has unknown keys: {sorted(extra)}")


def _require_id(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise CandidateValidationError(f"{context} must be a valid identifier")
    return value


def _require_text(value: Any, *, context: str, maximum: int = 1024) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value) > maximum
    ):
        raise CandidateValidationError(f"{context} must be non-empty text")
    return value


def _optional_number(
    value: Any,
    *,
    context: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateValidationError(f"{context} must be a number or null")
    result = float(value)
    if not math.isfinite(result):
        raise CandidateValidationError(f"{context} must be finite")
    if minimum is not None and result < minimum:
        raise CandidateValidationError(f"{context} must be >= {minimum:g}")
    if maximum is not None and result > maximum:
        raise CandidateValidationError(f"{context} must be <= {maximum:g}")
    return result


class CandidateStage(str, Enum):
    OBSERVATION_PROFILE = "observation_profile_id"
    HYPOTHESIS = "hypothesis_id"
    MASK = "mask_id"
    RECOVERY = "recovery_id"
    OBJECT = "object_id"
    GRASP = "grasp_id"
    TRAJECTORY = "trajectory_id"
    TOOL = "tool_id"
    OUTCOME = "outcome"


class SafetyStatus(str, Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    UNVERIFIED = "unverified"


class FeasibilityStatus(str, Enum):
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNVERIFIED = "unverified"


class EvidenceKind(str, Enum):
    IMAGE = "image"
    REPORT = "report"


def _coerce_enum(enum_type: type[Enum], value: Any, *, context: str) -> Enum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise CandidateValidationError(f"unsupported {context}: {value}") from exc


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    """A replayable visual or planner evidence artifact."""

    kind: EvidenceKind
    path: str
    view: str
    label: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EvidenceKind):
            object.__setattr__(
                self,
                "kind",
                _coerce_enum(EvidenceKind, self.kind, context="evidence kind"),
            )
        _require_text(self.path, context="evidence path", maximum=4096)
        _require_text(self.view, context="evidence view", maximum=128)
        _require_text(self.label, context="evidence label", maximum=512)

    @classmethod
    def image(cls, path: str, *, view: str, label: str) -> "EvidenceRef":
        return cls(kind=EvidenceKind.IMAGE, path=path, view=view, label=label)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceRef":
        if not isinstance(value, Mapping):
            raise CandidateValidationError("evidence must be an object")
        _require_exact_keys(
            value,
            required={"kind", "path", "view", "label"},
            context="evidence",
        )
        return cls(
            kind=value["kind"],
            path=value["path"],
            view=value["view"],
            label=value["label"],
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "kind": self.kind.value,
            "path": self.path,
            "view": self.view,
            "label": self.label,
        }


@dataclass(frozen=True, slots=True)
class CandidateMetrics:
    """Small scalar summaries; never joint arrays, waypoints, or motor commands."""

    quality_score: float | None = None
    confidence: float | None = None
    minimum_clearance_m: float | None = None
    path_length_m: float | None = None
    expected_duration_s: float | None = None
    joint_limit_margin_rad: float | None = None
    collision_free: bool | None = None

    _FIELDS = (
        "quality_score",
        "confidence",
        "minimum_clearance_m",
        "path_length_m",
        "expected_duration_s",
        "joint_limit_margin_rad",
        "collision_free",
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "quality_score",
            _optional_number(
                self.quality_score,
                context="quality_score",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "confidence",
            _optional_number(
                self.confidence,
                context="confidence",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        for name in (
            "minimum_clearance_m",
            "path_length_m",
            "expected_duration_s",
            "joint_limit_margin_rad",
        ):
            object.__setattr__(
                self,
                name,
                _optional_number(getattr(self, name), context=name, minimum=0.0),
            )
        if self.collision_free is not None and not isinstance(self.collision_free, bool):
            raise CandidateValidationError("collision_free must be a boolean or null")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateMetrics":
        if not isinstance(value, Mapping):
            raise CandidateValidationError("candidate metrics must be an object")
        _require_exact_keys(value, required=set(cls._FIELDS), context="candidate metrics")
        return cls(**{name: value[name] for name in cls._FIELDS})

    def to_dict(self, *, omit_null: bool = False) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {
            "quality_score": self.quality_score,
            "confidence": self.confidence,
            "minimum_clearance_m": self.minimum_clearance_m,
            "path_length_m": self.path_length_m,
            "expected_duration_s": self.expected_duration_s,
            "joint_limit_margin_rad": self.joint_limit_margin_rad,
            "collision_free": self.collision_free,
        }
        if omit_null:
            return {key: value for key, value in result.items() if value is not None}
        return result


def deterministic_candidate_id(
    stage: CandidateStage,
    stable_key: str,
    *,
    parent_candidate_id: str | None = None,
) -> str:
    """Derive an opaque stable ID from planner-owned semantic identity."""

    if not isinstance(stage, CandidateStage):
        stage = _coerce_enum(CandidateStage, stage, context="candidate stage")  # type: ignore[assignment]
    if not isinstance(stable_key, str) or not _STABLE_KEY_RE.fullmatch(stable_key):
        raise CandidateValidationError(
            "stable_key must contain 1-256 printable characters"
        )
    if parent_candidate_id is not None:
        _require_id(parent_candidate_id, context="parent_candidate_id")
    canonical = json.dumps(
        {
            "parent_candidate_id": parent_candidate_id,
            "stable_key": stable_key,
            "stage": stage.value,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()[:20]
    return f"{stage.value}:{digest}"


@dataclass(frozen=True, slots=True)
class Candidate:
    """One planner-generated option whose low-level payload stays out of the LLM."""

    candidate_id: str
    stage: CandidateStage
    stable_key: str
    parent_candidate_id: str | None
    label: str
    safety: SafetyStatus
    feasibility: FeasibilityStatus
    status_detail: str
    metrics: CandidateMetrics = field(default_factory=CandidateMetrics)
    evidence: tuple[EvidenceRef, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.stage, CandidateStage):
            object.__setattr__(
                self,
                "stage",
                _coerce_enum(CandidateStage, self.stage, context="candidate stage"),
            )
        if not isinstance(self.safety, SafetyStatus):
            object.__setattr__(
                self,
                "safety",
                _coerce_enum(SafetyStatus, self.safety, context="safety status"),
            )
        if not isinstance(self.feasibility, FeasibilityStatus):
            object.__setattr__(
                self,
                "feasibility",
                _coerce_enum(
                    FeasibilityStatus,
                    self.feasibility,
                    context="feasibility status",
                ),
            )
        _require_id(self.candidate_id, context="candidate_id")
        if not isinstance(self.stable_key, str) or not _STABLE_KEY_RE.fullmatch(
            self.stable_key
        ):
            raise CandidateValidationError(
                "stable_key must contain 1-256 printable characters"
            )
        if self.parent_candidate_id is not None:
            _require_id(self.parent_candidate_id, context="parent_candidate_id")
        expected_id = deterministic_candidate_id(
            self.stage,
            self.stable_key,
            parent_candidate_id=self.parent_candidate_id,
        )
        if self.candidate_id != expected_id:
            raise CandidateValidationError(
                f"candidate_id is not deterministic; expected {expected_id}"
            )
        _require_text(self.label, context="candidate label")
        if not isinstance(self.status_detail, str) or "\x00" in self.status_detail:
            raise CandidateValidationError("status_detail must be text")
        if not self.eligible and not self.status_detail.strip():
            raise CandidateValidationError(
                "filtered candidates require a safety/feasibility explanation"
            )
        if not isinstance(self.metrics, CandidateMetrics):
            raise CandidateValidationError("metrics must be CandidateMetrics")
        evidence = tuple(self.evidence)
        if not all(isinstance(item, EvidenceRef) for item in evidence):
            raise CandidateValidationError("evidence must contain EvidenceRef values")
        if len(set(evidence)) != len(evidence):
            raise CandidateValidationError("candidate evidence must be unique")
        object.__setattr__(self, "evidence", evidence)
        if self.metrics.collision_free is False and self.safety is SafetyStatus.SAFE:
            raise CandidateValidationError(
                "a collision candidate cannot be marked safe"
            )
        if self.stage is CandidateStage.TRAJECTORY and self.eligible:
            if self.metrics.collision_free is not True:
                raise CandidateValidationError(
                    "eligible trajectories require collision_free=true"
                )
        if self.eligible and not any(
            item.kind is EvidenceKind.IMAGE for item in self.evidence
        ):
            raise CandidateValidationError(
                "eligible candidates require at least one visual evidence image"
            )

    @property
    def eligible(self) -> bool:
        return (
            self.safety is SafetyStatus.SAFE
            and self.feasibility is FeasibilityStatus.FEASIBLE
            and self.metrics.collision_free is not False
        )

    @classmethod
    def create(
        cls,
        *,
        stage: CandidateStage,
        stable_key: str,
        label: str,
        parent_candidate_id: str | None = None,
        safety: SafetyStatus = SafetyStatus.SAFE,
        feasibility: FeasibilityStatus = FeasibilityStatus.FEASIBLE,
        status_detail: str = "",
        metrics: CandidateMetrics | None = None,
        evidence: tuple[EvidenceRef, ...] = (),
    ) -> "Candidate":
        return cls(
            candidate_id=deterministic_candidate_id(
                stage,
                stable_key,
                parent_candidate_id=parent_candidate_id,
            ),
            stage=stage,
            stable_key=stable_key,
            parent_candidate_id=parent_candidate_id,
            label=label,
            safety=safety,
            feasibility=feasibility,
            status_detail=status_detail,
            metrics=metrics or CandidateMetrics(),
            evidence=evidence,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Candidate":
        if not isinstance(value, Mapping):
            raise CandidateValidationError("candidate must be an object")
        _require_exact_keys(
            value,
            required={
                "candidate_id",
                "stage",
                "stable_key",
                "parent_candidate_id",
                "label",
                "safety",
                "feasibility",
                "status_detail",
                "metrics",
                "evidence",
            },
            context="candidate",
        )
        raw_evidence = value["evidence"]
        if not isinstance(raw_evidence, list):
            raise CandidateValidationError("candidate evidence must be an array")
        return cls(
            candidate_id=value["candidate_id"],
            stage=value["stage"],
            stable_key=value["stable_key"],
            parent_candidate_id=value["parent_candidate_id"],
            label=value["label"],
            safety=value["safety"],
            feasibility=value["feasibility"],
            status_detail=value["status_detail"],
            metrics=CandidateMetrics.from_dict(value["metrics"]),
            evidence=tuple(EvidenceRef.from_dict(item) for item in raw_evidence),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "candidate_id": self.candidate_id,
            "stage": self.stage.value,
            "stable_key": self.stable_key,
            "parent_candidate_id": self.parent_candidate_id,
            "label": self.label,
            "safety": self.safety.value,
            "feasibility": self.feasibility.value,
            "status_detail": self.status_detail,
            "metrics": self.metrics.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
        }

    def to_prompt_dict(self) -> dict[str, JsonValue]:
        if not self.eligible:
            raise CandidateSelectionError(
                f"filtered candidate cannot be exposed to the selector: {self.candidate_id}"
            )
        return {
            "candidate_id": self.candidate_id,
            "label": self.label,
            "safety": self.safety.value,
            "feasibility": self.feasibility.value,
            "metrics": self.metrics.to_dict(omit_null=True),
            "evidence": [item.to_dict() for item in self.evidence],
        }


def _candidate_batch_id(
    stage: CandidateStage,
    parent_candidate_id: str | None,
    candidates: tuple[Candidate, ...],
) -> str:
    canonical = json.dumps(
        {
            "candidates": [candidate.to_dict() for candidate in candidates],
            "parent_candidate_id": parent_candidate_id,
            "stage": stage.value,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()[:20]
    return f"batch:{stage.value}:{digest}"


@dataclass(frozen=True, slots=True)
class CandidateBatch:
    """All generated candidates, including options filtered before prompting."""

    batch_id: str
    stage: CandidateStage
    parent_candidate_id: str | None
    candidates: tuple[Candidate, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.stage, CandidateStage):
            object.__setattr__(
                self,
                "stage",
                _coerce_enum(CandidateStage, self.stage, context="batch stage"),
            )
        if self.parent_candidate_id is not None:
            _require_id(self.parent_candidate_id, context="batch parent_candidate_id")
        candidates = tuple(sorted(tuple(self.candidates), key=lambda item: item.candidate_id))
        if not candidates:
            raise CandidateValidationError("candidate batch cannot be empty")
        if not all(isinstance(item, Candidate) for item in candidates):
            raise CandidateValidationError("candidate batch contains an invalid value")
        ids = [item.candidate_id for item in candidates]
        if len(set(ids)) != len(ids):
            raise CandidateValidationError("candidate IDs must be unique within a batch")
        for candidate in candidates:
            if candidate.stage is not self.stage:
                raise CandidateValidationError("candidate stage does not match its batch")
            if candidate.parent_candidate_id != self.parent_candidate_id:
                raise CandidateValidationError("candidate parent does not match its batch")
        object.__setattr__(self, "candidates", candidates)
        _require_id(self.batch_id, context="batch_id")
        expected_id = _candidate_batch_id(
            self.stage,
            self.parent_candidate_id,
            candidates,
        )
        if self.batch_id != expected_id:
            raise CandidateValidationError(
                f"batch_id is not deterministic; expected {expected_id}"
            )

    @classmethod
    def create(
        cls,
        *,
        stage: CandidateStage,
        parent_candidate_id: str | None,
        candidates: tuple[Candidate, ...],
    ) -> "CandidateBatch":
        ordered = tuple(sorted(tuple(candidates), key=lambda item: item.candidate_id))
        return cls(
            batch_id=_candidate_batch_id(stage, parent_candidate_id, ordered),
            stage=stage,
            parent_candidate_id=parent_candidate_id,
            candidates=ordered,
        )

    @property
    def eligible_candidates(self) -> tuple[Candidate, ...]:
        return tuple(candidate for candidate in self.candidates if candidate.eligible)

    @property
    def filtered_candidates(self) -> tuple[Candidate, ...]:
        return tuple(candidate for candidate in self.candidates if not candidate.eligible)

    def image_paths(self) -> tuple[str, ...]:
        """Return de-duplicated selector-visible image paths in stable order."""

        return tuple(
            dict.fromkeys(
                evidence.path
                for candidate in self.eligible_candidates
                for evidence in candidate.evidence
                if evidence.kind is EvidenceKind.IMAGE
            )
        )

    def prompt_manifest(self) -> dict[str, JsonValue]:
        eligible = self.eligible_candidates
        if not eligible:
            raise CandidateSelectionError(
                f"batch {self.batch_id} contains no safe feasible candidates"
            )
        return {
            "batch_id": self.batch_id,
            "stage": self.stage.value,
            "parent_candidate_id": self.parent_candidate_id,
            "selection_contract": "Return exactly one candidate_id from candidates.",
            "candidates": [candidate.to_prompt_dict() for candidate in eligible],
        }

    def output_schema(self) -> dict[str, JsonValue]:
        ids = [candidate.candidate_id for candidate in self.eligible_candidates]
        if not ids:
            raise CandidateSelectionError(
                f"batch {self.batch_id} contains no safe feasible candidates"
            )
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "FiniteRobotCandidateChoice",
            "type": "object",
            "additionalProperties": False,
            "required": ["candidate_id"],
            "properties": {
                "candidate_id": {
                    "type": "string",
                    "enum": ids,
                }
            },
        }

    def resolve(self, candidate_id: str) -> Candidate:
        candidate = next(
            (item for item in self.candidates if item.candidate_id == candidate_id),
            None,
        )
        if candidate is None:
            raise CandidateSelectionError(
                f"candidate_id is not in finite batch {self.batch_id}: {candidate_id}"
            )
        if not candidate.eligible:
            raise CandidateSelectionError(
                f"candidate_id was filtered as unsafe or infeasible: {candidate_id}"
            )
        return candidate

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateBatch":
        if not isinstance(value, Mapping):
            raise CandidateValidationError("candidate batch must be an object")
        _require_exact_keys(
            value,
            required={"batch_id", "stage", "parent_candidate_id", "candidates"},
            context="candidate batch",
        )
        raw_candidates = value["candidates"]
        if not isinstance(raw_candidates, list):
            raise CandidateValidationError("batch candidates must be an array")
        return cls(
            batch_id=value["batch_id"],
            stage=value["stage"],
            parent_candidate_id=value["parent_candidate_id"],
            candidates=tuple(Candidate.from_dict(item) for item in raw_candidates),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "batch_id": self.batch_id,
            "stage": self.stage.value,
            "parent_candidate_id": self.parent_candidate_id,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True, slots=True)
class CandidateChoice:
    """The entire untrusted LLM output surface: one opaque finite ID."""

    candidate_id: str

    def __post_init__(self) -> None:
        _require_id(self.candidate_id, context="choice candidate_id")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateChoice":
        if not isinstance(value, Mapping):
            raise CandidateValidationError("candidate choice must be an object")
        _require_exact_keys(
            value,
            required={"candidate_id"},
            context="candidate choice",
        )
        return cls(candidate_id=value["candidate_id"])

    def to_dict(self) -> dict[str, JsonValue]:
        return {"candidate_id": self.candidate_id}


@dataclass(frozen=True, slots=True)
class CandidateDecision:
    batch_id: str
    stage: CandidateStage
    candidate_id: str

    def __post_init__(self) -> None:
        _require_id(self.batch_id, context="decision batch_id")
        if not isinstance(self.stage, CandidateStage):
            object.__setattr__(
                self,
                "stage",
                _coerce_enum(CandidateStage, self.stage, context="decision stage"),
            )
        _require_id(self.candidate_id, context="decision candidate_id")

    @classmethod
    def from_choice(
        cls,
        batch: CandidateBatch,
        choice: CandidateChoice,
    ) -> "CandidateDecision":
        candidate = batch.resolve(choice.candidate_id)
        return cls(
            batch_id=batch.batch_id,
            stage=batch.stage,
            candidate_id=candidate.candidate_id,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateDecision":
        if not isinstance(value, Mapping):
            raise CandidateValidationError("candidate decision must be an object")
        _require_exact_keys(
            value,
            required={"batch_id", "stage", "candidate_id"},
            context="candidate decision",
        )
        return cls(
            batch_id=value["batch_id"],
            stage=value["stage"],
            candidate_id=value["candidate_id"],
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "batch_id": self.batch_id,
            "stage": self.stage.value,
            "candidate_id": self.candidate_id,
        }


@dataclass(frozen=True, slots=True)
class StageSelection:
    """One accepted decision; stages may repeat during recovery loops."""

    stage: CandidateStage
    candidate_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.stage, CandidateStage):
            object.__setattr__(
                self,
                "stage",
                _coerce_enum(CandidateStage, self.stage, context="selection stage"),
            )
        _require_id(self.candidate_id, context="selected candidate_id")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StageSelection":
        if not isinstance(value, Mapping):
            raise CandidateValidationError("stage selection must be an object")
        _require_exact_keys(
            value,
            required={"stage", "candidate_id"},
            context="stage selection",
        )
        return cls(stage=value["stage"], candidate_id=value["candidate_id"])

    def to_dict(self) -> dict[str, JsonValue]:
        return {"stage": self.stage.value, "candidate_id": self.candidate_id}
