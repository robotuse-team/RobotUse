"""Real-robot observation contract: provenance tagging with transitive closure.

Every value that can reach the agent, the playbook, an executable skill, or the
critic carries a :class:`ProvenanceRecord`.  A value is *admissible* only when
every root of its derivation graph is something a real robot would have: camera
frames, calibrated camera geometry, or proprioception.

This is a whitelist with transitive closure, deliberately stronger than a
blacklist of known-privileged strings.  A blacklist catches a leaked body name;
it does not catch a ground-truth pose leaked as bare floats.  Here a value
derived from privileged input is refused no matter how it is encoded, because
admissibility is decided by the graph rather than by the payload.

The ledger is append-only and an input must already exist before it can be
referenced, so the derivation graph is acyclic by construction.

This module has no third-party dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from typing import Any, Iterable, Iterator, Mapping, Sequence


CONTRACT_SCHEMA_VERSION = 1

_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.:\-]{0,127}$")
_PRODUCER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,63}@v[0-9]+$")


class ContractError(RuntimeError):
    """Base error for observation-contract violations."""


class ContractValidationError(ContractError):
    """Raised when a provenance record or ledger operation is malformed."""


class ContractViolation(ContractError):
    """Raised when inadmissible information would reach the agent or critic.

    This is always a failed run, never a warning.
    """


class ProvenanceTag(str, Enum):
    """Closed set of provenance roots and the one derived form."""

    RGB = "rgb"
    DEPTH = "depth"
    CAMERA_CALIB = "camera_calib"
    PROPRIO = "proprio"
    DERIVED = "derived"
    SIM_PRIVILEGED = "sim_privileged"
    ORACLE = "oracle"


#: Roots a real robot genuinely has.
SENSOR_ROOTS: frozenset[ProvenanceTag] = frozenset(
    {
        ProvenanceTag.RGB,
        ProvenanceTag.DEPTH,
        ProvenanceTag.CAMERA_CALIB,
        ProvenanceTag.PROPRIO,
    }
)

#: Roots that must never reach the agent, the playbook, a skill, or the critic.
REFUSED_ROOTS: frozenset[ProvenanceTag] = frozenset(
    {ProvenanceTag.SIM_PRIVILEGED, ProvenanceTag.ORACLE}
)

_ROOT_TAGS: frozenset[ProvenanceTag] = SENSOR_ROOTS | REFUSED_ROOTS


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_of(value: Mapping[str, Any]) -> str:
    """Canonical SHA-256 of a JSON-safe mapping."""

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_value_id(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not _ID_PATTERN.match(value):
        raise ContractValidationError(
            f"{context} must be a lowercase finite identifier, got {value!r}"
        )
    return value


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    """One node in an episode's derivation graph."""

    value_id: str
    tag: ProvenanceTag
    producer: str | None = None
    inputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_value_id(self.value_id, context="value_id")

        tag = self.tag
        if not isinstance(tag, ProvenanceTag):
            try:
                tag = ProvenanceTag(tag)
            except (TypeError, ValueError) as exc:
                raise ContractValidationError(
                    f"unsupported provenance tag: {self.tag!r}"
                ) from exc
            object.__setattr__(self, "tag", tag)

        inputs = tuple(self.inputs)
        for item in inputs:
            _require_value_id(item, context="provenance input")
        if len(inputs) != len(set(inputs)):
            raise ContractValidationError("provenance inputs must be unique")
        if self.value_id in inputs:
            raise ContractValidationError("a value cannot derive from itself")
        object.__setattr__(self, "inputs", inputs)

        if tag is ProvenanceTag.DERIVED:
            if not isinstance(self.producer, str) or not _PRODUCER_PATTERN.match(
                self.producer
            ):
                raise ContractValidationError(
                    "derived values require a producer of the form 'tool@vN', "
                    f"got {self.producer!r}"
                )
            if not inputs:
                raise ContractValidationError(
                    "derived values require at least one input"
                )
        else:
            if self.producer is not None:
                raise ContractValidationError(
                    f"root provenance {tag.value} must not name a producer"
                )
            if inputs:
                raise ContractValidationError(
                    f"root provenance {tag.value} must not declare inputs"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "value_id": self.value_id,
            "tag": self.tag.value,
            "producer": self.producer,
            "inputs": list(self.inputs),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProvenanceRecord":
        if not isinstance(value, Mapping):
            raise ContractValidationError("provenance record must be an object")
        required = {"value_id", "tag", "producer", "inputs"}
        if set(value) != required:
            raise ContractValidationError(
                f"provenance record keys must be exactly {sorted(required)}"
            )
        raw_inputs = value["inputs"]
        if not isinstance(raw_inputs, list):
            raise ContractValidationError("provenance inputs must be an array")
        return cls(
            value_id=value["value_id"],
            tag=value["tag"],
            producer=value["producer"],
            inputs=tuple(raw_inputs),
        )


@dataclass(frozen=True, slots=True)
class AdmissibilityReport:
    """Why a value is or is not allowed to reach the agent."""

    value_id: str
    admissible: bool
    roots: frozenset[ProvenanceTag]
    offending_path: tuple[str, ...] = ()

    def describe(self) -> str:
        if self.admissible:
            names = ", ".join(sorted(tag.value for tag in self.roots))
            return f"{self.value_id} is admissible (roots: {names})"
        path = " <- ".join(self.offending_path)
        return f"{self.value_id} is inadmissible via {path}"


class ProvenanceLedger:
    """Append-only derivation graph for a single episode.

    An input must be recorded before anything may reference it, which makes the
    graph acyclic by construction and removes the need for cycle detection.
    """

    def __init__(self, *, episode_id: str) -> None:
        self.episode_id = _require_value_id(episode_id, context="episode_id")
        self._records: dict[str, ProvenanceRecord] = {}
        self._order: list[str] = []

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, value_id: object) -> bool:
        return value_id in self._records

    def __iter__(self) -> Iterator[ProvenanceRecord]:
        return (self._records[key] for key in self._order)

    def get(self, value_id: str) -> ProvenanceRecord:
        _require_value_id(value_id, context="value_id")
        try:
            return self._records[value_id]
        except KeyError:
            raise ContractValidationError(
                f"value is not in the provenance ledger: {value_id}"
            ) from None

    def record(self, record: ProvenanceRecord) -> ProvenanceRecord:
        """Append one node.  Rejects duplicates and forward references."""

        if not isinstance(record, ProvenanceRecord):
            raise ContractValidationError("ledger requires a ProvenanceRecord")
        if record.value_id in self._records:
            raise ContractValidationError(
                f"value_id already recorded: {record.value_id}"
            )
        for item in record.inputs:
            if item not in self._records:
                raise ContractValidationError(
                    "provenance input must be recorded before it is referenced: "
                    f"{item}"
                )
        self._records[record.value_id] = record
        self._order.append(record.value_id)
        return record

    def declare_root(self, value_id: str, tag: ProvenanceTag) -> ProvenanceRecord:
        """Convenience for sensor roots and for marking privileged sources."""

        return self.record(ProvenanceRecord(value_id=value_id, tag=tag))

    def declare_derived(
        self,
        value_id: str,
        *,
        producer: str,
        inputs: Sequence[str],
    ) -> ProvenanceRecord:
        return self.record(
            ProvenanceRecord(
                value_id=value_id,
                tag=ProvenanceTag.DERIVED,
                producer=producer,
                inputs=tuple(inputs),
            )
        )

    def analyse(self, value_id: str) -> AdmissibilityReport:
        """Resolve the root tag set, reporting the first inadmissible path."""

        record = self.get(value_id)
        roots: set[ProvenanceTag] = set()
        # Depth-first, tracking the path so a violation can be explained.
        stack: list[tuple[ProvenanceRecord, tuple[str, ...]]] = [
            (record, (record.value_id,))
        ]
        seen: set[str] = set()
        offending: tuple[str, ...] = ()
        while stack:
            current, path = stack.pop()
            if current.tag is not ProvenanceTag.DERIVED:
                roots.add(current.tag)
                if current.tag in REFUSED_ROOTS and not offending:
                    offending = path
                continue
            if current.value_id in seen:
                continue
            seen.add(current.value_id)
            for item in current.inputs:
                stack.append((self.get(item), path + (item,)))

        admissible = bool(roots) and roots <= SENSOR_ROOTS
        return AdmissibilityReport(
            value_id=value_id,
            admissible=admissible,
            roots=frozenset(roots),
            offending_path=offending if not admissible else (),
        )

    def is_admissible(self, value_id: str) -> bool:
        return self.analyse(value_id).admissible

    def assert_admissible(self, value_id: str, *, context: str) -> None:
        """Raise :class:`ContractViolation` if the value may not reach the agent."""

        report = self.analyse(value_id)
        if not report.admissible:
            raise ContractViolation(f"{context}: {report.describe()}")

    def assert_all_admissible(
        self,
        value_ids: Iterable[str],
        *,
        context: str,
    ) -> None:
        for value_id in value_ids:
            self.assert_admissible(value_id, context=context)

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "records": [self._records[key].to_dict() for key in self._order],
        }
        return {**body, "ledger_sha256": sha256_of(body)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProvenanceLedger":
        if not isinstance(value, Mapping):
            raise ContractValidationError("ledger must be an object")
        required = {"schema_version", "episode_id", "records", "ledger_sha256"}
        if set(value) != required:
            raise ContractValidationError(
                f"ledger keys must be exactly {sorted(required)}"
            )
        if value["schema_version"] != CONTRACT_SCHEMA_VERSION:
            raise ContractValidationError("unsupported contract schema version")
        records = value["records"]
        if not isinstance(records, list):
            raise ContractValidationError("ledger records must be an array")
        ledger = cls(episode_id=value["episode_id"])
        for item in records:
            ledger.record(ProvenanceRecord.from_dict(item))
        rebuilt = ledger.to_dict()
        if rebuilt["ledger_sha256"] != value["ledger_sha256"]:
            raise ContractValidationError("ledger hash mismatches")
        return ledger
