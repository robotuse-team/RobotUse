"""Existing agent/backend contracts, shared by both import layouts."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


class Backend(Protocol):
    """Return mappings with the public schemas below; extra fields stay private.

    observe: observation_id, views (list[{view_id, image_ref}])
    point: point_ref, observation_id, image_refs (optional list[str])
    grasp_candidates: candidates (list[{candidate_ref, image_refs?}])
    validate_grasp: candidate_ref, validation_ref, accepted (bool)
    execute_grasp: execution_ref, status ('succeeded'|'failed'|'unknown')
    place: execution_ref, status ('succeeded'|'failed'|'unknown')

    point segments a model-selected normalized pixel (u,v in [0,1000]).
    Backend must reject stale/unknown refs; it must never derive targets from BDDL.
    Candidate previews are sensor-derived, not simulator geometry overlays.
    """

    def observe(self) -> Mapping[str, Any]: ...
    def point(self, observation_id: str, view_id: str, u: float, v: float) -> Mapping[str, Any]: ...
    def grasp_candidates(self, point_ref: str) -> Mapping[str, Any]: ...
    def validate_grasp(self, candidate_ref: str) -> Mapping[str, Any]: ...
    def execute_grasp(self, candidate_ref: str, validation_ref: str) -> Mapping[str, Any]: ...
    def place(self, point_ref: str) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class Action:
    """Exactly one tool call, or finish with a structured result."""

    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


class AgentSession(Protocol):
    def next_action(self, messages: Sequence[Mapping[str, Any]], tools: Sequence[str]) -> Action: ...
    def close(self) -> None: ...


class SessionFactory(Protocol):
    def new_session(self, role: str, session_id: str) -> AgentSession: ...


@dataclass(frozen=True)
class Budgets:
    prime_steps: int = 40
    child_steps: int = 24
    max_delegations: int = 32
    max_tool_calls: int = 256

    def __post_init__(self) -> None:
        for value in (self.prime_steps, self.child_steps, self.max_delegations, self.max_tool_calls):
            if type(value) is not int or value < 1:
                raise ValueError('budgets must be positive integers')


@dataclass(frozen=True)
class RunResult:
    status: str
    result: Mapping[str, Any]
    events: tuple[Mapping[str, Any], ...]
    tool_calls: int
    delegations: int


class BoundaryError(ValueError):
    """Invalid action or malformed public backend result."""


class BudgetExceeded(RuntimeError):
    pass


class EpisodeEnded(RuntimeError):
    def __init__(self, budget):
        self.result = dict(status='unknown', verified=False, terminal=True,
            reason_code=budget['reason_code'], simulation_budget=deepcopy(budget))


class FreshAttemptRequired(RuntimeError):
    def __init__(self, cause, execution_ref):
        self.result = {"cause": cause, "execution_ref": execution_ref, "verified": False}


class FirstGraspFinished(RuntimeError):
    def __init__(self, result):
        self.result = result


class AuditError(RuntimeError):
    pass
