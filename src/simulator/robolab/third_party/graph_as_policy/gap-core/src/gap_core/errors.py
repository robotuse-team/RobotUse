"""Error hierarchy for the gap runtime and skill library.

Skills raise :class:`PipelineError` subclasses; the executor wraps node
failures in :class:`NodeExecutionError` and routes subgraph ``on_error``
exits. :class:`GuardLimitExceeded` inherits from ``BaseException`` so a bare
``except Exception:`` in skill code cannot silence a safety limit.
"""

from __future__ import annotations

from dataclasses import dataclass


class PipelineError(Exception):
    """Base error for pipeline execution failures."""


class PerceptionFailed(PipelineError):
    """Object not detected or segmentation failed."""


class PlanningFailed(PipelineError):
    """Motion planning failed after retries."""


class GraspFailed(PipelineError):
    """Gripper position indicates empty grasp or object dropped."""


class ValidationFailed(PipelineError):
    """VLM validation check rejected a result."""


class VerificationFailed(PipelineError):
    """Post-execution verification (checkpoint) failed."""


class ToolError(PipelineError):
    """A tool call failed (connector tool or tool-bundle function)."""

    def __init__(self, tool: str, detail: str = ""):
        self.tool = tool
        self.detail = detail
        super().__init__(f"{tool}: {detail}")


class WorkflowValidationError(PipelineError):
    """Workflow JSON is structurally invalid."""


class NodeExecutionError(PipelineError):
    """A node failed during execution."""

    def __init__(self, node_id: str, cause: Exception):
        self.node_id = node_id
        self.cause = cause
        super().__init__(f"Node '{node_id}' failed: {cause}")


class TaskCancelled(PipelineError):
    """A long-running skill / parallel branch was cancelled cooperatively.

    Raised by ``ctx.cancel_token.raise_if_set()`` inside skill loops when the
    owning scope exits or a sibling branch pre-empts. Catches as a regular
    PipelineError so workflows can route via ``on_error`` if desired.
    """


class StreamUnavailable(PipelineError):
    """Observation stream did not produce a first sample within the timeout.

    Raised by ``ObservationStream.latest()`` when the background poller has
    not completed a successful observation read before the caller's timeout.
    The underlying poll error (if any) is attached as ``cause``.
    """

    def __init__(self, cause: Exception | None = None):
        self.cause = cause
        msg = "observation stream is not yet available"
        if cause is not None:
            msg += f" (last poll error: {cause})"
        super().__init__(msg)


class GuardLimitExceeded(BaseException):
    """A safety call-count guard was exceeded.

    Inherits from ``BaseException`` so skill code cannot accidentally
    swallow it with ``except Exception:`` — guard violations must always
    terminate the workflow.
    """


@dataclass
class ValidationIssue:
    """A single validation finding (error or warning)."""

    severity: str  # "error" or "warning"
    node_id: str
    field: str | None
    message: str

    def __str__(self) -> str:
        loc = self.node_id
        if self.field:
            loc = f"{self.node_id}.{self.field}"
        return f"[{self.severity}] {loc}: {self.message}"


class GraphValidationError(WorkflowValidationError):
    """Pre-execution graph validation found type or connection errors."""

    def __init__(self, issues: list[ValidationIssue]):
        self.issues = issues
        summary = "; ".join(str(i) for i in issues[:5])
        if len(issues) > 5:
            summary += f" ... and {len(issues) - 5} more"
        super().__init__(f"Graph validation failed ({len(issues)} issues): {summary}")
