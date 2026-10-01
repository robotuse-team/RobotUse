"""CodegenContext — per-codegen-run state shared across subagent invocations.

Holds the registries (skill bundles + tools), the workflow output
directory, and the side-effect ledger (missing capabilities, inline
scripts the coder emitted) that meta-tools update.

A new context is created for each :func:`gap.agent.generate` invocation.
The same instance is threaded through coordinator → subgraph_agent →
checkpoint_agent → coder so all side effects accumulate in one place;
the orchestrator inspects the ledger at end-of-run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gap_core.tools import ToolRegistry

    from gap.skills import SkillsRegistry

    from .subgraph_runner import SubgraphRunner


@dataclass
class CodegenContext:
    skills_registry: SkillsRegistry
    tool_registry: ToolRegistry
    workflow_dir: Path
    """Output directory; inline scripts get written here as
    ``scripts/<sg>/<name>.py``."""

    config: Any | None = None
    """The pipeline config (LLM credentials, composition knobs, etc.)."""

    subgraph_runner: SubgraphRunner | None = None
    """Set by the runner just before any LLM call so meta-tools can
    recurse (e.g. ``request_inline_script`` invokes the coder)."""

    current_subgraph_name: str | None = None
    """Set by the runner when invoking subgraph_agent for a specific
    subgraph; meta-tools that emit per-subgraph artifacts read this."""

    missing_capabilities: list[dict] = field(default_factory=list)
    inline_scripts: dict[str, str] = field(default_factory=dict)
    """``{path: content}`` for every script the coder emitted. The path
    is workflow-relative (e.g. ``scripts/grasp/compute_align_pose.py``)."""

    trace_dir: Path | None = None

    feedback: str | None = None
    """Optional free-text feedback from a prior attempt (e.g. a failed
    execution's checkpoint summary). When set, the runner prepends it to
    every agent system prompt so the LLM rewrites the right surface on
    retry. Replaces the source pipeline's rehearsal-feedback hook with a
    plain string parameter."""
