"""Python authoring library for v3 workflow JSON.

Two peer classes mirror how LLM agents divide workflow authoring:

- :class:`Subgraph` is what a subagent authors and returns. It is fully
  serializable and inspectable on its own.
- :class:`Workflow` is what a coordinator authors. It imports the
  subagents' Subgraph objects via :meth:`Workflow.add_subgraph` and wires
  them together at the top level.

API mirrors LangGraph: ``add_node``, ``add_edge``, ``add_conditional_edges``.
See module-level docstring of :mod:`gap.builder.core` for details.
"""

from .core import (
    END,
    START,
    BuilderError,
    Ref,
    Subgraph,
    Workflow,
    WorkflowSpec,
)

__all__ = [
    "BuilderError",
    "END",
    "Ref",
    "START",
    "Subgraph",
    "Workflow",
    "WorkflowSpec",
]
