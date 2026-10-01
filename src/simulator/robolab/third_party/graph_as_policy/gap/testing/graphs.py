"""Graph assertion helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def assert_graph_valid(graph: dict | str | Path, *, skill_registry: Any = None,
                       tool_registry: Any = None) -> None:
    """Load + structurally validate a graph; raise AssertionError with the
    full issue list on error-level findings."""
    import json
    import tempfile

    from gap.runtime.validate import validate_workflow
    from gap.runtime.workflow import load_workflow

    if isinstance(graph, dict):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "workflow.json"
            p.write_text(json.dumps(graph))
            wf = load_workflow(p)
            issues = validate_workflow(
                wf, skill_registry=skill_registry, tool_registry=tool_registry
            )
    else:
        p = Path(graph)
        if p.is_dir():
            p = p / "workflow.json"
        wf = load_workflow(p)
        issues = validate_workflow(
            wf, skill_registry=skill_registry, tool_registry=tool_registry
        )
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        raise AssertionError(
            "graph failed validation:\n" + "\n".join(str(i) for i in errors)
        )
