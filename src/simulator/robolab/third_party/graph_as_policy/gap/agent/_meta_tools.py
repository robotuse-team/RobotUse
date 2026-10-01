"""Codegen-time meta-tools: read_skill_reference, read_skill_example,
report_missing_capability, request_inline_script.

These are ``scope="codegen"`` tools — bound to the codegen LLM via the
provider tool-use loop, never present in the runtime workflow tool
catalog (``type: tool`` states cannot reference them).

Registration goes through :func:`register_codegen_meta_tools` (called by
:func:`gap.agent._catalog.build_codegen_tool_registry`) rather than the
``@tool`` decorator: the decorator's pending list is drained exactly once
per process, while the codegen pipeline builds a fresh ToolRegistry per
run and must be able to re-register these on each one.

The ``ctx`` parameter is the current
:class:`gap.agent.codegen_context.CodegenContext` injected by
:class:`gap.agent.subgraph_runner.SubgraphRunner`. Skill paths resolve
through ``ctx.skills_registry``, so the tools work regardless of where
bundles physically live.
"""

from __future__ import annotations

from typing import TypedDict


def _bundle_dir(ctx, skill_name: str):
    if ctx is None or getattr(ctx, "skills_registry", None) is None:
        raise RuntimeError(
            "skill-reference tool invoked without a CodegenContext.skills_registry"
        )
    try:
        info = ctx.skills_registry.get(skill_name)
    except KeyError as e:
        raise FileNotFoundError(
            f"skill {skill_name!r} is not registered; check the catalog "
            f"or the configured open-robot-skills checkout"
        ) from e
    return info.bundle_dir


def read_skill_reference(ctx, skill_name: str, doc_name: str) -> str:
    """Returns the markdown content of
    ``<bundle_dir>/references/<doc_name>.md`` for the registered skill.

    Use when a skill's SKILL.md body links a reference doc and you need
    the deeper rationale. Progressive disclosure: zero-cost until called.
    """
    bundle = _bundle_dir(ctx, skill_name)
    refs_dir = bundle / "references"
    for p in (refs_dir / f"{doc_name}.md", refs_dir / doc_name):
        if p.is_file():
            return p.read_text()
    listed = sorted(p.stem for p in refs_dir.glob("*.md")) if refs_dir.is_dir() else []
    raise FileNotFoundError(
        f"reference {doc_name!r} not found for skill {skill_name!r}. "
        f"Available: {listed}"
    )


def read_skill_example(ctx, skill_name: str, example_name: str) -> str:
    """Returns the JSON content of
    ``<bundle_dir>/examples/<example_name>.json`` for the registered skill."""
    bundle = _bundle_dir(ctx, skill_name)
    examples_dir = bundle / "examples"
    for p in (examples_dir / f"{example_name}.json", examples_dir / example_name):
        if p.is_file():
            return p.read_text()
    listed = sorted(p.stem for p in examples_dir.glob("*")) if examples_dir.is_dir() else []
    raise FileNotFoundError(
        f"example {example_name!r} not found for skill {skill_name!r}. "
        f"Available: {listed}"
    )


class MissingCapabilityReport(TypedDict):
    recorded: bool
    name: str


def report_missing_capability(ctx, name: str, why: str) -> MissingCapabilityReport:
    """Records a structured gap on the CodegenContext. The orchestrator
    aggregates these at end-of-run and aborts the build if non-empty."""
    if ctx is not None:
        ctx.missing_capabilities.append({"name": name, "why": why})
    return {"recorded": True, "name": name}


class InlineScriptResult(TypedDict):
    script_path: str
    schema: dict
    content: str


def request_inline_script(
    ctx,
    name: str,
    signature: str,
    purpose: str,
    body_hint: str = "",
) -> InlineScriptResult:
    """Spawns a coder subagent invocation with the provided spec; returns
    the script path + schema. The subgraph_agent then references the path
    in a `type: script` state.

    `name`      — basename without ``.py`` (e.g. ``"compute_align_pose"``).
    `signature` — type-annotated ``def run(...) -> Output`` line.
    `purpose`   — natural-language description of what the script computes.
    `body_hint` — optional pseudocode or constraints.
    """
    if ctx is None or ctx.subgraph_runner is None:
        raise RuntimeError(
            "request_inline_script invoked without an active CodegenContext / SubgraphRunner"
        )
    spec = {
        "name": name,
        "signature": signature,
        "purpose": purpose,
        "body_hint": body_hint,
        "subgraph_name": ctx.current_subgraph_name or "_global",
    }
    result = ctx.subgraph_runner.run_coder_sync(spec, ctx)
    return {
        "script_path": result["script_path"],
        "schema": result.get("schema", {}),
        "content": result["content"],
    }


#: (name, summary, fn) rows registered by :func:`register_codegen_meta_tools`.
_META_TOOLS: tuple[tuple[str, str, object], ...] = (
    (
        "read_skill_reference",
        "Load a long-form reference doc bundled with a skill. Returns markdown.",
        read_skill_reference,
    ),
    (
        "read_skill_example",
        "Load a sample subgraph JSON bundled with a skill. Returns the file content.",
        read_skill_example,
    ),
    (
        "report_missing_capability",
        "Flag that no skill or tool covers a needed primitive. The build "
        "aborts with the structured list at end-of-run.",
        report_missing_capability,
    ),
    (
        "request_inline_script",
        "Delegate to the coder subagent to emit a Python script. Returns the "
        "file path the subgraph should reference in a type:script state.",
        request_inline_script,
    ),
)


def register_codegen_meta_tools(registry) -> None:
    """Register the codegen meta-tools into *registry* (idempotent-per-registry)."""
    for name, summary, fn in _META_TOOLS:
        if name in registry:
            continue
        registry.register_callable(
            name, fn, summary=summary, tags=("meta",), scope="codegen",
        )
