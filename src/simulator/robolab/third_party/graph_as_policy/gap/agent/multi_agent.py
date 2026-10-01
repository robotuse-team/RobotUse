"""Multi-agent workflow generation pipeline.

Pipeline stages:

1. **Coordinator** — emits the workflow topology as a ``WorkflowSpec``
   (per-subgraph skill name, inputs/outputs schemas, conditional edges).
2. **subgraph_agent** (universal) — for each subgraph, generates the
   inner state machine + inline scripts.
3. **checkpoint_agent** (whole-workflow, behind
   ``composition.checkpoint_agent`` — default ON) — one LLM call that
   authors postcondition checkpoints for every subgraph; sidecars land
   in ``<wf_dir>/checkpoints/<sg>.py``.
4. **Assemble** — stitch the per-subgraph dicts into workflow.json.
5. **Post-generation graph validation** + LLM script fix loop
   (≤ ``composition.max_validation_retries`` rounds).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._catalog import load_codegen_registries
from ._registry import default_agent_registry
from .codegen_context import CodegenContext
from .config import PipelineConfig
from .subgraph_runner import CodegenError, SubgraphRunner

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    success: bool = False
    workflow_json: str = ""
    scripts: dict[str, str] = field(default_factory=dict)
    checkpoint_modules: dict[str, str] = field(default_factory=dict)
    """Per-subgraph checkpoint sidecar source, keyed by subgraph name.
    Materialized to ``<wf_dir>/checkpoints/<sg>.py`` by
    :func:`_write_workflow_folder`."""
    workflow_dir: Path | None = None
    execution_stderr: str = ""
    attempts: int = 0
    validation_errors: list = field(default_factory=list)
    """Error-severity ``ValidationIssue``s still present after the
    post-generation fix loop. Non-empty ⇒ ``success`` is False — the
    pipeline never reports success on a structurally-invalid graph."""


def _write_workflow_folder(
    output_dir: Path,
    workflow_json: str,
    scripts: dict[str, str],
    skills_registry: Any = None,
    checkpoint_modules: dict[str, str] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "workflow.json").write_text(workflow_json)
    for filename, content in scripts.items():
        script_path = output_dir / filename
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(content)
    if checkpoint_modules:
        checkpoints_dir = output_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        for sg_name, module_source in checkpoint_modules.items():
            if not module_source:
                continue
            # Sanitize the filename — sg names are already validated by
            # the builder but defensive belt-and-suspenders here.
            safe_name = re.sub(r"[^A-Za-z0-9_]", "_", sg_name)
            (checkpoints_dir / f"{safe_name}.py").write_text(module_source)
    if skills_registry is not None:
        _materialize_canonical_scripts(output_dir, workflow_json, skills_registry)


def _materialize_canonical_scripts(
    output_dir: Path,
    workflow_json: str,
    skills_registry: Any,
) -> None:
    """Copy canonical (skill-owned) scripts referenced by any subgraph into
    ``output_dir`` so the runtime can resolve them from ``workflow_dir/<rel>``.

    The subgraph_agent emits ``type: script`` nodes pointing at canonical
    scripts by their stem (e.g. ``perceive_dino_vlm``). Inline LLM-emitted
    scripts already live under the writer-provided ``scripts`` dict; this
    function only fills in the canonical ones.

    Resolution: for each script-typed node in any subgraph, if the
    subgraph's skill has a canonical script whose stem matches the path's
    stem, copy that bundle file to ``output_dir / <script_rel>``.
    """
    try:
        wf = json.loads(workflow_json)
    except json.JSONDecodeError:
        return
    subgraphs = wf.get("subgraphs", {})
    if not isinstance(subgraphs, dict):
        return
    for _sg_name, sg in subgraphs.items():
        if not isinstance(sg, dict):
            continue
        skill_name = sg.get("skill")
        if not skill_name or skill_name not in skills_registry:
            continue
        info = skills_registry.get(skill_name)
        canonical = getattr(info, "canonical_scripts", None) or {}
        if not canonical:
            continue
        canonical_by_stem = {
            Path(s.bundle_relative).stem: s for s in canonical.values()
        }
        for node in sg.get("nodes", {}).values():
            if not isinstance(node, dict):
                continue
            # `script` and `router` nodes both carry a canonical `script`
            # field (a router's script returns the route, e.g. a loop's
            # decide/route_next_object). Field-based routers have no script
            # and are skipped by the empty-string guard below.
            if node.get("type") not in ("script", "router"):
                continue
            script_rel = node.get("script") or ""
            if not script_rel:
                continue
            target = output_dir / script_rel
            stem = Path(script_rel).stem
            sinfo = canonical_by_stem.get(stem)
            if sinfo is None:
                # Non-canonical stem — leave any LLM-emitted inline script
                # in place. Nothing to do.
                continue
            try:
                content = Path(sinfo.path).read_text()
            except OSError:
                continue
            # Canonical wins for any path whose stem matches a canonical
            # script — overwrite any LLM-emitted inline file. Inline
            # scripts only persist for non-canonical stems (handled
            # above by the early `continue`). Reason: LLM reimplementations
            # of canonical scripts have repeatedly introduced regressions
            # (wrong imports, simplified pose math, dropped parameters)
            # and the canonical is the authoritative version.
            if target.exists():
                logger.debug(
                    "Overriding LLM-emitted %s with canonical %s::%s",
                    target, skill_name, stem,
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
            logger.debug(
                "Materialized canonical script %s::%s -> %s",
                skill_name, stem, target,
            )


def _make_runner_and_ctx(
    config: PipelineConfig,
    workflow_dir: Path,
    trace_dir: Path | None = None,
) -> tuple[SubgraphRunner, CodegenContext]:
    if config.skills is None:
        raise CodegenError(
            "PipelineConfig.skills must point to an open-robot-skills checkout "
            "(the codegen registries are built from it)"
        )
    agents = default_agent_registry()
    skills, tools = load_codegen_registries(config.skills)
    runner = SubgraphRunner(agents, skills, tools)
    ctx = CodegenContext(
        skills_registry=skills,
        tool_registry=tools,
        workflow_dir=workflow_dir,
        config=config,
        trace_dir=trace_dir,
    )
    return runner, ctx


async def generate_workflow(
    task_id: int,
    task_prompt: str,
    config: PipelineConfig,
    output_dir: Path,
) -> tuple[str, dict[str, str], dict[str, str]]:
    """Multi-agent workflow generation pipeline.

    Returns ``(workflow_json_str, scripts_dict, checkpoint_modules_dict)``.
    ``checkpoint_modules_dict`` maps subgraph name → sidecar source for
    each subgraph that declared postcondition checkpoints; empty when the
    checkpoint agent is disabled or no subgraph emitted any.
    """
    workflow_dir = output_dir / f"task_{task_id:02d}"
    trace_dir = workflow_dir / "agent_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    runner, ctx = _make_runner_and_ctx(config, workflow_dir, trace_dir=trace_dir)

    # 1. Coordinator → workflow topology
    coord = await runner.run_coordinator(task_prompt, ctx)
    workflow_spec = coord.workflow_spec

    # 2. Per-subgraph generation (sequential; preserves graph-as-policy
    #    contract). Each call produces structure only — checkpoints are
    #    authored in a separate whole-workflow call below.
    subgraph_dicts: dict[str, dict] = {}
    sg_sources: dict[str, str] = {}
    upstream_outputs: dict[str, dict[str, str]] = {}
    for sg_name, sg_spec in workflow_spec.get("subgraphs", {}).items():
        if not isinstance(sg_spec, dict):
            continue

        skill_name = sg_spec.get("skill")
        if not skill_name:
            raise CodegenError(f"subgraph {sg_name!r} has no `skill` field")

        spec_for_runner = {
            "name": sg_name,
            "description": sg_spec.get("description", ""),
            "inputs": sg_spec.get("inputs", {}),
            "outputs": sg_spec.get("outputs", {}),
            "exit": sg_spec.get("exit", {}),
            "on_error": sg_spec.get("on_error"),
            "context": sg_spec.get("context", {}),
            # Canonical pick-and-place stage tag — preserved verbatim so
            # downstream refinement tooling can group subgraphs by stage
            # when computing per-stage pass-rates.
            "stage": sg_spec.get("stage"),
            # Invented-skill marker: tells run_subgraph_agent / the prompt
            # assembler to synthesize the skill contract from this spec
            # instead of loading a (non-existent) bundle.
            "generated": sg_spec.get("generated", False),
        }
        result = await runner.run_subgraph_agent(
            skill_name=skill_name,
            subgraph_spec=spec_for_runner,
            upstream_outputs=upstream_outputs,
            ctx=ctx,
        )
        subgraph_dicts[sg_name] = result.subgraph_dict
        sg_sources[sg_name] = result.sg_source
        upstream_outputs[sg_name] = dict(spec_for_runner["outputs"])

    if ctx.missing_capabilities:
        gaps = "\n".join(f"  - {m['name']}: {m['why']}" for m in ctx.missing_capabilities)
        raise CodegenError(f"coordinator/subgraph_agent reported missing capabilities:\n{gaps}")

    workflow_json = _assemble(task_id, task_prompt, workflow_spec, subgraph_dicts)

    # 3. Whole-workflow checkpoint_agent — one LLM call that authors
    #    postconditions for every subgraph at once. Sees every subgraph's
    #    builder source + bound outputs + the original task description.
    #    Behind config flag (default ON).
    checkpoint_modules: dict[str, str] = {}
    if config.composition.checkpoint_agent:
        workflow_for_checkpoint = json.loads(workflow_json)
        checkpoint_modules = await runner.run_checkpoint_agent(
            workflow_dict=workflow_for_checkpoint,
            sg_sources=sg_sources,
            task_prompt=task_prompt,
            ctx=ctx,
        )

    return workflow_json, dict(ctx.inline_scripts), checkpoint_modules


def _assemble(
    task_id: int,
    task_prompt: str,
    workflow_spec: dict,
    subgraph_dicts: dict[str, dict],
) -> str:
    workflow = {
        "version": 3,
        "meta": {
            "name": f"task_{task_id:02d}",
            "description": task_prompt,
        },
        "nodes": workflow_spec.get("nodes", {}),
        "edges": workflow_spec.get("edges", []),
        "conditional_edges": workflow_spec.get("conditional_edges", {}),
        "subgraphs": subgraph_dicts,
    }
    return json.dumps(workflow, indent=2)


def assemble_workflow_json(
    task_id: int,
    task_prompt: str,
    workflow_spec: dict,
    subgraph_dicts: dict[str, dict],
) -> str:
    """Public alias for the v3 workflow stitcher."""
    return _assemble(task_id, task_prompt, workflow_spec, subgraph_dicts)


# ---------------------------------------------------------------------------
# Generate-and-write entry point (used by the gap.agent.generate facade)
# ---------------------------------------------------------------------------


async def run_codegen(
    *,
    task_id: int,
    task_prompt: str,
    config: PipelineConfig,
    output_dir: Path,
) -> PipelineResult:
    """Generate a valid workflow, regenerating from scratch when a whole graph
    is still structurally invalid after the per-attempt script-fix loop.

    Fix-first, regenerate-as-fallback: each attempt (:func:`_codegen_attempt`)
    already repairs *script* errors via the LLM fix loop; but coordinator-level
    structural mistakes (e.g. a W8 — a subgraph declaring an input with no
    upstream producer) are not script-repairable, so we re-roll the stochastic
    pipeline up to ``composition.max_codegen_regenerations`` extra times and
    return the first fully-valid graph. When every attempt fails, return the
    most *diagnostic* failure, not the last one: an attempt that wrote a graph
    and reports concrete residual ``validation_errors`` must not be clobbered
    by a later re-roll that crashed on an infrastructure error (LLM outage,
    exhausted stub queue in tests) with nothing to show — that would violate
    the honest-reporting contract (see ``_codegen_attempt``).
    """
    max_regen = max(0, config.composition.max_codegen_regenerations)

    def _diagnostic_rank(r: PipelineResult) -> tuple[int, int]:
        # Concrete residual validation errors beat a written-but-unexplained
        # graph, which beats a bare crash with no artifact at all.
        return (int(bool(r.validation_errors)), int(r.workflow_dir is not None))

    best_failure: PipelineResult | None = None
    for attempt in range(max_regen + 1):
        result = await _codegen_attempt(
            task_id=task_id, task_prompt=task_prompt,
            config=config, output_dir=output_dir,
        )
        if result.success:
            if attempt > 0:
                logger.info(
                    "Task %d: regeneration attempt %d/%d produced a valid graph",
                    task_id, attempt, max_regen,
                )
            result.attempts = attempt + 1
            return result
        if best_failure is None or _diagnostic_rank(result) > _diagnostic_rank(best_failure):
            best_failure = result
        if attempt < max_regen:
            logger.warning(
                "Task %d: graph still invalid after script-fixes (%s); "
                "regenerating from scratch (attempt %d/%d)",
                task_id, (result.execution_stderr or "")[:140],
                attempt + 1, max_regen,
            )
    if best_failure is not None:
        best_failure.attempts = max_regen + 1
    return best_failure if best_failure is not None else PipelineResult(
        success=False, execution_stderr="codegen produced no result", attempts=1,
    )


async def _codegen_attempt(
    *,
    task_id: int,
    task_prompt: str,
    config: PipelineConfig,
    output_dir: Path,
) -> PipelineResult:
    """One generate → write → validate → script-fix pass. Returns a
    :class:`PipelineResult` (never raises on workflow failure — the error
    lands in ``execution_stderr``)."""
    try:
        workflow_json, scripts, checkpoint_modules = await generate_workflow(
            task_id, task_prompt, config, output_dir,
        )

        wf_dir = output_dir / f"task_{task_id:02d}"
        from gap.skills import load_registry_set
        skills_registry = load_registry_set(config.skills) if config.skills else None
        _write_workflow_folder(
            wf_dir, workflow_json, scripts, skills_registry,
            checkpoint_modules=checkpoint_modules,
        )
        (wf_dir / "multi_agent_meta.json").write_text(json.dumps({
            "pipeline": "gap_multi_agent",
            "schema_version": 3,
        }))

        try:
            from gap.runtime.workflow import load_workflow
            load_workflow(wf_dir / "workflow.json")
        except Exception as e:
            logger.error("Post-generation validation failed: %s", e)
            return PipelineResult(
                success=False,
                workflow_json=workflow_json,
                scripts=scripts,
                checkpoint_modules=checkpoint_modules,
                workflow_dir=wf_dir,
                execution_stderr=f"Post-generation validation failed: {e}",
                attempts=1,
            )

        max_fix = config.composition.max_validation_retries
        trace_dir = wf_dir / "agent_traces"
        validation_errors: list = []
        for fix_attempt in range(max_fix + 1):
            validation_errors = _run_graph_validation(wf_dir, config)
            if not validation_errors:
                logger.info("Task %d: graph validation passed", task_id)
                break
            if fix_attempt == max_fix:
                logger.warning(
                    "Task %d: validation still has %d error(s) after %d fix attempt(s)",
                    task_id, len(validation_errors), max_fix,
                )
                break
            fixed_scripts = await _fix_script_errors(
                validation_errors, workflow_json, scripts, config,
                trace_dir=trace_dir, attempt=fix_attempt,
            )
            if not fixed_scripts:
                break
            scripts.update(fixed_scripts)
            _write_workflow_folder(
                wf_dir, workflow_json, scripts, skills_registry,
                checkpoint_modules=checkpoint_modules,
            )

        # Honest reporting: success reflects structural validity. Any
        # error-severity issue still present after the fix loop (e.g.
        # cross-subgraph W8 / workflow-level errors that the per-subgraph
        # feedback loop and the script-body fixer cannot repair) flips
        # success to False so we never ship a broken graph as "OK".
        # `validation_errors` is the loop's last reading and reflects the
        # final on-disk state in every loop-exit path.
        residual = validation_errors
        if residual:
            logger.warning(
                "Task %d: %d residual validation error(s) remain",
                task_id, len(residual),
            )
        return PipelineResult(
            success=not residual,
            workflow_json=workflow_json,
            scripts=scripts,
            checkpoint_modules=checkpoint_modules,
            workflow_dir=wf_dir,
            execution_stderr=(
                "" if not residual
                else f"{len(residual)} residual validation error(s): "
                + "; ".join(str(i) for i in residual[:5])
            ),
            validation_errors=residual,
            attempts=1,
        )

    except Exception as e:
        logger.error("Multi-agent codegen failed: %s", e, exc_info=True)
        return PipelineResult(
            success=False,
            execution_stderr=str(e),
            attempts=1,
        )


def _run_graph_validation(wf_dir: Path, config: PipelineConfig) -> list:
    try:
        from gap.runtime.validate import validate_workflow
        from gap.runtime.workflow import load_workflow

        wf = load_workflow(wf_dir / "workflow.json")
        skill_registry = None
        tool_registry = None
        if config.skills:
            try:
                skill_registry, tool_registry = load_codegen_registries(config.skills)
            except Exception:
                logger.warning("could not build registries for validation", exc_info=True)

        issues = validate_workflow(
            wf,
            agent_registry=None,    # agent field is informational here
            skill_registry=skill_registry,
            tool_registry=tool_registry,
        )
        return [i for i in issues if i.severity == "error"]
    except Exception as e:
        logger.warning("Graph validation could not run: %s", e)
        return []


async def _fix_script_errors(
    errors: list,
    workflow_json: str,
    scripts: dict[str, str],
    config: PipelineConfig,
    trace_dir: Path | None = None,
    attempt: int = 0,
) -> dict[str, str]:
    """LLM-driven fix loop for script validation errors."""
    from .llm import complete

    wf = json.loads(workflow_json)
    state_to_script: dict[str, str] = {}
    for sg_name, sg_def in wf.get("subgraphs", {}).items():
        if not isinstance(sg_def, dict):
            continue
        for node_name, node_def in sg_def.get("nodes", {}).items():
            if isinstance(node_def, dict) and node_def.get("type") in ("script", "router"):
                full_id = f"subgraphs.{sg_name}.nodes.{node_name}"
                state_to_script[full_id] = node_def.get("script", "")

    errors_by_script: dict[str, list[str]] = {}
    for issue in errors:
        script_path = state_to_script.get(issue.node_id, "")
        if script_path and script_path in scripts:
            errors_by_script.setdefault(script_path, []).append(str(issue))

    if not errors_by_script:
        return {}

    fixed: dict[str, str] = {}
    for script_path, error_msgs in errors_by_script.items():
        content = scripts[script_path]
        error_text = "\n".join(f"  - {e}" for e in error_msgs)
        system_prompt = (
            "You are a Python code fixer for a robotics workflow system. "
            "Fix ONLY the validation errors listed below. Return the "
            "complete fixed script in a single ```python code block. "
            "Do not change function signatures, imports, or logic beyond "
            "what is strictly needed to resolve the errors. "
            "Preserve all type annotations."
        )
        user_msg = (
            f"Fix the following validation errors in `{script_path}`:\n\n"
            f"Errors:\n{error_text}\n\n"
            f"Current script:\n```python\n{content}\n```"
        )
        raw_response = await complete(
            config.llm, system=system_prompt,
            messages=[{"role": "user", "content": user_msg}],
        )
        fixed_code = _extract_python_block(raw_response)
        if fixed_code:
            fixed[script_path] = fixed_code
        if trace_dir:
            fix_dir = trace_dir / "_validation_fix" / f"attempt_{attempt}"
            fix_dir.mkdir(parents=True, exist_ok=True)
            safe = script_path.replace("/", "_").replace(".", "_")
            (fix_dir / f"{safe}_prompt.txt").write_text(f"System:\n{system_prompt}\n\nUser:\n{user_msg}")
            (fix_dir / f"{safe}_response.txt").write_text(raw_response)
            if fixed_code:
                (fix_dir / f"{safe}_fixed.py").write_text(fixed_code)
    return fixed


def _extract_python_block(raw: str) -> str:
    match = re.search(r"```(?:python|py)\n(.*?)```", raw, re.DOTALL)
    return match.group(1).strip() if match else ""
