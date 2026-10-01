"""Codegen LLM call/parse/validate/retry loop.

Driven by an :class:`gap.agent._registry.AgentSpec` (markdown-defined
subagent) and the :class:`PromptAssembler`.

The runner handles three roles via ``run_*`` entry points:

- ``run_coordinator`` — task → workflow topology (a ``WorkflowSpec``
  builder block bound to ``spec``).
- ``run_subgraph_agent`` — per-subgraph state machine + inline scripts
  (a ``Subgraph`` builder block bound to ``sg``).
- ``run_checkpoint_agent`` — one whole-workflow pass that authors
  postcondition checkpoints for every subgraph.
- ``run_coder_sync`` — single ad-hoc script (invoked through the
  ``request_inline_script`` meta-tool).

Each entry point uses the same retry-on-validation-error loop. The LLM
emits fenced ``python`` / ``python:scripts/...`` blocks; codegen
meta-tools (``read_skill_reference`` etc.) are bound through
:func:`gap.agent.llm.complete_with_tools` on every provider.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from gap_core.tools import ToolRegistry

from gap.skills import SkillsRegistry

from ._registry import AgentRegistry
from .codegen_context import CodegenContext
from .prompt_assembler import PromptAssembler

logger = logging.getLogger(__name__)


@dataclass
class CoordinatorResult:
    workflow_spec: dict
    """Parsed top-level workflow topology JSON."""

    raw_text: str = ""


@dataclass
class SubgraphResult:
    subgraph_dict: dict
    scripts: dict[str, str] = field(default_factory=dict)
    checkpoint_module: str | None = None
    """Always ``None`` after the checkpoint_agent split — this field is
    kept only for backwards compatibility with the field name. The
    sidecar is now produced by :meth:`SubgraphRunner.run_checkpoint_agent`
    in a follow-on pass over the whole workflow."""
    sg_source: str = ""
    """The verbatim Python builder block the LLM emitted (the block that
    bound ``sg = Subgraph(...)``). Consumed by ``run_checkpoint_agent``
    to materialize the subgraph object in a sandbox where the
    workflow-wide checkpoint script can attach postconditions to it."""
    raw_text: str = ""


@dataclass
class CoderResult:
    script_path: str
    content: str
    schema: dict = field(default_factory=dict)
    raw_text: str = ""


class SubgraphRunner:
    """Drives the codegen LLM loop for each function-only subagent role."""

    def __init__(
        self,
        agents: AgentRegistry,
        skills: SkillsRegistry,
        tools: ToolRegistry,
        assembler: PromptAssembler | None = None,
        max_retries: int = 2,
    ) -> None:
        self.agents = agents
        self.skills = skills
        self.tools = tools
        self.assembler = assembler or PromptAssembler(agents, skills, tools)
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # coordinator
    # ------------------------------------------------------------------

    async def run_coordinator(
        self,
        task_prompt: str,
        ctx: CodegenContext,
    ) -> CoordinatorResult:
        ctx.subgraph_runner = self
        prompt = self.assembler.assemble_coordinator(task_prompt)
        system_prompt = _inject_feedback(prompt.system_prompt, ctx.feedback)

        if ctx.trace_dir:
            d = ctx.trace_dir / "_coordinator"
            d.mkdir(parents=True, exist_ok=True)
            (d / "system_prompt.md").write_text(system_prompt, encoding="utf-8")

        conversation: list[dict] = [{"role": "user", "content": task_prompt}]
        last_error = ""
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                conversation.append({
                    "role": "user",
                    "content": (
                        f"Your previous response had errors:\n```\n{last_error}\n```\n\n"
                        f"Fix every error and re-emit the full ```python``` builder block "
                        f"that binds `spec = WorkflowSpec(...)`."
                    ),
                })
            llm_cfg = _resolve_llm(ctx.config, role="coordinator")
            raw = await self._call_llm(
                llm_cfg, system_prompt, conversation,
                bound_codegen_tools=prompt.bound_codegen_tools, ctx=ctx,
            )
            conversation.append({"role": "assistant", "content": raw})

            if ctx.trace_dir:
                d = ctx.trace_dir / "_coordinator"
                (d / f"llm_response_attempt_{attempt}.md").write_text(raw, encoding="utf-8")

            spec_dict = _parse_coordinator_response(raw)
            missing = _extract_missing(spec_dict, raw)
            if missing:
                ctx.missing_capabilities.extend(missing)
                last_error = f"coordinator reported missing capabilities: {missing!r}"
                continue
            if spec_dict is None:
                last_error = (
                    "no parseable workflow spec found; emit one ```python``` "
                    "block (no file path) that imports `from gap.builder import "
                    "WorkflowSpec, START`, builds the workflow scaffold, and "
                    "binds it to a module-level variable named `spec`."
                )
                continue
            err = _validate_workflow_spec(spec_dict, self.skills)
            if err:
                last_error = err
                continue
            return CoordinatorResult(workflow_spec=spec_dict, raw_text=raw)

        raise CodegenError(f"coordinator failed after {self.max_retries + 1} attempts: {last_error}")

    # ------------------------------------------------------------------
    # subgraph_agent
    # ------------------------------------------------------------------

    async def run_subgraph_agent(
        self,
        skill_name: str,
        subgraph_spec: dict,
        upstream_outputs: dict[str, dict[str, str]],
        ctx: CodegenContext,
    ) -> SubgraphResult:
        ctx.subgraph_runner = self
        ctx.current_subgraph_name = subgraph_spec.get("name")
        prompt = self.assembler.assemble_subgraph_agent(skill_name, subgraph_spec, upstream_outputs)
        system_prompt = _inject_feedback(prompt.system_prompt, ctx.feedback)

        sg_name = subgraph_spec.get("name", "_subgraph")
        if ctx.trace_dir:
            d = ctx.trace_dir / sg_name
            d.mkdir(parents=True, exist_ok=True)
            (d / "system_prompt.md").write_text(system_prompt, encoding="utf-8")
            (d / "subgraph_spec.json").write_text(json.dumps(subgraph_spec, indent=2), encoding="utf-8")

        user_msg = subgraph_spec.get("description") or f"Generate the {sg_name!r} subgraph using {skill_name!r}."
        conversation: list[dict] = [{"role": "user", "content": user_msg}]
        last_error = ""
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                conversation.append({
                    "role": "user",
                    "content": (
                        f"Your previous response had validation errors:\n```\n{last_error}\n```\n\n"
                        f"Fix every error and re-emit the full ```python``` builder block "
                        f"that binds `sg = Subgraph(...)` (plus any updated `python:scripts/<sg>/<file>.py` blocks)."
                    ),
                })
            llm_cfg = _resolve_llm(ctx.config, role="subgraph_agent")
            raw = await self._call_llm(
                llm_cfg, system_prompt, conversation,
                bound_codegen_tools=prompt.bound_codegen_tools, ctx=ctx,
            )
            conversation.append({"role": "assistant", "content": raw})

            if ctx.trace_dir:
                d = ctx.trace_dir / sg_name
                (d / f"llm_response_attempt_{attempt}.md").write_text(raw, encoding="utf-8")

            sg_dict, scripts, _legacy_cp_module, _legacy_cp_meta, sg_source, builder_error = (
                _parse_subgraph_response(raw)
            )
            missing = _extract_missing(sg_dict, raw)
            if missing:
                ctx.missing_capabilities.extend(missing)
                last_error = f"subgraph_agent reported missing capabilities: {missing!r}"
                continue
            if sg_dict is None:
                # A builder block that execs to an error (e.g. a literal
                # where a Ref() was required) gets its real message fed
                # back so the agent can fix it; otherwise the model emitted
                # no usable ```python``` block at all.
                last_error = builder_error or (
                    "no parseable subgraph found; emit one ```python``` block "
                    "(no file path) that imports `from gap.builder import "
                    "Subgraph, Ref, START, END`, builds the subgraph, and "
                    "binds it to a module-level variable named `sg`."
                )
                continue
            sg_dict["skill"] = skill_name
            # Merge the coordinator's declared inputs with the ones the
            # subgraph_agent actually declared via `sg.add_input(...)`,
            # letting the agent's win on conflict. Do NOT overwrite: the
            # agent's node `Ref("in.<name>")` usages must match its own
            # declarations, and clobbering them (e.g. with an empty dict for
            # an invented/generated skill) silently drops inputs the agent
            # needs and makes the structural validator below flag the
            # agent's own correct refs as undeclared (S5). Coordinator
            # inputs the agent omitted are still kept (contract preserved).
            declared_inputs = dict(subgraph_spec.get("inputs", {}))
            authored_inputs = dict(sg_dict.get("inputs", {}))
            sg_dict["inputs"] = {**declared_inputs, **authored_inputs}
            # Carry the canonical pick-and-place stage tag through to
            # workflow.json so downstream refinement tooling can group
            # failures by stage without re-inferring on every read.
            spec_stage = subgraph_spec.get("stage")
            if spec_stage:
                sg_dict["stage"] = spec_stage
            # `transitions` was a v2 field; v3 SubgraphDef has no such key.
            sg_dict.pop("transitions", None)
            scripts = _namespace_scripts(sg_name, scripts, sg_dict)
            err = _validate_subgraph(sg_dict, subgraph_spec, self.skills)
            if err:
                last_error = err
                continue
            # Full per-subgraph structural validation (S1-S11), reusing the
            # authoritative runtime validator. The shallow _validate_subgraph
            # above only catches parse-level mistakes; this catches $ref/
            # in.<name> validity, reachability, output binding, conditional
            # edges, etc. — and feeds them back to THIS agent (which has the
            # full builder context to fix them) via the same retry loop.
            # Structure-only: cross-subgraph (W8) and script-schema checks
            # remain at post-assembly (they need the whole workflow / files
            # on disk).
            struct_err = _structural_subgraph_errors(
                sg_name, sg_dict, self.skills,
            )
            if struct_err:
                last_error = struct_err
                continue
            # Checkpoint authoring has moved out of subgraph_agent's
            # responsibility (run_checkpoint_agent handles it post-hoc).
            for path, content in scripts.items():
                ctx.inline_scripts[path] = content
            # Strip any spurious sg.add_checkpoint(...) calls the LLM
            # emitted before stashing the source — the checkpoint_agent
            # re-execs this string and must NOT crash on a half-formed
            # predicate kwarg.
            stashed_source = sg_source or ""
            if stashed_source:
                try:
                    stashed_source = _strip_add_checkpoint_calls(stashed_source)
                except Exception:
                    pass
            return SubgraphResult(
                subgraph_dict=sg_dict,
                scripts=scripts,
                checkpoint_module=None,
                sg_source=stashed_source,
                raw_text=raw,
            )

        raise CodegenError(
            f"subgraph_agent for {sg_name!r} (skill={skill_name!r}) "
            f"failed after {self.max_retries + 1} attempts: {last_error}"
        )

    # ------------------------------------------------------------------
    # checkpoint_agent (whole-workflow)
    # ------------------------------------------------------------------

    async def run_checkpoint_agent(
        self,
        workflow_dict: dict,
        sg_sources: dict[str, str],
        task_prompt: str,
        ctx: CodegenContext,
    ) -> dict[str, str]:
        """Single LLM call that authors checkpoints for every subgraph.

        Materializes each subgraph (by re-execing its builder source)
        into a ``subgraphs`` dict, prompts the LLM to attach
        checkpoints via ``subgraphs["<sg>"].add_checkpoint(...)``,
        execs the response in that sandbox, AST-splits the script
        per-subgraph, and renders one sidecar per subgraph.

        Returns ``{sg_name: sidecar_source_text}`` ready to write to
        ``<wf_dir>/checkpoints/<sg>.py``.

        Raises :class:`CodegenError` if the LLM fails to produce a
        valid block after :attr:`max_retries`.
        """
        ctx.subgraph_runner = self

        # 1. Re-exec each subgraph_agent source to materialize Subgraph
        #    objects for the sandbox. These are the SAME objects whose
        #    _set_source_block + _render_checkpoints_module will produce
        #    the final sidecars.
        subgraphs_by_name: dict[str, Any] = {}
        for sg_name, src in sg_sources.items():
            sub_dict, _, _, _ = _exec_subgraph_builder(src)
            if sub_dict is None:
                raise CodegenError(
                    f"checkpoint_agent: failed to re-exec subgraph_agent "
                    f"source for {sg_name!r}"
                )
            # _exec_subgraph_builder only returns the dict; we also need
            # the raw sg object. Re-exec into a fresh sandbox we own.
            sandbox = _new_checkpoint_sandbox()
            exec(compile(src, f"<sg:{sg_name}>", "exec"), sandbox)
            sg_obj = sandbox.get("sg")
            if sg_obj is None:
                raise CodegenError(
                    f"checkpoint_agent: builder block for {sg_name!r} "
                    f"did not bind `sg`"
                )
            subgraphs_by_name[sg_name] = sg_obj

        # 2. Assemble prompt + retry loop.
        prompt = self.assembler.assemble_checkpoint_agent(
            workflow_dict=workflow_dict,
            sg_sources=sg_sources,
            task_prompt=task_prompt,
            subgraphs=subgraphs_by_name,
        )
        system_prompt = _inject_feedback(prompt.system_prompt, ctx.feedback)

        if ctx.trace_dir:
            d = ctx.trace_dir / "_checkpoint_agent"
            d.mkdir(parents=True, exist_ok=True)
            (d / "system_prompt.md").write_text(system_prompt, encoding="utf-8")
            (d / "user_prompt.md").write_text(prompt.user_prompt or "", encoding="utf-8")

        conversation: list[dict] = [
            {"role": "user", "content": prompt.user_prompt or ""},
        ]
        last_error = ""
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                conversation.append({
                    "role": "user",
                    "content": (
                        "Your previous response had validation errors:\n"
                        f"```\n{last_error}\n```\n\nFix every error and re-emit the "
                        "full ```python``` block containing only "
                        "`subgraphs[\"<sg>\"].add_checkpoint(...)` calls."
                    ),
                })
            llm_cfg = _resolve_llm(ctx.config, role="checkpoint_agent")
            raw = await self._call_llm(
                llm_cfg, system_prompt, conversation,
                bound_codegen_tools=prompt.bound_codegen_tools, ctx=ctx,
            )
            conversation.append({"role": "assistant", "content": raw})
            if ctx.trace_dir:
                d = ctx.trace_dir / "_checkpoint_agent"
                (d / f"llm_response_attempt_{attempt}.md").write_text(raw, encoding="utf-8")

            try:
                block = _parse_checkpoint_block(raw)
                _validate_checkpoint_block_static(block, subgraphs_by_name)
                # Reset _checkpoints on each subgraph before exec'ing
                # the candidate block — retries shouldn't accumulate
                # half-applied checkpoint declarations across attempts.
                for sg_obj in subgraphs_by_name.values():
                    sg_obj._checkpoints = []
                _exec_checkpoint_block(block, subgraphs_by_name)
                _validate_checkpoints_per_subgraph(subgraphs_by_name, block)
            except _CheckpointAuthorError as exc:
                last_error = str(exc)
                continue
            break
        else:
            raise CodegenError(
                f"checkpoint_agent failed after {self.max_retries + 1} "
                f"attempts: {last_error}"
            )

        # 3. AST-split per subgraph; render sidecars.
        per_sg_block = _split_block_by_subgraph(block, set(subgraphs_by_name))
        sidecars: dict[str, str] = {}
        for sg_name, sg_obj in subgraphs_by_name.items():
            structure_src = sg_sources[sg_name].rstrip("\n")
            cp_src = per_sg_block.get(sg_name, "").rstrip("\n")
            combined = (
                structure_src + "\n\n" + cp_src + "\n"
                if cp_src else structure_src + "\n"
            )
            sg_obj._set_source_block(combined)
            sidecars[sg_name] = sg_obj._render_checkpoints_module()
        return sidecars

    # ------------------------------------------------------------------
    # coder
    # ------------------------------------------------------------------

    def run_coder_sync(self, coder_spec: dict, ctx: CodegenContext) -> dict:
        """Synchronous helper used by ``request_inline_script``.

        Most ad-hoc scripts can be inlined directly by subgraph_agent in
        the same response (the preferred path). This entry exists for
        the cases where the subgraph_agent explicitly hands off via the
        meta-tool. Synchronous because the meta-tool is invoked from
        within the calling LLM's tool dispatch.
        """
        import asyncio
        import concurrent.futures

        from .llm import complete

        prompt = self.assembler.assemble_coder(coder_spec)
        sg_name = coder_spec.get("subgraph_name", "_global")

        async def _go() -> str:
            conversation = [{"role": "user", "content": coder_spec.get("purpose", "")}]
            return await complete(
                _resolve_llm(ctx.config, role="coder"),
                system=prompt.system_prompt, messages=conversation,
            )

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            raw = asyncio.run(_go())
        else:
            # Invoked from inside the provider's async tool loop — run the
            # nested LLM call on its own event loop in a worker thread.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                raw = pool.submit(asyncio.run, _go()).result()

        scripts = _parse_python_blocks(raw)
        if not scripts:
            raise CodegenError(f"coder produced no python script for {coder_spec['name']!r}")
        # Take the first emitted script.
        script_path, content = next(iter(scripts.items()))
        if not script_path.startswith("scripts/"):
            script_path = f"scripts/{sg_name}/{coder_spec['name']}.py"
        ctx.inline_scripts[script_path] = content
        return {"script_path": script_path, "content": content, "schema": {}, "raw": raw}

    # ------------------------------------------------------------------
    # LLM dispatch (tool-use loop lives in gap.agent.llm)
    # ------------------------------------------------------------------

    async def _call_llm(
        self,
        config: Any,
        system_prompt: str,
        conversation: list[dict],
        *,
        bound_codegen_tools: list,
        ctx: CodegenContext,
        max_tool_rounds: int = 6,
    ) -> str:
        """Single LLM call; codegen meta-tools bind through the provider
        tool-use loop when the agent declared any."""
        from .llm import complete, complete_with_tools

        messages = [dict(m) for m in conversation]
        if not bound_codegen_tools:
            return await complete(config, system=system_prompt, messages=messages)

        tool_descriptors = [
            _to_tool_schema(d) for d in bound_codegen_tools
        ]

        def handler(name: str, kwargs: dict) -> Any:
            return self._invoke_codegen_tool(name, kwargs, ctx)

        return await complete_with_tools(
            config,
            system=system_prompt,
            messages=messages,
            tools=tool_descriptors,
            tool_handler=handler,
            max_rounds=max_tool_rounds,
        )

    def _invoke_codegen_tool(self, name: str, kwargs: dict, ctx: CodegenContext) -> Any:
        descriptor = ctx.tool_registry.codegen_tools().get(name)
        if descriptor is None:
            raise KeyError(f"codegen tool {name!r} not registered")
        return ctx.tool_registry.invoke(name, ctx=ctx, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class CodegenError(RuntimeError):
    pass


class _CheckpointAuthorError(RuntimeError):
    """Raised inside the checkpoint_agent retry loop on a recoverable
    validation failure — parse error, AST shape violation, missing
    bound_outputs key, or per-subgraph rule miss. The retry handler
    appends the message to the next attempt's conversation; if all
    attempts exhaust, the runner re-raises as :class:`CodegenError`."""


def _new_checkpoint_sandbox(extra: dict | None = None) -> dict:
    """Build the exec sandbox shared by subgraph re-exec and the
    checkpoint-agent's block. Matches the pre-binding shape the
    builder/core test suite already relies on (`np`, `math`,
    `Subgraph`, `Ref`, `START`, `END`)."""
    import math as _math

    import numpy as _np

    from gap.builder import END as _END
    from gap.builder import START as _START
    from gap.builder import Ref as _Ref
    from gap.builder import Subgraph as _Subgraph
    sandbox: dict[str, Any] = {
        "__name__": "__checkpoint_agent__",
        "np": _np,
        "math": _math,
        "Subgraph": _Subgraph,
        "Ref": _Ref,
        "START": _START,
        "END": _END,
    }
    if extra:
        sandbox.update(extra)
    return sandbox


def _parse_checkpoint_block(raw: str) -> str:
    """Extract the single ``python`` fenced block (no file path) from the
    checkpoint_agent's response. Raises :class:`_CheckpointAuthorError`
    when zero or multiple candidate blocks are present."""
    blocks: list[str] = []
    for m in _FENCE_RE.finditer(raw):
        lang, fname, body = m.group(1), m.group(2), m.group(3)
        if lang in ("python", "py") and not fname:
            blocks.append(body.strip())
    if not blocks:
        raise _CheckpointAuthorError(
            "no parseable checkpoint block found; emit exactly ONE "
            "```python``` fenced block (no file path) containing only "
            "`subgraphs[\"<sg>\"].add_checkpoint(...)` statements."
        )
    if len(blocks) > 1:
        raise _CheckpointAuthorError(
            f"multiple python blocks found ({len(blocks)}); emit exactly "
            "ONE ```python``` block."
        )
    return blocks[0]


def _validate_checkpoint_block_static(
    block: str, subgraphs: dict[str, Any],
) -> None:
    """AST-walk-only checks (no exec) so retry hints are precise.

    Rejects anything other than top-level
    ``subgraphs["<sg>"].add_checkpoint(...)`` calls. Confirms the
    referenced ``<sg>`` is a known subgraph. Confirms 2-arg lambdas
    only read ``o[...]`` keys that exist in the receiver subgraph's
    bound outputs.
    """
    import ast as _ast
    try:
        tree = _ast.parse(block)
    except SyntaxError as exc:
        raise _CheckpointAuthorError(f"block has SyntaxError: {exc}") from None

    bound_outputs_by_sg: dict[str, set[str]] = {
        name: set((sg._outputs or {}).keys())
        for name, sg in subgraphs.items()
    }

    for stmt in tree.body:
        if not isinstance(stmt, _ast.Expr) or not isinstance(stmt.value, _ast.Call):
            raise _CheckpointAuthorError(
                f"only `subgraphs[...].add_checkpoint(...)` expressions are "
                f"allowed at module top level; saw {_ast.dump(stmt)[:120]}"
            )
        call = stmt.value
        func = call.func
        if not (
            isinstance(func, _ast.Attribute)
            and func.attr == "add_checkpoint"
            and isinstance(func.value, _ast.Subscript)
            and isinstance(func.value.value, _ast.Name)
            and func.value.value.id == "subgraphs"
            and isinstance(func.value.slice, _ast.Constant)
            and isinstance(func.value.slice.value, str)
        ):
            raise _CheckpointAuthorError(
                "every statement must call "
                "`subgraphs[\"<sg>\"].add_checkpoint(...)` with a string-"
                "literal subgraph name."
            )
        sg_name = func.value.slice.value
        if sg_name not in subgraphs:
            raise _CheckpointAuthorError(
                f"`subgraphs[{sg_name!r}]` is not a real subgraph; valid "
                f"names: {sorted(subgraphs)!r}"
            )
        # Inspect the predicate kwarg for 2-arg o[...] reads.
        for kw in call.keywords:
            if kw.arg != "predicate":
                continue
            if not isinstance(kw.value, _ast.Lambda):
                continue
            args = kw.value.args.args
            if len(args) < 2:
                continue
            o_arg_name = args[1].arg
            allowed = bound_outputs_by_sg.get(sg_name, set())
            for node in _ast.walk(kw.value.body):
                if (
                    isinstance(node, _ast.Subscript)
                    and isinstance(node.value, _ast.Name)
                    and node.value.id == o_arg_name
                    and isinstance(node.slice, _ast.Constant)
                    and isinstance(node.slice.value, str)
                ):
                    key = node.slice.value
                    if key not in allowed:
                        raise _CheckpointAuthorError(
                            f"checkpoint on `{sg_name}` reads `{o_arg_name}[{key!r}]` "
                            f"but `{sg_name}` does not bind that key in "
                            f"set_outputs(...). Bound keys: {sorted(allowed)!r}."
                        )


def _exec_checkpoint_block(
    block: str, subgraphs: dict[str, Any],
) -> None:
    """Exec the validated block in a sandbox where `subgraphs` is the
    dict of already-built :class:`Subgraph` objects. After this call,
    each subgraph's ``_checkpoints`` list is populated."""
    sandbox = _new_checkpoint_sandbox(extra={"subgraphs": subgraphs})
    try:
        exec(compile(block, "<checkpoint_agent>", "exec"), sandbox)
    except Exception as exc:
        raise _CheckpointAuthorError(
            f"block exec raised {type(exc).__name__}: {exc}"
        ) from None


_GRASP_SKILLS: frozenset[str] = frozenset(
    {"grasping-with-planner", "grasping-short-axis", "grasping-direct-ik"}
)
_TRANSPORT_SKILLS: frozenset[str] = frozenset({"transporting-objects"})


def _is_subscript_chain_on(node: Any, o_name: str) -> bool:
    """True when *node* is ``<o_name>[...]...[...]`` (any subscript depth)."""
    import ast as _ast
    while isinstance(node, _ast.Subscript):
        node = node.value
    return isinstance(node, _ast.Name) and node.id == o_name


def _block_references_z_for_sg(block_source: str, sg_name: str) -> bool:
    """Walk the LLM-emitted checkpoint block's AST. Return True if any
    ``subgraphs["<sg_name>"].add_checkpoint(...)`` call carries a
    predicate (2-arg lambda) whose body reads
    ``<o>[...]["position"]["z"]`` / ``<o>[...]["center"]["z"]`` (or the
    bare ``<o>[...]["z"]`` for Vec3-valued outputs like drop_position) —
    the output-anchored z-clearance pattern over gap.types TypedDicts.
    """
    import ast as _ast
    try:
        tree = _ast.parse(block_source)
    except SyntaxError:
        return False
    for stmt in tree.body:
        if not isinstance(stmt, _ast.Expr) or not isinstance(stmt.value, _ast.Call):
            continue
        call = stmt.value
        f = call.func
        # Match subgraphs["<sg_name>"].add_checkpoint(...)
        if not (
            isinstance(f, _ast.Attribute)
            and f.attr == "add_checkpoint"
            and isinstance(f.value, _ast.Subscript)
            and isinstance(f.value.value, _ast.Name)
            and f.value.value.id == "subgraphs"
            and isinstance(f.value.slice, _ast.Constant)
            and f.value.slice.value == sg_name
        ):
            continue
        for kw in call.keywords:
            if kw.arg != "predicate" or not isinstance(kw.value, _ast.Lambda):
                continue
            args = kw.value.args.args
            if len(args) < 2:
                continue
            o_name = args[1].arg
            for node in _ast.walk(kw.value.body):
                # A subscript whose key is the string "z" ...
                if not (
                    isinstance(node, _ast.Subscript)
                    and isinstance(node.slice, _ast.Constant)
                    and node.slice.value == "z"
                ):
                    continue
                inner = node.value
                # Case A: <o>[key]["position"]["z"] or <o>[key]["center"]["z"]
                if (
                    isinstance(inner, _ast.Subscript)
                    and isinstance(inner.slice, _ast.Constant)
                    and inner.slice.value in ("position", "center")
                    and _is_subscript_chain_on(inner.value, o_name)
                ):
                    return True
                # Case B: <o>[key]["z"] (the value at that key is a Vec3
                # — drop_position is the canonical example).
                if _is_subscript_chain_on(inner, o_name):
                    return True
    return False


def _validate_checkpoints_per_subgraph(
    subgraphs: dict[str, Any], block_source: str = "",
) -> None:
    """Confirm every subgraph received ≥ 1 ``validate=True`` checkpoint
    with non-empty rationale, ≤ 6 total. Plus per-skill rules:
    grasp/transport subgraphs MUST include at least one 2-arg predicate
    that checks the z-axis (``o["..."]["position"]["z"]`` /
    ``["center"]["z"]``), otherwise a sub-table grasp pose or a
    mis-computed drop height surfaces only downstream as a
    cascade-failure and the feedback misidentifies the root cause."""
    issues: list[str] = []
    for sg_name, sg in subgraphs.items():
        cps = list(getattr(sg, "_checkpoints", []) or [])
        if len(cps) == 0:
            issues.append(
                f"`{sg_name}` has no checkpoints; declare ≥ 1 `validate=True`."
            )
            continue
        if len(cps) > 6:
            issues.append(
                f"`{sg_name}` has {len(cps)} checkpoints (max 6); trim to "
                f"the most informative ones."
            )
        has_validate = any(bool(c.validate) for c in cps)
        if not has_validate:
            issues.append(
                f"`{sg_name}` has no `validate=True` checkpoint; the hard "
                f"postcondition is required for verification + feedback."
            )
        for c in cps:
            if bool(c.validate) and not (c.rationale and c.rationale.strip()):
                issues.append(
                    f"`{sg_name}.{c.name}` has validate=True but empty "
                    f"`rationale=`; the feedback prompt needs the "
                    f"rationale to explain what failed."
                )
        # Per-skill axis-coverage rule. Avoids the "xy-only check leaves
        # an upstream z bug invisible" failure mode observed on
        # cream-cheese (compute_grasp emitted z=-0.02; xy was fine; the
        # grasp_pose_over_target predicate passed; target_held failed
        # downstream and the feedback misattributed it to 'planning').
        skill = (getattr(sg, "_skill", None) or "").strip()
        if not block_source:
            continue  # can't AST-check without the source
        bound_keys = set((getattr(sg, "_outputs", None) or {}).keys())
        if skill in _GRASP_SKILLS:
            # Only enforce when the subgraph actually binds a grasp-pose-like
            # output (otherwise the LLM has nothing to anchor on). Prefer
            # `grasp_pose` (the computed planner target) over
            # `ee_pose_at_grasp` (the live EE pose at observe, which is the
            # approach pose and would pass z>0.01 trivially).
            if "grasp_pose" in bound_keys or "ee_pose_at_grasp" in bound_keys:
                if not _block_references_z_for_sg(block_source, sg_name):
                    target_key = (
                        "grasp_pose" if "grasp_pose" in bound_keys
                        else "ee_pose_at_grasp"
                    )
                    issues.append(
                        f"`{sg_name}` (skill={skill!r}) binds `{target_key}` "
                        f"but no checkpoint reads `o[...]['position']['z']` — "
                        f"required to catch sub-table grasp poses before "
                        f"they cascade into a `target_held` failure. Add: "
                        f"`predicate=lambda w, o: o[{target_key!r}]['position']['z'] "
                        f"> 0.01` with validate=True. Use `grasp_pose` "
                        f"(planner target) when available; "
                        f"`ee_pose_at_grasp` is the live approach pose at "
                        f"z≈0.35 and trivially passes."
                    )
        elif skill in _TRANSPORT_SKILLS:
            # Only enforce when the subgraph binds a drop-position-like
            # output. Some transport variants don't expose any output —
            # those rely on 1-arg privileged-only checks (`target_in_container`).
            if any(k in bound_keys for k in ("drop_position", "drop_pose")):
                if not _block_references_z_for_sg(block_source, sg_name):
                    issues.append(
                        f"`{sg_name}` (skill={skill!r}) binds a drop-position "
                        f"output but no checkpoint reads `o[...]['z']` — required "
                        f"to catch mis-computed drop heights. Add a checkpoint "
                        f"like: `predicate=lambda w, o: o['drop_position']['z'] > "
                        f"w.body('<container>').cavity_lower[2] - 0.01`."
                    )
    if issues:
        raise _CheckpointAuthorError("\n".join(issues))


def _split_block_by_subgraph(
    block: str, known_sg_names: set[str],
) -> dict[str, str]:
    """AST-walk the validated block; emit per-subgraph Python text with
    each ``subgraphs["<X>"].add_checkpoint(...)`` rewritten as
    ``sg.add_checkpoint(...)`` so the sidecar (which only has a bare
    `sg` in scope) re-execs cleanly."""
    import ast as _ast
    tree = _ast.parse(block)
    by_sg: dict[str, list[str]] = {name: [] for name in known_sg_names}
    for stmt in tree.body:
        call = stmt.value
        sg_name = call.func.value.slice.value
        # Rewrite receiver to `sg`.
        rewritten = _ast.Expr(
            value=_ast.Call(
                func=_ast.Attribute(
                    value=_ast.Name(id="sg", ctx=_ast.Load()),
                    attr="add_checkpoint",
                    ctx=_ast.Load(),
                ),
                args=call.args,
                keywords=call.keywords,
            )
        )
        _ast.fix_missing_locations(rewritten)
        by_sg[sg_name].append(_ast.unparse(rewritten))
    return {name: "\n".join(lines) for name, lines in by_sg.items() if lines}


def _to_tool_schema(descriptor: Any) -> dict:
    """Translate a ToolDescriptor into the canonical ``{name, description,
    input_schema}`` tool shape (translated per-provider in gap.agent.llm)."""
    schema = descriptor.schema
    properties: dict[str, dict] = {}
    required: list[str] = []
    for fname, finfo in schema.inputs.items():
        properties[fname] = _field_info_to_json_schema(finfo)
        if finfo.required:
            required.append(fname)
    input_schema: dict = {"type": "object", "properties": properties}
    if required:
        input_schema["required"] = required
    return {
        "name": descriptor.name,
        "description": descriptor.summary or "",
        "input_schema": input_schema,
    }


def _field_info_to_json_schema(finfo: Any) -> dict:
    type_str = (finfo.type_str or "").lower()
    if type_str.startswith("list"):
        return {"type": "array", "description": finfo.description or ""}
    if "int" in type_str:
        return {"type": "integer", "description": finfo.description or ""}
    if "float" in type_str or "double" in type_str:
        return {"type": "number", "description": finfo.description or ""}
    if "bool" in type_str:
        return {"type": "boolean", "description": finfo.description or ""}
    if "dict" in type_str:
        return {"type": "object", "description": finfo.description or ""}
    return {"type": "string", "description": finfo.description or ""}


_FENCE_RE = re.compile(r"```(\w+)(?::([^\n]+))?\n(.*?)```", re.DOTALL)


def _parse_first_json(text: str) -> dict | None:
    for m in _FENCE_RE.finditer(text):
        lang, _, body = m.group(1), m.group(2), m.group(3)
        if lang == "json":
            try:
                parsed = json.loads(body.strip())
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def _parse_coordinator_response(raw: str) -> dict | None:
    """Parse a coordinator response.

    Prefers a path-less ``python`` fenced block that builds a
    ``gap.builder.WorkflowSpec`` and binds it to ``spec``. Falls back to
    a legacy ``json`` block if no Python block is present.
    """
    builder_block: str | None = None
    json_dict: dict | None = None
    for m in _FENCE_RE.finditer(raw):
        lang, fname, body = m.group(1), m.group(2), m.group(3)
        body = body.strip()
        if lang in ("python", "py") and not fname:
            if builder_block is None and (
                "WorkflowSpec(" in body or "WorkflowSpec." in body
            ):
                builder_block = body
            elif builder_block is None:
                builder_block = body
        elif lang == "json" and json_dict is None:
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                json_dict = parsed
    if builder_block is not None:
        py_dict = _exec_workflow_spec_builder(builder_block)
        if py_dict is not None:
            return py_dict
    return json_dict


def _exec_workflow_spec_builder(code: str) -> dict | None:
    """Exec a Python WorkflowSpec block; return the resulting spec dict.

    Returns ``None`` on any execution error or if the block does not bind
    a ``WorkflowSpec`` instance to a module-level ``spec`` variable.
    """
    try:
        from gap.builder import START, WorkflowSpec
        from gap.runtime.workflow import ToolCall
    except Exception:
        logger.exception("gap.builder import failed")
        return None
    sandbox: dict = {
        "__builtins__": __builtins__,
        "WorkflowSpec": WorkflowSpec,
        "START": START,
        "ToolCall": ToolCall,
    }
    try:
        exec(compile(code, "<coordinator>", "exec"), sandbox)
    except Exception as e:
        logger.warning("coordinator builder exec failed: %s", e)
        return None
    spec = sandbox.get("spec")
    if spec is None or not isinstance(spec, WorkflowSpec):
        logger.warning(
            "coordinator block did not bind a WorkflowSpec to `spec` (got %r)",
            type(spec).__name__ if spec is not None else None,
        )
        return None
    try:
        return spec.to_dict()
    except Exception as e:
        logger.warning("WorkflowSpec.to_dict() failed: %s", e)
        return None


def _parse_subgraph_response(
    raw: str,
) -> tuple[dict | None, dict[str, str], str | None, list[dict], str | None, str | None]:
    """Parse a subgraph_agent response.

    Looks for one of:
      - A ``python`` fenced block (no file path) that builds a
        ``gap.builder.Subgraph`` and binds it to a module-level ``sg``
        variable. The block is exec'd in a sandbox and the resulting
        subgraph is converted to a v3 SubgraphDef dict via ``.to_dict()``.
      - Failing that, a legacy ``json`` fenced block holding the
        SubgraphDef directly.

    Inline script blocks (``python:scripts/<sg>/<file>.py``) are collected
    separately regardless of which subgraph form was used.

    Returns ``(sg_dict_or_None, scripts, checkpoint_module_or_None,
    checkpoint_meta, builder_block_or_None, builder_error_or_None)``. The
    ``builder_block`` is the verbatim Python source the LLM emitted (the
    block that bound ``sg``), used downstream by the checkpoint_agent's
    sandbox to re-materialize the subgraph object; ``None`` on the
    legacy JSON path. ``builder_error`` describes why a present builder
    block failed to produce a subgraph (so the retry loop can feed the
    real mistake back), and is ``None`` when the block succeeded or none
    was present.
    """
    sg_dict: dict | None = None
    scripts: dict[str, str] = {}
    checkpoint_module: str | None = None
    checkpoint_meta: list[dict] = []
    builder_block: str | None = None
    for m in _FENCE_RE.finditer(raw):
        lang, fname, body = m.group(1), m.group(2), m.group(3)
        body = body.strip()
        if lang in ("python", "py"):
            if fname:
                scripts[fname] = body
            elif builder_block is None and "Subgraph(" in body:
                builder_block = body
            elif builder_block is None:
                # First path-less python block, even without Subgraph(...),
                # is a candidate; exec will fail later if it's not the
                # builder block, but we keep it for the parse attempt.
                builder_block = body
        elif lang == "json" and sg_dict is None:
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                if "nodes" in parsed and "edges" in parsed:
                    sg_dict = parsed
                elif len(parsed) == 1:
                    inner = next(iter(parsed.values()))
                    if isinstance(inner, dict) and "nodes" in inner and "edges" in inner:
                        sg_dict = inner
    # Python builder block wins over legacy JSON when both are present.
    builder_error: str | None = None
    if builder_block is not None:
        py_dict, checkpoint_module, checkpoint_meta, builder_error = (
            _exec_subgraph_builder(builder_block)
        )
        if py_dict is not None:
            sg_dict = py_dict
            builder_error = None
    # LLMs occasionally emit the prompt's path TEMPLATE literally —
    # ``scripts/<sg>/file.py`` — in both the fence path and the node's
    # ``script:`` field. Substitute the real subgraph name on both sides
    # (they reference each other, so the rewrite must be consistent).
    sg_name = (sg_dict or {}).get("name")
    if sg_dict is not None and isinstance(sg_name, str) and sg_name:
        scripts = {
            key.replace("<sg>", sg_name): body
            for key, body in scripts.items()
        }
        for node in (sg_dict.get("nodes") or {}).values():
            script = node.get("script") if isinstance(node, dict) else None
            if isinstance(script, str) and "<sg>" in script:
                node["script"] = script.replace("<sg>", sg_name)
    return sg_dict, scripts, checkpoint_module, checkpoint_meta, builder_block, builder_error


def _checkpoint_meta_from_subgraph(sg: Any) -> list[dict]:
    """Lightweight, JSON-able view of a subgraph's declared checkpoints.

    Mirrors :class:`gap.builder.core._CheckpointDef`.
    """
    out: list[dict] = []
    for c in getattr(sg, "_checkpoints", []) or []:
        out.append({
            "name": c.name,
            "validate": bool(c.validate),
            "rationale": str(c.rationale or ""),
            "has_diagnostics": c.diagnostics is not None,
        })
    return out


def _exec_subgraph_builder(
    code: str,
) -> tuple[dict | None, str | None, list[dict], str | None]:
    """Exec a Python builder block; return ``(sg_dict, checkpoint_module,
    checkpoint_meta, error)``.

    ``sg_dict`` is ``None`` on any execution error or if the block does
    not bind a ``Subgraph`` instance to a module-level ``sg`` variable
    (in which case ``checkpoint_module`` is ``None`` and ``checkpoint_meta``
    is ``[]``). ``error`` is a human-readable description of the failure on
    the ``None`` path (fed back to the agent's retry loop so it can fix the
    actual builder mistake — e.g. a literal where a ``Ref()`` was required —
    instead of the generic "no parseable subgraph" guidance) and ``None`` on
    success. ``checkpoint_module`` is the rendered sidecar source (str)
    when the subgraph declared any ``add_checkpoint`` calls, else ``None``.

    ``numpy`` and ``math`` are pre-bound in the sandbox under the names
    ``np`` and ``math`` so that builder-block expressions outside of
    lambdas (e.g. ``inputs={"target_z": float(np.clip(...))}``) work
    without an explicit ``import``. Predicate lambdas reach numpy/math
    through the same sandbox via closure.
    """
    try:
        from gap.builder import END, START, Ref, Subgraph
    except Exception as e:
        logger.exception("gap.builder import failed")
        return None, None, [], f"gap.builder import failed: {e}"
    try:
        import math as _math

        import numpy as _np
    except Exception:
        _math = None
        _np = None
    sandbox: dict = {
        "__builtins__": __builtins__,
        "Subgraph": Subgraph,
        "Ref": Ref,
        "START": START,
        "END": END,
    }
    if _np is not None:
        sandbox["np"] = _np
    if _math is not None:
        sandbox["math"] = _math
    # Robustness: the subgraph_agent's prompt instructs it NOT to call
    # `sg.add_checkpoint(...)` (the whole-workflow checkpoint_agent
    # handles that). The LLM sometimes ignores the instruction and emits
    # one anyway — frequently with a string predicate that would crash
    # exec. Strip any top-level `sg.add_checkpoint(...)` call before
    # exec so structure generation is independent of checkpoint-emission
    # noise. The checkpoint_agent re-execs the same source later in its
    # own sandbox; that path adds the real checkpoints.
    try:
        code = _strip_add_checkpoint_calls(code)
    except Exception:
        pass  # best-effort; if AST parse fails, exec will surface it.
    try:
        exec(compile(code, "<subgraph_agent>", "exec"), sandbox)
    except Exception as e:
        logger.warning("subgraph builder exec failed: %s", e)
        return None, None, [], f"builder block raised {type(e).__name__}: {e}"
    sg = sandbox.get("sg")
    if sg is None or not isinstance(sg, Subgraph):
        got = type(sg).__name__ if sg is not None else None
        logger.warning(
            "subgraph builder block did not bind a Subgraph to `sg` (got %r)",
            got,
        )
        return None, None, [], (
            f"builder block did not bind a `Subgraph` instance to a "
            f"module-level `sg` variable (got {got!r})"
        )
    try:
        sg_dict = sg.to_dict()
    except Exception as e:
        logger.warning("Subgraph.to_dict() failed: %s", e)
        return None, None, [], f"Subgraph.to_dict() raised {type(e).__name__}: {e}"
    checkpoint_meta = _checkpoint_meta_from_subgraph(sg)
    checkpoint_module: str | None = None
    if sg._checkpoints:
        sg._set_source_block(code)
        try:
            checkpoint_module = sg._render_checkpoints_module()
        except Exception as e:
            logger.warning(
                "Subgraph._render_checkpoints_module() failed: %s", e,
            )
            checkpoint_module = None
    return sg_dict, checkpoint_module, checkpoint_meta, None


def _strip_add_checkpoint_calls(code: str) -> str:
    """Drop top-level ``sg.add_checkpoint(...)`` calls from a builder
    block. The subgraph_agent isn't supposed to emit them under the
    two-agent split — but the LLM sometimes does anyway, and a
    malformed predicate kwarg (e.g. a string instead of a lambda)
    crashes the structure exec. Stripping is silent: the
    checkpoint_agent re-execs the LLM source in its own sandbox and
    adds the real checkpoints there.
    """
    import ast as _ast
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return code
    new_body: list[_ast.stmt] = []
    for stmt in tree.body:
        if (
            isinstance(stmt, _ast.Expr)
            and isinstance(stmt.value, _ast.Call)
            and isinstance(stmt.value.func, _ast.Attribute)
            and stmt.value.func.attr == "add_checkpoint"
        ):
            # Drop this statement.
            continue
        new_body.append(stmt)
    tree.body = new_body
    return _ast.unparse(tree)


def _parse_python_blocks(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in _FENCE_RE.finditer(raw):
        lang, fname, body = m.group(1), m.group(2), m.group(3)
        if lang in ("python", "py"):
            key = fname or f"scripts/_unnamed_{len(out)}.py"
            out[key] = body.strip()
    return out


def _extract_missing(parsed: dict | None, raw: str) -> list[dict]:
    if isinstance(parsed, dict) and "missing" in parsed and isinstance(parsed["missing"], list):
        return [
            {"name": str(x.get("name", "")), "why": str(x.get("why", ""))}
            for x in parsed["missing"]
            if isinstance(x, dict)
        ]
    return []


def _namespace_scripts(
    sg_name: str, scripts: dict[str, str], sg_dict: dict,
) -> dict[str, str]:
    target_prefix = f"scripts/{sg_name}/"
    out: dict[str, str] = {}
    rename: dict[str, str] = {}
    for path, content in scripts.items():
        if path.startswith(target_prefix):
            out[path] = content
            continue
        bare = path.removeprefix("scripts/")
        new_path = target_prefix + bare
        out[new_path] = content
        rename[path] = new_path
    if rename:
        for node in sg_dict.get("nodes", {}).values():
            if isinstance(node, dict) and node.get("type") in ("script", "router"):
                old = node.get("script")
                if isinstance(old, str) and old in rename:
                    node["script"] = rename[old]
    return out


def _resolve_llm(config: Any, role: str) -> Any:
    """Per-role LLM config override (uses composition.coordinator_model etc.)."""
    from dataclasses import replace
    if config is None or not hasattr(config, "llm"):
        raise ValueError("CodegenContext.config must carry an `llm` field")
    base = config.llm
    comp = getattr(config, "composition", None)
    if comp is None:
        return base
    model = base.model
    temperature = base.temperature
    if role == "coordinator" and comp.coordinator_model:
        model = comp.coordinator_model
    elif role in ("subgraph_agent", "checkpoint_agent", "coder") and getattr(comp, "subgraph_model", None):
        model = comp.subgraph_model
        if comp.subgraph_temperature is not None:
            temperature = comp.subgraph_temperature
    return replace(base, model=model, temperature=temperature)


def _validate_workflow_spec(spec: dict, skills: SkillsRegistry) -> str:
    """Light parse-time check on the coordinator's v3 WorkflowSpec output."""
    issues: list[str] = []
    nodes = spec.get("nodes")
    if not isinstance(nodes, dict) or not nodes:
        issues.append("`nodes` must be a non-empty dict")
    edges = spec.get("edges")
    if not isinstance(edges, list) or not edges:
        issues.append("`edges` must be a non-empty list with at least [START, ...]")
    sgs = spec.get("subgraphs")
    if not isinstance(sgs, dict) or not sgs:
        issues.append("`subgraphs` must be a non-empty dict")
        return "\n".join(issues)
    for name, sg in sgs.items():
        if not isinstance(sg, dict):
            issues.append(f"subgraph {name!r} is not a dict")
            continue
        skill = sg.get("skill")
        if skill is None:
            issues.append(f"subgraph {name!r} missing `skill` field")
            continue
        if sg.get("generated"):
            # Invented skill: `skill` names a brand-new skill (NOT a
            # registered bundle) whose contract the coordinator authored
            # inline. Skip the registry-membership check; require only a
            # non-empty name + description so subgraph_agent has something
            # to implement.
            if not isinstance(skill, str) or not skill:
                issues.append(
                    f"generated subgraph {name!r} needs a non-empty `skill` name"
                )
            if not (sg.get("description") or "").strip():
                issues.append(
                    f"generated subgraph {name!r} needs a `description` "
                    f"(the contract the subgraph_agent implements)"
                )
            continue
        if skill not in skills:
            issues.append(
                f"subgraph {name!r} references unknown skill {skill!r} "
                f"(set generated=True on declare_subgraph to invent a new "
                f"skill, otherwise pick one from the Available Skills table)"
            )
    return "\n".join(issues)


def _structural_subgraph_errors(
    sg_name: str,
    sg_dict: dict,
    skills: SkillsRegistry,
) -> str:
    """Run the authoritative per-subgraph structural rules (S1-S11) on a
    just-generated subgraph, in-memory, and return a newline-joined string
    of error-severity issues (empty when clean).

    Reuses :func:`gap.runtime.validate._check_subgraph_level` so the
    authoring-time check matches the post-assembly validator exactly. The
    registry args mirror the post-assembly call in
    :func:`gap.agent.multi_agent._run_graph_validation`
    (``skill_registry=skills``, ``agent_registry=None``) so the two passes
    never disagree.

    Cross-subgraph rules (W8) and script-schema introspection are NOT run
    here — they need the assembled workflow / files on disk and stay at
    post-assembly. ``_parse_subgraph`` raises on malformed dicts; that
    message becomes feedback rather than crashing the pipeline.
    """
    from gap_core.errors import WorkflowValidationError

    from gap.runtime.validate import _check_subgraph_level
    from gap.runtime.workflow import _parse_subgraph

    try:
        sg_def = _parse_subgraph(sg_name, sg_dict)
    except WorkflowValidationError as e:
        return str(e)
    issues = _check_subgraph_level(sg_name, sg_def, skills, None)
    return "\n".join(str(i) for i in issues if i.severity == "error")


def _validate_subgraph(
    sg_dict: dict,
    spec: dict,
    skills: SkillsRegistry,
) -> str:
    """Light parse-time check on a v3 SubgraphDef.

    Defers heavyweight structural validation (reachability, $ref typing,
    streaming-flag/skill-contract consistency) to
    ``gap.runtime.validate.validate_workflow`` once the assembled
    workflow is ready. Here we only catch the most common LLM mistakes
    (missing nodes/edges, exit values mismatched to skill exit
    conditions, edges referencing unknown nodes).
    """
    issues: list[str] = []
    nodes = sg_dict.get("nodes")
    if not isinstance(nodes, dict) or not nodes:
        return "missing or empty `nodes`"
    edges = sg_dict.get("edges")
    if not isinstance(edges, list) or not edges:
        issues.append("missing or empty `edges`")
    valid_targets = {"START", "END"} | set(nodes.keys())
    for i, e in enumerate(edges or []):
        if not isinstance(e, list) or len(e) != 2:
            issues.append(f"edges[{i}] must be a [src, dst] pair")
            continue
        src, dst = e
        if src not in valid_targets:
            issues.append(f"edges[{i}] src {src!r} is not declared")
        if dst not in valid_targets:
            issues.append(f"edges[{i}] dst {dst!r} is not declared")

    exit_block = sg_dict.get("exit") or {}
    if "values" in exit_block:
        issues.append(
            "`exit.values` is no longer supported — use `exit.success_values` "
            "(success-path exits) and put the single failure exit in the "
            "subgraph's sibling `on_error` field"
        )
    success_values = list(exit_block.get("success_values") or [])
    if not success_values:
        issues.append("`exit.success_values` must be a non-empty list")
    on_error = sg_dict.get("on_error")

    # Exit conditions must match the contract. For a registered skill the
    # contract is the bundle's exit_conditions; for an invented (generated)
    # skill the contract is what the coordinator declared on the spec.
    skill_name = sg_dict.get("skill")
    if spec.get("generated"):
        declared = set((spec.get("exit") or {}).get("success_values") or [])
        spec_on_error = spec.get("on_error")
        if spec_on_error:
            declared.add(spec_on_error)
        present = set(success_values)
        if on_error is not None:
            present.add(on_error)
        if declared:
            missing = declared - present
            extra = present - declared
            if missing:
                issues.append(
                    f"generated skill missing declared exit values: {sorted(missing)}"
                )
            if extra:
                issues.append(
                    f"generated skill has exit values not in its declared "
                    f"contract: {sorted(extra)}"
                )
    elif skill_name and skill_name in skills:
        info = skills.get(skill_name)
        declared = set(info.meta.exit_conditions.keys())
        present = set(success_values)
        if on_error is not None:
            present.add(on_error)
        missing = declared - present
        extra = present - declared
        if missing:
            issues.append(f"missing required exit values: {sorted(missing)}")
        if extra:
            issues.append(f"unexpected exit values (not in skill exit_conditions): {sorted(extra)}")

    # on_error must NOT collide with a node name (S9) and must NOT appear in
    # success_values (it's the failure exit, not a success).
    if on_error is not None:
        if on_error in nodes:
            issues.append(
                f"on_error={on_error!r} must not be a declared node — "
                f"failure exits surface only via on_error, not as nodes"
            )
        if on_error in success_values:
            issues.append(
                f"on_error={on_error!r} must not also appear in "
                f"exit.success_values — it is the failure exit"
            )

    # Streaming nodes must have no outgoing edges
    streaming_nodes = {
        n for n, nd in nodes.items()
        if isinstance(nd, dict) and nd.get("streaming") is True
    }
    for _i, e in enumerate(edges or []):
        if isinstance(e, list) and len(e) == 2 and e[0] in streaming_nodes:
            issues.append(
                f"streaming node {e[0]!r} has outgoing edge to {e[1]!r}; "
                f"streaming nodes must be pure sources"
            )

    # Outputs
    declared_out = set(spec.get("outputs", {}).keys())
    bound_out = set((sg_dict.get("outputs") or {}).keys())
    missing_out = declared_out - bound_out
    if missing_out:
        issues.append(f"declared outputs not bound: {sorted(missing_out)}")

    return "\n".join(issues)


# Soft cap mirrors gap.builder.core._CHECKPOINT_SOFT_CAP — kept here so the
# parse-time check doesn't import a private builder constant.
_MAX_CHECKPOINTS_PER_SUBGRAPH = 6


# ---------------------------------------------------------------------------
# Prior-attempt feedback injection
# ---------------------------------------------------------------------------


def _inject_feedback(system_prompt: str, feedback: str | None) -> str:
    """Prepend a prior-attempt feedback section to a system prompt.

    The source pipeline's typed rehearsal/refine feedback hooks are not
    ported; ``CodegenContext.feedback`` is a plain string the caller may
    populate (e.g. with a failed run's checkpoint summary).
    """
    if not feedback or not feedback.strip():
        return system_prompt
    section = [
        "## Prior attempt feedback",
        "",
        feedback.strip(),
        "",
        "---",
        "",
    ]
    return "\n".join(section) + system_prompt
