"""The concrete ablation modes + a factory.

Each is a few lines: the heavy lifting (one native ``launch()`` over
the whole suite) is the shared ``BenchmarkMode.run`` → ``eval_core``.
The only per-mode logic is *how* the per-task workflows are produced
(or that they are zero-shot codegen'd by ``launch()``).

The source's ``monolithic`` scaffold is deleted — it never scored a
cell and only produced error rows.
"""

from __future__ import annotations

from .modes import BenchmarkMode, ModeRequest
from .workflow_materialize import materialize_for_task


class LlmGenerationMode(BenchmarkMode):
    """Zero-shot multi-agent graph codegen authors the workflow.

    ``produce_workflows`` returns ``None``: ``eval_core``/``launch()``
    runs :func:`gap.agent.multi_agent.run_codegen` per task natively
    (one shot, no rehearsal, no refine).
    """

    name = "llm_generation"

    async def produce_workflows(self, req: ModeRequest) -> dict[int, str] | None:
        return None


class _TemplateMode(BenchmarkMode):
    """Shared base for modes that materialize a fixed workflow template
    per task. Subclasses set ``name``; the template comes from
    ``mode_overrides.<name>.workflow_dir`` in the benchmark YAML. The
    workflow embeds a ``run_policy`` node referencing a ``{{policy_id}}``
    placeholder; ``policy_id`` resolves from the harness' policy axis
    (``cfg.policies``) or the mode override (single-policy default).

    Returns ``{task_id: workflow_dir}`` for every task in the suite, so
    a single native ``launch()`` covers the whole suite with per-task
    target literals.
    """

    name = ""
    requires_policy = True

    async def produce_workflows(self, req: ModeRequest) -> dict[int, str] | None:
        template = req.extra.get("workflow_dir")
        if not template:
            raise ValueError(
                f"{self.name} requires mode_overrides.{self.name}."
                f"workflow_dir (the workflow template) in the benchmark YAML"
            )
        # ``policy_id`` is an optional knob — ``materialize_for_task``
        # falls back to ``pi05-libero`` when None.
        policy_id = req.extra.get("policy_id") or "pi05-libero"
        out: dict[int, str] = {}
        for tid in req.task_ids:
            out[tid] = materialize_for_task(
                template_dir=template,
                dest_parent=req.produce_dir / f"task_{tid:02d}",
                suite_name=req.suite_name,
                task_id=tid,
                policy_id=policy_id,
            )
        return out


class LlmPlusPolicyMode(_TemplateMode):
    """LLM-authored OBB approach + a VLA policy.

    The steered-policy template (setup + OBB target + Cartesian
    approach + run_policy), materialized per task — the OBB nodes +
    run_policy prompt carry per-task target literals.
    """

    name = "llm_plus_policy"


class PolicyOnlyMode(_TemplateMode):
    """Bare policy, no graph scaffolding (baseline for ``llm_plus_policy``).

    Home/open-gripper bring-up + the ``run_policy`` node — no
    perception, no OBB, no approach. Isolates exactly what the
    LLM-authored OBB pre-positioning in ``llm_plus_policy`` buys over
    the raw VLA on the OOD variations.
    """

    name = "policy_only"


_REGISTRY: dict[str, type[BenchmarkMode]] = {
    LlmGenerationMode.name: LlmGenerationMode,
    LlmPlusPolicyMode.name: LlmPlusPolicyMode,
    PolicyOnlyMode.name: PolicyOnlyMode,
}


def build_mode(name: str) -> BenchmarkMode:
    """Instantiate a mode by its config name."""
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise ValueError(
            f"unknown benchmark mode {name!r}; valid: {sorted(_REGISTRY)}"
        ) from None
