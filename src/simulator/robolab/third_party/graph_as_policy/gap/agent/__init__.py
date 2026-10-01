"""gap.agent — LLM graph generation.

Compile a language instruction into a typed, verified workflow graph::

    import gap

    graph = gap.agent.generate_sync(
        "pick up the alphabet soup and put it in the basket",
    )                            # open-robot-skills checkout auto-discovered
    print(graph)                 # the compiled graph as terminal text
    result = gap.execute(graph.path, connector)

The pipeline runs a coordinator (topology), one subgraph_agent per
subgraph (structure + scripts), and a whole-workflow checkpoint_agent
(postcondition sidecars), then validates the result with
:func:`gap.runtime.validate.validate_workflow` and runs a bounded LLM
script-fix loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from .config import CompositionConfig, PipelineConfig
from .llm import LlmConfig, complete, complete_with_tools

logger = logging.getLogger(__name__)

__all__ = [
    "CompositionConfig",
    "GeneratedGraph",
    "LlmConfig",
    "PipelineConfig",
    "complete",
    "complete_with_tools",
    "generate",
    "generate_sync",
]


@dataclass
class GeneratedGraph:
    """Result of one :func:`generate` call."""

    path: Path
    """The written workflow folder (workflow.json + scripts/ +
    checkpoints/ + multi_agent_meta.json)."""

    workflow: dict
    """The parsed workflow.json contents."""

    code: dict[str, str]
    """Every generated source file keyed by workflow-relative path —
    inline scripts plus checkpoint sidecars."""

    def __str__(self) -> str:
        """The graph as box-drawing terminal text (:func:`gap.viz.to_text`)."""
        from gap.viz.text import to_text

        return to_text(self.workflow)


async def generate(
    instruction: str,
    *,
    skills: str | Path | Sequence[str | Path] | None = None,
    model: str | None = None,
    provider: str | None = None,
    out_dir: str | Path | None = None,
    config: PipelineConfig | str | Path | None = None,
) -> GeneratedGraph:
    """Generate a workflow graph from a language instruction.

    Args:
        instruction: The task in natural language.
        skills: Skill registry root(s) — one path or a precedence-ordered
            sequence (bundle discovery roots). When omitted, the active
            registries are resolved (``$GAP_SKILLS_PATH`` list > project
            ``[tool.gap]`` > user config > the open-robot-skills checkout
            next to the gap checkout); generation requires at least one,
            so resolution failure raises :class:`FileNotFoundError`
            listing what was tried.
        model: Optional LLM model override (default: provider default,
            ``gemini-3.1-flash-lite-preview``).
        provider: Optional LLM provider override
            (``openrouter`` | ``vertex``).
        out_dir: Output directory; the workflow folder is written to
            ``<out_dir>/task_00``. Defaults to
            ``outputs/generated_<timestamp>``.
        config: Optional :class:`PipelineConfig` (or a YAML path) for
            full control; ``skills``/``model``/``provider`` arguments
            override the corresponding config fields.

    Returns:
        :class:`GeneratedGraph`.

    Raises:
        RuntimeError: when generation fails (LLM retries exhausted,
            missing capabilities reported, or the assembled workflow
            does not load).
        FileNotFoundError: no skills path given, none in the config, and
            none discoverable.
    """
    from .multi_agent import run_codegen

    if config is None:
        cfg = PipelineConfig()
    elif isinstance(cfg_path := config, (str, Path)):
        cfg = PipelineConfig.from_yaml(cfg_path)
    else:
        cfg = config

    if skills is not None:
        from gap.skills import as_registry_paths

        cfg.skills = as_registry_paths(skills)
    elif cfg.skills is None:
        from gap.skills import resolve_registries

        cfg.skills = resolve_registries(required=True).paths()
    llm = cfg.llm
    if provider is not None:
        llm = replace(llm, provider=provider)
    if model is not None:
        llm = replace(llm, model=model)
    cfg.llm = llm

    if out_dir is None:
        out_dir = Path("outputs") / f"generated_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(out_dir)

    result = await run_codegen(
        task_id=0,
        task_prompt=instruction,
        config=cfg,
        output_dir=out_dir,
    )
    if not result.success or result.workflow_dir is None:
        raise RuntimeError(
            f"graph generation failed: {result.execution_stderr or 'unknown error'}"
        )

    code: dict[str, str] = dict(result.scripts)
    for sg_name, module_source in result.checkpoint_modules.items():
        if module_source:
            code[f"checkpoints/{sg_name}.py"] = module_source

    return GeneratedGraph(
        path=result.workflow_dir,
        workflow=json.loads(result.workflow_json),
        code=code,
    )


def generate_sync(
    instruction: str,
    *,
    skills: str | Path | Sequence[str | Path] | None = None,
    model: str | None = None,
    provider: str | None = None,
    out_dir: str | Path | None = None,
    config: PipelineConfig | str | Path | None = None,
) -> GeneratedGraph:
    """Synchronous wrapper around :func:`generate`."""
    return asyncio.run(generate(
        instruction,
        skills=skills,
        model=model,
        provider=provider,
        out_dir=out_dir,
        config=config,
    ))
