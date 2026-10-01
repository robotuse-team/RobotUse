"""Ablation modes — thin adapters over the shared produce→score seam.

Every mode implements one method, :meth:`BenchmarkMode.produce_workflows`,
which returns the per-task workflow map for a whole
``(family, variation)`` suite (or ``None`` to defer to ``launch()``'s
zero-shot codegen). :meth:`BenchmarkMode.run` then scores the whole
suite with the *same* :func:`gap.benchmark.eval_core.score_suite`
(one native ``launch()``), so the only thing that differs between
ablations is *how* the workflows are produced.

``produce_workflows`` returns either:
  * ``{task_id: workflow_dir}``  — per-task pre-built/templated
    workflows (``llm_plus_policy`` / ``policy_only``); or
  * ``None``  — ``launch()`` zero-shot-codegens each task
    (``llm_generation``).

A ``produce_workflows`` that raises becomes one ``error`` cell for the
whole ``(mode, family, variation)`` so a failed mode never aborts the
grid.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gap.agent.config import PipelineConfig

from .eval_core import score_suite
from .report import ModeResult, error_cell, normalize_task_results


@dataclass
class ModeRequest:
    """Everything a mode needs to produce + score one suite.

    A *cell* is one ``(mode, family, variation)`` = one suite run over
    ``task_ids`` (× ``n_seeds`` trials), scored by a single native
    ``launch()``.
    """

    family: str
    variation: str
    suite_name: str
    task_ids: list[int]
    n_seeds: int
    num_workers: int
    pipeline_config: PipelineConfig
    artifact_dir: Path
    record_video: bool = False
    # Mode-tuning knobs resolved from BenchmarkConfig.effective(mode):
    # e.g. {"workflow_dir": "<abs template>", "policy_id": ...}.
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def produce_dir(self) -> Path:
        """Where ``produce_workflows`` writes per-task workflows — a
        *sibling* of the score dir, because ``eval_core``/``launch()``
        cleans ``artifact_dir`` at start and would otherwise wipe them.
        """
        d = Path(self.artifact_dir)
        return d.with_name(d.name + "__wf")


class BenchmarkMode(ABC):
    """One ablation. Subclasses only implement ``produce_workflows``."""

    name: str = ""
    # Whether this mode's per-task workflow names a policy via
    # ``{{policy_id}}``. The harness uses this flag to decide whether to
    # iterate over ``cfg.policies`` for the mode (True) or run a single
    # cell with no policy axis (False — ``llm_generation``).
    requires_policy: bool = False

    @abstractmethod
    async def produce_workflows(
        self, req: ModeRequest
    ) -> dict[int, str] | None:
        """Per-task workflow map for the suite, or ``None`` for zero-shot.

        Raising is acceptable — :meth:`run` records one error cell so
        the grid keeps going.
        """
        raise NotImplementedError

    async def run(self, req: ModeRequest) -> ModeResult:
        """Produce (mode-specific) then score (shared) → one cell."""
        produce_dir = str(req.produce_dir)
        cell_policy_id = (
            (req.extra.get("policy_id") or "") if self.requires_policy else ""
        )
        t0 = time.perf_counter()
        try:
            workflow_dir_map = await self.produce_workflows(req)
        except Exception as e:  # noqa: BLE001 — never abort the grid
            return error_cell(
                mode=self.name,
                family=req.family,
                variation=req.variation,
                suite_name=req.suite_name,
                policy_id=cell_policy_id,
                error=f"{type(e).__name__}: {e}",
                out_dir=produce_dir,
            )
        produce_wall = time.perf_counter() - t0

        task_results, eval_wall = await score_suite(
            suite_name=req.suite_name,
            task_ids=req.task_ids,
            n_seeds=req.n_seeds,
            pipeline_config=req.pipeline_config,
            artifact_dir=req.artifact_dir,
            num_workers=req.num_workers,
            workflow_dir_map=workflow_dir_map,
            record_video=req.record_video,
        )
        return normalize_task_results(
            mode=self.name,
            family=req.family,
            variation=req.variation,
            suite_name=req.suite_name,
            policy_id=cell_policy_id,
            task_results=task_results,
            produce_wall_s=produce_wall,
            eval_wall_s=eval_wall,
            out_dir=str(req.artifact_dir),
        )
