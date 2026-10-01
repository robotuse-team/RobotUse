"""The single real-eval scoring core shared by every ablation mode.

One :func:`gap.agent.launcher.launch` call scores a whole
``(family, variation)`` suite over **all** its tasks — the same engine
``gap run``-style trial execution uses — so the benchmark adds *no*
parallel execution path of its own:

* ``workflow_dir_map=None``  → ``launch()`` zero-shot-codegens per task
  (mode ``llm_generation``);
* ``workflow_dir_map={tid: dir}``  → per-task pre-built/templated
  workflows (modes ``llm_plus_policy`` / ``policy_only``).

``launch()`` builds ``task_ids × n_seeds`` WorkItems and runs them
across ``trials.num_workers`` natively (``GAP_MUJOCO_EGL_DEVICES``
round-robins workers over GPUs), and ``_aggregate_tasks`` returns one
``TaskResult`` per task. No subprocess sharding, no hand-built CLI arg
list, no flag-drop risk.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import TYPE_CHECKING

from gap.agent.config import PipelineConfig, SuiteSpec

if TYPE_CHECKING:
    from gap.agent.launcher import TaskResult


async def score_suite(
    *,
    suite_name: str,
    task_ids: list[int],
    n_seeds: int,
    pipeline_config: PipelineConfig,
    artifact_dir: Path,
    num_workers: int,
    workflow_dir_map: dict[int, str] | None = None,
    record_video: bool = False,
) -> tuple[list[TaskResult], float]:
    """Score one suite over ``task_ids × n_seeds`` via a single ``launch()``.

    Args:
        suite_name: gap suite name for this ``(family, variation)``.
        task_ids: Tasks to run (one ``SuiteSpec``, native parallelism).
        n_seeds: Trials/task; seed ``i`` → LIBERO init ``(i-1) % len``
            via ``LiberoHandle.reset`` — identical for every mode.
        pipeline_config: Shared parsed config. Deep-copied; the original
            is never mutated.
        artifact_dir: Per-cell output dir. ``launch()`` cleans it at
            start, so it must be unique per cell.
        num_workers: Native parallel sim workers (``trials.num_workers``).
        workflow_dir_map: ``{task_id: workflow_dir}`` for template modes;
            ``None`` → ``launch()`` codegens per task.
        record_video: Forwarded to ``trials.record_video``.

    Returns:
        ``(task_results, wall_clock_s)`` — one ``TaskResult`` per task.
    """
    from gap.agent.launcher import launch  # heavy; import lazily

    config = copy.deepcopy(pipeline_config)
    config.task = "auto"  # auto-resolve prompts from LIBERO metadata
    config.suites = [SuiteSpec(suite_name=suite_name, task_ids=list(task_ids))]
    config.trials.trials_per_generation = n_seeds
    config.trials.task_ids = list(task_ids)
    config.trials.code_generations = 1
    config.trials.regenerate_code_per_trial = False
    config.trials.num_workers = num_workers
    config.trials.output_dir = Path(artifact_dir).resolve()
    config.trials.record_video = record_video

    t0 = time.perf_counter()
    result = await launch(config, workflow_dir_map=workflow_dir_map)
    wall = time.perf_counter() - t0
    return list(result.task_results), wall
