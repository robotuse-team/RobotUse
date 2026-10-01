"""Normalize per-cell results into a comparable benchmark summary.

A *cell* is one ``(mode, family, variation)`` (grid mode) or one suite
(suites mode). Within a cell the harness runs ``n_tasks × n_seeds``
trials and the launcher's ``_aggregate_tasks`` produces one
:class:`gap.agent.launcher.TaskResult` per task. We read the metrics
those dataclasses already computed (``success_rate`` /
``completion_rate`` / ``avg_reward``) rather than re-deriving them, then
emit a single ``summary.json`` + ``summary.tsv`` (with mode×variation
pivots) so ablations can be eyeballed side by side.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gap.agent.launcher import TaskResult


@dataclass
class ModeResult:
    """One grid cell, mode-agnostic."""

    mode: str
    variation: str
    suite_name: str
    family: str = ""
    # Empty string when the mode is policy-agnostic (e.g.
    # ``llm_generation``) — the harness still groups by it so cells from
    # different policies don't get merged.
    policy_id: str = ""
    n_trials: int = 0
    n_success: int = 0
    success_rate: float = 0.0
    # Partial-credit metric: mean fraction of sub-goals completed per
    # trial. For multi-item suites (grocery_packing) this is non-zero
    # even when no trial fully succeeds; for single-goal suites it
    # equals ``success_rate``.
    completion_rate: float = 0.0
    avg_reward: float = 0.0
    wall_clock_s: float = 0.0
    produce_wall_s: float = 0.0
    eval_wall_s: float = 0.0
    avg_physical_execution_s: float = 0.0
    """Trial-weighted mean of the physical-execution estimate (s).
    ``control_steps / control_freq + non-physics wall``; mirrors what
    real hardware would take. Zero when the env didn't report latency."""
    avg_control_steps: float = 0.0
    """Trial-weighted mean of cumulative env steps per trial."""
    avg_sim_physics_wall_s: float = 0.0
    """Trial-weighted mean of wall time spent inside ``env.step``."""
    out_dir: str = ""
    error: str | None = None
    # [{task_id, success_rate, completion_rate, avg_reward, n_trials, n_success}]
    per_task: list[dict[str, Any]] = field(default_factory=list)


def normalize_task_results(
    *,
    mode: str,
    variation: str,
    suite_name: str,
    family: str = "",
    policy_id: str = "",
    task_results: list[TaskResult],
    produce_wall_s: float = 0.0,
    eval_wall_s: float = 0.0,
    out_dir: str = "",
    error: str | None = None,
) -> ModeResult:
    """Fold the launcher's per-task ``TaskResult``s into a ``ModeResult``.

    Metrics are read straight off ``TaskResult`` (computed by
    ``_aggregate_tasks``) — never recomputed from raw trials. Cell-level
    rates are trial-weighted so a cell's number equals its pooled trial
    rate.
    """
    n_trials = sum(t.total_trials for t in task_results)
    n_success = sum(t.success_count for t in task_results)
    if n_trials > 0:
        success_rate = n_success / n_trials
        completion_rate = (
            sum(t.completion_rate * t.total_trials for t in task_results)
            / n_trials
        )
        avg_reward = (
            sum(t.avg_reward * t.total_trials for t in task_results) / n_trials
        )
        avg_physical_execution_s = (
            sum(
                t.avg_physical_execution_s * t.total_trials
                for t in task_results
            )
            / n_trials
        )
        avg_control_steps = (
            sum(t.avg_control_steps * t.total_trials for t in task_results)
            / n_trials
        )
        avg_sim_physics_wall_s = (
            sum(
                t.avg_sim_physics_wall_s * t.total_trials for t in task_results
            )
            / n_trials
        )
    else:
        success_rate = 0.0
        completion_rate = 0.0
        avg_reward = 0.0
        avg_physical_execution_s = 0.0
        avg_control_steps = 0.0
        avg_sim_physics_wall_s = 0.0
    return ModeResult(
        mode=mode,
        family=family,
        variation=variation,
        suite_name=suite_name,
        policy_id=policy_id,
        n_trials=n_trials,
        n_success=n_success,
        success_rate=success_rate,
        completion_rate=completion_rate,
        avg_reward=avg_reward,
        wall_clock_s=produce_wall_s + eval_wall_s,
        produce_wall_s=produce_wall_s,
        eval_wall_s=eval_wall_s,
        avg_physical_execution_s=avg_physical_execution_s,
        avg_control_steps=avg_control_steps,
        avg_sim_physics_wall_s=avg_sim_physics_wall_s,
        out_dir=out_dir,
        error=error,
        per_task=[
            {
                "task_id": t.task_id,
                "success_rate": t.success_rate,
                "completion_rate": t.completion_rate,
                "avg_reward": t.avg_reward,
                "avg_physical_execution_s": t.avg_physical_execution_s,
                "avg_control_steps": t.avg_control_steps,
                "avg_sim_physics_wall_s": t.avg_sim_physics_wall_s,
                "n_trials": t.total_trials,
                "n_success": t.success_count,
            }
            for t in task_results
        ],
    )


def error_cell(
    *,
    mode: str,
    variation: str,
    suite_name: str,
    error: str,
    family: str = "",
    policy_id: str = "",
    out_dir: str = "",
) -> ModeResult:
    """A cell that failed to produce/score — keeps the grid completing."""
    return ModeResult(
        mode=mode,
        family=family,
        variation=variation,
        suite_name=suite_name,
        policy_id=policy_id,
        out_dir=out_dir,
        error=error,
    )


def merge_cells(task_cells: list[ModeResult]) -> ModeResult:
    """Pool per-task ``ModeResult``s into one ``(mode, variation)`` cell.

    Trials/successes are pooled (trial-weighted reward); ``per_task`` is
    concatenated; walls summed. ``error`` is set only when **every** task
    errored — a partial failure still reports the tasks that scored.
    """
    if not task_cells:
        raise ValueError("merge_cells: empty task_cells")
    head = task_cells[0]
    errored = [c for c in task_cells if c.error]
    scored = [c for c in task_cells if not c.error]
    if not scored:
        return ModeResult(
            mode=head.mode,
            family=head.family,
            variation=head.variation,
            suite_name=head.suite_name,
            out_dir=head.out_dir,
            error=errored[0].error,
        )
    n_trials = sum(c.n_trials for c in scored)
    n_success = sum(c.n_success for c in scored)
    avg_reward = (
        sum(c.avg_reward * c.n_trials for c in scored) / n_trials
        if n_trials
        else 0.0
    )
    completion_rate = (
        sum(c.completion_rate * c.n_trials for c in scored) / n_trials
        if n_trials
        else 0.0
    )
    avg_physical_execution_s = (
        sum(c.avg_physical_execution_s * c.n_trials for c in scored) / n_trials
        if n_trials
        else 0.0
    )
    avg_control_steps = (
        sum(c.avg_control_steps * c.n_trials for c in scored) / n_trials
        if n_trials
        else 0.0
    )
    avg_sim_physics_wall_s = (
        sum(c.avg_sim_physics_wall_s * c.n_trials for c in scored) / n_trials
        if n_trials
        else 0.0
    )
    per_task: list[dict[str, Any]] = []
    for c in scored:
        per_task.extend(c.per_task)
    return ModeResult(
        mode=head.mode,
        family=head.family,
        variation=head.variation,
        suite_name=head.suite_name,
        n_trials=n_trials,
        n_success=n_success,
        success_rate=n_success / n_trials if n_trials else 0.0,
        completion_rate=completion_rate,
        avg_reward=avg_reward,
        wall_clock_s=sum(c.wall_clock_s for c in task_cells),
        produce_wall_s=sum(c.produce_wall_s for c in task_cells),
        eval_wall_s=sum(c.eval_wall_s for c in task_cells),
        avg_physical_execution_s=avg_physical_execution_s,
        avg_control_steps=avg_control_steps,
        avg_sim_physics_wall_s=avg_sim_physics_wall_s,
        out_dir=head.out_dir,
        # Partial task failures do NOT mark the whole cell errored — it
        # still has scored tasks and a meaningful success_rate.
        error=None,
        per_task=per_task,
    )


def build_summary(
    cells: list[ModeResult],
    *,
    out_dir: Path,
    grid: dict[str, Any],
) -> dict[str, Any]:
    """Write ``summary.json`` + ``summary.tsv`` and return the summary dict.

    ``summary.tsv`` has one row per cell plus trailing mode×variation
    pivots (success_rate and completion_rate) for at-a-glance comparison.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Row key disambiguates by policy_id when present so pi05 vs
    # molmoact cells under the same (mode, family, variation) don't
    # collapse into a single number in the pivot.
    def _row_key(c: ModeResult) -> str:
        return f"{c.mode}@{c.policy_id}" if c.policy_id else c.mode

    # matrix[row_key]["<family>/<variation>"] = {...}
    matrix: dict[str, dict[str, Any]] = {}
    for c in cells:
        col = f"{c.family}/{c.variation}" if c.family else c.variation
        matrix.setdefault(_row_key(c), {})[col] = {
            "success_rate": c.success_rate,
            "completion_rate": c.completion_rate,
            "n_trials": c.n_trials,
            "error": c.error,
            "policy_id": c.policy_id,
            "mode": c.mode,
        }

    summary = {
        "grid": grid,
        "cells": [asdict(c) for c in cells],
        "matrix": matrix,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    rows = list(dict.fromkeys(_row_key(c) for c in cells))
    cols = list(
        dict.fromkeys(
            (f"{c.family}/{c.variation}" if c.family else c.variation)
            for c in cells
        )
    )
    with (out_dir / "summary.tsv").open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(
            [
                "mode",
                "policy_id",
                "family",
                "variation",
                "suite_name",
                "success_rate",
                "completion_rate",
                "avg_reward",
                "n_success",
                "n_trials",
                "wall_clock_s",
                "avg_physical_execution_s",
                "avg_control_steps",
                "avg_sim_physics_wall_s",
                "error",
            ]
        )
        for c in sorted(
            cells,
            key=lambda x: (x.mode, x.policy_id, x.family, x.variation),
        ):
            w.writerow(
                [
                    c.mode,
                    c.policy_id,
                    c.family,
                    c.variation,
                    c.suite_name,
                    f"{c.success_rate:.4f}",
                    f"{c.completion_rate:.4f}",
                    f"{c.avg_reward:.4f}",
                    c.n_success,
                    c.n_trials,
                    f"{c.wall_clock_s:.1f}",
                    f"{c.avg_physical_execution_s:.2f}",
                    f"{c.avg_control_steps:.1f}",
                    f"{c.avg_sim_physics_wall_s:.2f}",
                    c.error or "",
                ]
            )
        # Pivot block: rows = mode@policy_id, cols = family/variation.
        # Two pivots — success_rate (all-or-nothing) and completion_rate
        # (partial credit).
        w.writerow([])
        w.writerow(["success_rate_pivot"] + cols)
        for r in rows:
            row = [r]
            for v in cols:
                cell = matrix.get(r, {}).get(v)
                if cell is None:
                    row.append("-")
                elif cell["error"]:
                    row.append("ERR")
                else:
                    row.append(f"{cell['success_rate']:.4f}")
            w.writerow(row)
        w.writerow([])
        w.writerow(["completion_rate_pivot"] + cols)
        for r in rows:
            row = [r]
            for v in cols:
                cell = matrix.get(r, {}).get(v)
                if cell is None:
                    row.append("-")
                elif cell["error"]:
                    row.append("ERR")
                else:
                    row.append(f"{cell['completion_rate']:.4f}")
            w.writerow(row)

    return summary
