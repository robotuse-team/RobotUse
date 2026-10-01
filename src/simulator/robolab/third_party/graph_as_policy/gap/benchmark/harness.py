"""Grid driver: sweep ``modes × (family, variation) × policies`` → one summary.

Thin wrapper over the one real engine. For each ``mode`` × each
``(family, variation, suite)`` cell, one native
:func:`gap.agent.launcher.launch` runs the whole suite
(``task_ids × n_seeds``) with native ``trials.num_workers`` parallelism.
When ``benchmark.policies`` is set, policy-dependent modes run once per
listed policy id so a single benchmark A/Bs the registered VLAs
end-to-end. Modes run sequentially (they contend on sim GPUs / the
shared policy server); tasks×seeds run in parallel *inside*
``launch()``. GPU spread is the standard
``GAP_MUJOCO_EGL_DEVICES=<csv>`` env (round-robins workers across GPUs).

Suites-mode configs (no ``benchmark:`` block — e.g. the grocery
acceptance recipe) treat each entry of the pipeline's ``suites:`` list
as one cell; all remaining cells run concurrently inside one event loop
(each is its own single-suite ``launch()``).

New relative to the source:

* ``gate=True`` — the returned :class:`BenchmarkSummary` carries
  ``ok=False`` (and ``gap benchmark --gate`` exits nonzero) when the
  trial-pooled overall success rate is below ``cfg.gate_threshold``
  (default 0.90), when any cell errored, or when no trial ran.
* ``resume=True`` — reuse the latest timestamped run dir under
  ``cfg.output_dir`` and skip cells whose ``cell_result.json`` already
  exists; the merged summary is rebuilt over old + new cells.

Each external (``url``-only) policy referenced by the run is
preflight-checked at startup so a missing policy server fails fast with
a clear diagnostic instead of crashing mid-suite.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .builtin_modes import build_mode
from .config import BenchmarkConfig
from .modes import ModeRequest
from .report import ModeResult, build_summary, error_cell, normalize_task_results

logger = logging.getLogger("gap.benchmark")

_STAMP_RX = re.compile(r"^\d{8}_\d{6}$")
_CELL_RESULT = "cell_result.json"


@dataclass
class BenchmarkSummary:
    """Outcome of one :func:`run_benchmark` call."""

    ok: bool
    """True unless gating was requested and failed (threshold missed,
    errored cell, or zero trials)."""
    gated: bool
    gate_threshold: float
    success_rate: float
    """Trial-pooled success rate over every scored cell."""
    completion_rate: float
    """Trial-weighted completion rate over every scored cell."""
    n_trials: int = 0
    n_success: int = 0
    avg_physical_execution_s: float = 0.0
    """Trial-weighted mean of the physical-execution estimate over every
    scored cell. Zero when no cell reported latency (real-robot suites,
    non-LIBERO envs)."""
    run_dir: Path | None = None
    cells: list[ModeResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    """The merged summary dict (also written to ``<run>/summary.json``)."""


# ---------------------------------------------------------------------------
# Policy preflight
# ---------------------------------------------------------------------------


async def preflight_policies(
    policies_cfg: dict[str, dict[str, Any]],
    policy_ids: list[str],
    *,
    timeout_s: float = 5.0,
) -> dict[str, str]:
    """Probe each external policy's WebSocket URL with a short handshake.

    Args:
        policies_cfg: The ``policies:`` registry
            (``{policy_id: {"url"|"start_cmd"|"preset", ...}}``).
        policy_ids: Ids to probe. Managed/preset policies (no ``url``)
            are skipped — the PolicyManager spawns them inline.
        timeout_s: Per-policy probe timeout.

    Returns:
        ``{policy_id: "ok"|"<error message>"}``. Caller decides whether
        to abort the run.
    """
    try:
        import websockets
    except ImportError:  # pragma: no cover - optional dep
        logger.warning(
            "preflight: 'websockets' not importable; skipping health check"
        )
        return {pid: "skipped (websockets not installed)" for pid in policy_ids}

    async def _probe(pid: str) -> tuple[str, str]:
        entry = policies_cfg.get(pid) or {}
        url = entry.get("url")
        if not url:
            return pid, "skipped (managed policy — PolicyManager owns lifecycle)"
        try:
            async with asyncio.timeout(timeout_s):
                async with websockets.connect(url, open_timeout=timeout_s) as ws:
                    # openpi's WebsocketPolicyServer sends a metadata
                    # packet first; receiving it confirms the handshake.
                    await ws.recv()
            return pid, "ok"
        except Exception as exc:  # noqa: BLE001 — surface to user
            return pid, f"{type(exc).__name__}: {exc}"

    results = await asyncio.gather(*[_probe(pid) for pid in policy_ids])
    return dict(results)


async def _preflight_or_raise(cfg: BenchmarkConfig) -> None:
    """Probe every external policy this run references; raise on failure."""
    assert cfg.pipeline_config is not None
    if cfg.suites_mode:
        return  # workers boot exactly what each workflow references
    policies_to_probe = list(
        dict.fromkeys(
            cfg.policies
            or [
                eff.get("policy_id")
                for eff in (cfg.effective(m) for m in cfg.modes)
                if eff.get("policy_id")
            ]
        )
    )
    if not policies_to_probe:
        return
    logger.info(
        "preflight: probing %d polic%s — %s",
        len(policies_to_probe),
        "y" if len(policies_to_probe) == 1 else "ies",
        ", ".join(policies_to_probe),
    )
    probe = await preflight_policies(
        cfg.pipeline_config.policies, policies_to_probe,
    )
    for pid, status in probe.items():
        level = logger.info if status.startswith(("ok", "skipped")) else logger.error
        level("preflight [%s] -> %s", pid, status)
    unreachable = [
        pid for pid, status in probe.items()
        if not status.startswith(("ok", "skipped"))
    ]
    if unreachable:
        raise RuntimeError(
            "policy preflight failed: " + ", ".join(unreachable)
            + ". Start the policy server(s) before re-running "
            "(gap policy serve <preset>)."
        )


# ---------------------------------------------------------------------------
# Run-dir + cell persistence (resume)
# ---------------------------------------------------------------------------


def _resolve_run_dir(output_dir: Path, resume: bool) -> Path:
    """New timestamped run dir, or the latest existing one when resuming."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        existing = sorted(
            d for d in output_dir.iterdir()
            if d.is_dir() and _STAMP_RX.match(d.name)
        )
        if existing:
            logger.info("resume: reusing run dir %s", existing[-1])
            return existing[-1]
    run_dir = output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_cell(cell_dir: Path, cell: ModeResult) -> None:
    """Persist a scored cell so ``--resume`` can skip it.

    Error cells are deliberately not persisted — a resumed run re-runs
    them.
    """
    if cell.error:
        return
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / _CELL_RESULT).write_text(json.dumps(asdict(cell), indent=2))


def _load_cell(cell_dir: Path) -> ModeResult | None:
    """Load a previously persisted cell result, or None."""
    path = Path(cell_dir) / _CELL_RESULT
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
        return ModeResult(**data)
    except Exception:
        logger.warning("resume: could not load %s — re-running cell", path)
        return None


# ---------------------------------------------------------------------------
# Grid mode
# ---------------------------------------------------------------------------


async def _run_grid(
    cfg: BenchmarkConfig, run_dir: Path, resume: bool,
) -> tuple[list[ModeResult], dict[str, Any]]:
    assert cfg.pipeline_config is not None
    grid_cells = cfg.grid_cells()
    task_ids = cfg.resolved_task_ids()
    logger.info(
        "benchmark grid: modes=%s cells=%s tasks=%s n_seeds=%d policies=%s -> %s",
        cfg.modes,
        [f"{f}/{v}" for f, v, _ in grid_cells],
        task_ids, cfg.n_seeds, cfg.policies or "(none)", run_dir,
    )

    t_start = time.perf_counter()
    cells: list[ModeResult] = []
    for mode_name in cfg.modes:  # sequential across modes (contention)
        mode = build_mode(mode_name)
        eff_base = cfg.effective(mode_name)
        # Policy axis: only iterate for policy-dependent modes; collapse
        # to a single pass for ``llm_generation``.
        if mode.requires_policy and cfg.policies:
            policy_axis: list[str | None] = list(cfg.policies)
        elif mode.requires_policy:
            policy_axis = [eff_base.get("policy_id")]
        else:
            policy_axis = [None]
        for policy_id in policy_axis:
            eff = dict(eff_base)
            if policy_id is not None:
                eff["policy_id"] = policy_id
            for family, variation, suite in grid_cells:
                # Disambiguate artifact dirs by policy so two VLAs don't
                # overwrite each other in the same run.
                cell_dir = run_dir / mode_name
                if mode.requires_policy and policy_id:
                    cell_dir = cell_dir / policy_id
                cell_dir = cell_dir / family / variation
                tag = f"{mode_name}{f'@{policy_id}' if policy_id else ''}"

                if resume:
                    prior = _load_cell(cell_dir)
                    if prior is not None:
                        logger.info(
                            "[%s %s/%s] resume: cell already scored "
                            "(success_rate=%.3f) — skipping",
                            tag, family, variation, prior.success_rate,
                        )
                        cells.append(prior)
                        continue

                req = ModeRequest(
                    family=family,
                    variation=variation,
                    suite_name=suite,
                    task_ids=task_ids,
                    n_seeds=eff["n_seeds"],
                    num_workers=eff["num_workers"],
                    pipeline_config=cfg.pipeline_config,
                    artifact_dir=cell_dir,
                    record_video=cfg.record_video,
                    extra=eff,
                )
                logger.info("[%s %s/%s] running suite %s ...",
                            tag, family, variation, suite)
                try:
                    cell = await mode.run(req)
                except Exception as e:  # noqa: BLE001 — never abort the grid
                    logger.exception(
                        "[%s %s/%s] crashed", tag, family, variation
                    )
                    cell = error_cell(
                        mode=mode_name, family=family, variation=variation,
                        suite_name=suite,
                        policy_id=policy_id or "",
                        error=f"{type(e).__name__}: {e}",
                        out_dir=str(cell_dir),
                    )
                _save_cell(cell_dir, cell)
                cells.append(cell)
                logger.info(
                    "[%s %s/%s] -> success_rate=%.3f (%d/%d) error=%s",
                    tag, family, variation, cell.success_rate,
                    cell.n_success, cell.n_trials, cell.error,
                )
        mdir = run_dir / mode_name
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / "aaa_done_flag.txt").write_text(f"mode {mode_name} done\n")

    grid = {
        "modes": cfg.modes,
        "families": cfg.families,
        "policies": cfg.policies,
        "cells": [f"{f}/{v}" for f, v, _ in grid_cells],
        "task_ids": task_ids,
        "n_seeds": cfg.n_seeds,
        "num_workers": cfg.num_workers,
        "gate_threshold": cfg.gate_threshold,
        "wall_clock_s": time.perf_counter() - t_start,
        "source_yaml": str(cfg.source_yaml),
        "resume": resume,
    }
    return cells, grid


# ---------------------------------------------------------------------------
# Suites mode
# ---------------------------------------------------------------------------


def _suite_cell_names(suites: list[Any]) -> list[str]:
    """Stable per-suite cell names; duplicates get _NN suffixes."""
    seen: dict[str, int] = {}
    names = []
    for s in suites:
        n = seen.get(s.suite_name, 0) + 1
        seen[s.suite_name] = n
        names.append(s.suite_name if n == 1 else f"{s.suite_name}_{n:02d}")
    return names


async def _run_suites(
    cfg: BenchmarkConfig, run_dir: Path, resume: bool,
) -> tuple[list[ModeResult], dict[str, Any]]:
    """Suites mode: each entry of the pipeline ``suites:`` list is a cell.

    All remaining cells launch concurrently (one single-suite
    ``launch()`` each), preserving the source recipe's concurrent-suite
    behavior — per-suite ``num_workers`` bounds each cell's pool.
    """
    from gap.agent.launcher import launch

    assert cfg.pipeline_config is not None
    pc = cfg.pipeline_config
    cell_names = _suite_cell_names(pc.suites)
    t_start = time.perf_counter()

    cells_by_idx: dict[int, ModeResult] = {}
    pending: list[tuple[int, str]] = []
    for idx, name in enumerate(cell_names):
        cell_dir = run_dir / "suites" / name
        if resume:
            prior = _load_cell(cell_dir)
            if prior is not None:
                logger.info(
                    "[suite %s] resume: cell already scored "
                    "(success_rate=%.3f) — skipping",
                    name, prior.success_rate,
                )
                cells_by_idx[idx] = prior
                continue
        pending.append((idx, name))

    # Per-cell device-slot offsets: concurrent single-suite launch()
    # calls must not all start their workers at device slot 0 (every
    # cell's model stack would land on the same GPU).
    cell_offsets: list[int] = []
    _off = 0
    for s_ in pc.suites:
        cell_offsets.append(_off)
        _off += max(1, int(getattr(s_, "num_workers", 1) or 1))

    async def _run_one(idx: int, name: str) -> tuple[int, ModeResult]:
        suite = pc.suites[idx]
        cell_dir = run_dir / "suites" / name
        sub = copy.deepcopy(pc)
        sub.suites = [copy.deepcopy(suite)]
        sub.trials.output_dir = cell_dir
        t0 = time.perf_counter()
        try:
            result = await launch(sub, device_slot_offset=cell_offsets[idx])
        except Exception as e:  # noqa: BLE001 — never abort the run
            logger.exception("[suite %s] crashed", name)
            return idx, error_cell(
                mode="llm_generation",
                family="suites",
                variation=name,
                suite_name=suite.suite_name,
                error=f"{type(e).__name__}: {e}",
                out_dir=str(cell_dir),
            )
        wall = time.perf_counter() - t0
        cell = normalize_task_results(
            mode="llm_generation",
            family="suites",
            variation=name,
            suite_name=suite.suite_name,
            task_results=list(result.task_results),
            eval_wall_s=wall,
            out_dir=str(cell_dir),
        )
        _save_cell(cell_dir, cell)
        return idx, cell

    if pending:
        logger.info(
            "suites mode: launching %d of %d suite cell(s) concurrently",
            len(pending), len(cell_names),
        )
        results = await asyncio.gather(
            *[_run_one(idx, name) for idx, name in pending]
        )
        for idx, cell in results:
            cells_by_idx[idx] = cell

    cells = [cells_by_idx[i] for i in sorted(cells_by_idx)]
    grid = {
        "modes": ["llm_generation"],
        "suites": cell_names,
        "trials_per_generation": pc.trials.trials_per_generation,
        "num_workers": pc.trials.num_workers,
        "gate_threshold": cfg.gate_threshold,
        "wall_clock_s": time.perf_counter() - t_start,
        "source_yaml": str(cfg.source_yaml),
        "resume": resume,
    }
    return cells, grid


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def run_benchmark(
    cfg: BenchmarkConfig,
    *,
    gate: bool = False,
    resume: bool = False,
) -> BenchmarkSummary:
    """Run the benchmark and write ``summary.json`` / ``summary.tsv``.

    Returns a :class:`BenchmarkSummary`; ``summary.summary`` is the
    merged dict also written to ``<run>/summary.json``.
    """
    if cfg.pipeline_config is None:
        raise ValueError(
            "BenchmarkConfig.pipeline_config is None — construct via "
            "BenchmarkConfig.from_yaml() so the shared pipeline config "
            "is parsed."
        )

    run_dir = _resolve_run_dir(Path(cfg.output_dir), resume)

    await _preflight_or_raise(cfg)

    if cfg.suites_mode:
        cells, grid = await _run_suites(cfg, run_dir, resume)
    else:
        cells, grid = await _run_grid(cfg, run_dir, resume)

    summary = build_summary(cells, out_dir=run_dir, grid=grid)
    if cfg.record_video:
        _collect_videos(run_dir)
    (run_dir / "aaa_done_flag.txt").write_text(
        f"benchmark done: {len(cells)} cells in "
        f"{grid['wall_clock_s']:.1f}s\n"
    )
    logger.info("benchmark complete -> %s/summary.tsv", run_dir)

    # --- Overall (trial-pooled) metrics + gate verdict ---
    n_trials = sum(c.n_trials for c in cells)
    n_success = sum(c.n_success for c in cells)
    success_rate = n_success / n_trials if n_trials else 0.0
    completion_rate = (
        sum(c.completion_rate * c.n_trials for c in cells) / n_trials
        if n_trials else 0.0
    )
    avg_physical_execution_s = (
        sum(c.avg_physical_execution_s * c.n_trials for c in cells) / n_trials
        if n_trials else 0.0
    )
    errored = [c for c in cells if c.error]
    ok = True
    if gate:
        ok = (
            n_trials > 0
            and not errored
            and success_rate >= cfg.gate_threshold
        )
        verdict = "PASS" if ok else "FAIL"
        logger.info(
            "gate %s: success_rate=%.4f threshold=%.4f trials=%d errors=%d",
            verdict, success_rate, cfg.gate_threshold, n_trials, len(errored),
        )

    return BenchmarkSummary(
        ok=ok,
        gated=gate,
        gate_threshold=cfg.gate_threshold,
        success_rate=success_rate,
        completion_rate=completion_rate,
        n_trials=n_trials,
        n_success=n_success,
        avg_physical_execution_s=avg_physical_execution_s,
        run_dir=run_dir,
        cells=cells,
        summary=summary,
    )


def _collect_videos(run_dir: Path) -> None:
    """Collate per-trial ``video.mp4`` into ``<run>/videos/``.

    Videos already live, well-defined, under each cell's
    ``.../task_NN/trial_*_{pass,fail}/video.mp4``. This hard-links them
    (no extra disk; copy fallback across filesystems) into a flat
    ``<run>/videos/`` with descriptive names, co-located with
    ``summary.tsv`` — so the run's own output dir is the single home for
    its results.
    """
    import shutil

    vid_dir = run_dir / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in sorted(run_dir.rglob("video.mp4")):
        if vid_dir in src.parents:
            continue
        rel = src.relative_to(run_dir)
        trial = rel.parts[-2]
        m = re.search(r"trial_(\d+).*_(pass|fail)\b", trial)
        if m:
            trial_tag = f"trial_{m.group(1)}__{m.group(2)}"
        else:
            trial_tag = trial
        dst = vid_dir / ("__".join(rel.parts[:-2]) + f"__{trial_tag}.mp4")
        try:
            if dst.exists():
                dst.unlink()
            try:
                os.link(src, dst)          # hard-link: no extra disk
            except OSError:
                shutil.copy2(src, dst)     # cross-fs fallback
            n += 1
        except OSError as e:  # noqa: PERF203
            logger.warning("could not collate video %s: %s", src, e)
    if n:
        logger.info("collated %d video(s) -> %s", n, vid_dir)
