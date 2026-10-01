"""Formal benchmark harness: family × variation × ablation grids.

One ``gap benchmark`` command sweeps a grid of benchmark *families*
(``libero`` / ``libero_pro`` / ``posvar`` / ``grocery_packing``) ×
per-family *variation* suites × pipeline *ablation* modes
(``llm_generation`` / ``llm_plus_policy`` / ``policy_only``). Every mode
produces a ``workflow.json`` in one shot (zero-shot — no rehearsal /
refine loop) and is then scored by the **same** in-process sim eval, so
the only difference between modes is how the workflow is produced.

Configs without a ``benchmark:`` block but with explicit ``suites:``
run in *suites mode* (the grocery acceptance shape): each suite is one
cell. See :mod:`gap.benchmark.config` for the grid definition and
:mod:`gap.benchmark.harness` for gate/resume semantics.

Public facade::

    import gap.benchmark
    summary = gap.benchmark.run("examples/benchmark/smoke.yaml")
    summary = gap.benchmark.run(cfg, gate=True, resume=True)
    assert summary.ok
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import (
    DEFAULT_GATE_THRESHOLD,
    FAMILY_SUITES,
    KNOWN_FAMILIES,
    KNOWN_MODES,
    POSVAR_SUITES,
    BenchmarkConfig,
    BenchmarkModeOverride,
)

if TYPE_CHECKING:
    from .harness import BenchmarkSummary

__all__ = [
    "DEFAULT_GATE_THRESHOLD",
    "FAMILY_SUITES",
    "KNOWN_FAMILIES",
    "KNOWN_MODES",
    "POSVAR_SUITES",
    "BenchmarkConfig",
    "BenchmarkModeOverride",
    "run",
    "run_benchmark",
]


def run(
    config: BenchmarkConfig | str | Path,
    *,
    gate: bool = False,
    resume: bool = False,
) -> BenchmarkSummary:
    """Run a benchmark from a config object or YAML path (synchronous).

    Args:
        config: A :class:`BenchmarkConfig` or a path to a benchmark YAML.
        gate: Apply the acceptance gate — the returned summary's ``ok``
            is False when the overall success rate is below
            ``config.gate_threshold`` (default 0.90), when any cell
            errored, or when no trial ran.
        resume: Reuse the latest run dir under ``config.output_dir``,
            skipping cells whose results already exist; the merged
            summary is rebuilt over old + new cells.

    Returns:
        :class:`gap.benchmark.harness.BenchmarkSummary`.
    """
    import asyncio

    from .harness import run_benchmark as _run

    cfg = (
        config
        if isinstance(config, BenchmarkConfig)
        else BenchmarkConfig.from_yaml(config)
    )
    return asyncio.run(_run(cfg, gate=gate, resume=resume))


def run_benchmark(*args: Any, **kwargs: Any):
    """Lazy proxy to :func:`gap.benchmark.harness.run_benchmark` (async).

    Imported lazily so ``import gap.benchmark`` (and the CLI
    registration) stays free of the heavy launcher dependency chain.
    """
    from .harness import run_benchmark as _run

    return _run(*args, **kwargs)
