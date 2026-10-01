"""Benchmark grid configuration.

A benchmark YAML is a standalone pipeline config (``llm:`` / ``skills:``
/ ``policies:`` / ``trials:`` …, parsed once via
:meth:`gap.agent.PipelineConfig.from_yaml`) plus an optional
``benchmark:`` block defining the sweep grid:

* **grid mode** (``benchmark:`` present) — sweep ``modes ×
  (family, variation)`` cells, each scored by one native
  :func:`gap.agent.launcher.launch` over ``task_ids × n_seeds``.
* **suites mode** (no ``benchmark:`` block) — the YAML's own ``suites:``
  list is the cell axis; each suite runs through ``launch()`` with the
  pipeline's trial settings. This is the shape of the grocery
  acceptance recipe (hand-curated per-task prompts + objects blocks).

``gate_threshold`` (default 0.90) is the acceptance bar applied by
``run_benchmark(..., gate=True)`` / ``gap benchmark --gate``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from gap.agent.config import PipelineConfig

# Benchmark families. Each family owns its own ``variation -> suite``
# map; the grid sweeps the union of ``(family, variation)`` over the
# selected ``families``. Suite names are the gap registry names (see
# gap.envs.registry — the registry also accepts the dev-config aliases
# libero_grocery_packing_object / libero_grocery_packing_permutation).
#
# * ``posvar``          — the 4 variance suites (10 tasks × 50 baked
#   init-state rows; ``all`` is the union of the perturbation kinds).
# * ``libero``          — the stock ``libero_object`` task suite.
# * ``libero_pro``      — the OOD ``object_swap`` perturbation suite
#   ("object" is deliberately omitted: it would duplicate libero/object).
# * ``grocery_packing`` — same floor scene as libero/object; success is
#   teleport-on-In monotonic delivery. ``object`` is the single-In goal,
#   ``permutation`` the full-table conjunction.
FAMILY_SUITES: dict[str, dict[str, str]] = {
    "libero": {
        "object": "libero_object",
    },
    "libero_pro": {
        "object_swap": "libero_object_swap",
    },
    "posvar": {
        "pos_var": "libero_object_target_pos_var20x20",
        "permutation": "libero_object_target_permutation_variance",
        "basket_swap": "libero_object_target_basket_swap_variance",
        "all": "libero_object_all_variance",
    },
    "grocery_packing": {
        "object": "libero_object_packing",
        "permutation": "permutation_packing",
    },
}

KNOWN_FAMILIES: tuple[str, ...] = tuple(FAMILY_SUITES)

# Back-compat alias: the posvar-only map.
POSVAR_SUITES: dict[str, str] = FAMILY_SUITES["posvar"]

# The ablation modes the harness knows how to build. ``policy_only`` is
# the bare-VLA baseline for ``llm_plus_policy``. (The source's
# ``monolithic`` scaffold is deleted — it never scored a cell.)
KNOWN_MODES: tuple[str, ...] = (
    "llm_generation",
    "llm_plus_policy",
    "policy_only",
)

#: Default acceptance bar for ``--gate`` runs.
DEFAULT_GATE_THRESHOLD = 0.90


@dataclass
class BenchmarkModeOverride:
    """Per-mode knobs that override the top-level grid sizing.

    ``None`` means "inherit the top-level value". ``workflow_dir`` is
    only meaningful for template-driven modes (``llm_plus_policy`` /
    ``policy_only``); it is absolutized against the YAML's directory in
    :meth:`BenchmarkConfig.from_yaml`.
    """

    n_seeds: int | None = None
    num_workers: int | None = None
    time_budget_s: float | None = None
    workflow_dir: str | None = None
    # Which policy SKILL the template-driven modes steer (== its preset /
    # bundle name, e.g. ``pi05-libero``), substituted into the template's
    # ``{{policy_id}}`` placeholder. ``None`` inherits the template default
    # (``pi05-libero``).
    policy_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkConfig:
    """Full benchmark spec (grid or suites mode)."""

    source_yaml: Path
    # Families to sweep (keys of FAMILY_SUITES). Defaults to ["posvar"]
    # so legacy posvar YAMLs behave exactly as before.
    families: list[str] = field(default_factory=lambda: ["posvar"])
    # Optional in-family variation-name filter. None -> every variation
    # of each selected family.
    variations: list[str] | None = None
    modes: list[str] = field(default_factory=lambda: list(KNOWN_MODES))
    n_tasks: int = 10
    # Explicit task-id subset. None -> range(n_tasks).
    task_ids: list[int] | None = None
    n_seeds: int = 50
    # Native parallelism: passed straight to launch() as
    # ``trials.num_workers``. GPU spread is GAP_MUJOCO_EGL_DEVICES=<csv>.
    num_workers: int = 8
    # Optional policy axis: when non-empty, every policy-dependent mode
    # runs once per entry so a single benchmark A/Bs the registered VLAs.
    policies: list[str] = field(default_factory=list)
    record_video: bool = False
    output_dir: Path = field(default_factory=lambda: Path("./benchmark_runs"))
    smoke: bool = False
    gate_threshold: float = DEFAULT_GATE_THRESHOLD
    mode_overrides: dict[str, BenchmarkModeOverride] = field(default_factory=dict)
    # Suites mode: no ``benchmark:`` block — the pipeline config's own
    # ``suites:`` list is the cell axis (grocery acceptance shape).
    suites_mode: bool = False
    # Parsed once from ``source_yaml``; the shared LLM/skills/policies
    # source for all modes. Not a YAML field.
    pipeline_config: PipelineConfig | None = None

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.suites_mode:
            return  # the family/mode grid is not used
        unknown_f = [f for f in self.families if f not in FAMILY_SUITES]
        if unknown_f:
            raise ValueError(
                f"unknown family(ies) {unknown_f}; "
                f"valid: {list(KNOWN_FAMILIES)}"
            )
        if self.variations is not None:
            valid = {v for f in self.families for v in FAMILY_SUITES[f]}
            unknown_v = [v for v in self.variations if v not in valid]
            if unknown_v:
                raise ValueError(
                    f"unknown variation(s) {unknown_v} for families "
                    f"{self.families}; valid: {sorted(valid)}"
                )
        unknown_m = [m for m in self.modes if m not in KNOWN_MODES]
        if unknown_m:
            raise ValueError(
                f"unknown mode(s) {unknown_m}; valid: {list(KNOWN_MODES)}"
            )

    # ------------------------------------------------------------------
    def suite_name(self, family: str, variation: str) -> str:
        """Map a ``(family, variation)`` pair to its gap suite name."""
        return FAMILY_SUITES[family][variation]

    def grid_cells(self) -> list[tuple[str, str, str]]:
        """The ``(family, variation, suite_name)`` cells to sweep.

        Union over ``families``; within each family, all variations
        unless ``variations`` filters them by name. Empty in suites mode.
        """
        if self.suites_mode:
            return []
        cells: list[tuple[str, str, str]] = []
        for fam in self.families:
            fam_map = FAMILY_SUITES[fam]
            names = (
                [v for v in fam_map if v in self.variations]
                if self.variations is not None
                else list(fam_map)
            )
            cells.extend((fam, v, fam_map[v]) for v in names)
        return cells

    def resolved_task_ids(self) -> list[int]:
        """The task ids to sweep: explicit subset or ``range(n_tasks)``."""
        return (
            list(self.task_ids)
            if self.task_ids is not None
            else list(range(self.n_tasks))
        )

    def effective(self, mode: str) -> dict[str, Any]:
        """Resolve per-mode grid knobs (override beats top-level)."""
        ov = self.mode_overrides.get(mode, BenchmarkModeOverride())
        return {
            "n_seeds": ov.n_seeds if ov.n_seeds is not None else self.n_seeds,
            "num_workers": (
                ov.num_workers
                if ov.num_workers is not None
                else self.num_workers
            ),
            "time_budget_s": ov.time_budget_s,
            "workflow_dir": ov.workflow_dir,
            "policy_id": ov.policy_id,
            "extra": dict(ov.extra),
        }

    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> BenchmarkConfig:
        """Load a benchmark YAML (standalone — no ``base:`` inheritance).

        The ``benchmark:`` block defines the grid; everything else (llm,
        skills, policies, trials, suites, …) is parsed by
        :meth:`PipelineConfig.from_yaml` and stashed on
        ``pipeline_config``. A YAML without a ``benchmark:`` block but
        with explicit ``suites:`` loads in *suites mode*.
        """
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        pipeline_config = PipelineConfig.from_yaml(path)  # raises on `base:`
        b = raw.get("benchmark")
        suites_mode = b is None
        if suites_mode and not pipeline_config.suites:
            raise ValueError(
                f"{path}: neither a 'benchmark:' grid block nor a "
                f"'suites:' list — nothing to run."
            )
        b = b or {}

        # Per-mode overrides; workflow_dir absolutized vs the YAML dir.
        overrides: dict[str, BenchmarkModeOverride] = {}
        for mode_name, ov_raw in (b.get("mode_overrides", {}) or {}).items():
            ov_raw = ov_raw or {}
            wf = ov_raw.get("workflow_dir")
            if wf:
                wf_p = Path(wf)
                wf = str(
                    wf_p.resolve()
                    if wf_p.is_absolute()
                    else (path.parent / wf_p).resolve()
                )
            overrides[mode_name] = BenchmarkModeOverride(
                n_seeds=ov_raw.get("n_seeds"),
                num_workers=ov_raw.get("num_workers"),
                time_budget_s=ov_raw.get("time_budget_s"),
                workflow_dir=wf,
                policy_id=ov_raw.get("policy_id"),
                extra=ov_raw.get("extra", {}) or {},
            )

        if suites_mode:
            # Suites-mode runs live under the pipeline's own output_dir.
            out_dir = pipeline_config.trials.output_dir
        else:
            out_dir = Path(b.get("output_dir", "./benchmark_runs"))
            if not out_dir.is_absolute():
                out_dir = (path.parent / out_dir).resolve()

        gate_threshold = float(
            b.get("gate_threshold", raw.get("gate_threshold", DEFAULT_GATE_THRESHOLD))
        )

        cfg = cls(
            source_yaml=path.resolve(),
            families=(
                list(b["families"]) if b.get("families") else ["posvar"]
            ),
            variations=(
                list(b["variations"]) if b.get("variations") else None
            ),
            modes=list(b.get("modes", list(KNOWN_MODES))),
            n_tasks=int(b.get("n_tasks", 10)),
            task_ids=(list(b["task_ids"]) if b.get("task_ids") else None),
            n_seeds=int(b.get("n_seeds", 50)),
            num_workers=int(b.get("num_workers", 8)),
            policies=list(b.get("policies") or []),
            record_video=bool(
                b.get("record_video", pipeline_config.trials.record_video)
            ),
            output_dir=out_dir,
            smoke=bool(b.get("smoke", False)),
            gate_threshold=gate_threshold,
            mode_overrides=overrides,
            suites_mode=suites_mode,
        )
        cfg.pipeline_config = pipeline_config
        return cfg
