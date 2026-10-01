"""Configuration dataclasses and YAML loader for the codegen pipeline.

Ported from the source pipeline's config, pruned to what the graph
generation pipeline actually consumes: ``task`` / ``suites`` / ``trials``
/ ``environment.cameras`` / ``safety_limits`` / ``llm`` / ``composition``
/ ``policies`` / ``policy_manager`` plus the new ``skills:`` path (the
open-robot-skills checkout the registries are built from).

Deleted relative to the source: rehearsal, scene_spec, GRPO, ray_serve,
services / api_servers, startup timeouts, sandbox and ``skill_bundles``
(replaced by the single ``skills:`` path). ``base:`` inheritance is gone
with platform.yaml — a config carrying ``base:`` errors with a clear
message instead of being silently merged.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .llm import LlmConfig

logger = logging.getLogger(__name__)


@dataclass
class TrialConfig:
    """Trial execution configuration."""

    trials_per_generation: int = 30
    task_ids: list[int] = field(default_factory=lambda: [0])
    code_generations: int = 1
    regenerate_code_per_trial: bool = False
    output_dir: Path = field(default_factory=lambda: Path("./outputs"))
    num_workers: int = 1
    record_video: bool = False
    enable_tracing: bool = False
    task_timeout_secs: int = 0
    """Per-trial wall-clock cap in seconds. ``0`` disables the watchdog.
    When > 0, a trial that exceeds the cap is hard-killed and recorded
    as a failed trial with ``exit_code=124`` (see gap.agent.parallel)."""


@dataclass
class SafetyLimitsConfig:
    """Runtime safety guard limits for LLM-generated code."""

    max_perception_calls: int = 50
    max_planning_calls: int = 20
    max_sim_steps: int = 5000


@dataclass
class EnvironmentConfig:
    """Simulation environment configuration."""

    cameras: list[str] = field(
        default_factory=lambda: ["agentview", "robot0_eye_in_hand"]
    )


@dataclass
class SuiteSpec:
    """Specification for a single evaluation suite."""

    suite_name: str = ""
    task_prompts: dict[int, str] = field(default_factory=dict)
    # Per-suite overrides — None means use top-level TrialConfig value.
    task_ids: list[int] | None = None
    trials_per_generation: int | None = None
    timeout_secs: int | None = None
    num_workers: int | None = None
    # Static workflow templating: ``{{<key>}}`` tokens substituted into a
    # copied workflow.json. Empty → no substitution.
    objects: dict[str, str] = field(default_factory=dict)

    def task_for(self, task_id: int, fallback: str = "") -> str:
        """Return the language prompt for *task_id*.

        The sentinel value ``"auto"`` is never returned as a usable
        prompt — it signals that resolution should have happened earlier.
        """
        if self.task_prompts and task_id in self.task_prompts:
            return self.task_prompts[task_id]
        if fallback == "auto":
            return ""
        return fallback


@dataclass
class CompositionConfig:
    """Multi-agent composition configuration."""

    subgraph_temperature: float | None = 0.3
    """Temperature for subagent LLM calls (lower = more focused)."""

    subgraph_model: str | None = None
    """Override model for subagents. Defaults to ``llm.model``."""

    coordinator_model: str | None = None
    """Override model for the coordinator agent. Defaults to ``llm.model``."""

    max_subgraph_retries: int = 2
    """Per-subagent retry limit when generation fails validation."""

    max_coordinator_retries: int = 2
    """Coordinator retry limit when decomposition fails validation."""

    max_validation_retries: int = 2
    """How many times to retry fixing graph validation errors via LLM."""

    max_codegen_regenerations: int = 2
    """If a whole graph is still structurally invalid after the per-attempt
    script-fix loop (e.g. a coordinator-level W8: a subgraph declares an input
    with no upstream producer — not repairable by the script-body fixer),
    regenerate the graph from scratch up to this many extra times. 0 disables
    (single attempt, matching the legacy behavior)."""

    checkpoint_agent: bool = True
    """Run the whole-workflow checkpoint_agent pass after the per-subgraph
    structure generation (default ON). When off, no postcondition sidecars
    are authored."""


@dataclass
class PipelineConfig:
    """Full pipeline configuration — loaded from a task YAML."""

    task: str = ""
    suites: list[SuiteSpec] = field(default_factory=list)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    trials: TrialConfig = field(default_factory=TrialConfig)
    safety_limits: SafetyLimitsConfig = field(default_factory=SafetyLimitsConfig)
    composition: CompositionConfig = field(default_factory=CompositionConfig)
    max_retries: int = 3

    skills: Path | list[Path] | None = None
    """Skill registry root(s) — one path or a precedence-ordered list
    (bundle discovery roots; see :mod:`gap.skills.registries`). The
    codegen pipeline builds its skill + tool registries from it."""

    # ``policies:`` — name → {start_cmd | url, env} entries used by the
    # learned-policy node type. See gap.runtime.policy_manager.
    policies: dict[str, dict[str, Any]] = field(default_factory=dict)

    # ``policy_manager: { startup_timeout_s, evict_grace_s }`` — tuning
    # knobs for the lifecycle manager.
    policy_manager: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load pipeline configuration from a YAML file.

        ``${VAR}`` env interpolation is applied to the ``skills:`` path;
        relative paths resolve against the YAML file's directory.
        """
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        if "base" in raw:
            raise ValueError(
                f"{path}: `base:` inheritance is not supported — "
                f"platform.yaml is gone. Inline the inherited keys into "
                f"this file instead."
            )

        # LLM config
        from .llm import default_model, default_provider

        llm_raw = raw.get("llm", {}) or {}
        llm = LlmConfig(
            provider=llm_raw.get("provider") or default_provider(),
            model=llm_raw.get("model") or default_model(),
            endpoint=llm_raw.get("endpoint"),
            api_key=llm_raw.get("api_key"),
            project_id=llm_raw.get("project_id"),
            region=llm_raw.get("region"),
            temperature=llm_raw.get("temperature", LlmConfig.temperature),
            max_tokens=llm_raw.get("max_tokens", LlmConfig.max_tokens),
            max_concurrent_requests=llm_raw.get(
                "max_concurrent_requests", LlmConfig.max_concurrent_requests,
            ),
            cache_dir=llm_raw.get("cache_dir"),
        )

        # Environment config
        env_raw = raw.get("environment", {}) or {}
        environment = EnvironmentConfig(
            cameras=env_raw.get("cameras", ["agentview", "robot0_eye_in_hand"]),
        )

        # Trials config
        trials_raw = raw.get("trials", {}) or {}
        trials = TrialConfig(
            trials_per_generation=trials_raw.get(
                "trials_per_generation", trials_raw.get("total", 30),
            ),
            task_ids=trials_raw.get("task_ids", [0]),
            code_generations=trials_raw.get("code_generations", 1),
            regenerate_code_per_trial=trials_raw.get(
                "regenerate_code_per_trial", False
            ),
            output_dir=Path(trials_raw.get("output_dir", "./outputs")).resolve(),
            num_workers=trials_raw.get("num_workers", 1),
            record_video=trials_raw.get("record_video", False),
            enable_tracing=trials_raw.get("enable_tracing", False),
            task_timeout_secs=int(trials_raw.get("task_timeout_secs", 0)),
        )

        # Suites
        suites = []
        for s in raw.get("suites", []) or []:
            raw_prompts = s.get("task_prompts", {})
            raw_objects = s.get("objects", {}) or {}
            suites.append(SuiteSpec(
                suite_name=s.get("suite_name", ""),
                task_prompts={int(k): str(v) for k, v in raw_prompts.items()},
                task_ids=s.get("task_ids"),
                trials_per_generation=s.get("trials_per_generation"),
                timeout_secs=s.get("timeout_secs"),
                num_workers=s.get("num_workers"),
                objects={str(k): str(v) for k, v in raw_objects.items()},
            ))

        # Safety limits
        safety_raw = raw.get("safety_limits", {}) or {}
        safety_limits = SafetyLimitsConfig(
            max_perception_calls=safety_raw.get("max_perception_calls", 50),
            max_planning_calls=safety_raw.get("max_planning_calls", 20),
            max_sim_steps=safety_raw.get("max_sim_steps", 5000),
        )

        # Composition config (multi-agent)
        comp_raw = raw.get("composition", {}) or {}
        composition = CompositionConfig(
            subgraph_temperature=comp_raw.get("subgraph_temperature", 0.3),
            subgraph_model=comp_raw.get("subgraph_model"),
            coordinator_model=comp_raw.get("coordinator_model"),
            max_subgraph_retries=comp_raw.get("max_subgraph_retries", 2),
            max_coordinator_retries=comp_raw.get("max_coordinator_retries", 2),
            max_validation_retries=comp_raw.get("max_validation_retries", 2),
            max_codegen_regenerations=comp_raw.get("max_codegen_regenerations", 2),
            checkpoint_agent=bool(comp_raw.get("checkpoint_agent", True)),
        )

        policies_raw = raw.get("policies") or {}
        if not isinstance(policies_raw, dict):
            raise ValueError(
                f"'policies:' must be a mapping of id -> entry, got "
                f"{type(policies_raw).__name__}"
            )
        policy_manager_raw = raw.get("policy_manager") or {}
        if not isinstance(policy_manager_raw, dict):
            raise ValueError(
                f"'policy_manager:' must be a mapping, got "
                f"{type(policy_manager_raw).__name__}"
            )

        # Skill registry root(s) — a single path or a precedence-ordered
        # list. ${VAR} env interpolation; relative paths resolve against
        # the YAML file's directory. When the key is omitted entirely,
        # the active registries are resolved ($GAP_SKILLS_PATH list >
        # project [tool.gap] > user config > an open-robot-skills
        # directory next to the gap checkout) — checked-in configs don't
        # need to hardcode it.
        def _skills_entry(entry: object) -> Path:
            p = Path(os.path.expandvars(str(entry))).expanduser()
            if not p.is_absolute():
                p = (path.parent / p).resolve()
            if not p.is_dir():
                raise ValueError(
                    f"'skills:' does not exist or is not a directory: {p}"
                )
            return p

        skills: Path | list[Path] | None = None
        skills_raw = raw.get("skills")
        if isinstance(skills_raw, list):
            skills = [_skills_entry(entry) for entry in skills_raw]
        elif skills_raw:
            skills = _skills_entry(skills_raw)
        else:
            from gap.skills import resolve_registries

            registry_set = resolve_registries()
            skills = registry_set.paths() if registry_set else None

        return cls(
            task=raw.get("task", ""),
            suites=suites,
            environment=environment,
            llm=llm,
            trials=trials,
            safety_limits=safety_limits,
            composition=composition,
            max_retries=raw.get("max_retries", llm_raw.get("max_retries", 3)),
            skills=skills,
            policies=policies_raw,
            policy_manager=policy_manager_raw,
        )
