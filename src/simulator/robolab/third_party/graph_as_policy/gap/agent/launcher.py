"""Pipeline launcher: code generation → parallel trials → aggregation.

Ported from the source ``compose/launcher.py`` and reworked for the gap
in-process runtime. Deleted relative to the source: rehearsal passes,
scene_spec resolution, Ray Serve / services.yaml generation, host
api_servers, and sim_bridge subprocess spawning — workers build their
own in-process connectors (see :mod:`gap.agent.parallel`).

Flow per suite:

1. Per-task code generation via :func:`gap.agent.multi_agent.run_codegen`
   (or a pre-built / templated workflow directory).
2. Build a :class:`~gap.agent.parallel.WorkerSetupConfig` + lightweight
   :class:`~gap.agent.parallel.WorkItem` list.
3. Dispatch via :func:`~gap.agent.parallel.run_parallel_trials`.
4. Aggregate per-task and per-suite results; save artifacts + done flag.

The ``objects:`` block of a :class:`~gap.agent.config.SuiteSpec` is
consumed twice (both load-bearing for the grocery acceptance gate):

* **codegen context** — :func:`codegen_prompt` appends the suite's
  object hints (``target`` / ``expected_label`` / ``shape_hint`` / …)
  to the task prompt, so the coordinator + subgraph agents choose
  unambiguous perception prompts and success labels;
* **workflow templating** — for pre-built workflows,
  :func:`materialize_workflow` substitutes ``{{<key>}}`` placeholders
  into a copied ``workflow.json``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .config import PipelineConfig, SuiteSpec
from .multi_agent import PipelineResult
from .parallel import (
    TrialResult,
    WorkerSetupConfig,
    WorkItem,
    run_parallel_trials,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TaskResult:
    """Aggregate result for a single task."""

    task_id: int = 0
    success_count: int = 0
    total_trials: int = 0
    success_rate: float = 0.0
    completion_rate: float = 0.0
    avg_reward: float = 0.0
    avg_physical_execution_s: float = 0.0
    """Mean across trials of the physical-execution estimate."""
    avg_control_steps: float = 0.0
    avg_sim_physics_wall_s: float = 0.0
    trial_results: list[TrialResult] = field(default_factory=list)


@dataclass
class SuiteLaunchResult:
    """Aggregate result for a single suite."""

    suite_name: str = ""
    task_results: list[TaskResult] = field(default_factory=list)
    trial_results: list[TrialResult] = field(default_factory=list)
    success_rate: float = 0.0
    completion_rate: float = 0.0
    total_trials: int = 0


@dataclass
class LaunchResult:
    """Aggregate result of the full launch pipeline."""

    suite_results: list[SuiteLaunchResult] = field(default_factory=list)
    task_results: list[TaskResult] = field(default_factory=list)
    trial_results: list[TrialResult] = field(default_factory=list)
    success_rate: float = 0.0
    completion_rate: float = 0.0
    total_trials: int = 0


# ---------------------------------------------------------------------------
# Task-prompt resolution (``task: "auto"``) + objects-block plumbing
# ---------------------------------------------------------------------------


def _resolve_libero_prompts(
    suite_name: str, task_ids: list[int],
) -> dict[int, str]:
    """Resolve language instructions from LIBERO task metadata.

    Lightweight: loads only task metadata, never constructs a sim env.
    Handles both fork layouts (see :mod:`gap.envs.loader`): vab task-dir
    suites read the task YAML's ``language``; classic suites go through
    the LIBERO-PRO benchmark registry. Suite aliases resolve through
    :func:`gap.envs.registry.registered_envs`.
    """
    try:
        from gap.envs.loader import (
            _activate_libero_fork,
            _vab_suite_dir,
            _vab_task_files,
        )
        from gap.envs.registry import registered_envs
    except ImportError:
        logger.debug("gap.envs unavailable — cannot auto-resolve prompts")
        return {}

    canonical = registered_envs().get(suite_name, suite_name)
    resolved: dict[int, str] = {}

    vab_dir = _vab_suite_dir(canonical)
    if vab_dir is not None:
        try:
            _activate_libero_fork("vab")
            from libero.vab import load_task  # type: ignore[import-not-found]
        except Exception:
            logger.debug("vab fork unavailable", exc_info=True)
            return {}
        # Classic LIBERO task numbering — MUST match the sim's ordering
        # (``_vab_task_files``). A bare ``sorted(glob)`` renumbers 8 of the 10
        # libero_object tasks alphabetically, so the generated graph targets a
        # different object than the env scores (see _vab_task_files docstring).
        task_files = _vab_task_files(vab_dir)
        for tid in task_ids:
            if not 0 <= tid < len(task_files):
                continue
            try:
                lang = load_task(task_files[tid]).language
            except Exception:
                continue
            if lang:
                resolved[tid] = str(lang)
        return resolved

    try:
        _activate_libero_fork("pro")
        from libero import benchmark  # type: ignore[import-not-found]

        benchmark_dict = benchmark.get_benchmark_dict(help=True)
        if canonical not in benchmark_dict:
            logger.debug("Suite %r not in LIBERO benchmark registry", canonical)
            return {}
        task_suite = benchmark_dict[canonical]()
    except Exception:
        logger.debug("Failed to load LIBERO suite %r", canonical, exc_info=True)
        return {}

    for tid in task_ids:
        try:
            lang = getattr(task_suite.get_task(tid), "language", None)
        except Exception:
            continue
        if lang:
            resolved[tid] = str(lang)
    return resolved


def codegen_prompt(suite: SuiteSpec, task_id: int, fallback: str = "") -> str:
    """The task prompt handed to the codegen pipeline for one task.

    Resolves the language prompt via :meth:`SuiteSpec.task_for` and, when
    the suite carries an ``objects:`` block, appends it as a structured
    object-context section. This is how the hand-curated ``target`` /
    ``expected_label`` / ``shape_hint`` hints of the grocery acceptance
    recipe reach the coordinator + subgraph-agent prompts.
    """
    prompt = suite.task_for(task_id, fallback)
    if not prompt:
        return ""
    if not suite.objects:
        return prompt
    lines = [
        prompt,
        "",
        "Object context (authoritative hints — use these to choose "
        "unambiguous perception prompts and expected labels):",
    ]
    lines.extend(f"- {key}: {value}" for key, value in suite.objects.items())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Static workflow templating
# ---------------------------------------------------------------------------


def materialize_workflow(
    src_workflow_dir: str,
    objects: dict[str, str],
    dest_parent: Path,
) -> str:
    """Copy ``src_workflow_dir`` to ``dest_parent/workflow`` and substitute
    ``{{<key>}}`` placeholders inside ``workflow.json``.

    Returns the materialized workflow directory path. The result is
    JSON-validated so a stray placeholder or bad value fails fast.
    """
    src = Path(src_workflow_dir).resolve()
    dst = (Path(dest_parent) / "workflow").resolve()
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    workflow_json_path = dst / "workflow.json"
    if not workflow_json_path.is_file():
        raise FileNotFoundError(
            f"Templated workflow {src} missing workflow.json"
        )
    text = workflow_json_path.read_text()
    for key, value in objects.items():
        text = text.replace("{{" + key + "}}", value)
    try:
        json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"workflow.json templating produced invalid JSON for "
            f"{src_workflow_dir} with objects={objects}: {e}"
        ) from e
    workflow_json_path.write_text(text)
    return str(dst)


# Backwards-compatible alias (the source helper's private name).
_materialize_workflow = materialize_workflow


# ---------------------------------------------------------------------------
# Suite execution
# ---------------------------------------------------------------------------


async def _codegen_for_suite(
    config: PipelineConfig,
    suite: SuiteSpec,
    suite_output_dir: Path,
    task_ids: list[int],
    trials_per_gen: int,
) -> tuple[
    dict[tuple[int, int], str],
    dict[tuple[int, int, int], str],
]:
    """Generate workflows for every (task, codegen[, trial]) of a suite.

    Returns ``(codegen_map, per_trial_wf)``:
    ``codegen_map[(task_id, codegen_id)] = workflow_dir`` and, when
    ``regenerate_code_per_trial`` is set,
    ``per_trial_wf[(task_id, codegen_id, trial_id)] = workflow_dir``.
    """
    from .multi_agent import run_codegen

    num_codegens = config.trials.code_generations
    codegen_map: dict[tuple[int, int], str] = {}
    per_trial_wf: dict[tuple[int, int, int], str] = {}

    def _prompt_or_raise(tid: int) -> str:
        prompt = codegen_prompt(suite, tid, config.task)
        if not prompt:
            raise ValueError(
                f"No task prompt for task {tid} in suite "
                f"'{suite.suite_name}'. Add it to task_prompts in the "
                f"YAML config, or ensure LIBERO is available for "
                f"auto-resolution (task: 'auto')."
            )
        return prompt

    if config.trials.regenerate_code_per_trial:
        async def _one(
            tid: int, cid: int, trial_id: int,
        ) -> tuple[int, int, int, PipelineResult]:
            trial_out = (
                suite_output_dir / f"task_{tid:02d}"
                / f"trial_{trial_id:02d}" / "codegen"
            )
            pr = await run_codegen(
                task_id=tid,
                task_prompt=_prompt_or_raise(tid),
                config=config,
                output_dir=trial_out,
            )
            return tid, cid, trial_id, pr

        coros = [
            _one(tid, cid, trial_id)
            for tid in task_ids
            for cid in range(1, num_codegens + 1)
            for trial_id in range(1, trials_per_gen + 1)
        ]
        logger.info(
            "regenerate_code_per_trial=True: launching %d codegen(s)",
            len(coros),
        )
        gathered = await asyncio.gather(*coros, return_exceptions=True)
        for result in gathered:
            if isinstance(result, BaseException):
                logger.error("Per-trial codegen raised: %s", result)
                continue
            tid, cid, trial_id, pr = result
            if pr.success and pr.workflow_dir is not None:
                per_trial_wf[(tid, cid, trial_id)] = str(pr.workflow_dir)
                codegen_map.setdefault((tid, cid), str(pr.workflow_dir))
            else:
                logger.error(
                    "Codegen failed for task %d codegen %d trial %d: %s",
                    tid, cid, trial_id, pr.execution_stderr,
                )
        return codegen_map, per_trial_wf

    async def _one_task(tid: int, cid: int) -> tuple[int, int, PipelineResult]:
        out_dir = suite_output_dir
        if num_codegens > 1:
            out_dir = out_dir / f"codegen_{cid:02d}"
        pr = await run_codegen(
            task_id=tid,
            task_prompt=_prompt_or_raise(tid),
            config=config,
            output_dir=out_dir,
        )
        return tid, cid, pr

    coros = [
        _one_task(tid, cid)
        for tid in task_ids
        for cid in range(1, num_codegens + 1)
    ]
    logger.info(
        "Launching graph workflow generation for %d task(s) x %d codegen(s)",
        len(task_ids), num_codegens,
    )
    gathered = await asyncio.gather(*coros, return_exceptions=True)
    for result in gathered:
        if isinstance(result, BaseException):
            logger.error("Codegen raised: %s", result)
            continue
        tid, cid, pr = result
        if pr.success and pr.workflow_dir is not None:
            codegen_map[(tid, cid)] = str(pr.workflow_dir)
        else:
            logger.error(
                "Codegen failed for task %d codegen %d: %s",
                tid, cid, pr.execution_stderr,
            )
    return codegen_map, per_trial_wf


async def _run_suite(
    config: PipelineConfig,
    suite: SuiteSpec,
    suite_output_dir: Path,
    *,
    workflow_dir: str | None = None,
    workflow_dir_map: dict[int, str] | None = None,
    device_slot_offset: int = 0,
) -> SuiteLaunchResult:
    """Run the codegen → trials → aggregate flow for a single suite."""
    task_ids = suite.task_ids or config.trials.task_ids
    trials_per_gen = (
        suite.trials_per_generation or config.trials.trials_per_generation
    )
    timeout_secs = suite.timeout_secs or config.trials.task_timeout_secs
    num_workers = suite.num_workers or config.trials.num_workers
    num_codegens = config.trials.code_generations
    multi_codegen = num_codegens > 1

    suite_output_dir.mkdir(parents=True, exist_ok=True)

    # --- Resolve "auto" task prompts from LIBERO metadata ---
    if config.task == "auto":
        missing = [tid for tid in task_ids if tid not in suite.task_prompts]
        if missing:
            resolved = _resolve_libero_prompts(suite.suite_name, missing)
            for tid, prompt in resolved.items():
                suite.task_prompts[tid] = prompt
                logger.info("Auto-resolved task %d prompt: %s", tid, prompt)
            still_missing = [tid for tid in missing if tid not in resolved]
            if still_missing:
                logger.warning(
                    "Suite %s: could not auto-resolve prompts for task(s) %s",
                    suite.suite_name, still_missing,
                )

    logger.info(
        "=== Suite %s: %d task(s), %d trial(s)/gen, %d worker(s) ===",
        suite.suite_name, len(task_ids), trials_per_gen, num_workers,
    )

    # --- Workflows: pre-built / per-task map / codegen ---
    codegen_map: dict[tuple[int, int], str] = {}
    per_trial_wf: dict[tuple[int, int, int], str] = {}

    if workflow_dir_map is not None:
        for tid in task_ids:
            wf = workflow_dir_map.get(tid)
            if wf is None:
                logger.warning(
                    "workflow_dir_map has no entry for task %d — skipping", tid,
                )
                continue
            for cid in range(1, num_codegens + 1):
                codegen_map[(tid, cid)] = wf
        logger.info(
            "Using %d pre-built per-task workflow(s) (skipping codegen)",
            len({wf for wf in workflow_dir_map.values()}),
        )
    elif workflow_dir:
        effective = workflow_dir
        if suite.objects:
            effective = materialize_workflow(
                workflow_dir, suite.objects, suite_output_dir,
            )
            logger.info(
                "Using pre-built workflow from %s (templated -> %s)",
                workflow_dir, effective,
            )
        else:
            logger.info("Using pre-built workflow from %s", workflow_dir)
        for tid in task_ids:
            for cid in range(1, num_codegens + 1):
                codegen_map[(tid, cid)] = effective
    else:
        codegen_map, per_trial_wf = await _codegen_for_suite(
            config, suite, suite_output_dir, task_ids, trials_per_gen,
        )

    if not codegen_map:
        logger.error("Suite %s: no workflows produced", suite.suite_name)
        return SuiteLaunchResult(suite_name=suite.suite_name)

    # --- Build WorkerSetupConfig ---
    sl = config.safety_limits
    setup_config = WorkerSetupConfig(
        suite_name=suite.suite_name,
        skills_path=(
            os.pathsep.join(
                str(p) for p in (
                    config.skills if isinstance(config.skills, list)
                    else [config.skills]
                )
            )
            if config.skills else ""
        ),
        camera_names=list(config.environment.cameras),
        record_video=config.trials.record_video,
        enable_tracing=config.trials.enable_tracing,
        task_timeout_secs=float(timeout_secs or 0),
        output_dir=str(suite_output_dir),
        multi_codegen=multi_codegen,
        safety_limits={
            "perception": sl.max_perception_calls,
            "planning": sl.max_planning_calls,
            "sim_step": sl.max_sim_steps,
        },
        policies=dict(config.policies),
        policy_manager=dict(config.policy_manager),
        extra_env=_llm_env_for_workers(config),
        device_slot_offset=device_slot_offset,
    )

    # --- Build WorkItems ---
    work_items: list[WorkItem] = []
    for task_id in task_ids:
        for codegen_id in range(1, num_codegens + 1):
            if (task_id, codegen_id) not in codegen_map:
                continue
            wf_dir = codegen_map[(task_id, codegen_id)]
            for trial_id in range(1, trials_per_gen + 1):
                if per_trial_wf:
                    key = (task_id, codegen_id, trial_id)
                    if key not in per_trial_wf:
                        logger.warning(
                            "No per-trial workflow for task %d codegen %d "
                            "trial %d, skipping",
                            task_id, codegen_id, trial_id,
                        )
                        continue
                    wf_dir = per_trial_wf[key]
                work_items.append(WorkItem(
                    task_id=task_id,
                    trial_id=trial_id,
                    codegen_id=codegen_id,
                    workflow_dir=wf_dir,
                ))

    # --- Dispatch ---
    logger.info(
        "Suite %s: dispatching %d work items across %d worker(s)",
        suite.suite_name, len(work_items), num_workers,
    )
    loop = asyncio.get_running_loop()

    def _on_complete(r: TrialResult) -> None:
        _save_trial_artifacts(suite_output_dir, num_codegens, r)

    all_results = await loop.run_in_executor(
        None,
        lambda: run_parallel_trials(
            work_items, num_workers, setup_config,
            on_complete=_on_complete,
        ),
    )

    # --- Aggregate ---
    task_results = _aggregate_tasks(suite_output_dir, task_ids, num_codegens, all_results)

    total = len(all_results)
    successes = sum(1 for r in all_results if r.task_completed)
    completion = (
        sum(r.completion_rate for r in all_results) / total if total else 0.0
    )
    suite_result = SuiteLaunchResult(
        suite_name=suite.suite_name,
        task_results=task_results,
        trial_results=all_results,
        success_rate=successes / total if total else 0.0,
        completion_rate=completion,
        total_trials=total,
    )

    _save_aggregate(suite_output_dir, suite_result)

    done_dir = suite_output_dir / "aaa_done_flag"
    done_dir.mkdir(parents=True, exist_ok=True)
    (done_dir / "aaa_done_flag.txt").write_text(
        f"Suite {suite.suite_name}: {total} trials. "
        f"Success rate: {suite_result.success_rate:.1%}\n"
    )

    _print_summary(suite_result)
    return suite_result


def _llm_env_for_workers(config: PipelineConfig) -> dict[str, str]:
    """Credential env vars propagated into spawned workers.

    Workers may invoke VLM-backed tools (perception bundles call the
    configured LLM provider), so the parent's relevant credential vars
    must survive the spawn — spawn inherits os.environ, but explicit
    config-derived values (api_key / project_id) only exist on the
    config object and are exported here.
    """
    env: dict[str, str] = {}
    llm = config.llm
    if llm.api_key:
        key_var = {
            "openrouter": "OPENROUTER_API_KEY",
        }.get(llm.provider)
        if key_var:
            env[key_var] = str(llm.api_key)
    if llm.project_id:
        env["GOOGLE_CLOUD_PROJECT"] = str(llm.project_id)
    if llm.region:
        env["GOOGLE_CLOUD_REGION"] = str(llm.region)
    return env


# ---------------------------------------------------------------------------
# Main launch entry point
# ---------------------------------------------------------------------------


def _suite_dir_names(suites: list[SuiteSpec]) -> list[str]:
    """Per-suite output dir names; duplicate suite names get _NN suffixes."""
    seen: dict[str, int] = {}
    names = []
    for s in suites:
        n = seen.get(s.suite_name, 0) + 1
        seen[s.suite_name] = n
        names.append(s.suite_name if n == 1 else f"{s.suite_name}_{n:02d}")
    return names


async def launch(
    config: PipelineConfig,
    *,
    workflow_dir: str | None = None,
    workflow_dir_map: dict[int, str] | None = None,
    clean_output: bool = True,
    device_slot_offset: int = 0,
) -> LaunchResult:
    """Run the full pipeline: codegen → parallel trials → aggregation.

    Args:
        config: Full pipeline configuration (from a task YAML).
        workflow_dir: Optional pre-built workflow directory used for every
            task (skips codegen). Suites carrying an ``objects:`` block
            get a ``{{key}}``-templated copy.
        workflow_dir_map: Optional per-task pre-built workflow map
            (``{task_id: dir}``) — the benchmark template modes use this.
        clean_output: Remove ``trials.output_dir`` before running
            (default, the source behavior). The benchmark harness passes
            False and manages per-cell directories itself.

    Returns:
        LaunchResult with per-suite, per-task, and aggregate outcomes.
    """
    base_output_dir = config.trials.output_dir
    multi_suite = len(config.suites) > 1

    if clean_output and base_output_dir.exists():
        def _force_remove(func, path, _exc_info):
            import os as _os
            import stat as _stat
            _os.chmod(path, _stat.S_IRUSR | _stat.S_IWUSR)
            func(path)
        shutil.rmtree(base_output_dir, onerror=_force_remove)
        logger.info("Cleaned output directory %s", base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # File handler so the run's logs land next to its artifacts.
    _file_handler = logging.FileHandler(base_output_dir / "launcher.log")
    _file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(_file_handler)

    try:
        suite_coros = []
        # Stagger concurrent cells across the EGL/CUDA device list: each
        # cell's workers start at a different device slot, so 10 cells x 1
        # worker spread over the GPUs instead of all landing on device[0].
        # ``device_slot_offset`` seeds the stagger for callers that run
        # several launch() invocations concurrently (the benchmark
        # harness's suites mode runs one single-suite launch per cell).
        device_slot = int(device_slot_offset)
        for suite, dir_name in zip(
            config.suites, _suite_dir_names(config.suites), strict=False
        ):
            suite_output_dir = (
                base_output_dir / dir_name if multi_suite else base_output_dir
            )
            suite_coros.append(_run_suite(
                config, suite, suite_output_dir,
                workflow_dir=workflow_dir,
                workflow_dir_map=workflow_dir_map,
                device_slot_offset=device_slot,
            ))
            device_slot += max(1, int(suite.num_workers or 1))
        suite_results = list(await asyncio.gather(*suite_coros))

        all_task_results = [tr for sr in suite_results for tr in sr.task_results]
        all_trial_results = [tr for sr in suite_results for tr in sr.trial_results]
        total = sum(sr.total_trials for sr in suite_results)
        successes = sum(
            1 for sr in suite_results
            for r in sr.trial_results if r.task_completed
        )
        completion = (
            sum(
                r.completion_rate
                for sr in suite_results for r in sr.trial_results
            ) / total if total else 0.0
        )

        launch_result = LaunchResult(
            suite_results=suite_results,
            task_results=all_task_results,
            trial_results=all_trial_results,
            success_rate=successes / total if total else 0.0,
            completion_rate=completion,
            total_trials=total,
        )

        if multi_suite:
            _save_cross_suite_aggregate(base_output_dir, suite_results)
            _print_multi_suite_summary(suite_results)

        done_dir = base_output_dir / "aaa_done_flag"
        done_dir.mkdir(parents=True, exist_ok=True)
        (done_dir / "aaa_done_flag.txt").write_text(
            f"Completed {total} trials across {len(config.suites)} suite(s). "
            f"Success rate: {launch_result.success_rate:.1%}\n"
        )
        return launch_result

    finally:
        logging.getLogger().removeHandler(_file_handler)
        _file_handler.close()


# ---------------------------------------------------------------------------
# Artifacts + aggregation
# ---------------------------------------------------------------------------


def _trial_dir(
    output_dir: Path, task_id: int, trial_id: int,
    codegen_id: int = 1, multi_codegen: bool = False,
) -> Path:
    task_dir = output_dir / f"task_{task_id:02d}"
    if multi_codegen:
        task_dir = task_dir / f"codegen_{codegen_id:02d}"
    return task_dir / f"trial_{trial_id:02d}"


def _save_trial_artifacts(
    output_dir: Path, code_generations: int, result: TrialResult,
) -> None:
    """Rename the trial directory to a descriptive name + save outputs.

    ``trial_NN`` → ``trial_NN_rc<code>_reward<r>_{pass,fail}`` so every
    artifact (result.json, trace, video, codegen copy) is consolidated
    and the outcome is visible from a directory listing.
    """
    status = "pass" if result.task_completed else "fail"
    trial_name = (
        f"trial_{result.trial_id:02d}"
        f"_rc{result.exit_code}"
        f"_reward{result.reward:.3f}"
        f"_{status}"
    )
    multi_codegen = code_generations > 1
    task_dir = output_dir / f"task_{result.task_id:02d}"
    if multi_codegen:
        task_dir = task_dir / f"codegen_{result.codegen_id:02d}"
    final_dir = task_dir / trial_name

    intermediate_dir = _trial_dir(
        output_dir, result.task_id, result.trial_id,
        codegen_id=result.codegen_id, multi_codegen=multi_codegen,
    )
    if intermediate_dir.exists() and not final_dir.exists():
        intermediate_dir.rename(final_dir)
    else:
        final_dir.mkdir(parents=True, exist_ok=True)

    if result.execution_stdout and not (final_dir / "stdout.txt").exists():
        (final_dir / "stdout.txt").write_text(result.execution_stdout)
    if result.execution_stderr and not (final_dir / "stderr.txt").exists():
        (final_dir / "stderr.txt").write_text(result.execution_stderr)

    # Track the video into the renamed directory.
    if result.video_path:
        video_src = Path(result.video_path)
        if str(video_src).startswith(str(intermediate_dir)):
            result.video_path = str(
                final_dir / video_src.relative_to(intermediate_dir)
            )
        elif video_src.exists() and not str(video_src).startswith(str(final_dir)):
            video_dest = final_dir / "video.mp4"
            if not video_dest.exists():
                shutil.move(str(video_src), str(video_dest))
                result.video_path = str(video_dest)


def _aggregate_tasks(
    output_dir: Path,
    task_ids: list[int],
    code_generations: int,
    all_results: list[TrialResult],
) -> list[TaskResult]:
    """Aggregate results per task and save per-task results.json."""
    from collections import defaultdict

    by_task: dict[int, list[TrialResult]] = defaultdict(list)
    for r in all_results:
        by_task[r.task_id].append(r)

    task_results = []
    for task_id in task_ids:
        trials = by_task.get(task_id, [])
        successes = sum(1 for t in trials if t.task_completed)
        total = len(trials)
        avg_reward = sum(t.reward for t in trials) / total if total else 0.0
        completion = (
            sum(t.completion_rate for t in trials) / total if total else 0.0
        )
        avg_physical_execution_s = (
            sum(t.physical_execution_s for t in trials) / total if total else 0.0
        )
        avg_control_steps = (
            sum(t.control_steps for t in trials) / total if total else 0.0
        )
        avg_sim_physics_wall_s = (
            sum(t.sim_physics_wall_s for t in trials) / total if total else 0.0
        )

        tr = TaskResult(
            task_id=task_id,
            success_count=successes,
            total_trials=total,
            success_rate=successes / total if total else 0.0,
            completion_rate=completion,
            avg_reward=avg_reward,
            avg_physical_execution_s=avg_physical_execution_s,
            avg_control_steps=avg_control_steps,
            avg_sim_physics_wall_s=avg_sim_physics_wall_s,
            trial_results=trials,
        )
        task_results.append(tr)

        task_dir = output_dir / f"task_{task_id:02d}"
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "results.json").write_text(json.dumps({
            "task_id": task_id,
            "code_generations": code_generations,
            "success_rate": tr.success_rate,
            "completion_rate": tr.completion_rate,
            "avg_reward": tr.avg_reward,
            "total_trials": total,
            "success_count": successes,
            "trials": [
                {
                    "trial_id": t.trial_id,
                    "codegen_id": t.codegen_id,
                    "seed": t.seed,
                    "task_completed": t.task_completed,
                    "reward": t.reward,
                    "completion_rate": t.completion_rate,
                    "exit_code": t.exit_code,
                    "duration_secs": t.duration_secs,
                    "video_path": t.video_path,
                }
                for t in trials
            ],
        }, indent=2))

    return task_results


def _save_aggregate(output_dir: Path, result: SuiteLaunchResult) -> None:
    """Save aggregate_results.json for a single suite."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "aggregate_results.json").write_text(json.dumps({
        "suite_name": result.suite_name,
        "success_rate": result.success_rate,
        "completion_rate": result.completion_rate,
        "total_trials": result.total_trials,
        "tasks": [
            {
                "task_id": tr.task_id,
                "success_rate": tr.success_rate,
                "completion_rate": tr.completion_rate,
                "avg_reward": tr.avg_reward,
                "total_trials": tr.total_trials,
                "success_count": tr.success_count,
            }
            for tr in result.task_results
        ],
    }, indent=2))
    logger.info("Aggregate results saved to %s", output_dir)


def _save_cross_suite_aggregate(
    base_output_dir: Path,
    suite_results: list[SuiteLaunchResult],
) -> None:
    """Save cross_suite_results.json combining all suites."""
    total = sum(sr.total_trials for sr in suite_results)
    successes = sum(
        sum(1 for r in sr.trial_results if r.task_completed)
        for sr in suite_results
    )
    (base_output_dir / "cross_suite_results.json").write_text(json.dumps({
        "success_rate": successes / total if total else 0.0,
        "total_trials": total,
        "suites": [
            {
                "suite_name": sr.suite_name,
                "success_rate": sr.success_rate,
                "completion_rate": sr.completion_rate,
                "total_trials": sr.total_trials,
                "tasks": [
                    {
                        "task_id": tr.task_id,
                        "success_rate": tr.success_rate,
                        "completion_rate": tr.completion_rate,
                        "avg_reward": tr.avg_reward,
                        "total_trials": tr.total_trials,
                        "success_count": tr.success_count,
                    }
                    for tr in sr.task_results
                ],
            }
            for sr in suite_results
        ],
    }, indent=2))
    logger.info("Cross-suite results saved to %s", base_output_dir)


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def _print_summary(result: SuiteLaunchResult) -> None:
    """Print per-task and aggregate summary tables for a single suite."""
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {result.suite_name}")
    print("=" * 60)

    print(f"\n{'Task':>6} {'Pass':>6} {'Total':>6} {'Rate':>8} {'Avg Reward':>12}")
    print("-" * 42)
    for tr in result.task_results:
        print(
            f"{tr.task_id:>6} {tr.success_count:>6} {tr.total_trials:>6} "
            f"{tr.success_rate:>7.1%} {tr.avg_reward:>12.3f}"
        )

    print("-" * 42)
    total_success = sum(tr.success_count for tr in result.task_results)
    total_trials = result.total_trials
    avg_reward = (
        sum(tr.avg_reward * tr.total_trials for tr in result.task_results)
        / total_trials
        if total_trials
        else 0.0
    )
    print(
        f"{'ALL':>6} {total_success:>6} {total_trials:>6} "
        f"{result.success_rate:>7.1%} {avg_reward:>12.3f}"
    )
    print("=" * 60)


def _print_multi_suite_summary(suite_results: list[SuiteLaunchResult]) -> None:
    """Print cross-suite summary table."""
    print(f"\n{'=' * 60}")
    print("CROSS-SUITE SUMMARY")
    print("=" * 60)

    print(f"\n{'Suite':<25} {'Pass':>6} {'Total':>6} {'Rate':>8}")
    print("-" * 48)

    grand_pass = 0
    grand_total = 0
    for sr in suite_results:
        passed = sum(1 for r in sr.trial_results if r.task_completed)
        grand_pass += passed
        grand_total += sr.total_trials
        print(
            f"{sr.suite_name:<25} {passed:>6} {sr.total_trials:>6} "
            f"{sr.success_rate:>7.1%}"
        )

    print("-" * 48)
    overall = grand_pass / grand_total if grand_total else 0.0
    print(f"{'ALL':<25} {grand_pass:>6} {grand_total:>6} {overall:>7.1%}")
    print("=" * 60)
