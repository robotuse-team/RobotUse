"""Parallel worker infrastructure for multi-trial execution.

Reworked from the source ``compose/parallel.py``. Workers no longer
spawn sim_bridge / skill-server subprocesses: each worker builds its own
**in-process** connector + tool registry inside :func:`worker_setup`,
post-fork (model weights and MuJoCo handles are not picklable), and runs
trials through the same :class:`gap.runtime.executor.WorkflowExecutor`
surface that ``gap.execute`` uses.

Kept from the source design:

* the ``mp.spawn`` worker pool (CUDA/MuJoCo safe) with poison-pill
  shutdown and liveness checking;
* the task-reuse optimization — work items are sorted by ``task_id`` so
  consecutive same-task trials on one worker skip env reconstruction and
  only ``reset(seed)`` (LIBERO scene-load is 5-15 s/trial);

Pool lifecycle (reworked after the 2026-06-10 livelock incident):

* workers persist across trials: they block on the work queue until the
  pool sends poison pills at shutdown — an idle worker never exits on
  its own, so a clean exit is never mistaken for a crash;
* every trial is tracked item-by-item: workers announce ``started`` /
  ``result`` messages, so when a worker process dies mid-trial (segfault
  in native sim/perception code, watchdog ``os._exit(124)``, OOM-kill)
  the in-flight item is recovered — re-enqueued up to
  ``max_item_retries`` times, then recorded as a failed
  :class:`TrialResult` (``exit_code=137``) instead of being silently
  lost;
* respawns are bounded: at most ``max_respawns_per_slot`` replacement
  processes per worker slot, and a replacement is spawned only when
  queued work actually exists. When every slot has burnt its budget the
  remaining trials are failed (with the crash diagnostics in
  ``execution_stderr``) and the pool concludes — never an infinite
  respawn/churn loop;
* positive completion: the pool returns exactly when every expected
  item has a result (real or synthesized), then poison-pills and joins
  the workers.
* EGL spread: ``GAP_MUJOCO_EGL_DEVICES=<csv>`` round-robins workers
  across GPUs via per-worker ``MUJOCO_EGL_DEVICE_ID``;
* the per-trial watchdog: ``task_timeout_secs > 0`` hard-kills a stuck
  trial and records a failed result with ``exit_code=124``. In a spawned
  worker the whole process exits (the pool respawns a replacement); in
  the in-process sequential path the hung trial thread is abandoned and
  the connector rebuilt for the next trial.

VRAM note: workers share nothing — each loads its own perception models
(sam3 + dino ≈ a few GB), so an A100-80GB holds several workers; spread
the rest with ``GAP_MUJOCO_EGL_DEVICES``. ``num_workers`` comes from the
task config (the G1 grocery recipe used 25). For shared skill actors see
the optional ``[ray]`` path in :mod:`gap.tools.ray_executor`.
"""

from __future__ import annotations

import importlib
import json
import logging
import multiprocessing as mp
import os
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Comma-separated GPU ids to round-robin MuJoCo EGL across, e.g.
#: ``GAP_MUJOCO_EGL_DEVICES=1,2,3`` keeps GPU 0 for a co-tenant model
#: server. Worker *i* gets ``MUJOCO_EGL_DEVICE_ID=devices[i % len]``.
EGL_DEVICES_ENV = "GAP_MUJOCO_EGL_DEVICES"

#: Seconds between worker process launches (EGL/MuJoCo init stampede guard).
_STAGGER_SECS = 1.0

#: Seconds the result loop waits for a message before running a
#: liveness/maintenance pass (reap dead workers, recover lost items,
#: respawn). Tests monkeypatch this down for fast crash recovery.
_POLL_SECS = 30.0

#: Default respawn budget per worker slot. A slot whose processes keep
#: dying (crash loop) is retired after this many replacements.
_DEFAULT_MAX_RESPAWNS_PER_SLOT = 3

#: Default re-run budget for an item whose worker died mid-trial. Beyond
#: this the item is recorded as a failed TrialResult (exit_code=137).
_DEFAULT_MAX_ITEM_RETRIES = 1

#: exit_code recorded for trials whose worker died and whose retry
#: budget is exhausted (128+SIGKILL by convention; distinct from the
#: watchdog's 124).
_CRASHED_EXIT_CODE = 137


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class WorkerSetupConfig:
    """Serializable config that crosses the mp.spawn boundary.

    Every field must be a picklable primitive (str / int / float / bool /
    dict / list) — no Path objects, no model handles, no connectors.
    """

    suite_name: str = ""
    """Default suite for work items that don't carry their own."""

    skills_path: str = ""
    """Skill registry root(s) the worker loads its skill/tool registries
    from — ``os.pathsep``-separated, precedence-ordered (picklable across
    the mp.spawn boundary). Empty string → no bundles (connector tools
    only)."""

    camera_names: list[str] = field(default_factory=list)
    """Camera override forwarded to the connector factory. Empty → env
    defaults."""

    record_video: bool = False
    enable_tracing: bool = False

    task_timeout_secs: float = 0.0
    """Per-trial wall-clock cap. ``0`` disables the watchdog."""

    output_dir: str = ""
    multi_codegen: bool = False
    checkpoints: str = "warn"
    """Checkpoint enforcement mode forwarded to the executor."""

    safety_limits: dict[str, int] = field(default_factory=dict)
    """``{perception, planning, sim_step}`` call caps applied per trial
    via :func:`gap.tools.guards.set_limits`."""

    policies: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Learned-policy registry (``policies:`` config block). Policies a
    workflow references are booted by the trial's PolicyManager."""

    policy_manager: dict[str, Any] = field(default_factory=dict)
    """PolicyManager tuning (``startup_timeout_s`` / ``evict_grace_s``)."""

    extra_env: dict[str, str] = field(default_factory=dict)
    """Environment variables applied in :func:`worker_setup` before any
    heavy import — VLM/LLM credentials (``OPENROUTER_API_KEY``,
    ``GOOGLE_CLOUD_PROJECT``, …) and sim knobs the bundles read."""

    device_slot_offset: int = 0
    """Global device-slot offset for this cell. Concurrent benchmark
    cells each run their own pool; without an offset every cell's
    worker 0 lands on the same GPU (all model stacks + EGL contexts pile
    onto device[0]). The launcher assigns each cell an increasing offset
    so workers spread as ``devices[(offset + worker_id) % len]``."""

    connector_factory: str = ""
    """Optional dotted ``module:attr`` path to a connector factory with
    the signature ``(suite_name, task_id, config) -> connector``. Empty →
    the default in-process :func:`gap.connector.sim` factory. Kept as a
    string so it crosses the spawn boundary."""


@dataclass
class TrialResult:
    """Result of a single trial execution."""

    trial_id: int = 0
    task_id: int = 0
    codegen_id: int = 1
    seed: int = 0
    task_completed: bool = False
    reward: float = 0.0
    completion_rate: float = 0.0
    """Partial-credit metric: 1.0 when the task completed, else the env's
    ``completion_rate()`` (delivered-item fraction for the packing suites)
    clamped to [0, 1], falling back to the reward for envs that don't
    expose one (single-goal suites report 0/1, making this equal to
    success)."""
    exit_code: int = -1
    execution_stdout: str = ""
    execution_stderr: str = ""
    video_path: str = ""
    duration_secs: float = 0.0
    control_steps: int = 0
    """Cumulative env-step count from the sim connector — actuator
    operations the policy issued. Real robot's actuator time is
    ``control_steps / control_freq``."""
    control_freq: float = 0.0
    """Sim env control frequency (Hz). 0 when latency wasn't reported."""
    sim_physics_wall_s: float = 0.0
    """Wall time spent inside ``env.step`` (MuJoCo physics + obs/reward)."""
    physical_execution_s: float = 0.0
    """Estimated real-robot wall time: ``control_steps / control_freq +
    (duration_secs - sim_physics_wall_s)``. ``compute_overhead`` is the
    non-physics share of trial wall time (LLM, planning, perception),
    which would also run on a real robot."""


@dataclass
class WorkItem:
    """A single unit of work: one trial of one (task, codegen)."""

    task_id: int
    trial_id: int
    codegen_id: int = 1
    workflow_dir: str = ""
    output_dir: str = ""
    """Per-trial override of ``WorkerSetupConfig.output_dir``."""
    suite_name: str = ""
    """Per-trial suite override (mixed-suite batches); empty falls back
    to ``WorkerSetupConfig.suite_name``."""


@dataclass
class WorkerState:
    """Non-serializable runtime state created inside the worker process."""

    worker_id: int
    config: WorkerSetupConfig
    connector_factory: Callable[[str, int, WorkerSetupConfig], Any]
    skill_registry: Any = None
    connector: Any = None
    tool_registry: Any = None
    tool_bundle_manager: Any = None
    """RPC tool-bundle manager (sam3/dino/vlm/geometry/curobo servers) booted
    ONCE per connector in ``_ensure_connector`` and reused across that
    ``(suite, task)``'s seeds. Booting + tearing it down per trial — while the
    registry persists across same-task seeds — orphaned the just-killed sam3
    subprocess into the reused registry, so every seed after the first on a
    worker failed perception with 'sam3 cannot write — stdin closed'."""
    last_init: tuple[str, int] | None = None
    """``(suite_name, task_id)`` of the live connector. When the next
    WorkItem matches we skip construction and just ``reset(seed)``."""
    hard_exit_on_timeout: bool = False
    """True in spawned pool workers: the watchdog pushes the failed
    result and ``os._exit(124)``s so a stuck MuJoCo/EGL call can't leak
    GPU state; the pool respawns a replacement. False in the in-process
    sequential path (tests, ``num_workers <= 1``)."""
    result_queue: Any = None
    """Result queue (spawned workers only) — the watchdog needs it to
    deliver the timeout result before hard-exiting."""

    tool_catalog: list = field(default_factory=list)
    """Persistent @tool catalog for this worker. ``discover_pending``
    drains the process-global pending queue exactly once per import, so
    without a per-worker catalog only the *first* connector's registry
    would see the bundle tools — a later connector (task switch on a
    persistent worker) would silently lose them."""

    trial_runner: Any = None
    """Persistent :class:`_TrialRunner` used by the watchdog path.
    MuJoCo EGL contexts are thread-affine: running each trial body on a
    fresh thread (the old watchdog design) made the *second* trial on a
    reused connector destroy/render the env from the wrong thread —
    SIGSEGV, killing the worker (the 2026-06-10 incident's worker
    deaths). One persistent thread keeps env affinity across trials."""


class _TrialTimeoutError(Exception):
    """Internal marker for the watchdog path (never escapes the worker)."""


class _TrialRunner:
    """Runs trial bodies on ONE persistent daemon thread.

    The watchdog cannot run trial bodies on a thread-per-trial basis:
    MuJoCo/EGL state is thread-affine, so a connector reused across
    trials (the task-reuse optimization) segfaults the worker the moment
    a second trial touches it from a new thread. A single long-lived
    runner thread keeps every trial of a worker — env construction,
    reset, stepping, rendering, destruction — on the same thread.

    After a watchdog timeout the wedged runner is abandoned (the hung
    body cannot be cancelled) and the caller installs a fresh runner;
    the connector is rebuilt anyway.
    """

    def __init__(self, name: str = "trial-runner") -> None:
        import queue as queue_mod

        self._queue: Any = queue_mod.Queue()
        self._thread = threading.Thread(
            target=self._loop, name=name, daemon=True,
        )
        self._thread.start()

    def _loop(self) -> None:
        while True:
            fn, done = self._queue.get()
            if fn is None:
                return
            try:
                fn()
            finally:
                done.set()

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def submit(self, fn: Callable[[], None]) -> threading.Event:
        """Queue *fn*; returns an Event set when it finished."""
        done = threading.Event()
        self._queue.put((fn, done))
        return done

    def close(self) -> None:
        """Ask the runner thread to exit (after queued work)."""
        self._queue.put((None, None))


# ---------------------------------------------------------------------------
# Connector factory
# ---------------------------------------------------------------------------


def default_connector_factory(
    suite_name: str, task_id: int, config: WorkerSetupConfig
) -> Any:
    """Build an in-process :class:`gap.connector.sim.SimConnector`.

    ``suite_name`` resolves through :mod:`gap.envs.registry`, so dev
    aliases (``libero_grocery_packing_object`` → ``libero_object_packing``)
    work transparently.
    """
    import gap.connector

    return gap.connector.sim(
        suite_name,
        task=task_id,
        cameras=list(config.camera_names) or None,
        record_video=config.record_video,
    )


def _load_factory(dotted: str) -> Callable[[str, int, WorkerSetupConfig], Any]:
    module_name, _, attr = dotted.partition(":")
    if not attr:
        raise ValueError(
            f"connector_factory {dotted!r} must be 'module.path:attr'"
        )
    return getattr(importlib.import_module(module_name), attr)


# ---------------------------------------------------------------------------
# Worker lifecycle
# ---------------------------------------------------------------------------


def worker_setup(
    worker_id: int,
    config: WorkerSetupConfig,
    *,
    connector_factory: Callable[[str, int, WorkerSetupConfig], Any] | None = None,
    hard_exit_on_timeout: bool = False,
    result_queue: Any = None,
) -> WorkerState:
    """Prepare a worker: env knobs + skill registry. Runs POST-fork.

    The connector itself is built lazily per ``(suite, task)`` inside
    :func:`run_trial_on_worker` — construction needs the task id and the
    task-reuse optimization wants to keep it across same-task trials.
    """
    logger.info("Worker %d: starting setup", worker_id)

    # 1. Environment knobs — must land before mujoco/robosuite import.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ["GAP_LIBERO_JOINT_MOTION_MODE"] = (
        "closed_loop" if config.policies else "teleport"
    )
    egl_raw = os.environ.get(EGL_DEVICES_ENV, "")
    devices = [d.strip() for d in egl_raw.split(",") if d.strip()]
    if devices:
        slot = (config.device_slot_offset + worker_id) % len(devices)
        device = devices[slot]
        if mp.current_process().name != "MainProcess":
            # Spawned worker, before any cuda import: pin the WHOLE stack
            # (torch models, curobo, EGL) to one physical GPU. The
            # robosuite EGL fork's binding_utils asserts that
            # MUJOCO_EGL_DEVICE_ID names a PHYSICAL id listed in
            # CUDA_VISIBLE_DEVICES (it maps physical -> EGL enumeration
            # internally), so both get the same physical id.
            os.environ["CUDA_VISIBLE_DEVICES"] = device
            os.environ["MUJOCO_EGL_DEVICE_ID"] = device
        else:
            os.environ["MUJOCO_EGL_DEVICE_ID"] = device
        logger.warning(
            "Worker %d (%s, pid=%d): device slot=%d -> physical GPU %s "
            "(offset=%d, CUDA_VISIBLE_DEVICES=%r)",
            worker_id, mp.current_process().name, os.getpid(), slot, device,
            config.device_slot_offset, os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
    for key, value in config.extra_env.items():
        os.environ[str(key)] = str(value)

    # 2. Skill registry (bundle discovery imports tools.py modules, which
    #    push @tool registrations onto the pending queue — drained into
    #    each connector's registry per trial).
    skill_registry = None
    if config.skills_path:
        from gap.skills import load_registry_set

        skill_registry = load_registry_set(
            config.skills_path.split(os.pathsep),
        )

    factory = connector_factory
    if factory is None and config.connector_factory:
        factory = _load_factory(config.connector_factory)
    if factory is None:
        factory = default_connector_factory

    logger.info("Worker %d: setup complete", worker_id)
    return WorkerState(
        worker_id=worker_id,
        config=config,
        connector_factory=factory,
        skill_registry=skill_registry,
        hard_exit_on_timeout=hard_exit_on_timeout,
        result_queue=result_queue,
    )


def _shutdown_bundles(state: WorkerState) -> None:
    """Tear down the worker's live RPC tool-bundle servers (sam3/dino/vlm/…)."""
    mgr = state.tool_bundle_manager
    if mgr is not None:
        try:
            mgr.shutdown_all()
        except Exception:
            logger.debug("tool_bundle_manager.shutdown_all failed", exc_info=True)
        state.tool_bundle_manager = None


def worker_cleanup(state: WorkerState | None) -> None:
    """Close the worker's connector. Best-effort, never raises.

    When the trials ran on the persistent runner thread (watchdog mode),
    the connector teardown is routed through that same thread — EGL
    destruction from a different thread can segfault just like use.
    """
    if state is None:
        return
    logger.info("Worker %d: cleaning up", state.worker_id)
    _shutdown_bundles(state)

    def _close() -> None:
        try:
            if state.connector is not None and hasattr(state.connector, "close"):
                state.connector.close()
        except Exception:
            logger.debug("worker_cleanup: connector.close failed", exc_info=True)

    runner = state.trial_runner
    try:
        if runner is not None and runner.alive:
            runner.submit(_close).wait(timeout=30)
            runner.close()
        else:
            _close()
    except Exception:
        logger.debug("worker_cleanup failed", exc_info=True)
    state.trial_runner = None
    state.connector = None
    state.tool_registry = None
    state.last_init = None


def _ensure_connector(state: WorkerState, suite_name: str, task_id: int,
                      workflow_dir: str | Path | None = None) -> Any:
    """Return a connector for ``(suite, task)``, reusing the live one.

    Task-reuse optimization: when the previous trial on this worker
    already built the same ``(suite, task)`` env we keep it and let the
    caller ``reset(seed)`` — skipping the scene re-load. The connector's RPC
    tool bundles (sam3/dino/vlm/…) are booted ONCE here and reused across that
    task's seeds; a task switch tears them down together with the connector.
    """
    init_key = (suite_name, task_id)
    if state.connector is not None and state.last_init == init_key:
        return state.connector
    # Task switch (or first build): tear down the previous connector AND its
    # tool bundles together — the bundles registered into that connector's
    # registry, which is discarded below.
    _shutdown_bundles(state)
    if state.connector is not None:
        try:
            state.connector.close()
        except Exception:
            logger.debug("connector.close failed", exc_info=True)
        state.connector = None
        state.tool_registry = None
    conn = state.connector_factory(suite_name, task_id, state.config)
    state.connector = conn
    state.last_init = init_key

    # Fresh registry per connector: robot.*/sim.* tools + the full bundle
    # @tool catalog. ``catalog=state.tool_catalog`` keeps the drained
    # @tool registrations on the WorkerState so every later registry on
    # this persistent worker (task switch -> new connector) re-registers
    # them too — the pending queue itself only yields them once.
    reg = getattr(conn, "tool_registry", None)
    if reg is None:
        from gap_core.tools import default_tool_registry

        reg = default_tool_registry()
    if hasattr(reg, "discover_pending"):
        try:
            reg.discover_pending(catalog=state.tool_catalog)
        except TypeError:  # registry without catalog support
            reg.discover_pending()
    state.tool_registry = reg
    # Boot the workflow's RPC tool bundles ONCE into this fresh registry; every
    # seed of this task reuses the same live sam3/dino/vlm servers. (Booting +
    # shutting them down per trial orphaned dead subprocesses into the reused
    # registry — see the WorkerState.tool_bundle_manager note.)
    if workflow_dir is not None and state.skill_registry is not None:
        try:
            from gap.runtime.tool_bundle_boot import boot_tool_bundles

            state.tool_bundle_manager = boot_tool_bundles(
                workflow_dir, state.skill_registry, state.tool_registry,
            )
        except Exception as exc:
            logger.error("tool bundle boot failed (task %s env): %s", task_id, exc)
            state.tool_bundle_manager = None
    return conn


# ---------------------------------------------------------------------------
# Trial execution
# ---------------------------------------------------------------------------


def _trial_dir(
    output_dir: str, task_id: int, trial_id: int,
    codegen_id: int = 1, multi_codegen: bool = False,
) -> Path:
    """Build the trial output directory path."""
    task_dir = Path(output_dir) / f"task_{task_id:02d}"
    if multi_codegen:
        task_dir = task_dir / f"codegen_{codegen_id:02d}"
    return task_dir / f"trial_{trial_id:02d}"


def _copy_codegen_to_trial(item: WorkItem, trial_dir: Path) -> None:
    """Copy generated workflow artifacts into the per-trial directory."""
    import shutil

    if not item.workflow_dir:
        return
    wf_dir = Path(item.workflow_dir)
    codegen_dest = trial_dir / "codegen"
    codegen_dest.mkdir(parents=True, exist_ok=True)
    for name in ("workflow.json", "multi_agent_meta.json"):
        src = wf_dir / name
        if src.exists():
            shutil.copy2(src, codegen_dest / name)
    for sub in ("scripts", "checkpoints"):
        src_dir = wf_dir / sub
        if src_dir.exists():
            shutil.copytree(src_dir, codegen_dest / sub, dirs_exist_ok=True)


def _execute_trial(
    state: WorkerState,
    item: WorkItem,
    trial_result: TrialResult,
    trial_dir: Path,
) -> None:
    """Core trial: connector → reset(seed) → workflow → success check → video."""
    from gap_core.tools import guards

    from gap.runtime.executor import WorkflowExecutor

    config = state.config
    suite_name = item.suite_name or config.suite_name

    guards.reset_counters()
    limits = config.safety_limits or {}
    guards.set_limits(
        perception=limits.get("perception"),
        planning=limits.get("planning"),
        sim_step=limits.get("sim_step"),
    )

    conn = _ensure_connector(state, suite_name, item.task_id, item.workflow_dir)
    conn.reset(seed=item.trial_id)

    # Boot every policy the workflow references before execution. Graph is
    # the source of truth — each policy skill owns its preset (auto-resolved
    # from PRESETS unless overridden in `policies:`); a missing/unstartable
    # server fails the trial here with a clear error.
    from gap.runtime.policy_boot import boot_policies

    policy_manager, policy_executor = boot_policies(
        item.workflow_dir,
        state.skill_registry,
        config_policies=config.policies,
        startup_timeout_s=float(
            config.policy_manager.get("startup_timeout_s", 120.0)
        ),
        evict_grace_s=float(
            config.policy_manager.get("evict_grace_s", 10.0)
        ),
    )

    # RPC tool bundles (sam3, grounding-dino, vlm, geometry, curobo, …) are
    # booted ONCE per connector in ``_ensure_connector`` and reused across this
    # task's seeds — their tools are already live in ``state.tool_registry``.
    # (Booting + tearing them down per trial orphaned the just-killed sam3
    # subprocess into the reused registry: "sam3 cannot write — stdin closed",
    # silently failing every seed after the first on a worker.)

    executor = None
    try:
        executor = WorkflowExecutor(
            item.workflow_dir,
            tool_registry=state.tool_registry,
            skill_registry=state.skill_registry,
            policy_executor=policy_executor,
            observation_poll_fn=getattr(conn, "get_observation", None),
            world_snapshot_fn=getattr(conn, "world_snapshot", None),
            trace_dir=trial_dir / "trace",
            checkpoints=config.checkpoints,
        )
        try:
            executor.execute()
            trial_result.exit_code = 0
        except Exception as exc:
            trial_result.exit_code = 1
            trial_result.execution_stderr += f"\n{exc}"
            logger.error(
                "Trial %d (task %d) workflow failed: %s",
                item.trial_id, item.task_id, exc,
            )
    finally:
        if executor is not None:
            try:
                executor.close()
            except Exception:
                pass
        # NOTE: tool bundles are NOT shut down here — they live on the
        # WorkerState (per connector) and are torn down on task switch /
        # worker cleanup, so this task's remaining seeds reuse them.
        if policy_executor is not None:
            policy_executor.close()
        if policy_manager is not None:
            policy_manager.shutdown_all()

    completed, reward = conn.check_success()
    trial_result.task_completed = bool(completed)
    trial_result.reward = float(reward)
    # Partial credit must come from the env's completion_rate — the VAB
    # suites' reward is BINARY (1.0 only at full completion), so clamping
    # the reward records a 5/6 pack as 0.0 and the aggregate can't tell
    # "delivered nothing" from "delivered almost everything".
    partial = float(reward)
    cr_fn: Any = getattr(getattr(conn, "env", None), "completion_rate", None)
    if callable(cr_fn):
        try:
            partial = float(cr_fn())
        except Exception:
            logger.debug("env completion_rate read failed", exc_info=True)
    trial_result.completion_rate = (
        1.0 if completed else min(max(partial, 0.0), 1.0)
    )

    latency_fn = getattr(conn, "get_latency_info", None)
    if latency_fn is not None:
        try:
            lat = dict(latency_fn())
        except Exception:
            lat = {}
        trial_result.control_steps = int(lat.get("control_steps", 0))
        trial_result.control_freq = float(lat.get("control_freq", 0.0))
        trial_result.sim_physics_wall_s = float(lat.get("sim_physics_wall_s", 0.0))

    if config.record_video and hasattr(conn, "save_video"):
        video_path = str(trial_dir / "video.mp4")
        resp = conn.save_video(video_path, clear=True)
        if isinstance(resp, dict) and resp.get("success"):
            trial_result.video_path = resp.get("file_path", video_path)


def run_trial_on_worker(state: WorkerState, item: WorkItem) -> TrialResult:
    """Execute one trial; honors the per-trial watchdog. Never raises."""
    start_time = time.monotonic()
    config = state.config

    output_dir = item.output_dir or config.output_dir
    trial_dir = _trial_dir(
        output_dir, item.task_id, item.trial_id,
        codegen_id=item.codegen_id, multi_codegen=config.multi_codegen,
    )
    trial_dir.mkdir(parents=True, exist_ok=True)

    trial_result = TrialResult(
        trial_id=item.trial_id,
        task_id=item.task_id,
        codegen_id=item.codegen_id,
        seed=item.trial_id,
    )

    if not item.workflow_dir:
        trial_result.execution_stderr = (
            f"WorkItem for task {item.task_id} trial {item.trial_id} "
            f"has no workflow_dir"
        )
        trial_result.duration_secs = time.monotonic() - start_time
        _write_result_json(trial_result, trial_dir)
        return trial_result

    try:
        _copy_codegen_to_trial(item, trial_dir)
    except Exception:
        logger.debug("Failed to copy codegen artifacts", exc_info=True)

    timeout_s = float(config.task_timeout_secs or 0.0)
    if timeout_s > 0:
        _run_with_watchdog(state, item, trial_result, trial_dir, timeout_s)
    else:
        try:
            _execute_trial(state, item, trial_result, trial_dir)
        except Exception as exc:
            trial_result.exit_code = 1
            trial_result.execution_stderr += (
                f"\n{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            )

    trial_result.duration_secs = time.monotonic() - start_time
    # Physical execution: actuator time + non-physics wall (LLM /
    # planning / perception — the share of the trial that would also
    # run on a real robot).
    if trial_result.control_freq > 0:
        control_s = trial_result.control_steps / trial_result.control_freq
        compute_s = max(
            0.0, trial_result.duration_secs - trial_result.sim_physics_wall_s
        )
        trial_result.physical_execution_s = control_s + compute_s
    _write_result_json(trial_result, trial_dir)
    return trial_result


def _run_with_watchdog(
    state: WorkerState,
    item: WorkItem,
    trial_result: TrialResult,
    trial_dir: Path,
    timeout_s: float,
) -> None:
    """Run the trial body on the worker's persistent trial-runner thread
    with a hard wall-clock cap.

    The body runs on ONE long-lived thread per worker (see
    :class:`_TrialRunner`) — never thread-per-trial, which broke
    MuJoCo/EGL thread affinity for connectors reused across trials and
    segfaulted the worker on its second trial (the incident's silent
    worker deaths).

    On expiry the trial is recorded as failed (``exit_code=124``,
    ``task_completed=False``). The hung body cannot be cancelled
    in-process (a stuck MuJoCo/EGL call never observes a token), so:

    * spawned worker (``hard_exit_on_timeout``): the result is pushed
      onto the result queue, ``result.json`` is written, and the process
      hard-exits — the pool respawns a replacement worker. This is the
      ported source behavior (SIGKILL the sim, fail the trial, no retry).
    * sequential / test path: the wedged runner thread is abandoned (a
      fresh one is created next trial) and the connector reference
      dropped so the next trial rebuilds a fresh env.
    """
    body_error: list[BaseException] = []

    def _body() -> None:
        try:
            _execute_trial(state, item, trial_result, trial_dir)
        except BaseException as exc:  # noqa: BLE001 — recorded below
            body_error.append(exc)

    runner = state.trial_runner
    if runner is None or not runner.alive:
        runner = _TrialRunner(name=f"trial-runner-w{state.worker_id}")
        state.trial_runner = runner
    done = runner.submit(_body)
    finished = done.wait(timeout=timeout_s)

    if finished:
        if body_error:
            exc = body_error[0]
            trial_result.exit_code = 1
            trial_result.execution_stderr += f"\n{type(exc).__name__}: {exc}"
        return

    # --- watchdog fired ---
    logger.warning(
        "Worker %d: trial timeout (task=%d trial=%d) after %.0fs",
        state.worker_id, item.task_id, item.trial_id, timeout_s,
    )
    trial_result.exit_code = 124
    trial_result.task_completed = False
    trial_result.reward = 0.0
    trial_result.completion_rate = 0.0
    trial_result.execution_stderr += (
        f"\ntrial wall-clock cap ({timeout_s:.0f}s) exceeded"
    )

    if state.hard_exit_on_timeout:
        trial_result.duration_secs = timeout_s
        _write_result_json(trial_result, trial_dir)
        if state.result_queue is not None:
            try:
                state.result_queue.put(
                    ("result", os.getpid(), _item_key(item), trial_result)
                )
                state.result_queue.close()
                state.result_queue.join_thread()
            except Exception:
                logger.debug("watchdog: result delivery failed", exc_info=True)
        os._exit(124)

    # Soft path: abandon the wedged runner thread + poison the connector
    # so the next trial gets a fresh runner and builds a fresh env on it
    # (the old one may be wedged mid-step).
    state.trial_runner = None
    state.connector = None
    state.tool_registry = None
    state.last_init = None


def _write_result_json(result: TrialResult, trial_dir: Path) -> None:
    """Write per-trial result.json so the outcome is visible immediately."""
    try:
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "result.json").write_text(json.dumps({
            "task_id": result.task_id,
            "trial_id": result.trial_id,
            "codegen_id": result.codegen_id,
            "seed": result.seed,
            "task_completed": result.task_completed,
            "reward": result.reward,
            "completion_rate": result.completion_rate,
            "exit_code": result.exit_code,
            "duration_secs": result.duration_secs,
            "video_path": result.video_path,
            "execution_stdout": result.execution_stdout,
            "execution_stderr": result.execution_stderr,
        }, indent=2))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Worker loop + parallel dispatch
# ---------------------------------------------------------------------------
#
# Pool protocol (parent <-> spawned workers), all over ``result_queue``:
#
#   ("setup_failed", pid, error_str)        worker_setup raised; the
#                                           process exits right after.
#   ("ready", pid)                          worker_setup finished; the
#                                           worker now blocks on the
#                                           work queue.
#   ("started", pid, item_key)              worker took an item off the
#                                           work queue.
#   ("result", pid, item_key, TrialResult)  trial finished (also sent by
#                                           the watchdog hard-exit path).
#
# ``item_key`` identifies a WorkItem; the parent uses started/result
# pairs to know exactly which trial a dead worker was holding so it can
# re-enqueue (or fail) it instead of losing it.


def _item_key(item: WorkItem) -> tuple:
    """Stable identity of a WorkItem for pool accounting."""
    return (
        item.suite_name, item.output_dir,
        item.task_id, item.codegen_id, item.trial_id,
    )


def _worker_loop(
    worker_id: int,
    work_queue: Any,
    result_queue: Any,
    setup_config: WorkerSetupConfig,
) -> None:
    """Process target: set up the worker, then pull items until poisoned.

    Workers persist across trials: the loop blocks on ``work_queue`` and
    only exits on a poison pill (``None``), a crash, or the watchdog's
    ``os._exit(124)``. Per-process globals (the @tool pending queue,
    ``gap.tools.guards`` counters/limits) are private to this spawned
    process — concurrent pools in the parent cannot race on them.
    """
    pid = os.getpid()
    state = None
    try:
        state = worker_setup(
            worker_id, setup_config,
            hard_exit_on_timeout=True,
            result_queue=result_queue,
        )
    except Exception as exc:
        logger.error(
            "Worker %d setup failed: %s\n%s",
            worker_id, exc, traceback.format_exc(),
        )
        result_queue.put((
            "setup_failed", pid,
            f"Worker {worker_id} setup failed: {exc}\n"
            f"{traceback.format_exc()}",
        ))
        return

    result_queue.put(("ready", pid))
    try:
        while True:
            item = work_queue.get()
            if item is None:  # poison pill
                break
            key = _item_key(item)
            result_queue.put(("started", pid, key))
            try:
                result = run_trial_on_worker(state, item)
            except Exception as exc:  # run_trial_on_worker should not raise
                result = TrialResult(
                    task_id=item.task_id,
                    trial_id=item.trial_id,
                    codegen_id=item.codegen_id,
                    seed=item.trial_id,
                    exit_code=1,
                    execution_stderr=(
                        f"Worker {worker_id} error: {exc}\n"
                        f"{traceback.format_exc()}"
                    ),
                )
            result_queue.put(("result", pid, key, result))
    finally:
        worker_cleanup(state)


def run_parallel_trials(
    work_items: list[WorkItem],
    num_workers: int,
    setup_config: WorkerSetupConfig,
    on_complete: Callable[[TrialResult], Any] | None = None,
    connector_factory: Callable[[str, int, WorkerSetupConfig], Any] | None = None,
    *,
    max_respawns_per_slot: int = _DEFAULT_MAX_RESPAWNS_PER_SLOT,
    max_item_retries: int = _DEFAULT_MAX_ITEM_RETRIES,
) -> list[TrialResult]:
    """Run trials sequentially (``num_workers <= 1``) or via a spawn pool.

    The pool guarantees termination with one TrialResult per work item:
    trials whose worker died repeatedly come back failed
    (``exit_code=137``) rather than hanging the pool, and respawns are
    bounded per worker slot (see the module docstring).

    Args:
        work_items: Items to execute (sorted by task_id for env reuse).
        num_workers: Worker processes; ``<= 1`` runs in-process.
        setup_config: Picklable worker config.
        on_complete: Optional callback per finished TrialResult.
        connector_factory: In-process factory override — only honored on
            the sequential path (tests / stub connectors). Spawned
            workers use ``setup_config.connector_factory`` (dotted path).
        max_respawns_per_slot: Replacement processes allowed per worker
            slot before the slot is retired.
        max_item_retries: Re-runs allowed for an item whose worker died
            mid-trial before it is recorded as failed.
    """
    if not work_items:
        return []

    # In-process sequential execution is for TESTS ONLY (explicit factory
    # override or GAP_PARALLEL_INPROC=1, used by tests that monkeypatch
    # module state a spawned child can't see). Production cells —
    # including num_workers=1, the dev-parity acceptance topology —
    # always spawn: concurrent cells in one process would share one GPU,
    # one GIL, and one set of process-global guard counters (measured:
    # 10 in-process cells -> every trial 900s-timeout or
    # GuardLimitExceeded from cross-cell counter pollution).
    if num_workers <= 1 and (
        connector_factory is not None
        or os.environ.get("GAP_PARALLEL_INPROC") == "1"
    ):
        state = worker_setup(
            0, setup_config, connector_factory=connector_factory,
        )
        try:
            results = []
            sorted_items = sorted(work_items, key=lambda wi: wi.task_id)
            for i, item in enumerate(sorted_items):
                logger.info(
                    "Trial %d/%d (task=%d, trial=%d)",
                    i + 1, len(work_items), item.task_id, item.trial_id,
                )
                result = run_trial_on_worker(state, item)
                if on_complete:
                    on_complete(result)
                results.append(result)
            return results
        finally:
            worker_cleanup(state)

    return _run_pool(
        work_items, max(1, num_workers), setup_config, on_complete,
        max_respawns_per_slot=max_respawns_per_slot,
        max_item_retries=max_item_retries,
    )


def _run_pool(
    work_items: list[WorkItem],
    num_workers: int,
    setup_config: WorkerSetupConfig,
    on_complete: Callable[[TrialResult], Any] | None,
    *,
    max_respawns_per_slot: int,
    max_item_retries: int,
) -> list[TrialResult]:
    """Spawn-pool execution with item-level crash accounting."""
    import queue as queue_mod

    # Parallel execution with spawn context (CUDA/MuJoCo safe).
    ctx = mp.get_context("spawn")
    work_queue = ctx.Queue()
    result_queue = ctx.Queue()

    # Sort by task_id so each worker sees consecutive same-task trials —
    # maximizes the connector-reuse win.
    sorted_items = sorted(work_items, key=lambda wi: wi.task_id)
    expected: dict[tuple, WorkItem] = {}
    for item in sorted_items:
        key = _item_key(item)
        if key in expected:
            raise ValueError(
                f"duplicate WorkItem {key!r}: pool accounting requires "
                "unique (suite, output_dir, task, codegen, trial) tuples"
            )
        expected[key] = item
    total = len(expected)

    # --- Pool state ------------------------------------------------------
    procs: dict[int, tuple[int, Any]] = {}   # pid -> (slot, Process)
    ready_pids: set[int] = set()             # workers past worker_setup
    in_flight: dict[int, tuple] = {}         # pid -> item key
    queued: dict[tuple, float] = {}          # keys believed in work_queue
    stale_seen: dict[tuple, float] = {}      # queued-while-idle first-seen
    completed: dict[tuple, TrialResult] = {}
    crashes: dict[tuple, int] = {}           # worker deaths per item
    respawns = [0] * num_workers             # replacements per slot
    setup_errors: list[str] = []
    results: list[TrialResult] = []

    now = time.monotonic()
    for item in sorted_items:
        work_queue.put(item)
        queued[_item_key(item)] = now

    def _spawn(slot: int) -> None:
        p = ctx.Process(
            target=_worker_loop,
            args=(slot, work_queue, result_queue, setup_config),
        )
        p.start()
        procs[p.pid] = (slot, p)

    for slot in range(num_workers):
        _spawn(slot)
        if slot + 1 < num_workers:
            time.sleep(_STAGGER_SECS)

    # --- Accounting helpers ----------------------------------------------

    def _finish(key: tuple, result: TrialResult) -> None:
        completed[key] = result
        results.append(result)
        if on_complete:
            on_complete(result)
        status = "PASS" if result.task_completed else "FAIL"
        logger.info(
            "Completed %d/%d: task=%d trial=%d %s reward=%.3f (%.1fs)",
            len(results), total, result.task_id, result.trial_id,
            status, result.reward, result.duration_secs,
        )

    def _fail_item(key: tuple, reason: str) -> None:
        """Record a synthesized failure for an item the pool cannot run."""
        item = expected[key]
        result = TrialResult(
            trial_id=item.trial_id,
            task_id=item.task_id,
            codegen_id=item.codegen_id,
            seed=item.trial_id,
            exit_code=_CRASHED_EXIT_CODE,
            execution_stderr=reason,
        )
        trial_dir = _trial_dir(
            item.output_dir or setup_config.output_dir,
            item.task_id, item.trial_id,
            codegen_id=item.codegen_id,
            multi_codegen=setup_config.multi_codegen,
        )
        _write_result_json(result, trial_dir)
        logger.error(
            "Failing trial task=%d trial=%d: %s",
            item.task_id, item.trial_id, reason,
        )
        _finish(key, result)

    def _crash_item(key: tuple, detail: str) -> None:
        """A worker died holding *key*: re-enqueue or fail it."""
        if key in completed:
            return
        crashes[key] = crashes.get(key, 0) + 1
        item = expected[key]
        if crashes[key] > max_item_retries:
            extra = f" Last worker error: {setup_errors[-1]}" if setup_errors else ""
            _fail_item(
                key,
                f"worker process died running this trial "
                f"{crashes[key]} time(s) (retry budget {max_item_retries} "
                f"exhausted): {detail}.{extra}",
            )
        else:
            logger.warning(
                "Worker died mid-trial (task=%d trial=%d): %s — "
                "re-enqueueing (attempt %d/%d)",
                item.task_id, item.trial_id, detail,
                crashes[key] + 1, max_item_retries + 1,
            )
            work_queue.put(item)
            queued[key] = time.monotonic()
            stale_seen.pop(key, None)

    def _handle(msg: Any) -> None:
        kind = msg[0]
        if kind == "ready":
            ready_pids.add(msg[1])
        elif kind == "started":
            _, pid, key = msg
            queued.pop(key, None)
            stale_seen.pop(key, None)
            if pid in procs:
                in_flight[pid] = key
            elif key not in completed:
                # The worker was already reaped; its 'started' flushed
                # late. The item is on no live worker — recover it.
                _crash_item(key, "worker exited right after taking the item")
        elif kind == "result":
            _, pid, key, result = msg
            in_flight.pop(pid, None)
            if key in completed:
                logger.debug("Duplicate result for %r ignored", key)
            else:
                _finish(key, result)
        elif kind == "setup_failed":
            _, pid, err = msg
            setup_errors.append(err.strip().splitlines()[-1] if err else err)
            logger.error("Worker setup failed (pid=%s): %s", pid, err)
        else:  # pragma: no cover - future-proofing
            logger.debug("Unknown pool message %r", msg)

    def _maintain() -> None:
        """Reap dead workers, recover their items, respawn (bounded)."""
        # 1. Reap dead processes; recover items they held.
        for pid in list(procs):
            slot, p = procs[pid]
            if p.is_alive():
                continue
            del procs[pid]
            ready_pids.discard(pid)
            key = in_flight.pop(pid, None)
            if key is not None:
                _crash_item(
                    key, f"worker slot {slot} (pid {pid}) exited with "
                    f"code {p.exitcode}",
                )
            elif p.exitcode == 124:
                # Watchdog hard-exit by design; its result was already
                # delivered. A replacement is spawned below if needed.
                logger.info(
                    "Worker slot %d (pid %d) hard-exited after a trial "
                    "timeout (by design)", slot, pid,
                )
            else:
                logger.warning(
                    "Worker slot %d (pid %d) exited idle with code %s",
                    slot, pid, p.exitcode,
                )

        # 2. Invariant: every unfinished item is queued or on a live
        #    worker. Anything else was lost by a dying worker whose
        #    messages never flushed.
        live_keys = set(in_flight.values())
        for key in expected:
            if (
                key not in completed
                and key not in queued
                and key not in live_keys
            ):
                _crash_item(key, "trial lost by a dying worker")

        # 3. Items stuck in the queue while *ready* live workers sit
        #    idle were dequeued by a worker that died before announcing
        #    'started'. Gated on readiness so a replacement still
        #    loading its models doesn't count as an idle consumer, and
        #    only flagged after persisting across two maintenance passes.
        idle_pids = [
            pid for pid in procs
            if pid in ready_pids and pid not in in_flight
        ]
        if idle_pids and queued:
            t = time.monotonic()
            for key in list(queued):
                first = stale_seen.setdefault(key, t)
                if t - first > 2 * _POLL_SECS:
                    queued.pop(key, None)
                    stale_seen.pop(key, None)
                    _crash_item(
                        key, "item vanished from the queue (worker died "
                        "before reporting start)",
                    )
        else:
            stale_seen.clear()

        # 4. Respawn replacements — only when queued work exceeds the
        #    live pool, and only on slots with respawn budget left.
        target = min(num_workers, len(queued) + len(in_flight))
        deficit = target - len(procs)
        if deficit > 0:
            slots_in_use = {slot for slot, _ in procs.values()}
            free = [
                s for s in range(num_workers)
                if s not in slots_in_use and respawns[s] < max_respawns_per_slot
            ]
            for slot in free[:deficit]:
                respawns[slot] += 1
                logger.warning(
                    "Respawning worker slot %d (attempt %d/%d) — "
                    "%d trial(s) queued, %d in flight",
                    slot, respawns[slot], max_respawns_per_slot,
                    len(queued), len(in_flight),
                )
                _spawn(slot)
            retired = [
                s for s in range(num_workers)
                if s not in slots_in_use and respawns[s] >= max_respawns_per_slot
            ]
            if len(free) < deficit and retired:
                logger.error(
                    "Worker slot(s) %s retired (respawn cap %d reached); "
                    "pool capacity reduced to %d",
                    retired, max_respawns_per_slot, len(procs),
                )

        # 5. Dead end: no live workers and no respawn budget — conclude
        #    by failing everything unfinished instead of spinning.
        if not procs and len(completed) < total:
            detail = (
                f" Last worker error: {setup_errors[-1]}"
                if setup_errors else ""
            )
            logger.error(
                "Worker pool exhausted: all %d slot(s) hit the respawn "
                "cap (%d). Failing %d unfinished trial(s).%s",
                num_workers, max_respawns_per_slot,
                total - len(completed), detail,
            )
            for key in [k for k in expected if k not in completed]:
                _fail_item(
                    key,
                    f"worker pool exhausted (respawn cap "
                    f"{max_respawns_per_slot}/slot reached).{detail}",
                )

    # --- Result loop -------------------------------------------------------
    last_maintenance = time.monotonic()
    while len(completed) < total:
        try:
            msg = result_queue.get(timeout=_POLL_SECS)
        except queue_mod.Empty:
            msg = None
        if msg is not None:
            _handle(msg)
            if time.monotonic() - last_maintenance < _POLL_SECS:
                continue
        # Drain anything already delivered before judging liveness, so a
        # dead worker's flushed messages are never double-counted.
        while True:
            try:
                _handle(result_queue.get_nowait())
            except queue_mod.Empty:
                break
        if len(completed) >= total:
            break
        _maintain()
        last_maintenance = time.monotonic()

    # --- Shutdown: poison-pill the live workers and join them ---------------
    for _ in procs:
        work_queue.put(None)
    for _, p in procs.values():
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join()

    return results
