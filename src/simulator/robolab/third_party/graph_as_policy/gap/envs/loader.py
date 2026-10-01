"""LIBERO task loading utilities.

Ported from HyRL's hyrl/integrations/libero.py — loads a LIBERO task by
suite name and task ID — and extended with the gap repo's source-of-truth
resolution across the vendored LIBERO forks:

1. **vab** (third_party/Variational-Automation-Benchmark): suites whose
   name matches a ``tasks/<suite_name>/`` directory in the vab repo load
   through the self-contained YAML task format (``libero.libero.vab``).
   Each YAML bakes the arena, objects, 50 init-state variations, and a
   success predicate; ``task_id`` indexes the suite dir's sorted ``*.yaml``
   files, and the per-trial seed selects one of the baked inits
   (``(seed - 1) % n_inits`` — same convention as the classic path's
   pruned-init indexing).
2. **LIBERO-PRO** (third_party/LIBERO-PRO): everything else goes through
   the classic benchmark registry + BDDL + pruned-init files (the source
   behavior).

Both forks ship a top-level python package named ``libero`` — the vab fork
is the pip-installed one (``gap[libero]`` wires it via [tool.uv.sources]);
LIBERO-PRO is vendored-only. :func:`_activate_libero_fork` swaps which one
``import libero`` resolves to by stashing/restoring ``sys.modules`` entries
and toggling a ``sys.path`` entry, so suites from both forks can be
constructed in the same process (live envs keep strong refs to their own
module objects, so an env from the inactive fork keeps working).
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gap import env_config

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

# Root of the gap repo (gap/envs/ -> ../..). Holds the vendored LIBERO
# forks under third_party/. Overridable per-fork via GAP_VAB_ROOT /
# GAP_LIBERO_PRO_ROOT for installs where the package doesn't live inside
# the repo checkout.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_VAB_ROOT = Path(
    env_config.vab_root()
    or (_REPO_ROOT / "third_party" / "Variational-Automation-Benchmark")
)
_LIBERO_PRO_ROOT = Path(
    env_config.libero_pro_root() or (_REPO_ROOT / "third_party" / "LIBERO-PRO")
)

# Installed-package root of the classic fork (contains bddl_files/,
# init_files/, benchmark/, ...).
_LIBERO_PRO_PKG_ROOT = _LIBERO_PRO_ROOT / "libero" / "libero"

# sys.path entries exposing each fork's top-level ``libero`` package.
_FORK_PATHS = {
    "vab": _VAB_ROOT / "libero",
    "pro": _LIBERO_PRO_ROOT / "libero",
}

_fork_module_stash: dict[str, dict[str, Any]] = {}
_active_fork: str | None = None


def _classify_libero_module(mod: Any) -> str:
    """Classify an imported ``libero`` module as the vab or classic fork."""
    pkg_dir = Path(mod.__file__).resolve().parent
    return "vab" if (pkg_dir / "vab").is_dir() else "pro"


def _activate_libero_fork(fork: str) -> None:
    """Make ``import libero`` resolve to the requested vendored fork.

    The two forks both install/ship a top-level ``libero`` package, so only
    one can occupy ``sys.modules`` at a time. Switching stashes the current
    fork's ``libero*`` modules (keeping any live envs working — their code
    holds strong refs to its own module objects) and restores the other
    fork's previously-imported modules. The vab fork is the pip-installed
    ``libero``, so its path is always on ``sys.path``; LIBERO-PRO wins by
    an explicit entry inserted at the front.
    """
    global _active_fork

    if _active_fork is None and "libero" in sys.modules:
        # Someone imported libero before the loader ran (e.g. directly via
        # ``from libero.vab import load_task``); adopt it.
        _active_fork = _classify_libero_module(sys.modules["libero"])

    if _active_fork != fork:
        if _active_fork is not None:
            _fork_module_stash[_active_fork] = {
                k: m
                for k, m in sys.modules.items()
                if k == "libero" or k.startswith("libero.")
            }
        for k in [
            k for k in sys.modules if k == "libero" or k.startswith("libero.")
        ]:
            del sys.modules[k]
        sys.modules.update(_fork_module_stash.pop(fork, {}))
        _active_fork = fork

    pro_path = str(_FORK_PATHS["pro"])
    while pro_path in sys.path:
        sys.path.remove(pro_path)
    if fork == "pro":
        if not _FORK_PATHS["pro"].is_dir():
            raise ModuleNotFoundError(
                f"LIBERO-PRO not found at {_LIBERO_PRO_ROOT}. Initialize the "
                "third_party/LIBERO-PRO submodule or set GAP_LIBERO_PRO_ROOT."
            )
        sys.path.insert(0, pro_path)


_LIBERO_PATHS_CONFIGURED = False


def _ensure_libero_paths() -> None:
    """Configure libero's path resolution to use the vendored classic fork.

    LIBERO's ``set_libero_path`` rewrites ``~/.libero/config.yaml`` by
    opening it in ``"w"`` mode (truncate-then-write). With many worker
    subprocesses calling this concurrently on every ``load_libero_task``,
    readers in other processes hit the empty truncate window and
    ``yaml.load("") → None`` raises ``TypeError``.

    Run once per process and skip thereafter — the config target is fixed
    for the process's lifetime. Only the classic (LIBERO-PRO) path consumes
    the config; the vab fork resolves assets package-relative.
    """
    global _LIBERO_PATHS_CONFIGURED
    if _LIBERO_PATHS_CONFIGURED:
        return
    if not _LIBERO_PRO_PKG_ROOT.is_dir():
        _LIBERO_PATHS_CONFIGURED = True
        return  # classic fork not vendored — fallback paths below handle it

    try:
        from libero.utils import set_libero_path  # type: ignore[import-not-found]
        set_libero_path(str(_LIBERO_PRO_PKG_ROOT))
    except Exception:
        pass  # non-fatal — fallback paths below will handle it
    _LIBERO_PATHS_CONFIGURED = True


@dataclass
class LiberoHandle:
    """Wrapper around a LIBERO OffScreenRenderEnv."""

    env: Any
    suite_name: str
    task_id: int
    task_language: str
    init_states: Any

    def reset(self, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        self.env.seed(seed)
        if self.init_states is not None and len(self.init_states) > 0:
            if seed is not None:
                state_idx = (seed - 1) % len(self.init_states)
            else:
                state_idx = 0
            self.env.set_init_state(self.init_states[state_idx])
        obs = self.env.reset()
        return obs, {}

    def step(self, action: list[float]) -> tuple[Any, float, bool, dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        return obs, float(reward), bool(done), info


@dataclass
class VABHandle(LiberoHandle):
    """LiberoHandle variant for vab tasks.

    ``init_states`` holds the task YAML's baked init list (object-id →
    7-pose dicts, one per trial seed) rather than flattened mujoco states;
    reset maps the seed onto an init index with the same ``(seed - 1) %
    n_inits`` convention as the classic path and hands it to
    ``VABEnv.reset(init_index=...)``, which applies the poses after the
    robosuite hard reset.
    """

    task: Any = None  # libero.vab schema.Task

    def reset(self, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        n_inits = len(self.init_states) if self.init_states is not None else 0
        if n_inits > 0 and seed is not None:
            init_index = (seed - 1) % n_inits
        elif self.task is not None:
            init_index = self.task.default_init_index
        else:
            init_index = 0
        obs = self.env.reset(init_index=init_index)
        return obs, {}


class VABControlEnv:
    """ControlEnv-compatible adapter around a raw-obs :class:`VABEnv`.

    The classic path's ``OffScreenRenderEnv`` *wraps* the robosuite env as
    ``.env`` and forwards ``sim`` / ``robots`` / ``check_success``;
    ``FrankaLiberoEnv`` codes against that two-layer surface (e.g.
    ``handle.env.env._action_dim`` in the controller swap). VABEnv *is* the
    robosuite env, so this shim restores the wrapper layer.
    """

    def __init__(self, env: Any) -> None:
        self.env = env

    @property
    def sim(self) -> Any:
        return self.env.sim

    @property
    def robots(self) -> Any:
        return self.env.robots

    @property
    def obj_body_id(self) -> dict[str, int]:
        return self.env.obj_body_id

    @property
    def language_instruction(self) -> str:
        return self.env.task.language

    def step(self, action: Any) -> Any:
        return self.env.step(action)

    def reset(self, init_index: int | None = None) -> Any:
        return self.env.reset(init_index=init_index)

    def check_success(self) -> bool:
        # Same teleport-on-In semantics as VABEnv.step: stateful predicates
        # (pack_all_into) re-evaluate, marking newly-contained objects as
        # delivered and teleporting them to the graveyard pose.
        return bool(self.env._check_success())

    def seed(self, seed: int | None) -> None:
        pass  # vab inits are baked in the task YAML; seeding is VABHandle's job

    def close(self) -> None:
        self.env.close()


_VAB_RAW_ENV_CLS: type | None = None


def _get_vab_raw_env_cls() -> type:
    """Build (once) the raw-obs VABEnv subclass. Lazy: imports libero."""
    global _VAB_RAW_ENV_CLS
    if _VAB_RAW_ENV_CLS is not None:
        return _VAB_RAW_ENV_CLS

    from libero.vab.env import VABEnv  # type: ignore[import-not-found]

    class _RawObsVABEnv(VABEnv):
        """VABEnv with the strict-obs filter disabled.

        VAB's benchmark contract strips observations down to
        ``{images, proprio}`` so policies can't peek at privileged state.
        The gap env layer is the *bridge*, not the policy — it needs the
        raw robosuite keys (``agentview_image``, ``robot0_joint_pos``,
        ``robot0_eef_quat``, ...) that ``FrankaLiberoEnv.get_observation``
        assembles from, plus ground-truth body poses for the connector's
        ObjectState. Privilege separation is re-imposed downstream by
        whatever consumes the connector's Observation.
        """

        def _filter_obs(self, raw: dict[str, Any]) -> dict[str, Any]:
            return raw

        @property
        def obj_body_id(self) -> dict[str, int]:
            # Mirror the classic problem envs' public attribute so the
            # ground-truth pose assembly (and the perturbed env's basket
            # resolution) works unchanged on the vab path.
            return self._obj_body_id

    _VAB_RAW_ENV_CLS = _RawObsVABEnv
    return _RawObsVABEnv


def _vab_suite_dir(suite_name: str) -> Path | None:
    """Return the vab task dir for ``suite_name`` if it exists."""
    suite_dir = _VAB_ROOT / "tasks" / suite_name
    return suite_dir if suite_dir.is_dir() else None


#: Classic LIBERO task numbering for the ``libero_object``-derived vab
#: suites, ported from LIBERO's ``libero_task_map["libero_object"]`` (see
#: third_party/LIBERO-PRO/libero/libero/benchmark/libero_suite_task_map.py).
#: The dev sim bridge resolved ``(suite, task_id)`` through LIBERO's
#: benchmark registry, so task 1 is the CREAM CHEESE task — NOT the
#: alphabetically-second bbq sauce. A bare ``sorted()`` over the YAML
#: filenames silently renumbers 8 of the 10 tasks, desynchronizing every
#: config that addresses tasks by classic id (the G1
#: ``examples/benchmark/grocery_acceptance.yaml`` prompts, the dev
#: ``libero_object_*`` recipes): the generated graph then picks the
#: object the config asked for while the env scores a different target.
_CLASSIC_LIBERO_OBJECT_TASK_ORDER: tuple[str, ...] = (
    "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
    "pick_up_the_cream_cheese_and_place_it_in_the_basket",
    "pick_up_the_salad_dressing_and_place_it_in_the_basket",
    "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
    "pick_up_the_ketchup_and_place_it_in_the_basket",
    "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
    "pick_up_the_butter_and_place_it_in_the_basket",
    "pick_up_the_milk_and_place_it_in_the_basket",
    "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
    "pick_up_the_orange_juice_and_place_it_in_the_basket",
)


def _vab_task_files(suite_dir: Path) -> list[Path]:
    """Suite task files in classic LIBERO numbering.

    When the suite's YAML stems are exactly the classic ``libero_object``
    task set (the four ``libero_object*variance`` suites), order them by
    :data:`_CLASSIC_LIBERO_OBJECT_TASK_ORDER`; otherwise (crate washing,
    popcorn, packing — version-/scene-numbered files) keep lexicographic
    order, which is their intended numbering.
    """
    files = sorted(suite_dir.glob("*.yaml"))
    by_stem = {f.stem: f for f in files}
    if set(by_stem) == set(_CLASSIC_LIBERO_OBJECT_TASK_ORDER):
        return [by_stem[s] for s in _CLASSIC_LIBERO_OBJECT_TASK_ORDER]
    return files


def _load_vab_task(
    suite_name: str,
    task_id: int,
    suite_dir: Path,
    *,
    cam_w: int,
    cam_h: int,
    controller: str,
    horizon: int,
    control_freq: int,
    camera_depths: bool,
) -> VABHandle:
    """Construct a vab task env satisfying the LiberoHandle surface.

    ``task_id`` indexes the suite's tasks in classic LIBERO numbering
    (see :func:`_vab_task_files`): for the ``libero_object``-derived
    variance suites that is the classic ``libero_object`` task order
    (0 = alphabet soup, 1 = cream cheese, ...), matching how the dev sim
    bridge resolved task ids through LIBERO's benchmark registry.
    """
    _activate_libero_fork("vab")
    try:
        from libero.vab import load_task  # type: ignore[import-not-found]
    except Exception as e:
        raise ModuleNotFoundError(
            "The vab LIBERO fork is not importable. Install the sim stack:\n"
            '  uv pip install -e "gap[libero]"\n'
            "(wires libero → third_party/Variational-Automation-Benchmark)"
        ) from e

    task_files = _vab_task_files(suite_dir)
    if not task_files:
        raise FileNotFoundError(f"No task YAMLs in vab suite dir {suite_dir}")
    if not 0 <= task_id < len(task_files):
        raise IndexError(
            f"task_id {task_id} out of range for vab suite "
            f"{suite_name!r} ({len(task_files)} tasks)"
        )

    task = load_task(task_files[task_id])
    # Render at the bridge's camera size and enable depth — the YAML's
    # 128×128 RGB-only spec targets the strict benchmark contract, while
    # FrankaLiberoEnv's obs assembly expects ``<cam>_depth`` buffers and
    # computes intrinsics for its own render size. (Segmentation is not
    # plumbed: VABEnv pins camera_segmentations=None.)
    task.camera_width = cam_w
    task.camera_height = cam_h
    task.camera_depth = bool(camera_depths)
    task.horizon = horizon

    env_cls = _get_vab_raw_env_cls()
    env = env_cls(task=task, controller=controller, control_freq=control_freq)

    return VABHandle(
        env=VABControlEnv(env),
        suite_name=suite_name,
        task_id=task_id,
        task_language=task.language,
        init_states=task.inits,
        task=task,
    )


def _extract_language_from_bddl(bddl_path: str) -> str | None:
    """Extract task language from a BDDL file."""
    try:
        with open(bddl_path) as f:
            content = f.read()
        match = re.search(
            r"\(:language\s+(.*?)\)", content, re.DOTALL | re.IGNORECASE
        )
        if match:
            return match.group(1).strip()
    except Exception as e:
        print(f"Warning: Could not extract language from {bddl_path}: {e}")
    return None


def resolve_bddl_path(suite_name: str, task_id: int) -> str:
    """Resolve the BDDL file path for a classic LIBERO ``(suite_name, task_id)``.

    Mirrors the resolution inside :func:`load_libero_task` (benchmark-registry
    lookup + vendored fallback) but constructs no simulation env — cheap
    enough to call from scene-spec generation. Raises ``ModuleNotFoundError``
    if LIBERO is not importable. vab suites have no BDDL (self-contained
    YAML) — use ``_vab_suite_dir`` to detect them before calling.
    """
    _activate_libero_fork("pro")
    _ensure_libero_paths()
    try:
        from libero import benchmark  # type: ignore[import-not-found]
        from libero.utils import get_libero_path  # type: ignore[import-not-found]
    except Exception as e:
        raise ModuleNotFoundError(
            "LIBERO not available; cannot resolve BDDL path."
        ) from e

    benchmark_dict = benchmark.get_benchmark_dict(help=True)
    task_suite = benchmark_dict[suite_name]()
    task = task_suite.get_task(task_id)

    bddl_file_path = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    if not os.path.exists(bddl_file_path):
        fallback_path = os.path.join(
            str(_LIBERO_PRO_PKG_ROOT / "bddl_files"),
            task.problem_folder,
            task.bddl_file,
        )
        if os.path.exists(fallback_path):
            bddl_file_path = fallback_path
    return bddl_file_path


def load_libero_task(
    suite_name: str,
    task_id: int,
    cam_w: int = 128,
    cam_h: int = 128,
    controller: str = "OSC_POSE",
    horizon: int = 1000,
    control_freq: int = 20,
    camera_depths: bool = True,
    camera_segmentations: str | None = None,
) -> LiberoHandle:
    """Load a LIBERO task by suite name and task ID.

    Source-of-truth resolution: a suite name matching a ``tasks/<name>``
    dir in the vendored Variational-Automation-Benchmark repo loads via
    the vab YAML path; everything else goes through the classic LIBERO-PRO
    benchmark registry with OffScreenRenderEnv.

    Args:
        suite_name: LIBERO suite name (vab task dir or benchmark registry).
        task_id: Task index within the suite.
        cam_w: Camera image width.
        cam_h: Camera image height.
        controller: Controller type (e.g. "JOINT_POSITION", "OSC_POSE").
        horizon: Maximum episode steps.
        control_freq: Control frequency in Hz.
        camera_depths: Whether to render depth images.
        camera_segmentations: When set (``"instance"`` / ``"class"`` /
            ``"element"``), robosuite renders per-camera segmentation
            buffers and returns them in the obs dict as
            ``<cam>_segmentation_<level>``. Default ``None`` skips
            segmentation rendering. Classic path only — the vab env pins
            segmentation off.

    Returns:
        LiberoHandle wrapping the environment.
    """
    vab_dir = _vab_suite_dir(suite_name)
    if vab_dir is not None:
        return _load_vab_task(
            suite_name,
            task_id,
            vab_dir,
            cam_w=cam_w,
            cam_h=cam_h,
            controller=controller,
            horizon=horizon,
            control_freq=control_freq,
            camera_depths=camera_depths,
        )

    _activate_libero_fork("pro")
    _ensure_libero_paths()

    try:
        from libero import benchmark  # type: ignore[import-not-found]
        from libero.envs import OffScreenRenderEnv  # type: ignore[import-not-found]
        from libero.utils import get_libero_path  # type: ignore[import-not-found]
    except Exception as e:
        raise ModuleNotFoundError(
            "LIBERO not available.  Install the sim stack:\n"
            '  uv pip install -e "gap[libero]"\n'
            "and initialize the third_party/LIBERO-PRO submodule."
        ) from e

    benchmark_dict = benchmark.get_benchmark_dict(help=True)
    task_suite = benchmark_dict[suite_name]()
    task = task_suite.get_task(task_id)

    # Resolve BDDL file path (with fallback to vendored location)
    bddl_file_path = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )

    if not os.path.exists(bddl_file_path):
        fallback_bddl_root = str(_LIBERO_PRO_PKG_ROOT / "bddl_files")
        fallback_path = os.path.join(
            fallback_bddl_root, task.problem_folder, task.bddl_file
        )
        if os.path.exists(fallback_path):
            print(f"Using vendored BDDL: {fallback_path}")
            bddl_file_path = fallback_path

    env_args = {
        "bddl_file_name": bddl_file_path,
        "camera_heights": cam_h,
        "camera_widths": cam_w,
        "controller": controller,
        "horizon": horizon,
        "control_freq": control_freq,
        "camera_depths": camera_depths,
    }
    if camera_segmentations is not None:
        env_args["camera_segmentations"] = camera_segmentations
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)

    # Extract task language
    task_language = _extract_language_from_bddl(bddl_file_path)
    if not task_language:
        task_language = task.language

    # Load init states (with fallback to vendored location).
    # PyTorch >=2.6 defaults weights_only=True which breaks LIBERO's
    # pickle files, so we catch broadly and retry with weights_only=False.
    import torch

    init_states = None
    try:
        init_states = task_suite.get_task_init_states(task_id)
    except Exception:
        pass

    if init_states is None:
        init_states_path = os.path.join(
            get_libero_path("init_states"),
            task.problem_folder,
            task.init_states_file,
        )
        if not os.path.exists(init_states_path):
            fallback_init_root = str(_LIBERO_PRO_PKG_ROOT / "init_files")
            init_states_path = os.path.join(
                fallback_init_root, task.problem_folder, task.init_states_file
            )
        if os.path.exists(init_states_path):
            init_states = torch.load(init_states_path, weights_only=False)
        else:
            # No pruned_init for this task (e.g. user-added BDDLs that
            # haven't gone through the init-sampling tool).
            # FrankaLiberoEnv doesn't actually consume init_states for the
            # snapshot path — it inits from the BDDL's :init clauses. Pass
            # an empty array through so callers that DO need a sampler get
            # an explicit "no samples available" signal rather than a torch
            # load crash at import.
            import numpy as np
            init_states = np.empty((0, 0), dtype=np.float64)

    return LiberoHandle(
        env=env,
        suite_name=suite_name,
        task_id=task_id,
        task_language=task_language,
        init_states=init_states,
    )
