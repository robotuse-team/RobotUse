"""FrankaLiberoEnv — standalone LIBERO environment wrapper.

Ported from HyRL's hyrl/envs/libero_low_level.py with viser debug
code stripped. Provides observation, control, and task completion
checking for the gap connector layer.

Dependencies (installed via the ``gap[libero]`` extra):
  - libero    (vendored at third_party/Variational-Automation-Benchmark;
               the classic benchmark suites load from third_party/LIBERO-PRO
               — see gap.envs.loader for the source-of-truth resolution)
  - robosuite (vendored at third_party/robosuite)
  - viser     (pip, for SE3/SO3 transforms)
  - mujoco, gymnasium, numpy, scipy, torch
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np
import viser.transforms as vtf

from gap import env_config

from .base_env import BaseEnv
from .loader import load_libero_task
from .registry import EnvConfig

logger = logging.getLogger(__name__)

# LIBERO default "home" joint configuration (canonical Franka home; same
# fallback the source sim bridge used for GoHome on the libero branch).
_FRANKA_HOME_JOINTS = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)


class FrankaLiberoEnv(BaseEnv):
    """Franka LIBERO environment.

    Wraps LIBERO's OffScreenRenderEnv with joint-position control,
    camera observation extraction, and task completion checking.
    """

    def __init__(
        self,
        suite_name: str,
        task_id: int,
        max_steps: int = 20000,
        seed: int | None = None,
        enable_render: bool = False,
        control_freq: int = 20,
        camera_names: list[str] | None = None,
        joint_motion_mode: str = "closed_loop",
    ) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.seed = seed
        self.enable_render = enable_render
        self.segmentation_level = "element"
        self._render_width = 800
        self._render_height = 512
        self.camera_names = camera_names or ["agentview", "robot0_eye_in_hand"]

        # The robosuite env is constructed with OSC_POSE because that's the
        # action space ``pi05_libero`` and other LIBERO π-series checkpoints
        # emit (see ``apply_policy_action``).  A second JointPositionController
        # is built post-init and swapped in for joint-space callers
        # (``move_to_joints_blocking``, GoHome, joint-trajectory execution)
        # so they get proper closed-loop tracking instead of qpos teleports
        # that fight a stale OSC interpolator.  See ``_use_controller``.
        self.handle = load_libero_task(
            suite_name=suite_name,
            task_id=task_id,
            cam_w=self._render_width,
            cam_h=self._render_height,
            controller="OSC_POSE",
            horizon=max_steps,
            control_freq=control_freq,
            camera_segmentations=self.segmentation_level,
        )

        # State tracking
        self._step_count = 0
        self._sim_step_count = 0
        self._sim_physics_wall_s = 0.0
        self._command_count = 0
        """Unique actuator commands issued — bumped once per ``step``,
        ``apply_policy_action``, or ``move_to_joints_blocking`` call.
        Decoupled from ``_sim_step_count`` (which also counts closed-loop
        tracking iterations and ``_step_once`` settles) so latency can
        report what a real robot at the same control rate would have
        sent. Real-robot motion time ≈ ``_command_count / control_freq``."""
        self._reset_step_baseline = 0
        self._reset_phys_baseline = 0.0
        self._reset_command_baseline = 0
        self._control_freq = control_freq
        self._rng = np.random.default_rng(self.seed)
        self._current_obs = None
        self._current_info = None
        self._current_reward = None
        self._current_done = None

        # Video capture. ``_subsample_rate`` records one frame per N sim steps;
        # save_video encodes at 20 fps, so subsample=4 gives a real-time video
        # for a 20 Hz sim. Bumped from 1 because closed-loop joint moves can
        # log ~10× more sim steps per trial than qpos teleports, blowing the
        # video duration up by the same factor.
        self._record_frames = False
        # Captured frames are streamed to disk (one PNG each) under a
        # per-session temp dir rather than held in memory; see _record_frame.
        self._frames_dir: str | None = None
        self._frames_seq: int = 0
        self._subsample_rate = 4

        # ``closed_loop`` (default) swaps in the JointPositionController and
        # tracks under physics — physically faithful, and leaves OSC's
        # interpolator clean, which matters when a policy node interleaves OSC
        # actions with joint trajectories in the same workflow.
        # ``teleport`` writes joint qpos directly and settles a few OSC
        # zero-action steps — faster (fewer sim steps), fine for pure graph
        # workflows that only need the arm to reach the IK solution. Selected
        # per worker via ``GAP_LIBERO_JOINT_MOTION_MODE`` (see ``make_env``).
        if joint_motion_mode not in ("teleport", "closed_loop"):
            raise ValueError(
                f"joint_motion_mode must be 'teleport' or 'closed_loop', "
                f"got {joint_motion_mode!r}"
            )
        self._joint_motion_mode = joint_motion_mode

        # Policy-like streaming execution. When enabled, planned joint
        # trajectories are followed by a synchronous path-following servo
        # (``stream_joint_trajectory``) at near-max controller speed instead of
        # per-waypoint convergence + settling — the dominant motion-time sink.
        # Kill-switch ``GAP_LIBERO_STREAM=0`` restores the legacy per-waypoint
        # path (gated in ``WorldAdapter._execute_trajectory``).
        self._stream_enabled = env_config.libero_stream()
        # Per-tick joint-step clamp as a fraction of ``output_max`` (≤1.0).
        # Lower it to be gentler on a carried payload at some speed cost.
        self._stream_max_step_frac = env_config.libero_stream_max_step_frac()

        # Skip per-step camera rendering during pure motion. robosuite renders
        # both cameras inside every ``handle.step`` (~16 ms/step here); geometric
        # moves don't need camera obs, so the connector deactivates the camera
        # observables for the duration of a motion/settle segment (and refreshes
        # once before perception). ``GAP_LIBERO_MOTION_RENDER=0`` enables it.
        self._motion_render = env_config.libero_motion_render()
        self._cam_obs_names: list[str] | None = None

        # Robot link indices for transforms
        self.gripper_metric_length = 0.04
        self.base_link_idx = self.handle.env.sim.model.body_name2id("robot0_base")
        self.gripper_link_idx = self.handle.env.sim.model.body_name2id("gripper0_eef")

        self.base_link_wxyz_xyz = np.concatenate([
            self.handle.env.sim.data.xquat[self.base_link_idx],
            self.handle.env.sim.data.xpos[self.base_link_idx],
        ])

        self.gripper_link_wxyz_xyz = np.concatenate([
            self.handle.env.sim.data.xquat[self.gripper_link_idx],
            self.handle.env.sim.data.xpos[self.gripper_link_idx],
        ])

        # Precompute Panda joint qpos addresses
        joint_names = [f"robot0_joint{i}" for i in range(1, 8)]
        self._panda_joint_qpos_addrs: list[int] = []
        for jn in joint_names:
            addr = self.handle.env.sim.model.get_joint_qpos_addr(jn)
            if isinstance(addr, tuple):
                addr = addr[0]
            self._panda_joint_qpos_addrs.append(int(addr))

        self.home_joint_position: np.ndarray | None = None

        # Controller refs are stale until the first reset() — robosuite rebuilds
        # robot.controller (and frees the MjSim) inside Robot.reset(). The pair
        # ``_osc_ctrl`` / ``_joint_ctrl`` is re-bound there.
        self._osc_ctrl = None
        self._joint_ctrl = None
        self._ctrl_mode: str = "osc"
        self._joint_output_max: float = 0.05

        self.reset()

    # ----------------------- Dual-controller plumbing -----------------------

    def _rebuild_controllers(self) -> None:
        """Capture the just-rebuilt OSC controller and build a parallel joint one.

        ``Robot.reset()`` calls ``_load_controller`` every hard reset and
        ``MjSim.free()`` deletes the old sim's ``model`` attribute, so any
        controller (or sim ref) cached pre-reset is unusable post-reset.
        Call this after each ``handle.reset()``.
        """
        from robosuite.controllers import controller_factory, load_controller_config

        robot = self.handle.env.robots[0]
        self._robot = robot
        self._osc_ctrl = robot.controller  # freshly built by Robot.reset()

        joint_cfg = load_controller_config(default_controller="JOINT_POSITION")
        joint_cfg.update({
            "robot_name": robot.name,
            "sim": robot.sim,
            "eef_name": robot.gripper.important_sites["grip_site"],
            "eef_rot_offset": robot.eef_rot_offset,
            "joint_indexes": {
                "joints": robot.joint_indexes,
                "qpos": robot._ref_joint_pos_indexes,
                "qvel": robot._ref_joint_vel_indexes,
            },
            "actuator_range": robot.torque_limits,
            "policy_freq": robot.control_freq,
            "ndim": len(robot.robot_joints),
        })
        self._joint_ctrl = controller_factory("JOINT_POSITION", joint_cfg)
        self._joint_ctrl.update_base_pose(robot.base_pos, robot.base_ori)
        # JOINT_POSITION action scaling: input ∈ [-1,1] maps to ≤ output_max
        # rad per joint per env step (default 0.05). Cache for the closed-loop
        # tracker in ``move_to_joints_blocking``.
        self._joint_output_max = float(joint_cfg.get("output_max", 0.05))
        # Robosuite left OSC active after its rebuild; start there.
        self._ctrl_mode = "osc"

    def _use_controller(self, mode: str) -> None:
        """Swap ``robots[0].controller`` and sync its internal state.

        After the swap, the new controller's goal is rebound to the current
        sim state — without this, OSC's interpolator (or the joint
        controller's stored ``goal_qpos``) carries the stale setpoint from
        the previous mode and drags the arm back on the first env.step.
        """
        if mode == self._ctrl_mode:
            return
        if mode not in ("osc", "joint"):
            raise ValueError(f"unknown controller mode: {mode!r}")
        new_ctrl = self._joint_ctrl if mode == "joint" else self._osc_ctrl
        self._robot.controller = new_ctrl
        new_ctrl.update(force=True)
        # ``LinearInterpolator.set_goal`` rotates ``start ← old goal`` on each
        # call, so a single ``reset_goal`` leaves ``start`` stale. Call twice
        # to flush ``start = goal = current``.
        new_ctrl.reset_goal()
        new_ctrl.reset_goal()
        from robosuite.utils.buffers import DeltaBuffer
        self._robot.recent_actions = DeltaBuffer(dim=self._robot.action_dim)
        # Robosuite caches the env-level action_dim at reset (sum of per-robot
        # action_dim); it's not a property, so we recompute here so the
        # _pre_action shape assertion passes after the swap.
        rs_env = self.handle.env.env
        rs_env._action_dim = sum(r.action_dim for r in rs_env.robots)
        self._ctrl_mode = mode

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        libero_obs, libero_info = self.handle.reset(seed=seed)

        self._current_obs = libero_obs
        self._current_info = libero_info

        self._step_count = 0
        self._sim_step_count = 0
        self._sim_physics_wall_s = 0.0

        self._current_joints = self.handle.env.sim.data.qpos[:7].copy()
        self.home_joint_position = np.array(
            libero_obs["robot0_joint_pos"], dtype=np.float64
        )
        self._gripper_fraction = 1.0
        self._gripper_width_target_m = None

        # robosuite hard_reset re-instantiated MjSim + rebuilt the OSC
        # controller; capture the new ref and build a fresh parallel joint
        # controller against the same sim.
        self._rebuild_controllers()

        # Settle simulation after reset
        for _ in range(40):
            self._step_once()

        # Latency baseline: the 40 settle steps above are sim warmup,
        # not actuator commands the policy issued — exclude them from
        # ``get_latency_info``. We don't zero ``_sim_step_count`` here
        # because ``max_steps`` truncation + ``get_current_time_s`` and
        # frame-recording downstream all read the legacy counter.
        self._reset_step_baseline = self._sim_step_count
        self._reset_phys_baseline = self._sim_physics_wall_s
        self._reset_command_baseline = self._command_count

        obs = self.get_observation()
        self.gripper_link_wxyz_xyz = np.concatenate([
            self.handle.env.sim.data.xquat[self.gripper_link_idx],
            self.handle.env.sim.data.xpos[self.gripper_link_idx],
        ])

        info = {"task_prompt": self.handle.task_language}
        return obs, info

    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Step the simulation with the given action."""
        self._step_count += 1
        self._command_count += 1
        _t0 = time.perf_counter()
        self._current_obs, self._current_reward, self._current_done, self._current_info = (
            self.handle.step(action)
        )
        self._sim_physics_wall_s += time.perf_counter() - _t0
        self._sim_step_count += 1
        self.gripper_link_wxyz_xyz = np.concatenate([
            self.handle.env.sim.data.xquat[self.gripper_link_idx],
            self.handle.env.sim.data.xpos[self.gripper_link_idx],
        ])
        if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
            self._record_frame()
        obs = self.get_observation()
        reward = self.compute_reward()
        terminated = bool(self._current_done)
        truncated = self._sim_step_count >= self.max_steps
        info: dict[str, Any] = dict(self._current_info or {})
        info["reward"] = float(reward)
        info.update(self.get_latency_info())
        return obs, reward, terminated, truncated, info

    def get_latency_info(self) -> dict[str, Any]:
        """Cumulative episode latency since the last :meth:`reset`.

        ``control_steps`` counts unique high-level actuator commands —
        one per ``step``, ``apply_policy_action``, or
        ``move_to_joints_blocking`` call. NOT the closed-loop tracking
        iterations a sim controller spins through to converge (those
        live in ``_sim_step_count``). On a real Franka with the same
        control rate, that count IS what gets sent to the actuator, so
        ``control_steps / control_freq`` is the real-robot motion time.

        Pair with the run's total wall time (outside this env) to derive
        ``compute_overhead = total_wall − sim_physics_wall`` and
        ``physical_execution = control_steps / control_freq +
        compute_overhead``.
        """
        return {
            "control_steps": int(self._command_count - self._reset_command_baseline),
            "control_freq": float(self._control_freq),
            "sim_physics_wall_s": float(
                self._sim_physics_wall_s - self._reset_phys_baseline
            ),
        }

    # ----------------------- Control Interface -----------------------

    def _camera_observable_names(self) -> list[str]:
        """Camera image/depth/segmentation observable names (cached)."""
        if self._cam_obs_names is None:
            try:
                rs = self.handle.env.env
                self._cam_obs_names = [
                    n for n in getattr(rs, "_observables", {})
                    if any(k in n for k in ("image", "depth", "segmentation"))
                ]
            except Exception:
                self._cam_obs_names = []
        return self._cam_obs_names

    def set_cameras_active(self, active: bool) -> None:
        """Activate/deactivate robosuite camera observables. Deactivating skips
        the per-step offscreen render (the dominant per-step wall cost); proprio
        observables stay active, so motion control is unaffected."""
        try:
            rs = self.handle.env.env
        except Exception:
            return
        for n in self._camera_observable_names():
            try:
                rs.modify_observable(n, "active", active)
                rs.modify_observable(n, "enabled", active)
            except Exception:
                pass

    def refresh_camera_obs(self) -> None:
        """One hold-step so ``_current_obs`` carries fresh camera images for the
        next perception read (cameras were off during a motion segment)."""
        action = np.zeros(self._robot.action_dim, dtype=np.float64)
        action[-1] = 1.0 - self._gripper_fraction * 2.0
        _t0 = time.perf_counter()
        self._current_obs, self._current_reward, self._current_done, self._current_info = (
            self.handle.step(action)
        )
        self._sim_physics_wall_s += time.perf_counter() - _t0
        self._sim_step_count += 1

    def move_to_joints_blocking(
        self,
        joints: np.ndarray,
        *,
        tolerance: float = 0.01,
        max_steps: int = 120,
        arm_id: int = 0,
    ) -> None:
        """Move to target joints. Implementation depends on ``_joint_motion_mode``.

        ``teleport`` writes ``qpos = target`` (warm-start) then settles a
        fixed-budget number of sim steps under the JointPositionController
        with the controller goal + interpolator pinned to the teleported
        pose. The fixed budget bounds the per-waypoint cost: contact-rich
        descents can't converge on a tight tolerance (joints oscillate
        against the object), and an open-ended convergence loop would
        balloon execute time. Empirically 10 steps is enough for contact
        forces to resolve without going past CuRobo's intended waypoint
        cadence.

        ``closed_loop`` iterates until ``error < tolerance`` so joints
        smoothly track the target under physics. Required for mixed
        graph+policy workflows where OSC and joint moves interleave.
        """
        target = np.asarray(joints, dtype=np.float64).reshape(7)
        self._current_joints = target
        # One logical motion command per call. The closed-loop branch
        # below may iterate many ``handle.step`` calls to converge, but
        # those are sim-controller tracking — a real Franka at the same
        # control rate gets sent ONE waypoint command here.
        self._command_count += 1

        # Teleport is only safe when the hand is empty. Once the gripper has
        # closed on something (``_gripper_fraction`` near 0 = closed), a qpos
        # jump displaces the held object's finger contacts instantly and the
        # next mj_step ejects it from the grasp. Fall back to closed-loop
        # tracking whenever we're carrying a payload (e.g. the transport /
        # descend-release moves).
        effective_mode = self._joint_motion_mode
        if effective_mode == "teleport" and self._gripper_fraction < 0.5:
            effective_mode = "closed_loop"

        if effective_mode == "teleport":
            # qpos teleport + sync joint controller goal/interpolator to the
            # new pose. _use_controller is a no-op after the first call (mode
            # already joint), but the LinearInterpolator rotates
            # ``start ← old_goal`` on every ``set_goal`` — so without an
            # explicit ``update + reset_goal ×2`` per waypoint, the
            # interpolator carries the *previous* waypoint's target as
            # ``start`` and drags the joints backwards through the
            # just-teleported gap. (Single reset_goal leaves ``start``
            # stale; second call flushes start ← goal = current = target.)
            self.handle.env.sim.data.qpos[self._panda_joint_qpos_addrs] = target
            self.handle.env.sim.forward()
            self._use_controller("joint")
            self._joint_ctrl.update(force=True)
            self._joint_ctrl.reset_goal()
            self._joint_ctrl.reset_goal()
            # Fixed budget. 10 sim steps ≈ 0.5 s of physics — enough for the
            # finger-vs-table or finger-vs-object contact to resolve before
            # the next teleport. ``min(.., max_steps)`` lets callers (tests,
            # nonblocking single moves) request a smaller budget.
            for _ in range(max(0, min(10, max_steps))):
                self._step_once()
            return

        self._use_controller("joint")
        sim = self.handle.env.sim
        addrs = self._panda_joint_qpos_addrs
        output_max = self._joint_output_max

        steps = 0
        while steps < max_steps:
            current = np.array(sim.data.qpos[addrs], dtype=np.float64)
            error = float(np.linalg.norm(current - target))
            if error < tolerance and steps > 0:
                break
            normalized = np.clip((target - current) / output_max, -1.0, 1.0)
            gripper_cmd = 1.0 - self._gripper_fraction * 2.0
            action = np.concatenate([normalized, [gripper_cmd]])
            _t0 = time.perf_counter()
            self._current_obs, self._current_reward, self._current_done, self._current_info = (
                self.handle.step(action)
            )
            self._sim_physics_wall_s += time.perf_counter() - _t0
            self._sim_step_count += 1
            self.gripper_link_wxyz_xyz = np.concatenate([
                self.handle.env.sim.data.xquat[self.gripper_link_idx],
                self.handle.env.sim.data.xpos[self.gripper_link_idx],
            ])
            if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
                self._record_frame()
            steps += 1

    def stream_joint_trajectory(
        self,
        waypoints: list,
        *,
        settle_tolerance: float = 0.01,
        settle_max_steps: int = 60,
        arm_id: int = 0,
    ) -> None:
        """Feed-forward joint-space path-following servo — policy-like motion.

        Streams a planned joint polyline at near-max JointPositionController
        speed instead of converging + settling at every waypoint. Each sim tick
        commands the clamped delta toward the *current* target waypoint and
        advances the pointer once the arm is within ~half a controller step of
        it, so a dense path runs ≈1 sim tick per waypoint and a coarse path
        never lags (the controller keeps moving at ``output_max``). One bounded
        final settle converges the LAST waypoint to ``settle_tolerance`` — that
        is what preserves grasp-pose / policy hand-off precision.

        This is the env hook ``WorldAdapter._execute_trajectory`` prefers when
        present (``gap/connector/core.py``): without it, trajectories fall back
        to per-waypoint convergence — the dominant motion-time sink. Mirrors
        the real-robot streaming model (``franka_real_env``) but stays
        synchronous (the sim only advances when we ``handle.step``); no
        background thread.

        Instrumentation: each *streamed* (commanded) tick counts one
        ``_command_count`` — matching the per-tick ``apply_policy_action``
        convention, so ``_command_count / control_freq`` stays an honest
        real-robot motion time and is directly comparable to a policy run.
        Final-settle ticks count toward ``_sim_step_count`` only (convergence
        tracking, like the closed-loop tail).
        """
        wps = [np.asarray(w, dtype=np.float64).reshape(-1) for w in waypoints]
        wps = [w for w in wps if w.shape[0] >= 7]
        if not wps:
            return
        wps = [w[:7] for w in wps]

        self._use_controller("joint")
        sim = self.handle.env.sim
        addrs = self._panda_joint_qpos_addrs
        output_max = self._joint_output_max
        # Per-tick joint-step clamp (≤ output_max). 1.0 = full controller speed.
        max_step = max(1e-4, output_max * self._stream_max_step_frac)
        advance_thresh = 0.5 * output_max

        def _tick(target: np.ndarray, *, command: bool) -> float:
            current = np.array(sim.data.qpos[addrs], dtype=np.float64)
            step_delta = np.clip(target - current, -max_step, max_step)
            normalized = np.clip(step_delta / output_max, -1.0, 1.0)
            gripper_cmd = 1.0 - self._gripper_fraction * 2.0
            action = np.concatenate([normalized, [gripper_cmd]])
            _t0 = time.perf_counter()
            self._current_obs, self._current_reward, self._current_done, self._current_info = (
                self.handle.step(action)
            )
            self._sim_physics_wall_s += time.perf_counter() - _t0
            self._sim_step_count += 1
            if command:
                self._command_count += 1
            self.gripper_link_wxyz_xyz = np.concatenate([
                self.handle.env.sim.data.xquat[self.gripper_link_idx],
                self.handle.env.sim.data.xpos[self.gripper_link_idx],
            ])
            if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
                self._record_frame()
            new = np.array(sim.data.qpos[addrs], dtype=np.float64)
            return float(np.linalg.norm(new - target))

        # ── Pursuit: stream along the polyline at clamped-max speed ──
        last = len(wps) - 1
        k = 0
        # Runaway guard only (joint-limit / singular target won't converge).
        max_pursuit = 10 * len(wps) + 50
        ticks = 0
        stuck = 0
        best_err = float("inf")
        while ticks < max_pursuit:
            current = np.array(sim.data.qpos[addrs], dtype=np.float64)
            if float(np.max(np.abs(wps[k] - current))) < advance_thresh:
                if k < last:
                    k += 1
                    stuck = 0
                    best_err = float("inf")
                    continue
                break  # within a controller step of the final waypoint
            err = _tick(wps[k], command=True)
            ticks += 1
            # No-progress guard: skip an unreachable waypoint (or stop at the
            # last) rather than spinning the full budget against a joint limit.
            if err < best_err - 1e-4:
                best_err = err
                stuck = 0
            else:
                stuck += 1
                if stuck >= 15:
                    if k < last:
                        k += 1
                        stuck = 0
                        best_err = float("inf")
                    else:
                        break

        # ── Final convergence settle (tracking-only) on the last waypoint ──
        target = wps[last]
        settle = 0
        while settle < settle_max_steps:
            if _tick(target, command=False) < settle_tolerance:
                break
            settle += 1

    def apply_policy_action(self, action: np.ndarray) -> None:
        """Send a single 7-dim OSC_POSE action to the underlying env.

        Action layout: ``[Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]`` where the
        gripper component matches LIBERO/robosuite convention (positive →
        close, negative → open).  This is the action space that
        ``pi05_libero`` and other LIBERO π-series checkpoints emit
        directly, so callers can pass each row of the policy's action
        chunk through verbatim.
        """
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.size != 7:
            raise ValueError(
                f"apply_policy_action: expected 7-dim action, got shape {a.shape}"
            )
        self._use_controller("osc")
        self._step_count += 1
        self._command_count += 1
        _t0 = time.perf_counter()
        self._current_obs, self._current_reward, self._current_done, self._current_info = (
            self.handle.step(a)
        )
        self._sim_physics_wall_s += time.perf_counter() - _t0
        self._sim_step_count += 1
        # Robosuite OSC convention: action[-1] = -1 → fully open, +1 → closed.
        # Map back to the bridge's open-fraction representation so
        # subsequent _step_once() calls hold the policy's gripper command.
        self._gripper_fraction = float(np.clip(0.5 - 0.5 * a[-1], 0.0, 1.0))
        self.gripper_link_wxyz_xyz = np.concatenate([
            self.handle.env.sim.data.xquat[self.gripper_link_idx],
            self.handle.env.sim.data.xpos[self.gripper_link_idx],
        ])
        if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
            self._record_frame()

    def _set_gripper(self, fraction: float, arm_id: int = 0) -> None:
        """Set gripper opening fraction (0.0 = closed, 1.0 = open)."""
        self._gripper_fraction = float(np.clip(fraction, 0.0, 1.0))
        self._gripper_width_target_m = None

    def _set_gripper_width(self, width_m: float, arm_id: int = 0) -> None:
        """Set Panda's position-controller target, then hold it with zero action.

        PandaGripper.format_action integrates the SIGN of its input; a
        fractional open/close command is not an absolute aperture. Seed the
        existing actuator target instead, without changing physical joint state.
        """
        from robosuite.models.grippers.panda_gripper import PandaGripper

        gripper = self._robot.gripper
        if arm_id != 0 or not isinstance(gripper, PandaGripper):
            raise NotImplementedError("absolute width requires the LIBERO Panda gripper")
        if not np.isfinite(width_m) or not 0.0 <= width_m <= 0.08:
            raise ValueError("Panda opening must be in [0, 0.08] metres")
        fraction = float(width_m) / 0.08
        gripper.current_action = np.array([2.0 * fraction - 1.0, 1.0 - 2.0 * fraction])
        self._gripper_fraction = 0.5  # Zero input preserves the position target.
        self._gripper_width_target_m = float(width_m)

    def _step_once(self) -> None:
        """Idle step: zero arm action + last gripper command.

        Action dimension follows the active controller (7 for OSC, 8 for
        JOINT_POSITION); a zero arm action means "hold current pose" under
        both — OSC's delta is zero, JOINT_POSITION's normalized delta is
        zero so ``goal_qpos`` stays at the current joint configuration.
        """
        action_dim = self._robot.action_dim
        action = np.zeros(action_dim, dtype=np.float64)
        action[-1] = 1.0 - self._gripper_fraction * 2.0

        _t0 = time.perf_counter()
        self._current_obs, self._current_reward, self._current_done, self._current_info = (
            self.handle.step(action)
        )
        self._sim_physics_wall_s += time.perf_counter() - _t0
        self._sim_step_count += 1
        # Settle / hold-pose cycles ARE control commands at 20 Hz on a
        # real robot (gripper open/close + post-move dampers all spin
        # the inner loop while no arm waypoint moves). Closed-loop
        # tracking inside ``move_to_joints_blocking`` bypasses
        # ``_step_once`` — it calls ``handle.step`` directly — so those
        # convergence iterations are NOT counted here.
        self._command_count += 1

        self.gripper_link_wxyz_xyz = np.concatenate([
            self.handle.env.sim.data.xquat[self.gripper_link_idx],
            self.handle.env.sim.data.xpos[self.gripper_link_idx],
        ])

        if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
            self._record_frame()

    # ----------------------- Observation -----------------------

    def compute_reward(self) -> float:
        return self._current_reward

    def get_observation(self) -> dict[str, Any]:
        """Get observation in structured format with camera data and robot state."""
        obs: dict[str, Any] = {}

        for camera_name in self.camera_names:
            if camera_name not in obs:
                obs[camera_name] = {}

            cam_world_wxyz_xyz = np.concatenate([
                vtf.SO3.from_matrix(
                    self.handle.env.sim.data.get_camera_xmat(camera_name)
                ).wxyz,
                self.handle.env.sim.data.get_camera_xpos(camera_name),
            ])

            cam_robot_tf = (
                (
                    vtf.SE3(wxyz_xyz=self.base_link_wxyz_xyz).inverse()
                    @ vtf.SE3(wxyz_xyz=cam_world_wxyz_xyz)
                )
                @ vtf.SE3.from_rotation_and_translation(
                    rotation=vtf.SO3.from_rpy_radians(0.0, np.pi, 0.0),
                    translation=np.array([0, 0, 0]),
                )
                @ vtf.SE3.from_rotation_and_translation(
                    rotation=vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi),
                    translation=np.array([0, 0, 0]),
                )
            )
            obs[camera_name]["pose"] = np.concatenate([
                cam_robot_tf.translation(),
                cam_robot_tf.rotation().wxyz,
            ])
            obs[camera_name]["pose_mat"] = cam_robot_tf.as_matrix()

            cam_id = self.handle.env.sim.model.camera_name2id(camera_name)
            fovy = self.handle.env.sim.model.cam_fovy[cam_id]
            f = 0.5 * self._render_height / np.tan(fovy * np.pi / 360.0)
            K = np.array([
                [f, 0, 0.5 * self._render_width],
                [0, f, 0.5 * self._render_height],
                [0, 0, 1],
            ])
            obs[camera_name]["intrinsics"] = K

            obs[camera_name]["images"] = {}
            if camera_name + "_image" in self._current_obs:
                obs[camera_name]["images"]["rgb"] = self._current_obs[
                    camera_name + "_image"
                ][::-1]
            if camera_name + "_depth" in self._current_obs:
                from robosuite.utils.camera_utils import get_real_depth_map as _get_depth

                # MuJoCo's EGL offscreen renderer occasionally returns a
                # handful of out-of-range pixels (NaN / Inf / values like
                # 1e7) that trip robosuite's [0,1] assertion in
                # ``get_real_depth_map``.  Clamp into the expected range
                # before conversion; the max-far value reads as unknown
                # downstream which matches what the bad pixels represent.
                raw_depth = np.nan_to_num(
                    self._current_obs[camera_name + "_depth"][::-1],
                    nan=1.0, posinf=1.0, neginf=0.0,
                )
                raw_depth = np.clip(raw_depth, 0.0, 1.0)
                depth_metric = _get_depth(self.handle.env.sim, raw_depth)
                obs[camera_name]["images"]["depth"] = depth_metric
            if (
                camera_name + "_segmentation_" + self.segmentation_level
                in self._current_obs
            ):
                obs[camera_name]["images"]["segmentation"] = self._current_obs[
                    camera_name + "_segmentation_" + self.segmentation_level
                ][::-1]

        gripper_robot_base = (
            vtf.SE3(wxyz_xyz=self.base_link_wxyz_xyz).inverse()
            @ vtf.SE3(wxyz_xyz=self.gripper_link_wxyz_xyz)
            @ vtf.SE3.from_rotation_and_translation(
                rotation=vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi / 2.0),
                translation=np.array([0, 0, -0.107]),
            )
        )
        obs["robot_joint_pos_0"] = np.concatenate([
            self._current_obs["robot0_joint_pos"],
            [self._current_obs["robot0_gripper_qpos"][0] / self.gripper_metric_length],
        ])
        obs["robot_cartesian_pos_0"] = np.concatenate([
            gripper_robot_base.translation(),
            gripper_robot_base.rotation().wxyz,
            [self._current_obs["robot0_gripper_qpos"][0] / self.gripper_metric_length],
        ])

        # Raw robosuite proprioception in the exact layout pi05_libero
        # was trained on: world-frame eef site position (3), world-frame
        # axis-angle of eef body quaternion (3), raw 2-finger gripper
        # qpos (2).  Surfaced as-is so policy encoders can pass it through
        # to the model without any frame conversion / TCP offset that
        # would distribution-shift it.
        eef_pos = np.asarray(
            self._current_obs.get("robot0_eef_pos", np.zeros(3)),
            dtype=np.float64,
        ).reshape(3)
        eef_quat_xyzw = np.asarray(
            self._current_obs.get("robot0_eef_quat", np.array([0.0, 0.0, 0.0, 1.0])),
            dtype=np.float64,
        ).reshape(4)
        gripper_qpos_raw = np.asarray(
            self._current_obs.get("robot0_gripper_qpos", np.zeros(2)),
            dtype=np.float64,
        ).reshape(-1)[:2]
        if gripper_qpos_raw.size < 2:
            gripper_qpos_raw = np.pad(gripper_qpos_raw, (0, 2 - gripper_qpos_raw.size))

        # Robosuite-style axisangle (matches openpi/examples/libero/main.py).
        w = float(np.clip(eef_quat_xyzw[3], -1.0, 1.0))
        den = np.sqrt(1.0 - w * w)
        if den < 1e-9:
            axisangle = np.zeros(3, dtype=np.float64)
        else:
            axisangle = (eef_quat_xyzw[:3] * 2.0 * np.arccos(w)) / den

        obs["robot_proprio_pi05_libero_0"] = np.concatenate([
            eef_pos.astype(np.float64),
            axisangle.astype(np.float64),
            gripper_qpos_raw.astype(np.float64),
        ])

        # Ground-truth object poses (gap extension). The connector reads
        # ``obs["cube_poses"]`` ({name: [x, y, z, qw, qx, qy, qz]}) for
        # ObjectState ground truth. The classic LIBERO problem envs surface
        # ``obj_body_id`` on the wrapped domain env and the vab adapter
        # mirrors it (see gap.envs.loader), so both paths land here.
        domain = getattr(self.handle.env, "env", self.handle.env)
        body_id_map = getattr(domain, "obj_body_id", None)
        if body_id_map:
            sim_data = self.handle.env.sim.data
            obs["cube_poses"] = {
                name: np.concatenate([
                    sim_data.xpos[int(body_id)],
                    sim_data.xquat[int(body_id)],
                ])
                for name, body_id in body_id_map.items()
            }
        return obs

    def get_simulation_time_s(self) -> float:
        """Native physics clock for synchronized capture and recording."""
        return float(self.handle.env.sim.data.time)

    def get_current_time_s(self) -> float:
        return self._sim_step_count / self._control_freq

    def task_completed(self) -> bool:
        # Prefer the success the env already computed in step() (cached on
        # _current_info). Re-invoking handle.env.check_success() RE-EVALUATES
        # the task predicate, which for stateful predicates is a side effect on
        # the sim: VAB's pack_all_into teleports every delivered object to a
        # graveyard pose each time it is evaluated. A loop that polls task
        # completion each iteration must therefore read the cached verdict, not
        # trigger a fresh evaluation. Fall back to a live check only when the
        # env surfaces no cached signal (non-LIBERO predicates).
        info = self._current_info or {}
        if "success" in info:
            return bool(info["success"])
        cr = info.get("completion_rate")
        if cr is not None:
            return float(cr) >= 1.0
        return self.handle.env.check_success()

    def completion_rate(self) -> float:
        """Fraction of items delivered (0..1), from the env's cached step info.
        This is the authoritative progress metric (the sparse ``compute_reward``
        only pays out at full completion). Falls back to the binary success flag
        when the predicate exposes no rate."""
        info = self._current_info or {}
        cr = info.get("completion_rate")
        if cr is not None:
            return float(cr)
        return 1.0 if info.get("success") else 0.0

    # ----------------------- Video Capture -----------------------
    #
    # Each captured frame is written to disk as its own PNG (into a
    # per-session temp dir) the moment it is rendered, instead of being
    # accumulated in an in-memory buffer. This keeps memory flat on long
    # rollouts and — crucially — means an interrupted or crashed run still
    # leaves every captured frame on disk to assemble. ``get_video_frames``
    # and ``save_video`` read the PNGs back from that dir.

    def enable_video_capture(
        self, enabled: bool = True, *, clear: bool = True
    ) -> None:
        self._record_frames = enabled
        if clear:
            self._reset_frames_dir()
        if enabled:
            self._record_frame()

    def _reset_frames_dir(self) -> None:
        """Start a fresh empty frames dir, removing any previous one."""
        import shutil
        import tempfile

        prev = self._frames_dir
        if prev and os.path.isdir(prev):
            shutil.rmtree(prev, ignore_errors=True)
        self._frames_dir = tempfile.mkdtemp(prefix="gap_video_frames_")
        self._frames_seq = 0

    def _frame_paths(self) -> list[str]:
        """Sorted PNG paths captured so far (empty if nothing recorded)."""
        d = self._frames_dir
        if not d or not os.path.isdir(d):
            return []
        return [
            os.path.join(d, f)
            for f in sorted(os.listdir(d))
            if f.startswith("frame_") and f.endswith(".png")
        ]

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        import imageio.v3 as iio

        frames = [np.asarray(iio.imread(p)) for p in self._frame_paths()]
        if clear:
            self._reset_frames_dir()
        return frames

    def save_video(
        self, output_path: str, *, fps: int = 20, clear: bool = False
    ) -> int:
        """Assemble the per-frame PNGs captured on disk into ``output_path``.

        Returns the number of frames written (0 when none were captured, in
        which case no file is created). 20 fps default pairs with
        ``_subsample_rate=4`` on the 20 Hz sim.
        """
        from pathlib import Path

        paths = self._frame_paths()
        if not paths:
            return 0
        import imageio.v3 as iio

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        stack = np.stack([np.asarray(iio.imread(p)) for p in paths])
        iio.imwrite(output_path, stack, fps=fps, codec="libx264")
        if clear:
            self._reset_frames_dir()
        return len(paths)

    def _record_frame(self) -> None:
        if not self._record_frames:
            return
        if self._frames_dir is None:
            self._reset_frames_dir()
        # MuJoCo offscreen render: ~tens of ms per call on GPU. Sim-only
        # work (a real robot uses real cameras), so attribute to
        # ``sim_physics_wall_s`` so compute_overhead doesn't double-count it.
        _t0 = time.perf_counter()
        frame = self.handle.env.sim.render(
            camera_name="agentview",
            width=self._render_width,
            height=self._render_height,
            depth=False,
        )
        self._sim_physics_wall_s += time.perf_counter() - _t0
        import imageio.v3 as iio

        assert self._frames_dir is not None
        iio.imwrite(
            os.path.join(self._frames_dir, f"frame_{self._frames_seq:06d}.png"),
            frame[::-1],
        )
        self._frames_seq += 1

    def render(
        self, mode: str = "rgb_array", *, camera_name: str = "agentview"
    ) -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError("Only rgb_array render mode is supported")
        _t0 = time.perf_counter()
        frame = self.handle.env.sim.render(
            camera_name=camera_name,
            width=self._render_width,
            height=self._render_height,
            depth=False,
        )
        self._sim_physics_wall_s += time.perf_counter() - _t0
        return frame[::-1]


# ---------------------------------------------------------------------------
# Registry factory
# ---------------------------------------------------------------------------

def make_env(
    suite_name: str,
    task_id: int,
    camera_names: list[str] | tuple[str, ...] | None = None,
    enable_render: bool = False,
    *,
    perturbed: bool | None = None,
    joint_motion_mode: str | None = None,
    **extra: Any,
) -> tuple[FrankaLiberoEnv, EnvConfig]:
    """Registry factory for the LIBERO suites (vab + classic benchmark).

    ``perturbed`` opts into the moving-basket variant
    (:class:`gap.envs.libero_perturbed_env.FrankaLiberoPerturbedEnv`);
    ``joint_motion_mode`` selects ``"closed_loop"`` (physics tracking,
    the default — physically faithful and safe for mixed policy/graph
    workflows) or ``"teleport"`` (fast qpos writes, fine for pure graph
    workflows). Both default to the ``GAP_LIBERO_PERTURBED``
    / ``GAP_LIBERO_JOINT_MOTION_MODE`` env vars so parallel workers can be
    configured per process without config plumbing. Remaining ``extra``
    kwargs (``max_steps``, ``seed``, ``control_freq``) pass through to the
    env constructor.
    """
    if joint_motion_mode is None:
        joint_motion_mode = env_config.libero_joint_motion_mode()
    if perturbed is None:
        perturbed = env_config.libero_perturbed()

    if perturbed:
        from .libero_perturbed_env import FrankaLiberoPerturbedEnv

        logger.info(
            "Creating perturbed env: %s task=%d (motion=%s)",
            suite_name, task_id, joint_motion_mode,
        )
        env_cls: type[FrankaLiberoEnv] = FrankaLiberoPerturbedEnv
    else:
        logger.info(
            "Creating env: %s task=%d (motion=%s)",
            suite_name, task_id, joint_motion_mode,
        )
        env_cls = FrankaLiberoEnv

    env = env_cls(
        suite_name=suite_name,
        task_id=task_id,
        enable_render=enable_render,
        camera_names=list(camera_names) if camera_names else None,
        joint_motion_mode=joint_motion_mode,
        **extra,
    )

    # Values lifted from the source sim bridge's Init libero branch:
    # velocity_joints action mode, 7-DOF single Franka, panda URDF, control
    # freq from the env; home_joints is the canonical Franka home the bridge
    # fell back to for GoHome; tcp_offset is its default panda fingertip
    # offset (-0.1 m along hand Z).
    config = EnvConfig(
        arm_dof=7,
        num_arms=1,
        action_mode="velocity_joints",
        control_freq=float(getattr(env, "_control_freq", 20.0)),
        home_joints=_FRANKA_HOME_JOINTS,
        # Ground truth measured from MuJoCo at home pose: hand
        # ``robot0_right_hand`` -> site ``gripper0_grip_site`` is +0.0970 m
        # along hand z (robosuite's canonical TCP). gap stores the negated
        # value (PyRoKi convention). The historical -0.1 was 3 mm too long
        # and the curobo bundle's 0.1029 comment is 5.9 mm too long.
        tcp_offset=(0.0, 0.0, -0.097),
        tcp_rotation_z=None,
        arm_bases=None,
        robot_urdf_path="panda_description",
        default_cameras=tuple(env.camera_names),
        is_real=False,
    )
    return env, config
