"""FrankaLiberoPerturbedEnv — LIBERO env with a scripted basket trajectory.

Subclass of :class:`FrankaLiberoEnv` that teleports the ``basket_1`` body along
a hard-coded trajectory every sim tick. Used as the target for the
moving-basket demo.

The base env is left untouched so the existing LIBERO benchmark path keeps
its exact previous behavior; this class is opted in by setting
``GAP_LIBERO_PERTURBED=1`` (or passing ``perturbed=True`` to
``gap.envs.libero_env.make_env``).

Trajectory parameters are class-level constants — edit them here to change
the motion. No external config plumbing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .libero_env import FrankaLiberoEnv

# --- Hard-coded trajectory ---------------------------------------------------
# kind: "circular" | "linear"
#  circular: radius (m), period_s (s)
#  linear:   velocity (3-vec, m/s), duration_s (s, clamps t)
#  target_bddl_name: BDDL object name to teleport (default basket_1).
_BASKET_MOTION: dict[str, Any] = {
    "kind": "circular",
    "radius": 0.05,
    "period_s": 8.0,
    "target_bddl_name": "basket_1",
}


class FrankaLiberoPerturbedEnv(FrankaLiberoEnv):
    """FrankaLiberoEnv variant with a scripted basket trajectory.

    The trajectory is the module-level ``_BASKET_MOTION`` constant; edit
    that to change the motion. Args are passed through to the base class.
    """

    def __init__(
        self,
        suite_name: str,
        task_id: int,
        **kwargs: Any,
    ) -> None:
        self._basket_motion = dict(_BASKET_MOTION)
        self._basket_qpos_addr: int | None = None
        self._basket_initial_pose: np.ndarray | None = None
        super().__init__(suite_name=suite_name, task_id=task_id, **kwargs)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # Run the standard reset + 40-step settle; THEN cache the basket's
        # rest pose so the trajectory is expressed as a delta from that pose.
        obs, info = super().reset(seed=seed, options=options)
        if self._basket_motion is not None:
            self._resolve_basket_qpos()
        return obs, info

    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        self._current_obs, self._current_reward, self._current_done, self._current_info = (
            self.handle.step(action)
        )
        self._sim_step_count += 1
        # Teleport the basket BEFORE rendering so cameras see the new pose.
        self._apply_basket_motion()
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
        return obs, reward, terminated, truncated, {}

    def move_to_joints_blocking(
        self,
        joints: np.ndarray,
        *,
        tolerance: float = 0.01,
        max_steps: int = 120,
        arm_id: int = 0,
    ) -> None:
        target = np.asarray(joints, dtype=np.float64).reshape(7)
        self._current_joints = target

        steps = 0
        while steps < max_steps:
            current = np.array(
                self.handle.env.sim.data.qpos[self._panda_joint_qpos_addrs],
                dtype=np.float64,
            )

            error = np.linalg.norm(current - target)
            if error < tolerance and steps > 0:
                break

            delta = (target - current) * self._control_freq
            action = np.concatenate([delta, [self._gripper_fraction]])
            action[-1] = 1.0 - action[-1] * 2.0

            self._current_obs, self._current_reward, self._current_done, self._current_info = (
                self.handle.step(action)
            )
            self._sim_step_count += 1
            self._apply_basket_motion()

            self.gripper_link_wxyz_xyz = np.concatenate([
                self.handle.env.sim.data.xquat[self.gripper_link_idx],
                self.handle.env.sim.data.xpos[self.gripper_link_idx],
            ])

            if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
                self._record_frame()

            steps += 1

    # ------------------------------------------------------------------
    # Perturbation helpers
    # ------------------------------------------------------------------

    def _resolve_basket_qpos(self) -> None:
        cfg = self._basket_motion
        if cfg is None:
            return
        bddl_name = str(cfg.get("target_bddl_name", "basket_1"))
        # ``obj_body_id`` lives on the inner Libero_*_Manipulation domain,
        # which OffScreenRenderEnv wraps; fall through both for robustness.
        domain = getattr(self.handle.env, "env", self.handle.env)
        body_id_map = getattr(domain, "obj_body_id", None)
        if not body_id_map or bddl_name not in body_id_map:
            raise RuntimeError(
                f"basket_motion: BDDL object '{bddl_name}' not found in "
                f"obj_body_id (have: {list(body_id_map.keys()) if body_id_map else 'none'})"
            )
        body_id = int(body_id_map[bddl_name])
        sim_model = self.handle.env.sim.model
        joint_id = int(sim_model.body_jntadr[body_id])
        if joint_id < 0:
            raise RuntimeError(
                f"basket_motion: body '{bddl_name}' has no joint (jntadr<0)"
            )
        qpos_addr = int(sim_model.jnt_qposadr[joint_id])
        self._basket_qpos_addr = qpos_addr
        self._basket_initial_pose = np.array(
            self.handle.env.sim.data.qpos[qpos_addr:qpos_addr + 7], dtype=np.float64
        ).copy()

    def _apply_basket_motion(self) -> None:
        cfg = self._basket_motion
        addr = self._basket_qpos_addr
        if cfg is None or addr is None or self._basket_initial_pose is None:
            return
        t = self.get_current_time_s()
        kind = str(cfg.get("kind", "circular"))
        x0, y0, z0 = self._basket_initial_pose[:3]
        qw, qx, qy, qz = self._basket_initial_pose[3:7]

        if kind == "circular":
            r = float(cfg.get("radius", 0.05))
            T = float(cfg.get("period_s", 8.0))
            phase = 2.0 * np.pi * t / max(T, 1e-3)
            x = x0 + r * float(np.cos(phase)) - r  # start at the rest position
            y = y0 + r * float(np.sin(phase))
            z = z0
        elif kind == "linear":
            v = np.asarray(cfg.get("velocity", [0.01, 0.0, 0.0]), dtype=float)
            duration = float(cfg.get("duration_s", 8.0))
            scale = min(t, duration)
            x = x0 + float(v[0]) * scale
            y = y0 + float(v[1]) * scale
            z = z0 + float(v[2]) * scale
        else:
            return  # unknown kind → no-op

        sim = self.handle.env.sim
        sim.data.qpos[addr:addr + 3] = [x, y, z]
        sim.data.qpos[addr + 3:addr + 7] = [qw, qx, qy, qz]
        # Cancel residual velocity to avoid the integrator fighting the teleport.
        sim.data.qvel[addr:addr + 6] = 0.0
        sim.forward()  # propagate FK → xpos / xquat / camera matrices
