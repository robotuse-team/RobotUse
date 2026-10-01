"""SimConnector — simulation backends (LIBERO et al.) + the ``sim()`` factory.

Absorbs the ``SimBridgeServicer`` method bodies (Reset / StepOnce /
CheckTaskCompletion / EnableVideoCapture / SaveVideo / GetState's
ground-truth object poses / ApplyPolicyAction) on top of the shared
:class:`gap.connector.core.Connector` robot-control logic.

    import gap
    conn = gap.connector.sim("libero", task="libero_object/0")
    result = gap.execute(graph, conn)   # open-robot-skills auto-discovered
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from gap_core.errors import ToolError
from gap_core.tools import ToolRegistry
from gap_core.types import Observation, make_pose

from gap.connector.core import Capabilities, Connector

logger = logging.getLogger(__name__)


def _parse_task(task: Any) -> tuple[str, int]:
    """Parse ``"suite/task_id"`` (or ``(suite, int)``) into components."""
    if isinstance(task, (tuple, list)) and len(task) == 2:
        return str(task[0]), int(task[1])
    text = str(task)
    if "/" in text:
        suite, _, tid = text.rpartition("/")
        return suite, int(tid)
    return text, 0


class SimConnector(Connector):
    """Connector over an in-process simulation environment."""

    def __init__(
        self,
        env: Any,
        config: Any,
        *,
        camera_names: list[str] | None = None,
        ik: Any | None = None,
        record_video: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__(env, config, camera_names=camera_names, ik=ik)
        self._record_video = bool(record_video)
        self._pending_seed = seed
        self._world_adapter: Any | None = None

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> Capabilities:
        env = self.env
        return Capabilities(
            reset=hasattr(env, "reset"),
            success_check=hasattr(env, "task_completed"),
            video=hasattr(env, "enable_video_capture"),
            world_state=self._supports_world_state(),
        )

    def _supports_world_state(self) -> bool:
        """Ground-truth world snapshots need a reachable MuJoCo sim."""
        try:
            from gap.connector.world_adapter import find_mujoco_sim

            return find_mujoco_sim(self.env) is not None
        except Exception:
            return False

    # ------------------------------------------------------------------
    # sim.* tools
    # ------------------------------------------------------------------

    def _register_extra_tools(self, reg: ToolRegistry) -> None:
        rc = reg.register_callable
        rc("sim.reset", self._tool_reset,
           summary="Reset the simulation to its initial state.")
        rc("sim.step", self._tool_step,
           summary="Advance the sim one control step holding pose + gripper.",
           tags=("sim_step",))
        rc("sim.check_success", self._tool_check_success,
           summary="Check task completion; returns task_completed + reward.")
        rc("sim.apply_policy_action", self._tool_apply_policy_action,
           summary="Forward one low-level policy action to the env controller.",
           tags=("sim_step",))
        rc("sim.enable_video", self._tool_enable_video,
           summary="Enable or disable in-sim video frame capture.")
        rc("sim.save_video", self.save_video,
           summary="Encode buffered video frames to an mp4 file.")

    # ------------------------------------------------------------------
    # Reset / step / success (absorbed SimBridgeServicer)
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> Observation:
        """Reset the env; returns the initial Observation.

        ``sim(seed=...)`` stashes a pending seed consumed by the first
        no-arg reset (one reset, not two).
        """
        if seed is None:
            seed, self._pending_seed = self._pending_seed, None
        obs, _info = self.env.reset(seed=seed)
        self._gripper_fraction = 1.0
        # AABBs are recomputed per reset (hard reset rebuilds the MjModel).
        if self._world_adapter is not None:
            try:
                self._world_adapter.refresh()
            except Exception:
                logger.debug("world adapter refresh failed", exc_info=True)
        if self._record_video:
            self.start_video()
        return self._build_observation(obs)

    def _tool_reset(self, seed: int = 0) -> dict:
        """sim.reset tool — source semantics: seed 0 means 'unseeded'."""
        return {"observation": self.reset(seed=seed if seed != 0 else None)}

    def _tool_step(self, gripper_fraction: float = -1.0) -> dict:
        """One held-pose sim step; optionally set the gripper first."""
        if gripper_fraction >= 0:
            self.set_gripper(gripper_fraction)
        obs, reward, terminated, truncated, _info = self.step_once()
        return {
            "observation": self._build_observation(obs),
            "reward": float(reward),
            "terminated": terminated,
            "truncated": truncated,
        }

    def check_success(self) -> tuple[bool, float]:
        """Return ``(task_completed, reward)`` from the env's own checker."""
        reward = float(self.env.compute_reward())
        return bool(self.env.task_completed()), reward

    def get_latency_info(self) -> dict[str, Any]:
        """Forward the env's cumulative episode latency, if exposed.

        ``{}`` when the underlying env doesn't implement
        :meth:`get_latency_info` (real-robot / UR-Zed envs today). The
        ``gap run`` and ``gap bench`` printouts gate on the dict being
        non-empty before reporting physical-execution time.
        """
        fn = getattr(self.env, "get_latency_info", None)
        return dict(fn()) if fn is not None else {}

    def _tool_check_success(self) -> dict:
        completed, reward = self.check_success()
        out = {"task_completed": completed, "reward": reward}
        cr_fn = getattr(self.env, "completion_rate", None)
        if cr_fn is not None:
            try:
                out["completion_rate"] = float(cr_fn())
            except Exception:
                pass
        return out

    def _tool_apply_policy_action(self, action: list[float]) -> None:
        """Forward a single low-level action to the env's controller.

        Backends without a passthrough controller return the source's
        FAILED_PRECONDITION error so the caller surfaces a clear error
        rather than driving the robot with the wrong action space.
        """
        action = list(action or [])
        if not action:
            raise ToolError(
                "sim.apply_policy_action", "ApplyPolicyAction: 'action' field is required"
            )
        env = self.env
        applier = getattr(env, "apply_policy_action", None)
        if applier is None:
            raise ToolError(
                "sim.apply_policy_action",
                f"ApplyPolicyAction not supported by env "
                f"{type(env).__name__}; the env class must expose an "
                f"apply_policy_action(action) method (currently only "
                f"FrankaLiberoEnv does).",
            )
        arr = np.asarray(action, dtype=np.float64)
        try:
            applier(arr)
        except Exception as exc:
            logger.exception("apply_policy_action failed")
            raise ToolError(
                "sim.apply_policy_action", f"apply_policy_action failed: {exc}"
            ) from exc
        try:
            obs = env.get_observation()
        except Exception:
            obs = {}
        self._emit_step(
            arr,
            obs,
            float(env.compute_reward() or 0.0),
            bool(getattr(env, "_current_done", False)),
        )

    # ------------------------------------------------------------------
    # Ground-truth state / world snapshots
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Ground-truth sim state (absorbed SimBridge.GetState).

        Returns ``{"arm_states": [...], "objects": {name: Se3Pose}}`` built
        from the env obs dict; object poses come from the env's
        ``cube_poses`` entry when the env surfaces it.
        """
        obs = self.env.get_observation()
        dof = self._arm_dof

        objects: dict[str, Any] = {}
        for name, pose_data in obs.get("cube_poses", {}).items():
            pose_arr = np.asarray(pose_data, dtype=np.float64)
            objects[name] = make_pose(pose_arr[:3], pose_arr[3:7])

        arm_states = []
        for arm_id in range(self.num_arms):
            joint_pos = np.asarray(obs.get(f"robot_joint_pos_{arm_id}", np.zeros(dof + 1)))
            cart = np.asarray(obs.get(f"robot_cartesian_pos_{arm_id}", np.zeros(8)))
            ee_pos = cart[:3].copy()
            ee_quat = cart[3:7] if len(cart) >= 7 else np.array([1.0, 0.0, 0.0, 0.0])
            ee_pos = self._base_frame_to_world(ee_pos, arm_id)
            arm_states.append({
                "joint_state": {"positions": joint_pos[:dof].astype(np.float64)},
                "gripper_fraction": (
                    float(joint_pos[dof]) if len(joint_pos) > dof else self._gripper_fraction
                ),
                "ee_pose": make_pose(ee_pos, ee_quat),
            })

        return {"arm_states": arm_states, "objects": objects}

    def world_snapshot(self):
        """Build a :class:`gap.runtime.verify.World` from sim ground truth."""
        if self._world_adapter is None:
            from gap.connector.world_adapter import LiberoWorldAdapter

            self._world_adapter = LiberoWorldAdapter(self.env)
        return self._world_adapter.snapshot()

    # ------------------------------------------------------------------
    # Video (absorbed EnableVideoCapture / SaveVideo)
    # ------------------------------------------------------------------

    def start_video(self, *, clear: bool = True) -> None:
        self._tool_enable_video(enabled=True, clear=clear)

    def _tool_enable_video(self, enabled: bool = True, clear: bool = True) -> None:
        if hasattr(self.env, "enable_video_capture"):
            self.env.enable_video_capture(enabled, clear=clear)
        else:
            logger.warning("Environment does not support enable_video_capture")

    def save_video(self, output_path: str, fps: int = 20, clear: bool = False) -> dict:
        """Encode buffered frames into ``output_path`` (libx264).

        Also writes the env's extra per-camera buffers as
        ``<stem>_<cam>.mp4`` next to the main video.
        """
        try:
            fps = fps if fps > 0 else 20

            if hasattr(self.env, "get_video_frames"):
                frames = self.env.get_video_frames(clear=clear)
            else:
                logger.warning("Environment does not support get_video_frames")
                return {"success": False, "file_path": output_path, "num_frames": 0}

            if not frames or len(frames) == 0:
                return {"success": True, "file_path": output_path, "num_frames": 0}

            import imageio.v3 as iio

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(output_path, np.stack(frames), fps=fps, codec="libx264")

            # Extra per-camera videos, captured at the same cadence as the
            # main buffer. Written as <stem>_<cam>.mp4 next to it.
            if hasattr(self.env, "get_extra_video_frames"):
                out_dir = Path(output_path).parent
                stem = Path(output_path).stem
                extra = self.env.get_extra_video_frames(clear=clear)
                for cam, cam_frames in extra.items():
                    if not cam_frames:
                        continue
                    cam_path = out_dir / f"{stem}_{cam}.mp4"
                    iio.imwrite(
                        str(cam_path), np.stack(cam_frames),
                        fps=fps, codec="libx264",
                    )
                    logger.info(
                        "save_video: wrote %s (%d frames)", cam_path, len(cam_frames),
                    )

            return {"success": True, "file_path": output_path, "num_frames": len(frames)}
        except Exception:
            logger.error("save_video failed", exc_info=True)
            return {"success": False, "file_path": output_path, "num_frames": 0}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def sim(
    env: str = "libero",
    *,
    task: Any = "libero_object/0",
    cameras: list[str] | None = None,
    headless: bool = True,
    seed: int | None = None,
    record_video: bool = False,
    **env_kwargs: Any,
) -> SimConnector:
    """Build a :class:`SimConnector` for a registered simulation backend.

    Args:
        env: Registry name (``"libero"``, ...) resolved via
            :func:`gap.envs.registry.resolve`.
        task: ``"suite/task_id"`` (e.g. ``"libero_object/0"``) or a
            ``(suite, task_id)`` tuple.
        cameras: Camera-name override (default: the env's
            ``EnvConfig.default_cameras``).
        headless: When False, enable on-screen/offline rendering even
            without video recording.
        seed: Seed applied by the connector's first ``reset()``.
        record_video: Construct with render enabled and start frame capture
            on reset.
        **env_kwargs: Extra keyword arguments forwarded to the env factory.
    """
    try:
        from gap.envs.registry import resolve
    except ImportError as e:
        raise ImportError(
            "gap.envs is not available — the simulation env layer is required "
            "for gap.connector.sim(). Install the [libero] extra and ensure "
            "gap.envs is importable."
        ) from e

    factory, key = resolve(env)
    if isinstance(task, int):
        suite_name, task_id = key, int(task)
    else:
        suite_name, task_id = _parse_task(task)
    enable_render = bool(record_video or not headless)
    env_obj, config = factory(
        suite_name,
        task_id,
        camera_names=cameras,
        enable_render=enable_render,
        **env_kwargs,
    )
    logger.info(
        "sim connector: env=%s key=%s suite=%s task=%d render=%s",
        env, key, suite_name, task_id, enable_render,
    )
    return SimConnector(
        env_obj,
        config,
        camera_names=cameras,
        record_video=record_video,
        seed=seed,
    )


__all__ = ["SimConnector", "sim"]
