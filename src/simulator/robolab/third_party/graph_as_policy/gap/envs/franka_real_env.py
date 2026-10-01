"""FrankaRealEnv — real Franka robot environment via robots_realtime.

Implements the msgpack TCP protocol spoken by the robots_realtime client
(vendored at ``third_party/robots_realtime``).

Architecture:
  - MsgpackNumpyServer listens on port 9000 for the robots_realtime client
  - robots_realtime sends observations (joints, camera) and receives actions
  - This env translates wire-format observations to the gap env obs format
  - Control is absolute joint targets republished at 50 Hz

The companion rr-session process (``gap.connector.rr_launcher.RRSession``
spawns it, or run it manually with ``uv run --directory
third_party/robots_realtime rr-session configs/franka/franka_robotiq_client.yaml``)
owns the realtime hardware loops and connects back to this server.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any

import numpy as np

from gap.envs.base_env import BaseEnv
from gap.envs.msgpack_bridge import MsgpackNumpyServer, start_server_in_background

logger = logging.getLogger(__name__)

# Franka Panda home joint configuration (radians)
_FRANKA_HOME_JOINTS = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)


class FrankaRealEnv(BaseEnv):
    """Real Franka environment connected via the robots_realtime msgpack bridge.

    - MsgpackNumpyServer on configurable port (default 9000)
    - Observation conversion from wire format to the gap obs format
    - Joint-position control with convergence monitoring
    - Gripper control with hardware settle timing
    """

    # TCP offset from panda_hand to Robotiq fingertip (meters, in EE frame)
    _TCP_OFFSET = np.array([0.0, 0.0, -0.157], dtype=np.float64)
    # Robotiq mounting rotation relative to panda_hand (pi/4 around Z)
    _TCP_ROTATION_Z = np.pi / 4

    def __init__(
        self,
        camera_names: list[str] | None = None,
        port: int = 9000,
        host: str = "127.0.0.1",
        heartbeat_period: float = 1.0,
        stale_after_s: float = 2.0,
    ) -> None:
        super().__init__()

        # Camera name mapping: robots_realtime names the camera node freely
        # in the session yaml; gap workflows expect configurable camera names
        self.camera_names = camera_names or ["robot0_robotview"]

        # Robot state
        self._current_joints = np.array(
            [0.0256, -0.5002, -0.0217, -2.374, -0.0109, 1.874, -2.346],
            dtype=np.float64,
        )
        self._gripper_fraction = 1.0  # open
        self._action_publish_period = 0.02  # 50 Hz

        # Msgpack server for robots_realtime communication.
        # Default is loopback-only. Pass the robot-LAN IP (e.g. "172.16.0.4")
        # when the Franka controller runs on a separate machine.
        # Construct first, pre-seed ``latest_action`` with a hold-home command,
        # then start serving. Without the seed the very first realtime
        # ``send_request`` returns ``{}`` and the client falls back to its
        # Viser IK gizmo, jolting the arm before the workflow has commanded
        # anything.
        self.server = MsgpackNumpyServer(host=host, port=port)
        self.server.latest_action = {
            "timestamp": time.time(),
            "left": {
                "joint_pos": self._current_joints.astype(np.float32).tolist(),
                "gripper": float(self._gripper_fraction),
            },
        }
        start_server_in_background(self.server)

        # Cached observations
        self._obs: dict[str, Any] = {}

        # Step tracking
        self._sim_step_count = 0
        self.max_steps = 999999

        # Video capture
        self._record_frames = False
        self._frame_buffer: list[np.ndarray] = []

        # URDF for forward kinematics (ee pose from joints)
        from robot_descriptions.loaders.yourdfpy import load_robot_description

        self._urdf = load_robot_description("panda_description")

        # Streaming target slot — sole source of truth for the wire command.
        # ``move_to_joints_blocking`` writes ``_target_joints`` under the lock;
        # the background republisher is the **only** writer of
        # ``self.server.latest_action``. Earlier code wrote to ``latest_action``
        # from both sites, which raced against ``MsgpackNumpyServer._handle``'s
        # send_framed and caused brief stale-command frames after a few seconds
        # of correct tracking.
        self._target_lock = threading.Lock()
        # Seed the target with the home pose so the republisher emits valid
        # actions from t=0; otherwise the realtime client's first response
        # frames are empty and it falls back to the Viser IK gizmo.
        self._target_joints: np.ndarray | None = self._current_joints.copy()
        self._target_gripper: float = self._gripper_fraction

        # Background republisher: at 50 Hz, re-stamp the current target into
        # `self.server.latest_action` so the realtime controller always sees
        # a fresh-timestamp command between servo ticks (~33 ms gaps).
        self._republish_period = self._action_publish_period
        self._republish_stop = threading.Event()
        # Counters/state for the heartbeat (read by ``_heartbeat_loop``):
        self._republish_count = 0
        self._target_update_count = 0
        self._last_target_update_t = time.time()
        self._republish_thread = threading.Thread(
            target=self._republish_loop, daemon=True,
        )
        self._republish_thread.start()

        # Heartbeat: every ``heartbeat_period`` seconds, log a one-line
        # status snapshot. Catches:
        #   - realtime-side hang: ``obs_age`` keeps growing
        #   - real-robot stuck: ``obs_age`` stays small but ``joint_max_diff``
        #     stays near 0 (joints frozen on hardware despite fresh wire frames)
        #   - gap-side hang: ``republish_hz`` drops to 0 (republisher died) or
        #     ``target_age`` grows large (move_to_joints stopped being called)
        # The heartbeat says exactly which side stopped.
        #
        # ``obs_stale`` flips to True when no fresh wire observation has
        # arrived for ``stale_after_s`` (client silent / disconnected); it
        # clears again on the next fresh frame.
        self._heartbeat_period = float(heartbeat_period)
        self._stale_after_s = float(stale_after_s)
        self.obs_stale = False
        self.last_heartbeat: dict[str, float] = {}
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True,
        )
        self._heartbeat_thread.start()

    def _republish_loop(self) -> None:
        try:
            while not self._republish_stop.is_set():
                with self._target_lock:
                    target = self._target_joints
                    gripper = self._target_gripper
                if target is not None:
                    self.server.latest_action = {
                        "timestamp": time.time(),
                        "left": {
                            "joint_pos": target.astype(np.float32).tolist(),
                            "gripper": float(gripper),
                        },
                    }
                    self._republish_count += 1
                time.sleep(self._republish_period)
        except Exception:
            # A silent thread death here would freeze the robot at the last
            # action, since ``latest_action`` would stop updating. Make it
            # loud — the heartbeat would also catch this via republish_hz=0
            # but we want a clear stack trace too.
            logger.exception("[FrankaRealEnv] REPUBLISHER CRASHED")

    def _heartbeat_loop(self) -> None:
        last_count = 0
        last_t = time.time()
        last_observed_joints: np.ndarray | None = None
        while not self._republish_stop.is_set():
            time.sleep(self._heartbeat_period)
            now = time.time()
            dt = max(now - last_t, 1e-6)

            # Republisher rate (target ~50 Hz). Anything <30 Hz is a problem.
            republish_hz = (self._republish_count - last_count) / dt
            last_count = self._republish_count
            last_t = now

            # Wire-observation age and joint motion since last heartbeat.
            wire = self.server.latest_observation
            ts_wire = None
            joint_max_diff = 0.0
            current_joints = None
            if isinstance(wire, dict):
                ts_wire = self._g(wire, "timestamp")
                arm = self._g(wire, "left")
                if isinstance(arm, dict):
                    raw = self._g(arm, "joint_pos")
                    if raw is not None:
                        current_joints = np.asarray(raw, dtype=np.float32)
                        if (
                            last_observed_joints is not None
                            and current_joints.shape == last_observed_joints.shape
                        ):
                            joint_max_diff = float(
                                np.abs(current_joints - last_observed_joints).max()
                            )
                        last_observed_joints = current_joints

            obs_age = (
                (now - float(ts_wire))
                if isinstance(ts_wire, (int, float))
                else float("inf")
            )
            target_age = now - self._last_target_update_t
            with self._target_lock:
                target = self._target_joints
                cmd_vs_obs = (
                    float(
                        np.abs(
                            target.astype(np.float64)
                            - current_joints[:7].astype(np.float64)
                        ).max()
                    )
                    if (
                        target is not None
                        and current_joints is not None
                        and current_joints.size >= 7
                    )
                    else float("nan")
                )

            self.obs_stale = obs_age > self._stale_after_s
            self.last_heartbeat = {
                "republish_hz": republish_hz,
                "obs_age": obs_age,
                "joint_max_diff": joint_max_diff,
                "target_age": target_age,
                "cmd_vs_obs": cmd_vs_obs,
            }

            log = logger.warning if self.obs_stale else logger.info
            log(
                "[FrankaRealEnv hb] republish_hz=%.1f obs_age=%.0fms "
                "joint_max_diff=%.4f target_age=%.0fms cmd_vs_obs=%.4f%s",
                republish_hz, obs_age * 1000, joint_max_diff,
                target_age * 1000, cmd_vs_obs,
                " STALE (no fresh wire observation — rr-session silent?)"
                if self.obs_stale else "",
            )

    # ------------------------------------------------------------------
    # Observation handling
    # ------------------------------------------------------------------

    def _update_from_network(self) -> None:
        """Pull latest observation from msgpack server and convert."""
        msg = self.server.latest_observation
        if msg is None:
            return

        new_obs = self._convert_observation(msg)

        # Always update joints; only overwrite camera data when fresh RGB arrives
        for k, v in new_obs.items():
            if k in self.camera_names:
                if k not in self._obs:
                    self._obs[k] = v
                elif v.get("images", {}).get("rgb") is not None:
                    self._obs[k] = v
            else:
                self._obs[k] = v

    @staticmethod
    def _g(d: dict, key: str) -> Any:
        """Get from dict trying both string and byte-string keys (msgpack compat)."""
        return d.get(key) if key in d else d.get(key.encode())

    # Top-level wire keys that are NOT camera subdicts (joints + bookkeeping).
    # Anything else that's a dict and carries an ``images`` (or ``depth_data``)
    # field is treated as the camera payload.
    _NON_CAMERA_KEYS = frozenset({
        "left", "right", "timestamp", "left_arm", "right_arm",
    })

    @classmethod
    def _find_camera_subdict(cls, msg: dict) -> dict | None:
        """Return the first dict-valued subdict that looks camera-shaped.

        rr-session names the camera node freely in the session yaml
        (``camera_top``, ``zed``, ``cam0``, …) — the wire reflects that
        name verbatim. Pick whichever non-arm subkey carries ``images``
        or ``depth_data``; ignore arm and bookkeeping keys.
        """
        for k, v in msg.items():
            key = k.decode() if isinstance(k, bytes) else str(k)
            if key in cls._NON_CAMERA_KEYS:
                continue
            if not isinstance(v, dict):
                continue
            if cls._g(v, "images") is not None or cls._g(v, "depth_data") is not None:
                return v
        return None

    def _convert_observation(self, msg: dict) -> dict[str, Any]:
        """Convert wire-format msgpack observation to the gap obs format.

        Wire format (byte-string keys from robots_realtime):
            msg[b'left'][b'joint_pos'] -> 8 floats (7 joints + gripper)
            msg[b'camera_top'][b'images'][b'left_rgb'] -> RGB
            msg[b'camera_top'][b'depth_data'] -> depth
            msg[b'camera_top'][b'intrinsics'][b'left'][b'intrinsics_matrix'] -> K
            msg[b'camera_top'][b'pose_mat'] -> 4x4 SE(3)

        gap format (string keys):
            obs["robot_joint_pos_0"] -> np.float32(8,)
            obs["<camera_name>"]["images"]["rgb"] -> np.ndarray(H,W,3)
            obs["<camera_name>"]["images"]["depth"] -> np.ndarray(H,W)
            obs["<camera_name>"]["intrinsics"] -> np.float32(3,3)
            obs["<camera_name>"]["pose"] -> [x,y,z, w,x,y,z]
            obs["robot_cartesian_pos_0"] -> [x,y,z, w,x,y,z, gripper_frac]
        """
        import viser.transforms as vtf

        obs: dict[str, Any] = {}
        _g = self._g

        # DEBUG: log keys on first observation
        if not self._obs:
            logger.info(
                "[FrankaRealEnv] First observation keys: %s", list(msg.keys())
            )
            for k, v in msg.items():
                if isinstance(v, dict):
                    logger.info("  [%s] sub-keys: %s", k, list(v.keys()))

        # --- Joints (8 elements: 7 arm + gripper) ---
        arm_data = _g(msg, "left")
        if arm_data is not None:
            joint_pos_raw = _g(arm_data, "joint_pos")
            if joint_pos_raw is not None:
                joint_pos = np.asarray(joint_pos_raw, dtype=np.float32)
                obs["robot_joint_pos_0"] = joint_pos

                # --- End-effector pose via FK ---
                joints_7 = joint_pos[:7].astype(np.float64)
                gripper_frac = (
                    float(joint_pos[7]) if len(joint_pos) > 7 else self._gripper_fraction
                )
                ee_pos, ee_quat_wxyz = self._compute_fk(joints_7)
                obs["robot_cartesian_pos_0"] = np.concatenate([
                    ee_pos, ee_quat_wxyz, [gripper_frac]
                ]).astype(np.float32)

        # --- Camera ---
        # The realtime client rate-limits camera fields (see
        # franka_osc_client_cartesian._strip_cameras_unless_due) — most
        # ticks omit the camera subdict entirely. The wire key for the
        # camera is whatever the rr-session yaml named the CameraNode
        # (e.g. ``camera_top``, ``zed``, etc.); auto-discover it by
        # finding the first dict-valued subkey that carries an ``images``
        # field. Only emit a fresh ``obs[cam_name]`` when the wire
        # actually carries an RGB payload; otherwise leave it absent so
        # ``_update_from_network`` keeps the previous good frame.
        cam = self._find_camera_subdict(msg)
        cam_name = self.camera_names[0] if self.camera_names else "robot0_robotview"
        if cam is None:
            return obs

        cam_data: dict[str, Any] = {"images": {}}
        images = _g(cam, "images") or {}
        # Camera frames may be keyed as ``left_rgb`` (mono ZED), ``rgb``
        # (concat-stereo or generic OpenCV), or even ``right_rgb``
        # (right-only mode). Try each in order and use whichever arrives.
        rgb = _g(images, "left_rgb")
        if rgb is None:
            rgb = _g(images, "rgb")
        if rgb is None:
            rgb = _g(images, "right_rgb")
        if rgb is not None:
            cam_data["images"]["rgb"] = np.asarray(rgb)

        depth = _g(cam, "depth_data")
        if depth is not None:
            cam_data["images"]["depth"] = np.asarray(depth)

        intrinsics = _g(cam, "intrinsics")
        if intrinsics is not None:
            left_intr = _g(intrinsics, "left") or {}
            mat = _g(left_intr, "intrinsics_matrix")
            if mat is not None:
                cam_data["intrinsics"] = np.asarray(mat, dtype=np.float32)

        # Camera pose — derive from pose_mat (4x4) for unambiguous convention
        pose_mat = _g(cam, "pose_mat")
        if pose_mat is not None:
            pose_mat = np.asarray(pose_mat, dtype=np.float32)
            cam_data["pose_mat"] = pose_mat
            se3 = vtf.SE3.from_matrix(pose_mat)
            cam_data["pose"] = np.concatenate([
                se3.translation(),
                se3.rotation().wxyz,
            ]).astype(np.float32)
        else:
            raw_pose = _g(cam, "pose")
            if raw_pose is not None:
                cam_data["pose"] = np.asarray(raw_pose, dtype=np.float32)

        # Only publish the camera entry if RGB actually arrived this tick
        # (intrinsics-only / pose-only payloads aren't useful to consumers).
        if cam_data["images"].get("rgb") is None:
            return obs
        obs[cam_name] = cam_data

        return obs

    def _compute_fk(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute tool-tip pose from 7 joint values using URDF FK + TCP offset.

        Computes panda_hand pose, then applies the Robotiq TCP offset and
        mounting rotation to report the actual fingertip position/orientation.

        Returns:
            (position(3,), quaternion_wxyz(4,))
        """
        import viser.transforms as vtf

        # Build joint config dict for yourdfpy
        joint_names = [f"panda_joint{i + 1}" for i in range(7)]
        cfg = dict(zip(joint_names, joints.tolist(), strict=True))
        self._urdf.update_cfg(cfg)

        # Get panda_hand link transform (base frame is panda_link0, not "world")
        link_tf = self._urdf.get_transform("panda_hand", "panda_link0")  # 4x4
        se3 = vtf.SE3.from_matrix(link_tf)

        # Apply TCP rotation (Robotiq mounting) and offset
        R_link = se3.rotation()
        R_tcp = vtf.SO3.from_rpy_radians(0.0, 0.0, self._TCP_ROTATION_Z)
        R_tool = R_link @ R_tcp

        # TCP offset is in the panda_hand frame (before tool rotation)
        tip_pos = np.asarray(se3.translation(), dtype=np.float64) + \
            R_link.as_matrix() @ self._TCP_OFFSET

        return (
            tip_pos,
            np.asarray(R_tool.wxyz, dtype=np.float64),
        )

    # ------------------------------------------------------------------
    # BaseEnv interface
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Block until first RGB observation arrives from robots_realtime."""
        cam_name = self.camera_names[0] if self.camera_names else "robot0_robotview"
        while True:
            self._update_from_network()
            cam_obs = self._obs.get(cam_name, {})
            if cam_obs.get("images", {}).get("rgb") is not None:
                break
            logger.info(
                "Waiting for observation from real environment... %s",
                self._diagnose_missing_rgb(cam_name),
            )
            time.sleep(1.0)

        self._sim_step_count = 0
        return self._obs, {}

    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Low-level step — not typically called directly."""
        self._sim_step_count += 1
        obs = self.get_observation()
        return obs, 0.0, False, False, {}

    def get_observation(self) -> dict[str, Any]:
        """Get latest observation from robots_realtime."""
        self._update_from_network()
        return self._obs

    def _diagnose_missing_rgb(self, cam_name: str) -> str:
        """Pinpoint where the camera path is breaking while ``reset`` waits.

        Reports which stage is empty: no wire connection, no
        camera-shaped subdict (rr-session CameraNode not publishing or
        rate-limited), wire payload missing ``images``/rgb keys, or RGB
        on wire but env not picking it up.
        """
        msg = self.server.latest_observation
        if msg is None:
            return "(latest_observation is None — rr-session not connected yet)"
        keys = [k.decode() if isinstance(k, bytes) else str(k) for k in msg.keys()]
        cam = self._find_camera_subdict(msg)
        if cam is None:
            return (
                f"(wire keys={keys}; no camera-shaped subdict on wire — "
                "rr-session CameraNode not publishing? check /tmp/rr_logs_*/<cam>.log)"
            )
        cam_keys = [
            k.decode() if isinstance(k, bytes) else str(k) for k in cam.keys()
        ]
        images = self._g(cam, "images")
        if images is None:
            return f"(camera subdict present but no 'images' field; cam keys={cam_keys})"
        img_keys = [
            k.decode() if isinstance(k, bytes) else str(k) for k in images.keys()
        ]
        rgb = self._g(images, "left_rgb") or self._g(images, "rgb") or self._g(
            images, "right_rgb"
        )
        if rgb is None:
            return f"(camera.images keys={img_keys}; no rgb / left_rgb / right_rgb field)"
        return (
            f"(wire has rgb shape={getattr(rgb, 'shape', '?')} "
            f"dtype={getattr(rgb, 'dtype', '?')}, but self._obs[{cam_name!r}] is empty — "
            "_convert_observation dropping it?)"
        )

    def compute_reward(self) -> float:
        """No reward on real hardware."""
        return 0.0

    def task_completed(self) -> bool:
        """No automatic task completion on real hardware."""
        return False

    # ------------------------------------------------------------------
    # Control interface
    # ------------------------------------------------------------------

    def move_to_joints_blocking(
        self,
        joints: np.ndarray,
        *,
        tolerance: float = 0.051,
        max_steps: int = 350,
        arm_id: int = 0,
    ) -> None:
        """Move to target joint positions. Publishes at 50Hz, polls for convergence.

        Args:
            joints: (7,) target joint positions in radians.
            tolerance: Joint position error tolerance for convergence.
            max_steps: Maximum polling iterations before timeout. ``max_steps == 0``
                publishes the command once and returns immediately — used by
                streaming visual-servo loops where each new call supersedes the
                previous target and the upstream 50 Hz publisher is responsible
                for actually executing the trajectory.
        """
        target = np.asarray(joints, dtype=np.float64).reshape(7)
        self._current_joints = target

        # Single source of truth for the wire command: the republisher reads
        # ``_target_joints`` under ``_target_lock`` and re-stamps
        # ``self.server.latest_action`` every ``_action_publish_period`` (50 Hz).
        # Don't write ``latest_action`` from this method — that races against
        # both the republisher and ``MsgpackNumpyServer._handle.send_framed``.
        with self._target_lock:
            self._target_joints = target
            self._target_gripper = self._gripper_fraction
        # Heartbeat tracking — lets the diagnostic thread report ``target_age``
        # and detect "GoToPose stopped being called" vs "republisher died".
        self._target_update_count += 1
        self._last_target_update_t = time.time()

        if max_steps == 0:
            # Fire-and-forget streaming path. Caller owns the loop rate; the
            # background republisher keeps the wire command fresh at 50 Hz.
            self._sim_step_count += 1
            return

        steps = 0
        while steps < max_steps:
            self._update_from_network()
            joints_obs = self._obs.get("robot_joint_pos_0")
            if joints_obs is not None:
                current = np.asarray(joints_obs[:7], dtype=np.float64)
                error = np.linalg.norm(current - target)
                if error < tolerance:
                    break

            time.sleep(0.01)
            steps += 1
            self._sim_step_count += 1

            if self._record_frames:
                self._record_frame()

    def _set_gripper(self, fraction: float, arm_id: int = 0) -> None:
        """Set gripper opening fraction (0.0=closed, 1.0=open)."""
        self._gripper_fraction = float(np.clip(fraction, 0.0, 1.0))
        with self._target_lock:
            self._target_gripper = self._gripper_fraction
        time.sleep(0.15)  # hardware settle

    def _step_once(self) -> None:
        """Hold the current joint target and pace one publish period.

        Updates ``_target_joints`` so the republisher continues emitting the
        held pose; doesn't write ``latest_action`` directly (the republisher
        is the sole writer to avoid racing ``MsgpackNumpyServer._handle``).
        """
        with self._target_lock:
            self._target_joints = self._current_joints.copy()
            self._target_gripper = self._gripper_fraction
        self._sim_step_count += 1

        time.sleep(self._action_publish_period)  # 0.02s = 50Hz pacing

        if self._record_frames:
            self._record_frame()

    # ------------------------------------------------------------------
    # Video capture
    # ------------------------------------------------------------------

    def enable_video_capture(self, enabled: bool = True, *, clear: bool = True) -> None:
        self._record_frames = enabled
        if clear:
            self._frame_buffer.clear()
        if enabled:
            self._record_frame()

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [f.copy() for f in self._frame_buffer]
        if clear:
            self._frame_buffer.clear()
        return frames

    def _record_frame(self) -> None:
        if not self._record_frames:
            return
        cam_name = self.camera_names[0] if self.camera_names else "robot0_robotview"
        cam_obs = self._obs.get(cam_name, {})
        rgb = cam_obs.get("images", {}).get("rgb")
        if rgb is not None:
            self._frame_buffer.append(rgb.copy())

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Clean shutdown: stop republisher/heartbeat threads + msgpack server."""
        self._republish_stop.set()
        try:
            self.server.stop()
        except Exception:
            logger.debug("msgpack server stop failed", exc_info=True)


# ---------------------------------------------------------------------------
# Factory (registry contract)
# ---------------------------------------------------------------------------


def make_env(
    suite_name: str = "franka_real",
    task_id: int = 0,
    camera_names: list[str] | None = None,
    enable_render: bool = False,
    *,
    port: int = 9000,
    host: str = "127.0.0.1",
    **extra: Any,
):
    """Build a :class:`FrankaRealEnv` + its EnvConfig.

    EnvConfig values are lifted from the source server's ``franka_real``
    Init branch: absolute joint control at 50 Hz, canonical Franka home,
    panda_description URDF, and the Robotiq gripper TCP (−0.157 m along
    panda_hand Z, mounted with a π/4 Z-rotation).

    ``task_id`` / ``enable_render`` are accepted for registry-contract
    compatibility and ignored — there is no task switch or render toggle
    on real hardware.
    """
    from gap.envs.registry import EnvConfig

    env = FrankaRealEnv(camera_names=camera_names, port=port, host=host)
    config = EnvConfig(
        arm_dof=7,
        num_arms=1,
        action_mode="absolute_joints",
        control_freq=50.0,
        home_joints=_FRANKA_HOME_JOINTS,
        # Robotiq gripper TCP: offset from panda_hand to fingertip;
        # Robotiq is mounted with pi/4 Z-rotation relative to panda_hand.
        tcp_offset=(0.0, 0.0, -0.157),
        tcp_rotation_z=math.pi / 4,
        robot_urdf_path="panda_description",
        default_cameras=tuple(env.camera_names),
        is_real=True,
    )
    return env, config
