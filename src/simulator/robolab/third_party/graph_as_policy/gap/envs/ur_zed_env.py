"""URZedEnv — UR robot + ZED stereo camera environment.

Directly wraps the ZED camera (via the pyzed SDK) and reads UR robot
state (via rtde_receive) to produce observations in the gap env obs
format. Perception-only — no robot control.

Hardware dependencies are imported lazily inside ``__init__`` so this
module always imports (registry resolution, validation, tests):

- ``pyzed`` ships with the ZED SDK and must be installed manually from
  https://www.stereolabs.com/docs/app-development/python/install
  (it is not on PyPI for every CUDA/python combination).
- ``rtde_receive`` comes from ``pip install 'graph-as-policy[real]'``
  (the ``ur-rtde`` package).

The URDF used for forward kinematics resolves from the ``GAP_UR_URDF``
environment variable (path to a UR URDF) or falls back to
``robot_descriptions``' ``ur5e_description``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from gap import env_config
from gap.envs.base_env import BaseEnv

logger = logging.getLogger(__name__)

_UR_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

# UR "elbow up, camera over table" home from the source server config.
_UR_HOME_JOINTS = (3.14159, -1.5708, -1.5708, -1.5708, 1.5708, -1.5708)


def _swap_mount_xy(T: np.ndarray) -> np.ndarray:
    """Swap local X/Y translation for the wrist-mounted camera offset.

    The calibrated orientation is correct, but this camera's mount position is
    mirrored into the wrong lateral slot unless we reinterpret the wrist-frame
    translation as having X/Y exchanged and flip the lateral sign.
    """
    out = np.array(T, copy=True)
    out[0, 3], out[1, 3] = -T[1, 3], T[0, 3]
    return out


def _import_pyzed():
    try:
        from pyzed import sl  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "URZedEnv needs the ZED SDK python bindings (pyzed), which are a "
            "manual install — they are not pip-installable from PyPI. Install "
            "the ZED SDK and its python API per "
            "https://www.stereolabs.com/docs/app-development/python/install "
            "then re-run."
        ) from e
    return sl


def _import_rtde_receive():
    try:
        import rtde_receive  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "URZedEnv needs the ur-rtde python bindings (rtde_receive). "
            "Install the real-hardware extra: pip install 'graph-as-policy[real]'"
        ) from e
    return rtde_receive


class ZedDepthCamera:
    """Minimal pyzed wrapper: synchronized left RGB + metric depth + intrinsics.

    Replaces the dexnet ``ZedWithDepth`` dependency of the original env,
    which provided exactly three things: ``get_images_and_depth()``
    (left/right RGB + depth in meters), the left-camera pinhole K, and
    ``close()``. This class provides the same surface directly on the
    pyzed SDK (see ``third_party/robots_realtime/.../zed_camera.py`` for
    the reference driver this mirrors).
    """

    _RESOLUTION_KEYS = {
        "2k": "HD2K", "1080p": "HD1080", "1200p": "HD1200",
        "720p": "HD720", "vga": "VGA", "svga": "SVGA",
    }

    def __init__(
        self,
        flip_mode: bool = True,
        resolution: str = "720p",
        fps: int = 15,
        depth_mode: str = "NEURAL",
    ) -> None:
        sl = _import_pyzed()
        self._sl = sl

        init_params = sl.InitParameters()
        res_key = self._RESOLUTION_KEYS.get(resolution.lower(), resolution)
        init_params.camera_resolution = getattr(sl.RESOLUTION, res_key)
        init_params.camera_fps = int(fps)
        init_params.depth_mode = getattr(sl.DEPTH_MODE, depth_mode)
        init_params.coordinate_units = sl.UNIT.METER
        if flip_mode:
            init_params.camera_image_flip = sl.FLIP_MODE.ON

        self.zed = sl.Camera()
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED camera open failed: {err!r}")

        self._left = sl.Mat()
        self._right = sl.Mat()
        self._depth = sl.Mat()
        self._runtime = sl.RuntimeParameters()

        calib = self.zed.get_camera_information().camera_configuration.calibration_parameters
        cam = calib.left_cam
        self._K = np.array(
            [[cam.fx, 0.0, cam.cx], [0.0, cam.fy, cam.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def get_intrinsics_matrix(self) -> np.ndarray:
        """Left-camera 3x3 pinhole K (rectified)."""
        return self._K.copy()

    def get_images_and_depth(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Grab one frame: (left_rgb uint8 HxWx3, right_rgb, depth_m float32 HxW)."""
        sl = self._sl
        if self.zed.grab(self._runtime) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("ZED camera grab failed")
        self.zed.retrieve_image(self._left, sl.VIEW.LEFT)
        self.zed.retrieve_image(self._right, sl.VIEW.RIGHT)
        self.zed.retrieve_measure(self._depth, sl.MEASURE.DEPTH)
        # pyzed returns BGRA; convert to contiguous RGB.
        left_rgb = np.ascontiguousarray(self._left.get_data()[:, :, :3][:, :, ::-1])
        right_rgb = np.ascontiguousarray(self._right.get_data()[:, :, :3][:, :, ::-1])
        depth_m = np.asarray(self._depth.get_data(), dtype=np.float32)
        # NaN/inf where stereo matching failed → 0 (the convention the
        # perception tools treat as "no depth").
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        return left_rgb, right_rgb, depth_m

    def close(self) -> None:
        self.zed.close()


def _load_ur_urdf():
    """Load the UR URDF for FK: ``GAP_UR_URDF`` path or ur5e_description."""
    import yourdfpy

    urdf_path = env_config.ur_urdf()
    if urdf_path:
        logger.info("[URZedEnv] Loading URDF from GAP_UR_URDF=%s", urdf_path)
        return yourdfpy.URDF.load(urdf_path)
    from robot_descriptions.loaders.yourdfpy import load_robot_description

    logger.info("[URZedEnv] Loading robot_descriptions ur5e_description")
    return load_robot_description("ur5e_description")


class URZedEnv(BaseEnv):
    """UR robot + ZED camera environment for real-hardware perception.

    Unlike FrankaRealEnv (which receives data via robots_realtime msgpack),
    this env directly captures from the ZED camera and reads UR state via
    RTDE. Designed for perception-only workflows — no robot control.

    Camera pose is computed via URDF FK (joint angles → wrist_3_link) plus
    the hand-eye calibration, NOT from getActualTCPPose() which includes
    tool offsets that don't match the physical camera mounting.
    """

    def __init__(
        self,
        camera_names: list[str] | None = None,
        robot_ip: str = "172.22.22.2",
        calibration_path: str | Path | None = None,
        resolution: str = "720p",
        fps: int = 15,
    ) -> None:
        super().__init__()

        self.camera_names = camera_names or ["zed_left"]

        # --- ZED camera (manual ZED SDK install; lazy import inside) ---
        logger.info("[URZedEnv] Initializing ZED camera...")
        self.zed = ZedDepthCamera(flip_mode=True, resolution=resolution, fps=fps)

        # --- UR robot (read-only joint state via RTDE) ---
        rtde_receive = _import_rtde_receive()
        logger.info("[URZedEnv] Connecting to UR robot at %s...", robot_ip)
        self.rtde_recv = rtde_receive.RTDEReceiveInterface(robot_ip)

        # --- URDF for FK (wrist flange pose from joint angles) ---
        self._urdf = _load_ur_urdf()

        # --- Camera calibration (4x4 camera→wrist) ---
        calibration_path = calibration_path or env_config.ur_zed_calib()
        if calibration_path:
            self._T_cam_to_wrist = np.load(str(calibration_path))
            logger.info("[URZedEnv] Loaded calibration from %s", calibration_path)
        else:
            self._T_cam_to_wrist = np.eye(4)
            logger.warning(
                "[URZedEnv] No hand-eye calibration provided (calibration_path "
                "kwarg or GAP_UR_ZED_CALIB env var, a 4x4 camera→wrist .npy) — "
                "using identity; camera poses will equal the wrist pose."
            )

        # --- State ---
        self._obs: dict[str, Any] = {}
        self._sim_step_count = 0
        self.max_steps = 999999

        logger.info("[URZedEnv] Ready.")

    # ------------------------------------------------------------------
    # FK + calibration
    # ------------------------------------------------------------------

    def _compute_camera_pose(self, joints: np.ndarray) -> np.ndarray:
        """Compute camera pose in base frame via URDF FK + hand-eye calibration.

        Returns [x, y, z, w, x, y, z] (position + WXYZ quaternion).
        """
        cfg = dict(zip(_UR_JOINT_NAMES, joints.tolist(), strict=True))
        self._urdf.update_cfg(cfg)

        # Wrist flange pose in base frame (4x4)
        T_wrist = self._urdf.get_transform("wrist_3_link")

        # Camera pose: T_base_camera = T_base_wrist @ T_wrist_camera
        # T_cam_to_wrist maps camera→wrist, so camera frame in base =
        # T_wrist @ inv(T_cam_to_wrist)  [inv because we want camera origin
        # in wrist coords, not wrist origin in camera coords]
        T_wrist_to_cam = _swap_mount_xy(np.linalg.inv(self._T_cam_to_wrist))
        T_cam_in_base = T_wrist @ T_wrist_to_cam

        pos = T_cam_in_base[:3, 3].astype(np.float32)
        q_xyzw = Rotation.from_matrix(T_cam_in_base[:3, :3]).as_quat()
        q_wxyz = np.array(
            [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]],
            dtype=np.float32,
        )
        return np.concatenate([pos, q_wxyz])

    def _compute_ee_pose(self, joints: np.ndarray) -> np.ndarray:
        """Compute EE pose in base frame via URDF FK.

        Returns [x, y, z, w, x, y, z, gripper_frac].
        """
        cfg = dict(zip(_UR_JOINT_NAMES, joints.tolist(), strict=True))
        self._urdf.update_cfg(cfg)

        ee_link = "ee_link" if "ee_link" in self._urdf.link_map else "wrist_3_link"
        T_ee = self._urdf.get_transform(ee_link)

        pos = T_ee[:3, 3].astype(np.float32)
        q_xyzw = Rotation.from_matrix(T_ee[:3, :3]).as_quat()
        q_wxyz = np.array(
            [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]],
            dtype=np.float32,
        )
        return np.concatenate([pos, q_wxyz, np.array([1.0], dtype=np.float32)])

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _capture_observation(self) -> dict[str, Any]:
        """Capture one frame from ZED + UR state, return the gap obs dict."""
        cam_name = self.camera_names[0] if self.camera_names else "zed_left"

        # Camera capture (RGB already in RGB order, depth in meters)
        left_rgb, _right_rgb, depth_meters = self.zed.get_images_and_depth()
        # The physical ZED mount is upside down relative to the calibrated camera
        # frame, so rotate both RGB and depth together to keep geometry aligned.
        left_rgb = np.ascontiguousarray(np.rot90(left_rgb, 2))
        depth_meters = np.ascontiguousarray(np.rot90(depth_meters, 2))
        K = self.zed.get_intrinsics_matrix()

        # Robot joint angles (read-only RTDE)
        joints = np.array(self.rtde_recv.getActualQ(), dtype=np.float64)

        # Camera and EE poses via URDF FK (not TCP which includes tool offset)
        cam_pose = self._compute_camera_pose(joints)
        ee_pose = self._compute_ee_pose(joints)

        obs: dict[str, Any] = {
            "robot_joint_pos_0": np.append(
                joints.astype(np.float32), np.float32(1.0)
            ),
            "robot_cartesian_pos_0": ee_pose,
            cam_name: {
                "images": {
                    "rgb": left_rgb,
                    "depth": depth_meters.astype(np.float32),
                },
                "intrinsics": K.astype(np.float32),
                "pose": cam_pose,
            },
        }
        return obs

    # ------------------------------------------------------------------
    # BaseEnv interface
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._sim_step_count = 0
        self._obs = self._capture_observation()
        return self._obs, {}

    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._sim_step_count += 1
        obs = self.get_observation()
        return obs, 0.0, False, False, {}

    def get_observation(self) -> dict[str, Any]:
        self._obs = self._capture_observation()
        return self._obs

    def compute_reward(self) -> float:
        return 0.0

    def task_completed(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        logger.info("[URZedEnv] Closing ZED camera...")
        self.zed.close()


# ---------------------------------------------------------------------------
# Factory (registry contract)
# ---------------------------------------------------------------------------


def make_env(
    suite_name: str = "ur_zed",
    task_id: int = 0,
    camera_names: list[str] | None = None,
    enable_render: bool = False,
    *,
    robot_ip: str = "172.22.22.2",
    calibration_path: str | Path | None = None,
    resolution: str = "720p",
    fps: int = 15,
    **extra: Any,
):
    """Build a :class:`URZedEnv` + its EnvConfig (perception-only).

    ``task_id`` / ``enable_render`` are accepted for registry-contract
    compatibility and ignored.
    """
    from gap.envs.registry import EnvConfig

    env = URZedEnv(
        camera_names=camera_names,
        robot_ip=robot_ip,
        calibration_path=calibration_path,
        resolution=resolution,
        fps=fps,
    )
    config = EnvConfig(
        arm_dof=6,
        num_arms=1,
        action_mode="absolute_joints",
        control_freq=15.0,
        home_joints=_UR_HOME_JOINTS,
        robot_urdf_path=env_config.ur_urdf() or "ur5e_description",
        default_cameras=tuple(env.camera_names),
        is_real=True,
    )
    return env, config
