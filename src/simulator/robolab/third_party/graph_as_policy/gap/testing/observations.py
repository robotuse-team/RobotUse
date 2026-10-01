"""Geometrically consistent synthetic observations.

``make_test_observation`` renders colored axis-aligned boxes with a real
pinhole camera model: the returned ``depth`` + ``intrinsics`` + camera
``pose`` reproject exactly onto the boxes' world-space surfaces. Perception
math (mask → points → OBB) can therefore be tested against true numerics,
not mocks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gap_core.types import ArmState, CameraFrame, Observation, Se3Pose, make_pose, pose_to_matrix

# Distinct, saturated colors assigned to objects in order.
_PALETTE: list[tuple[int, int, int]] = [
    (220, 40, 40),  # red
    (40, 90, 220),  # blue
    (40, 180, 60),  # green
    (230, 200, 40),  # yellow
    (170, 60, 200),  # purple
]
_TABLE_COLOR = (110, 110, 110)
_SKY_COLOR = (235, 235, 235)


def _look_at_pose(eye: np.ndarray, target: np.ndarray) -> Se3Pose:
    """Camera-to-world pose with +z forward (toward target), +x right, +y down."""
    from scipy.spatial.transform import Rotation

    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    if np.linalg.norm(right) < 1e-6:  # looking straight down
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    down = np.cross(fwd, right)
    rot = np.stack([right, down, fwd], axis=1)  # columns: x, y, z axes
    x, y, z, w = Rotation.from_matrix(rot).as_quat()
    return make_pose(eye, (w, x, y, z))


def make_test_observation(
    objects: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]] | None = None,
    *,
    camera_name: str = "test_cam",
    image_hw: tuple[int, int] = (120, 160),
    camera_eye: tuple[float, float, float] = (0.0, -0.7, 0.9),
    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.1),
    table_z: float = 0.0,
    fov_deg: float = 60.0,
    arm_joints: int = 7,
) -> tuple[Observation, dict[str, Any]]:
    """Render a synthetic tabletop scene into a gap Observation.

    Args:
        objects: list of ``(name, center_xyz, size_xyz)`` axis-aligned boxes
            (full extents, meters). Default: one 6 cm cube at the origin
            sitting on the table.
        image_hw, camera_eye, camera_target, table_z, fov_deg: scene knobs.
        arm_joints: dof of the stub arm state.

    Returns:
        ``(observation, ground_truth)`` where ground_truth maps object name →
        {"center", "size", "color", "mask"} (mask = boolean pixel ownership),
        plus "camera_pose_mat" (4x4 camera-to-world).
    """
    if objects is None:
        objects = [("cube", (0.0, 0.0, table_z + 0.03), (0.06, 0.06, 0.06))]

    h, w = image_hw
    fx = fy = (w / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    intrinsics = np.array(
        [[fx, 0.0, w / 2.0], [0.0, fy, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )

    cam_pose = _look_at_pose(np.asarray(camera_eye, dtype=float), np.asarray(camera_target, dtype=float))
    cam_mat = pose_to_matrix(cam_pose)
    rot_cw = cam_mat[:3, :3]
    origin = cam_mat[:3, 3]

    # Rays in camera frame with z == 1, so the slab-test parameter t IS the
    # camera z-depth at each pixel.
    us, vs = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    dirs_cam = np.stack(
        [(us - intrinsics[0, 2]) / fx, (vs - intrinsics[1, 2]) / fy, np.ones_like(us)],
        axis=-1,
    )
    dirs_world = dirs_cam @ rot_cw.T  # [H, W, 3]

    depth = np.full((h, w), np.inf, dtype=np.float64)
    rgb = np.empty((h, w, 3), dtype=np.uint8)
    rgb[:] = _SKY_COLOR
    owner = np.full((h, w), -1, dtype=np.int32)  # -1 sky, -2 table, i = object

    # Table plane z = table_z (only where rays head downward toward it).
    dz = dirs_world[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_table = (table_z - origin[2]) / dz
    hit_table = (t_table > 1e-6) & np.isfinite(t_table)
    np.copyto(depth, t_table, where=hit_table & (t_table < depth))
    owner[hit_table & (depth == t_table)] = -2

    ground_truth: dict[str, Any] = {"camera_pose_mat": cam_mat}
    for i, (name, center, size) in enumerate(objects):
        lo = np.asarray(center, dtype=float) - np.asarray(size, dtype=float) / 2.0
        hi = np.asarray(center, dtype=float) + np.asarray(size, dtype=float) / 2.0
        # Vectorized slab test.
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo - origin) / dirs_world
            t2 = (hi - origin) / dirs_world
        tmin = np.minimum(t1, t2).max(axis=-1)
        tmax = np.maximum(t1, t2).min(axis=-1)
        hit = (tmax >= tmin) & (tmin > 1e-6)
        closer = hit & (tmin < depth)
        depth[closer] = tmin[closer]
        owner[closer] = i
        color = _PALETTE[i % len(_PALETTE)]
        ground_truth[name] = {
            "center": np.asarray(center, dtype=float),
            "size": np.asarray(size, dtype=float),
            "color": color,
        }

    for i, (name, _, _) in enumerate(objects):
        mask = owner == i
        rgb[mask] = ground_truth[name]["color"]
        ground_truth[name]["mask"] = mask
    rgb[owner == -2] = _TABLE_COLOR
    depth[~np.isfinite(depth)] = 0.0  # sky → 0 (invalid), matching sim conventions

    frame: CameraFrame = {
        "name": camera_name,
        "rgb": rgb,
        "depth": depth.astype(np.float32),
        "intrinsics": intrinsics,
        "pose": cam_pose,
    }
    arm: ArmState = {
        "joint_state": {"positions": np.zeros(arm_joints)},
        "gripper_fraction": 1.0,
        "ee_pose": make_pose((0.3, 0.0, 0.4), (1.0, 0.0, 0.0, 0.0)),
    }
    obs: Observation = {"cameras": [frame], "arms": [arm]}
    return obs, ground_truth
