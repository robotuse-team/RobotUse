"""The gap data vocabulary — plain typed dicts + numpy arrays.

These types replace the protobuf messages of the original research codebase.
They are deliberately boring: TypedDicts (so graph ``$ref`` dataflow walks
them as plain dicts and traces serialize them directly) carrying numpy arrays
(no byte packing in-process — conversions from simulator buffers happen once,
inside env classes).

Conventions
-----------
- **Quaternions are wxyz (scalar-first).** LIBERO/MuJoCo use xyzw internally;
  env classes convert at the boundary with :func:`quat_xyzw_to_wxyz`.
- ``rgb`` is ``uint8 [H, W, 3]``; ``depth`` is ``float32 [H, W]`` in meters;
  ``Mask`` is ``uint8 [H, W]`` (0 = background, 255 = foreground).
- Camera ``intrinsics`` is the ``float64 [3, 3]`` pinhole matrix K; camera
  ``pose`` is camera-to-world.
- ``OrientedBoundingBox.extent`` holds **half**-extents along local axes.
- ``ArmState.proprio_state`` is the policy-training-exact layout (e.g. the
  openpi-LIBERO ``[eef_pos(3), axisangle(3), gripper_qpos(2)]``) and must
  never be transformed by the runtime.
"""

# NOTE: no `from __future__ import annotations` here — PEP 563 string
# annotations break TypedDict's NotRequired detection (__required_keys__).

import sys
from typing import TypedDict

if sys.version_info >= (3, 11):
    from typing import NotRequired
else:  # pragma: no cover
    from typing_extensions import NotRequired

import numpy as np

__all__ = [
    "Vec3",
    "Quaternion",
    "Se3Pose",
    "OrientedBoundingBox",
    "BoundingBox2D",
    "CameraFrame",
    "Mask",
    "PointCloud",
    "JointState",
    "Trajectory",
    "GripperState",
    "ArmState",
    "Observation",
    "CollisionMesh",
    "WorldConfig",
    "GraspCandidates",
    "identity_pose",
    "make_pose",
    "quat_xyzw_to_wxyz",
    "quat_wxyz_to_xyzw",
    "pose_to_matrix",
    "matrix_to_pose",
]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


class Vec3(TypedDict):
    x: float
    y: float
    z: float


class Quaternion(TypedDict):
    """Unit quaternion, **wxyz scalar-first**."""

    w: float
    x: float
    y: float
    z: float


class Se3Pose(TypedDict):
    """6-DOF rigid transform: position + wxyz rotation."""

    position: Vec3
    rotation: Quaternion


class OrientedBoundingBox(TypedDict):
    """Oriented box: center, **half**-extents along local axes, orientation."""

    center: Vec3
    extent: Vec3
    orientation: Quaternion


class BoundingBox2D(TypedDict):
    """Axis-aligned 2D box in pixel coordinates, top-left → bottom-right."""

    x1: float
    y1: float
    x2: float
    y2: float


# ---------------------------------------------------------------------------
# Sensor data
# ---------------------------------------------------------------------------

#: ``uint8 [H, W]`` — 0 = background, 255 = foreground.
Mask = np.ndarray


class CameraFrame(TypedDict):
    """One camera's synchronized RGB-D frame + calibration."""

    name: str
    rgb: np.ndarray  # uint8 [H, W, 3]
    depth: np.ndarray  # float32 [H, W], meters
    intrinsics: np.ndarray  # float64 [3, 3] pinhole K
    pose: Se3Pose  # camera-to-world


class PointCloud(TypedDict):
    points: np.ndarray  # float32 [N, 3]
    colors: NotRequired[np.ndarray]  # float32 [N, 3] in [0, 1]


# ---------------------------------------------------------------------------
# Robot state
# ---------------------------------------------------------------------------


class JointState(TypedDict):
    positions: np.ndarray  # float64 [dof], radians
    names: NotRequired[list[str]]


class Trajectory(TypedDict):
    waypoints: list[JointState]


class GripperState(TypedDict):
    position: float  # meters; 0.0 = closed


class ArmState(TypedDict):
    joint_state: JointState
    gripper_fraction: float  # 0.0 closed → 1.0 open
    ee_pose: Se3Pose  # end-effector in world frame
    gripper_qpos: NotRequired[np.ndarray]  # raw per-finger qpos (sim-specific)
    proprio_state: NotRequired[np.ndarray]  # policy-training-exact — never transform


class Observation(TypedDict):
    """The full observation surface: all cameras + all arms."""

    cameras: list[CameraFrame]
    arms: list[ArmState]


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


class CollisionMesh(TypedDict):
    name: str
    vertices: np.ndarray  # float32 [V, 3]
    faces: np.ndarray  # int32 [F, 3]
    pose: Se3Pose


class WorldConfig(TypedDict):
    """Planner-agnostic collision scene."""

    meshes: list[CollisionMesh]


class GraspCandidates(TypedDict):
    poses: list[Se3Pose]  # best-first
    scores: NotRequired[list[float]]


# ---------------------------------------------------------------------------
# Helpers (boundary conversions + small constructors)
# ---------------------------------------------------------------------------


def identity_pose() -> Se3Pose:
    return {
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }


def make_pose(xyz, quat_wxyz) -> Se3Pose:
    """Build an Se3Pose from a 3-sequence and a wxyz 4-sequence."""
    x, y, z = (float(v) for v in xyz)
    w, qx, qy, qz = (float(v) for v in quat_wxyz)
    return {
        "position": {"x": x, "y": y, "z": z},
        "rotation": {"w": w, "x": qx, "y": qy, "z": qz},
    }


def quat_xyzw_to_wxyz(q) -> tuple[float, float, float, float]:
    """Convert an xyzw quaternion (MuJoCo/LIBERO/scipy order) to wxyz."""
    x, y, z, w = (float(v) for v in q)
    return (w, x, y, z)


def quat_wxyz_to_xyzw(q) -> tuple[float, float, float, float]:
    """Convert a wxyz quaternion (gap order) to xyzw."""
    w, x, y, z = (float(v) for v in q)
    return (x, y, z, w)


def pose_to_matrix(pose: Se3Pose) -> np.ndarray:
    """Se3Pose → 4x4 homogeneous transform (float64)."""
    from scipy.spatial.transform import Rotation

    rot = pose["rotation"]
    pos = pose["position"]
    mat = np.eye(4)
    mat[:3, :3] = Rotation.from_quat(
        [rot["x"], rot["y"], rot["z"], rot["w"]]
    ).as_matrix()
    mat[:3, 3] = [pos["x"], pos["y"], pos["z"]]
    return mat


def matrix_to_pose(mat: np.ndarray) -> Se3Pose:
    """4x4 homogeneous transform → Se3Pose."""
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(np.asarray(mat)[:3, :3]).as_quat()
    tx, ty, tz = np.asarray(mat)[:3, 3]
    return make_pose((tx, ty, tz), (w, x, y, z))
