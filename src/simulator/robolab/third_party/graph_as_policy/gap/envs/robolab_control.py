"""Simulator-independent units and frame contract for the RoboLab DROID bridge.

Public poses are the Robotiq base_link flange, in the robot base frame.
Matrices map child coordinates to parent coordinates. Joint actions are absolute
radians, with the Robotiq driving joint in radians; no hidden home offset.
"""
from __future__ import annotations
import math
import numpy as np

ROBOTIQ_MAX_WIDTH_M = .085
HOME_JOINTS = (0., -np.pi / 5, 0., -4 * np.pi / 5, 0., 3 * np.pi / 5, 0.)


def validate_motion_speed_scale(value):
    scale = float(value)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('motion_speed_scale must be positive and finite')
    return scale


def retime_joint_targets(start, targets, speed_scale):
    """Sample the joint polyline at a scaled rate, retaining its exact endpoint.

    Each target originally represents one control tick after the previous pose.
    Physics dt, controller gains, gripper holds and endpoint settling are separate.
    Inputs are validated by the environment before this iterator is consumed.
    """
    scale = validate_motion_speed_scale(speed_scale)
    if scale == 1.:
        yield from targets
        return
    count = len(targets)
    samples = math.ceil(count / scale)
    path = [start, *targets]
    for step in range(1, samples + 1):
        position = min(step * scale, count)
        if position >= count:
            yield targets[-1]
        else:
            lower = math.floor(position)
            fraction = position - lower
            yield path[lower] + fraction * (path[lower + 1] - path[lower])


def finite_vector(value, size, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f'{name} must contain {size} finite values')
    return array.copy()


def rigid_pose(value):
    pose = np.array(value, dtype=float, copy=True)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1])
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5, rtol=0)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-5, rtol=0)):
        raise ValueError('pose must be a finite rigid transform')
    # Learned float32 rotation products accumulate a few parts per million of
    # drift. Project only this bounded roundoff onto SO(3), retaining the exact
    # translation and source artifact. Reflections, scale and shear above the
    # tolerance remain invalid. Already-rigid matrices remain bitwise unchanged.
    rotation = pose[:3, :3]
    if np.max(np.abs(rotation.T @ rotation - np.eye(3))) > 1e-10:
        u, _, vt = np.linalg.svd(rotation)
        pose[:3, :3] = u @ vt
    return pose


def relative_pose_cm(pose, delta_cm):
    result = rigid_pose(pose)
    result[:3, 3] += finite_vector(delta_cm, 3, 'delta_cm') / 100.
    return result


def pose_in_base(world_from_base, world_from_sensor):
    return np.linalg.inv(rigid_pose(world_from_base)) @ rigid_pose(world_from_sensor)


def validate_joint_target(joints, limits=None):
    target = finite_vector(joints, 7, 'joint target')
    if limits is not None:
        limits = np.asarray(limits, dtype=float)
        if limits.shape != (7, 2) or not np.isfinite(limits).all():
            raise ValueError('invalid joint limits')
        if np.any(target < limits[:, 0]) or np.any(target > limits[:, 1]):
            raise ValueError('joint target exceeds robot limits')
    return target


def robotiq_width_to_angle(width_m):
    """2F-85 parallel-jaw linkage (URDF dimensions), width in metres.

    Physical opening is measured separately; commands never substitute for it.
    Small negative angles at the nominal 85 mm limit are clamped to open.
    """
    if not np.isfinite(width_m) or not 0 <= width_m <= ROBOTIQ_MAX_WIDTH_M:
        raise ValueError('Robotiq total opening must be in [0, .085] metres')
    # The native USD driver stops at 45 degrees (about 2 mm residual aperture).
    return float(np.clip(.715 - np.arcsin((width_m - .010) / .1143), 0, np.pi / 4))


def joint_action(joints, width_m):
    return np.r_[validate_joint_target(joints), robotiq_width_to_angle(width_m)]
