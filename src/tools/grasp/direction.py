"""Agent-specified angular filtering of unchanged Contact-GraspNet poses.

These filters only constrain TCP +Z (the approach axis) in robot-base coordinates.
CGN scores retain their original meaning; they are neither reweighted nor turned
into success probabilities. IK and collision validation remain separate stages.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from src.tools.grasp.cgn_client import ContactCenterGrasps, camera_to_robot_base


class DirectionFilterResult(NamedTuple):
    indices: np.ndarray
    metadata: dict[str, Any]


def _angle(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a number in [{minimum}, {maximum}]")
    result = float(value)
    if not np.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be a number in [{minimum}, {maximum}]")
    return result


def _poses_and_scores(poses: Any, scores: Any) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(poses)
    confidences = np.asarray(scores)
    if values.dtype.kind not in "iuf" or confidences.dtype.kind not in "iuf":
        raise ValueError("poses and scores must be finite real numbers")
    values = np.asarray(values, dtype=np.float64)
    confidences = np.asarray(confidences, dtype=np.float64)
    if values.shape == (0,):
        values = values.reshape(0, 4, 4)
    if values.ndim != 3 or values.shape[1:] != (4, 4) or confidences.shape != (len(values),):
        raise ValueError("poses must be Nx4x4 and scores must be N")
    if not np.isfinite(values).all() or not np.isfinite(confidences).all():
        raise ValueError("poses and scores must be finite real numbers")
    if not np.allclose(values[:, 3, :], [0, 0, 0, 1], atol=1e-6, rtol=0):
        raise ValueError("poses must have homogeneous bottom row [0, 0, 0, 1]")
    rotations = values[:, :3, :3]
    if not np.allclose(rotations.swapaxes(-2, -1) @ rotations, np.eye(3), atol=1e-4, rtol=0) or not np.allclose(
        np.linalg.det(rotations), 1, atol=1e-4, rtol=0
    ):
        raise ValueError("pose rotations must be in SO(3)")
    return values, confidences


def filter_cgn_directions(
    contact_poses_base: Any, scores: Any, *, direction: str, tolerance_deg: float,
    azimuth_deg: float | None = None, polar_deg: float | None = None,
) -> DirectionFilterResult:
    """Return accepted indices ordered by unchanged descending CGN score.

    ``vertical`` (alias ``top_down``) is a cone around base -Z; ``horizontal``
    is a band around the base XY plane. ``custom`` uses polar degrees from -Z
    and azimuth degrees counterclockwise from +X toward +Y. Cone tolerance is
    0..180 degrees, band tolerance 0..90 degrees. Boundaries are inclusive.
    This selects existing poses and never forces a new gripper rotation.
    """
    if direction == "top_down":
        direction = "vertical"
    if direction not in ("vertical", "horizontal", "custom"):
        raise ValueError("direction must be vertical, horizontal, or custom")
    tolerance = _angle(tolerance_deg, "tolerance_deg", 0, 90 if direction == "horizontal" else 180)
    desired = None
    if direction == "custom":
        polar = _angle(polar_deg, "polar_deg", 0, 180)
        azimuth = _angle(azimuth_deg, "azimuth_deg", -360, 360)
        theta, phi = np.deg2rad([polar, azimuth])
        desired = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), -np.cos(theta)])
    else:
        if azimuth_deg is not None or polar_deg is not None:
            raise ValueError("azimuth_deg and polar_deg apply only to custom direction")
        polar = azimuth = None
        if direction == "vertical":
            desired = np.array([0., 0., -1.])

    poses, confidences = _poses_and_scores(contact_poses_base, scores)
    approach = poses[:, :3, 2]
    approach = approach / np.linalg.norm(approach, axis=1, keepdims=True)
    if direction == "horizontal":
        errors = np.rad2deg(np.arcsin(np.clip(np.abs(approach[:, 2]), 0, 1)))
    else:
        errors = np.rad2deg(np.arccos(np.clip(approach @ desired, -1, 1)))
    accepted = errors <= tolerance + 1e-9  # roundoff only; retain inclusive angle boundary
    indices = np.flatnonzero(accepted)
    indices = indices[np.argsort(-confidences[indices], kind="stable")]
    indices.setflags(write=False)
    return DirectionFilterResult(indices, {
        "frame": "robot_base", "approach_axis": "+Z", "direction": direction,
        "tolerance_deg": tolerance, "boundary": "inclusive",
        "polar_deg": polar, "azimuth_deg": azimuth,
        "desired_approach_base": None if desired is None else desired.tolist(),
        "angular_errors_deg": errors.tolist(), "accepted_mask": accepted.tolist(),
        "candidate_count": len(poses), "accepted_count": len(indices),
        "ranking": "unchanged_cgn_score_descending", "rotation_modified": False,
    })


def select_top_down_grasp(
    contact_camera: ContactCenterGrasps, camera_to_base: Any,
    vertical_threshold: float = 0.8,
) -> tuple[np.ndarray | None, float]:
    """CaP-X's strict -approach_z > threshold selection with a frame guard.

    Input must be contact-center candidates tagged camera_optical, including
    their scores. Returns the highest-scoring accepted base-frame pose, or
    (None, -inf). Unlike angular filtering above, the threshold is strict,
    matching the original helper; threshold=1 accepts no candidates.
    """
    threshold = _angle(vertical_threshold, "vertical_threshold", -1, 1)
    base = camera_to_robot_base(contact_camera, camera_to_base)
    indices = np.flatnonzero(-base.poses[:, 2, 2] > threshold)
    if not len(indices):
        return None, float("-inf")
    best = indices[np.argmax(base.scores[indices])]
    return base.poses[best].copy(), float(base.scores[best])
