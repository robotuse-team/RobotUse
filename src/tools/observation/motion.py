"""Sensor-selected high wrist observation; no target oracle or grasp-pose changes."""
from __future__ import annotations

import numpy as np
from src.tools.motion.tolerances import cartesian_tolerances, requires_profile_tracking


def execute_observation_move(connector, *, target_points, config, viewpoint="top",
                             holding=False, recorder=None, point_ref=None):
    """Bounded, calibrated alternate view; caller captures/reselects AFTER return.

    This separate opt-in path deliberately
    disables collisions, follows environment tracking limits, retains IK/numerical
    validity, and reports actual view geometry rather than promising coverage.
    No gripper operation or grasp orientation is changed.
    """
    from dataclasses import replace
    from types import SimpleNamespace
    from src.tools.motion.planning import _points, _plan, _execute_checked
    from src.tools.observation.views import fresh_wrist_state, requested_view, view_metrics, VIEWPOINT_DEFINITIONS
    if holding:
        raise ValueError("view motion is forbidden while holding an object")
    points = _points(target_points)
    before = fresh_wrist_state(connector)
    height = max(float(config.high_transit_z_m), float(before["ee"][2, 3]), float(points[:, 2].max()) + .25)
    # Validate requested side and calibration before any motion.
    target, optical, ee = requested_view(points, before["ee_from_optical"], viewpoint=viewpoint, high_z_m=height)
    motion_config = replace(config, collision_checks_enabled=False)
    rise = before["ee"].copy(); rise[2, 3] = height
    diagnostics, labels, transforms = [], [], []
    def move(transform, label):
        segments, targets, *_ = _plan(connector, (transform,), np.empty((0, 3)), motion_config, None)
        if recorder is not None:
            recorder.register_plan(SimpleNamespace(segments=segments, targets=targets,
                target_labels=(label,), segment_labels=(label,), transit_policy="high",
                high_transit_z_m=height), kind="observation", point_ref=point_ref)
        diagnostics.extend(_execute_checked(connector, segments, targets, collision_checks_enabled=False))
        labels.append(label); transforms.append(transform.tolist())
    move(rise, "observation_rise")
    risen = fresh_wrist_state(connector)
    position_limit, _ = cartesian_tolerances(connector)
    if risen["ee"][2, 3] < height - position_limit:
        # A sequencing prerequisite, not a tracking safety certificate.
        raise RuntimeError("rise did not reach observation altitude; refusing low reorientation; reobserve required")
    target, optical, ee = requested_view(points, risen["ee_from_optical"], viewpoint=viewpoint,
                                       high_z_m=max(height, risen["ee"][2, 3]))
    move(ee, "observation_high_" + viewpoint)
    actual = fresh_wrist_state(connector)
    metrics = view_metrics(before["optical"], actual["optical"], target)
    requested_error = view_metrics(optical, actual["optical"], target)
    return {"target_labels": labels, "target_transforms": transforms, "diagnostics": diagnostics,
            "source": "selected_observed_cloud_centroid", "viewpoint": viewpoint,
            "viewpoint_definitions": VIEWPOINT_DEFINITIONS, "target_centroid_base": target.tolist(),
            "requested_view": {"optical_to_base": optical.tolist(), "ee_to_base": ee.tolist()},
            "actual_view": {"optical_to_base": actual["optical"].tolist(), "ee_to_base": actual["ee"].tolist(), "epoch": actual["epoch"]},
            "before_view": {"optical_to_base": before["optical"].tolist(), "ee_to_base": before["ee"].tolist(), "epoch": before["epoch"]},
            "view_novelty": metrics, "requested_camera_position_error_m": requested_error["camera_translation_m"],
            "requested_camera_axis_error_deg": requested_error["optical_axis_change_deg"],
            "actual_robot_translation_m": float(np.linalg.norm(actual["ee"][:3, 3] - before["ee"][:3, 3])),
            "actual_robot_rotation_deg": float(np.degrees(np.arccos(np.clip((np.trace(before["ee"][:3, :3].T @ actual["ee"][:3, :3]) - 1.) / 2., -1., 1.)))),
            "collision_checks_enabled": False, "tracking_safety_gate": requires_profile_tracking(connector),
            "invalidated_point_ref": point_ref, "requires_fresh_observation": True,
            "requires_same_object_reselection": True, "grasp_pose_modified": False}


def observe_above_cloud(connector, *, target_points, config, holding=False, recorder=None, point_ref=None):
    """Plan both observation legs before moving; retry 2 cm lower on rejection.

    Preserve orientation while adjusting height, then move above the cloud
    facing down. Never attempt heights at or below cloud top plus 5 cm.
    This observation pose does not replace a GraspGen prediction.
    """
    from src.tools.motion.planning import (
        MotionPlanningError, _points, _pose_transform, _plan, _execute_checked,
    )
    if holding:
        raise ValueError("view motion is forbidden while holding an object")
    points = _points(target_points)
    current = _pose_transform(connector.get_ee_pose())
    cloud_top = float(points[:, 2].max())
    initial_height = max(float(config.high_transit_z_m), float(current[2, 3]),
                         cloud_top + 0.25)
    minimum_height = cloud_top + 0.05
    if not np.isfinite(initial_height):
        raise ValueError("invalid observation height")
    obstacles = np.empty((0, 3))
    step = 0
    while True:
        height = initial_height - step * 0.02
        # Exclude equality despite floating point rounding at the boundary.
        if height <= minimum_height + 1e-9:
            raise MotionPlanningError(
                "no feasible observation path above cloud top + 0.05 m "
                f"after {step} height attempts")
        rise = current.copy()
        rise[2, 3] = height
        above = rise.copy()
        above[:3, :3] = np.diag([1., -1., -1.])
        above[:2, 3] = np.median(points[:, :2], axis=0)
        try:
            segments, targets, *_ = _plan(connector, (rise, above), obstacles, config, None)
        except MotionPlanningError as exc:
            # Invalid state, planner exceptions and malformed output are not
            # evidence of an unreachable height; propagate them immediately.
            if not str(exc).startswith("planner rejected segment "):
                raise
            step += 1
            continue
        break

    def record(segment, target, label):
        if recorder is not None:
            from types import SimpleNamespace
            recorder.register_plan(SimpleNamespace(segments=(segment,), targets=(target,),
                target_labels=(label,), segment_labels=(label,), transit_policy="high",
                high_transit_z_m=height), kind="observation", point_ref=point_ref)
    record(segments[0], targets[0], "observation_rise")
    diagnostics = _execute_checked(connector, segments[:1], targets[:1],
                                    collision_checks_enabled=config.collision_checks_enabled)
    # Check upward AND downward adjustment before lateral motion. Execution
    # failures stop immediately rather than triggering more height attempts.
    actual = _pose_transform(connector.get_ee_pose())
    position_limit, _ = cartesian_tolerances(connector)
    if abs(actual[2, 3] - height) > position_limit or actual[2, 3] <= minimum_height:
        raise RuntimeError("rise/height adjustment did not reach observation altitude; reobserve required")
    record(segments[1], targets[1], "observation_high_downward")
    diagnostics += _execute_checked(connector, segments[1:], targets[1:],
                                     collision_checks_enabled=config.collision_checks_enabled)
    actual = _pose_transform(connector.get_ee_pose())
    if (abs(actual[2, 3] - height) > position_limit or actual[2, 3] <= minimum_height or
            np.linalg.norm(actual[:2, 3] - above[:2, 3]) > 0.03 or
            float(-actual[2, 2]) < 0.95):
        raise RuntimeError("high downward observation pose not reached; reobserve required")
    return {"target_labels": ["observation_rise", "observation_high_downward"],
            "target_transforms": [rise.tolist(), above.tolist()],
            "diagnostics": diagnostics, "source": "selected_observed_cloud_xy",
            "collision_checks_enabled": config.collision_checks_enabled}
