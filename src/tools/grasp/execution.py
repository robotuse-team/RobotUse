"""Grasp transit with independently selected contact-centre heights.

Inputs use the existing canonical hand frame (+Z approaches, +X closes).
The RoboLab public EE is the flange, so an absolute contact-centre height
must be converted through both the hand/EE calibration and the jaw offset.
No height is raised, clamped, or replaced by an automatic route preference.
"""
from __future__ import annotations

import numbers

import numpy as np

from src.tools.motion import planning as motion


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f'{name} must be a finite number in metres')
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0):
        raise ValueError(f'{name} must be finite' + (' and positive' if positive else ''))
    return result


def contact_center(ee_transform, grasp_to_ee, *, jaw_offset_m=.136):
    """Measured/target jaw centre in robot-base XYZ, from a public EE pose."""
    hand = motion._transform(ee_transform) @ np.linalg.inv(motion._transform(grasp_to_ee))
    offset = _number(jaw_offset_m, 'jaw_offset_m', positive=True)
    return hand[:3, 3] + offset * hand[:3, 2]


def explicit_grasp_waypoints(*, current_ee, grasp_transform, grasp_to_ee,
                             pre_pick_z_m, post_pick_z_m, approach_m=.10,
                             jaw_offset_m=.136, lift_after_grasp=True,
                             resume_from_pregrasp=False):
    """Return ``(EE transforms, semantic labels)`` without planning or motion.

    ``pre_pick_z_m`` and ``post_pick_z_m`` are absolute robot-base Z values
    of the contact centre, not flange Z and not relative lift distances.
    The incoming route moves vertically at the current pose, then reorients
    above the grasp at the chosen incoming height. Tilted grasps additionally
    align above their pregrasp before descending along the retained approach.
    ``resume_from_pregrasp`` omits incoming transit after a paused correction;
    the corrected pregrasp and original selected outgoing height are retained.
    """
    if type(lift_after_grasp) is not bool or type(resume_from_pregrasp) is not bool:
        raise ValueError('grasp route switches must be booleans')
    incoming_z = _number(pre_pick_z_m, 'pre_pick_z_m')
    outgoing_z = _number(post_pick_z_m, 'post_pick_z_m')
    approach_distance = _number(approach_m, 'approach_m', positive=True)
    offset = _number(jaw_offset_m, 'jaw_offset_m', positive=True)
    current = motion._transform(current_ee)
    hand = motion._transform(grasp_transform)
    calibration = motion._transform(grasp_to_ee)
    ee = hand @ calibration

    def at_height(pose, height):
        target = pose.copy()
        target[2, 3] += height - contact_center(target, calibration, jaw_offset_m=offset)[2]
        return target

    approach = ee.copy()
    approach[:3, 3] -= approach_distance * hand[:3, 2]
    targets, labels = [], []
    if not resume_from_pregrasp:
        targets.extend((at_height(current, incoming_z), at_height(ee, incoming_z)))
        labels.extend(('initial_lift', 'high_transit'))
        if not np.allclose(approach[:2, 3], ee[:2, 3], atol=1e-10, rtol=0):
            targets.append(at_height(approach, incoming_z))
            labels.append('high_pregrasp_align')
    targets.extend((approach, ee))
    labels.extend(('pregrasp', 'grasp'))
    if lift_after_grasp:
        targets.append(at_height(ee, outgoing_z))
        labels.append('lift')
    return tuple(targets), tuple(labels)


def plan_explicit_grasp(connector, *, grasp_transform, grasp_to_ee,
                        target_points, obstacle_points, pre_pick_z_m, post_pick_z_m,
                        config=motion.MotionConfig(), jaw_offset_m=.136,
                        lift_after_grasp=True, resume_from_pregrasp=False,
                        open_width_m=.08, max_width_m=.08, trajectory_validator=None,
                        chain_planner=None):
    """Plan a grasp using the shared GraspPlan executor and checkpoints.

    A caller with observed-world routing can inject
    ``chain_planner(connector, targets, obstacles, config, None,
    target_labels=labels)``. It returns the same five items as ``_plan``:
    segments, public target poses, starting joints, clearance, and validation
    status. Each target has exactly one segment, whose joint waypoints may
    describe an obstacle-aware detour. The callback must perform its route's
    robot/scene checks; this function accepts its returned chain and applies
    ``trajectory_validator`` only when collision checks are enabled.
    There is no fallback from a rejected callback to a different route.

    With no callback, the existing ``_plan`` supplies planning/check boundaries.
    The returned plan stores ``pre_pick_z_m``, ``post_pick_z_m`` and
    ``jaw_offset_m`` for a subsequent paused refinement using this function
    again with ``resume_from_pregrasp=True``.
    """
    from src.tools.motion.robot_state import _current_robot_state
    maximum = _number(max_width_m, 'max_width_m', positive=True)
    opening = _number(open_width_m, 'open_width_m', positive=True)
    if opening > maximum:
        raise ValueError(f'gripper opening must be in (0, {maximum}] meters')
    current_pose, _ = _current_robot_state(connector)
    hand = motion._transform(grasp_transform)
    calibration = motion._transform(grasp_to_ee)
    obj = motion._points(target_points)
    obstacles = motion._points(obstacle_points)
    targets, labels = explicit_grasp_waypoints(current_ee=motion._pose_transform(current_pose),
        grasp_transform=hand, grasp_to_ee=calibration, pre_pick_z_m=pre_pick_z_m,
        post_pick_z_m=post_pick_z_m, approach_m=config.approach_m,
        jaw_offset_m=jaw_offset_m, lift_after_grasp=lift_after_grasp,
        resume_from_pregrasp=resume_from_pregrasp)
    if config.collision_checks_enabled and lift_after_grasp:
        displacement = targets[-1][:3, 3] - targets[labels.index('grasp')][:3, 3]
        motion._payload_clearance(obj, [np.zeros(3), displacement], obstacles,
            config.clearance_m, allow_initial_support=displacement[2] > config.clearance_m)
    try:
        if chain_planner is None:
            result = motion._plan(connector, targets, obstacles, config, trajectory_validator)
        else:
            result = chain_planner(connector, targets, obstacles, config, None, target_labels=labels)
        segments, poses, joints, clearance, validated = result
        segments, poses, joints = tuple(segments), tuple(poses), tuple(joints)
    except motion.MotionPlanningError as exc:
        index = exc.planning_feedback.get('segment_index')
        if type(index) is int and 0 <= index < len(labels):
            exc.planning_feedback['segment'] = labels[index]
        raise
    if chain_planner is not None and config.collision_checks_enabled and trajectory_validator is not None:
        validated = trajectory_validator(segments, joints, obstacles.copy()) is True
        if not validated:
            raise motion.MotionPlanningError('external trajectory validator rejected motion path')
    plan = motion.GraspPlan(segments, poses, joints, hand, calibration, obj, obstacles,
        clearance, validated, connector=connector, target_labels=labels,
        transit_policy='explicit', high_transit_z_m=None, open_width_m=opening,
        collision_checks_enabled=config.collision_checks_enabled,
        limitations=motion.LIMITATIONS if config.collision_checks_enabled else motion.COLLISION_DISABLED_LIMITATIONS)
    plan.pre_pick_z_m = float(pre_pick_z_m)
    plan.post_pick_z_m = float(post_pick_z_m)
    plan.jaw_offset_m = float(jaw_offset_m)
    plan.resume_from_pregrasp = resume_from_pregrasp
    return plan
