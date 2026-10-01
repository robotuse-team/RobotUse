"""Convert learned object placements through the measured grasp attachment."""
from __future__ import annotations

import numpy as np

from src.tools.motion.planning import (
    _transform, _pose_transform, _plan, PlacePlan, MotionPlanningError,
    COLLISION_DISABLED_LIMITATIONS,
    execute_place,
)


def release_pose(relative_transform, closed_ee_pose):
    """For a rigid hold: desired EE = delta(object input -> placed) @ EE(close).

    The object cloud stays in its original measured base frame. The actual
    close pose binds that frame to the gripper, including grasp tracking error.
    This does not estimate slip or object movement caused by finger contact.
    """
    return _transform(relative_transform) @ _transform(closed_ee_pose)


def preview_anyplace(connector, *, relative_transform, closed_ee_pose, config):
    """Four Cartesian goals for display, with no path planning or rejection."""
    from types import SimpleNamespace
    from src.tools.motion.planning import transform_to_pose
    current = _pose_transform(connector.get_ee_pose())
    release = release_pose(relative_transform, closed_ee_pose)
    # Keep the learned target orientation. Clearance is a separately recorded
    # release offset above the learned stable placement, not a model prediction.
    release[2, 3] += config.release_clearance_m
    height = max(config.high_transit_z_m, current[2, 3], release[2, 3] + config.lift_m)
    lift, above, retreat = current.copy(), release.copy(), release.copy()
    lift[2, 3] = above[2, 3] = retreat[2, 3] = height
    return SimpleNamespace(targets=tuple(transform_to_pose(t) for t in (lift, above, release, retreat)),
        target_labels=('lift', 'high_transit', 'release', 'retreat'), release_clearance_m=config.release_clearance_m,
        high_transit_z_m=height)


def plan_anyplace(connector, *, grasp_plan, relative_transform, closed_ee_pose, config, jaw_width_m=.08):
    if grasp_plan.connector is not connector or not grasp_plan.grasp_executed:
        raise MotionPlanningError("AnyPlace needs this connector's executed grasp")
    preview = preview_anyplace(connector, relative_transform=relative_transform,
                              closed_ee_pose=closed_ee_pose, config=config)
    segments, poses, joints, clearance, validated = _plan(
        connector, tuple(_pose_transform(p) for p in preview.targets), np.empty((0, 3)), config, None)
    from src.tools.place.self_collision import check_release_self_collision
    from src.tools.motion.robot_state import _trajectory_end
    # A route solver can choose another IK branch than the prefilter. Check
    # only its third endpoint too, never the intermediate trajectory samples.
    native = getattr(connector.ik, 'check_release_self_collision', None)
    if callable(native):
        self_check = native([_trajectory_end(segments[2])], jaw_width_m=jaw_width_m)[0]
    else:
        self_check = check_release_self_collision(connector.ik._robot_file,
            [_trajectory_end(segments[2])], jaw_width_m=jaw_width_m)[0]
    if not self_check['accepted']:
        error = MotionPlanningError('selected release endpoint has robot self collision')
        error.evidence = self_check
        raise error
    plan = PlacePlan(segments, poses, joints, clearance, validated, connector=connector,
                     transit_policy="anyplace_high", high_transit_z_m=preview.high_transit_z_m,
                     collision_checks_enabled=False, limitations=(
                         'full-arm/world and path self collision checks disabled in simulation',
                         'only third release goal checked for closed-gripper/held-object geometry, static IK and robot self collision',
                         'opening, approach, retreat and joint-path collision checks not performed',
                         "rigid attachment estimated from observed pre-grasp cloud and actual close EE pose; slip unmeasured",))
    plan.release_clearance_m = config.release_clearance_m
    plan.release_self_collision = self_check
    return plan


def execute_anyplace(connector, plan, *, require_release_tracking=False):
    """Optionally separate release tracking from simulation collision policy.

    Arrival tolerances follow the environment profile (15 mm / 0.15 rad fallback).
    These are the same limits as src.tools.motion.planning._execute_checked,
    not placement-model scores or task-dependent contact rules. The caller
    explicitly selects whether to require release tracking.
    """
    if not require_release_tracking:
        return {**execute_place(connector, plan, allow_partial_safety=True), 'release_commanded': True}
    from src.runtime.budget import (gripper_settle_steps, require_motion_budget,
                                    admit_place_motion, retreat_fits_budget)
    plan.release_commanded = False
    from src.tools.motion.planning import _action_boundary, _execution_guard, _execute_checked
    boundary = _action_boundary(plan, 'release')
    _execution_guard(connector, plan, True)
    opening_steps = gripper_settle_steps(connector, 'open', 60)
    admit_place_motion(connector, plan, boundary, opening_steps)
    diagnostics = _execute_checked(connector, plan.segments[:boundary], plan.targets[:boundary],
                                   collision_checks_enabled=plan.collision_checks_enabled)
    release = diagnostics[-1]
    from src.tools.motion.tolerances import cartesian_tolerances
    position_limit, angle_limit = cartesian_tolerances(connector)
    reached = release['position_error_m'] <= position_limit and release['orientation_error_rad'] <= angle_limit
    check = {'accepted': reached, 'position_tolerance_m': position_limit, 'orientation_tolerance_rad': angle_limit,
             'position_error_m': release['position_error_m'],
             'orientation_error_rad': release['orientation_error_rad'],
             'policy': 'existing Cartesian endpoint tolerances, independent of simulation collision policy'}
    result = {'status': 'held_at_incomplete_release', 'release_commanded': False,
              'success_verified': False, 'full_arm_safety_certified': False,
              'execution_diagnostics': diagnostics, 'release_tracking_check': check,
              'collision_checks_enabled': plan.collision_checks_enabled,
              'limitations': list(plan.limitations)}
    if not reached:
        # Keep the hand closed and attachment available for fresh observation
        # and replanning. The attempted route is consumed and cannot be reused.
        return result
    pause = getattr(plan, 'release_pause', None)
    retreat = (plan.segments[boundary:], plan.targets[boundary:])
    if pause is not None:
        # Hand rests at the measured release goal, object still held. A caller
        # may replace the release point with a short corrected move from HERE.
        replacement = pause(plan)
        if replacement is not None and replacement.get('aborted'):
            result['release_correction'] = {'accepted': False, 'aborted': True}
            result['release_tracking_check'] = {**check, 'accepted': False, 'stage': 'refiner_abort',
                                                'policy': 'paused Refiner declined to release at the measured goal'}
            return result
        if replacement is not None:
            require_motion_budget(connector, replacement['segments'], gripper_steps=opening_steps)
            diagnostics += _execute_checked(connector, replacement['segments'], replacement['targets'],
                                            collision_checks_enabled=plan.collision_checks_enabled)
            moved = diagnostics[-1]
            corrected = moved['position_error_m'] <= position_limit and moved['orientation_error_rad'] <= angle_limit
            result['release_correction'] = {'accepted': corrected, 'position_error_m': moved['position_error_m'],
                                            'orientation_error_rad': moved['orientation_error_rad'],
                                            **{k: v for k, v in replacement.items() if k not in ('segments', 'targets', 'retreat')}}
            if not corrected:
                result['release_tracking_check'] = {**check, 'accepted': False, 'stage': 'release_correction',
                                                    'position_error_m': moved['position_error_m'],
                                                    'orientation_error_rad': moved['orientation_error_rad']}
                return result
            if replacement.get('retreat') is not None:
                retreat = replacement['retreat']
    require_motion_budget(connector, (), gripper_steps=opening_steps)
    plan.release_commanded = True
    connector.open_gripper(settle_steps=opening_steps)
    result.update(status='released', release_commanded=True)
    if not retreat_fits_budget(connector, retreat[0]):
        result.update(retreat_executed=False, retreat_reason_code='insufficient_simulation_time')
        return result
    diagnostics += _execute_checked(connector, *retreat,
                                    collision_checks_enabled=plan.collision_checks_enabled)
    result['retreat_executed'] = True
    return result
