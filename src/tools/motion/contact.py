"""Bounded contact motions using the existing planner."""
from contextlib import contextmanager

import numpy as np

from src.tools.motion import planning as motion


TURN_LIMIT_DEG = 180.0


def turn_angle(value):
    if type(value) not in (int, float) or not np.isfinite(value) or not 0 < abs(value) <= TURN_LIMIT_DEG:
        raise ValueError(f'turn requires a nonzero angle within +/-{TURN_LIMIT_DEG:g} degrees')
    return float(value)


def turn_target(current, angle_deg):
    """Rotate about the current TCP's local Z; keep its position fixed.

    Positive follows the right-hand rule about local +Z. This does not infer a
    knob axis or pivot: the grasp must already align the tool and object axes.
    """
    angle = np.deg2rad(turn_angle(angle_deg))
    target = motion._transform(current).copy()
    c, s = np.cos(angle), np.sin(angle)
    target[:3, :3] = target[:3, :3] @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return target


@contextmanager
def _ik_attempts(connector, config):
    """Observe actual IK results in the pinned synchronous planner call.

    A None trajectory alone does not distinguish IK from trajectory failure.
    Restore the solver method even on errors; no solver settings are changed.
    Unsupported planners produce no evidence, so cannot enable the fallback.
    """
    counts = []
    backend = connector.ik
    if (config.collision_checks_enabled or type(backend).__name__ != 'CuRoboBackend'
            or type(backend).__module__ != 'gap.connector.ik'):
        yield counts
        return
    impl = backend._import_impl()
    solver = impl._get_directed_planner(backend._robot_file).ik_solver
    original = solver.solve_pose
    had_override = 'solve_pose' in vars(solver)

    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        counts.append(int(result.success.count_nonzero().item()))
        return result

    solver.solve_pose = observed
    try:
        yield counts
    finally:
        if had_override:
            solver.solve_pose = original
        else:
            del solver.solve_pose


def plan_contact_stroke(connector, target, config, *, record):
    """Try 0, +5, -5, +10, -10 degrees about base Z; execute nothing here.

    Ordered search selects the smallest feasible rotation among these candidates.
    Endpoint position, planner validation and gripper execution stay unchanged.
    Only confirmed IK failure of the original orientation enables alternatives.
    """
    original_error = None
    for angle in (0, 5, -5, 10, -10):
        candidate = target.copy()
        radians = np.deg2rad(angle)
        c, s = np.cos(radians), np.sin(radians)
        candidate[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ target[:3, :3]
        with _ik_attempts(connector, config) as counts:
            try:
                result = motion._plan(connector, (candidate,), np.empty((0, 3)), config, None)
            except motion.MotionPlanningError as exc:
                record({'axis': 'connector_base_z', 'yaw_degrees': angle,
                        'selected': False, 'ik_success_counts': counts.copy(), 'error': str(exc)})
                if angle == 0:
                    if str(exc) != 'planner rejected segment 0' or not counts or any(counts):
                        raise
                    original_error = exc
                continue
        record({'axis': 'connector_base_z', 'yaw_degrees': angle,
                'selected': True, 'ik_success_counts': counts.copy()})
        return result
    raise original_error
