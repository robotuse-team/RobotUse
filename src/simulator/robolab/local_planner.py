"""Bounded native-robot transit search against measured scene points.

Tries a straight Cartesian route and six explicit midpoint detours. Each route
uses native IK and the common whole-robot sweep check; no Panda planner model.
This is a finite local search, not cuRobo or a complete free-space planner.
"""
from types import SimpleNamespace
import numpy as np
from src.tools.motion.path_collision import make_candidate_path_collision
from src.tools.motion.planning import MotionPlanningError


def plan_observed_transit(connector, target, start_joints, world):
    ik = connector.ik
    start = ik.model.fk(start_joints)
    target = np.asarray(target, dtype=float)
    scene = np.asarray(world.observed_points, dtype=float).reshape(-1, 3)
    checker = make_candidate_path_collision(connector)
    width = connector.env.gripper_width()
    routes = [[target]]
    for axis, sign in ((2, 1), (0, 1), (0, -1), (1, 1), (1, -1), (2, -1)):
        middle = start.copy()
        middle[:3, 3] = (start[:3, 3] + target[:3, 3]) / 2
        middle[axis, 3] += .10 * sign
        routes.append([middle, target])
    attempts = []
    for goals in routes:
        q, pose, segments = list(start_joints), start, []
        for goal in goals:
            segment = ik.plan_linear(pose, goal, seed_joints=q)
            if segment is None:
                break
            segments.append(segment)
            q, pose = segment['waypoints'][-1]['positions'], goal
        if len(segments) != len(goals):
            attempts.append({'accepted': False, 'kind': 'ik'})
            continue
        labels = tuple(f'transit_{i}' for i in range(len(segments)))
        plan = SimpleNamespace(start_joints=start_joints, segments=segments, target_labels=labels)
        checked = checker.check(plan, scene, stop_label=labels[-1], jaw_width_m=width)
        attempts.append(checked)
        if checked['accepted']:
            return {'waypoints': [row for segment in segments for row in segment['waypoints']],
                'planner': 'native IK with bounded Cartesian detours', 'attempts': attempts}
    raise MotionPlanningError('native transit search found no checked route',
        planning_feedback={'kind': 'planning', 'planner_reason_code': 'world_route_failed', 'attempts': attempts})
