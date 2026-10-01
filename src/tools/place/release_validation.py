"""Only the third (closed-gripper release) goal filters placement candidates."""
from functools import lru_cache

import numpy as np
from scipy.spatial import cKDTree

from src.tools.place.collision import gripper_meshes, mesh_points_collision
from src.tools.pose_editor.inspection import transform_points
from src.tools.motion.planning import _transform, transform_to_pose

POLICY = 'release_goal_only'
SCOPE = ('third goal only: closed gripper and held object versus observed scene, static IK, and environment-profile robot self collision; '
         'no pregrasp placement gate, opening sweep, approach, retreat or joint-path collision checks')


def check_release_geometry(object_points, scene_points, delta, closed_ee, grasp_to_ee,
                           *, jaw_width_m, release_clearance_m=.015, mesh_source=None):
    delta = _transform(delta)
    release_ee = delta @ _transform(closed_ee)
    release_ee[2, 3] += release_clearance_m
    hand = release_ee @ np.linalg.inv(_transform(grasp_to_ee))
    scene = np.asarray(scene_points)
    local = transform_points(scene, np.linalg.inv(hand))
    result = {'compatible': True, 'stage': 'release_goal', 'policy': POLICY, 'scope': SCOPE,
              'release_ee_pose': release_ee.tolist(), 'release_hand_pose': hand.tolist(),
              'jaw_width_m': float(jaw_width_m), 'release_clearance_m': release_clearance_m}
    for body, mesh in gripper_meshes(float(jaw_width_m), mesh_source=mesh_source).items():
        hit = mesh_points_collision(mesh, local)
        if hit:
            return {**result, 'compatible': False, 'reason_code': 'gripper_scene_collision', 'body': body, **hit}
    payload = transform_points(object_points, delta) + [0, 0, release_clearance_m]
    distances, _ = cKDTree(scene).query(payload)
    if distances.min() <= .002:
        return {**result, 'compatible': False, 'reason_code': 'payload_scene_collision',
                'min_distance_m': float(distances.min()), 'colliding_points': int((distances <= .002).sum())}
    return result


@lru_cache(maxsize=4)
def _ik_solver(robot_file, batch_size):
    from curobo._src.solver.seed_ik.seed_ik_solver import SeedIKSolver
    from curobo._src.solver.seed_ik.seed_ik_solver_cfg import SeedIKSolverCfg
    # Native LM IK has only pose error and joint limits. The higher-level
    # optimizer in this installed version fails with empty costs when all
    # collision terms are disabled; do not substitute a trajectory planner.
    config = SeedIKSolverCfg.create(robot=robot_file, num_seeds=32, sampler_seed=7,
        max_iterations=32, inner_iterations=16, use_cuda_graph=False,
        position_tolerance=.005, orientation_tolerance=.05)
    return SeedIKSolver(config)


def release_goal_ik(backend, release_ee_poses, *, jaw_width_m=.08):
    """cuRobo native IK; never call backend.solve_ik (that plans a trajectory)."""
    native = getattr(backend, 'release_goal_ik', None)
    if callable(native):
        return native(release_ee_poses, jaw_width_m=jaw_width_m)
    import torch
    from curobo.types import Pose, GoalToolPose
    from scipy.spatial.transform import Rotation
    if not len(release_ee_poses):
        return []
    solver = _ik_solver(backend._robot_file, len(release_ee_poses))
    converted = [backend._pose_for_curobo(transform_to_pose(pose), 0) for pose in release_ee_poses]
    position = np.stack([p for p, q in converted])
    quaternion = np.stack([q for p, q in converted])
    goals = Pose(position=torch.tensor(position, device='cuda', dtype=torch.float32),
                 quaternion=torch.tensor(quaternion, device='cuda', dtype=torch.float32))
    solver.reset_seed()
    goal = GoalToolPose.from_poses({solver.kinematics.tool_frames[0]: goals}, num_goalset=1)
    solved = solver.solve_batch(goal, return_seeds=1)
    joints = solved.js_solution.position.detach().cpu().numpy().reshape(len(converted), -1)
    fk = solver.compute_kinematics(solved.js_solution).tool_poses
    actual_xyz = fk.position.detach().cpu().numpy().reshape(-1, 3)
    actual_quat = fk.quaternion.detach().cpu().numpy().reshape(-1, 4)
    position_error = np.linalg.norm(actual_xyz-position, axis=1)
    desired_rotation = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]])
    actual_rotation = Rotation.from_quat(actual_quat[:, [1, 2, 3, 0]])
    orientation_error = (desired_rotation.inv()*actual_rotation).magnitude()
    success = solved.success.detach().cpu().numpy().reshape(len(converted), -1).all(axis=1)
    limits = solver.joint_limits.position.detach().cpu().numpy()
    within_limits = ((joints >= limits[0]) & (joints <= limits[1])).all(axis=1)
    from src.tools.place.self_collision import check_release_self_collision
    self_checks = check_release_self_collision(backend._robot_file, joints, jaw_width_m=jaw_width_m)
    return [{'accepted': bool(success[i] and within_limits[i] and position_error[i] <= .005 and orientation_error[i] <= .05 and self_checks[i]['accepted']),
             'stage': 'release_goal_ik', 'joint_positions': joints[i].tolist(),
             'position_error_m': float(position_error[i]), 'orientation_error_rad': float(orientation_error[i]),
             'num_seeds': 32, 'random_seed': 7, 'solver': 'cuRobo native LM SeedIKSolver',
             'within_joint_limits': bool(within_limits[i]), 'self_collision_check': True,
             'self_collision':self_checks[i], 'path_planned': False}
            for i in range(len(converted))]
