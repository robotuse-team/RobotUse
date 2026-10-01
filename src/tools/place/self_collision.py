"""Native cuRobo self collision at one release joint configuration only."""
from functools import lru_cache
import hashlib
from pathlib import Path
import numpy as np


@lru_cache(maxsize=8)
def _checker(robot_file, jaw_width_m):
    from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
    from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml
    path = Path(join_path(get_robot_configs_path(), robot_file))
    robot = load_yaml(str(path))['robot_cfg']
    # Same symmetric closed-jaw mesh convention as the observed-scene check.
    # Retain native link-pair ignore rules, sphere radii and per-link padding.
    robot['kinematics']['lock_joints'].update(
        panda_finger_joint1=jaw_width_m/2, panda_finger_joint2=jaw_width_m/2)
    config = RobotCollisionCheckerCfg.load_from_config(robot_config=robot, scene_model=None,
        collision_activation_distance=0.0, self_collision_activation_distance=0.0)
    checker = RobotCollisionChecker(config)
    return checker, {'robot_file': str(path), 'robot_config_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'modeled_jaw_width_m':jaw_width_m, 'geometry':'native cuRobo collision spheres',
        'pair_policy':'native robot self_collision_ignore and self_collision_buffer retained',
        'scene_collision_checked':False, 'joint_path_checked':False,
        'sphere_count':checker.kinematics.get_self_collision_config().num_spheres,
        'checked_sphere_pair_count':len(checker.kinematics.get_self_collision_config().collision_pairs)}


def check_release_self_collision(robot_file, joint_positions, *, jaw_width_m):
    """Check supplied static IK configurations; no IK or trajectory fallback."""
    import torch
    q = np.asarray(joint_positions, dtype=float)
    if q.ndim != 2 or q.shape[1] != 7 or not np.isfinite(q).all():
        raise ValueError('self collision requires finite N x 7 release joints')
    if not 0 <= jaw_width_m <= .08:
        raise ValueError('closed jaw width outside Panda range')
    if not len(q):
        return []
    checker, metadata = _checker(robot_file, float(jaw_width_m))
    checker.setup_batch_tensors(len(q), 1)
    joints = torch.tensor(q, device=checker.device_cfg.device, dtype=checker.device_cfg.dtype)[:, None, :]
    state = checker.get_kinematics(joints)
    # Native kernel returns maximum positive penetration; 0 means no detected
    # penetration for configured pairs. This is not a signed clearance distance.
    penetration = checker.get_self_collision_distance(state.robot_spheres).detach().cpu().numpy().reshape(len(q))
    if not np.isfinite(penetration).all():
        raise RuntimeError('nonfinite native self-collision result')
    return [{'accepted':bool(value <= 0), 'stage':'release_goal_self_collision',
        'maximum_padded_penetration_m':float(value), 'self_collision_check':True,
        'reason_code':'clear' if value <= 0 else 'robot_self_collision', **metadata} for value in penetration]
