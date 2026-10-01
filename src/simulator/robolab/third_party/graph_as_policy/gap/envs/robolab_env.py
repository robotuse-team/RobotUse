"""RoboLab DROID at the same BaseEnv/EnvConfig boundary as LIBERO.

Isaac imports and application startup happen only on construction. Native
Franka + Robotiq 2F-85 assets, tasks and success predicates are retained.
"""
from __future__ import annotations

from contextlib import contextmanager
import time
import math
import numpy as np
from scipy.spatial.transform import Rotation
from .base_env import BaseEnv
from .registry import EnvConfig
from .robolab_control import (HOME_JOINTS, ROBOTIQ_MAX_WIDTH_M, joint_action,
                              pose_in_base, relative_pose_cm, rigid_pose,
                              validate_joint_target, validate_motion_speed_scale,
                              retime_joint_targets)

CAMERAS = {'agentview': 'over_shoulder_left_camera', 'robot0_eye_in_hand': 'wrist_cam'}


def _matrix(position, quaternion):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix()
    matrix[:3, 3] = position
    return matrix


def _pose_array(matrix):
    q = Rotation.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]
    return np.r_[matrix[:3, 3], q]


class RoboLabEnv(BaseEnv):
    motion_position_tolerance_m = .005
    motion_orientation_tolerance_rad = np.deg2rad(1.)
    motion_joint_tolerance_rad = None
    enforce_motion_tracking = True
    robot_profile = 'robolab_droid_robotiq_2f85'
    max_gripper_width_m = ROBOTIQ_MAX_WIDTH_M
    gripper_open_settle_steps = 20
    gripper_close_settle_steps = 20

    def __init__(self, task='BananaInBowlTask', *, camera_names=None, seed=0,
                 device='cuda:0', headless=True, max_steps=None,
                 camera_width=640, camera_height=360, output_dir=None,
                 motion_speed_scale=1.5, gripper_open_settle_steps=20,
                 gripper_close_settle_steps=20, motion_position_tolerance_m=None,
                 motion_orientation_tolerance_rad=None, motion_joint_tolerance_rad=None,
                 randomize_init_pose=False, init_pose_xy_range_m=.1, initial_seed=None):
        from .robolab_randomization import validate_xy_range
        init_pose_xy_range_m = validate_xy_range(init_pose_xy_range_m)
        if initial_seed is not None:
            seed = initial_seed
        self.initial_pose_randomization = None
        for name, value in (('motion_position_tolerance_m', motion_position_tolerance_m),
                            ('motion_orientation_tolerance_rad', motion_orientation_tolerance_rad),
                            ('motion_joint_tolerance_rad', motion_joint_tolerance_rad)):
            if value is not None:
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(f'{name} must be positive and finite')
                setattr(self, name, float(value))
        self.motion_speed_scale = validate_motion_speed_scale(motion_speed_scale)
        for name, value in (('gripper_open_settle_steps', gripper_open_settle_steps),
                            ('gripper_close_settle_steps', gripper_close_settle_steps)):
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
            setattr(self, name, value)
        self.output_dir = output_dir
        self.camera_names = list(camera_names or CAMERAS)
        if any(name not in CAMERAS for name in self.camera_names):
            raise ValueError(f'RoboLab cameras must be selected from {list(CAMERAS)}')
        if max_steps is not None and max_steps < 1:
            raise ValueError('max_steps must be positive')
        self.max_steps = int(max_steps) if max_steps is not None else None
        self._render_width, self._render_height = int(camera_width), int(camera_height)
        self._record_frames, self._subsample_rate = False, 1
        self._frames = []
        self._closed = False
        self._app = self._env = None
        self._sim_step_count = 0
        self._physics_wall_s = 0.
        self._current_done = self._current_truncated = False
        self._current_reward = 0.
        self._width_target = ROBOTIQ_MAX_WIDTH_M
        try:
            # cv2 before Isaac avoids a known extension-load conflict in containers.
            import cv2  # noqa: F401
            from isaaclab.app import AppLauncher
            # Keep Python alive after close so shared runners can write summaries.
            self._launcher = AppLauncher(headless=headless, enable_cameras=True, device=device, fast_shutdown=False)
            self._app = self._launcher.app
            import torch
            import isaaclab.envs.mdp as mdp
            from isaaclab.utils import configclass
            from robolab.constants import set_output_dir
            # Recorder configs capture this directory during configuration
            # composition, before create_env. Set it before importing factories.
            if output_dir is not None:
                set_output_dir(str(output_dir))
            from robolab.core.environments.factory import create_env_cfg
            from robolab.core.environments.runtime import create_env
            from robolab.core.observations.observation_utils import generate_image_obs_from_cameras, generate_obs_cfg
            from robolab.robots.droid import DroidCfg, contact_gripper
            from robolab.variations.camera import OverShoulderLeftCameraCfg
            from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
            from robolab.variations.lighting import SphereLightCfg
            self._torch = torch
            robot_cfg = DroidCfg()
            camera_cfg = OverShoulderLeftCameraCfg()
            for cam in (robot_cfg.wrist_cam, camera_cfg.over_shoulder_left_camera):
                cam.data_types = ['rgb', 'distance_to_image_plane']
                cam.width, cam.height = self._render_width, self._render_height
            # A continuous driver preserves native mimic joints and enables metric opening.
            @configclass
            class Actions:
                arm = mdp.JointPositionActionCfg(asset_name='robot',
                    joint_names=[f'panda_joint{i}' for i in range(1, 8)], preserve_order=True,
                    scale=1., use_default_offset=False)
                gripper = mdp.JointPositionActionCfg(asset_name='robot',
                    joint_names=['finger_joint'], scale=1., use_default_offset=False)
            # Config composition requires classes; these copies never mutate upstream defaults.
            @configclass
            class Robot(DroidCfg):
                robot = robot_cfg.robot
                wrist_cam = robot_cfg.wrist_cam
            @configclass
            class Front(OverShoulderLeftCameraCfg):
                over_shoulder_left_camera = camera_cfg.over_shoulder_left_camera
            @configclass
            class Wrist:
                wrist_cam = robot_cfg.wrist_cam
            observations = generate_obs_cfg({'image_obs': generate_image_obs_from_cameras([Front, Wrist])()})
            cfg_type = create_env_cfg(task, env_postfix='GapDroid',
                observations_cfg=observations(), actions_cfg=Actions(), robot_cfg=Robot,
                camera_cfg=[Front], lighting_cfg=SphereLightCfg,
                background_cfg=HomeOfficeBackgroundCfg, contact_gripper=contact_gripper,
                dt=1 / 120, render_interval=8, decimation=8, seed=seed)
            cfg = cfg_type()
            self._configure_recording_output(cfg)
            cfg.scene.num_envs = 1
            cfg.sim.device = device
            cfg.seed = seed
            if randomize_init_pose:
                from .robolab_randomization import configure_initial_pose
                self.initial_pose_randomization = configure_initial_pose(cfg, init_pose_xy_range_m)
            self._configure_episode_limit(cfg)
            if output_dir is not None:
                set_output_dir(str(output_dir))
            self._env, self.cfg = create_env(cfg, device=device, seed=seed)
            self._control_freq = 1. / self._env.step_dt
            self.robot = self._env.scene['robot']
            self._arm_ids, _ = self.robot.find_joints([f'panda_joint{i}' for i in range(1, 8)], preserve_order=True)
            self._eef_id = self.robot.find_bodies('base_link')[0][0]
            self.task_language = str(self.cfg.instruction)
            self.reset(seed=seed)
        except BaseException:
            import traceback
            traceback.print_exc()
            self.close()
            raise

    def _configure_recording_output(self, cfg):
        """Bind the native dataset to this episode, including cached configs."""
        if self.output_dir is None:
            return
        from pathlib import Path
        target = str(Path(self.output_dir).resolve())
        recorder = getattr(cfg, 'recorders', None)
        if recorder is None or not hasattr(recorder, 'dataset_export_dir_path'):
            raise RuntimeError('native recorder output cannot be isolated before environment startup')
        recorder.dataset_export_dir_path = target
        if str(Path(recorder.dataset_export_dir_path).resolve()) != target:
            raise RuntimeError('native recorder output directory does not match episode')

    def _configure_episode_limit(self, cfg):
        """Keep native simulation time fixed; max_steps is an additional cap."""
        step_dt = cfg.decimation * cfg.sim.dt
        self._step_duration_s = step_dt
        # Match IsaacLab's max_episode_length rounding and connector step guard.
        native_steps = math.ceil(cfg.episode_length_s / step_dt)
        self.max_steps = native_steps if self.max_steps is None else min(self.max_steps, native_steps)

    def _numpy(self, tensor):
        return tensor.detach().cpu().numpy().copy()

    def _base_matrix(self):
        return _matrix(self._numpy(self.robot.data.root_pos_w[0]), self._numpy(self.robot.data.root_quat_w[0]))

    def ee_matrix(self):
        state = self._numpy(self.robot.data.body_state_w[0, self._eef_id, :7])
        return pose_in_base(self._base_matrix(), _matrix(state[:3], state[3:]))

    def make_path_collision(self, connector, *, clearance_m=.002):
        from robot_skill_selector.robolab_collision import RoboLabPathCollision
        return RoboLabPathCollision(connector, clearance_m=clearance_m)

    def gripper_angle(self):
        jid = self.robot.find_joints('finger_joint')[0][0]
        return float(self.robot.data.joint_pos[0, jid])

    def create_ik_backend(self):
        from .robolab_kinematics import RoboLabIK
        return RoboLabIK(self)

    def joints(self):
        return self._numpy(self.robot.data.joint_pos[0, self._arm_ids])

    def reset(self, *, seed=None, options=None):
        self._env.reset_eval_state()
        self._env.reset(seed=seed)
        from .robolab_randomization import record_initial_pose
        record_initial_pose(self, seed)
        self._sim_step_count = 0
        self._physics_wall_s = 0.
        self._current_reward = 0.
        self._current_done = self._current_truncated = False
        self._width_target = ROBOTIQ_MAX_WIDTH_M
        self._target_joints = self.joints()
        return self.get_observation(), {}

    def simulation_budget(self):
        remaining = max(0, self.max_steps - self._sim_step_count)
        terminal = self._current_done or self._current_truncated or remaining == 0
        return dict(elapsed_s=self._sim_step_count * self._step_duration_s,
            limit_s=self.max_steps * self._step_duration_s,
            remaining_s=0. if terminal else remaining * self._step_duration_s,
            remaining_steps=0 if terminal else remaining, step_duration_s=self._step_duration_s,
            terminal=bool(terminal), reason_code=('simulation_time_limit' if
                self._current_truncated or remaining == 0 else 'episode_terminated') if terminal else None)

    def _check_active(self):
        if self._closed:
            raise RuntimeError('RoboLab environment is closed')
        if self._current_done or self._current_truncated or self._sim_step_count >= self.max_steps:
            raise RuntimeError('RoboLab episode ended; explicit reset required before motion')

    def _step_once(self):
        self._check_active()
        action = joint_action(self._target_joints, self._width_target)
        tensor = self._torch.as_tensor(action, device=self._env.device, dtype=self._torch.float32)[None]
        started = time.monotonic()
        _, reward, done, truncated, _ = self._env.step(tensor)
        self._physics_wall_s += time.monotonic() - started
        self._sim_step_count += 1
        self._current_reward = float(reward[0])
        # Upstream discards termination artifacts during its first two steps.
        # Only its recorded verifier result constitutes task success.
        self._current_done = self.task_completed()
        self._current_truncated = bool(truncated[0]) or self._sim_step_count >= self.max_steps
        if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
            self._record_frame()

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (8,):
            raise ValueError('action is seven absolute joint radians and total gripper width in metres')
        self._target_joints = self._validate_joints(action[:7])
        joint_action(self._target_joints, action[7])  # validate before changing any target
        self._width_target = float(action[7])
        self._step_once()
        return self.get_observation(), self._current_reward, self._current_done, self._current_truncated, {}

    def _validate_joints(self, joints):
        return validate_joint_target(joints, self._numpy(self.robot.data.joint_pos_limits[0, self._arm_ids]))

    @contextmanager
    def stationary_grasp_arrival(self, target_joints):
        """Require a settled grasp endpoint within the existing motion step cap.

        Only the final joint target is guarded, including on the non-streaming
        route. Transit, intermediate waypoints and post-close lift keep their
        existing arrival policy. The yielded evidence is recorded by the caller.
        """
        evidence = dict(accepted=False, linear_speed_limit_m_s=.005,
                        angular_speed_limit_rad_s=math.radians(3.),
                        required_consecutive_steps=5, consecutive_steps=0)
        previous = getattr(self, '_stationary_grasp_arrival', None)
        self._stationary_grasp_arrival = (self._validate_joints(target_joints).copy(), evidence)
        try:
            yield evidence
        finally:
            self._stationary_grasp_arrival = previous

    def move_to_joints_blocking(self, target, tolerance=.001, max_steps=120, arm_id=0):
        if arm_id != 0:
            raise ValueError('DROID has one arm')
        if tolerance <= 0 or not np.isfinite(tolerance) or max_steps < 0:
            raise ValueError('invalid convergence limits')
        if self.motion_joint_tolerance_rad is not None:
            tolerance = self.motion_joint_tolerance_rad
        self._check_active()
        self._target_joints = self._validate_joints(target)
        if max_steps == 0:
            self._step_once()
            return
        # The requested joint tolerance and the Cartesian arrival limits are
        # separate contracts. An arbitrary tighter joint cutoff can reject an
        # otherwise acceptable flange pose under load. FK uses robot state only.
        target_pose = self.create_ik_backend().model.fk(self._target_joints)
        request = getattr(self, '_stationary_grasp_arrival', None)
        arrival = request[1] if request is not None and np.array_equal(request[0], self._target_joints) else None
        if arrival is not None:
            # Sample actual physics steps, never wall time or repeated observations.
            step_dt = self._step_duration_s
            if not np.isfinite(step_dt) or step_dt <= 0:
                raise ValueError('positive finite control step duration required')
            previous_pose = self.ee_matrix().copy()
            arrival.update(accepted=False, consecutive_steps=0, step_duration_s=step_dt)
        for _ in range(max_steps):
            self._step_once()
            actual = self.ee_matrix()
            position_error = float(np.linalg.norm(actual[:3, 3] - target_pose[:3, 3]))
            angle_error = float(Rotation.from_matrix(target_pose[:3, :3] @ actual[:3, :3].T).magnitude())
            near_target = (np.max(np.abs(self.joints() - self._target_joints)) <= tolerance
                           and position_error <= self.motion_position_tolerance_m
                           and angle_error <= self.motion_orientation_tolerance_rad)
            if arrival is not None:
                linear_speed = float(np.linalg.norm(actual[:3, 3] - previous_pose[:3, 3]) / step_dt)
                angular_speed = float(Rotation.from_matrix(
                    actual[:3, :3] @ previous_pose[:3, :3].T).magnitude() / step_dt)
                stopped = (linear_speed <= arrival['linear_speed_limit_m_s']
                           and angular_speed <= arrival['angular_speed_limit_rad_s'])
                count = arrival['consecutive_steps'] + 1 if near_target and stopped else 0
                arrival.update(linear_speed_m_s=linear_speed, angular_speed_rad_s=angular_speed,
                               consecutive_steps=count, accepted=count >= arrival['required_consecutive_steps'])
                previous_pose = actual.copy()
                if arrival['accepted']:
                    return
            elif near_target:
                return
        raise RuntimeError(f'joint motion did not converge (max joint error '
            f'{np.max(np.abs(self.joints() - self._target_joints)):.6f} rad, '
            f'Cartesian error {position_error:.6f} m / {angle_error:.6f} rad)')

    def stream_joint_trajectory(self, waypoints, *, settle_tolerance=.001, settle_max_steps=120, arm_id=0):
        targets = [self._validate_joints(q) for q in waypoints]
        if not targets:
            raise ValueError('empty joint trajectory')
        if arm_id != 0:
            raise ValueError('DROID has one arm')
        self._check_active()
        for q in retime_joint_targets(self.joints(), targets, self.motion_speed_scale):
            self._target_joints = q
            self._step_once()
        self.move_to_joints_blocking(targets[-1], settle_tolerance, settle_max_steps, arm_id)

    def move_to_pose(self, target, *, tolerance_m=.0005, tolerance_rad=.005, max_steps=120):
        """Native differential IK acceptance path: measured flange target in base frame.

        Does not teleport joints or read object state. F2 path planning remains
        a separate capability; this method verifies native control and frames.
        """
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
        from isaaclab.utils.math import subtract_frame_transforms
        target = rigid_pose(target)
        ctl = DifferentialIKController(DifferentialIKControllerCfg(command_type='pose',
            use_relative_mode=False, ik_method='dls'), num_envs=1, device=self._env.device)
        cmd = self._torch.as_tensor(_pose_array(target), dtype=self._torch.float32, device=self._env.device)[None]
        ctl.set_command(cmd)
        for _ in range(max_steps):
            robot = self.robot
            pos, quat = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                robot.data.body_pos_w[:, self._eef_id], robot.data.body_quat_w[:, self._eef_id])
            jac = robot.root_physx_view.get_jacobians()[:, self._eef_id - 1, :, self._arm_ids].clone()
            # PhysX Jacobians use world axes; controller errors use root axes.
            r = self._torch.as_tensor(self._base_matrix()[:3, :3].T, dtype=jac.dtype, device=jac.device)
            jac[:, :3] = r @ jac[:, :3]
            jac[:, 3:] = r @ jac[:, 3:]
            center = (r @ (robot.data.body_com_pos_w[:, self._eef_id]
                           - robot.data.root_pos_w).unsqueeze(-1)).squeeze(-1)
            jac[:, :3] += self._torch.linalg.cross(jac[:, 3:].transpose(1, 2),
                (pos - center)[:, None, :], dim=-1).transpose(1, 2)
            q = ctl.compute(pos, quat, jac, robot.data.joint_pos[:, self._arm_ids])
            self._target_joints = self._validate_joints(self._numpy(q[0]))
            self._step_once()
            actual = self.ee_matrix()
            pos_error = np.linalg.norm(actual[:3, 3] - target[:3, 3])
            rot_error = Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).magnitude()
            if pos_error <= tolerance_m and rot_error <= tolerance_rad:
                return {'position_error_m': float(pos_error), 'rotation_error_rad': float(rot_error)}
        raise RuntimeError(f'Cartesian motion did not converge: {pos_error:.6f} m, {rot_error:.6f} rad')

    def move_relative_cm(self, delta_cm, **kwargs):
        return self.move_to_pose(relative_pose_cm(self.ee_matrix(), delta_cm), **kwargs)

    def _set_gripper(self, fraction, arm_id=0):
        if not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError('gripper fraction must be in [0, 1]')
        self._set_gripper_width(float(fraction) * ROBOTIQ_MAX_WIDTH_M, arm_id)

    def _set_gripper_width(self, width_m, arm_id=0):
        if arm_id != 0:
            raise ValueError('DROID has one arm')
        joint_action(self._target_joints, width_m)
        self._width_target = float(width_m)

    def gripper_width(self):
        jid = self.robot.find_joints('finger_joint')[0][0]
        angle = float(self.robot.data.joint_pos[0, jid])
        return float(np.clip(.010 + .1143 * np.sin(.715 - angle), 0, ROBOTIQ_MAX_WIDTH_M))

    def get_observation(self):
        width = self.gripper_width()
        result = {'robot_joint_pos_0': np.r_[self.joints(), width / ROBOTIQ_MAX_WIDTH_M],
                  'robot_cartesian_pos_0': np.r_[_pose_array(self.ee_matrix()), width / ROBOTIQ_MAX_WIDTH_M],
                  'robot_gripper_width_m_0': width}
        for alias in self.camera_names:
            data = self._env.scene[CAMERAS[alias]].data
            pose = self.camera_optical_matrix(alias)
            depth = self._numpy(data.output['distance_to_image_plane'][0]).squeeze(-1)
            # Non-return rays remain unknown (zero); never fabricate far surfaces.
            depth[~np.isfinite(depth) | (depth <= 0)] = 0
            result[alias] = {'pose': _pose_array(pose), 'pose_mat': pose,
                'intrinsics': self._numpy(data.intrinsic_matrices[0]),
                'images': {'rgb': self._numpy(data.output['rgb'][0])[..., :3], 'depth': depth}}
        return result

    def camera_optical_matrix(self, name):
        camera = self._env.scene[CAMERAS[name]]
        if name == 'robot0_eye_in_hand':
            # This sensor is rigidly mounted directly under the measured
            # base_link flange. IsaacLab's USD and XForm-view camera poses can
            # remain at initialization under Fabric; use robot proprioception
            # and the authored camera mount, never a scene-object pose.
            offset = camera.cfg.offset
            if not camera.cfg.prim_path.endswith('/base_link/wrist_cam') or offset.convention != 'opengl':
                raise ValueError('unsupported RoboLab wrist camera mounting contract')
            flange_from_opengl = _matrix(offset.pos, offset.rot)
            return self.ee_matrix() @ flange_from_opengl @ np.diag([1., -1., -1., 1.])
        data = camera.data
        return pose_in_base(self._base_matrix(), _matrix(self._numpy(data.pos_w[0]),
            self._numpy(data.quat_w_ros[0])))

    def capture_rgbd(self, *, refresh=True):
        from gap_core.types import matrix_to_pose
        if refresh:
            self.refresh_camera_obs()
        observation = self.get_observation()
        return [dict(name=name, **observation[name]['images'],
            intrinsics=observation[name]['intrinsics'], pose=matrix_to_pose(observation[name]['pose_mat']))
            for name in self.camera_names]

    def render_rgb(self, name):
        return self._numpy(self._env.scene[CAMERAS[name]].data.output['rgb'][0])[..., :3]

    def get_simulation_time_s(self):
        """Native Isaac physics clock for synchronized capture and recording."""
        return float(self._env.sim.current_time)

    def get_current_time_s(self):
        return float(self._env.sim.current_time)

    def refresh_camera_obs(self):
        # Render and refresh sensors without advancing the physics clock.
        self._env.sim.render()
        for name in CAMERAS.values():
            self._env.scene[name].update(0., force_recompute=True)

    def set_cameras_active(self, active):
        # Cameras remain synchronous with physics; the common connector may nest this hook.
        pass

    def compute_reward(self):
        return self._current_reward

    def task_completed(self):
        return self._env.get_env_results()[0]['success'] is True

    def get_latency_info(self):
        return {'simulator_steps': self._sim_step_count, 'physics_wall_s': self._physics_wall_s,
                'simulation_time_s': self.get_current_time_s(),
                'motion_speed_scale': self.motion_speed_scale}

    def enable_video_capture(self, enabled=True, *, clear=True):
        self._record_frames = bool(enabled)
        if clear:
            self._frames.clear()

    def _record_frame(self):
        self._frames.append(self.get_observation()[self.camera_names[0]]['images']['rgb'])

    def get_video_frames(self, *, clear=False):
        frames = list(self._frames)
        if clear:
            self._frames.clear()
        return frames

    def close(self):
        if getattr(self, '_closed', False):
            return
        self._closed = True
        if self._env is not None:
            self._env.close()
        if self._app is not None:
            import sys
            cli = sys.modules.get('robot_skill_selector.robolab_cli')
            # Standalone ownership uses Kit's fast-exit lifetime, but waits until
            # the runner has saved its summary and closed model workers. Calling
            # app.close here either exits too early or runs unstable plugin
            # finalization. Embedded/library users still control full app close.
            if not getattr(cli, 'OWNS_PROCESS', False):
                self._app.close()


def make_env(suite_name, task_id=0, camera_names=None, enable_render=False, **extra):
    if task_id != 0:
        raise ValueError('RoboLab selects a task class by name; the variant index must be 0')
    env = RoboLabEnv(suite_name, camera_names=camera_names, **extra)
    return env, EnvConfig(arm_dof=7, num_arms=1, action_mode='absolute_joints',
        control_freq=env._control_freq, home_joints=HOME_JOINTS,
        tcp_offset=(0., 0., 0.), robot_urdf_path=None,
        default_cameras=tuple(env.camera_names), is_real=False)
