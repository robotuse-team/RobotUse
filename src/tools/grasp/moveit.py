"""MoveIt Grasps I/O adapter; upstream owns sampling and direction scoring.

Measured target points enter OBB fitting; native MoveIt scene/path checks precede
the capped cuRobo loop. The caller owns path checks, opening calculation,
preview, refinement and execution flow.
"""
from pathlib import Path
import ctypes
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import time

import numpy as np

from src.tools.grasp.preference import normalize_preference, normalize_grasp_type
from src.tools.grasp.prediction import GraspPrediction, checked_points

from src.runtime.paths import REPOSITORY_ROOT as ROOT
MOVEIT_SOURCE = Path(__file__).resolve().parent/'moveit_grasps'
RUNTIME_ROOT = Path(os.environ.get('ROBOTUSE_RUNTIME_ROOT', ROOT/'runtime'))
MOVEIT_RUNTIME = RUNTIME_ROOT/'moveit-grasps'
DEFAULT_EXECUTABLE = MOVEIT_RUNTIME/'build/generate_candidates'
DEFAULT_URDF = MOVEIT_SOURCE/'config/generator.urdf'
DEFAULT_PREFIX = RUNTIME_ROOT/'moveit-env'
DEFAULT_SCENE_EXECUTABLE = MOVEIT_RUNTIME/'build/scene_filter'
DEFAULT_SCENE_ASSETS = MOVEIT_SOURCE/'config'
UPSTREAM_REVISION = 'e70b88f97ed223dbcaecd3343316b6900029dacd'


class MoveItGraspsBackend:
    def __init__(self, *, executable=None, urdf=None, prefix=None, max_path_checks=48,
                 ros_master_uri='http://127.0.0.1:11329', scene_filter_enabled=True,
                 scene_executable=None, scene_assets=None, robot_profile='libero_panda'):
        if robot_profile not in ('libero_panda', 'robolab_robotiq'):
            raise ValueError('unsupported MoveIt robot profile')
        self.robot_profile = robot_profile
        self.executable = Path(executable or DEFAULT_EXECUTABLE)
        self.urdf = Path(urdf or DEFAULT_URDF)
        if robot_profile == 'robolab_robotiq' and self.urdf == DEFAULT_URDF:
            self.urdf = MOVEIT_SOURCE/'config/robotiq_sampling.urdf'
        self.prefix = Path(prefix or DEFAULT_PREFIX)
        if type(max_path_checks) is not int or max_path_checks < 1:
            raise ValueError('positive MoveIt path-check cap required')
        self.max_path_checks = max_path_checks
        self.ros_master_uri = ros_master_uri
        self.preferred_direction = None
        self.grasp_type = 'face'
        self.motion_policy = 'source-order'
        self.motion_score_tolerance = 0.
        self.last_generation = {}
        self.pose_dedup = None
        self.scene_filter_enabled = scene_filter_enabled
        self.scene_filter = None
        self.scene_executable = Path(scene_executable or DEFAULT_SCENE_EXECUTABLE)
        self.scene_assets = Path(scene_assets or DEFAULT_SCENE_ASSETS)

    def preflight(self):
        """Validate the caller's runtime before any candidate budget is reserved."""
        problems = []
        if not (self.prefix/'lib').is_dir():
            problems.append(f'MoveIt environment missing: {self.prefix}')
        usb = self.prefix/'lib/libusb-1.0.so.0'
        try:
            if usb.is_file():
                ctypes.CDLL(str(usb), mode=ctypes.RTLD_GLOBAL)
            for name in ('open3d', 'scipy.spatial.transform', 'cv2'):
                importlib.import_module(name)
        except (ImportError, OSError) as exc:
            problems.append(f'caller Python dependency: {exc}')
        assets = [self.urdf]
        binaries = [self.executable]
        if self.scene_filter_enabled:
            binaries.append(self.scene_executable)
            if self.robot_profile == 'libero_panda':
                assets += [self.scene_assets/'panda.urdf', self.scene_assets/'panda.srdf']
                assets.append(self.prefix/'share/franka_description/package.xml')
        for path in assets:
            if not path.is_file():
                problems.append(f'asset missing: {path}')
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = str(self.prefix/'lib')
        for path in binaries:
            if not path.is_file() or not os.access(path, os.X_OK):
                problems.append(f'executable missing or not executable: {path}')
                continue
            try:
                result = subprocess.run(['ldd', str(path)], env=env, capture_output=True,
                                        text=True, timeout=15)
                if result.returncode or 'not found' in result.stdout:
                    problems.append(f'native dependencies for {path}: {result.stdout} {result.stderr}')
                elif self.robot_profile == 'robolab_robotiq':
                    capability = subprocess.run([str(path), '--supports-robolab'], env=env,
                        capture_output=True, text=True, timeout=15)
                    if capability.returncode:
                        problems.append(f'outdated binary lacks RoboLab profile: {path}; rebuild MoveIt tools')
            except (OSError, subprocess.TimeoutExpired) as exc:
                problems.append(f'native dependency check for {path}: {exc}')
        if problems:
            raise RuntimeError('MoveIt runtime preflight failed; configure the executables and runtime prefix '
                               'with their required Python and native dependencies. ' + '; '.join(problems))

    def prediction(self, row):
        native = self.robot_profile == 'robolab_robotiq'
        return GraspPrediction(np.asarray(row['hand']), float(row['score']),
            gripper_adapter=None if native else 'libero_panda',
            gripper_name='robotiq_2f_85' if native else 'franka_panda')

    def predict_with_path_filter(self, object_points, scene_points, accept_candidate, *, rank_batch=None):
        from scipy.spatial.transform import Rotation
        preference = normalize_preference(self.preferred_direction)
        family = normalize_grasp_type(self.grasp_type) or 'face'
        obj = checked_points(object_points, minimum=100)
        scene = checked_points(scene_points)
        output = Path(self.output_dir)
        output.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(output/'observed_points.npz', object_points=obj, scene_points=scene)
        # Use the pinned OBB fit with Open3D dependencies from the configured runtime.
        usb = self.prefix/'lib/libusb-1.0.so.0'
        if usb.is_file():
            ctypes.CDLL(str(usb), mode=ctypes.RTLD_GLOBAL)
        spec = importlib.util.spec_from_file_location('moveit_measured_geometry',
            ROOT/'src/tools/grasp/third_party/open_robot_skills/tools/geometry/_impl.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        obb = module.compute_obb(obj)
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat([obb['orientation'][k] for k in 'xyzw']).as_matrix()
        matrix[:3, 3] = [obb['center'][k] for k in 'xyz']
        size = np.array([2*obb['extent'][k] for k in 'xyz'])
        np.savetxt(output/'obb.txt', np.r_[matrix.ravel(), size][None, :])
        (output/'obb.json').write_text(json.dumps(dict(obb=obb, point_count=len(obj),
            source='measured RGB-D target; compute_obb; fitted cuboid is an approximation'), indent=2))
        env = os.environ.copy()
        env.update(ROS_MASTER_URI=self.ros_master_uri, ROS_HOSTNAME='127.0.0.1',
            LD_LIBRARY_PATH=str(self.prefix/'lib'), ROS_PACKAGE_PATH=str(self.prefix/'share'))
        started = time.monotonic()
        command = [str(self.executable), str(output/'obb.txt'), str(self.urdf),
                   preference or 'none', str(output/'upstream_candidates.json'), family]
        if self.robot_profile == 'robolab_robotiq':
            command.append(self.robot_profile)
        with (output/'generator.log').open('x') as log:
            subprocess.run(command,
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
        rows = json.loads((output/'upstream_candidates.json').read_text())
        # Generic TCP closes along Y; both canonical hand frames close along X.
        # The jaw centre offset is profile-specific, never a Panda retargeting.
        tcp_to_hand = np.eye(4)
        tcp_to_hand[:3, :3] = Rotation.from_euler('z', -np.pi/2).as_matrix()
        tcp_to_hand[2, 3] = -.136 if self.robot_profile == 'robolab_robotiq' else -.099
        unique = {}
        for row in rows:
            hand = np.asarray(row['tcp'], dtype=float) @ tcp_to_hand
            key = tuple(np.round(hand.ravel(), 7))+(round(row['open_width_m'], 7),)
            if key not in unique or row['score'] > unique[key]['score']:
                unique[key] = dict(row, hand=hand.tolist())
        ranked = sorted(unique.values(), key=lambda row: -row['score'])
        for index, row in enumerate(ranked, 1):
            row['original_rank'] = index
        # Preserve the complete generated batch even if the native filter errors.
        (output/'ranked_candidates.json').write_text(json.dumps(ranked, indent=2))
        eligible, prefilter = ranked, None
        dedup_audit = None
        if self.pose_dedup is not None:
            indices, dedup_audit = self.pose_dedup.select(
                [r['hand'] for r in ranked], ids=[r['original_rank'] for r in ranked],
                widths=[r['open_width_m'] for r in ranked])
            eligible = [ranked[i] for i in indices]
            (output/'pose_dedup.json').write_text(json.dumps(dedup_audit, indent=2))
        retained_rows = eligible
        if self.scene_filter_enabled:
            if self.scene_filter is None:
                raise RuntimeError('MoveIt scene filter requires current robot/scene context')
            eligible, prefilter = self.scene_filter(eligible, output)
        from src.tools.grasp.motion_selection import (validate_policy, motion_cost,
            rank_symmetric_candidates, same_parallel_jaw_grasp)
        validate_policy(self.motion_policy, self.motion_score_tolerance)
        if self.motion_policy == 'low-motion':
            for row in eligible:
                row['motion_cost'] = motion_cost(row['hand'], self.current_ee, self.grasp_to_ee)
            eligible = rank_symmetric_candidates(eligible, tolerance=self.motion_score_tolerance)
        checked, accepted, checked_ranks, accepted_ranks = 0, [], [], []
        accepted_rows, skipped = [], []
        for row in eligible:
            if checked >= self.max_path_checks:
                break
            if self.motion_policy == 'low-motion':
                representative = next((other for other in accepted_rows
                    if abs(other['score']-row['score']) <= self.motion_score_tolerance+1e-12
                    and other['motion_cost']['rotation_deg'] <= row['motion_cost']['rotation_deg']+1e-6
                    and same_parallel_jaw_grasp(row, other)), None)
                if representative is not None:
                    skipped.append(dict(original_rank=row['original_rank'],
                        representative_original_rank=representative['original_rank']))
                    continue
            prediction = self.prediction(row)
            checked += 1
            checked_ranks.append(row['original_rank'])
            if accept_candidate(prediction):
                accepted.append(prediction)
                accepted_rows.append(row)
                accepted_ranks.append(row['original_rank'])
            if len(accepted) >= self.target_candidates:
                break
        self.refinement_only_rows = (sorted(retained_rows, key=lambda r: (not r.get('moveit_scene_result', {}).get('grasp_ik', False), r['original_rank']))[:6] if not accepted and prefilter else [])
        if getattr(self, 'expose_partial_diagnostics', False):
            # Only real prefilter rejections: path rejections were already
            # published by the caller, and untested/accepted poses are not failures.
            rejected_rows = [r for r in ranked
                if r.get('moveit_scene_result', {}).get('status') not in (None, 'accepted')]
            self.refinement_only_rows = sorted(rejected_rows,
                key=lambda r: (not r['moveit_scene_result'].get('grasp_ik', False), r['original_rank']))[:6]
        self.last_generation = dict(generator='moveit_grasps', robot_profile=self.robot_profile, upstream_commit=UPSTREAM_REVISION,
            preferred_direction=preference, grasp_type=family, raw_count=len(rows), unique_count=len(ranked),
            path_checked=checked, accepted_count=len(accepted), path_check_cap=self.max_path_checks,
            unchecked_count=len(eligible)-checked, target_candidates=self.target_candidates,
            pose_dedup=dedup_audit,
            moveit_scene_filter=prefilter, eligible_count=len(eligible),
            path_checked_original_ranks=checked_ranks, accepted_original_ranks=accepted_ranks,
            motion_policy=self.motion_policy, motion_score_tolerance=self.motion_score_tolerance,
            motion_selection_scope='equivalent_180_degree_jaw_symmetries_only',
            symmetry_skipped_count=len(skipped),
            elapsed_s=time.monotonic()-started, tcp_to_hand=tcp_to_hand.tolist(),
            executable_sha256=hashlib.sha256(self.executable.read_bytes()).hexdigest(),
            opening_policy=('native Robotiq full opening .085m; upstream width saved as evidence'
                if self.robot_profile == 'robolab_robotiq' else
                'measured-target adaptive opening shared with GraspGen; upstream width saved as evidence'),
            ranking='upstream orientation Z weight 10, depth 1, width .1; maximum score over four XY azimuths for horizontal')
        # Keep the full ordering/audit in the artifact, not the agent's payload.
        (output/'generation.json').write_text(json.dumps(dict(**self.last_generation, candidates=ranked,
            motion_order_original_ranks=[r['original_rank'] for r in eligible],
            skipped_symmetric_candidates=skipped), indent=2))
        return tuple(accepted)
