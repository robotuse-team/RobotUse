"""MoveIt-native scene/path prefilter before the bounded cuRobo candidate loop."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation

from src.tools.grasp.moveit import DEFAULT_SCENE_EXECUTABLE, DEFAULT_SCENE_ASSETS

class MoveItSceneFilter:
    def __init__(self, backend, geometry, point_ref, *, prefix, ros_master_uri, executable=None, assets=None, robot_profile='libero_panda'):
        if robot_profile not in ('libero_panda', 'robolab_robotiq'):
            raise ValueError('unsupported MoveIt scene profile')
        self.robot_profile = robot_profile
        self.backend, self.geometry, self.point_ref = backend, geometry, point_ref
        self.prefix, self.ros_master_uri = Path(prefix), ros_master_uri
        self.executable = Path(executable or DEFAULT_SCENE_EXECUTABLE)
        self.assets = Path(assets or DEFAULT_SCENE_ASSETS)

    def __call__(self, ranked, output):
        from src.tools.motion.planning import transform_to_pose, _pose_transform
        from src.tools.motion.robot_state import _current_robot_state
        backend = self.backend
        native = self.robot_profile == 'robolab_robotiq'
        out = Path(output)/'moveit_scene_filter'
        out.mkdir(exist_ok=False)
        checker, scene = backend._candidate_path_scene(self.point_ref)
        scene_relaxed = bool(getattr(backend, '_relax_grasp_generation', False))
        observed_count = len(scene)
        if scene_relaxed:
            np.savetxt(out/'observed-scene.xyz', scene)
            scene = np.empty((0, 3))
        pose, joints = _current_robot_state(backend.connector)
        def link_pose(p):
            if native:
                return _pose_transform(p) @ np.linalg.inv(backend.grasp_to_ee)
            pos, quat = backend.connector.ik._pose_for_curobo(p, 0)
            t = np.eye(4)
            t[:3, 3] = pos
            t[:3, :3] = Rotation.from_quat([*quat[1:], quat[0]]).as_matrix()
            return t
        np.savetxt(out/'scene.xyz', scene)
        np.savetxt(out/'start.txt', np.r_[joints, link_pose(pose).ravel()][None])
        inputs = []
        for row in ranked:
            hand = np.asarray(row['hand'])
            options = backend._grasp_options(hand, self.geometry,
                SimpleNamespace(gripper_adapter=None if native else 'libero_panda'), 'moveit_prefilter')
            inputs.append(np.r_[options['open_width_m'], link_pose(transform_to_pose(hand@backend.grasp_to_ee)).ravel()])
        np.savetxt(out/'candidates.txt', np.asarray(inputs).reshape(-1, 17))
        assets = self.assets
        binary = self.executable
        if native:
            from src.simulator.robolab.moveit_model import export_moveit_model
            model_dir = getattr(backend, '_moveit_native_model_dir', None)
            if model_dir is None:
                model_dir = export_moveit_model(checker.native, Path(backend.output_dir)/'moveit_native_model',
                    asset_path=backend.gripper_assets.asset_path)
                backend._moveit_native_model_dir = model_dir
            urdf, srdf = model_dir/'robot.urdf', model_dir/'robot.srdf'
            np.savetxt(out/'start-width.txt', [backend.connector.env.gripper_width()])
        else:
            urdf, srdf = assets/'panda.urdf', assets/'panda.srdf'
        command = [str(binary), str(urdf), str(srdf), str(out), str(out/'results.jsonl')]
        if native:
            command.append(self.robot_profile)
        env = os.environ.copy()
        env.update(CMAKE_PREFIX_PATH=str(self.prefix.resolve()), ROS_MASTER_URI=self.ros_master_uri,
            ROS_HOSTNAME='127.0.0.1', ROS_PACKAGE_PATH=str(self.prefix/'share'), LD_LIBRARY_PATH=str(self.prefix/'lib'))
        def write(name, value):
            with (out/name).open('x') as f: json.dump(value, f, indent=2)
        started = datetime.now(timezone.utc).isoformat(); tick = time.monotonic(); code = None; error = None
        write('configuration.json', dict(started_at=started, command=command, robot_profile=self.robot_profile, candidate_count=len(ranked),
            scene_points=len(scene), voxel_m=.005, pregrasp_distance_m=.10, ik_timeout_s=.08,
            scene_relaxed=scene_relaxed, observed_scene_points=observed_count,
            ompl_timeout_s=1, approach_step_m=.002, target_excluded=True,
            scope='MoveIt KDL + FCL full robot/self/world + OMPL to pregrasp + axial IK/interpolated collision; observed scene only; bounded search',
            source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (binary, Path(__file__).resolve().parent/'moveit_grasps/scene_filter.cpp', urdf, srdf)}))
        try:
            with (out/'controller.log').open('x') as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                    timeout=max(120, len(ranked)*5), check=True)
            code = result.returncode
            results = [json.loads(line) for line in (out/'results.jsonl').read_text().splitlines()]
            if [r['index'] for r in results] != list(range(1, len(ranked)+1)):
                raise RuntimeError('Incomplete or reordered MoveIt prefilter output')
            for row, result in zip(ranked, results):
                row['moveit_scene_result'] = result
            survivors = [row for row, result in zip(ranked, results) if result['status']=='accepted']
            summary = dict(checked_count=len(results), accepted_count=len(survivors),
                rejected_count=len(results)-len(survivors), status_counts=dict(Counter(r['status'] for r in results)),
                accepted_original_ranks=[r['original_rank'] for r in survivors], evidence=str(out/'results.jsonl'))
            write('summary.json', summary)
            return survivors, summary
        except Exception as exc:
            error = repr(exc); code = getattr(exc, 'returncode', code)
            raise
        finally:
            write('execution.json', dict(started_at=started, finished_at=datetime.now(timezone.utc).isoformat(),
                elapsed_s=time.monotonic()-tick, exit_code=code, error=error))
