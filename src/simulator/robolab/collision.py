"""RoboLab adapter for the shared observed-point whole-robot sweep checks."""
import numpy as np
import trimesh
from src.tools.motion.path_collision import CandidatePathCollision, motion_radii, world_stationary_geom
from src.simulator.robolab.robot_model import RoboLabRobotModel


class RoboLabPathCollision(CandidatePathCollision):
    def __init__(self, connector, *, clearance_m=.002):
        import mujoco
        if not trimesh.ray.has_embree:
            raise RuntimeError('RoboLab visual-mesh ray queries require embreex; install it in the simulator interpreter')
        if not np.isfinite(clearance_m) or clearance_m < 0:
            raise ValueError('invalid collision clearance')
        env = connector.env
        from pathlib import Path
        self._query_log = (Path(env.output_dir)/'robot-geometry-worker.log'
                           if getattr(env, 'output_dir', None) is not None else None)
        self.native = RoboLabRobotModel(env.robot.cfg.spawn.usd_path)
        self.model, self.data = self.native.model, self.native.data
        self.native.set_joints(env.joints(), env.gripper_angle())
        self.captured_qpos = self.data.qpos.copy()
        self.clearance_m = float(clearance_m)
        self.robot = list(range(self.model.ngeom))
        self.base = self.model.body('panda_link0').id
        self.addresses = self.native.arm_addresses
        self.fingers = []
        self.names = {g: self.native.geom_body[self.model.geom(g).name]+'/'+self.model.geom(g).name for g in self.robot}
        self.radii = motion_radii(self.model, self.robot, [self.model.joint(f'panda_joint{i}').id for i in range(1,8)])
        self.meshes = {g:self._mesh(g) for g in self.robot}
        self.mesh_radii = {g:float(np.linalg.norm(mesh.vertices, axis=1).max())+self.clearance_m for g,mesh in self.meshes.items()}
        self.stationary_geoms = {g for g in self.robot if world_stationary_geom(self.model,g)}
        self.capture_poses = {g:self._pose(g) for g in self.robot}
        self.capture_visual_meshes = {}
        self.capture_visual_poses = {}
        for name,triangles in self.native.visual_triangles.items():
            self.capture_visual_meshes[name] = trimesh.Trimesh(vertices=triangles.reshape(-1,3),
                faces=np.arange(triangles.size//3).reshape(-1,3),process=True)
            pose=self.native.body_matrix(self.native.geom_body[name])
            self.capture_visual_poses[name]=(pose[:3,3],pose[:3,:3])
        from src.simulator.robolab.moveit_model import excluded_pair
        parents = {j['child'].rsplit('/', 1)[-1]: j['parent'].rsplit('/', 1)[-1]
                   for j in self.native.joints}
        self.pairs = [(a,b) for a in self.robot for b in self.robot if a<b
                      and not excluded_pair(self.native.geom_body[self.model.geom(a).name],
                                            self.native.geom_body[self.model.geom(b).name], parents)]

    def _set_gripper_width(self, jaw_width_m):
        from gap.envs.robolab_control import robotiq_width_to_angle
        q = self.data.qpos[self.addresses].copy()
        self.native.set_joints(q, min(np.pi/4, robotiq_width_to_angle(jaw_width_m)))

    def remove_captured_robot(self, points):
        """Same visual-surface/closed-volume rule, using triangle BVHs.

        Building Python R-trees for the high-resolution native hand under Kit
        takes minutes. Open3D queries the original triangles without simplifying
        them or replacing the empty space between fingers with a hull.
        """
        from src.simulator.robolab.visual_query import remove_visual_points
        return remove_visual_points(points, self.capture_visual_meshes, self.capture_visual_poses,
                                    log=getattr(self, '_query_log', None))

    def check_joint_self_collision(self, joints, *, jaw_width_m):
        from types import SimpleNamespace
        from gap.envs.robolab_control import validate_joint_target
        q = validate_joint_target(joints)
        plan = SimpleNamespace(start_joints=q, target_labels=('release',),
            segments=({'waypoints': [{'positions': q.tolist()}]},))
        result = self.check(plan, np.empty((0, 3)), stop_label='release',
                            jaw_width_m=jaw_width_m, skip_scene=True)
        return {**result, 'stage': 'release_goal_self_collision',
            'geometry': 'convex hulls of native RoboLab robot USD meshes',
            'pair_policy': 'exclude graph distance <=2 and internal gripper mechanism pairs',
            'scene_collision_checked': False, 'joint_path_checked': False}
