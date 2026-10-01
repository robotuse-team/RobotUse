"""Obstacle-aware transit planning from the observed scene cloud.

The configured planner uses a fused, target-excluded, robot-removed scene cloud.
Occupied voxels above the table form an obstacle mesh alongside a table slab.
Transit segments may detour; final approach and lift segments remain directed.
Path verification and fallback behavior follow the calling backend's policy.
"""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np

from src.backend.paused_refinement import PauseRefineIntentBackend
from src.tools.motion.planning import MotionPlanningError

VOXEL_M = .015
TABLE_Z_M = .01          # scene points below this are the table surface
PRE_APPROACH_MIN_Z_M = .25   # world transit ends at least this high, clear of table-top clutter
PRE_APPROACH_LOCAL_RADIUS_M = .10   # ... and above the tallest observed point near the goal XY
PRE_APPROACH_LOCAL_MARGIN_M = .06
WORLD_MAX_REACH_M = .78         # horizontal distance of the pre-approach from the base; farther goals fail IK
WORLD_MAX_APPROACH_TILT_DEG = 60.  # side grasps use straight routes
MAX_VOXELS = 20000
_CUBE_VERTICES = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                           [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
_CUBE_FACES = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                        [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=int)


def voxel_obstacle_mesh(points, *, voxel_m=VOXEL_M, table_z_m=TABLE_Z_M, inflate=1.05):
    """One triangle mesh of inflated cubes on the occupied voxels above the table."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] >= table_z_m)]
    if not len(points):
        return None, dict(voxel_m=voxel_m, voxels=0)
    size = float(voxel_m)
    keys = np.unique(np.floor(points / size).astype(np.int64), axis=0)
    while len(keys) > MAX_VOXELS:
        size *= 2.
        keys = np.unique(np.floor(points / size).astype(np.int64), axis=0)
    corners = (keys[:, None, :] + (_CUBE_VERTICES[None] - .5) * inflate + .5) * size
    vertices = corners.reshape(-1, 3)
    faces = (_CUBE_FACES[None] + 8 * np.arange(len(keys))[:, None, None]).reshape(-1, 3)
    return SimpleNamespace(name='observed_obstacles', pose=None, vertices=vertices.tolist(),
                           faces=faces.tolist()), dict(voxel_m=size, voxels=int(len(keys)))


def table_slab_mesh(points, *, top_z_m=0., thickness_m=.04, margin_m=.15):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if not len(points):
        return None
    lo, hi = points[:, :2].min(axis=0) - margin_m, points[:, :2].max(axis=0) + margin_m
    box = np.array([[lo[0], lo[1], top_z_m - thickness_m], [hi[0], hi[1], top_z_m]])
    vertices = box[0] + _CUBE_VERTICES * (box[1] - box[0])
    return SimpleNamespace(name='table_slab', pose=None, vertices=vertices.tolist(),
                           faces=_CUBE_FACES.tolist())


START_CONTACT_CLEARANCE_M = .03   # points this close to the robot NOW are pre-existing contacts


def current_contact_indices(checker, scene, joints, clearance_m=START_CONTACT_CLEARANCE_M):
    """Scene points within ``clearance_m`` of a moving robot geom at the CURRENT joints.

    A planner cannot leave a start state that already touches the world; those
    points (an object the hand just pushed, or the neighbour it stalled against)
    are excluded from the planning world, never from the sweep verification.
    """
    from scipy.spatial import cKDTree
    from src.tools.motion.path_collision import initial_contact_indices
    scene = np.asarray(scene, dtype=float)
    if not len(scene) or joints is None:
        return np.zeros(0, dtype=int)
    try:
        import mujoco
        checker.data.qpos[checker.addresses] = np.asarray(joints, dtype=float)[:len(checker.addresses)]
        mujoco.mj_kinematics(checker.model, checker.data)
    except (AttributeError, ImportError, ValueError):
        pass  # test fakes carry their own poses
    proxy = SimpleNamespace(meshes=checker.meshes, stationary_geoms=checker.stationary_geoms, _pose=checker._pose,
                            clearance_m=clearance_m,
                            mesh_radii={g: r - checker.clearance_m + clearance_m for g, r in checker.mesh_radii.items()})
    return initial_contact_indices(proxy, scene, cKDTree(scene))


def pre_approach_height(approach_z, approach_xy, scene, approach_m):
    """Pre-approach height: above the pregrasp, above the floor threshold and above nearby clutter."""
    scene = np.asarray(scene, dtype=float).reshape(-1, 3)
    top = 0.
    if len(scene):
        near = np.linalg.norm(scene[:, :2] - np.asarray(approach_xy, dtype=float), axis=1) <= PRE_APPROACH_LOCAL_RADIUS_M
        if near.any():
            top = float(scene[near, 2].max())
    return max(float(approach_z) + approach_m, PRE_APPROACH_MIN_Z_M, top + PRE_APPROACH_LOCAL_MARGIN_M)


_WORLD_SERIAL = [0]


def observed_world(scene_points, *, checker=None, joints=None):
    """cuRobo world (``.mesh`` list) for a target-excluded, robot-removed scene cloud.

    Mesh names are unique per build: cuRobo keys its mesh cache by name and
    silently reuses the old geometry for a same-named mesh, so a re-observed
    scene would otherwise plan against the first world of the episode.
    """
    scene = np.asarray(scene_points, dtype=float).reshape(-1, 3)
    removed = 0
    if checker is not None and hasattr(checker, 'meshes') and len(scene):
        contacts = current_contact_indices(checker, scene, joints)
        if len(contacts):
            keep = np.ones(len(scene), dtype=bool)
            keep[contacts] = False
            scene, removed = scene[keep], int(len(contacts))
    obstacles, summary = voxel_obstacle_mesh(scene)
    summary['current_contact_points_removed'] = removed
    _WORLD_SERIAL[0] += 1
    serial = _WORLD_SERIAL[0]
    meshes = [m for m in (table_slab_mesh(scene), obstacles) if m is not None]
    for m in meshes:
        m.name = f'{m.name}_{serial}'
    summary['world_serial'] = serial
    return SimpleNamespace(mesh=meshes, observed_points=scene.copy()), summary


class WorldPlanIntentBackend(PauseRefineIntentBackend):
    def __init__(self, *, observed_transit_planner=None, **kwargs):
        super().__init__(**kwargs)
        self.observed_transit_planner = observed_transit_planner
        self._world_cache = {}

    # ---- world -----------------------------------------------------------
    def _obstacle_world(self, scene, checker=None, joints=None):
        scene = np.asarray(scene, dtype=float)
        pose_key = tuple(np.round(np.asarray(joints, dtype=float), 3).tolist()) if joints is not None else None
        key = (id(scene), scene.shape, float(scene[:, 2].sum()) if len(scene) else 0., pose_key, checker is not None)
        cached = self._world_cache.get(key)
        if cached is None:
            cached = self._world_cache[key] = observed_world(scene, checker=checker, joints=joints)
            if len(self._world_cache) > 8:
                self._world_cache.pop(next(iter(self._world_cache)))
        return cached

    # ---- planning --------------------------------------------------------
    def _plan_world_segment(self, target_transform, start_joints, world):
        """Joint trajectory from ``start_joints`` to the EE transform, avoiding ``world``."""
        from src.tools.motion.planning import transform_to_pose
        planner = getattr(self, 'observed_transit_planner', None)
        if planner is not None:
            return planner(self.connector, target_transform, start_joints, world)
        ik = self.connector.ik
        impl = ik._import_impl()
        position, quat = ik._pose_for_curobo(transform_to_pose(target_transform), 0)
        success, trajectory = impl.plan_to_pose(
            np.asarray(position, dtype=float), np.asarray(quat, dtype=float),
            ik._resolve_seed(list(start_joints)), robot_file=ik._robot_file, tcp_offset=None,
            world_config=world, use_cuda_graph=getattr(ik, '_use_cuda_graph', False))
        if not success or trajectory is None:
            raise MotionPlanningError('world planner found no obstacle-free route',
                                      planning_feedback={'kind': 'planning', 'planner_reason_code': 'world_route_failed'})
        rows = np.asarray(trajectory, dtype=float)
        if rows.ndim != 2 or rows.shape[1] < 7 or not np.isfinite(rows).all() or len(rows) < 2:
            raise MotionPlanningError('world planner returned an invalid trajectory',
                                      planning_feedback={'kind': 'planning', 'planner_reason_code': 'invalid_trajectory'})
        return {'waypoints': [{'positions': row[:7].tolist()} for row in rows]}

    def _world_route(self, pose, target_points, obstacle_points, scene, options, checker=None):
        from src.tools.motion.planning import (GraspPlan, COLLISION_DISABLED_LIMITATIONS, _transform,
                                      _world_points, transform_to_pose, _collision_disabled_planner)
        from src.tools.motion.robot_state import _current_robot_state, _trajectory_end, _segments_safe
        config = self.motion_config
        grasp = _transform(pose)
        calibration = _transform(self.grasp_to_ee)
        ee = grasp @ calibration
        approach, lift = ee.copy(), ee.copy()
        approach[:3, 3] -= config.approach_m * grasp[:3, 2]
        lift[2, 3] += config.lift_m
        # The obstacle-aware transit ends ABOVE the clutter, not at the pregrasp:
        # a pregrasp beside neighbouring objects can lie inside the planner's
        # collision margin. The final descent stays
        # straight and is still verified by the sweep check.
        tilt = float(np.degrees(np.arccos(np.clip(-grasp[2, 2], -1., 1.))))  # approach axis vs straight down
        if tilt > WORLD_MAX_APPROACH_TILT_DEG:
            raise MotionPlanningError('side approach; world transit reserved for near-vertical approaches',
                                      planning_feedback={'kind': 'planning', 'planner_reason_code': 'world_skipped_side_approach',
                                                         'approach_tilt_deg': round(tilt, 1)})
        pre_approach = approach.copy()
        pre_approach[2, 3] = pre_approach_height(approach[2, 3], approach[:2, 3], scene, config.approach_m)
        reach = float(np.linalg.norm(pre_approach[:2, 3]))
        if reach > WORLD_MAX_REACH_M:
            raise MotionPlanningError('pre-approach beyond comfortable reach',
                                      planning_feedback={'kind': 'planning', 'planner_reason_code': 'world_skipped_reach',
                                                         'reach_m': round(reach, 3)})
        _, joints = _current_robot_state(self.connector)
        world, summary = self._obstacle_world(scene, checker, joints)
        transit = self._plan_world_segment(pre_approach, joints, world)
        planner_name = transit.get('planner', 'cuRobo observed-voxel planner')
        summary = {**summary, 'planner': planner_name}
        planner = _collision_disabled_planner(self.connector).ik
        seed = _trajectory_end(transit)
        to_pregrasp = planner.plan_linear(transform_to_pose(pre_approach), transform_to_pose(approach), seed_joints=seed)
        if not to_pregrasp:
            raise MotionPlanningError('straight descent to the pregrasp failed',
                                      planning_feedback={'kind': 'planning', 'segment': 'pregrasp',
                                                         'planner_reason_code': planner.planning_failure_code or 'no_usable_route'})
        descent = planner.plan_linear(transform_to_pose(approach), transform_to_pose(ee),
                                      seed_joints=_trajectory_end(to_pregrasp))
        if not descent:
            raise MotionPlanningError('straight descent from the pregrasp failed',
                                      planning_feedback={'kind': 'planning', 'segment': 'grasp',
                                                         'planner_reason_code': planner.planning_failure_code or 'no_usable_route'})
        segments = [transit, to_pregrasp, descent]
        targets, labels = [pre_approach, approach, ee], ['world_transit', 'pregrasp', 'grasp']
        if options.get('lift_after_grasp', True):
            rise = planner.plan_linear(transform_to_pose(ee), transform_to_pose(lift),
                                       seed_joints=_trajectory_end(descent))
            if not rise:
                raise MotionPlanningError('straight lift after the grasp failed',
                                          planning_feedback={'kind': 'planning', 'segment': 'lift',
                                                             'planner_reason_code': planner.planning_failure_code or 'no_usable_route'})
            segments.append(rise); targets.append(lift); labels.append('lift')
        if not _segments_safe(segments, start_joints=joints):
            raise MotionPlanningError('world route violates the joint-path continuity guard',
                                      planning_feedback={'kind': 'planning', 'planner_reason_code': 'continuity_rejected'})
        plan = GraspPlan(tuple(segments), tuple(transform_to_pose(t) for t in targets), tuple(float(v) for v in joints),
                         grasp, calibration, _world_points(target_points, np.eye(4)),
                         _world_points(obstacle_points, np.eye(4)), None, False, connector=self.connector,
                         target_labels=tuple(labels), transit_policy='world', high_transit_z_m=None,
                         open_width_m=options.get('open_width_m', .08), collision_checks_enabled=False,
                         limitations=COLLISION_DISABLED_LIMITATIONS + (
                             f'transit to a pre-approach above clutter: {planner_name}; '
                             'target excluded; descent to pregrasp, grasp and lift straight',))
        plan.world_summary = summary
        return plan

    def _routes(self, pose, target, obstacles, checker, scene, options, *, relaxed=False):
        attempts = []
        if not relaxed and getattr(self, '_route_policy_override', 'auto') == 'auto':
            plan = None
            try:
                plan = self._world_route(pose, target, obstacles, scene, options, checker)
                evidence = checker.check(plan, scene, stop_label='grasp', jaw_width_m=plan.open_width_m)
                evidence['world_obstacles'] = getattr(plan, 'world_summary', None)
            except MotionPlanningError as exc:
                evidence = {'kind': 'planning', 'error': str(exc),
                            **getattr(exc, 'planning_feedback', {}), 'accepted': False}
            attempts.append(dict(policy='world', relaxed=False, evidence=deepcopy(evidence)))
            self._record('grasp_route_attempt', attempts[-1])
            if evidence.get('accepted') and plan is not None:
                return plan, {**evidence, 'route_attempts': attempts, 'transit_policy': 'world'}
        plan, evidence = super()._routes(pose, target, obstacles, checker, scene, options, relaxed=relaxed)
        evidence = dict(evidence)
        evidence['route_attempts'] = attempts + list(evidence.get('route_attempts', []))
        return plan, evidence

    def _plan_waypoint(self, p, scene, checker):
        if p.get('motion') == 'linear':
            return super()._plan_waypoint(p, scene, checker)
        from src.tools.motion.planning import transform_to_pose
        from src.tools.motion.robot_state import _current_robot_state, _segments_safe
        try:
            _, joints = _current_robot_state(self.connector)
            world, summary = self._obstacle_world(np.asarray(scene, dtype=float), checker, joints)
            segment = self._plan_world_segment(p['pose'], joints, world)
            if not _segments_safe([segment], start_joints=joints):
                raise MotionPlanningError('world waypoint route violates the continuity guard')
            self._record('world_waypoint_plan', dict(waypoint_motion='world', **summary,
                                                     waypoints=len(segment['waypoints'])))
            return (segment,), (transform_to_pose(p['pose']),), tuple(float(v) for v in joints), None, False
        except MotionPlanningError as exc:
            self._record('world_waypoint_plan', dict(waypoint_motion='world', error=str(exc), fallback='straight'))
            return super()._plan_waypoint(p, scene, checker)
