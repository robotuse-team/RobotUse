"""Candidate-only Panda path collision checks against captured scene points.

Uses a kinematic displacement bound and native MuJoCo geometry distances.
Scene geometry is never used as an obstacle oracle.
All FK runs on a private copy; neither live joints nor live collision flags change.
"""
import copy
import math
import re
from numbers import Integral, Real

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from src.tools.place.collision import mesh_points_collision

CLEARANCE_M = .002
CAPTURE_REMOVAL_MARGIN_M = .005
SAMPLE_MOTION_M = .0005
# The table top (base-frame z ~ 0) is a known support surface: fingertips may
# touch it (penetration still rejected); every other geom keeps the clearance.
TABLE_CONTACT_Z_M = .004
FINGER_GEOM_NAMES = ('gripper0_finger1_collision', 'gripper0_finger2_collision',
                     'gripper0_finger1_pad_collision', 'gripper0_finger2_pad_collision')



def collision_scene_witness(scene, indices, hit, *, sample_index, segment, robotgeom, clearance_m):
    """Copy the first reported scene point; malformed evidence never alters a verdict."""
    try:
        query_index = hit['point_index']
        if (not isinstance(query_index, Integral) or isinstance(query_index, (bool, np.bool_))
                or not 0 <= query_index < len(indices)):
            return None
        scene_index = indices[query_index]
        if (not isinstance(scene_index, Integral) or isinstance(scene_index, (bool, np.bool_))
                or not 0 <= scene_index < len(scene)):
            return None
        point = np.asarray(scene[scene_index])
        if (point.shape != (3,) or point.dtype.kind not in 'fiu' or not np.isfinite(point).all()
                or not isinstance(sample_index, Integral) or isinstance(sample_index, (bool, np.bool_))
                or sample_index < 1 or not isinstance(segment, str) or not isinstance(robotgeom, str)
                or not isinstance(clearance_m, Real) or isinstance(clearance_m, (bool, np.bool_))
                or not np.isfinite(clearance_m) or clearance_m < 0):
            return None
        return {'frame': 'connector_base', 'scene_point_xyz_m': point.tolist(),
                'scene_point_index': int(scene_index), 'query_point_index': int(query_index),
                'scene_index_scope': 'input_scene_after_capture_filter',
                'query_index_scope': 'geom_radius_query_subset',
                'sample_index_1based': int(sample_index), 'segment': segment,
                'robotgeom': robotgeom, 'applied_clearance_m': float(clearance_m)}
    except (KeyError, IndexError, TypeError, ValueError, OverflowError):
        return None


def world_stationary_geom(model, gid):
    """Prove world stationarity from every ancestor, never endpoint equality.

    Any joint (including a mobile/free base or finger slide) or mocap body
    invalidates this classification, even if that joint is idle in this plan.
    """
    body = int(model.geom_bodyid[gid])
    while True:
        if model.body_jntnum[body] or model.body_mocapid[body] >= 0:
            return False
        if body == 0:
            return True
        body = int(model.body_parentid[body])


def captured_robot_visual_geoms(model, base, mesh_type):
    """Select known attached robot visuals by body ancestry, not name prefix.

    Mounts have their own naming namespace but are children of the robot base.
    Scene siblings remain outside this subtree, regardless of their names.
    """
    def attached(body):
        while body != base and body != 0:
            body = int(model.body_parentid[body])
        return body == base

    return [g for g in range(model.ngeom)
            if model.geom_group[g] == 1 and model.geom_type[g] == mesh_type
            and attached(int(model.geom_bodyid[g]))]


def path_samples(plan, radii, stop_label):
    """Bound displacement of every downstream robot point, not just the TCP."""
    previous = np.asarray(plan.start_joints, dtype=float)
    yield previous, 'initial'
    end = plan.target_labels.index(stop_label) + 1
    for label, segment in zip(plan.target_labels[:end], plan.segments[:end]):
        for waypoint in segment['waypoints']:
            q = np.asarray(waypoint['positions'], dtype=float)[:7]
            steps = max(1, math.ceil(float(radii @ np.abs(q - previous)) / SAMPLE_MOTION_M))
            for i in range(1, steps + 1):
                yield previous + (q - previous) * (i / steps), label
            previous = q


def initial_contact_indices(checker, scene, tree):
    """Scene points already within clearance of a MOVING robot geom at the start pose.

    The robot is standing there before any commanded motion, so these are
    pre-existing contacts (for example an object it just pushed), not
    collisions created by the checked path. They are excluded from the sweep;
    every other scene point, including the rest of that same object, remains.
    """
    found = []
    for gid, mesh in checker.meshes.items():
        if gid in checker.stationary_geoms:
            continue
        position, rotation = checker._pose(gid)
        indices = tree.query_ball_point(position, checker.mesh_radii[gid])
        if not indices:
            continue
        indices = np.asarray(indices, dtype=int)
        local = (scene[indices] - position) @ rotation
        lo, hi = mesh.bounds
        selected = np.flatnonzero(np.all((local >= lo - checker.clearance_m)
                                         & (local <= hi + checker.clearance_m), axis=1))
        if not len(selected):
            continue
        distance = trimesh.proximity.signed_distance(mesh, local[selected])
        found.extend(indices[selected[distance >= -checker.clearance_m]].tolist())
    return np.unique(np.asarray(found, dtype=int))


def link_name(name):
    match = re.match(r'robot0_link(\d+)_collision', name)
    if match:
        return 'panda_link' + match[1]
    if name == 'gripper0_hand_collision':
        return 'panda_hand'
    if name.startswith('gripper0_finger1'):
        return 'panda_leftfinger'
    if name.startswith('gripper0_finger2'):
        return 'panda_rightfinger'
    return name


def motion_radii(model, robot, joint_ids):
    """Triangle-inequality bound on full-arm displacement along a path."""
    import mujoco
    bounds = []
    for jid in joint_ids:
        ancestor = int(model.jnt_bodyid[jid])
        best = 0.
        for gid in robot:
            body = int(model.geom_bodyid[gid])
            radius = float(np.linalg.norm(model.geom_pos[gid]) + model.geom_rbound[gid])
            while body and body != ancestor:
                radius += float(np.linalg.norm(model.body_pos[body]))
                for k in range(int(model.body_jntadr[body]), int(model.body_jntadr[body] + model.body_jntnum[body])):
                    if model.jnt_type[k] == mujoco.mjtJoint.mjJNT_SLIDE:
                        radius += float(np.max(np.abs(model.jnt_range[k])))
                    else:
                        radius += 2 * float(np.linalg.norm(model.jnt_pos[k]))
                body = int(model.body_parentid[body])
            if body == ancestor:
                best = max(best, radius + float(np.linalg.norm(model.jnt_pos[jid])))
        bounds.append(best)
    return np.asarray(bounds)


class CandidatePathCollision:
    def __init__(self, connector, *, clearance_m=CLEARANCE_M):
        if not np.isfinite(clearance_m) or clearance_m < 0:
            raise ValueError("invalid collision clearance")
        self.clearance_m = float(clearance_m)
        import mujoco
        from src.tools.motion.model_access import find_model_and_data
        from src.tools.motion.self_collision_runtime import PANDA_ROBOT_CONTACT_GEOMS, PANDA_GRIPPER_CONTACT_GEOMS
        from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml
        model, data = find_model_and_data(connector)
        self.model = copy.copy(getattr(model, '_model', model))
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = data.qpos
        self.captured_qpos = self.data.qpos.copy()
        self.model.geom_contype[:] = 0
        self.model.geom_conaffinity[:] = 0
        self.robot = [self._id(mujoco.mjtObj.mjOBJ_GEOM, name)
                      for name in (*PANDA_ROBOT_CONTACT_GEOMS, *PANDA_GRIPPER_CONTACT_GEOMS)]
        self.names = {g: mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in self.robot}
        joints = [self._id(mujoco.mjtObj.mjOBJ_JOINT, f'robot0_joint{i}') for i in range(1, 8)]
        self.addresses = self.model.jnt_qposadr[joints]
        self.radii = motion_radii(self.model, self.robot, joints)
        self.fingers = [self.model.jnt_qposadr[self._id(mujoco.mjtObj.mjOBJ_JOINT, name)]
                        for name in ('gripper0_finger_joint1', 'gripper0_finger_joint2')]
        self.base = self._id(mujoco.mjtObj.mjOBJ_BODY, 'robot0_base')
        config = load_yaml(join_path(get_robot_configs_path(), connector.ik._robot_file))['robot_cfg']['kinematics']
        ignored = {frozenset((a, b)) for a, bs in config['self_collision_ignore'].items() for b in bs}
        self.pairs = [(a, b) for i, a in enumerate(self.robot) for b in self.robot[i+1:]
                      if self.model.geom_bodyid[a] != self.model.geom_bodyid[b]
                      and link_name(self.names[a]) != link_name(self.names[b])
                      and frozenset((link_name(self.names[a]), link_name(self.names[b]))) not in ignored]
        self.meshes = {g: self._mesh(g) for g in self.robot}
        self.stationary_geoms = {g for g in self.robot if world_stationary_geom(self.model, g)}
        self.mesh_radii = {g: float(np.max(np.linalg.norm(mesh.vertices, axis=1))) + self.clearance_m
                           for g, mesh in self.meshes.items()}
        mujoco.mj_kinematics(self.model, self.data)
        self.capture_poses = {g: self._pose(g) for g in self.robot}
        # RGBD renders group-1 visual triangles, whose surface can extend beyond
        # the collision hulls. Use those triangles only for captured-self removal.
        visual = captured_robot_visual_geoms(self.model, self.base, mujoco.mjtGeom.mjGEOM_MESH)
        if not visual:
            raise ValueError('captured Panda visual meshes missing')
        self.capture_visual_meshes = {g: self._visual_mesh(g) for g in visual}
        self.capture_visual_poses = {g: self._pose(g) for g in visual}


    def _id(self, kind, name):
        import mujoco
        index = mujoco.mj_name2id(self.model, kind, name)
        if index < 0:
            raise ValueError(f'candidate collision geometry missing: {name}')
        return index

    def _mesh(self, gid):
        import mujoco
        kind = self.model.geom_type[gid]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mid = self.model.geom_dataid[gid]
            start, count = self.model.mesh_vertadr[mid], self.model.mesh_vertnum[mid]
            # Native MuJoCo mesh collision also uses the convex hull.
            return trimesh.convex.convex_hull(self.model.mesh_vert[start:start+count])
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            return trimesh.creation.box(extents=2*self.model.geom_size[gid])
        raise ValueError(f'unsupported Panda collision shape: {kind}')

    def _visual_mesh(self, gid):
        mid = self.model.geom_dataid[gid]
        v0, vn = self.model.mesh_vertadr[mid], self.model.mesh_vertnum[mid]
        f0, fn = self.model.mesh_faceadr[mid], self.model.mesh_facenum[mid]
        return trimesh.Trimesh(vertices=self.model.mesh_vert[v0:v0+vn].copy(),
                              faces=self.model.mesh_face[f0:f0+fn].copy(), process=True)

    def _pose(self, gid):
        rotation = self.data.xmat[self.base].reshape(3, 3)
        return ((self.data.geom_xpos[gid] - self.data.xpos[self.base]) @ rotation,
                rotation.T @ self.data.geom_xmat[gid].reshape(3, 3))

    def remove_captured_robot(self, points):
        """Remove captured visual surfaces; only closed visual volumes have an inside."""
        points = np.asarray(points)
        removed = np.zeros(len(points), dtype=bool)
        for gid, (position, rotation) in self.capture_visual_poses.items():
            local = (points - position) @ rotation
            mesh = self.capture_visual_meshes[gid]
            lo, hi = mesh.bounds
            indices = np.flatnonzero(np.all((local >= lo-CAPTURE_REMOVAL_MARGIN_M) & (local <= hi+CAPTURE_REMOVAL_MARGIN_M), axis=1))
            if len(indices):
                _, distance, _ = trimesh.proximity.closest_point(mesh, local[indices])
                hit = distance <= CAPTURE_REMOVAL_MARGIN_M
                # Material-split visual pieces can be open. Their signed
                # distance does not define an interior; never clear that half-space.
                if mesh.is_volume:
                    hit |= mesh.contains(local[indices])
                removed[indices[hit]] = True
        return points[~removed].copy(), {'input_points': len(points), 'removed_robot_points': int(removed.sum()),
            'capture_policy': 'captured robot visual surfaces within 5mm and closed visual interiors; never proposed poses',
            'capture_removal_margin_m': CAPTURE_REMOVAL_MARGIN_M,
            'capture_visual_mesh_count': len(self.capture_visual_meshes)}

    def _set_gripper_width(self, jaw_width_m):
        self.data.qpos[self.fingers] = [jaw_width_m/2, -jaw_width_m/2]

    def check(self, plan, scene, *, stop_label, jaw_width_m, skip_scene=False, ignore_initial_contacts=False):
        import mujoco
        self.data.qpos[:] = self.captured_qpos
        self._set_gripper_width(jaw_width_m)
        scene = np.asarray(scene)
        tree = None if skip_scene else cKDTree(scene)
        table_points = (scene[:, 2] < TABLE_CONTACT_Z_M) if len(scene) else np.zeros(0, dtype=bool)
        last_clear_pose = {}
        evidence = {'accepted': True, 'clearance_m': self.clearance_m, 'sample_motion_bound_m': SAMPLE_MOTION_M,
                    'checked_until': stop_label, 'environment': 'observed non-target scene points',
                    'checks': ['whole_robot_scene_collision', 'configured_robot_self_collision'],
                    'stationary_scene_clearance_m': 0., 'moving_scene_clearance_m': self.clearance_m,
                    'stationary_scene_geoms': [self.names[g] for g in sorted(self.stationary_geoms)],
                    'stationary_scene_policy': 'world-stationary by full joint/mocap ancestry; collision-proxy penetration checked once; no physical geometry guarantee'}
        if skip_scene:
            evidence.update(environment='explicitly ignored', checks=['configured_robot_self_collision'])
        for count, (q, label) in enumerate(path_samples(plan, self.radii, stop_label), 1):
            self.data.qpos[self.addresses] = q
            mujoco.mj_kinematics(self.model, self.data)
            for a, b in self.pairs:
                distance = mujoco.mj_geomDistance(self.model, self.data, a, b, self.clearance_m, None)
                if distance < self.clearance_m:
                    return {**evidence, 'accepted': False, 'sample_count': count, 'segment': label,
                            'kind': 'self', 'geoms': [self.names[a], self.names[b]], 'distance_m': distance}
            if skip_scene:
                continue
            if count == 1 and ignore_initial_contacts and tree is not None:
                removed = initial_contact_indices(self, scene, tree)
                if len(removed):
                    keep = np.ones(len(scene), dtype=bool)
                    keep[removed] = False
                    scene = scene[keep]
                    table_points = table_points[keep]
                    tree = cKDTree(scene) if len(scene) else None
                    evidence['initial_contact_points_removed'] = int(len(removed))
                    evidence['initial_contact_policy'] = ('scene points within clearance of the robot at its '
                        'current pose are pre-existing contacts, not path collisions; excluded from this sweep')
            if tree is None:
                continue
            for gid, mesh in self.meshes.items():
                stationary = gid in self.stationary_geoms
                if stationary and count > 1:
                    continue
                position, rotation = self._pose(gid)
                previous = last_clear_pose.get(gid)
                if previous is not None and np.array_equal(position, previous[0]) and np.array_equal(rotation, previous[1]):
                    continue
                last_clear_pose[gid] = (position.copy(), rotation.copy())
                indices = tree.query_ball_point(position, self.mesh_radii[gid])
                if not indices:
                    continue
                scene_clearance = 0. if stationary else self.clearance_m
                indices = np.asarray(indices, dtype=int)
                if self.names[gid] in FINGER_GEOM_NAMES and table_points[indices].any():
                    # Fingertips may rest on the support surface. Measured table
                    # points sit up to TABLE_CONTACT_Z_M above the physical plane,
                    # so a fingertip on the real table penetrates them by that much:
                    # table points are rejected only beyond that depth, other
                    # points keep the clearance.
                    on_table = table_points[indices]
                    hit = mesh_points_collision(mesh, (scene[indices[~on_table]]-position) @ rotation,
                                                clearance_m=scene_clearance) if (~on_table).any() else None
                    if hit is not None:
                        indices = indices[~on_table]
                    else:
                        hit = mesh_points_collision(mesh, (scene[indices[on_table]]-position) @ rotation,
                                                    clearance_m=-TABLE_CONTACT_Z_M)
                        if hit is not None:
                            indices, scene_clearance = indices[on_table], -TABLE_CONTACT_Z_M
                            evidence['table_contact_policy'] = ('fingertip penetration of table points '
                                                                f'deeper than {TABLE_CONTACT_Z_M*1e3:g} mm')
                else:
                    hit = mesh_points_collision(mesh, (scene[indices]-position) @ rotation, clearance_m=scene_clearance)
                if hit is not None:
                    # mesh_points_collision reports an index into this radius
                    # query, not the complete filtered scene. Retain the exact
                    # first reported measurement for private RGB evidence;
                    # caller supplies its captured observation identity.
                    witness = collision_scene_witness(scene, indices, hit, sample_index=count,
                        segment=label, robotgeom=self.names[gid], clearance_m=scene_clearance)
                    return {**evidence, 'accepted': False, 'sample_count': count, 'segment': label,
                            'kind': 'environment', 'geom': self.names[gid],
                            'geom_scene_clearance_m': scene_clearance,
                            'geom_world_stationary': stationary, **hit,
                            **({'collision_witness': witness} if witness is not None else {})}
        return {**evidence, 'sample_count': count}


def make_candidate_path_collision(connector, *, clearance_m=CLEARANCE_M):
    """Dispatch at the environment boundary; LIBERO keeps its existing checker."""
    factory = getattr(getattr(connector, 'env', None), 'make_path_collision', None)
    if callable(factory):
        return factory(connector, clearance_m=clearance_m)
    return CandidatePathCollision(connector, clearance_m=clearance_m)
