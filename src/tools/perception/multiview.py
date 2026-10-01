"""Measured-only, single-epoch front/wrist snapshots and deterministic fusion."""
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
from src.tools.perception.runtime import capture_connector_rgbd, quaternion_camera_to_base
from src.tools.perception.geometry import ObservationProfileId

CAMERAS = ("agentview", "robot0_eye_in_hand")

def simulation_epoch(connector):
    env = connector.env
    from src.runtime.clock import simulation_time_s
    seconds = simulation_time_s(env)
    return (int(env._sim_step_count), float(seconds))

def capture_fresh_pair(*, connector, output_dir, capture_revision):
    """Read camera/robot calibration and render twice without advancing physics.

    Caller must serialize simulation mutations with this operation. A changed
    control-step/time guard rejects racing capture; no cached observation getter.
    """
    capture = getattr(connector.env, 'capture_rgbd', None)
    if callable(capture):
        before = simulation_epoch(connector)
        cameras = capture()
        if simulation_epoch(connector) != before:
            raise ValueError('simulation epoch changed during RGBD capture')
        frames = capture_connector_rgbd(connector=SimpleNamespace(get_observation=lambda: {'cameras': cameras}),
            output_dir=output_dir, capture_revision=capture_revision,
            actor_profile=ObservationProfileId.MULTIVIEW_TRACK)
        return frames, before
    from robosuite.utils.camera_utils import get_real_depth_map
    from scipy.spatial.transform import Rotation
    env, sim = connector.env, connector.env.handle.env.sim
    before = simulation_epoch(connector)
    base = np.asarray(env.base_link_wxyz_xyz, dtype=float)
    if base.shape != (7,) or not np.isfinite(base).all():
        raise ValueError("invalid robot base calibration")
    rigid = quaternion_camera_to_base({"position": dict(zip("xyz", base[4:])),
                                      "rotation": dict(zip("wxyz", base[:4]))})
    world_base = np.eye(4)
    world_base[:3, :3], world_base[:3, 3] = rigid.rotation, rigid.translation
    base_world = np.linalg.inv(world_base)
    w, h = int(env._render_width), int(env._render_height)
    cameras = []
    for name in CAMERAS:
        world_camera = np.eye(4)
        world_camera[:3, :3] = np.asarray(sim.data.get_camera_xmat(name)).reshape(3, 3)
        world_camera[:3, 3] = sim.data.get_camera_xpos(name)
        optical = base_world @ world_camera @ np.diag([1., -1., -1., 1.])
        q = Rotation.from_matrix(optical[:3, :3]).as_quat()  # xyzw
        fovy = float(sim.model.cam_fovy[sim.model.camera_name2id(name)])
        f = .5*h / np.tan(fovy*np.pi/360.)
        rgb, raw = sim.render(camera_name=name, width=w, height=h, depth=True)
        # Preserve established renderer flips and normalized->metric depth units.
        normalized = np.clip(np.nan_to_num(np.asarray(raw)[::-1], nan=1., posinf=1., neginf=0.), 0., 1.)
        depth = np.asarray(get_real_depth_map(sim, normalized), dtype=np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        cameras.append(dict(name=name, rgb=np.ascontiguousarray(rgb[::-1]), depth=depth,
            intrinsics=((f, 0., w/2), (0., f, h/2), (0., 0., 1.)),
            pose={"position": dict(zip("xyz", optical[:3, 3])),
                  "rotation": dict(zip("wxyz", [q[3], *q[:3]]))}))
    if simulation_epoch(connector) != before:
        raise ValueError("simulation epoch changed during RGBD capture")
    frames = capture_connector_rgbd(connector=SimpleNamespace(get_observation=lambda: {"cameras": cameras}),
        output_dir=output_dir, capture_revision=capture_revision, actor_profile=ObservationProfileId.MULTIVIEW_TRACK)
    return frames, before

@dataclass(frozen=True)
class VoxelCloud:
    points: np.ndarray
    view_bits: np.ndarray
    view_counts: np.ndarray
    sample_counts: np.ndarray
    source_view: np.ndarray
    source_index: np.ndarray
    view_ids: tuple

def voxel_merge(clouds, voxel_size_m=.002):
    """Lexicographically smallest actual measurement per voxel, NOT a centroid.

    View counts count distinct cameras, not samples. Representatives retain exact
    source indices; all source clouds remain available on the fusion record.
    """
    if not np.isfinite(voxel_size_m) or voxel_size_m <= 0:
        raise ValueError("voxel size must be positive finite meters")
    ids = tuple(sorted(clouds))
    if not 1 <= len(ids) <= 63:
        raise ValueError("expected 1..63 distinct views")
    arrays = [np.asarray(clouds[v], dtype=np.float32).reshape(-1, 3) for v in ids]
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("cloud has nonfinite points")
    points = np.concatenate(arrays)
    views = np.concatenate([np.full(len(a), i, dtype=np.int64) for i, a in enumerate(arrays)])
    indices = np.concatenate([np.arange(len(a)) for a in arrays])
    groups = {}
    for i, key in enumerate(np.floor(points.astype(float)/voxel_size_m).astype(np.int64)):
        groups.setdefault(tuple(key), []).append(i)
    representatives, bits, counts = [], [], []
    for key in sorted(groups):
        rows = groups[key]
        representatives.append(min(rows, key=lambda i: (*points[i], views[i], indices[i])))
        bits.append(sum(1 << int(v) for v in set(views[rows])))
        counts.append(len(rows))
    r = np.asarray(representatives, dtype=int)
    return VoxelCloud(points[r], np.asarray(bits, dtype=np.uint64),
        np.asarray([b.bit_count() for b in bits]), np.asarray(counts), views[r], indices[r], ids)

@dataclass(frozen=True)
class FusedGeometry:
    observation_id: str
    epoch: object
    same_object: str
    per_view: tuple
    object_cloud: VoxelCloud
    scene_cloud: VoxelCloud
    voxel_size_m: float
    @property
    def source_views(self): return self.object_cloud.view_ids
    @property
    def front(self):
        """Front selection for single-view previews; not a fused mask."""
        return next((g for g in self.per_view if g.view_id == CAMERAS[0]), self.per_view[0])
    @property
    def frame(self): return self.front.frame
    @property
    def view_id(self): return self.front.view_id
    @property
    def role(self): return self.front.role
    @property
    def mask_path(self): return self.front.mask_path
    @property
    def overlay_path(self): return self.front.overlay_path
    @property
    def object_points(self): return self.object_cloud.points
    @property
    def scene_points(self): return self.scene_cloud.points

def fuse_selected(geometries, *, observation_id, epoch, same_object, voxel_size_m=.002, role="pick"):
    selected = tuple(geometries)
    if not isinstance(same_object, str) or not same_object.strip():
        raise ValueError("explicit agent same_object selection required")
    if len(selected) != 2 or {g.view_id for g in selected} != set(CAMERAS):
        raise ValueError("independent front and wrist selections required")
    if any(g.observation_id != observation_id or g.epoch != epoch for g in selected):
        raise ValueError("mismatched observation/epoch; stale accumulation prohibited")
    if role not in ("pick", "place") or any(g.role != role or len(g.object_points) == 0 for g in selected):
        raise ValueError("two nonblank selections of the requested role required")
    obj = voxel_merge({g.view_id: g.object_points for g in selected}, voxel_size_m)
    scene = voxel_merge({g.view_id: g.scene_points for g in selected}, voxel_size_m)
    # Deliberately retain contradictory other-view scene evidence conservatively;
    # never project a selected mask into the other image or erase hidden geometry.
    return FusedGeometry(observation_id, epoch, same_object, selected, obj, scene, voxel_size_m)


def compare_measured_clouds(previous, current, *, distance_threshold_m=.005):
    """Comparison only: NEVER merge measurements across time or certify a back face.

    A moved object, mismatched agent selection, noise or occluder can cause novel
    samples too. The caller must explicitly assert same object by reselection;
    these base-coordinate distances are not tracking or visibility certificates.
    """
    from scipy.spatial import cKDTree
    if not np.isfinite(distance_threshold_m) or distance_threshold_m <= 0:
        raise ValueError("positive finite cloud comparison threshold required")
    arrays = [np.asarray(g.object_points, float) for g in (previous, current)]
    if any(a.ndim != 2 or a.shape[1] != 3 or not len(a) or not np.isfinite(a).all() for a in arrays):
        raise ValueError("comparison requires nonempty measured clouds")
    old, new = arrays
    distance = cKDTree(old).query(new)[0]
    reverse = cKDTree(new).query(old)[0]
    novel = distance > distance_threshold_m
    per_view = []
    for selection in getattr(current, "per_view", (current,)):
        samples = np.asarray(selection.object_points, float)
        distances = cKDTree(old).query(samples)[0]
        frame = getattr(selection, "frame", None)
        per_view.append({"view_id": getattr(selection, "view_id", None),
                         "frame_id": getattr(frame, "frame_id", None),
                         "calibration_path": str(getattr(frame, "calibration_path", "")),
                         "point_count": len(samples),
                         "new_measured_point_count": int((distances > distance_threshold_m).sum()),
                         "new_measured_fraction": float((distances > distance_threshold_m).mean())})
    return {"policy": "fresh_pair_replace_no_temporal_merge",
            "previous_observation_id": previous.observation_id, "observation_id": current.observation_id,
            "previous_epoch": previous.epoch, "epoch": current.epoch,
            "previous_point_count": len(old), "point_count": len(new), "per_view": per_view,
            "distance_threshold_m": float(distance_threshold_m),
            "new_measured_point_count": int(novel.sum()), "new_measured_fraction": float(novel.mean()),
            "previous_points_not_reobserved_fraction": float((reverse > distance_threshold_m).mean()),
            "new_to_previous_distance_m": {"median": float(np.median(distance)), "p95": float(np.percentile(distance, 95))},
            "measured_centroid_shift_m": float(np.linalg.norm(np.median(new, axis=0) - np.median(old, axis=0))),
            "missing_surface_coverage_verified": False,
            "limitations": "distance novelty is not object tracking, identity proof, occlusion clearance or semantic back-surface coverage"}
