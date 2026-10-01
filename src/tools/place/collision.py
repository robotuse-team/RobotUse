"""Observed-scene mesh helpers for release-only placement validation."""
from functools import lru_cache
import numpy as np
import trimesh
from src.tools.motion.planning import _transform
from src.tools.pose_editor.inspection import load_panda_mesh, transform_points


@lru_cache(maxsize=64)
def gripper_meshes(opening_m, *, mesh_source=None):
    parts, _ = load_panda_mesh(mesh_source, expected_open_width_m=opening_m)
    meshes = {}
    for name, triangles in parts.items():
        mesh = trimesh.Trimesh(vertices=triangles.reshape(-1, 3),
            faces=np.arange(triangles.size//3).reshape(-1, 3), process=True)
        if mesh_source is not None:
            # USD visual parts can be open surfaces. Native whole-arm checks
            # use these same conservative per-part convex hulls.
            mesh = mesh.convex_hull
        if not mesh.is_watertight or not mesh.is_winding_consistent:
            raise ValueError('gripper collision mesh must be a consistently oriented solid')
        meshes[name] = mesh
    return meshes

def mesh_points_collision(mesh, points_local, *, clearance_m=.002):
    """Inside-solid or near-surface measured points, including triangle interiors."""
    p = np.asarray(points_local)
    lo, hi = mesh.bounds
    selected = np.flatnonzero(np.all((p >= lo-clearance_m) & (p <= hi+clearance_m), axis=1))
    if not len(selected):
        return None
    distance = trimesh.proximity.signed_distance(mesh, p[selected])
    hit = distance >= -clearance_m
    if not hit.any():
        return None
    return {'colliding_points': int(hit.sum()), 'max_inside_distance_m': float(distance.max()),
            'point_index': int(selected[np.flatnonzero(hit)[0]])}

def remove_captured_gripper(scene_points, captured_hand, captured_width_m, *, mesh_source=None):
    """Remove the known hand seen at capture, never the hypothetical new pose.

    Only observed non-target scene points within the captured solid gripper
    (plus 2 mm sensor/model tolerance) are removed. This does not clear unknown
    space, the arm, or an arbitrary volume around a proposed placement.
    """
    scene = np.asarray(scene_points)
    if mesh_source is not None:
        return mesh_source.remove_captured_points(scene, captured_hand, captured_width_m)
    local = transform_points(scene, np.linalg.inv(_transform(captured_hand)))
    removed = np.zeros(len(scene), dtype=bool)
    counts = {}
    meshes = gripper_meshes(float(captured_width_m))
    for name, mesh in meshes.items():
        lo, hi = mesh.bounds
        indices = np.flatnonzero(np.all((local >= lo-.002) & (local <= hi+.002), axis=1))
        if len(indices):
            inside = trimesh.proximity.signed_distance(mesh, local[indices]) >= -.002
            removed[indices[inside]] = True
            counts[name] = int(inside.sum())
        else:
            counts[name] = 0
    return scene[~removed].copy(), {'input_points':len(scene), 'removed_points':int(removed.sum()),
        'retained_points':int((~removed).sum()), 'removed_by_body':counts,
        'tolerance_m':.002, 'policy':'captured known gripper mesh only; target points preserved; arm and unknown space not cleared'}

def sample_agent_choices(eligible, *, seed, maximum=4):
    """Only sample after all candidates have completed feasibility checks."""
    if maximum != 4:
        raise ValueError('agent placement choice budget is four')
    indices = np.random.default_rng(seed).choice(len(eligible), min(maximum,len(eligible)), replace=False)
    return [eligible[int(i)] for i in indices]
