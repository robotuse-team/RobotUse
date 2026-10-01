"""Process-isolated triangle queries: Open3D and Kit must not share a process."""
import os
from pathlib import Path
import shutil
import sys
import tempfile
import numpy as np

MARGIN_M = .005


def remove_visual_points(points, meshes, poses, *, log=None, margin_m=MARGIN_M):
    from src.runtime.worker import call_worker
    points = np.asarray(points, dtype=float)
    if not np.isfinite(margin_m) or margin_m < 0:
        raise ValueError('invalid captured visual surface tolerance')
    directory = Path(tempfile.mkdtemp(prefix='robolab-visual-query-'))
    log = Path(log) if log is not None else Path(tempfile.gettempdir())/f'robolab-visual-query-{os.getpid()}.log'
    data = {'points': points, 'count': np.array(len(meshes))}
    for i, (gid, mesh) in enumerate(meshes.items()):
        position, rotation = poses[gid]
        data.update({f'vertices_{i}': mesh.vertices, f'faces_{i}': mesh.faces,
            f'position_{i}': position, f'rotation_{i}': rotation, f'volume_{i}': np.array(mesh.is_volume)})
    source, output = directory/'input.npz', directory/'removed.npy'
    np.savez(source, **data)
    # Successful scratch IPC is disposable; failed inputs remain for diagnosis.
    call_worker(sys.executable, Path(__file__).resolve(), dict(input=str(source), output=str(output), margin_m=float(margin_m)),
                log, 120., startup_timeout=120.)
    removed = np.load(output, allow_pickle=False)
    if removed.dtype != np.bool_ or removed.shape != (len(points),):
        raise ValueError(f'invalid native visual query output; evidence retained in {directory}')
    shutil.rmtree(directory)
    return points[~removed].copy(), dict(input_points=len(points), removed_robot_points=int(removed.sum()),
        capture_policy='captured robot visual surfaces within tolerance and closed visual interiors; never proposed poses',
        capture_removal_margin_m=float(margin_m), capture_visual_mesh_count=len(meshes),
        query_backend='process-isolated Open3D triangle BVH')


def _worker(args):
    import open3d as o3d
    margin = args.margin_m
    with np.load(args.input, allow_pickle=False) as data:
        points = data['points']
        removed = np.zeros(len(points), dtype=bool)
        for i in range(int(data['count'])):
            local = (points-data[f'position_{i}']) @ data[f'rotation_{i}']
            vertices, faces = data[f'vertices_{i}'], data[f'faces_{i}']
            indices = np.flatnonzero(~removed & np.all(
                (local >= vertices.min(0)-margin) & (local <= vertices.max(0)+margin), axis=1))
            if not len(indices):
                continue
            query = o3d.t.geometry.RaycastingScene(nthreads=2)
            query.add_triangles(o3d.core.Tensor(np.asarray(vertices, dtype=np.float32)),
                                o3d.core.Tensor(np.asarray(faces, dtype=np.uint32)))
            samples = o3d.core.Tensor(np.asarray(local[indices], dtype=np.float32))
            hit = query.compute_distance(samples, nthreads=2).numpy() <= margin
            if bool(data[f'volume_{i}']):
                hit |= query.compute_occupancy(samples, nthreads=2, nsamples=3).numpy() > .5
            removed[indices[hit]] = True
    np.save(args.output, removed)
