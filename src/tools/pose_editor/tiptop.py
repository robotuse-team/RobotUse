"""NumPy-only TiPToP depth adapter for observed preview geometry.

Adapted from tiptop/perception/utils.py, depth_to_xyz, MIT licensed:
https://github.com/tiptop-robot/tiptop/blob/d8f5afdaa94a7432220c3042f9f80be5ab45aae8/tiptop/perception/utils.py
Original source and license are preserved in src/tools/pose_editor/third_party/tiptop. Hardware camera,
cuTAMP and Open3D imports are intentionally not required by this adapter.
"""
import numpy as np

REVISION = 'd8f5afdaa94a7432220c3042f9f80be5ab45aae8'


def depth_to_xyz(depth, intrinsics):
    depth, k = np.asarray(depth), np.asarray(intrinsics, dtype=float)
    if depth.ndim != 2 or k.shape != (3, 3) or not np.isfinite(k).all():
        raise ValueError('registered depth and calibrated 3x3 intrinsics required')
    if k[0, 0] <= 0 or k[1, 1] <= 0 or k[0, 1] != 0 or k[1, 0] != 0:
        raise ValueError('positive focal lengths and zero-skew intrinsics required')
    u, v = np.meshgrid(np.arange(depth.shape[1], dtype=np.float32),
                       np.arange(depth.shape[0], dtype=np.float32))
    return np.stack(((u-k[0, 2])*depth/k[0, 0], (v-k[1, 2])*depth/k[1, 1], depth), axis=-1)


def observed_frame_points(frame, stride=6):
    """Only measured valid depth; no shape completion or simulator object state."""
    depth = np.asarray(frame.depth_m)
    xyz = depth_to_xyz(depth, frame.intrinsics)[::stride, ::stride]
    valid = np.isfinite(xyz).all(axis=-1) & (xyz[..., 2] > 0)
    transform = frame.camera_to_base
    return xyz[valid] @ np.asarray(transform.rotation).T + np.asarray(transform.translation)
