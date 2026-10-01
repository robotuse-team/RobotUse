"""Measured geometry and agent-specified contact poses.

Inputs are already in ``connector_base`` metres; these helpers never infer a
camera transform, object height, gripper offset, or a missing agent decision.
Contact poses locate the point between the jaws, not the hand/flange origin.
The caller must apply the robot's explicit contact-to-hand/EE calibration.
"""
from __future__ import annotations

import numpy as np


FRAME = 'connector_base'
SOURCE = 'observed_sam_segment'
_DOWN = np.array([0., 0., -1.])


def _numeric(value, name):
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must contain finite real numbers') from exc
    if array.dtype.kind not in 'iuf' or not np.isfinite(array).all():
        raise ValueError(f'{name} must contain finite real numbers')
    return array.astype(float, copy=True)


def _scalar(value, name):
    array = _numeric(value, name)
    if array.shape != ():
        raise ValueError(f'{name} must be a finite real scalar')
    return float(array)


def _metadata(source, frame):
    if not isinstance(source, str) or not source.strip():
        raise ValueError('source must identify the observed geometry')
    if frame != FRAME:
        raise ValueError('geometry must already be in connector_base coordinates')
    return dict(source=source, frame=frame, units='metres')


def observed_cloud_statistics(points, *, source=SOURCE, frame=FRAME):
    """Return JSON-safe statistics of a nonempty, finite observed Nx3 cloud.

    Invalid rows are rejected rather than silently removed. Bounds and Z
    quantiles describe visible measured surfaces, not complete object geometry.
    """
    metadata = _metadata(source, frame)
    cloud = _numeric(points, 'observed points')
    if cloud.ndim != 2 or cloud.shape[1:] != (3,) or not len(cloud):
        raise ValueError('observed points must be a nonempty Nx3 cloud')
    with np.errstate(over='ignore', invalid='ignore'):
        mean = cloud.mean(axis=0)
        median = np.median(cloud, axis=0)
        lo, hi = cloud.min(axis=0), cloud.max(axis=0)
        extents = hi - lo
        quantiles = np.quantile(cloud[:, 2], [.05, .25, .5, .75, .95])
    if not all(np.isfinite(value).all() for value in (mean, median, extents, quantiles)):
        raise ValueError('observed cloud statistics must remain finite')
    xy = cloud[:, :2] - mean[:2]
    covariance = xy.T @ xy / len(cloud)
    if not np.isfinite(covariance).all():
        raise ValueError('observed cloud covariance must remain finite')
    values, vectors = np.linalg.eigh(covariance)
    axis = None
    if values[-1] > 0 and values[-1] - values[0] > 1e-6 * values[-1]:
        axis = float(np.degrees(np.arctan2(vectors[1, -1], vectors[0, -1])) % 180.)
    return dict(**metadata, count=len(cloud), mean_xyz_m=mean.tolist(),
                median_xyz_m=median.tolist(), min_xyz_m=lo.tolist(),
                max_xyz_m=hi.tolist(), extents_xyz_m=extents.tolist(),
                z_quantiles_m=dict(zip(('p05', 'p25', 'p50', 'p75', 'p95'),
                                      quantiles.tolist())),
                principal_xy_axis_yaw_deg=axis,
                principal_xy_axis_scope='undirected observed surface axis modulo 180 degrees; not a grasp rotation',
                scope='observed surfaces only; hidden surfaces remain unknown')


def xy_candidates(points, *, clicked_xyz_m=None, source=SOURCE, frame=FRAME):
    """Offer an optional measured click, then componentwise median and mean.

    Each candidate includes its observed reference Z for an explicit relative
    height decision. It does not choose the eventual contact Z. Pass ``None``
    when the click has no valid measured depth; malformed supplied clicks fail.
    Candidates retain distinct provenance even when their coordinates coincide.
    """
    statistics = observed_cloud_statistics(points, source=source, frame=frame)
    centers = []
    if clicked_xyz_m is not None:
        click = _numeric(clicked_xyz_m, 'clicked_xyz_m')
        if click.shape != (3,):
            raise ValueError('clicked_xyz_m must be a measured XYZ point')
        centers.append(('clicked', click.tolist(), 'observed_depth_at_selected_pixel'))
    centers.extend((('median', statistics['median_xyz_m'], source),
                    ('mean', statistics['mean_xyz_m'], source)))
    return [dict(candidate_id=kind, source=center_source, frame=frame, units='metres',
                 xy_m=xyz[:2], reference_z_m=xyz[2],
                 reference_z_source=f'{kind}_observed_xyz',
                 observed_xyz_m=xyz)
            for kind, xyz, center_source in centers]


def resolve_height(value_m, *, mode, reference_z_m=None):
    """Resolve an explicit absolute base Z or observed-surface-relative offset.

    ``absolute`` takes the requested Z directly. ``surface_relative`` adds the
    requested signed offset to the explicitly selected observed reference Z.
    There are no defaults, clearance offsets, clamps, or fallback estimates.
    """
    value = _scalar(value_m, 'height value_m')
    if mode == 'absolute':
        if reference_z_m is not None:
            raise ValueError('absolute height does not take a surface reference')
        return value
    if mode != 'surface_relative':
        raise ValueError('height mode must be absolute or surface_relative')
    if reference_z_m is None:
        raise ValueError('surface_relative height requires an observed reference_z_m')
    reference = _scalar(reference_z_m, 'reference_z_m')
    result = value + reference
    if not np.isfinite(result):
        raise ValueError('resolved height must be finite')
    return result


def top_down_contact_pose(xy_m, z_m, yaw_deg):
    """Construct a contact-center pose with local +Z pointing straight down.

    Agent yaw is the heading of local +X in the base XY plane, positive about
    base +Z. At yaw zero, the rotation is diag(1, -1, -1). Translation is the
    supplied contact-center XYZ; no hand/flange or clearance offset is applied.
    """
    xy = _numeric(xy_m, 'xy_m')
    if xy.shape != (2,):
        raise ValueError('xy_m must contain exactly two coordinates')
    z = _scalar(z_m, 'z_m')
    yaw = np.deg2rad(_scalar(yaw_deg, 'yaw_deg'))
    if not np.isfinite(yaw):
        raise ValueError('yaw must remain finite in radians')
    c, s = np.cos(yaw), np.sin(yaw)
    pose = np.eye(4)
    pose[:3, :3] = ((c, s, 0.), (s, -c, 0.), (0., 0., -1.))
    pose[:3, 3] = (xy[0], xy[1], z)
    return pose


def validate_top_down_contact_pose(pose):
    """Return a checked copy; reject nonrigid transforms or tilted approaches."""
    matrix = _numeric(pose, 'contact pose')
    if matrix.shape != (4, 4):
        raise ValueError('contact pose must be a 4x4 rigid transform')
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0., 0., 0., 1.], atol=1e-7, rtol=0.)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6, rtol=0.)):
        raise ValueError('contact pose must be a right-handed rigid transform')
    if not np.allclose(rotation[:, 2], _DOWN, atol=1e-7, rtol=0.):
        raise ValueError('geometric center grasps must retain a downward approach')
    return matrix


def refine_top_down_contact_pose(pose, *, dx_m=0., dy_m=0., dz_m=0., yaw_deg=0.,
                                 roll_deg=0., pitch_deg=0.):
    """Apply base XYZ translation and base-Z yaw while preserving top-down.

    Reject any nonzero roll/pitch request instead of silently discarding it.
    Step/cumulative budgets belong to the caller; this helper makes no clamps.
    """
    matrix = validate_top_down_contact_pose(pose)
    if _scalar(roll_deg, 'roll_deg') != 0. or _scalar(pitch_deg, 'pitch_deg') != 0.:
        raise ValueError('geometric center grasp refinement allows yaw only')
    shift = np.array([_scalar(dx_m, 'dx_m'), _scalar(dy_m, 'dy_m'),
                      _scalar(dz_m, 'dz_m')])
    yaw = np.rad2deg(np.arctan2(matrix[1, 0], matrix[0, 0]))
    yaw += _scalar(yaw_deg, 'yaw_deg')
    with np.errstate(over='ignore', invalid='ignore'):
        position = matrix[:3, 3] + shift
    return top_down_contact_pose(position[:2], position[2], yaw)
