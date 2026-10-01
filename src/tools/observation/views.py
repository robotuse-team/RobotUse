"""Observed-centroid viewpoints in connector BASE axes, never object/camera axes.

front=-X, rear=+X, left=+Y, right=-Y. These names do not identify an
object's semantic front/back. Camera optical convention is +Z forward, +Y down.
No object poses, rendering, physics stepping, collision or tracking safety gate.
"""
import numpy as np

VIEWPOINT_OFFSETS = {
    "front_oblique": (-.30, 0.), "rear_oblique": (.30, 0.),
    "left_oblique": (0., .30), "right_oblique": (0., -.30), "top": (0., 0.),
}
VIEWPOINT_DEFINITIONS = "connector_base axes: front=-X, rear=+X, left=+Y, right=-Y; not object-semantic or camera-relative"


def rigid_matrix(value):
    t = np.asarray(value, dtype=float)
    if (t.shape != (4, 4) or not np.isfinite(t).all() or
            not np.allclose(t[3], [0, 0, 0, 1], atol=1e-7) or
            not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5) or
            not np.isclose(np.linalg.det(t[:3, :3]), 1., atol=1e-5)):
        raise ValueError("invalid rigid camera/robot calibration")
    return t.copy()


def fresh_wrist_state(connector):
    """Read actual robot/base/camera matrices in a single unchanged sim epoch."""
    from src.tools.motion.planning import _pose_transform
    from src.tools.perception.multiview import simulation_epoch
    from src.tools.perception.runtime import quaternion_camera_to_base
    before = simulation_epoch(connector)
    ee = rigid_matrix(_pose_transform(connector.get_ee_pose()))
    env = connector.env
    camera_pose = getattr(env, 'camera_optical_matrix', None)
    if callable(camera_pose):
        optical = rigid_matrix(camera_pose('robot0_eye_in_hand'))
        if simulation_epoch(connector) != before:
            raise ValueError('simulation epoch changed during wrist calibration')
        return {'ee': ee, 'optical': optical,
                'ee_from_optical': rigid_matrix(np.linalg.inv(ee) @ optical), 'epoch': before}
    base = np.asarray(env.base_link_wxyz_xyz, dtype=float)
    if base.shape != (7,) or not np.isfinite(base).all():
        raise ValueError("invalid robot base calibration")
    b = quaternion_camera_to_base({"position": dict(zip("xyz", base[4:])),
                                   "rotation": dict(zip("wxyz", base[:4]))})
    world_base = np.eye(4)
    world_base[:3, :3], world_base[:3, 3] = b.rotation, b.translation
    data = env.handle.env.sim.data
    world_camera = np.eye(4)
    world_camera[:3, :3] = np.asarray(data.get_camera_xmat("robot0_eye_in_hand")).reshape(3, 3)
    world_camera[:3, 3] = data.get_camera_xpos("robot0_eye_in_hand")
    optical = rigid_matrix(np.linalg.inv(rigid_matrix(world_base)) @ rigid_matrix(world_camera)
                           @ np.diag([1., -1., -1., 1.]))
    if simulation_epoch(connector) != before:
        raise ValueError("simulation epoch changed during wrist calibration")
    return {"ee": ee, "optical": optical, "ee_from_optical": rigid_matrix(np.linalg.inv(ee) @ optical),
            "epoch": before}


def look_at_optical(position, target):
    p, target = np.asarray(position, float), np.asarray(target, float)
    if p.shape != (3,) or target.shape != (3,) or not np.isfinite([p, target]).all():
        raise ValueError("finite 3D look-at coordinates required")
    z = target - p
    if np.linalg.norm(z) < 1e-6:
        raise ValueError("look-at camera coincides with target")
    z /= np.linalg.norm(z)
    up = np.array([0., 0., 1.])
    if abs(z @ up) > .98:
        up = np.array([0., 1., 0.])
    x = np.cross(z, up); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    t = np.eye(4); t[:3, :3] = np.column_stack((x, y, z)); t[:3, 3] = p
    return rigid_matrix(t)


def requested_view(target_points, ee_from_optical, *, viewpoint, high_z_m,
                   clearance_m=.25):
    if viewpoint not in VIEWPOINT_OFFSETS:
        raise ValueError("unknown viewpoint; choose " + ", ".join(VIEWPOINT_OFFSETS))
    points = np.asarray(target_points, float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("nonempty finite observed target cloud required")
    calibration = rigid_matrix(ee_from_optical)
    target = np.median(points, axis=0)
    if not np.isfinite(clearance_m) or clearance_m < 0:
        raise ValueError("observation clearance must be finite and nonnegative")
    height = max(float(high_z_m), float(points[:, 2].max()) + clearance_m)
    if not np.isfinite(height):
        raise ValueError("invalid observation height")
    # Optical origin stays above high EEF altitude even for rotated wrist offset.
    position = target.copy(); position[:2] += VIEWPOINT_OFFSETS[viewpoint]
    position[2] = height + np.linalg.norm(calibration[:3, 3])
    optical = look_at_optical(position, target)
    ee = rigid_matrix(optical @ np.linalg.inv(calibration))
    return target, optical, ee


def view_metrics(before, after, target):
    before, after = rigid_matrix(before), rigid_matrix(after)
    target = np.asarray(target, float)
    a, b = before[:3, 3] - target, after[:3, 3] - target
    def angle(a, b):
        n = np.linalg.norm(a) * np.linalg.norm(b)
        return None if n < 1e-10 else float(np.degrees(np.arccos(np.clip(a @ b / n, -1., 1.))))
    orbit = angle(a, b)
    azimuth = angle(a[:2], b[:2])
    aim = angle(after[:3, 2], -b)
    translation = float(np.linalg.norm(after[:3, 3] - before[:3, 3]))
    return {"camera_translation_m": translation, "optical_axis_change_deg": angle(before[:3, 2], after[:3, 2]),
            "target_orbit_angle_deg": orbit, "base_xy_azimuth_change_deg": azimuth,
            "opposite_base_xy_hemisphere": bool(azimuth is not None and azimuth > 90.),
            "target_ray_error_deg": aim,
            "geometrically_new_view": bool(translation > .02 and orbit is not None and orbit > 10. and aim is not None and aim < 10.),
            "target_visibility": "unknown_until_fresh_depth_and_same_object_reselection",
            "missing_surface_coverage_verified": False}
