"""Find a reachable camera pose directly above an agent-selected destination."""
from types import SimpleNamespace
import numpy as np


def destination_top_views(points, ee_from_optical, *, preferred_height):
    """Same top-down view intent, bounded height and camera-roll alternatives.

    All candidates keep the optical origin above the measured destination XY.
    Camera roll changes image orientation, not the destination or viewing ray.
    """
    from src.tools.observation.views import requested_view
    points = np.asarray(points, dtype=float)
    clearance_m = .05
    minimum = max(.25, float(points[:, 2].max()) + clearance_m)
    heights = sorted({max(minimum, float(preferred_height)), max(minimum, .40),
                      max(minimum, .35), max(minimum, .30)}, reverse=True)
    for height in heights:
        target, optical, _ = requested_view(points, ee_from_optical, viewpoint='top',
                                           high_z_m=height, clearance_m=clearance_m)
        for degrees in (0, 90, -90, 180):
            radians = np.deg2rad(degrees)
            roll = np.array([[np.cos(radians), -np.sin(radians), 0],
                             [np.sin(radians), np.cos(radians), 0], [0, 0, 1]])
            camera = optical.copy()
            camera[:3, :3] = camera[:3, :3] @ roll
            yield {'height_m': height, 'camera_roll_deg': degrees, 'target': target,
                   'camera': camera, 'ee': camera @ np.linalg.inv(ee_from_optical)}


def observe_destination_top(connector, *, points, config, recorder=None, point_ref=None):
    from src.tools.observation.views import fresh_wrist_state, view_metrics
    from src.tools.motion.planning import _plan, _execute_checked, MotionPlanningError
    before = fresh_wrist_state(connector)
    failures, selected = [], None
    obstacles = np.empty((0, 3))
    # Plan complete camera-goal moves from actual current state before committing
    # any movement. Each IK rejection leaves the original sensor state valid.
    for candidate in destination_top_views(points, before['ee_from_optical'],
                                            preferred_height=config.high_transit_z_m):
        try:
            segments, poses, *_ = _plan(connector, (candidate['ee'],), obstacles, config, None)
        except Exception as exc:
            failures.append({'height_m': candidate['height_m'], 'camera_roll_deg': candidate['camera_roll_deg'],
                             'reason': repr(exc)})
            continue
        selected = candidate
        break
    audit = {'policy': 'reachable_top_down_destination', 'point_ref': point_ref,
             'planning_rejections': failures, 'selected': None if selected is None else {
                 k: selected[k] for k in ('height_m', 'camera_roll_deg')}}
    if recorder:
        recorder.event('destination_view_search', audit)
    if selected is None:
        raise MotionPlanningError('no reachable directly-overhead destination view in bounded height/roll search')
    if recorder:
        recorder.register_plan(SimpleNamespace(segments=segments, targets=poses,
            target_labels=('destination_top_observation',), segment_labels=('destination_top_observation',),
            transit_policy='destination_top', high_transit_z_m=selected['height_m']),
            kind='observation', point_ref=point_ref)
    diagnostics = _execute_checked(connector, segments, poses,
                                   collision_checks_enabled=config.collision_checks_enabled)
    after = fresh_wrist_state(connector)
    metrics = view_metrics(before['optical'], after['optical'], selected['target'])
    xy_error = float(np.linalg.norm(after['optical'][:2, 3] - selected['target'][:2]))
    result = {**audit, 'view_novelty': metrics, 'camera_over_destination_xy_error_m': xy_error,
              'target_transforms': [selected['ee'].tolist()], 'diagnostics': diagnostics,
              'requires_fresh_observation': True, 'requires_same_object_reselection': True}
    if xy_error > .04 or metrics['target_ray_error_deg'] is None or metrics['target_ray_error_deg'] > 10:
        if recorder:
            recorder.event('destination_top_view_missed', result)
        raise RuntimeError('requested overhead destination camera pose not achieved; refresh and reselect')
    if recorder:
        recorder.event('destination_top_view_achieved', result)
    return result
