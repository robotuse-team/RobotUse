"""Public, bounded fields describing an unexecuted planner segment.

Never forward raw planner strings, absolute poses, joint arrays or exceptions.
These fields describe the requested motion, not a diagnosis of reachability.
"""
import math


REASON_CODES = frozenset({'no_planner', 'no_usable_route', 'ik_failed',
    'planner_failed', 'invalid_trajectory', 'continuity_rejected', 'planner_exception', 'world_route_failed'})
SEGMENTS = frozenset({'initial_lift', 'high_transit', 'high_pregrasp_align',
    'pregrasp', 'grasp', 'lift', 'release', 'retreat', 'waypoint', 'transit', 'pregrasp_align'})


def segment_planning_feedback(index, label, start, target, reason):
    """Describe a known requested segment; do not infer why its route failed."""
    import numpy as np
    delta = target[:3, 3] - start[:3, 3]
    angle = math.degrees(math.acos(float(np.clip(
        (np.trace(start[:3, :3].T @ target[:3, :3]) - 1.) / 2., -1., 1.))))
    return public_planning_feedback(dict(kind='planning', segment_index=index, segment=label,
        planner_reason_code=reason if isinstance(reason, str) and reason in REASON_CODES else 'planner_failed',
        start_reference='current_ee' if index == 0 else 'previous_planned_waypoint',
        requested_translation_m=dict(zip(('dx_m', 'dy_m', 'dz_m'), map(float, delta))),
        orientation_change_deg=angle))


def public_planning_feedback(value):
    if not isinstance(value, dict) or value.get('kind') != 'planning':
        return {}
    output = {'kind': 'planning'}
    index = value.get('segment_index')
    if type(index) is int and index >= 0:
        output['segment_index'] = index
    if isinstance(value.get('segment'), str) and value['segment'] in SEGMENTS:
        output['segment'] = value['segment']
    if isinstance(value.get('planner_reason_code'), str) and value['planner_reason_code'] in REASON_CODES:
        output['planner_reason_code'] = value['planner_reason_code']
    if value.get('start_reference') in ('current_ee', 'previous_planned_waypoint'):
        output['start_reference'] = value['start_reference']
    delta = value.get('requested_translation_m')
    axes = ('dx_m', 'dy_m', 'dz_m')
    if isinstance(delta, dict) and all(type(delta.get(k)) in (int, float)
            and math.isfinite(delta[k]) for k in axes):
        output['requested_translation_m'] = {k: float(delta[k]) for k in axes}
    angle = value.get('orientation_change_deg')
    if type(angle) in (int, float) and math.isfinite(angle) and 0 <= angle <= 180:
        output['orientation_change_deg'] = float(angle)
    return output
