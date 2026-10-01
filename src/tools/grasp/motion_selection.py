"""Prefer shorter 180-degree jaw symmetries without ranking unrelated grasps."""
import math

import numpy as np

from src.tools.observation.views import rigid_matrix

POLICIES = ('source-order', 'low-motion')


def validate_policy(policy, score_tolerance):
    if policy not in POLICIES:
        raise ValueError('unknown grasp motion policy')
    if not math.isfinite(score_tolerance) or not 0 <= score_tolerance <= 1:
        raise ValueError('grasp score tolerance must be finite and in [0, 1]')


def motion_cost(hand, current_ee, grasp_to_ee):
    """Compare in the measured public EE frame, including native calibration."""
    target = rigid_matrix(hand) @ rigid_matrix(grasp_to_ee)
    current = rigid_matrix(current_ee)
    cosine = np.clip((np.trace(current[:3, :3].T @ target[:3, :3])-1.)/2., -1., 1.)
    return dict(rotation_deg=float(np.degrees(np.arccos(cosine))),
                translation_m=float(np.linalg.norm(target[:3, 3]-current[:3, 3])))


def rank_symmetric_candidates(items, *, tolerance):
    """Swap only equivalent grasp slots; preserve every unrelated candidate slot.

    Keep both poses for reachability fallback. The caller may skip a symmetric
    sibling only after an equivalent pose has passed its path checks. Quality
    tolerance is anchored at the first member, never chained through siblings.
    """
    validate_policy('low-motion', tolerance)
    ordered = list(items)
    assigned = set()
    for index, anchor in enumerate(items):
        if index in assigned:
            continue
        slots = [index]
        slots.extend(i for i in range(index + 1, len(items))
                     if i not in assigned
                     and abs(anchor['score']-items[i]['score']) <= tolerance+1e-12
                     and same_parallel_jaw_grasp(anchor, items[i]))
        assigned.update(slots)
        siblings = sorted((items[i] for i in slots),
                          key=lambda row: round(row['motion_cost']['rotation_deg'], 6))
        for slot, sibling in zip(slots, siblings):
            ordered[slot] = sibling
    return ordered


def same_parallel_jaw_grasp(a, b):
    """Only duplicate/180-degree jaw swaps; different contacts/approaches survive.

    These tight tolerances absorb serialization roundoff, not motion tolerance.
    The symmetry does not establish arm reachability: a representative must
    independently pass the configured path checks before a sibling is skipped.
    """
    x, y = np.asarray(a['hand']), np.asarray(b['hand'])
    relative = x[:3, :3].T @ y[:3, :3]
    return (np.linalg.norm(x[:3, 3]-y[:3, 3]) <= 1e-5
            and abs(a['open_width_m']-b['open_width_m']) <= 1e-6
            and any(np.allclose(relative, r, atol=1e-5, rtol=0)
                    for r in (np.eye(3), np.diag([-1., -1., 1.]))))


def estimated_motion_seconds(plan, env):
    """Native streaming ticks only; excludes settling, gripper holds and LLMs.

    Mirrors RoboLab retime_joint_targets: each segment is sampled separately.
    Missing timing metadata stays unknown instead of becoming a zero duration.
    """
    if (not callable(getattr(env, 'stream_joint_trajectory', None))
            or not getattr(env, '_stream_enabled', True)):
        return None
    frequency = getattr(env, '_control_freq', None)
    scale = getattr(env, 'motion_speed_scale', None)
    if (frequency is None or scale is None or not math.isfinite(frequency)
            or not math.isfinite(scale) or frequency <= 0 or scale <= 0):
        return None
    counts = [len(segment.get('waypoints', ())) for segment in plan.segments]
    if not counts or not all(counts):
        return None
    return sum(math.ceil(n/scale) for n in counts)/frequency
