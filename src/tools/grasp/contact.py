"""Observed Panda grasp opening; no object label, LLM, or simulator state."""
import numpy as np

CONTACT_PADDING_M = .010  # Per side; opening gains 20 mm in total.
MAX_OPENING_M = .080
# Bounds of the pinned finger STL, relative to its attachment (see mesh loader).
FINGER_Y_M = .010494819842278957
FINGER_Z_M = (.0001316962443525, .053849034011364)


def contact_opening(pose, object_points, *, libero_adapter=True):
    # Match the observed float32 cloud sent to the inference worker exactly.
    points = np.asarray(object_points, dtype=np.float32)
    pose = np.asarray(pose, dtype=float)
    if (pose.shape != (4, 4) or points.ndim != 2 or points.shape[1] != 3
            or not np.isfinite(pose).all() or not np.isfinite(points).all()):
        raise ValueError('finite grasp pose and observed Nx3 cloud required')
    local = (points - pose[:3, 3]) @ pose[:3, :3]
    attachment = .0524 if libero_adapter else .0584
    inside = ((np.abs(local[:, 0]) <= MAX_OPENING_M / 2)
        & (np.abs(local[:, 1]) <= FINGER_Y_M)
        & (local[:, 2] >= attachment + FINGER_Z_M[0])
        & (local[:, 2] <= attachment + FINGER_Z_M[1]))
    captured = local[inside]
    # The pose centre stays fixed, so an off-centre patch needs twice its
    # furthest distance from the closing-axis origin, not just max(X)-min(X).
    required = float(2 * np.max(np.abs(captured[:, 0]))) if len(captured) >= 3 else None
    requested = required + 2 * CONTACT_PADDING_M if required is not None else MAX_OPENING_M
    # Round upward to 1 mm; identical openings can share the official filter.
    opening = float(min(MAX_OPENING_M, np.ceil(requested * 1000) / 1000))
    return {'open_width_m': opening, 'required_width_m': required,
            'padding_per_side_m': CONTACT_PADDING_M,
            'effective_padding_per_side_m': (opening-required)/2 if required is not None else None,
            'clamped_to_maximum': requested > MAX_OPENING_M,
            'observed_capture_points': len(captured),
            'method': 'observed points in finger closing volume; fixed grasp centre; 1mm upward rounding'
                if required is not None else 'insufficient observed capture points; use maximum opening'}
