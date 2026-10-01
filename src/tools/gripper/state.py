"""Measured jaw opening in metres with caller-supplied bounds."""
import numpy as np


def measured_opening(connector, *, max_width_m=.08, tolerance_m=0.):
    if (not np.isfinite(max_width_m) or max_width_m <= 0
            or not np.isfinite(tolerance_m) or tolerance_m < 0):
        raise ValueError('invalid gripper width limits')
    arm = connector.get_observation()['arms'][0]
    if 'gripper_width_m' in arm:
        width = float(arm['gripper_width_m'])
        raw = []
    else:
        qpos = np.asarray(arm.get('gripper_qpos', []), dtype=float)
        if qpos.shape != (2,) or not np.isfinite(qpos).all():
            raise ValueError('measured two-finger proprioception required')
        raw = qpos.tolist()
        width = float(abs(qpos[0] - qpos[1]))
    if not np.isfinite(width) or not 0 <= width <= max_width_m + tolerance_m:
        raise ValueError('measured gripper opening outside configured range')
    return min(width, max_width_m), raw, max_width_m
