"""Environment-specific Cartesian arrival limits, independent of collision policy."""
import math


def cartesian_tolerances(connector, *, position=.015, orientation=.15):
    env = getattr(connector, 'env', None)
    position = float(getattr(env, 'motion_position_tolerance_m', position))
    orientation = float(getattr(env, 'motion_orientation_tolerance_rad', orientation))
    if not all(math.isfinite(v) and v > 0 for v in (position, orientation)):
        raise ValueError('positive finite Cartesian arrival tolerances required')
    return position, orientation


def requires_profile_tracking(connector):
    return bool(getattr(getattr(connector, 'env', None), 'enforce_motion_tracking', False))
