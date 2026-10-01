"""Shared request contract for optional grasp approach preferences."""

MAX_GRASP_CANDIDATES = 6
APPROACH_MODES = ('vertical', 'horizontal')


def normalize_approach(value=None):
    """Validate the public direction choice."""
    if value is None:
        return None
    if isinstance(value, str) and value in APPROACH_MODES:
        return value
    raise ValueError('preferred_direction must be vertical, horizontal or null; omission is allowed')


def resolve_approach(value=None):
    """Translate a public choice to the generator's direction and family."""
    mode = normalize_approach(value)
    if mode is None:
        return None, None
    return mode, None  # Direction alone keeps the backend's default face family.


def normalize_preference(value=None):
    if value is None or (isinstance(value, str) and value == 'none'):
        return None
    if isinstance(value, str) and value in ('vertical', 'horizontal'):
        return value
    raise ValueError('preferred_direction must be vertical, horizontal, none or null; omission is allowed')


def preference_schema():
    return dict(type=['string', 'null'], enum=[*APPROACH_MODES, None],
        description="Preferred gripper approach direction, not the object's orientation. "
                    'vertical: generate fitted-box face candidates, preferring downward approach from above (base -Z). '
                    'horizontal: generate fitted-box face candidates, preferring side approach parallel to the base XY plane. '
                    'Directions are ranking preferences, not exact angle constraints. '
                    "When grasping an object's edge, rim, handle, knob or button, omit preferred_direction or pass null. "
                    'If the approach direction is uncertain, omit preferred_direction or pass null. '
                    'At most 6 candidates are shown.')


def normalize_grasp_type(value=None):
    if value is None or (isinstance(value, str) and value == "none"):
        return None
    if isinstance(value, str) and value in ('face', 'edge', 'corner', 'all'):
        return value
    raise ValueError('grasp_type must be face, edge, corner, all, none or null; omission is allowed')
