"""Numeric pose decisions shared by provider schemas and runtime boundaries."""
from copy import deepcopy
import math

from src.backend.controller import BoundaryError


HEIGHT_REFERENCES = ('absolute', 'grasp', 'current_tcp', 'clicked_point',
                     'segment_median', 'segment_mean', 'observed_min', 'observed_max')
from .schemas import HEIGHT_SCHEMA, TRANSIT_SCHEMA


def property_schema(key, *, tool=None, unrestricted_pose_rotation=False,
                    unrestricted_pose_translation=False, clicked_grasp_candidates=False):
    """Return a pose-decision schema, or None to use the caller's schema."""
    schemas = {
        'geometric_height': HEIGHT_SCHEMA, 'height': HEIGHT_SCHEMA, 'transit_height': HEIGHT_SCHEMA, 'transit': TRANSIT_SCHEMA,
        'xy_m': dict(type='array', items=dict(type='number'), minItems=2, maxItems=2),
        'xy_source': dict(type='string', enum=['clicked', 'median', 'mean']),
        'tolerance_deg': dict(type='number', minimum=0, maximum=180),
        'polar_deg': dict(type=['number', 'null'], minimum=0, maximum=180),
    }
    if tool == 'explicit_grasp_candidates':
        schemas.update(direction=dict(type='string', enum=['vertical', 'custom', 'mean', 'median']),
                       azimuth_deg=dict(type=['number', 'null']),
                       tolerance_deg=dict(type=['number', 'null'], minimum=0, maximum=180),
                       geometric_height={**HEIGHT_SCHEMA, 'type': ['object', 'null']})
        if clicked_grasp_candidates:
            schemas['direction']['enum'].append('clicked')
    if tool == 'validate_view':
        schemas['motion'] = dict(type='string', enum=['planned', 'linear'],
            description='planned follows the proposed waypoint orientation; linear preserves current orientation along a straight translation.')
    if tool == 'move_vertical':
        schemas['dz_m'] = dict(type='number', description='Signed base-Z displacement in metres; preserves gripper command.')
    if tool == 'explicit_adjust_place':
        schemas.update({name: dict(type='number', minimum=-30, maximum=30)
                        for name in ('dx_mm', 'dy_mm', 'dz_mm')})
        schemas.update({name: dict(type='number', minimum=-10, maximum=10)
                        for name in ('roll_deg', 'pitch_deg', 'yaw_deg')})
    if (unrestricted_pose_rotation and key in ('roll_deg', 'pitch_deg', 'yaw_deg')
            and tool in ('adjust_grasp', 'nudge_grasp', 'explicit_adjust_place')):
        return dict(type='number', description='Finite degrees about the gripper local axis; no angular magnitude budget. The resulting pose and motion are checked.')
    if (unrestricted_pose_translation and key in ('dx_mm', 'dy_mm', 'dz_mm')
            and tool in ('adjust_grasp', 'nudge_grasp', 'explicit_adjust_place')):
        return dict(type='number', description='Finite signed BASE XYZ millimetres; no per-step or cumulative magnitude cap. The resulting pose and motion are checked.')
    return deepcopy(schemas.get(key))


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise BoundaryError(f'{name} must be a finite number')
    return float(value)


def validate_height(value):
    if not isinstance(value, dict) or set(value) != {'value_m', 'reference'}:
        raise BoundaryError('height requires exactly reference and value_m')
    if value['reference'] not in HEIGHT_REFERENCES:
        raise BoundaryError('unknown height reference')
    return dict(value_m=_number(value['value_m'], 'height value_m'), reference=value['reference'])


def validate_argument(tool, key, value):
    """Validate pose-decision fields; the caller handles all remaining fields."""
    schema = property_schema(key, tool=tool)
    if schema is None:
        raise KeyError(key)
    if value is None and isinstance(schema.get('type'), list) and 'null' in schema['type']:
        return None
    if key in ('height', 'geometric_height', 'transit_height'):
        return validate_height(value)
    if key == 'transit':
        if not isinstance(value, dict) or set(value) != {'pre', 'post'}:
            raise BoundaryError('transit requires explicit pre and post heights')
        return {name: validate_height(value[name]) for name in ('pre', 'post')}
    if key == 'xy_m':
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise BoundaryError('xy_m requires exactly two base coordinates')
        return [_number(item, 'xy_m') for item in value]
    if key in ('azimuth_deg', 'polar_deg') and value is None:
        return None
    if schema['type'] == 'string':
        if value not in schema['enum']:
            raise BoundaryError(f'{key} must be one of {schema["enum"]}')
        return value
    number = _number(value, key)
    if not schema.get('minimum', -math.inf) <= number <= schema.get('maximum', math.inf):
        raise BoundaryError(f'{key} is outside its declared range')
    return number
