"""Definitions owned by observation tools."""

from ..base_tool import define_tool
from ..schema import bounded_text, normalized_pixel

TOOLS = (
    define_tool('review_observation', {
        'observation_id': {'type': 'string'},
    }),
    define_tool('propose_waypoint', {
        'observation_id': {'type': 'string'},
        'u': {'type': 'number', 'minimum': 0, 'maximum': 1000},
        'v': {'type': 'number', 'minimum': 0, 'maximum': 1000},
        'purpose': {'type': 'string', 'enum': ['observe', 'transport', 'contact']},
        'height_offset_m': {'type': 'number', 'minimum': -0.5, 'maximum': 0.5},
        'dx_m': {'type': 'number', 'minimum': -0.5, 'maximum': 0.5},
        'dy_m': {'type': 'number', 'minimum': -0.5, 'maximum': 0.5},
        'dz_m': {'type': 'number', 'minimum': -0.5, 'maximum': 0.5},
    }, argument_validators={'u': normalized_pixel, 'v': normalized_pixel, 'purpose': bounded_text}),
    define_tool('propose_downward_waypoint', {
        'observation_id': {'type': 'string'},
        'u': {'type': 'number', 'minimum': 0, 'maximum': 1000},
        'v': {'type': 'number', 'minimum': 0, 'maximum': 1000},
        'height_offset_m': {'type': 'number', 'minimum': -0.5, 'maximum': 0.5},
        'dx_m': {'type': 'number', 'minimum': -0.5, 'maximum': 0.5},
        'dy_m': {'type': 'number', 'minimum': -0.5, 'maximum': 0.5},
        'dz_m': {'type': 'number', 'minimum': -0.5, 'maximum': 0.5},
    }, argument_validators={'u': normalized_pixel, 'v': normalized_pixel}),
    define_tool('shift_waypoint', {
        'observation_id': {'type': 'string'},
        'purpose': {'type': 'string', 'enum': ['observe', 'transport', 'contact']},
        'dx_m': {'type': 'number', 'minimum': -0.5, 'maximum': 0.5},
        'dy_m': {'type': 'number', 'minimum': -0.5, 'maximum': 0.5},
        'dz_m': {'type': 'number', 'minimum': -0.5, 'maximum': 0.5},
    }, argument_validators={'purpose': bounded_text}),
    define_tool('preview_view', {
        'waypoint_ref': {'type': 'string'},
        'azimuth_deg': {'type': 'number', 'minimum': -180.0, 'maximum': 180.0},
        'elevation_deg': {'type': 'number', 'minimum': -85.0, 'maximum': 85.0},
        'zoom': {'type': 'number', 'minimum': 0.5, 'maximum': 3.0},
    }),
    define_tool('validate_view', {
        'waypoint_ref': {'type': 'string'},
        'motion': {'type': 'string', 'enum': ['planned', 'linear'], 'description': 'planned follows the proposed waypoint orientation; linear preserves current orientation along a straight translation.'},
    }),
    define_tool('execute_view', {
        'waypoint_ref': {'type': 'string'},
        'validation_ref': {'type': 'string'},
    }),
)
