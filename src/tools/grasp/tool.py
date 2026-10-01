"""Definitions owned by grasp; learned proposals retain their existing client."""

from ..base_tool import define_tool
from ..schema import bounded_text
from .schemas import HEIGHT_SCHEMA, TRANSIT_SCHEMA

VALIDATE_GRASP = define_tool('validate_grasp', {
        'candidate_ref': {'type': 'string'},
    }, visibility='internal')

EXECUTE_GRASP = define_tool('execute_grasp', {
        'candidate_ref': {'type': 'string'},
        'validation_ref': {'type': 'string'},
    }, visibility='internal')

TOOLS = (
    define_tool('grasp_candidates', {
        'point_ref': {'type': 'string'},
        'direction': {'type': 'string', 'enum': ['vertical', 'custom', 'mean', 'median', 'clicked']},
        'tolerance_deg': {'type': ['number', 'null'], 'minimum': 0, 'maximum': 180},
        'azimuth_deg': {'type': ['number', 'null']},
        'polar_deg': {'type': ['number', 'null'], 'minimum': 0, 'maximum': 180},
        'geometric_height': {**HEIGHT_SCHEMA, 'type': ['object', 'null']},
        'transit': TRANSIT_SCHEMA,
    }, internal_name='explicit_grasp_candidates'),
    define_tool('inspect_candidate', {
        'candidate_ref': {'type': 'string'},
    }),
    define_tool('preview_candidate', {
        'candidate_ref': {'type': 'string'},
        'azimuth_deg': {'type': 'number', 'minimum': -180.0, 'maximum': 180.0},
        'elevation_deg': {'type': 'number', 'minimum': -85.0, 'maximum': 85.0},
        'zoom': {'type': 'number', 'minimum': 0.5, 'maximum': 3.0},
    }),
    define_tool('set_grasp_mode', {
        'mode': {'type': 'string', 'enum': ['transport', 'contact']},
    }, argument_validators={'mode': bounded_text}),
    define_tool('replan_grasp', {
        'candidate_ref': {'type': 'string'},
        'transit_policy': {'type': 'string'},
    }),
    VALIDATE_GRASP,
    EXECUTE_GRASP,
)
