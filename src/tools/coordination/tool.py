"""Definitions owned by the coordination tools."""

from ..base_tool import ToolSpec, define_tool, dispatch_existing
from ..schema import bounded_text
from .schemas import finish_schema, finish_description

DELEGATE_REFINER = define_tool('delegate_refiner', {
        'instruction': {'type': 'string'},
        'candidate_ref': {'type': 'string'},
    }, visibility='internal')

TOOLS = (
    define_tool('delegate_point', {
        'instruction': {'type': 'string'},
    }),
    define_tool('delegate_waypoint', {
        'instruction': {'type': 'string'},
    }),
    define_tool('delegate_grasp', {
        'instruction': {'type': 'string'},
        'point_ref': {'type': 'string'},
    }),
    define_tool('delegate_destination', {
        'instruction': {'type': 'string'},
    }),
    define_tool('delegate_place', {
        'instruction': {'type': 'string'},
        'destination_ref': {'type': 'string'},
        'hold_assessment': {'type': 'string', 'enum': ['held', 'empty', 'uncertain']},
        'destination_assessment': {'type': 'string', 'enum': ['unchanged', 'changed', 'uncertain']},
    }, argument_validators={'hold_assessment': bounded_text, 'destination_assessment': bounded_text}),
    DELEGATE_REFINER,
    ToolSpec("finish", (), dispatch_existing, kind="control", input_schema=finish_schema, description=finish_description),
)
