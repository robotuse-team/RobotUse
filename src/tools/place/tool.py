"""Definitions owned by place; RobotUse does not load a placement prediction model."""

from ..base_tool import ToolSpec, define_tool, dispatch_existing

from ..grasp.schemas import HEIGHT_SCHEMA

SAVE_DESTINATION = define_tool('save_destination', {
        'point_ref': {'type': 'string'},
    }, visibility='internal')

EXECUTE_PLACE = define_tool('execute_place', {
        'candidate_ref': {'type': 'string'},
    }, internal_name='explicit_execute_place', visibility='internal')

TOOLS = (
    define_tool('place_candidates', {
        'destination_ref': {'type': 'string'},
    }, internal_name='explicit_place_candidates'),
    define_tool('prepare_place', {
        'destination_ref': {'type': 'string'},
        'xy_source': {'type': 'string', 'enum': ['clicked', 'median', 'mean']},
        'xy_m': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2},
        'height': HEIGHT_SCHEMA,
        'transit_height': HEIGHT_SCHEMA,
    }, internal_name='explicit_prepare_place'),
    define_tool('inspect_place_candidate', {
        'candidate_ref': {'type': 'string'},
    }),
    SAVE_DESTINATION,
    EXECUTE_PLACE,
)
