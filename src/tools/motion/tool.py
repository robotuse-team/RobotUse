"""Definitions owned by motion; the configured native planner is unchanged."""

from ..base_tool import define_tool
from ..schema import validate_value
from src.core.contracts import BoundaryError


def _turn_angle(value, schema):
    angle = validate_value(schema, value, "angle_deg")
    if angle == 0:
        raise BoundaryError('turn requires a nonzero angle within +/-180 degrees')
    return angle

TOOLS = (
    define_tool('move_vertical', {
        'dz_m': {'type': 'number', 'description': 'Signed base-Z displacement in metres; preserves gripper command.'},
    }),
    define_tool('goto_home_joint_position', {
    }),
    define_tool('turn', {
        'angle_deg': {'type': 'number', 'minimum': -180.0, 'maximum': 180.0},
    }, argument_validators={'angle_deg': _turn_angle}),
)
