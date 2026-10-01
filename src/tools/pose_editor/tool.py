"""Definitions owned by pose editing; existing boundary checks remain in force."""

from ..base_tool import ToolSpec, define_tool, dispatch_existing
from .schemas import nudge_place_schema, TRANSLATION, ROTATION

REFINE_CANDIDATE = define_tool('refine_candidate', {
        'candidate_ref': {'type': 'string'},
        'roll_deg': {'type': 'number', 'minimum': -10.0, 'maximum': 10.0},
        'pitch_deg': {'type': 'number', 'minimum': -10.0, 'maximum': 10.0},
        'yaw_deg': {'type': 'number', 'minimum': -10.0, 'maximum': 10.0},
    })

TOOLS = (
    define_tool('adjust_grasp', {
        'candidate_ref': {'type': 'string'},
        'dx_mm': TRANSLATION,
        'dy_mm': TRANSLATION,
        'dz_mm': TRANSLATION,
        'roll_deg': ROTATION,
        'pitch_deg': ROTATION,
        'yaw_deg': ROTATION,
    }),
    define_tool('adjust_place', {
        'candidate_ref': {'type': 'string'},
        'dx_mm': TRANSLATION,
        'dy_mm': TRANSLATION,
        'dz_mm': TRANSLATION,
        'roll_deg': ROTATION,
        'pitch_deg': ROTATION,
        'yaw_deg': ROTATION,
    }, internal_name='explicit_adjust_place'),
    define_tool('refine_view', {
        'waypoint_ref': {'type': 'string'},
        'dx_mm': {'type': 'number', 'minimum': -10, 'maximum': 10},
        'dy_mm': {'type': 'number', 'minimum': -10, 'maximum': 10},
        'dz_mm': {'type': 'number', 'minimum': -10, 'maximum': 10},
        'roll_deg': {'type': 'number', 'minimum': -10.0, 'maximum': 10.0},
        'pitch_deg': {'type': 'number', 'minimum': -10.0, 'maximum': 10.0},
        'yaw_deg': {'type': 'number', 'minimum': -10.0, 'maximum': 10.0},
    }),
    define_tool('nudge_grasp', {
        'dx_mm': TRANSLATION,
        'dy_mm': TRANSLATION,
        'dz_mm': TRANSLATION,
        'roll_deg': ROTATION,
        'pitch_deg': ROTATION,
        'yaw_deg': ROTATION,
    }),
    ToolSpec("nudge_place", ("dx_mm", "dy_mm", "dz_mm"), dispatch_existing, input_schema=nudge_place_schema),
    define_tool('relax_candidate', {
        'candidate_ref': {'type': 'string'},
    }),
    # The agent loop tracks this tool's inspection budget separately.
    REFINE_CANDIDATE,
)
