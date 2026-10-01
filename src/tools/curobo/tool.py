"""Register an internal optional planner without adding an agent action."""

from ..base_tool import ToolSpec, define_tool, dispatch_existing

TOOLS = (
    define_tool('curobo_plan', {
        'target': {'type': 'array', 'minItems': 4, 'maxItems': 4, 'items': {'type': 'array', 'minItems': 4, 'maxItems': 4, 'items': {'type': 'number'}}},
        'start_joints': {'type': 'array', 'minItems': 7, 'maxItems': 7, 'items': {'type': 'number'}},
        'world': {'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False},
    }, kind='backend', visibility='internal'),
)
