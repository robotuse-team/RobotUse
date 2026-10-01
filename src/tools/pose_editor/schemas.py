"""The optional paused-release angles retain the configured RobotUse policy."""
from ..schema import object_schema


TRANSLATION = {'type': 'number', 'description': 'Finite signed BASE XYZ millimetres; no per-step or cumulative magnitude cap. The resulting pose and motion are checked.'}
ROTATION = {'type': 'number', 'description': 'Finite degrees about the gripper local axis; no angular magnitude budget. The resulting pose and motion are checked.'}

def nudge_place_schema(*, context=None, role=None, tools=()):
    properties = {key: {'type':'number', 'minimum':-30, 'maximum':30}
                  for key in ('dx_mm', 'dy_mm', 'dz_mm')}
    required = tuple(properties)
    features = getattr(context, 'interaction_features', None)
    if getattr(context, 'place_rotation', getattr(features, 'place_rotation', False)):
        properties.update({key: {'type':'number', 'minimum':-10., 'maximum':10.}
                           for key in ('roll_deg', 'pitch_deg', 'yaw_deg')})
    return object_schema(properties, required)
