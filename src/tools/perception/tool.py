"""Point selection uses the existing SAM2 and calibrated RGB-D pipeline."""

from ..base_tool import define_tool
from ..schema import normalized_pixel

TOOLS = (
    define_tool('select_region', {
        'observation_id': {'type': 'string'},
        'view_id': {'type': 'string'},
        'u': {'type': 'number', 'minimum': 0, 'maximum': 1000},
        'v': {'type': 'number', 'minimum': 0, 'maximum': 1000},
    }, argument_validators={'u': normalized_pixel, 'v': normalized_pixel}),
)
