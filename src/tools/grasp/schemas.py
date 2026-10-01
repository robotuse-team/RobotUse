"""Shared contact-height geometry for grasp and place tool inputs."""

HEIGHT_SCHEMA = {'type': 'object',
 'properties': {'value_m': {'type': 'number'},
                'reference': {'type': 'string',
                              'enum': ['absolute',
                                       'grasp',
                                       'current_tcp',
                                       'clicked_point',
                                       'segment_median',
                                       'segment_mean',
                                       'observed_min',
                                       'observed_max']}},
 'required': ['reference', 'value_m'],
 'additionalProperties': False,
 'description': 'Contact-center base Z in metres: absolute uses value_m directly; other '
                'references add a signed offset. grasp means this candidate grasp Z or '
                'the held grasp Z; current_tcp is measured when planning. Unavailable '
                'references reject.'}

TRANSIT_SCHEMA = {
    'type': 'object', 'properties': {'pre': HEIGHT_SCHEMA, 'post': HEIGHT_SCHEMA},
    'required': ['pre', 'post'], 'additionalProperties': False,
    'description': 'Independent contact-center heights before approach and after pickup.',
}
