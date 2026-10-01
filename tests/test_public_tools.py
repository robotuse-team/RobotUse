"""Public tool names preserve backend schemas, dispatch and audit without global aliases."""
import json

import pytest

from src.backend.controller import ARGUMENTS, BoundaryError
from src.backend.delegation import DelegationOrchestrator
from src.tools.grasp.arguments import HEIGHT_SCHEMA, TRANSIT_SCHEMA
from src.backend.orchestrator import AgentOrchestrator
from src.tools.names import TOOL_SCHEMA_ALIASES
from test_prime_delegation import Factory
from test_orchestrator import Backend
from test_provider import send, functions, grasp_request, height


@pytest.mark.parametrize('native', [True, False])
@pytest.mark.parametrize('tool,arguments', [
    ('grasp_candidates', grasp_request('vertical')),
    ('place_candidates', {'destination_ref': 'dest'}),
    ('prepare_place', dict(destination_ref='dest', xy_source='median', xy_m=[.4, -.2],
                          height=height(.2), transit_height=height(.6))),
    ('adjust_place', dict(candidate_ref='candidate', dx_mm=5., dy_mm=0., dz_mm=0.,
                         roll_deg=0., pitch_deg=0., yaw_deg=0.)),
])
def test_public_names_and_original_schemas_on_wire(tmp_path, monkeypatch, native, tool, arguments):
    role = 'grasp' if tool == 'grasp_candidates' else 'place'
    factory = Factory()
    runner = AgentOrchestrator(Backend(), factory)
    body = send(tmp_path, monkeypatch, role=role, tools=runner._tools(role, {}), native=native,
                action=dict(tool=tool, arguments=arguments),
                tool_schema_aliases=factory.tool_schema_aliases,
                unrestricted_pose_rotation=factory.unrestricted_pose_rotation,
                unrestricted_pose_translation=factory.unrestricted_pose_translation,
                clicked_grasp_candidates=factory.clicked_grasp_candidates,
                requested_refine_routes=factory.requested_refine_routes,
                messages=[{'role': 'system', 'content': runner._prompt(role)}])
    assert all(internal not in json.dumps(body) for internal in TOOL_SCHEMA_ALIASES.values())
    if native:
        schema = functions(body)[tool]['parameters']
        assert set(schema['required']) == set(arguments)
        assert schema['additionalProperties'] is False
        properties = schema['properties']
    else:
        raw = body['messages'][0]['content'].split(' Tool argument types: ', 1)[1]
        schema_map, _ = json.JSONDecoder().raw_decode(raw)
        properties = schema_map[tool]
        assert set(properties) == set(arguments)
    if tool == 'grasp_candidates':
        assert properties['direction']['enum'] == ['vertical', 'custom', 'mean', 'median', 'clicked']
        assert properties['geometric_height'] == {**HEIGHT_SCHEMA, 'type': ['object', 'null']}
        assert properties['transit'] == TRANSIT_SCHEMA
    elif tool == 'prepare_place':
        assert properties['height'] == properties['transit_height'] == HEIGHT_SCHEMA
    elif tool == 'adjust_place':
        assert 'maximum' not in properties['dx_mm'] and 'minimum' not in properties['dx_mm']
        assert 'maximum' not in properties['yaw_deg'] and 'minimum' not in properties['yaw_deg']


def test_public_contracts_are_isolated_from_base_tool_contracts():
    original = dict(ARGUMENTS)
    controller = AgentOrchestrator(Backend(), Factory())
    base = DelegationOrchestrator(Backend(), Factory())
    for role in ('prime', 'point', 'grasp', 'place', 'refiner'):
        assert not any(t.startswith('explicit_') for t in controller._tools(role, {}))
        assert not any(t in controller._prompt(role) for t in TOOL_SCHEMA_ALIASES.values())
    assert 'adjust_place' in controller._tools('refiner', {'place_refinement': True})
    assert 'explicit_prepare_place' in base._tools('place', {})
    assert not hasattr(base.factory, 'tool_schema_aliases')
    assert ARGUMENTS == original
    assert controller._required_arguments('grasp_candidates') == ARGUMENTS['explicit_grasp_candidates']
    assert controller._required_arguments('grasp_candidates') != ARGUMENTS['grasp_candidates']
    assert controller._tool_argument('grasp_candidates', 'direction', 'median') == 'median'
    assert controller._tool_argument('grasp_candidates', 'geometric_height', None) is None
    assert controller._tool_argument('adjust_place', 'dx_mm', 310.) == 310.
    with pytest.raises(BoundaryError):
        base._tool_argument('explicit_adjust_place', 'dx_mm', 310.)
    with pytest.raises(BoundaryError):
        controller._tool_argument('prepare_place', 'height', {'value_m': .2})
