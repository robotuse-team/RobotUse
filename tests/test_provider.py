"""Exercise RobotUse provider requests with the shared offline transport fixture.

Cover tool policies, clicked pose selection, and unbounded editing.
"""
import json
from types import SimpleNamespace

import pytest

from src.llm.manager import ImageRegistry, OpenRouterSession
from src.backend.delegation import DelegationOrchestrator  # Registers unique tool names.
from src.tools.grasp.arguments import HEIGHT_SCHEMA, TRANSIT_SCHEMA


def height(value, mode='absolute', reference='none'):
    refs = dict(none='absolute', clicked='clicked_point', median='segment_median',
                mean='segment_mean', min='observed_min', max='observed_max')
    return dict(value_m=value, reference=refs.get(reference, reference) if mode == 'absolute' or reference != 'none' else 'none')


def grasp_request(direction):
    return dict(point_ref='p_current', direction=direction, tolerance_deg=20.,
        azimuth_deg=60. if direction == 'custom' else None,
        polar_deg=70. if direction == 'custom' else None,
        geometric_height=None,
        transit=dict(pre=height(.4), post=height(.6)))


def send(tmp_path, monkeypatch, *, role, tools, action, native=True, pose_editor=True, policy='agent-choice',
         messages=None, images=None, image_history_policy='current_turn', tool_schema_aliases=None,
         unrestricted_pose_rotation=False, requested_refine_routes=False,
         unrestricted_pose_translation=False, clicked_grasp_candidates=False):
    factory = SimpleNamespace(images=ImageRegistry(), model='offline-test', key='offline-test',
        output_dir=tmp_path, json_action_fallback=not native, active_perception=True,
        intent_driven=True, review_driven=True, target_intent_mode=True,
        explicit_geometry_enabled=pose_editor, grasp_policy=policy, image_history_policy=image_history_policy,
        tool_schema_aliases=tool_schema_aliases or {}, unrestricted_pose_rotation=unrestricted_pose_rotation,
        unrestricted_pose_translation=unrestricted_pose_translation, clicked_grasp_candidates=clicked_grasp_candidates,
        requested_refine_routes=requested_refine_routes)
    session = OpenRouterSession(factory, role, 'tool-schema-session')
    if images is not None:
        factory.images = images
    message = ({'tool_calls': [{'id': 'test-call', 'type': 'function', 'function': {
        'name': action['tool'], 'arguments': json.dumps(action['arguments'])}}]} if native else
        {'content': json.dumps(action)})
    payload = {'choices': [{'finish_reason': 'tool_calls' if native else 'stop', 'message': message}]}
    captured = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps(payload).encode()

    def request(req, **kwargs):
        captured.append(json.loads(req.data))
        return Response()

    monkeypatch.setattr('src.llm.manager.urllib.request.urlopen', request)
    selected = session.next_action(messages or [{'role': 'user', 'content': 'Choose from the current measured geometry.'}], tools)
    assert selected.tool == action['tool'] and selected.arguments == action['arguments']
    assert len(captured) == 1
    return captured[0]


def functions(body):
    return {tool['function']['name']: tool['function'] for tool in body['tools']}


@pytest.mark.parametrize('image_history_policy', ['current_turn', 'bounded_history'])
@pytest.mark.parametrize('tool', ['inspect_candidate', 'adjust_grasp'])
def test_provider_deduplicates_file_aliases(tmp_path, monkeypatch, image_history_policy, tool):
    from PIL import Image
    images = ImageRegistry()
    card = tmp_path / 'card.png'
    cyan = tmp_path / 'cyan.png'
    Image.new('RGB', (8, 8), 'orange').save(card)
    Image.new('RGB', (8, 8), 'cyan').save(cyan)
    alias = tmp_path / 'alias.png'
    alias.symlink_to(card)
    # Distinct opaque IDs (and a path alias) must not attach the same file twice.
    refs = [images.add(card), images.add(card), images.add(alias), images.add(cyan)]
    body = send(tmp_path, monkeypatch, role='grasp', tools=('finish',),
        action=dict(tool='finish', arguments=dict(status='failed')), images=images,
        image_history_policy=image_history_policy,
        messages=[{'role': 'tool', 'tool': tool, 'content': {'image_refs': refs}}])
    urls = [b['image_url']['url'] for m in body['messages'] if isinstance(m['content'], list)
            for b in m['content'] if b.get('type') == 'image_url']
    assert len(urls) == len(set(urls)) == 2


@pytest.mark.parametrize('pose_editor,tool,preview_present,expected', [
    (True, 'inspect_candidate', True, 1),
    (False, 'inspect_candidate', True, 2),
    (True, 'preview_candidate', True, 2),
    (True, 'inspect_candidate', False, 1),
])
def test_inspection_omits_scene_only_for_pose_editor_with_preview(tmp_path, monkeypatch, pose_editor, tool, preview_present, expected):
    from PIL import Image
    images = ImageRegistry()
    refs = []
    for name, color in [('scene', 'red'), ('preview', 'cyan')]:
        path = tmp_path / (name + '.png')
        Image.new('RGB', (8, 8), color).save(path)
        refs.append(images.add(path))
    body = send(tmp_path, monkeypatch, role='grasp', tools=('finish',),
        action=dict(tool='finish', arguments=dict(status='failed')), pose_editor=pose_editor, images=images,
        messages=[{'role': 'user', 'content': 'Inspect this candidate.'},
                  {'role': 'tool', 'tool': tool, 'content': {'image_refs': refs[1:] if preview_present else []}},
                  {'role': 'user', 'content': {'current_observation': {'views': [{'image_ref': refs[0]}]}}}])
    blocks = [b for m in body['messages'] if isinstance(m['content'], list) for b in m['content']]
    assert sum(b.get('type') == 'image_url' for b in blocks) == expected
    attached_labels = [b.get('text') for b in blocks if b.get('text') in refs]
    assert (refs[0] not in attached_labels) == (pose_editor and tool == 'inspect_candidate' and preview_present)


def test_observation_motion_choices_reach_actual_provider_request(tmp_path, monkeypatch):
    body = send(tmp_path, monkeypatch, role='prime', tools=('validate_view',),
        action=dict(tool='validate_view', arguments=dict(waypoint_ref='wp_current', motion='planned')))
    schema = functions(body)['validate_view']['parameters']['properties']['motion']
    assert schema['enum'] == ['planned', 'linear']
    from src.tools.grasp.arguments import validate_argument
    from src.backend.controller import BoundaryError
    with pytest.raises(BoundaryError, match='motion must be one of'):
        validate_argument('validate_view', 'motion', 'move_arm')


@pytest.mark.parametrize('direction', ['vertical', 'custom'])
def test_native_grasp_request_declares_nested_heights_and_nullable_direction_angles(tmp_path, monkeypatch, direction):
    arguments = grasp_request(direction)
    body = send(tmp_path, monkeypatch, role='grasp', tools=('grasp_candidates', 'finish'),
                action=dict(tool='grasp_candidates', arguments=arguments))
    tool = functions(body)['grasp_candidates']['parameters']
    assert set(tool['required']) == set(arguments)
    assert tool['additionalProperties'] is False
    properties = tool['properties']
    assert properties['geometric_height'] == {**HEIGHT_SCHEMA, 'type': ['object', 'null']}
    assert properties['transit'] == TRANSIT_SCHEMA
    assert properties['direction']['enum'] == ['vertical', 'custom', 'mean', 'median', 'clicked']
    assert properties['azimuth_deg']['type'] == ['number', 'null']
    assert properties['polar_deg'] == {'type': ['number', 'null'], 'minimum': 0, 'maximum': 180}
    assert properties['tolerance_deg'] == {'type': ['number', 'null'], 'minimum': 0, 'maximum': 180}
    assert 'geometric_yaw_deg' not in properties
    assert body['parallel_tool_calls'] is False and body['tool_choice'] == 'required'
    assert body['session_id'] == 'tool-schema-session'
    assert body['max_tokens'] == 8192
    instructions = body['messages'][0]['content']
    assert 'Use the declared metric coordinate, height and angular fields.' in instructions
    assert 'Never output code, metric poses or joints.' not in instructions


@pytest.mark.parametrize('native', [True, False])
@pytest.mark.parametrize('direction', ['vertical', 'custom'])
def test_json_and_native_transport_preserve_agent_numbers_nulls_and_nested_choices(tmp_path, monkeypatch, native, direction):
    arguments = grasp_request(direction)
    body = send(tmp_path, monkeypatch, native=native, role='grasp', tools=('grasp_candidates', 'finish'),
                action=dict(tool='grasp_candidates', arguments=arguments))
    if native:
        schemas = functions(body)['grasp_candidates']['parameters']['properties']
    else:
        assert body['response_format'] == {'type': 'json_object'}
        assert 'tools' not in body and 'parallel_tool_calls' not in body
        instructions = body['messages'][0]['content']
        raw = instructions.split(' Tool argument types: ', 1)[1]
        schema_map, _ = json.JSONDecoder().raw_decode(raw)
        schemas = schema_map['grasp_candidates']
        assert 'Never output code, metric poses or joints.' not in instructions
    assert schemas['geometric_height'] == {**HEIGHT_SCHEMA, 'type': ['object', 'null']}
    assert schemas['transit'] == TRANSIT_SCHEMA
    assert schemas['polar_deg']['type'] == ['number', 'null']
    audit = json.loads((tmp_path/'provider_audit.jsonl').read_text())
    assert audit['response']['arguments'] == arguments
    assert audit['native_tool_calling'] is native


def test_place_native_schema_supports_finite_unbounded_adjustments_without_releasing(tmp_path, monkeypatch):
    body = send(tmp_path, monkeypatch, role='place',
        tools=('place_candidates', 'prepare_place', 'adjust_place', 'inspect_place_candidate', 'finish'),
        action=dict(tool='prepare_place', arguments=dict(destination_ref='dest', xy_source='mean',
            xy_m=[.43, -.24], height=height(.18), transit_height=height(.6))))
    tools = functions(body)
    prepare = tools['prepare_place']['parameters']
    assert prepare['properties']['xy_m'] == dict(type='array', items={'type': 'number'}, minItems=2, maxItems=2)
    assert prepare['properties']['xy_source']['enum'] == ['clicked', 'median', 'mean']
    assert prepare['properties']['height'] == HEIGHT_SCHEMA
    assert prepare['properties']['transit_height'] == HEIGHT_SCHEMA
    assert 'transit_height' in prepare['required']
    assert {'current_tcp', 'grasp'} <= set(HEIGHT_SCHEMA['properties']['reference']['enum'])
    adjustment = tools['adjust_place']['parameters']
    assert set(adjustment['required']) == {'candidate_ref', 'dx_mm', 'dy_mm', 'dz_mm', 'roll_deg', 'pitch_deg', 'yaw_deg'}
    for key in ('dx_mm', 'dy_mm', 'dz_mm', 'roll_deg', 'pitch_deg', 'yaw_deg'):
        assert adjustment['properties'][key]['type'] == 'number'
        assert 'minimum' not in adjustment['properties'][key]
        assert 'maximum' not in adjustment['properties'][key]
    assert 'release' not in tools


def test_delegation_only_accepts_instruction_and_point_reference(tmp_path, monkeypatch):
    body = send(tmp_path, monkeypatch, role='prime',
        tools=('delegate_grasp', 'finish'), action=dict(tool='delegate_grasp',
        arguments=dict(point_ref='p1', instruction='pick')))
    schema = functions(body)['delegate_grasp']['parameters']
    assert set(schema['required']) == {'point_ref', 'instruction'}
    assert set(schema['properties']) == {'point_ref', 'instruction'}
    assert 'preferred_direction' not in schema['properties']
    assert 'batch_size' not in schema['properties']
    assert 'Use the declared metric coordinate' in body['messages'][0]['content']


def test_prime_motion_schema_keeps_separate_explicit_release(tmp_path, monkeypatch):
    body = send(tmp_path, monkeypatch, role='prime',
        tools=('move_vertical', 'goto_home_joint_position', 'release', 'finish'),
        action=dict(tool='move_vertical', arguments={'dz_m': -.12}))
    tools = functions(body)
    assert tools['move_vertical']['parameters']['properties']['dz_m']['type'] == 'number'
    assert tools['move_vertical']['parameters']['required'] == ['dz_m']
    assert tools['goto_home_joint_position']['parameters']['required'] == []
    assert tools['release']['parameters']['required'] == []


@pytest.mark.parametrize('native', [False, True])
def test_keeps_bounded_preview_and_unbounded_pose_edit_schemas(tmp_path, monkeypatch, native):
    body = send(tmp_path, monkeypatch, role='refiner', native=native,
        tools=('preview_candidate', 'adjust_grasp', 'finish'), action=dict(tool='finish', arguments={'status': 'failed'}))
    if native:
        schemas = {name: definition['parameters']['properties'] for name, definition in functions(body).items()}
    else:
        raw = body['messages'][0]['content'].split(' Tool argument types: ', 1)[1]
        schemas, _ = json.JSONDecoder().raw_decode(raw)
    assert schemas['preview_candidate']['azimuth_deg'] == {
        'type': 'number', 'minimum': -180., 'maximum': 180.}
    for key in ('dx_mm', 'dy_mm', 'dz_mm', 'roll_deg', 'pitch_deg', 'yaw_deg'):
        assert schemas['adjust_grasp'][key]['type'] == 'number'
        assert 'minimum' not in schemas['adjust_grasp'][key]
        assert 'maximum' not in schemas['adjust_grasp'][key]


@pytest.mark.parametrize('paused', [False, True])
def test_refiner_finish_schema_matches_runtime_decisions(tmp_path, monkeypatch, paused):
    tools = ('nudge_grasp', 'finish') if paused else ('inspect_candidate', 'adjust_grasp', 'finish')
    body = send(tmp_path, monkeypatch, role='refiner', tools=tools,
        action=dict(tool='finish', arguments={'status': 'abort' if paused else 'needs_observation'}))
    schema = functions(body)['finish']['parameters']
    if paused:
        assert schema['properties']['status']['enum'] == ['continue', 'abort']
        assert schema['required'] == ['status']
    else:
        assert 'needs_observation' in schema['properties']['status']['enum']


@pytest.mark.parametrize('mode', ['mean', 'median'])
@pytest.mark.parametrize('native', [True, False])
def test_center_modes_reach_provider_without_unused_yaw_or_angles(tmp_path, monkeypatch, mode, native):
    arguments = dict(point_ref='p_current', direction=mode, tolerance_deg=None,
        azimuth_deg=None, polar_deg=None, geometric_height=height(.04),
        transit=dict(pre=height(.4), post=height(.25, reference='grasp')))
    send(tmp_path, monkeypatch, role='grasp', native=native, tools=('explicit_grasp_candidates',),
         action=dict(tool='explicit_grasp_candidates', arguments=arguments))
