"""RobotUse integration: real CPU previews/plans and scripted Refiner sessions."""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
from PIL import Image

from src.llm.manager import ImageRegistry
from src.tools.pose_editor.adapter import PoseEditorMixin
from src.backend.orchestrator import AgentOrchestrator
from src.tools.names import public_tool_name
from test_backend import setup, candidates, paused
from test_place_motion import Backend as PlaceBackend, prepare
from test_orchestrator import Backend as AgentBackend, grasp_actions, pointer, height
from test_prime_delegation import Factory, a


def enable(backend):
    backend.__class__ = type('EditorBackend', (PoseEditorMixin, type(backend)), {})
    triangles = np.array([[[-.04, 0., 0.], [.04, 0., 0.], [0., 0., .10]]])
    if not hasattr(backend, 'gripper_assets'):
        backend.gripper_assets = NS()
    backend.gripper_assets.load_gripper_mesh = lambda width: ({'hand': triangles}, {})
    return backend


def deltas(**changes):
    return dict(dict(dx_mm=0., dy_mm=0., dz_mm=0., roll_deg=0., pitch_deg=0., yaw_deg=0.), **changes)


def assert_one_card(backend, result):
    paths = [backend.images.paths[ref].resolve() for ref in result['image_refs']]
    assert len(paths) == len(set(paths))
    cards = [path for path in paths if path.parent.name == 'pose_editor']
    assert len(cards) == 1
    assert Image.open(cards[0]).size == (1440, 1280)
    return cards[0]


@pytest.mark.parametrize('rotation,local_offset', [
    (np.diag([1., -1., -1.]), [25., 17., -1.]),
    (np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]]), [-17., 25., -1.]),
])
def test_card_serializes_base_offsets_without_mislabeled_local_values(setup, rotation, local_offset):
    backend = enable(setup[0])
    pose = np.eye(4); pose[:3, :3] = rotation
    pivot = np.array([.49, -.10, .02])
    pose[:3, 3] = pivot - backend.jaw_offset_m * rotation[:, 2]
    # A known world displacement must retain its signs even when the hand turns.
    center = pivot + np.array([.025, -.017, .001])
    points = center + np.array([[-.04, -.06, -.005], [.04, .06, .005]])
    backend._pose_card('grasp', 'frame-regression', points, points, pose, .075)
    cues = json.loads((backend.output_dir/'pose_editor/frame-regression.cues.json').read_text())
    assert cues['translation_frame'] == cues['target_center_mm_frame'] == 'base'
    assert cues['target_center_mm'] == cues['target_center_base_mm'] == [25., -17., 1.]
    np.testing.assert_allclose(cues['target_center_local_mm'], local_offset)
    assert 'X +25   Y -17   Z +1' in cues['lines'][0]
    assert cues['target_center_is_grasp_target'] is False
    assert cues['measured_width_scope'] == 'whole_observed_cloud'
    assert 'NOT a grasp target' in cues['scope']


def test_grasp_preview_edit_chain_and_base_coordinates(setup, monkeypatch):
    from unittest.mock import Mock
    from src.tools.pose_editor import adapter as pose_editor
    render = Mock(wraps=pose_editor.render_pose_card)
    monkeypatch.setattr(pose_editor, 'render_pose_card', render)
    backend = enable(setup[0])
    initial = candidates(backend, direction='median')['candidates'][0]
    ref = initial['candidate_ref']
    assert initial['image_refs']
    assert not backend.pose_editor_feedback(ref)
    assert render.call_count == 0
    assert not (backend.output_dir / 'pose_editor').exists()
    first_inspection = backend.inspect_candidate(ref)
    image = assert_one_card(backend, first_inspection)
    assert render.call_count == 1
    again = backend.inspect_candidate(ref)
    assert assert_one_card(backend, again) == image
    assert again['image_refs'][0] == first_inspection['image_refs'][0]
    assert render.call_count == 1
    assert all('pose_editor' not in str(p) for p in backend._candidate_inspection_cache[ref])
    cues = json.loads(image.with_suffix('.cues.json').read_text())
    assert cues['translation_frame'] == 'base' and cues['translation_unit'] == 'mm'
    edited = backend.refine_candidate(ref, **deltas(dx_mm=5, yaw_deg=5))
    assert edited['accepted']
    assert assert_one_card(backend, edited) != image
    assert render.call_count == 2
    from src.backend.controller import public_result
    assert public_result('refine_candidate', edited)['adjustment_deg']['yaw_deg'] == 5.
    assert edited['pose_editor']['applied'] == {'dx_mm': 5., 'yaw_deg': 5.}
    second = backend.refine_candidate(edited['candidate_ref'], **deltas(dx_mm=5))
    second_image = assert_one_card(backend, second)
    assert render.call_count == 3
    assert second['pose_editor']['cumulative_from_original']['dx_mm'] == 10.
    np.testing.assert_allclose(np.array(second['contact_center_xyz_m']) - initial['contact_center_xyz_m'], [.01, 0, 0])
    inspection = backend.inspect_candidate(second['candidate_ref'])
    assert assert_one_card(backend, inspection) == second_image
    assert render.call_count == 3
    assert inspection['pose_editor']['step'] == 2
    assert 'BASE mm' in inspection['geometry_cues']['lines'][0]
    restarted = backend.refine_candidate(ref, **deltas(dy_mm=3))
    assert restarted['pose_editor']['cumulative_from_original'] == {'dy_mm': 3.}
    assert not backend.connector.events


def test_place_preview_uses_measured_payload_and_mm_history(tmp_path):
    backend = enable(PlaceBackend())
    backend.output_dir = tmp_path
    backend.images = ImageRegistry()
    # The inherited test backend supplies calibrated projection placeholders.
    # Supply a real RGB overlay so the preview availability gate is exercised.
    rgb = tmp_path / 'rgb.png'; Image.new('RGB', (32, 32)).save(rgb)
    backend._explicit_project_place_hand = lambda *args, **kwargs: [backend.images.add(rgb)]
    original = prepare(backend)
    assert not backend.pose_editor_feedback(original['candidate_ref'])
    assert not (tmp_path / 'pose_editor').exists()
    inspection = backend.explicit_inspect_place(original['candidate_ref'])
    original_image = assert_one_card(backend, inspection)
    modified_at = original_image.stat().st_mtime_ns
    assert assert_one_card(backend, backend.explicit_inspect_place(original['candidate_ref'])) == original_image
    assert original_image.stat().st_mtime_ns == modified_at
    feedback = backend.pose_editor_feedback(original['candidate_ref'])
    assert feedback['geometry_cues']['kind'] == 'place'
    assert feedback['geometry_cues']['translation_unit'] == 'mm'
    assert feedback['geometry_cues']['support_z_m'] == pytest.approx(
        np.percentile(backend.destinations['dest'].points[:, 2], 98))
    first = backend.explicit_adjust_place(original['candidate_ref'], **deltas(dz_mm=10))
    assert assert_one_card(backend, first) != original_image
    second = backend.explicit_adjust_place(first['candidate_ref'], **deltas(dz_mm=10))
    assert_one_card(backend, second)
    assert second['pose_editor']['cumulative_from_original'] == {'dz_mm': 20.}
    assert backend.pose_editor_feedback(second['candidate_ref'])['pose_editor']['step'] == 2
    assert not backend.connector.events


def test_paused_nudges_show_new_card_and_rejection_keeps_history(setup):
    backend, checker, plan, entry = paused(setup)
    del backend._inflight_preview  # Exercise the preview implementation instead of the fixture.
    enable(backend)
    backend._paused_ref = 'initial'
    backend.inflight_feedback = {}
    backend.point_adapter.frames['obs'] = [NS(depth_m=np.array([[.3, .4], [.4, .5]]),
        intrinsics=np.eye(3), camera_to_base=NS(rotation=np.eye(3), translation=np.zeros(3)))]
    from src.tools.grasp.backend import hand_to_contact
    pivot_before = hand_to_contact(backend._inflight['current_pose'], backend.jaw_offset_m)[:3, 3]
    result = backend.nudge_inflight_grasp(**deltas(dx_mm=5, dy_mm=-7, dz_mm=3))
    assert result['accepted']
    assert_one_card(backend, result)
    assert result['image_refs'][0] == backend.inflight_feedback['image_refs'][0]
    assert backend.inflight_feedback['pose_editor']['cumulative_from_original'] == {
        'dx_mm': 5., 'dy_mm': -7., 'dz_mm': 3.}
    cues = backend.inflight_feedback['geometry_cues']
    np.testing.assert_allclose(np.array(cues['pivot']) - pivot_before, [.005, -.007, .003])
    expected_offset = ((np.array(cues['target_center']) - cues['pivot']) * 1000).round(1)
    np.testing.assert_allclose(cues['target_center_mm'], expected_offset)
    assert cues['target_center_mm'] == cues['target_center_base_mm']
    before = backend._inflight['current_pose'].copy()
    checker.accepted = False
    rejected = backend.nudge_inflight_grasp(**deltas(dx_mm=5))
    assert not rejected['accepted']
    np.testing.assert_array_equal(backend._inflight['current_pose'], before)
    assert backend.inflight_feedback['pose_editor']['step'] == 1


class EditorAgentBackend(AgentBackend):
    def pose_editor_feedback(self, candidate_ref):
        return dict(geometry_cues={'translation_frame': 'base', 'translation_unit': 'mm'},
                    pose_editor={'step': 1}, image_refs=['editor_card'])

    def adjust_grasp(self, **kwargs):
        self.calls.append(('adjust_grasp', kwargs))
        angles = {key: 0. for key in ('roll_deg', 'pitch_deg', 'yaw_deg')}
        return dict(candidate_ref='edited', accepted=True, image_refs=['editor_card'],
                    adjustment_deg=angles, cumulative_deg=angles)


def test_grasp_refiner_receives_editor_and_execution_uses_reviewed_candidate():
    backend = EditorAgentBackend()
    factory = Factory(prime=[[a('delegate_point', instruction='object'),
        a('delegate_grasp', instruction='pick', point_ref='p1'), a('finish', status='unknown')]],
        point=[pointer()], grasp=[[
            a(public_tool_name(action.tool), **action.arguments) for action in grasp_actions()]], refiner=[[
            a('inspect_candidate', candidate_ref='c1'),
            a('adjust_grasp', candidate_ref='c1', **deltas(dx_mm=5)),
            a('finish', status='success', candidate_ref='edited')]])
    runner = AgentOrchestrator(backend, factory, mandatory_observation=True, auto_refine_routes=True)
    runner._overhead_done = True
    result = runner.run('pick')
    assert not [e for e in result.events if e['kind'] in ('action_rejected', 'backend_error')]
    assert next(args['candidate_ref'] for tool, args in backend.calls if tool == 'execute_grasp') == 'edited'
    assert any(e.get('tool') == 'grasp_candidates' for e in result.events)
    assert not any(e.get('tool', '').startswith('explicit_') for e in result.events)
    refiner = next(s for role, _, s in factory.sessions if role == 'refiner')
    outputs = [m['content'] for messages, _ in refiner.inputs for m in messages
               if m.get('role') == 'tool' and m.get('tool') == 'adjust_grasp']
    assert outputs[-1]['pose_editor']['step'] == 1 and 'editor_card' in outputs[-1]['image_refs']
    # Metadata restoration must not append another image from feedback.
    inspections = [m['content'] for messages, _ in refiner.inputs for m in messages
                   if m.get('role') == 'tool' and m.get('tool') == 'inspect_candidate']
    assert 'editor_card' not in inspections[-1]['image_refs']


@pytest.mark.parametrize('review_enabled,accept', [(True, True), (True, False), (False, True)])
def test_place_refiner_edits_or_vetoes_only_when_enabled(review_enabled, accept):
    backend = EditorAgentBackend(); backend.held_plan = object(); backend.grasp_attempted = True
    review = ([a('inspect_place_candidate', candidate_ref='pc'),
               a('adjust_place', candidate_ref='pc', **deltas(dz_mm=5)),
               a('finish', status='success', candidate_ref='pc_adjusted')] if accept else
              [a('finish', status='failed', reason='No supported placement')])
    factory = Factory(prime=[[a('delegate_place', instruction='place', destination_ref='dest',
        hold_assessment='held', destination_assessment='unchanged'), a('finish', status='unknown')]],
        place=[[a('place_candidates', destination_ref='dest'),
                a('prepare_place', destination_ref='dest', xy_source='median', xy_m=[.4, -.2],
                  height=height(.2), transit_height=height(.6)), a('finish', status='success', candidate_ref='pc')]],
        refiner=[review])
    runner = AgentOrchestrator(backend, factory, auto_refine_routes=review_enabled)
    runner.destinations.add('dest')
    result = runner.run('place')
    assert not [e for e in result.events if e['kind'] in ('action_rejected', 'backend_error')]
    executions = [args['candidate_ref'] for tool, args in backend.calls if tool == 'explicit_execute_place']
    assert executions == (['pc'] if not review_enabled else ['pc_adjusted'] if accept else [])
    assert any(role == 'refiner' for role, _, _ in factory.sessions) is review_enabled
    assert any(e.get('tool') == 'prepare_place' for e in result.events)
    assert any(e.get('tool') == 'execute_place' for e in result.events) is accept
    assert not any(e.get('tool', '').startswith('explicit_') for e in result.events)
    assert not any(tool == 'release' for tool, _ in backend.calls)


def test_current_turn_image_selection_attaches_editor_card(tmp_path):
    images = ImageRegistry()
    refs = []
    for name in ('rgb', 'editor', 'overlay'):
        path = tmp_path / (name + '.png'); Image.new('RGB', (32, 32)).save(path)
        refs.append(images.add(path))
    messages = [{'role': 'user', 'content': {'current_observation': {'views': [{'image_ref': refs[0]}]}}},
                {'role': 'tool', 'tool': 'adjust_grasp',
                 'content': {'image_refs': refs[1:], 'pose_editor': {'step': 1}}}]
    selected = images.current_turn_refs(messages, role='refiner')
    assert refs[1] in selected


def test_paused_refiner_receives_cues_and_edit_history_after_nudge():
    backend = EditorAgentBackend()
    backend.inflight_feedback = backend.pose_editor_feedback('pending')
    backend.nudge_inflight_grasp = lambda **kwargs: dict(
        accepted=True, image_refs=['edited_card'], pose_editor={'step': 99})
    factory = Factory(refiner=[[a('nudge_grasp', **deltas(dx_mm=5)), a('finish', status='continue')]])
    runner = AgentOrchestrator(backend, factory)
    runner._current_observation = backend.observe()
    result = runner._delegate('refiner', dict(inflight_refinement=True, stage='pregrasp',
        instruction='review before closing', image_refs=['pending_card']), {'candidates': {}}, 'parent')
    assert result['result']['status'] == 'continue'
    refiner = factory.sessions[0][2]
    task = next(m['content'] for m in refiner.inputs[0][0] if m['role'] == 'user')
    assert task['geometry_cues']['translation_unit'] == 'mm'
    output = next(m['content'] for m in refiner.inputs[1][0]
                  if m.get('role') == 'tool' and m.get('tool') == 'nudge_grasp')
    assert output['pose_editor']['step'] == 1
    assert output['image_refs'] == ['edited_card']
