"""On-demand RobotUse review and unrestricted rotation keep validation/execution gates."""
import json

import numpy as np
import pytest

from src.tools.pose_editor.refinement import checked_adjustment, RefinementError
from src.backend.controller import BoundaryError
from src.backend.orchestrator import AgentOrchestrator
from src.tools.pose_editor.adapter import PoseEditorMixin
from src.tools.names import public_tool_name
from src.tools.grasp.backend import hand_to_contact
from test_prime_delegation import Factory, a
from test_orchestrator import grasp_actions, pointer, height
from test_backend import setup, candidates, paused
from test_place_motion import Backend as PlaceBackend, prepare
from test_provider import send, functions
from test_pose_editor import EditorAgentBackend, deltas, enable


def grasp_runner(status, review):
    backend = EditorAgentBackend()
    actions = [a(public_tool_name(action.tool), **action.arguments) for action in grasp_actions()]
    actions[-1] = a('finish', status=status, candidate_ref='c1', reason='Move contact onto rim')
    factory = Factory(prime=[[a('delegate_point', instruction='object'),
        a('delegate_grasp', instruction='pick', point_ref='p1'), a('finish', status='unknown')]],
        point=[pointer()], grasp=[actions], refiner=[review])
    return AgentOrchestrator(backend, factory, auto_refine_routes=False), backend, factory


@pytest.mark.parametrize('review_status', ['success', 'failed', 'needs_observation', 'missing'])
def test_explicit_grasp_request_reviews_one_candidate_and_requires_success(review_status):
    review = ([a('inspect_candidate', candidate_ref='c1'),
               a('adjust_grasp', candidate_ref='c1', **deltas(pitch_deg=180.)),
               a('finish', status='success', candidate_ref='edited')] if review_status == 'success' else
              [] if review_status == 'missing' else
              [a('finish', status=review_status, reason='No usable pose')])
    runner, backend, factory = grasp_runner('needs_refinement', review)
    result = runner.run('pick')
    assert [role for role, _, _ in factory.sessions].count('refiner') == 1
    executions = [args['candidate_ref'] for tool, args in backend.calls if tool == 'execute_grasp']
    assert executions == (['edited'] if review_status == 'success' else [])
    events = [e for e in result.events if e['kind'] == 'candidate_refinement']
    assert len(events) == 1 and events[0]['trigger'] == 'requested'
    assert events[0]['original_candidate_ref'] == 'c1'
    refiner = next(session for role, _, session in factory.sessions if role == 'refiner')
    task = refiner.inputs[0][0][1]['content']
    assert task['candidate_ref'] == 'c1' and 'Move contact onto rim' in task['instruction']
    assert not any(tool == 'release' for tool, _ in backend.calls)


def test_suitable_grasp_skips_preexecution_refiner_when_not_requested():
    runner, backend, factory = grasp_runner('success', [])
    runner.run('pick')
    assert 'refiner' not in [role for role, _, _ in factory.sessions]
    assert [args['candidate_ref'] for tool, args in backend.calls if tool == 'execute_grasp'] == ['c1']


@pytest.mark.parametrize('accept', [True, False])
def test_explicit_place_request_reviews_selected_pose_and_never_releases(accept):
    backend = EditorAgentBackend(); backend.held_plan = object(); backend.grasp_attempted = True
    review = ([a('inspect_place_candidate', candidate_ref='pc'),
               a('adjust_place', candidate_ref='pc', **deltas(pitch_deg=180.)),
               a('finish', status='success', candidate_ref='pc_adjusted')] if accept else
              [a('finish', status='failed', reason='Cannot reach requested orientation')])
    factory = Factory(prime=[[a('delegate_place', instruction='flip into bowl', destination_ref='dest',
        hold_assessment='held', destination_assessment='unchanged'), a('finish', status='unknown')]],
        place=[[a('place_candidates', destination_ref='dest'),
                a('prepare_place', destination_ref='dest', xy_source='median', xy_m=[.4, -.2],
                  height=height(.2), transit_height=height(.6)),
                a('finish', status='needs_refinement', candidate_ref='pc', reason='Invert held object')]],
        refiner=[review])
    runner = AgentOrchestrator(backend, factory, auto_refine_routes=False)
    runner.destinations.add('dest')
    result = runner.run('place')
    assert not [e for e in result.events if e['kind'] in ('action_rejected', 'backend_error')]
    assert [role for role, _, _ in factory.sessions].count('refiner') == 1
    assert [args['candidate_ref'] for tool, args in backend.calls if tool == 'explicit_execute_place'] == (
        ['pc_adjusted'] if accept else [])
    assert not any(tool == 'release' for tool, _ in backend.calls)


@pytest.mark.parametrize('tool', ['adjust_grasp', 'nudge_grasp', 'adjust_place'])
@pytest.mark.parametrize('native', [True, False])
def test_unrestricted_angles_match_actual_provider_and_boundary(tmp_path, monkeypatch, tool, native):
    runner = AgentOrchestrator(EditorAgentBackend(), Factory())
    args = deltas(roll_deg=180., pitch_deg=-450., yaw_deg=1080.)
    if tool != 'nudge_grasp':
        args['candidate_ref'] = 'chosen'
    body = send(tmp_path, monkeypatch, role='refiner', tools=(tool, 'finish'), native=native,
        action=dict(tool=tool, arguments=args), tool_schema_aliases=runner.factory.tool_schema_aliases,
        unrestricted_pose_rotation=runner.factory.unrestricted_pose_rotation,
        requested_refine_routes=runner.factory.requested_refine_routes)
    if native:
        props = functions(body)[tool]['parameters']['properties']
    else:
        raw = body['messages'][0]['content'].split(' Tool argument types: ', 1)[1]
        props = json.JSONDecoder().raw_decode(raw)[0][tool]
    for key in ('roll_deg', 'pitch_deg', 'yaw_deg'):
        assert props[key]['type'] == 'number'
        assert 'maximum' not in props[key] and 'minimum' not in props[key]
        assert runner._tool_argument(tool, key, args[key]) == args[key]
        for invalid in (True, '180', float('nan'), float('inf')):
            with pytest.raises(BoundaryError):
                runner._tool_argument(tool, key, invalid)
    assert runner._tool_argument(tool, 'dx_mm', 1000.) == 1000.


def test_place_provider_declares_explicit_refinement_request(tmp_path, monkeypatch):
    body = send(tmp_path, monkeypatch, role='place', tools=('finish',),
        action=dict(tool='finish', arguments=dict(status='needs_refinement', candidate_ref='chosen')),
        requested_refine_routes=True)
    assert 'needs_refinement' in functions(body)['finish']['parameters']['properties']['status']['enum']


def test_large_angles_still_require_finite_values_and_preserve_legacy_budget():
    step, total = checked_adjustment((180., -450., 1080.), (500., 0., 300.), unrestricted=True)
    assert step == (180., -450., 1080.) and total == (680., -450., 1380.)
    with pytest.raises(RefinementError):
        checked_adjustment((180., 0., 0.))
    for step, applied in [((float('nan'), 0., 0.), (0., 0., 0.)),
                          ((float('inf'), 0., 0.), (0., 0., 0.)),
                          ((1e308, 0., 0.), (1e308, 0., 0.))]:
        with pytest.raises(RefinementError):
            checked_adjustment(step, applied, unrestricted=True)


def test_geometric_large_rotation_is_replanned_preserves_pivot_and_origin(setup):
    backend = enable(setup[0]); checker = setup[2]
    original = candidates(backend, direction='median')['candidates'][0]
    ref = original['candidate_ref']
    pose = backend.candidates[ref][2].pose.copy()
    before_checks = len(checker.calls)
    edited = backend.refine_candidate(ref, **deltas(pitch_deg=90., yaw_deg=720.))
    assert edited['accepted'] and len(checker.calls) > before_checks
    assert edited['source'] == original['source'] == 'median'
    assert edited['source_top_down_only'] is True and edited['top_down_only'] is False
    assert backend.grasp_contracts[ref]['top_down_only'] is True
    np.testing.assert_array_equal(backend.candidates[ref][2].pose, pose)
    np.testing.assert_allclose(edited['contact_center_xyz_m'], original['contact_center_xyz_m'])
    assert not np.allclose(edited['approach_direction_base'], [0, 0, -1])
    second = backend.refine_candidate(edited['candidate_ref'], **deltas(pitch_deg=90., yaw_deg=720.))
    assert second['accepted']
    assert backend.candidate_adjustments[second['candidate_ref']] == (0., 180., 1440.)
    checker.accepted = False
    rejected = backend.refine_candidate(second['candidate_ref'], **deltas(roll_deg=180.))
    assert not rejected['accepted'] and not rejected['executable']
    assert not backend.connector.events


def test_paused_geometric_large_rotation_retains_transit_and_rejects_bad_path(setup):
    backend, checker, original, entry = paused(setup)
    backend.unrestricted_pose_rotation = PoseEditorMixin.unrestricted_pose_rotation
    before = backend._inflight['current_pose'].copy()
    result = backend.nudge_inflight_grasp(**deltas(pitch_deg=90., yaw_deg=720.))
    assert result['accepted']
    updated = backend._inflight['replacement']
    assert updated.grasp_contract['top_down_only'] is False
    assert updated.grasp_contract['source_top_down_only'] is True
    assert updated.pre_pick_z_m == original.pre_pick_z_m
    assert updated.post_pick_z_m == original.post_pick_z_m
    np.testing.assert_allclose(hand_to_contact(before, backend.jaw_offset_m)[:3, 3],
        hand_to_contact(updated.grasp_transform, backend.jaw_offset_m)[:3, 3])
    # Continuing from an edited geometric pose must not reapply its original yaw-only constraint.
    assert backend.nudge_inflight_grasp(**deltas(yaw_deg=450.))['accepted']
    accepted = backend._inflight['replacement']
    checker.accepted = False
    assert not backend.nudge_inflight_grasp(**deltas(roll_deg=180.))['accepted']
    assert backend._inflight['replacement'] is accepted
    assert not backend.connector.events


def test_place_can_flip_without_angle_budget_but_rejected_route_cannot_execute():
    backend = PlaceBackend()
    backend.unrestricted_pose_rotation = PoseEditorMixin.unrestricted_pose_rotation
    first = prepare(backend)
    ref = first['candidate_ref']
    initial_pose = backend._view(backend._explicit_place_proposal(ref)['waypoint_ref'])['pose'].copy()
    flip = backend.explicit_adjust_place(ref, **deltas(pitch_deg=180.))
    assert flip['accepted']
    flipped_pose = backend._view(backend._explicit_place_proposal(flip['candidate_ref'])['waypoint_ref'])['pose']
    assert not np.allclose(initial_pose[:3, :3], flipped_pose[:3, :3])
    np.testing.assert_allclose(flip['contact_center_xyz_m'], first['contact_center_xyz_m'])
    assert flip['transit_z_m'] == first['transit_z_m']
    more = backend.explicit_adjust_place(flip['candidate_ref'], **deltas(pitch_deg=540.))
    assert more['accepted']
    assert backend._explicit_place_proposal(more['candidate_ref'])['rotation_deg'] == [0., 720., 0.]
    backend.reject = True
    rejected = backend.explicit_adjust_place(more['candidate_ref'], **deltas(roll_deg=180.))
    assert not rejected['accepted']
    blocked = backend.explicit_execute_place(rejected['candidate_ref'])
    assert blocked['status'] == 'not_executed' and not blocked['executed']
    assert backend.connector.events == []


def test_requested_diagnostic_is_refined_before_validation_or_execution():
    review = [a('inspect_candidate', candidate_ref='c1'),
              a('adjust_grasp', candidate_ref='c1', **deltas(yaw_deg=90.)),
              a('finish', status='success', candidate_ref='edited')]
    runner, backend, factory = grasp_runner('needs_refinement', review)
    generate = backend.explicit_grasp_candidates
    def diagnostic(**args):
        result = generate(**args)
        result['diagnostic_candidates'] = result.pop('candidates')
        result['diagnostic_candidates'][0].update(executable=False, reason_code='path_rejected',
            source_view='front', score=0., rejection_reason='path rejected')
        result['candidates'] = []
        return result
    backend.explicit_grasp_candidates = diagnostic
    result = runner.run('pick')
    assert not [e for e in result.events if e['kind'] in ('action_rejected', 'backend_error')]
    assert [args['candidate_ref'] for tool, args in backend.calls if tool == 'execute_grasp'] == ['edited']
    assert [role for role, _, _ in factory.sessions].count('refiner') == 1


def test_nonfinite_backend_adjustment_retains_actual_reason_and_original_pose(setup):
    backend = enable(setup[0])
    first = candidates(backend, direction='median')['candidates'][0]
    ref = first['candidate_ref']
    original = backend.candidates[ref][2].pose.copy()
    result = backend.refine_candidate(ref, **deltas(pitch_deg=float('inf')))
    assert not result['accepted']
    assert result['error_details']['category'] == 'input'
    assert result['error_details']['phase'] == 'pose_adjustment'
    assert 'finite' in result['error_details']['message']
    np.testing.assert_array_equal(backend.candidates[ref][2].pose, original)
    assert not backend.connector.events


def test_paused_handoff_matches_unrestricted_rotation_policy(monkeypatch):
    runner = AgentOrchestrator(EditorAgentBackend(), Factory())
    runner._inflight_call = ({}, {'instruction': 'lift object'}, 'parent', 'prime')
    tasks = []
    def delegate(role, task, scope, sid):
        tasks.append(task)
        return dict(status='completed', result=dict(status='continue'))
    monkeypatch.setattr(runner, '_delegate', delegate)
    assert runner._inflight_refinement('pregrasp', {})['status'] == 'continue'
    assert 'without angular magnitude limits' in tasks[0]['instruction']
    assert 'yaw only' not in tasks[0]['instruction']


def test_live_agent_inputs_and_provider_payloads_report_unrestricted_rotation(tmp_path, monkeypatch):
    backend = EditorAgentBackend()
    actions = [a(public_tool_name(action.tool), **action.arguments) for action in grasp_actions()]
    actions[-1] = a('finish', status='needs_refinement', candidate_ref='c1', reason='Rotate contact')
    factory = Factory(prime=[[a('delegate_point', instruction='object'),
        a('delegate_grasp', instruction='pick', point_ref='p1'),
        a('delegate_place', instruction='place', destination_ref='dest',
          hold_assessment='held', destination_assessment='unchanged'), a('finish', status='unknown')]],
        point=[pointer()], grasp=[actions],
        refiner=[[a('inspect_candidate', candidate_ref='c1'),
                  a('finish', status='success', candidate_ref='c1')]],
        place=[[a('place_candidates', destination_ref='dest'),
                a('prepare_place', destination_ref='dest', xy_source='median', xy_m=[.4, -.2],
                  height=height(.2), transit_height=height(.6)),
                a('finish', status='success', candidate_ref='pc')]])
    runner = AgentOrchestrator(backend, factory, auto_refine_routes=False)
    runner.destinations.add('dest')
    result = runner.run('transfer object')
    assert not [e for e in result.events if e['kind'] in ('action_rejected', 'backend_error')]
    inputs = [e for e in result.events if e['kind'] == 'agent_input']
    assert {'prime', 'point', 'grasp', 'refiner', 'place'} == {e['role'] for e in inputs}
    for event in inputs:
        features = event['interaction_features']
        assert features['rotation_step_limit_deg'] is None
        assert features['rotation_cumulative_limit_deg'] is None
        assert features['place_pose_rotation'] is True
        assert 'place_rotation' not in features
        assert features['legacy_nudge_place_enabled'] is False
        assert features['rotation_limit_scope'] == ['adjust_grasp', 'nudge_grasp', 'adjust_place']
    # Exercise the real provider serializer with the exact messages received by
    # all four decision roles, rather than checking metadata() in isolation.
    for role in ('prime', 'grasp', 'refiner', 'place'):
        session = next(session for kind, _, session in factory.sessions if kind == role)
        messages, allowed = session.inputs[0]
        out = tmp_path / role; out.mkdir()
        body = send(out, monkeypatch, role=role, tools=allowed, messages=messages,
            action=dict(tool='finish', arguments=dict(status='failed')),
            tool_schema_aliases=runner.factory.tool_schema_aliases,
            unrestricted_pose_rotation=True, requested_refine_routes=True)
        wire = '\n'.join(message['content'] if isinstance(message['content'], str)
            else '\n'.join(block.get('text', '') for block in message['content'])
            for message in body['messages'])
        # Public context is JSON text inside the provider message text block.
        assert '\"rotation_step_limit_deg\": null' in wire
        assert '\"rotation_cumulative_limit_deg\": null' in wire
        assert '\"rotation_step_limit_deg\": 10' not in wire
        assert '\"rotation_cumulative_limit_deg\": 30' not in wire


def test_backend_and_orchestrator_feature_metadata_preserve_legacy_switches(setup):
    from src.backend.interaction import InteractionFeatures
    from src.tools.grasp.backend import GraspBackend
    from src.backend.delegation import DelegationOrchestrator
    backend = enable(setup[0])
    metadata = backend.interaction_features.metadata()
    assert metadata['rotation_step_limit_deg'] is None
    assert metadata['rotation_cumulative_limit_deg'] is None
    assert metadata['place_pose_rotation'] is True
    assert metadata['legacy_nudge_place']['agent_tool_exposed'] is False
    assert backend.interaction_features.place_rotation == GraspBackend.interaction_features.place_rotation is True
    assert AgentOrchestrator.interaction_features.place_rotation == DelegationOrchestrator.interaction_features.place_rotation is False
    for legacy in (InteractionFeatures(), GraspBackend.interaction_features, DelegationOrchestrator.interaction_features):
        assert legacy.metadata()['rotation_step_limit_deg'] == 10
        assert legacy.metadata()['rotation_cumulative_limit_deg'] == 30
        assert 'place_rotation' in legacy.metadata()
    assert 'nudge_place' not in AgentOrchestrator(backend, Factory())._tools('refiner', {})
