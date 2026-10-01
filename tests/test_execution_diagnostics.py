"""Known planner stages and execution provenance survive actual public handoffs."""
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from src.tools.motion import planning as motion
from src.backend.robot import IntentBackend
from src.backend.intent import public_operation_result
from src.core.planning_feedback import public_planning_feedback
from src.backend.orchestrator import AgentOrchestrator
from test_prime_delegation import Factory, a
from test_backend import setup, candidates
from test_orchestrator import pointer, grasp_actions, height
from test_pose_editor import enable, deltas, EditorAgentBackend
from src.tools.names import public_tool_name


@pytest.mark.parametrize('failed_index', [0, 1, 2])
@pytest.mark.parametrize('failure', ['none', 'known', 'exception'])
def test_chain_reports_exact_failure_without_executing_or_leaking_errors(setup, failed_index, failure):
    backend = setup[0]
    current = motion._pose_transform(backend.connector.pose)
    targets = [current.copy() for _ in range(3)]
    targets[0][2, 3] += .2
    targets[1][:3, 3] += [.1, 0, .2]
    targets[2][:3, 3] += [.1, 0, -.1]
    backend._obstacle_world = lambda *args, **kwargs: (object(), {})
    original = backend.connector.ik.plan_linear
    calls = []
    def plan(start, target, *, seed_joints):
        index = len(calls)
        calls.append(index)
        if index == failed_index:
            if failure == 'known':
                raise motion.MotionPlanningError('SECRET', planning_feedback={
                    'kind': 'planning', 'planner_reason_code': 'world_route_failed', 'attempts': ['SECRET']})
            if failure == 'exception':
                raise RuntimeError('SECRET')
            return None
        return original(start, target, seed_joints=seed_joints)
    backend.connector.ik.plan_linear = plan
    backend._plan_world_segment = lambda target, seed, world: plan(
        motion.transform_to_pose(targets[0]), motion.transform_to_pose(target), seed_joints=seed)
    with pytest.raises(motion.MotionPlanningError) as caught:
        backend._explicit_chain_planner(backend.connector, targets, np.zeros((1, 3)), backend.motion_config,
            None, target_labels=('initial_lift', 'transit', 'waypoint'))
    feedback = public_planning_feedback(caught.value.planning_feedback)
    assert feedback['segment_index'] == failed_index
    assert feedback['segment'] == ('initial_lift', 'transit', 'waypoint')[failed_index]
    assert feedback['start_reference'] == ('current_ee' if failed_index == 0 else 'previous_planned_waypoint')
    expected = targets[failed_index][:3, 3] - (current if failed_index == 0 else targets[failed_index-1])[:3, 3]
    assert feedback['requested_translation_m'] == pytest.approx(dict(zip(('dx_m', 'dy_m', 'dz_m'), expected)))
    assert feedback['planner_reason_code'] == {'none': 'no_usable_route', 'known': 'world_route_failed',
                                               'exception': 'planner_exception'}[failure]
    assert 'SECRET' not in json.dumps(feedback)
    assert len(calls) == failed_index + 1 and not backend.connector.events


def test_validate_view_and_operation_handoff_preserve_known_planning_details(setup, monkeypatch):
    backend = setup[0]
    checker = NS(remove_captured_robot=lambda scene: (scene, {}))
    monkeypatch.setattr('src.tools.motion.path_collision.make_candidate_path_collision', lambda *a, **k: checker)
    backend.view_proposals['wp'] = dict(epoch=backend.epoch, observation_id='obs', start=backend._robot_state(),
        scene=np.zeros((1, 3)), purpose='transport', pose=np.eye(4), orientation_policy='preserve_current')
    known = dict(kind='planning', segment_index=1, segment='transit', planner_reason_code='world_route_failed')
    def plan(*args):
        raise motion.MotionPlanningError('SECRET', planning_feedback={**known, 'attempts': ['SECRET']})
    backend._plan_waypoint = plan
    output = backend.validate_view('wp')
    assert not output['accepted'] and output['validation_feedback'] == known
    assert public_operation_result({'validation': output})['planning_feedback'] == known
    assert 'SECRET' not in json.dumps(output)
    assert not backend.view_validations and not backend.connector.events


def test_failed_place_session_returns_tool_stage_to_prime_without_model_repetition():
    backend = EditorAgentBackend()
    known = dict(kind='planning', segment_index=2, segment='waypoint', planner_reason_code='ik_failed')
    backend.explicit_prepare_place = lambda **args: dict(candidate_ref='bad', accepted=False,
        validation={'accepted': False, 'validation_feedback': {**known, 'joints': ['SECRET']}}, image_refs=['preview'])
    factory = Factory(prime=[[a('delegate_place', instruction='place', destination_ref='dest',
        hold_assessment='held', destination_assessment='unchanged'), a('finish', status='unknown')]],
        place=[[a('prepare_place', destination_ref='dest', xy_source='median', xy_m=[.4, -.2],
            height=height(.2), transit_height=height(.6)), a('finish', status='failed', reason='cannot plan')]])
    runner = AgentOrchestrator(backend, factory)
    runner.destinations.add('dest')
    result = runner.run('place')
    output = next(e['result'] for e in result.events if e['kind'] == 'tool_result' and e.get('tool') == 'delegate_place')
    assert output['planning_diagnostics'][0]['planning_feedback'] == known
    assert 'SECRET' not in json.dumps(output['planning_diagnostics'])
    assert not runner._planning_stack
    assert not any(tool == 'explicit_execute_place' for tool, _ in backend.calls)


def test_original_selected_and_execution_pose_are_distinct_in_prime_feedback():
    backend = EditorAgentBackend()
    pose = np.eye(4).tolist()
    def execute(**args):
        backend.epoch += 1
        backend.last_execution_pose = dict(candidate_ref=args['candidate_ref'],
            reference_only=True, execution_target_pose_base=pose, paused_edit_applied=True)
        return dict(status='succeeded', execution_ref='exec')
    backend.execute_grasp = execute
    factory = Factory(prime=[[a('delegate_point', instruction='bagel'),
        a('delegate_grasp', instruction='pick bagel', point_ref='p1'), a('finish', status='unknown')]],
        point=[pointer()], grasp=[[
            a(public_tool_name(action.tool), **action.arguments) for action in grasp_actions()]], refiner=[[
            a('inspect_candidate', candidate_ref='c1'), a('adjust_grasp', candidate_ref='c1', **deltas(dx_mm=60.)),
            a('finish', status='success', candidate_ref='edited')]])
    runner = AgentOrchestrator(backend, factory, auto_refine_routes=True)
    result = runner.run('pick')
    feedback = next(e['result']['grasp_feedback'] for e in result.events
        if e['kind'] == 'tool_result' and e.get('tool') == 'delegate_grasp')
    assert feedback['original_selection']['candidate_ref'] == 'c1'
    assert feedback['selected_candidate']['candidate_ref'] == 'edited'
    assert feedback['execution_pose']['candidate_ref'] == 'edited'
    assert feedback['execution_pose']['execution_target_pose_base'] == pose
    assert runner._previous_grasp_feedback == feedback


@pytest.mark.parametrize('resumed', [True, False])
def test_backend_reports_paused_command_target_without_claiming_abort_was_executed(setup, monkeypatch, resumed):
    from src.backend.paused_refinement import PauseRefineIntentBackend
    backend = enable(setup[0])
    entry = candidates(backend, direction='median')['candidates'][0]
    ref = entry['candidate_ref']
    validation = backend.validate_grasp(ref)
    original = backend.candidate_routes[ref][1]
    replacement = deepcopy(original)
    replacement.grasp_transform[0, 3] += .16
    replacement.grasp_executed = resumed
    def execute(self, candidate_ref, validation_ref):
        self.epoch += 1
        self._inflight = dict(replacement=replacement, resume_authorized=resumed)
        return dict(status='succeeded' if resumed else 'failed', execution_ref='exec')
    monkeypatch.setattr(PauseRefineIntentBackend, 'execute_grasp', execute)
    out = backend.execute_grasp(ref, validation['validation_ref'])['execution_pose']
    assert out['paused_edit_applied'] is resumed
    assert out['grasp_plan_marked_executed'] is resumed
    assert out['execution_started'] is True
    original_pose = np.array(out['selected_target_pose_base'])
    pending = np.array(out['pending_edited_target_pose_base'])
    assert pending[0, 3] - original_pose[0, 3] == pytest.approx(.16)
    np.testing.assert_allclose(out['execution_target_pose_base'], pending if resumed else original_pose)
    assert 'not measured arrival' in out['scope']
