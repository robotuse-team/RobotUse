"""Scripted sessions cover optional RobotUse observation and the shared strict gate."""
import pytest

from src.backend.delegation import DelegationOrchestrator
from src.backend.orchestrator import AgentOrchestrator
from src.tools.names import public_tool_name
from test_prime_delegation import Factory, a
from test_orchestrator import Backend, grasp_actions


def pointer(obs='o0'):
    # Scripted Point reply for selecting a visible region.
    return [a('select_region', observation_id=obs, view_id='front', u=500, v=500),
            a('finish', point_ref='p1', reason='visible evidence',
              recommendation='inspect next state', evidence_image_refs=['front_rgb'])]


class ObservationBackend(Backend):
    def pose_editor_feedback(self, candidate_ref):
        return {}

    def propose_waypoint(self, **kwargs):
        return dict(waypoint_ref='view1', observation_id=kwargs['observation_id'], image_refs=['pose'])

    def validate_view(self, **kwargs):
        return dict(accepted=True, waypoint_ref='view1', validation_ref='vv')

    def execute_view(self, **kwargs):
        self.calls.append(('execute_view', kwargs))
        self.epoch += 1
        return {**self.observe(), 'view_status': 'achieved', 'purpose': 'observe',
                'target_update': {'tracking_status': 'needs_pointer'}}


def waypoint():
    return [a('propose_waypoint', observation_id='o0', u=500, v=500, purpose='observe',
              height_offset_m=.3, dx_m=0, dy_m=0, dz_m=0), a('finish', waypoint_ref='view1')]


def rejections(result):
    return [e['reason_code'] for e in result.events if e['kind'] == 'action_precondition_rejected']


@pytest.mark.parametrize('failed_attempts', [0, 2, 10])
def test_explicit_mandatory_policy_blocks_grasp_even_after_repeated_failures(failed_attempts):
    backend = ObservationBackend()
    factory = Factory(prime=[[a('delegate_point', instruction='object'),
        a('delegate_grasp', instruction='pick', point_ref='p1'), a('finish', status='unknown')]],
        point=[pointer()])
    orchestrator = DelegationOrchestrator(backend, factory, observe_before_grasp=False, mandatory_observation=True)
    orchestrator._overhead_attempts = failed_attempts
    result = orchestrator.run('pick')
    assert rejections(result) == ['observe_from_above_first']
    assert not any(role == 'grasp' for role, _, _ in factory.sessions)
    assert not any(tool in ('explicit_geometry', 'explicit_grasp_candidates', 'execute_grasp') for tool, _ in backend.calls)
    assert 'failed attempts never waive' in orchestrator._prompt('prime')
    assert 'requirement is waived' not in orchestrator._prompt('prime')


def test_observation_unlocks_grasp_and_release_requires_new_observation():
    backend = ObservationBackend()
    factory = Factory(prime=[[a('delegate_point', instruction='object'),
        a('delegate_grasp', instruction='pick', point_ref='p1'),
        a('delegate_waypoint', instruction='look from above'),
        a('validate_view', waypoint_ref='view1', motion='planned'),
        a('execute_view', waypoint_ref='view1', validation_ref='vv'),
        a('delegate_point', instruction='object in fresh image'),
        a('delegate_grasp', instruction='pick', point_ref='p1'),
        a('release'), a('delegate_point', instruction='object after release'),
        a('delegate_grasp', instruction='pick again', point_ref='p1'),
        a('finish', status='unknown')]],
        point=[pointer(), waypoint(), pointer('o1'), pointer('o3')], grasp=[grasp_actions()])
    orchestrator = DelegationOrchestrator(backend, factory, mandatory_observation=True)
    result = orchestrator.run('pick and release')
    assert not [e for e in result.events if e['kind'] in ('action_rejected', 'backend_error')]
    assert rejections(result) == ['observe_from_above_first', 'observe_from_above_first']
    assert [tool for tool, _ in backend.calls if tool in ('execute_view', 'execute_grasp', 'release')] == [
        'execute_view', 'execute_grasp', 'release']
    assert not orchestrator._overhead_done


@pytest.mark.parametrize('orchestrator_type', [DelegationOrchestrator, AgentOrchestrator])
@pytest.mark.parametrize('mandatory', [False, True])
def test_two_rejected_waypoints_waive_optional_policy(orchestrator_type, mandatory):
    backend = ObservationBackend()
    backend.validate_view = lambda **kwargs: dict(accepted=False, waypoint_ref='view1',
                                                 reason_code='waypoint_path_rejected')
    factory = Factory(prime=[[a('delegate_point', instruction='object'),
        a('delegate_waypoint', instruction='above'),
        a('validate_view', waypoint_ref='view1', motion='planned'),
        a('delegate_waypoint', instruction='above again'),
        a('validate_view', waypoint_ref='view1', motion='planned'),
        a('delegate_grasp', instruction='pick', point_ref='p1'), a('finish', status='unknown')]],
        point=[pointer(), waypoint(), waypoint()], grasp=[[
            a(public_tool_name(action.tool) if orchestrator_type is AgentOrchestrator else action.tool,
              **action.arguments) for action in grasp_actions()]])
    orchestrator = orchestrator_type(backend, factory, observe_before_grasp=True, mandatory_observation=mandatory)
    result = orchestrator.run('pick')
    assert orchestrator._overhead_attempts == 2
    assert rejections(result) == (['observe_from_above_first'] if mandatory else [])
    assert any(tool == 'execute_grasp' for tool, _ in backend.calls) is (not mandatory)


@pytest.mark.parametrize('observe_first', [False, True])
def test_default_allows_direct_grasp_or_agent_chosen_observation(observe_first):
    backend = ObservationBackend()
    observe = [a('delegate_waypoint', instruction='look from above'),
               a('validate_view', waypoint_ref='view1', motion='planned'),
               a('execute_view', waypoint_ref='view1', validation_ref='vv'),
               a('delegate_point', instruction='object in fresh image')]
    factory = Factory(prime=[[a('delegate_point', instruction='object'),
        *(observe if observe_first else []),
        a('delegate_grasp', instruction='pick', point_ref='p1'), a('finish', status='unknown')]],
        point=[pointer(), *([waypoint(), pointer('o1')] if observe_first else [])],
        grasp=[[a(public_tool_name(action.tool), **action.arguments) for action in grasp_actions()]])
    orchestrator = AgentOrchestrator(backend, factory)
    result = orchestrator.run('pick')
    assert not [e for e in result.events if e['kind'] in (
        'action_rejected', 'backend_error', 'action_precondition_rejected')]
    assert [tool for tool, _ in backend.calls if tool in ('execute_view', 'execute_grasp')] == (
        ['execute_view', 'execute_grasp'] if observe_first else ['execute_grasp'])
    assert not any(role == 'refiner' for role, _, _ in factory.sessions)
    assert 'Overhead observation before grasp is optional' in orchestrator._prompt('prime')
