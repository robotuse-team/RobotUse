"""Observed error categories, public redaction and agent delivery paths."""
import json
import urllib.error

import pytest

from src.core.action_feedback import ActionPreconditionError
from src.tools.motion.planning import MotionPlanningError
from src.backend.intent import public_operation_result
from src.backend.controller import BoundaryError, public_result
from src.tools.grasp.cgn_client import ContactGraspNetClient, CGNResponseError, CGNServiceError
from src.core.errors import add_result_error, safe_error_message, structured_error
from src.tools.grasp.backend import GraspBackend
from test_backend import setup, candidates
from test_cgn_client import rgbd
from test_orchestrator import Backend, Factory, DelegationOrchestrator, a, pointer, request


@pytest.mark.parametrize('failure,category,code', [
    (CGNServiceError('connection failed'), 'service', 'cgn_service_unavailable'),
    (CGNResponseError('malformed arrays'), 'service', 'cgn_invalid_response'),
    (ValueError('height requires exactly reference and value_m'), 'input', 'invalid_or_unavailable_input'),
    (TypeError('unexpected keyword argument'), 'contract', 'backend_contract_error'),
    (BoundaryError('grasp target is stale'), 'state', 'stale_or_unowned_reference'),
    (ActionPreconditionError('grasp_requires_release'), 'state', 'grasp_requires_release'),
    (MotionPlanningError('no usable route'), 'planning', 'motion_plan_rejected'),
])
def test_typed_reason_does_not_guess_physical_failure(failure, category, code):
    detail = structured_error(failure, phase='tool_dispatch', tool='sample')
    assert detail['category'] == category and detail['code'] == code
    assert detail['retry']['same_request'] is False
    assert detail['message'] and detail['exception_type'] == type(failure).__name__


def test_motion_error_after_execution_is_not_preflight_infeasibility():
    detail = structured_error(MotionPlanningError('invalid gripper proprioception'),
                              phase='motion_execution', tool='execute_grasp')
    assert detail['category'] == 'execution'
    assert detail['code'] == 'motion_execution_failed'


def test_public_exception_text_redacts_urls_credentials_and_limits_size():
    value = ('https://user:password@example.test/plan?token=private '
             'api_key=secretvalue Authorization: Bearer hiddenvalue password="othersecret" '
             'access_token: anothersecret\n' + 'x'*600)
    clean = safe_error_message(value)
    assert len(clean) <= 400 and '\n' not in clean
    for secret in ('user:password', 'example.test', 'private', 'secretvalue', 'hiddenvalue', 'othersecret', 'anothersecret'):
        assert secret not in clean


@pytest.mark.parametrize('exc,code,status', [
    (urllib.error.URLError(ConnectionRefusedError()), 'cgn_service_unavailable', None),
    (urllib.error.URLError(TimeoutError()), 'cgn_service_timeout', None),
    (urllib.error.HTTPError('http://secret/plan', 500, 'private server body', {}, None), 'cgn_http_error', 500),
])
def test_cgn_http_failures_expose_code_not_server_body(monkeypatch, exc, code, status):
    def fail(*args, **kwargs):
        raise exc
    monkeypatch.setattr('urllib.request.urlopen', fail)
    with pytest.raises(CGNServiceError) as captured:
        ContactGraspNetClient().plan(*rgbd())
    detail = structured_error(captured.value, phase='cgn_generation')
    assert detail['code'] == code and detail.get('http_status') == status
    assert 'secret' not in json.dumps(detail) and 'private server body' not in json.dumps(detail)


@pytest.mark.parametrize('failure,reason', [('service', 'cgn_service_unavailable'),
                                         ('malformed', 'cgn_invalid_response'),
                                         ('empty', 'no_cgn_proposals')])
def test_cgn_generation_result_is_not_mislabeled_as_planning_failure(setup, failure, reason):
    backend, client, _, _ = setup
    if failure == 'service':
        client.error = CGNServiceError('service unavailable')
    elif failure == 'malformed':
        client.error = CGNResponseError('bad shape')
    else:
        from src.tools.grasp.cgn_client import RawCGNGrasps, ROBOT_BASE
        client.result = RawCGNGrasps([], [], [], ROBOT_BASE)
    output = candidates(backend)
    assert output['reason_code'] == reason
    assert output['error_details']['code'] == reason
    assert output['error_details']['category'] != 'planning'
    assert not output['candidates'] and backend.grasp_reserved == 0


def test_empty_grasp_and_unknown_execution_cause_stay_distinct():
    public = {'execution_ref': 'exec_1', 'status': 'failed'}
    unknown = add_result_error(dict(public), phase='grasp_execution')
    assert unknown['error_details']['category'] == 'execution'
    observed = GraspBackend._public_grasp_feedback(None, dict(public), {
        'held_state': 'not_held', 'grasp_evidence': 'empty_closed_gripper', 'private_state': 'never expose'})
    assert observed['error_details']['category'] == 'physical'
    assert observed['error_details']['code'] == 'empty_grasp_observed'
    assert 'private_state' not in observed
    filtered = public_result('execute_grasp', observed)
    assert filtered['error_details'] == observed['error_details']
    assert public_operation_result(filtered)['error_details'] == observed['error_details']
    provisional = GraspBackend._public_grasp_feedback(None, {'status': 'succeeded'}, {
        'held_state': 'unknown', 'grasp_evidence': 'provisional'})
    assert 'error_details' not in provisional


def runner(backend, grasp_actions):
    factory = Factory(prime=[[a('delegate_point', instruction='object'),
        a('delegate_grasp', instruction='pick object', point_ref='p1'), a('finish', status='unknown')]],
        point=[pointer()], grasp=[grasp_actions], refiner=[])
    value = DelegationOrchestrator(backend, factory, auto_refine_routes=False)
    value._overhead_done = True
    return value, factory


def test_backend_value_error_reaches_agent_with_reason_and_retains_legacy_keys():
    backend = Backend()
    def invalid(**args):
        raise ValueError('requested height reference is unavailable: grasp')
    backend.explicit_grasp_candidates = invalid
    value, _ = runner(backend, [a('explicit_grasp_candidates', **request()),
                               a('finish', status='failed', reason='invalid input')])
    result = value.run('pick')
    output = next(event['result'] for event in result.events
                  if event['kind'] == 'tool_result' and event.get('tool') == 'explicit_grasp_candidates')
    assert output['error'] == 'tool_operation_failed'
    assert output['exception_type'] == 'ValueError'
    assert output['error_details']['category'] == 'input'
    assert 'height reference' in output['error_details']['message']


def test_stale_role_reference_rejection_is_logged_and_delivered():
    args = request()
    args['point_ref'] = 'old_selection'
    value, factory = runner(Backend(), [a('explicit_grasp_candidates', **args),
                                     a('finish', status='failed', reason='refresh required')])
    result = value.run('pick')
    rejected = next(event for event in result.events if event['kind'] == 'action_rejected')
    assert rejected['error_details']['category'] == 'state'
    session = next(session for role, _, session in factory.sessions if role == 'grasp')
    delivered = [message['content'] for message in session.inputs[-1][0]
                 if message.get('role') == 'tool']
    assert any(content.get('error_details', {}).get('category') == 'state' for content in delivered)


def test_tool_input_type_and_backend_type_errors_are_distinct():
    assert structured_error(TypeError('expected number'), phase='tool_validation')['category'] == 'input'
    assert structured_error(TypeError('unexpected keyword'), phase='tool_dispatch')['category'] == 'contract'


def test_optional_reason_code_and_budget_only_results_remain_safe():
    assert add_result_error({'accepted': False, 'reason_code': None}, phase='test')['error_details']['category'] == 'planning'
    assert add_result_error({'reason_code': 'grasp_budget_exhausted'}, phase='test')['error_details']['category'] == 'budget'


def test_refiner_validation_failure_keeps_actual_input_reason():
    detail = structured_error(ValueError('angles must contain finite numbers'),
                              phase='pose_adjustment', tool='adjust_grasp')
    backend = Backend()
    backend.adjust_grasp = lambda **kwargs: dict(accepted=False,
        reason_code='adjustment_budget_exhausted', error_details=detail)
    value, _ = runner(backend, [])
    output = value._dispatch('refiner', 'adjust_grasp', {'candidate_ref': 'c1'},
                             {'candidates': {'c1': value._epoch}}, {}, 'test-refiner')
    assert output['error_details']['category'] == 'input'
    assert output['error_details']['message'] == 'angles must contain finite numbers'
