"""CPU-only tests of the isolated two-layer delegation contract."""
import json
from copy import deepcopy

import pytest

from src.backend.controller import (
    Action, BoundaryError, Budgets, PrimeOrchestrator, public_result,
)


class FakeBackend:
    def __init__(self):
        self.calls = []
        self.points = 0
        self.executions = 0
        self.accepted = True

    def observe(self):
        self.calls.append(('observe', {}))
        return {'observation_id': 'o', 'views': [{'view_id': 'front', 'image_ref': 'rgb',
                'hidden_geometry': 'SECRET'}], 'bddl': 'SECRET'}

    def point(self, **kwargs):
        self.calls.append(('point', kwargs))
        self.points += 1
        return {'point_ref': f'p{self.points}', 'observation_id': kwargs['observation_id'],
                'image_refs': ['mask'], 'geometry': {'xyz': 'SECRET'}}

    def grasp_candidates(self, **kwargs):
        self.calls.append(('grasp_candidates', kwargs))
        return {'candidates': [{'candidate_ref': 'c1', 'image_refs': ['preview1'],
                                'pose': 'SECRET'}, {'candidate_ref': 'c2'}],
                'bddl': 'SECRET'}

    def validate_grasp(self, **kwargs):
        self.calls.append(('validate_grasp', kwargs))
        return {'candidate_ref': kwargs['candidate_ref'], 'validation_ref': 'v',
                'accepted': self.accepted, 'error': {'text': 'SECRET'}}

    def execute_grasp(self, **kwargs):
        self.calls.append(('execute_grasp', kwargs))
        self.executions += 1
        return {'execution_ref': f'e{self.executions}',
                'status': 'failed' if self.executions == 1 else 'succeeded',
                'private': 'SECRET'}

    def place(self, **kwargs):
        self.calls.append(('place', kwargs))
        return {'execution_ref': 'placement', 'status': 'succeeded'}


class ScriptedSession:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.inputs = []
        self.closed = False

    def next_action(self, messages, tools):
        self.inputs.append((deepcopy(messages), tuple(tools)))
        return next(self.actions)

    def close(self):
        self.closed = True


class Factory:
    def __init__(self, **scripts):
        self.scripts = {role: iter(values) for role, values in scripts.items()}
        self.sessions = []

    def new_session(self, role, session_id):
        session = ScriptedSession(next(self.scripts[role]))
        self.sessions.append((role, session_id, session))
        return session


def a(tool, **arguments):
    return Action(tool, arguments)


def point_script(ref='p1'):
    return [a('observe'), a('point', observation_id='o', view_id='front', u=500, v=250),
            a('finish', point_ref=ref)]


def grasp_script(point_ref='p1', execution_ref='e1'):
    return [a('grasp_candidates', point_ref=point_ref), a('validate_grasp', candidate_ref='c2'),
            a('execute_grasp', candidate_ref='c2', validation_ref='v'),
            a('observe'), a('finish', execution_ref=execution_ref)]


def test_dynamic_point_grasp_failure_point_grasp_success_and_isolation():
    backend = FakeBackend()
    factory = Factory(prime=[[
        a('delegate_point', instruction='pick first'),
        a('delegate_grasp', instruction='try first', point_ref='p1'),
        a('delegate_point', instruction='pick another'),
        a('delegate_grasp', instruction='retry', point_ref='p2'),
        a('finish', status='completed'),
    ]], point=[point_script('p1'), point_script('p2')],
        grasp=[grasp_script(), grasp_script('p2', 'e2')])
    result = PrimeOrchestrator(backend, factory).run('move visible object')
    assert result.status == 'completed'
    assert result.result == {'status': 'completed', 'verified': False}
    assert result.delegations == 4
    assert len(set(sid for _, sid, _ in factory.sessions)) == 5
    assert [role for role, _, _ in factory.sessions] == ['prime', 'point', 'grasp', 'point', 'grasp']
    for role, _, session in factory.sessions:
        assert session.closed
        if role != 'prime':
            assert len(session.inputs[0][0]) == 2
            assert 'delegate_point' not in session.inputs[0][1]
            assert 'delegate_grasp' not in session.inputs[0][1]
    second_point = factory.sessions[3][2].inputs[0][0]
    assert second_point[1]['content'] == {'instruction': 'pick another'}
    assert 'try first' not in json.dumps(second_point)
    serialized = json.dumps(result.events) + json.dumps([s.inputs for _, _, s in factory.sessions])
    assert 'SECRET' not in serialized
    child_results = [e['result']['result'] for e in result.events
                     if e['kind'] == 'tool_result' and e['tool'] == 'delegate_grasp']
    assert [r['status'] for r in child_results] == ['failed', 'succeeded']
    assert [e['seq'] for e in result.events] == list(range(len(result.events)))


def test_point_iterates_pixels_and_accepts_only_issued_reference():
    backend = FakeBackend()
    factory = Factory(prime=[[a('delegate_point', instruction='target'), a('finish', status='unknown')]],
        point=[[a('finish', point_ref='invented'), a('observe'),
                a('point', observation_id='o', view_id='front', u=0, v=1000),
                a('point', observation_id='o', view_id='front', u=1000, v=0),
                a('finish', point_ref='p2')]])
    result = PrimeOrchestrator(backend, factory).run('task')
    assert backend.points == 2
    assert any(e['kind'] == 'action_rejected' for e in result.events)
    assert any(e.get('result') == {'point_ref': 'p2'} for e in result.events)
    point_session = factory.sessions[1][2]
    assert any(m.get('content', {}).get('image_refs') == ['mask']
               for m in point_session.inputs[-1][0] if isinstance(m.get('content'), dict))


@pytest.mark.parametrize('u,v,view', [(True, 1, 'front'), (-1, 1, 'front'),
    (1001, 1, 'front'), (float('nan'), 1, 'front'), (0, float('inf'), 'front'),
    (0, 0, 'hidden')])
def test_rejects_invalid_pixels_and_unknown_views(u, v, view):
    backend = FakeBackend()
    factory = Factory(prime=[[a('delegate_point', instruction='target'), a('finish', status='unknown')]],
        point=[[a('observe'), a('point', observation_id='o', view_id=view, u=u, v=v)]])
    PrimeOrchestrator(backend, factory).run('task')
    assert backend.points == 0


def test_validation_gate_reselect_and_single_use_with_staleness():
    backend = FakeBackend()
    factory = Factory(prime=[[a('delegate_point', instruction='target'),
        a('delegate_grasp', instruction='grasp', point_ref='p1'),
        a('place', point_ref='p1'), a('delegate_grasp', instruction='stale', point_ref='p1'),
        a('finish', status='unknown')]], point=[point_script()], grasp=[[
        a('grasp_candidates', point_ref='p1'),
        a('execute_grasp', candidate_ref='c1', validation_ref='invented'),
        a('validate_grasp', candidate_ref='c1'),
        a('execute_grasp', candidate_ref='c2', validation_ref='v'),
        a('validate_grasp', candidate_ref='c2'),
        a('execute_grasp', candidate_ref='c2', validation_ref='v'),
        a('execute_grasp', candidate_ref='c2', validation_ref='v'),
        a('observe'), a('finish', execution_ref='e1')]])
    result = PrimeOrchestrator(backend, factory).run('task')
    assert backend.executions == 1
    assert not any(tool == 'place' for tool, _ in backend.calls)
    assert result.delegations == 2
    assert sum(e['kind'] == 'action_rejected' for e in result.events) == 5


def test_rejected_validation_cannot_execute():
    backend = FakeBackend()
    backend.accepted = False
    factory = Factory(prime=[[a('delegate_point', instruction='target'),
        a('delegate_grasp', instruction='grasp', point_ref='p1'), a('finish', status='unknown')]],
        point=[point_script()], grasp=[grasp_script()])
    PrimeOrchestrator(backend, factory).run('task')
    assert backend.executions == 0


def test_place_invalidates_prime_observations_and_point_refs():
    backend = FakeBackend()
    factory = Factory(prime=[[a('delegate_point', instruction='destination'),
        a('place', point_ref='p1'), a('place', point_ref='p1'), a('finish', status='unknown')]],
        point=[point_script()])
    PrimeOrchestrator(backend, factory).run('task')
    assert sum(tool == 'place' for tool, _ in backend.calls) == 1


def test_backend_errors_never_leak_exception_text():
    backend = FakeBackend()
    def broken():
        raise RuntimeError('SECRET BDDL hidden geometry')
    backend.observe = broken
    factory = Factory(prime=[[a('observe'), a('finish', status='unknown')]])
    result = PrimeOrchestrator(backend, factory).run('task')
    assert 'SECRET' not in json.dumps(result.events)
    assert 'SECRET' not in json.dumps(factory.sessions[0][2].inputs)
    assert any(e['kind'] == 'backend_error' for e in result.events)


def test_fresh_session_identity_enforced():
    session = ScriptedSession([a('delegate_point', instruction='target'), a('finish', status='unknown')])
    class ReusingFactory:
        def new_session(self, role, session_id):
            return session
    result = PrimeOrchestrator(FakeBackend(), ReusingFactory()).run('task')
    assert any(e['kind'] == 'session_rejected' for e in result.events)
    assert session.closed


@pytest.mark.parametrize('budgets', [Budgets(prime_steps=1), Budgets(max_tool_calls=1),
                                     Budgets(max_delegations=1), Budgets(child_steps=1)])
def test_budgets_bound_work(budgets):
    factory = Factory(prime=[[a('delegate_point', instruction='one'),
        a('delegate_point', instruction='two'), a('observe'), a('observe')]],
        point=[point_script(), point_script('p2')])
    result = PrimeOrchestrator(FakeBackend(), factory, budgets=budgets).run('task')
    assert result.tool_calls <= budgets.max_tool_calls
    assert result.delegations <= budgets.max_delegations
    assert all(len(s.inputs) <= (budgets.prime_steps if role == 'prime' else budgets.child_steps)
               for role, _, s in factory.sessions)
    assert all(s.closed for _, _, s in factory.sessions)


@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_budget_validation(value):
    with pytest.raises(ValueError):
        Budgets(child_steps=value)


def test_projection_rejects_missing_or_malformed_fields():
    with pytest.raises(BoundaryError):
        public_result('observe', {'image_refs': ['rgb']})
    with pytest.raises(BoundaryError):
        public_result('validate_grasp', {'candidate_ref': 'c', 'validation_ref': 'v', 'accepted': 'yes'})
    with pytest.raises(BoundaryError):
        public_result('grasp_candidates', {'candidates': [{'candidate_ref': 'c'}, {'candidate_ref': 'c'}]})


def test_audit_is_detached_and_single_use():
    sink_events = []
    def sink(event):
        sink_events.append(deepcopy(event))
        event['kind'] = 'tampered'
    runner = PrimeOrchestrator(FakeBackend(), Factory(prime=[[a('finish', status='completed')]]), audit_sink=sink)
    result = runner.run('task')
    assert list(result.events) == sink_events
    with pytest.raises(RuntimeError, match='single-use'):
        runner.run('again')


def test_audit_failure_prevents_backend_execution():
    backend = FakeBackend()
    def sink(event):
        if event['kind'] == 'tool_called':
            raise RuntimeError('audit unavailable')
    factory = Factory(prime=[[a('observe')]])
    with pytest.raises(RuntimeError, match='audit unavailable'):
        PrimeOrchestrator(backend, factory, audit_sink=sink).run('task')
    assert backend.calls == []
    assert factory.sessions[0][2].closed


def test_grasp_can_explicitly_request_repoint_without_execution():
    backend = FakeBackend()
    factory = Factory(prime=[[a('delegate_point', instruction='target'),
        a('delegate_grasp', instruction='grasp', point_ref='p1'),
        a('delegate_point', instruction='correct target'), a('finish', status='unknown')]],
        point=[point_script(), point_script('p2')], grasp=[[
            a('grasp_candidates', point_ref='p1'),
            a('finish', status='needs_point', reason='wrong_target')]])
    result = PrimeOrchestrator(backend, factory).run('task')
    assert backend.executions == 0
    assert result.delegations == 3
    assert any(e.get('result') == {'status': 'needs_point', 'reason': 'wrong_target'}
               for e in result.events)


def test_point_can_fail_without_inventing_target():
    factory = Factory(prime=[[a('delegate_point', instruction='target'), a('finish', status='unknown')]],
        point=[[a('observe'), a('finish', status='failed', reason='no_valid_mask')]])
    result = PrimeOrchestrator(FakeBackend(), factory).run('task')
    assert any(e.get('result') == {'status': 'failed', 'reason': 'no_valid_mask'} for e in result.events)


def test_malformed_arguments_and_role_escalation_are_rejected():
    backend = FakeBackend()
    factory = Factory(prime=[[Action('observe', 'not a mapping'),
        a('execute_grasp', candidate_ref='c', validation_ref='v'),
        a('delegate_point', instruction='target'), a('finish', status='unknown')]],
        point=[[a('delegate_grasp', instruction='escalate', point_ref='x'),
                a('finish', status='failed', reason='no_valid_mask')]])
    result = PrimeOrchestrator(backend, factory).run('task')
    assert sum(e['kind'] == 'action_rejected' for e in result.events) == 3
    assert backend.calls == []


def test_factory_errors_are_sanitized_and_audited():
    class BrokenFactory:
        def new_session(self, role, session_id):
            raise RuntimeError('SECRET hidden provider state')
    result = PrimeOrchestrator(FakeBackend(), BrokenFactory()).run('task')
    assert result.status == 'error'
    assert result.result == {'error': 'factory_error'}
    assert 'SECRET' not in json.dumps(result.events)
    assert result.events[-1]['kind'] == 'session_closed'


def test_model_mutating_messages_does_not_mutate_history_or_audit():
    class Mutator(ScriptedSession):
        def next_action(self, messages, tools):
            action = super().next_action(messages, tools)
            messages[1]['content']['instruction'] = 'tampered'
            return action
    session = Mutator([a('observe'), a('finish', status='unknown')])
    class MutatingFactory:
        def new_session(self, role, session_id):
            return session
    PrimeOrchestrator(FakeBackend(), MutatingFactory()).run('original')
    assert session.inputs[-1][0][1]['content'] == {'instruction': 'original'}


def test_prime_branches_on_child_result_instead_of_fixed_sequence():
    class AdaptivePrime(ScriptedSession):
        def next_action(self, messages, tools):
            self.inputs.append((deepcopy(messages), tuple(tools)))
            results = [m['content'] for m in messages if m['role'] == 'tool']
            if not results:
                return a('delegate_point', instruction='first target')
            latest = results[-1]['result']
            if latest.get('point_ref'):
                return a('delegate_grasp', instruction='grasp', point_ref=latest['point_ref'])
            if latest.get('status') == 'needs_point':
                return a('delegate_point', instruction='correct target')
            return a('finish', status='completed')
    class AdaptiveFactory(Factory):
        def new_session(self, role, session_id):
            if role != 'prime':
                return super().new_session(role, session_id)
            session = AdaptivePrime([])
            self.sessions.append((role, session_id, session))
            return session
    backend = FakeBackend()
    backend.executions = 1
    factory = AdaptiveFactory(point=[point_script(), point_script('p2')], grasp=[[
        a('finish', status='needs_point', reason='wrong_target')], grasp_script('p2', 'e2')])
    result = PrimeOrchestrator(backend, factory).run('task')
    assert result.status == 'completed'
    assert [role for role, _, _ in factory.sessions] == ['prime', 'point', 'grasp', 'point', 'grasp']
    assert backend.executions == 2


@pytest.mark.parametrize('error', [TypeError, BoundaryError])
def test_audit_boundary_typed_failure_is_still_fatal(error):
    backend = FakeBackend()
    def sink(event):
        if event['kind'] == 'tool_called':
            raise error('SECRET sink error')
    factory = Factory(prime=[[a('observe'), a('observe')]])
    with pytest.raises(RuntimeError, match='audit unavailable'):
        PrimeOrchestrator(backend, factory, audit_sink=sink).run('task')
    assert backend.calls == []
    assert len(factory.sessions[0][2].inputs) == 1


def test_grasp_finish_requires_observation_after_execution_not_before():
    backend = FakeBackend()
    backend.executions = 1
    factory = Factory(prime=[[a('delegate_point', instruction='target'),
        a('delegate_grasp', instruction='grasp', point_ref='p1'), a('finish', status='unknown')]],
        point=[point_script()], grasp=[[
            a('observe'), a('grasp_candidates', point_ref='p1'),
            a('validate_grasp', candidate_ref='c2'),
            a('execute_grasp', candidate_ref='c2', validation_ref='v'),
            a('finish', execution_ref='e2'),  # old RGB cannot establish post-motion evidence
            a('observe'), a('finish', execution_ref='e2')]])
    result = PrimeOrchestrator(backend, factory).run('task')
    assert sum(e['kind'] == 'action_rejected' for e in result.events) == 1
    finished = [e['result'] for e in result.events if e['kind'] == 'finished' and e['role'] == 'grasp']
    assert finished == [{'execution_ref': 'e2', 'status': 'succeeded',
                         'observation_id': 'o', 'verified': False}]


def test_grasp_visual_empty_grip_can_request_new_point_after_command_success():
    backend = FakeBackend()
    backend.executions = 1
    factory = Factory(prime=[[a('delegate_point', instruction='target'),
        a('delegate_grasp', instruction='grasp', point_ref='p1'),
        a('delegate_point', instruction='retry empty grasp'), a('finish', status='unknown')]],
        point=[point_script(), point_script('p2')], grasp=[[
            a('grasp_candidates', point_ref='p1'), a('validate_grasp', candidate_ref='c2'),
            a('execute_grasp', candidate_ref='c2', validation_ref='v'), a('observe'),
            a('finish', status='needs_point', reason='empty_grip')]])
    result = PrimeOrchestrator(backend, factory).run('task')
    assert result.delegations == 3
    assert any(e.get('result') == {'status': 'needs_point', 'reason': 'empty_grip'}
               for e in result.events)


def test_failed_post_execution_observation_does_not_unlock_finish():
    backend = FakeBackend()
    backend.executions = 1
    original_observe = backend.observe
    def observe():
        if backend.executions > 1:
            raise RuntimeError('camera failed')
        return original_observe()
    backend.observe = observe
    factory = Factory(prime=[[a('delegate_point', instruction='target'),
        a('delegate_grasp', instruction='grasp', point_ref='p1'), a('finish', status='unknown')]],
        point=[point_script()], grasp=[grasp_script('p1', 'e2') + [
            a('finish', status='needs_point', reason='needs_reobserve')]])
    result = PrimeOrchestrator(backend, factory).run('task')
    assert not any(e['kind'] == 'finished' and 'execution_ref' in e['result'] for e in result.events)
    assert any(e.get('result') == {'status': 'needs_point', 'reason': 'needs_reobserve'}
               for e in result.events)
