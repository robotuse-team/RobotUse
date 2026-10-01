"""Clicked contacts must cross the real RobotUse dispatch boundary, not only schemas."""
import pytest

from src.backend.controller import BoundaryError
from src.backend.delegation import DelegationOrchestrator
from src.backend.orchestrator import AgentOrchestrator
from test_prime_delegation import Factory
from test_orchestrator import Backend, height


def request(**changes):
    return dict(dict(point_ref='p1', direction='clicked', tolerance_deg=None,
                     azimuth_deg=None, polar_deg=None,
                     geometric_height=dict(reference='clicked_point', value_m=-.015),
                     transit=dict(pre=height(.4), post=height(.55))), **changes)


def dispatch(runner, args, *, epoch=None):
    runner._point_refs['p1'] = runner._epoch if epoch is None else epoch
    scope = {'candidates': {}}
    tool = 'grasp_candidates' if isinstance(runner, AgentOrchestrator) else 'explicit_grasp_candidates'
    result = runner._dispatch('grasp', tool, args, scope,
                              dict(point_ref='p1', instruction='pinch the selected rim'), 'grasp')
    return result, scope


def test_clicked_public_dispatch_reaches_backend_and_registers_candidate():
    runner = AgentOrchestrator(Backend(), Factory())
    args = request()
    result, scope = dispatch(runner, args)
    assert next(value for tool, value in runner.backend.calls
                if tool == 'explicit_grasp_candidates') == args
    assert result['candidates'][0]['candidate_ref'] == 'c1'
    assert scope['candidates']['c1'] == runner._epoch
    assert not any(tool == 'execute_grasp' for tool, _ in runner.backend.calls)


@pytest.mark.parametrize('changes', [
    {'geometric_height': None}, {'tolerance_deg': 10.},
    {'azimuth_deg': 0.}, {'polar_deg': 0.},
])
def test_clicked_dispatch_keeps_geometric_height_and_null_filter_contract(changes):
    runner = AgentOrchestrator(Backend(), Factory())
    with pytest.raises(BoundaryError):
        dispatch(runner, request(**changes))
    assert not runner.backend.calls


def test_clicked_dispatch_preserves_current_target_guard():
    runner = AgentOrchestrator(Backend(), Factory())
    with pytest.raises(BoundaryError, match='stale'):
        dispatch(runner, request(), epoch=runner._epoch - 1)
    assert not runner.backend.calls


def test_base_dispatch_rejects_clicked_contacts():
    runner = DelegationOrchestrator(Backend(), Factory())
    with pytest.raises(BoundaryError, match='direction must be one of'):
        dispatch(runner, request())
    assert not runner.backend.calls
