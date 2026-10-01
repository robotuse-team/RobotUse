"""Measured contact seeds use existing candidate budgets, validation and review."""
from copy import deepcopy
import json

import numpy as np
import pytest

from src.backend.controller import BoundaryError
from src.backend.orchestrator import AgentOrchestrator
from test_prime_delegation import Factory
from test_backend import setup, OBJECT, candidates, TRANSIT
from test_provider import send, functions
from test_pose_editor import enable, deltas, EditorAgentBackend


def request(**changes):
    return dict(dict(point_ref='point', direction='clicked', tolerance_deg=None,
        azimuth_deg=None, polar_deg=None, geometric_height=dict(reference='clicked_point', value_m=0.),
        transit=deepcopy(TRANSIT)), **changes)


def test_edge_seed_keeps_measured_xy_instead_of_recentering_and_plans_without_cgn(setup):
    backend, client, checker, geometry = setup
    enable(backend)
    backend.clicked_points['point'] = OBJECT[-1].copy()
    result = backend.explicit_grasp_candidates(**request())
    assert len(result['candidates']) == result['candidate_limit'] == 1
    assert not result['diagnostic_candidates'] and result['generator'] == 'clicked'
    entry = result['candidates'][0]
    np.testing.assert_allclose(entry['contact_center_xyz_m'], OBJECT[-1])
    # Use a target offset from the centroid to exercise unrestricted contact selection.
    assert np.linalg.norm(OBJECT[-1] - np.median(OBJECT, axis=0)) > .06
    assert entry['source'] == 'clicked' and entry['yaw_deg'] == 0.
    assert backend.grasp_reserved == 1 and len(checker.calls) == 1
    assert backend.validate_grasp(entry['candidate_ref'])['accepted']
    revised = backend.refine_candidate(entry['candidate_ref'], **deltas(pitch_deg=90., dy_mm=80.))
    assert revised['accepted'] and revised['source'] == 'clicked'
    assert not revised['top_down_only']
    assert not client.requests and not backend.connector.events


@pytest.mark.parametrize('failure', ['depth', 'background', 'stale', 'budget'])
def test_unavailable_click_never_falls_back_to_center_or_calls_model(setup, failure):
    backend, client, checker, _ = setup
    enable(backend)
    if failure == 'depth':
        backend.clicked_points.pop('point')
    elif failure == 'background':
        backend.clicked_points['point'] = np.array([1., 1., 1.])
    else:
        backend.clicked_points['point'] = OBJECT[-1].copy()
    if failure == 'stale':
        backend.epoch += 1
    if failure == 'budget':
        backend.grasp_reserved = backend.task_grasp_budget
        assert backend.explicit_grasp_candidates(**request())['reason_code'] == 'grasp_budget_exhausted'
    else:
        with pytest.raises(ValueError):
            backend.explicit_grasp_candidates(**request())
    assert not client.requests and not checker.calls and not backend.connector.events


def test_one_remaining_slot_and_rejected_path_publish_only_nonexecutable_diagnostic(setup):
    backend, client, checker, _ = setup
    enable(backend)
    backend.clicked_points['point'] = OBJECT[-1].copy()
    backend.grasp_reserved = backend.task_grasp_budget - 1
    checker.accepted = False
    result = backend.explicit_grasp_candidates(**request())
    assert not result['candidates'] and len(result['diagnostic_candidates']) == 1
    ref = result['diagnostic_candidates'][0]['candidate_ref']
    assert not backend.validate_grasp(ref)['accepted'] and ref not in backend.candidate_routes
    with pytest.raises(ValueError, match='diagnostic candidate'):
        backend.execute_grasp(ref, 'invented')
    assert backend.grasp_reserved == backend.task_grasp_budget
    assert not client.requests and not backend.connector.events


def test_cgn_failure_does_not_automatically_enable_clicked_mode(setup):
    backend, client, _, _ = setup
    enable(backend)
    client.error = ConnectionError('offline')
    backend.clicked_points['point'] = OBJECT[-1].copy()
    assert not candidates(backend)['candidates']
    assert backend.grasp_reserved == 0
    assert backend.explicit_grasp_candidates(**request())['candidates']
    assert len(client.requests) == 1


@pytest.mark.parametrize('native', [True, False])
def test_clicked_mode_is_exposed_only_by_provider(tmp_path, monkeypatch, native):
    runner = AgentOrchestrator(EditorAgentBackend(), Factory())
    action = dict(tool='grasp_candidates', arguments=request())
    body = send(tmp_path, monkeypatch, role='grasp', tools=('grasp_candidates',), action=action,
        native=native, tool_schema_aliases=runner.factory.tool_schema_aliases,
        clicked_grasp_candidates=True)
    if native:
        props = functions(body)['grasp_candidates']['parameters']['properties']
    else:
        raw = body['messages'][0]['content'].split(' Tool argument types: ', 1)[1]
        props = json.JSONDecoder().raw_decode(raw)[0]['grasp_candidates']
    assert 'clicked' in props['direction']['enum']
    assert runner._tool_argument('grasp_candidates', 'direction', 'clicked') == 'clicked'
    from src.tools.grasp.arguments import property_schema
    assert 'clicked' not in property_schema('direction', tool='explicit_grasp_candidates')['enum']


def test_base_backend_does_not_expose_clicked_generator(setup):
    with pytest.raises(ValueError):
        setup[0].explicit_grasp_candidates(**request())
