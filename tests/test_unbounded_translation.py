"""RobotUse edit sizes are unrestricted; legacy bounds and execution checks remain."""
import json

import numpy as np
import pytest

from src.backend.controller import BoundaryError
from src.tools.pose_editor.geometry import checked_translation
from src.backend.orchestrator import AgentOrchestrator
from test_prime_delegation import Factory
from test_backend import setup, candidates, paused
from test_place_motion import Backend as PlaceBackend, prepare
from test_provider import send, functions
from test_pose_editor import enable, deltas, EditorAgentBackend


@pytest.mark.parametrize('native', [True, False])
@pytest.mark.parametrize('tool', ['adjust_grasp', 'nudge_grasp', 'adjust_place'])
def test_large_all_axis_edits_survive_provider_and_runtime(tmp_path, monkeypatch, native, tool):
    runner = AgentOrchestrator(EditorAgentBackend(), Factory())
    args = deltas(dx_mm=600., dy_mm=-500., dz_mm=400., roll_deg=180., pitch_deg=-450., yaw_deg=1080.)
    if tool != 'nudge_grasp':
        args['candidate_ref'] = 'current'
    body = send(tmp_path, monkeypatch, role='refiner', tools=(tool,), native=native,
        action=dict(tool=tool, arguments=args), tool_schema_aliases=runner.factory.tool_schema_aliases,
        unrestricted_pose_rotation=True, unrestricted_pose_translation=True,
        clicked_grasp_candidates=True)
    if native:
        props = functions(body)[tool]['parameters']['properties']
    else:
        raw = body['messages'][0]['content'].split(' Tool argument types: ', 1)[1]
        props = json.JSONDecoder().raw_decode(raw)[0][tool]
    for key, value in args.items():
        assert runner._tool_argument(tool, key, value) == value
        if key != 'candidate_ref':
            assert 'minimum' not in props[key] and 'maximum' not in props[key]
    for value in (float('nan'), float('inf'), True, '100'):
        with pytest.raises(BoundaryError):
            runner._tool_argument(tool, 'dx_mm', value)


def test_unbounded_totals_and_legacy_defaults():
    assert checked_translation((500., -600., 700.), (900., 0., 0.), unrestricted=True)[1] == (1400., -600., 700.)
    with pytest.raises(ValueError):
        checked_translation((31., 0., 0.))
    for step, applied in [((float('inf'), 0., 0.), (0., 0., 0.)),
                          ((1e308, 0., 0.), (1e308, 0., 0.)),
                          ((True, 0., 0.), (0., 0., 0.))]:
        with pytest.raises(ValueError):
            checked_translation(step, applied, unrestricted=True)


def test_large_grasp_translation_and_rotation_preserve_checks(setup):
    backend = enable(setup[0])
    checker = setup[2]
    first = candidates(backend, direction='median')['candidates'][0]
    edited = backend.refine_candidate(first['candidate_ref'], **deltas(dx_mm=60., pitch_deg=90.))
    assert edited['accepted']
    np.testing.assert_allclose(np.array(edited['contact_center_xyz_m']) - first['contact_center_xyz_m'], [.06, 0, 0])
    second = backend.refine_candidate(edited['candidate_ref'], **deltas(dx_mm=150., yaw_deg=720.))
    assert second['accepted']
    assert second['pose_editor']['cumulative_from_original']['dx_mm'] == 210.
    checker.accepted = False
    rejected = backend.refine_candidate(second['candidate_ref'], **deltas(dx_mm=500.))
    assert not rejected['accepted'] and not rejected['executable']
    assert rejected['candidate_ref'] not in backend.candidate_routes
    assert backend.validate_grasp(rejected['candidate_ref'])['accepted'] is False
    assert not backend.connector.events


def test_large_paused_translation_keeps_pending_pose_when_rejected(setup):
    backend, checker, original, _ = paused(setup)
    backend.unrestricted_pose_translation = backend.unrestricted_pose_rotation = True
    accepted = backend.nudge_inflight_grasp(**deltas(dx_mm=160., pitch_deg=90.))
    assert accepted['accepted']
    assert backend._inflight['translation'] == (160., 0., 0.)
    current = backend._inflight['current_pose'].copy()
    checker.accepted = False
    rejected = backend.nudge_inflight_grasp(**deltas(dx_mm=200.))
    assert not rejected['accepted']
    np.testing.assert_array_equal(current, backend._inflight['current_pose'])
    assert backend._inflight['translation'] == (160., 0., 0.)
    assert not backend.connector.events


def test_large_place_translation_uses_same_validation_and_fixed_transit():
    backend = PlaceBackend()
    backend.unrestricted_pose_translation = backend.unrestricted_pose_rotation = True
    first = prepare(backend)
    revised = backend.explicit_adjust_place(first['candidate_ref'], **deltas(dx_mm=160., yaw_deg=180.))
    assert revised['accepted'] and revised['transit_z_m'] == first['transit_z_m']
    np.testing.assert_allclose(np.array(revised['contact_center_xyz_m']) - first['contact_center_xyz_m'], [.16, 0, 0], atol=1e-12)
    backend.reject = True
    rejected = backend.explicit_adjust_place(revised['candidate_ref'], **deltas(dx_mm=200.))
    assert not rejected['accepted'] and not backend.connector.events


def test_role_prompts_and_metadata_have_no_old_translation_budget():
    runner = AgentOrchestrator(EditorAgentBackend(), Factory())
    for role in ('prime', 'grasp', 'place', 'refiner'):
        prompt = runner._prompt(role)
        for obsolete in ('30 per step', '90 cumulative', '30 per axis', 'with bounded base XYZ', 'retain translation bounds'):
            assert obsolete not in prompt
    features = runner.interaction_features.metadata()
    assert features['translation_step_limit_mm'] is features['translation_cumulative_limit_mm'] is None
    assert features['grasp_candidate_translation_cumulative_limit_mm'] is None
