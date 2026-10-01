"""Backend integration on CPU: fake perception/model/IK, real contracts/plans.

These tests publish synthetic previews and are not live robot success evidence.
"""
from copy import deepcopy
from types import SimpleNamespace as NS
import json

import numpy as np
import pytest
from PIL import Image

from src.core.action_feedback import ActionPreconditionError
from src.tools.motion import planning as motion; from src.backend import robot_base as prime_backend; from src.backend import robot as intent_backend
from src.llm.manager import ImageRegistry
from src.tools.grasp.backend import GraspBackend, hand_to_contact
from src.tools.grasp.cgn_client import RawCGNGrasps, ROBOT_BASE, CGNResponseError, CGNServiceError
from src.tools.grasp.geometry import top_down_contact_pose
from test_graspgen_motion import Connector


OBJECT = np.array([[.38, -.02, .28], [.40, 0., .30], [.46, .03, .32]])
SCENE = np.array([[1., 1., .1], [1., 1., .3]])
HEIGHT = dict(mode='absolute', reference='none', value_m=.30)
TRANSIT = dict(pre=dict(mode='absolute', reference='none', value_m=.55),
               post=dict(mode='surface_relative', reference='max', value_m=.43))


class Client:
    def __init__(self):
        self.requests = []
        self.error = None
        poses = []
        for i in range(6):
            poses.append(top_down_contact_pose([.4+i*.002, 0.], .4034, 0))
        self.result = RawCGNGrasps(np.stack(poses), [.1, .7, .3, .9, .5, .8],
                                   np.tile([.4, 0, .3], (6, 1)), ROBOT_BASE)

    def plan_point_clouds(self, full, segment, *, input_frame):
        self.requests.append((full.copy(), segment.copy(), input_frame))
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def setup(tmp_path, monkeypatch):
    client = Client()
    connector = Connector()
    source = NS(observation_id='obs', view_id='front', role='pick',
                object_points=OBJECT[:2], scene_points=SCENE, frame=NS())
    geometry = NS(observation_id='obs', view_id='front', role='pick',
                  object_points=OBJECT, scene_points=SCENE, per_view=(source,))
    adapter = NS(latest='obs', frames={'obs': []}, _check_current=lambda obs: None)
    legacy = NS(predict=lambda *args, **kwargs: pytest.fail('legacy inference must not run'))
    backend = GraspBackend(connector=connector, point_adapter=adapter, graspgen=legacy,
        anyplace=legacy, moveit_grasps=legacy, cgn_client=client, output_dir=tmp_path,
        images=ImageRegistry(), grasp_to_ee=np.eye(4),
        gripper_assets=NS(jaw_center_offset_m=.136), max_gripper_width_m=.085,
        motion_config=motion.MotionConfig(collision_checks_enabled=False),
        task_grasp_budget=12)
    backend.latest_observation_id = 'obs'
    backend.latest_fused_ref = 'point'
    backend.points['point'] = (backend.epoch, geometry)
    backend.clicked_points['point'] = np.array([.39, -.01, .29])
    backend.scene_grippers['obs'] = {}  # captured sensor provenance; avoid native mesh read
    checker = NS(accepted=True, calls=[], clearance_m=.002)

    def check(plan, scene, **kwargs):
        checker.calls.append((plan, scene.copy(), kwargs))
        return dict(accepted=checker.accepted, kind='environment', segment='lift')

    checker.check = check
    backend.candidate_path_scenes['point'] = (checker, SCENE)
    backend._candidate_path_scene = lambda point_ref: (checker, SCENE)
    backend.events = []
    backend._record = lambda kind, value: backend.events.append((kind, deepcopy(value)))

    def render(geometry, prediction, ref, path, **kwargs):
        Image.new('RGB', (120, 80), 'white').save(path)

    def project(frames, pose, directory, **kwargs):
        assert kwargs['mesh_source'].preview_supersampling == 4
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/'preview-front.png'
        Image.new('RGB', (120, 80), 'cyan').save(path)
        return [str(path)]
    from src.tools.pose_editor import refinement as pose_refinement
    monkeypatch.setattr(pose_refinement, 'preview_refined_pose', project)
    monkeypatch.setattr(prime_backend, 'render_candidate', render)
    monkeypatch.setattr(intent_backend, 'render_candidate', render)
    return backend, client, checker, geometry


def candidates(backend, **changes):
    arguments = dict(point_ref='point', direction='vertical', tolerance_deg=30,
        azimuth_deg=None, polar_deg=None, geometric_height=deepcopy(HEIGHT),
        transit=deepcopy(TRANSIT))
    arguments.update(changes)
    if arguments['direction'] in ('mean', 'median'):
        arguments['tolerance_deg'] = None
    else:
        arguments['geometric_height'] = None
    return backend.explicit_grasp_candidates(**arguments)


def test_cgn_full_cloud_contains_segment_and_only_four_cgn_candidates_use_contact_frame(setup):
    backend, client, checker, geometry = setup
    result = candidates(backend)
    assert backend.anyplace is None and backend.moveit_grasps is None
    assert len(result['candidates']) == 4 and not result['diagnostic_candidates']
    full, segment, frame = client.requests[0]
    np.testing.assert_array_equal(full, np.concatenate([SCENE, OBJECT]))
    np.testing.assert_array_equal(segment, OBJECT)
    assert frame == ROBOT_BASE
    entries = result['candidates']
    assert [entry['source'] for entry in entries] == ['contact_graspnet']*4
    assert [entry['source_candidate_index'] for entry in entries] == [3, 5, 1, 4]
    assert [entry['score'] for entry in entries] == [.9, .8, .7, .5]
    for entry in entries:
        ref = entry['candidate_ref']
        plan = backend.candidate_routes[ref][1]
        contact = hand_to_contact(plan.grasp_transform, .136)
        assert contact[2, 3] == pytest.approx(.3)
        assert plan.grasp_transform[2, 3] == pytest.approx(.436)
        np.testing.assert_allclose(entry['contact_center_pose_base'], contact)
        np.testing.assert_allclose(plan.target_points, OBJECT)
        assert entry['transit'] == dict(pre_pick_z_m=.55, post_pick_z_m=.75)
        assert plan.grasp_contract['post_pick_z_m'] == .75
        assert backend.images.paths[entry['image_refs'][0]].is_file()
    assert all(kwargs['stop_label'] == 'lift' for _, _, kwargs in checker.calls)
    assert backend.grasp_reserved == 4 and backend.grasp_published == 4
    assert backend._active_grasp_contract is None
    assert not backend.connector.events  # generation planned but did not move


@pytest.mark.parametrize('failure', ['empty', 'direction', 'service', 'malformed', 'frame'])
def test_cgn_unavailability_never_silently_adds_geometry_alternatives(setup, failure):
    backend, client, _, _ = setup
    kwargs = {}
    if failure == 'empty':
        client.result = RawCGNGrasps([], [], [], ROBOT_BASE)
    elif failure == 'direction':
        kwargs = dict(direction='horizontal', tolerance_deg=0)
    elif failure == 'service':
        client.error = CGNServiceError('unavailable')
    elif failure == 'malformed':
        client.error = CGNResponseError('scores and poses have different lengths')
    else:
        raw = client.result
        client.result = RawCGNGrasps(raw.poses, raw.scores, raw.contact_points, 'camera_optical')
    result = candidates(backend, **kwargs)
    assert not result['candidates']
    if failure in ('service', 'malformed', 'frame'):
        assert result['source_reports'][0]['error_type']
    else:
        assert result['source_reports'][0]['direction_survivors'] == 0


def test_geometry_exposes_measured_click_centers_height_statistics_and_calibrated_current_tcp(setup):
    backend, _, _, _ = setup
    data = backend.explicit_geometry('point')
    assert data['statistics']['count'] == 3
    assert data['statistics']['max_xyz_m'][2] == .32
    assert [x['candidate_id'] for x in data['xy_candidates']] == ['clicked', 'median', 'mean']
    np.testing.assert_allclose(data['contact_center_xyz_m'], [.4, 0, .464])
    assert data['frame'] == 'connector_base' and data['position_reference'] == 'jaw_contact_center'


def test_opening_policy_uses_same_fused_target_as_plan(setup):
    backend, _, _, _ = setup
    targets = []
    backend.grasp_opening_policy = lambda pose, points: (
        targets.append(points.copy()) or dict(open_width_m=.075))
    candidates(backend)
    assert len(targets) == 4
    for target in targets:
        np.testing.assert_array_equal(target, OBJECT)


def test_diagnostic_publication_retains_source_geometry_and_refines_through_same_checks(setup):
    backend, client, checker, _ = setup
    checker.accepted = False
    result = candidates(backend, direction="median")
    assert not result['candidates'] and len(result['diagnostic_candidates']) == 4
    entry = result['diagnostic_candidates'][0]
    ref = entry['candidate_ref']
    assert backend.inspect_candidate(ref)['source'] == 'median'
    assert backend.validate_grasp(ref)['accepted'] is False
    assert ref not in backend.candidate_routes
    checker.accepted = True
    refined = backend.refine_candidate(ref, 0, 0, 5, dx_mm=3, dz_mm=2)
    assert refined['accepted'] and refined['source'] == 'median'
    new = refined['candidate_ref']
    assert new != ref and new in backend.candidate_routes
    assert backend.candidate_routes[new][1].grasp_contract['pre_pick_z_m'] == .55
    assert backend.candidate_routes[new][1].grasp_contract['post_pick_z_m'] == .75
    assert backend.grasp_contracts[new]['contact_center_xyz_m'] == refined['contact_center_xyz_m']
    np.testing.assert_allclose(np.array(refined['contact_center_xyz_m'])-entry['contact_center_xyz_m'], [.003, 0, .002])
    np.testing.assert_allclose(refined['approach_direction_base'], [0, 0, -1], atol=1e-12)
    assert backend.validate_grasp(new)['accepted']
    plan = backend.candidate_routes[new][1]
    assert plan.scene_observation_id == 'obs' and plan.scene_point_ref == 'point'
    assert len(client.requests) == 0  # refinement never invokes a model
    assert backend.grasp_reserved == 4  # four median yaw poses were checked


def test_world_planning_failure_with_existing_kind_still_publishes_diagnostics(setup):
    backend, _, _, _ = setup

    def fail(*args, **kwargs):
        raise motion.MotionPlanningError('no transit route', planning_feedback={
            'kind': 'planning', 'segment': 'high_transit', 'planner_reason_code': 'no_usable_route'})

    backend._plan_grasp_candidate = fail
    result = candidates(backend)
    assert not result['candidates'] and len(result['diagnostic_candidates']) == 4
    assert result['diagnostic_candidates'][0]['validation_feedback']['segment'] == 'high_transit'


def test_lower_score_feasible_cgn_is_not_hidden_by_four_higher_score_path_rejections(setup):
    backend, client, checker, _ = setup

    def check(plan, scene, **kwargs):
        # The highest four CGN source indices (3,5,1,4) are unreachable.
        rejected_x = [.406, .410, .402, .408]
        x = hand_to_contact(plan.grasp_transform, .136)[0, 3]
        return dict(accepted=not any(np.isclose(x, value) for value in rejected_x), kind='planning')

    checker.check = check
    result = candidates(backend)
    valid = [entry for entry in result['candidates'] if entry['source'] == 'contact_graspnet']
    assert [entry['source_candidate_index'] for entry in valid] == [2, 0]
    assert len(result['candidates']) + len(result['diagnostic_candidates']) == 4
    assert len(backend.candidates) == 4  # unshown rejections were never published
    assert backend.grasp_reserved == 6
    report = result['source_reports'][0]
    assert report['planning_checked_count'] == 6 and report['planning_accepted_count'] == 2
    assert report['search_stop_reason'] == 'direction_candidates_exhausted'
    assert report['direction_filter']['tolerance_deg'] == 30


@pytest.mark.parametrize('remaining', [0, 1, 2, 3])
def test_geometry_reserves_four_slots_without_calling_cgn(setup, remaining):
    backend, client, _, _ = setup
    backend.grasp_reserved = backend.task_grasp_budget - remaining
    result = candidates(backend, direction='median')
    assert not client.requests and not result['candidates']
    assert result['reason_code'] == 'grasp_budget_exhausted'
    assert backend.grasp_reserved == backend.task_grasp_budget - remaining


def test_cgn_search_reports_budget_limit_without_claiming_all_candidates_invalid(setup):
    backend, _, checker, _ = setup
    backend.task_grasp_budget = 5
    checker.accepted = False
    result = candidates(backend)
    assert len(result['diagnostic_candidates']) == 4
    report = result['source_reports'][0]
    assert report['planning_checked_count'] == 5 and report['direction_survivors'] == 6
    assert report['search_stop_reason'] == 'task_planning_budget_exhausted'


def paused(setup, *, geometric=True):
    backend, _, checker, _ = setup
    entries = candidates(backend, direction='median' if geometric else 'vertical')['candidates']
    entry = entries[0]
    plan = backend.candidate_routes[entry['candidate_ref']][1]
    plan.scene_observation_id = 'obs'
    plan.scene_point_ref = 'point'
    backend.connector.pose = plan.targets[plan.target_labels.index('pregrasp')]
    backend._inflight = dict(original=plan, current_pose=plan.grasp_transform.copy(),
        replacement=None, adjustment=(0., 0., 0.), translation=(0., 0., 0.),
        attempts=[], observation={'observation_id': 'obs'})
    backend._inflight_context = dict(candidate_ref=entry['candidate_ref'], checker_scene=(checker, SCENE))
    backend._inflight_preview = lambda *args: ['preview']
    return backend, checker, plan, entry


def test_paused_nudge_keeps_transit_and_scene_metadata_updates_contact_and_preserves_topdown(setup):
    backend, checker, original, entry = paused(setup)
    result = backend.nudge_inflight_grasp(dx_mm=10, dz_mm=5, yaw_deg=5)
    assert result['accepted']
    replacement = backend._inflight['replacement']
    assert replacement.resume_from_pregrasp and replacement.target_labels == ('pregrasp', 'grasp', 'lift')
    assert replacement.pre_pick_z_m == .55 and replacement.post_pick_z_m == .75
    assert replacement.scene_point_ref == 'point' and replacement.scene_observation_id == 'obs'
    np.testing.assert_allclose(replacement.grasp_contract['contact_center_xyz_m'],
                               np.array(entry['contact_center_xyz_m'])+[.01, 0, .005])
    np.testing.assert_allclose(replacement.grasp_transform[:3, 2], [0, 0, -1], atol=1e-12)
    assert original.grasp_contract['contact_center_xyz_m'] == entry['contact_center_xyz_m']
    assert checker.calls[-1][2]['stop_label'] == 'lift'
    backend.nudge_inflight_grasp(dy_mm=4)
    assert backend._inflight['translation'] == (10, 4, 5)
    np.testing.assert_allclose(backend._inflight['replacement'].grasp_contract['contact_center_xyz_m'],
                               np.array(entry['contact_center_xyz_m'])+[.01, .004, .005])


@pytest.mark.parametrize('failure', ['path', 'preview', 'missing_scene', 'budget'])
def test_rejected_paused_nudge_preserves_pending_pose_and_records_actual_acceptance(setup, failure):
    backend, checker, _, _ = paused(setup)
    before = backend._inflight['current_pose'].copy()
    kwargs = dict(dx_mm=5)
    if failure == 'path':
        checker.accepted = False
    elif failure == 'preview':
        backend._inflight_preview = lambda *args: []
    elif failure == 'missing_scene':
        backend._inflight_context['checker_scene'] = None
    else:
        kwargs = dict(dx_mm=31)
    result = backend.nudge_inflight_grasp(**kwargs)
    assert not result['accepted'] and backend._inflight['replacement'] is None
    np.testing.assert_array_equal(before, backend._inflight['current_pose'])
    assert backend._inflight['translation'] == (0, 0, 0)
    if failure in ('path', 'preview'):
        assert backend._inflight['attempts'][-1]['accepted'] is False
        assert backend.events[-1][1]['accepted'] is False


def test_geometric_tilt_rejected_with_structured_feedback_but_cgn_rotation_allowed(setup):
    backend, _, _, _ = paused(setup)
    with pytest.raises(ActionPreconditionError) as raised:
        backend.nudge_inflight_grasp(roll_deg=2)
    feedback = raised.value.public_feedback()
    assert feedback['reason_code'] == 'geometric_grasp_requires_yaw_only'
    assert feedback['executed'] is False
    entry = candidates(backend)['candidates'][0]['candidate_ref']
    result = backend.refine_candidate(entry, 2, 3, 0)
    assert result['accepted'] and result['source'] == 'contact_graspnet'
    assert not np.allclose(result['approach_direction_base'], [0, 0, -1])


@pytest.mark.parametrize('grasp_equation', [True, False])
def test_attachment_archive_uses_executed_paused_replacement_before_inherited_wrapper_returns(setup, monkeypatch, grasp_equation):
    backend, _, _, _ = setup
    expected_equation = backend.attachment_release_equation
    if not grasp_equation:
        monkeypatch.delattr(GraspBackend, 'attachment_release_equation')
        expected_equation = 'AnyPlace_relative_transform @ actual_closed_ee'
    entry = candidates(backend, direction='median')['candidates'][0]
    ref = entry['candidate_ref']
    validation = backend.validate_grasp(ref)
    original = backend.candidate_routes[ref][1]
    replacement = deepcopy(original)
    replacement.grasp_transform[0, 3] += .01
    replacement.grasp_executed = True
    replacement.grasp_contract['contact_center_xyz_m'][0] += .01

    def simulate_execution_boundary(self, candidate_ref, validation_ref):
        self.held_plan = original  # shared LiveBackend returns its original argument
        self._inflight = dict(replacement=replacement, refinement=None)
        self.first_grasp_evidence = dict(execution_ref='fake-execution', observations={
            'post_close': {'robot_ee_pose': replacement.grasp_transform.tolist()}})
        return dict(status='succeeded', execution_ref='fake-execution')

    monkeypatch.setattr(prime_backend.LiveBackend, '_execute_grasp', simulate_execution_boundary)
    monkeypatch.setattr('src.tools.gripper.state.measured_opening',
                        lambda *args, **kwargs: (.02, [.01, .01], {}))
    result = backend.execute_grasp(ref, validation['validation_ref'])
    assert result['status'] == 'succeeded' and backend.held_plan is replacement
    archive = next(backend.output_dir.glob('attachment_*/attachment.npz'))
    with np.load(archive, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['grasp_transform'], replacement.grasp_transform)
        np.testing.assert_array_equal(saved['actual_closed_ee'], replacement.grasp_transform)
    assert backend.held_plan.grasp_contract['contact_center_xyz_m'][0] == pytest.approx(.41)
    metadata = json.loads(archive.with_suffix('.json').read_text())
    assert metadata['release_equation'] == expected_equation
    event = next(value for kind, value in backend.events if kind == 'grasp_attachment_saved')
    assert event['release_equation'] == expected_equation


@pytest.mark.parametrize('operation', ['grasp_candidates', 'place_candidates', '_placement_pool',
                                       'relax_candidate', 'execute_place_candidate'])
def test_legacy_entrypoints_are_blocked_by_trusted_structured_feedback(setup, operation):
    backend, client, _, _ = setup
    with pytest.raises(ActionPreconditionError) as raised:
        getattr(backend, operation)('irrelevant')
    feedback = raised.value.public_feedback()
    assert feedback['error'] == 'action_precondition_failed' and feedback['state_changed'] is False
    assert not client.requests and not backend.connector.events


def test_held_grasp_gate_and_invalid_policy_do_not_invoke_cgn(setup):
    backend, client, _, _ = setup
    with pytest.raises(ValueError):
        candidates(backend, tolerance_deg=-1)
    backend.held_plan = object()
    with pytest.raises(ActionPreconditionError) as raised:
        candidates(backend)
    assert raised.value.reason_code == 'grasp_requires_release'
    assert not client.requests


def test_grasp_sweep_disabled_for_candidates_and_refinement(setup):
    backend, _, checker, _ = setup
    backend.grasp_path_collision_checks = False
    backend.records = []
    backend._record = lambda kind, value: backend.records.append((kind, value))
    checker.accepted = False
    entries = candidates(backend)['candidates']
    assert len(entries) == 4 and not checker.calls
    refined = backend.refine_candidate(entries[0]['candidate_ref'], roll_deg=0, pitch_deg=0, yaw_deg=5)
    assert refined['accepted'] and not checker.calls
    records = [value for kind, value in backend.records if kind == 'grasp_path_validation']
    assert records and all(r['collision_check_status'] == 'disabled_by_configuration' for r in records)
    assert all(r['grasp_path_collision_checks'] is False for r in records)


@pytest.mark.parametrize('enabled', [False, True])
def test_grasp_sweep_switch_applies_to_paused_nudge(setup, enabled):
    backend, checker, _, _ = paused(setup)
    backend.grasp_path_collision_checks = enabled
    checker.accepted = False
    checker.calls.clear()
    result = backend.nudge_inflight_grasp(dx_mm=5)
    assert result['accepted'] is (not enabled)
    assert bool(checker.calls) is enabled


def test_remote_height_references_resolve_against_each_candidate_and_current_tcp(setup):
    backend, client, _, _ = setup
    from src.tools.grasp.cgn_client import RawCGNGrasps
    poses = client.result.poses.copy()
    poses[:, 2, 3] += np.arange(len(poses)) * .003
    client.result = RawCGNGrasps(poses, client.result.scores, client.result.contact_points, ROBOT_BASE)
    context = backend.explicit_geometry('point')
    result = candidates(backend, geometric_height=dict(reference='current_tcp', value_m=-.1),
        transit=dict(pre=dict(reference='grasp', value_m=.3), post=dict(reference='grasp', value_m=.4)))
    for item in result['candidates'] + result['diagnostic_candidates']:
        contract = backend.grasp_contracts[item['candidate_ref']]
        z = contract['contact_center_xyz_m'][2]
        assert contract['pre_pick_z_m'] == pytest.approx(z + .3)
        assert contract['post_pick_z_m'] == pytest.approx(z + .4)
        if contract['top_down_only']:
            assert z == pytest.approx(context['current_tcp_z_m'] - .1)
    assert len(client.requests) == 1


def test_agent_geometry_includes_full_tcp_axis_width_and_preapproach(setup):
    backend, _, _, _ = setup
    context = backend.explicit_geometry('point')
    tcp = hand_to_contact(motion._pose_transform(backend.connector.get_ee_pose()), .136)
    np.testing.assert_allclose(context['current_tcp_pose_base'], tcp)
    assert context['gripper_max_opening_m'] == .085
    assert context['grasp_axis_preapproach_distance_m'] == backend.motion_config.approach_m
    assert context['statistics']['principal_xy_axis_yaw_deg'] is not None
    assert context['per_view_statistics'][0]['view_id'] == 'front'
    assert context['last_grasp_target_tcp_z_m'] is None
    backend.held_plan = NS(grasp_contract={'contact_center_xyz_m': [.4, 0, .3]})
    assert backend.explicit_geometry('point')['last_grasp_target_tcp_z_m'] == .3


def test_unavailable_grasp_height_rejects_before_cgn(setup):
    backend, client, _, _ = setup
    with pytest.raises(ValueError, match='unavailable: grasp'):
        candidates(backend, direction='median', geometric_height=dict(reference='grasp', value_m=0.))
    assert not client.requests


def test_pick_projection_uses_cyan_native_mesh_and_inspection_cache(setup, monkeypatch):
    backend, _, checker, _ = setup
    from src.tools.pose_editor import refinement as pose_refinement
    calls = []
    backend.point_adapter.frames['obs'] = ['saved-front', 'saved-wrist']
    triangles = np.arange(9).reshape(1, 3, 3)
    backend.gripper_assets.load_gripper_mesh = lambda width: ({'hand': triangles}, {})
    def project(frames, pose, directory, **kwargs):
        parts, _ = kwargs['mesh_source'].load_gripper_mesh(kwargs['expected_open_width_m'])
        assert set(parts) == {'left_finger'}
        np.testing.assert_array_equal(parts['left_finger'], triangles)
        assert frames == ['saved-front', 'saved-wrist']
        calls.append(pose.copy())
        directory.mkdir(parents=True)
        path = directory/'preview.png';Image.new('RGB', (12, 12), 'cyan').save(path)
        return [str(path)]
    monkeypatch.setattr(pose_refinement, 'preview_refined_pose', project)
    monkeypatch.setattr(prime_backend, 'render_candidate', lambda *a, **k: pytest.fail('legacy glyph'))
    monkeypatch.setattr(intent_backend, 'render_candidate', lambda *a, **k: pytest.fail('legacy glyph'))
    entries = candidates(backend)['candidates']
    assert len(calls) == len(entries) == 4
    ref = entries[0]['candidate_ref']
    backend.inspect_candidate(ref)
    assert len(calls) == 4
    checker.accepted = False
    result = backend.refine_candidate(ref, roll_deg=0, pitch_deg=0, yaw_deg=5)
    assert not result['accepted'] and len(calls) == 5


@pytest.mark.parametrize('mode', ['mean', 'median'])
def test_center_mode_returns_only_four_fixed_base_yaws_without_cgn(setup, mode):
    backend, client, _, _ = setup
    client.error = CGNServiceError('must never call service for center mode')
    result = candidates(backend, direction=mode)
    entries = result['candidates']
    assert len(entries) == 4 and not result['diagnostic_candidates']
    assert not client.requests
    assert result['candidate_limit'] == 4 and result['generator'] == mode
    xy = getattr(np, mode)(OBJECT, axis=0)[:2]
    for entry, yaw in zip(entries, [-45, 0, 45, 90]):
        assert entry['source'] == mode and entry['top_down_only']
        assert entry['yaw_deg'] == pytest.approx(yaw)
        np.testing.assert_allclose(entry['contact_center_pose_base'], top_down_contact_pose(xy, .3, yaw))
        assert entry['transit'] == dict(pre_pick_z_m=.55, post_pick_z_m=.75)


def test_cgn_failure_requires_explicit_subsequent_center_request(setup):
    backend, client, _, _ = setup
    client.error = CGNServiceError('down')
    assert candidates(backend)['candidates'] == []
    assert len(client.requests) == 1
    entries = candidates(backend, direction='mean')['candidates']
    assert len(entries) == 4 and len(client.requests) == 1
