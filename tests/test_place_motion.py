"""CPU contracts for target-free motion and the separate release action."""
from types import SimpleNamespace as N
import json

import numpy as np
import pytest

from src.tools.motion import planning as motion
from src.core.action_feedback import ActionPreconditionError
from src.tools.grasp.backend import hand_to_contact
from src.tools.place.execution import PlacementMotionMixin, observed_scene
from test_motion import GRASP_TO_EE, hand_pose
from test_graspgen_motion import Connector


ABS = dict(mode='absolute', value_m=.45, reference='none')


class CommandEnv:
    enforce_motion_tracking = True
    motion_position_tolerance_m = .01
    motion_orientation_tolerance_rad = .05
    motion_joint_tolerance_rad = .01
    def __init__(self):
        self._width_target = .03
    def _set_gripper_width(self, width):
        self._width_target = width
    def _validate_joints(self, target):
        if np.any(np.abs(target) > 3.):
            raise ValueError('joint target outside limits')
        return target


class MovingConnector(Connector):
    def __init__(self):
        super().__init__()
        self.env = CommandEnv()
        self._gripper_fraction = .35
        self._home_joints = [.2] * 7
        self.ik.model = N(fk=self.fk)
        self.pose = motion.transform_to_pose(hand_pose() @ GRASP_TO_EE)
        self.command_during_motion = []
    def fk(self, joints):
        result = hand_pose() @ GRASP_TO_EE
        result[0, 3] += float(joints[0])
        return result
    def set_gripper(self, fraction):
        self._gripper_fraction = fraction
        self.env._set_gripper_width(fraction * .085)
    def execute_trajectory(self, segment):
        self.command_during_motion.append(self.env._width_target)
        self.events.append('trajectory')
        self.q = segment['waypoints'][-1]['positions']
        self.pose = segment.get('target', motion.transform_to_pose(self.fk(self.q)))
    def open_gripper(self, *, settle_steps):
        self.events.append('open')
        self.set_gripper(1.)


class Base:
    """View boundary fake; motion must pass through validation and execution."""
    def __init__(self):
        self.connector = MovingConnector()
        self.motion_config = motion.MotionConfig(collision_checks_enabled=False)
        self.grasp_to_ee = GRASP_TO_EE.copy()
        self.jaw_offset_m = .136
        self.grasp_jaw_width_m = .03
        self.max_gripper_width_m = .085
        self.held_plan = N(grasp_executed=True, target_points=np.array([[.4, 0., .3]]))
        self.grasp_attachment = motion._pose_transform(self.connector.pose)
        self.closed_push = False
        self.grasp_attempted = True
        self.plan_only = False
        self.recorder = None
        self.epoch = 0
        self.latest_observation_id = 'obs0'
        self.latest_fused_ref = None
        self.observation_image_refs = {'obs0': ['rgb0']}
        self.frame = N(depth_m=np.array([[.3, .4], [.5, .6]]),
            intrinsics=np.array([[2., 0., 0.], [0., 2., 0.], [0., 0., 1.]]),
            camera_to_base=N(rotation=np.eye(3), translation=np.zeros(3)))
        self.point_adapter = N(latest='obs0', frames={'obs0': [self.frame]}, _check_current=self.check_current)
        self.destinations = {'dest': N(point_ref='point', observation_id='obs0', epoch=0,
            points=np.array([[.4, .1, .1], [.5, .2, .2], [.8, .3, .3]]),
            anchor=np.array([.43, .14, .12]))}
        self.clicked_points = {'point': np.array([.43, .14, .12])}
        self.view_proposals, self.view_validations, self.validations = {}, {}, {}
        self.last_place = None
        self.placement_execution = None
        self.records, self.validated, self.projections = [], [], []
        self.reject = False
        self.fail_arrival = False
        self.preview_available = True
    def _record(self, kind, value):
        self.records.append((kind, value))
    def check_current(self, observation_id):
        if observation_id != self.point_adapter.latest:
            raise ValueError('stale observation')
    def _robot_state(self):
        return json.dumps(dict(joints=self.connector.q, pose=self.connector.pose), sort_keys=True)
    def _view(self, ref):
        value = self.view_proposals[ref]
        if value['epoch'] != self.epoch:
            raise ValueError('stale waypoint')
        self.check_current(value['observation_id'])
        return value
    def _explicit_project_place_hand(self, observation_id, hand, *, caption, height_status):
        self.preview_hand = hand.copy()
        self.projections.append(dict(observation_id=observation_id, hand=hand.copy(),
                                    caption=caption, height_status=height_status))
        return ['pose-preview'] if self.preview_available else []
    def validate_view(self, ref, motion='planned'):
        p = self._view(ref)
        self.validated.append((p, motion))
        token = 'v'+str(len(self.validated))
        self.view_validations[token] = (ref, p['start'])
        return dict(accepted=not self.reject, validation_ref=token, reason_code='path_rejected' if self.reject else 'accepted')
    def _execute_waypoint(self, plan):
        for segment in plan.segments:
            self.connector.execute_trajectory(segment)
    def execute_view(self, ref, token):
        p = self._view(ref)
        saved, state = self.view_validations.pop(token)
        if saved != ref or state != self._robot_state():
            raise ValueError('stale validation')
        if self.plan_only:
            return dict(view_status='not_executed', observation_id=self.latest_observation_id)
        self.epoch += 1
        self.validations.clear()
        self.view_validations.clear()
        if self.held_plan is not None or self.closed_push:
            self.connector.set_gripper(0.)  # Existing executor's reminder.
        target = p['pose'].copy()
        if self.fail_arrival:
            target[2, 3] += .1
        segment = dict(waypoints=[dict(positions=self.connector.q.copy())],
                       target=motion.transform_to_pose(target))
        self._execute_waypoint(N(segments=(segment,)))
        return dict(**self.observe(), view_status='requested_view_failed' if self.fail_arrival else 'achieved')
    def observe(self):
        observation_id = 'obs'+str(self.epoch)
        self.latest_observation_id = self.point_adapter.latest = observation_id
        self.point_adapter.frames[observation_id] = [self.frame]
        return dict(observation_id=observation_id, views=[dict(image_ref='fresh-'+observation_id)])


class Backend(PlacementMotionMixin, Base):
    _explicit_project_place_hand = Base._explicit_project_place_hand


def prepare(b, **kw):
    args = dict(destination_ref='dest', xy_source='median', xy_m=[.51, .22], height=ABS, transit_height=dict(reference='absolute', value_m=.65))
    args.update(kw)
    return b.explicit_prepare_place(**args)


def test_measured_candidates_offer_click_median_mean_with_distinct_sources():
    b = Backend()
    result = b.explicit_place_candidates('dest')
    rows = result['xy_candidates']
    assert [r['candidate_id'] for r in rows] == ['clicked', 'median', 'mean']
    assert rows[0]['xy_m'] == [.43, .14]
    assert rows[1]['xy_m'] == [.5, .2]
    np.testing.assert_allclose(rows[2]['xy_m'], [.5666666667, .2])
    assert result['statistics']['max_xyz_m'][2] == .3
    assert result['height_decision_required'] is True
    assert [p['xy_source'] for p in result['candidate_previews']] == ['clicked', 'median', 'mean']
    for projection, row, preview in zip(b.projections, rows, result['candidate_previews']):
        contact = hand_to_contact(projection['hand'], b.jaw_offset_m)
        expected = np.asarray(row['observed_xyz_m']) + [0., 0., .05]
        np.testing.assert_allclose(contact[:3, 3], expected)
        np.testing.assert_allclose(preview['preview_contact_center_xyz_m'], expected)
        assert preview['reference_contact_center_xyz_m'] == row['observed_xyz_m']
        assert preview['preview_z_offset_m'] == .05
        assert projection['height_status'] == 'PREVIEW: surface Z + 0.05 m; execution height not selected'
    assert not any(p['executable'] for p in result['candidate_previews'])
    assert b.connector.events == []


@pytest.mark.parametrize('reference,z', [('clicked', .22), ('median', .3), ('mean', .3), ('min', .2), ('max', .4)])
def test_place_height_resolves_explicit_surface_reference_and_preserves_orientation(reference, z):
    b = Backend()
    before = motion._pose_transform(b.connector.pose)
    result = prepare(b, height=dict(mode='surface_relative', value_m=.1, reference=reference))
    assert result['contact_center_xyz_m'] == pytest.approx([.51, .22, z])
    target = b._view(result['waypoint_ref'])['pose']
    np.testing.assert_allclose(target[:3, :3], before[:3, :3])
    contact = hand_to_contact(target @ np.linalg.inv(GRASP_TO_EE), .136)
    np.testing.assert_allclose(contact[:3, 3], [.51, .22, z])
    assert target[2, 3] == pytest.approx(z+.136)
    assert result['robot_motion'] is False and result['release_commanded'] is False
    assert b.connector.events == []
    assert b.validated[0][0]['purpose'] == 'transport'
    assert b.validated[0][0]['target_ref'] is None


def test_place_achieves_goal_holding_then_only_explicit_release_opens_and_clears_state():
    b = Backend()
    candidate = prepare(b)
    result = b.explicit_execute_place(candidate['candidate_ref'])
    assert result['view_status'] == 'achieved' and result['release_required']
    assert result['arrival_current'] and not b._explicit_place_release_blocked
    assert result['views'][0]['image_ref'] == 'fresh-obs1'
    assert result['execution_ref'] and result['executed'] and result['state_changed']
    assert result['status'] == 'succeeded' and result['observation']['observation_id'] == 'obs1'
    assert b.connector.events == ['trajectory']
    assert b.connector.command_during_motion == [.03]
    assert b.connector.env._width_target == .03 and b.connector._gripper_fraction == .35
    assert b.held_plan is not None and b.grasp_attachment is not None
    assert b.placement_execution is None
    released = b.release()
    assert released['status'] == 'succeeded' and released['release_commanded'] is True
    assert released['success_verified'] is False
    assert b.connector.events == ['trajectory', 'open']
    assert b.held_plan is b.grasp_attachment is None
    assert not b.destinations and not b.grasp_attempted
    assert b.placement_execution == released and b.epoch == 2


@pytest.mark.parametrize('failure', ['path', 'tracking', 'plan_only'])
def test_rejected_unexecuted_or_missed_place_never_opens_and_blocks_release(failure):
    b = Backend()
    b.reject = failure == 'path'
    b.fail_arrival = failure == 'tracking'
    b.plan_only = failure == 'plan_only'
    result = b.explicit_execute_place(prepare(b)['candidate_ref'])
    assert result['arrival_current'] is False and b._explicit_place_release_blocked
    assert 'open' not in b.connector.events
    with pytest.raises(ActionPreconditionError) as caught:
        b.release()
    assert caught.value.reason_code == 'release_requires_placement_retry'
    assert b.held_plan is not None


def test_retry_reaches_explicit_corrected_goal_and_unblocks_release():
    b = Backend()
    b.fail_arrival = True
    b.explicit_execute_place(prepare(b)['candidate_ref'])
    b.fail_arrival = False
    assert b.explicit_execute_place(prepare(b)['candidate_ref'])['arrival_current']
    assert b.release()['release_commanded'] is True


def test_adjust_place_rotates_about_contact_center_and_translates_in_base_axes():
    b = Backend()
    original = prepare(b)
    updated = b.explicit_adjust_place(original['candidate_ref'], 5, -6, 7, 8, 4, -3)
    np.testing.assert_allclose(updated['contact_center_xyz_m'], [.515, .214, .457])
    first = b._view(original['waypoint_ref'])['pose']
    second = b._view(updated['waypoint_ref'])['pose']
    assert not np.allclose(first[:3, :3], second[:3, :3])
    assert updated['translation_mm'] == [5, -6, 7] and updated['rotation_deg'] == [8, 4, -3]
    assert b.explicit_inspect_place(updated['candidate_ref'])['image_refs'] == ['pose-preview']
    assert not b.connector.events
    with pytest.raises(ValueError):
        b.explicit_adjust_place(updated['candidate_ref'], 0, 0, 0, 11, 0, 0)
    with pytest.raises(ValueError):
        b.explicit_adjust_place(updated['candidate_ref'], 31, 0, 0, 0, 0, 0)


def test_adjustment_cumulative_limit_and_epoch_prevent_bypasses():
    b = Backend()
    candidate = prepare(b)['candidate_ref']
    for _ in range(3):
        candidate = b.explicit_adjust_place(candidate, 30, 0, 0, 0, 0, 10)['candidate_ref']
    with pytest.raises(ValueError):
        b.explicit_adjust_place(candidate, 1, 0, 0, 0, 0, 0)
    with pytest.raises(ValueError):
        b.explicit_adjust_place(candidate, 0, 0, 0, 0, 0, 1)
    b.move_vertical(.02)
    with pytest.raises(ValueError, match='current placement'):
        b.explicit_execute_place(candidate)


@pytest.mark.parametrize('state', ['open', 'held', 'pusher'])
def test_vertical_is_target_free_linear_and_preserves_current_command(state):
    b = Backend()
    if state != 'held':
        b.held_plan = b.grasp_attachment = None
    b.closed_push = state == 'pusher'
    before = motion._pose_transform(b.connector.pose)
    result = b.move_vertical(-.08)
    after = motion._pose_transform(b.connector.pose)
    np.testing.assert_allclose(after[:3, :3], before[:3, :3], atol=1e-12)
    np.testing.assert_allclose(after[:3, 3]-before[:3, 3], [0., 0., -.08], atol=1e-12)
    assert b.validated[0][1] == 'linear'
    assert b.validated[0][0]['target_ref'] is None
    assert b.validated[0][0]['purpose'] != 'contact'
    assert b.connector.command_during_motion == [.03]
    assert result['view_status'] == 'achieved' and result['release_commanded'] is False
    assert b.connector.events == ['trajectory']
    assert result['execution_ref'] and result['observation']['observation_id'] == 'obs1'


def test_move_after_place_invalidates_release_arrival_but_failed_validation_does_not_move():
    b = Backend()
    b.explicit_execute_place(prepare(b)['candidate_ref'])
    b.move_vertical(.03)
    assert b._explicit_place_release_blocked
    with pytest.raises(ActionPreconditionError):
        b.release()
    b = Backend()
    b.reject = True
    rejected = b.move_vertical(.05)
    assert rejected['status'] == 'not_executed' and rejected['executed'] is False
    assert 'execution_ref' not in rejected
    assert not b.connector.events


def test_empty_grasp_recovery_release_is_available_without_placement():
    b = Backend()
    b.held_plan = b.grasp_attachment = None
    assert b.release()['status'] == 'succeeded'
    assert b.connector.events == ['open']


def test_current_full_rgbd_scene_transform_and_invalid_pixels():
    frame = N(depth_m=np.array([[.5, np.nan], [0., 3.1]]), intrinsics=np.eye(3),
              camera_to_base=N(rotation=np.eye(3), translation=np.array([1., 2., 3.])))
    np.testing.assert_allclose(observed_scene([frame]), [[1., 2., 3.5]])
    frame.depth_m[:] = np.nan
    with pytest.raises(ValueError, match='no valid'):
        observed_scene([frame])


def test_cyan_closeup_selects_left_view_and_magnifies_with_padding(tmp_path):
    from PIL import Image
    from src.tools.pose_editor.refinement import preview_refined_pose
    from src.tools.place.execution import _CyanGripperMesh
    frame = N(view_id='agentview', rgb=np.zeros((360, 640, 3), dtype=np.uint8),
              intrinsics=np.array([[250., 0, 320], [0, 250., 180], [0, 0, 1.]]),
              camera_to_base=N(rotation=np.eye(3), translation=np.zeros(3)))
    wrist = N(**{**vars(frame), 'view_id': 'robot0_eye_in_hand'})
    triangles = np.array([[[-.04, -.04, 1.], [.04, -.04, 1.], [0., .04, 1.]]])
    mesh = _CyanGripperMesh(N(load_gripper_mesh=lambda width: ({'hand': triangles}, {})))
    zoom = preview_refined_pose([frame, wrist], np.eye(4), tmp_path/'zoom',
                               tcp_offset_z_m=0., mesh_source=mesh)
    full = preview_refined_pose([frame, wrist], np.eye(4), tmp_path/'full',
                               tcp_offset_z_m=0., mesh_source=mesh, closeup=False)
    assert len(zoom) == len(full) == 1 and 'agentview' in zoom[0]
    spans = []
    for path in [full[0], zoom[0]]:
        pixels = np.asarray(Image.open(path)).astype(int)
        assert pixels.shape == (360, 640, 3)
        y, x = np.where((pixels[:, :, 1] > pixels[:, :, 0] + 20)
                        & (pixels[:, :, 2] > pixels[:, :, 0] + 20))
        spans.append(x.max() - x.min())
        assert x.min() > 0 and x.max() < 639 and y.min() > 0 and y.max() < 359
    assert spans[1] > spans[0] * 2
    assert not frame.rgb.any()


def test_cyan_projection_reuses_refiner_and_original_native_mesh_without_mutation(tmp_path, monkeypatch):
    b = Backend()
    b.output_dir = tmp_path
    b.images = N(add=lambda path: str(path))
    b.frame.rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    b.frame.view_id = 'agentview'
    wrist = N(**vars(b.frame)); wrist.view_id = 'robot0_eye_in_hand'
    b.point_adapter.frames['obs0'].append(wrist)
    part_a = np.array([[[-.02, -.02, 0.], [.02, -.02, 0.], [0., .02, 0.]]])
    part_b = part_a + [0., 0., .01]
    b.gripper_assets = N(load_gripper_mesh=lambda width: ({'hand': part_a.copy(), 'finger': part_b.copy()}, {}))
    from src.tools.pose_editor import refinement as pose_refinement; from src.tools.grasp import input_cards as grasp_input_cards
    original = pose_refinement.preview_refined_pose
    calls = []
    def render(frames, hand, directory, **kwargs):
        assert [f.view_id for f in frames] == ['agentview']
        assert kwargs['mesh_source'].preview_supersampling == 4
        parts, _ = kwargs['mesh_source'].load_gripper_mesh(kwargs['expected_open_width_m'])
        calls.append((hand.copy(), parts, kwargs))
        return original(frames, hand, directory, **kwargs)
    monkeypatch.setattr(pose_refinement, 'preview_refined_pose', render)
    b._explicit_project_place_hand = lambda *args, **kwargs: PlacementMotionMixin._explicit_project_place_hand(b, *args, **kwargs)
    result = b.explicit_place_candidates('dest')
    assert len(result['image_refs']) == 3 and len(calls) == 3
    assert grasp_input_cards.COLORS['left_finger'] == (29, 167, 207)
    for (hand, parts, options), row in zip(calls, result['xy_candidates']):
        assert set(parts) == {'left_finger'}
        np.testing.assert_array_equal(parts['left_finger'], np.concatenate([part_a, part_b]))
        np.testing.assert_allclose(hand_to_contact(hand, .136)[:3, 3],
                                   np.asarray(row['observed_xyz_m']) + [0., 0., .05])
        assert options['tcp_offset_z_m'] == .136 and options['expected_open_width_m'] == .03
    from PIL import Image
    im = np.asarray(Image.open(result['image_refs'][0]))
    assert im.shape == (128, 128, 3) and np.any(im != 0)
    assert np.all(b.frame.rgb == 0)
    selected = prepare(b)
    np.testing.assert_allclose(hand_to_contact(calls[-1][0], .136)[:3, 3], [.51, .22, .45])
    assert selected['image_refs']
    b.explicit_adjust_place(selected['candidate_ref'], 5, -6, 7, 8, 4, -3)
    np.testing.assert_allclose(hand_to_contact(calls[-1][0], .136)[:3, 3], [.515, .214, .457])
    b.grasp_jaw_width_m = 0.
    b.explicit_inspect_place(selected['candidate_ref'])
    assert calls[-1][2]['expected_open_width_m'] == 0.


def test_cyan_closeup_retains_subpixel_mesh_detail(tmp_path):
    """Two edges within one source pixel must separate after crop magnification."""
    from PIL import Image
    from src.tools.pose_editor.refinement import preview_refined_pose
    from src.tools.place.execution import _CyanGripperMesh
    frame = N(view_id='agentview', rgb=np.zeros((360, 640, 3), dtype=np.uint8),
              intrinsics=np.array([[250., 0, 320], [0, 250., 180], [0, 0, 1.]]),
              camera_to_base=N(rotation=np.eye(3), translation=np.zeros(3)))
    # Thin vertical rectangle, just 0.3 source pixels across: rasterizing
    # before the ~5x crop enlargement would inflate it to a full source pixel.
    vertices = np.array([[.0004, -.04, 1.], [.0016, -.04, 1.],
                         [.0016, .04, 1.], [.0004, .04, 1.]])
    triangles = vertices[[[0, 1, 2], [0, 2, 3]]]
    mesh = _CyanGripperMesh(N(load_gripper_mesh=lambda width: ({'hand': triangles}, {})))
    path, = preview_refined_pose([frame], np.eye(4), tmp_path,
                                 tcp_offset_z_m=0., mesh_source=mesh)
    pixels = np.asarray(Image.open(path)).astype(float)
    row = pixels[180, :, 2]
    # A narrow antialiased line, not the broad source-pixel block.
    assert 1 <= np.count_nonzero(row > row.max() * .5) <= 3
    assert np.any((row > 0) & (row < row.max() * .5))
    assert not frame.rgb.any()


def test_missing_observation_recaptures_and_invalid_request_never_moves():
    b = Backend()
    b.latest_observation_id = None
    b.move_vertical(.03)
    assert b.latest_observation_id == 'obs1'
    for value in (float('nan'), True, '0.1'):
        with pytest.raises(ValueError):
            b.move_vertical(value)
    b = Backend()
    b.preview_available = False
    with pytest.raises(ValueError, match='preview unavailable'):
        prepare(b)
    assert not b.connector.events


class Checker:
    def __init__(self, accepted=True):
        self.accepted = accepted
        self.calls = []
    def remove_captured_robot(self, scene):
        return scene.copy(), {'removed': 0}
    def check(self, plan, scene, **kwargs):
        self.calls.append((plan, scene.copy(), kwargs))
        return dict(accepted=self.accepted)


def home_backend(held=False, accepted=True):
    b = Backend()
    if not held:
        b.held_plan = b.grasp_attachment = None
    checker = Checker(accepted)
    b._explicit_make_checker = lambda: checker
    # Sensor cloud filtering is shared with existing held waypoint semantics;
    # this test isolates home route creation/execution and held sweep dispatch.
    if held:
        b._explicit_home_scene = lambda checker, scene: (scene, {'removed_held_samples': 0})
    b.held_checks = []
    def held_check(checker, plan, scene):
        b.held_checks.append((plan, scene))
        return dict(accepted=True)
    b._check_held_path = held_check
    return b, checker


@pytest.mark.parametrize('held', [False, True])
def test_home_uses_native_joints_checks_full_path_and_preserves_command(held):
    b, checker = home_backend(held)
    b.home_joints = [.3] * 7
    result = b.goto_home_joint_position()
    assert result['status'] == 'succeeded' and result['success_verified'] is False
    np.testing.assert_allclose(b.connector.q, [.3]*7)
    plan, scene, kwargs = checker.calls[0]
    assert kwargs['stop_label'] == 'waypoint'
    assert len(plan.segments[0]['waypoints']) >= 13
    assert len(scene) == 4
    assert bool(b.held_checks) == held
    assert b.connector.events == ['trajectory']
    assert b.connector.command_during_motion == [.03]
    assert b.connector.env._width_target == .03
    assert result['views'][0]['image_ref'] == 'fresh-obs1'


def test_home_collision_or_payload_rejection_cannot_execute():
    b, _ = home_backend(accepted=False)
    assert b.goto_home_joint_position()['reason_code'] == 'home_path_rejected'
    assert not b.connector.events
    b, _ = home_backend(held=True)
    b._check_held_path = lambda *args: dict(accepted=False, kind='held_object_scene')
    assert b.goto_home_joint_position()['status'] == 'not_executed'
    assert not b.connector.events


def test_home_missing_configuration_invalid_joint_or_plan_only_never_moves():
    b, _ = home_backend()
    b.connector._home_joints = None
    with pytest.raises(ValueError, match='home joints required'):
        b.goto_home_joint_position()
    b.home_joints = [4.] * 7
    with pytest.raises(ValueError, match='outside limits'):
        b.goto_home_joint_position()
    b.home_joints = [.1] * 7
    b.plan_only = True
    assert b.goto_home_joint_position()['reason_code'] == 'plan_only'
    assert not b.connector.events


def test_home_wrong_measured_arrival_reports_failure_and_does_not_release():
    b, _ = home_backend()
    def miss(segment):
        b.connector.events.append('attempt')
        b.connector.q = [.05] * 7
        b.connector.pose = motion.transform_to_pose(b.connector.fk(b.connector.q))
    b.connector.execute_trajectory = miss
    result = b.goto_home_joint_position()
    assert result['status'] == 'failed' and result['reason_code'] == 'home_execution_incomplete'
    assert result['joint_max_error_rad'] == pytest.approx(.15)
    assert b.connector.events == ['attempt']


def test_release_failure_keeps_attachment_as_unknown_and_reports_no_success():
    b = Backend()
    def fail(**kwargs):
        raise RuntimeError('actuation failed')
    b.connector.open_gripper = fail
    result = b.release()
    assert result['status'] == 'failed' and result['release_commanded'] is None
    assert b.held_plan is not None and b.grasp_attachment is not None


def test_release_accepts_verified_episode_completion_during_settling():
    b = Backend()
    b.connector.env.simulation_budget = lambda: dict(terminal=True, reason_code='episode_terminated')
    b.connector.check_success = lambda: (True, 0.)
    def open_then_end(**kwargs):
        MovingConnector.open_gripper(b.connector, **kwargs)
        raise RuntimeError('RoboLab episode ended; explicit reset required before motion')
    b.connector.open_gripper = open_then_end
    result = b.release()
    assert result['status'] == 'succeeded' and result['release_commanded'] is True
    assert result['terminal'] is True and result['reason_code'] == 'episode_terminated'
    assert 'error' not in result and result['success_verified'] is False
    assert result['execution_feedback']['task_success'] is None
    assert 'during settling' in result['execution_feedback']['interpretation']
    assert b.connector.events == ['open']
    assert b.held_plan is b.grasp_attachment is None
    assert not b.destinations and not b.grasp_attempted
    assert result['observation']['observation_id'] == 'obs1'
    assert b.placement_execution == result


@pytest.mark.parametrize('case', [
    'time_limit', 'nonterminal', 'verifier_false', 'verifier_error',
    'budget_error', 'missing_budget', 'actuation_error', 'wrong_exception_type',
])
def test_release_does_not_hide_unverified_or_unrelated_errors(case):
    b = Backend()
    def budget():
        if case == 'budget_error':
            raise RuntimeError('budget unavailable')
        return dict(terminal=case != 'nonterminal',
                    reason_code='simulation_time_limit' if case == 'time_limit' else 'episode_terminated')
    def verify():
        if case == 'verifier_error':
            raise RuntimeError('verifier unavailable')
        return case != 'verifier_false', 0.
    if case != 'missing_budget':
        b.connector.env.simulation_budget = budget
    b.connector.check_success = verify
    error = RuntimeError('RoboLab episode ended; explicit reset required before motion')
    if case == 'actuation_error':
        error = RuntimeError('actuation failed')
    elif case == 'wrong_exception_type':
        error = ValueError(str(error))
    def fail(**kwargs):
        raise error
    b.connector.open_gripper = fail
    result = b.release()
    assert result['status'] == 'failed' and result['release_commanded'] is None
    assert result['reason_code'] == 'release_execution_incomplete'
    assert result['error'] == repr(error)
    assert b.held_plan is not None and b.grasp_attachment is not None


@pytest.mark.parametrize('stage', ['enter', 'exit'])
def test_release_does_not_hide_recorder_errors_after_task_success(stage):
    from contextlib import contextmanager
    b = Backend()
    b.connector.env.simulation_budget = lambda: dict(terminal=True, reason_code='episode_terminated')
    b.connector.check_success = lambda: (True, 0.)
    @contextmanager
    def record(*args, **kwargs):
        if stage == 'exit':
            yield
        raise RuntimeError('RoboLab episode ended; explicit reset required before motion')
    b.recorder = N(active=record)
    result = b.release()
    assert result['status'] == 'failed' and result['release_commanded'] is None
    assert b.held_plan is not None and b.grasp_attachment is not None


def test_place_independent_transit_survives_adjustment_and_checks_whole_route(monkeypatch):
    from src.backend.robot import IntentBackend
    from src.tools.motion import path_collision as candidate_path_collision
    b = Backend()
    checker = Checker()
    monkeypatch.setattr(candidate_path_collision, 'make_candidate_path_collision', lambda *a, **kw: checker)
    b.validate_view = lambda ref, motion='planned': IntentBackend.validate_view(b, ref)
    held_checks = []
    b._check_held_path = lambda checker, plan, scene: held_checks.append(plan) or dict(accepted=True)
    route_calls = []
    def plan(connector, targets, scene, config, validator, *, target_labels):
        route_calls.append((targets, target_labels))
        segments = tuple(dict(waypoints=[dict(positions=connector.q.copy())],
                              target=motion.transform_to_pose(p)) for p in targets)
        return segments, tuple(motion.transform_to_pose(p) for p in targets), tuple(connector.q), None, False
    b._explicit_chain_planner = plan
    current = hand_to_contact(motion._pose_transform(b.connector.pose) @ np.linalg.inv(GRASP_TO_EE), .136)
    choice = prepare(b, height=dict(reference='absolute', value_m=.4),
                     transit_height=dict(reference='current_tcp', value_m=.3))
    assert choice['accepted']
    travel = current[2, 3] + .3
    assert choice['transit_z_m'] == pytest.approx(travel)
    poses, labels = route_calls[-1]
    contacts = [hand_to_contact(p @ np.linalg.inv(GRASP_TO_EE), .136) for p in poses]
    np.testing.assert_allclose(contacts[0][:3, 3], [*current[:2, 3], travel])
    np.testing.assert_allclose(contacts[1][:3, 3], [.51, .22, travel])
    np.testing.assert_allclose(contacts[2][:3, 3], [.51, .22, .4])
    assert len(checker.calls[-1][0].segments) == 3 and len(held_checks[-1].segments) == 3
    assert labels == ('initial_lift', 'transit', 'waypoint')
    adjusted = b.explicit_adjust_place(choice['candidate_ref'], 5, 0, 10, 0, 0, 3)
    assert adjusted['transit_z_m'] == travel
    assert len(route_calls[-1][0]) == 3
    assert adjusted['contact_center_xyz_m'][2] == pytest.approx(.41)
    assert not b.connector.events
    from types import MethodType
    b.execute_view = MethodType(IntentBackend.execute_view, b)
    result = b.explicit_execute_place(adjusted['candidate_ref'])
    assert result['status'] == 'succeeded' and result['release_required']
    assert b.connector.events == ['trajectory'] * 3
    assert b.connector.command_during_motion == [.03] * 3
    assert b.held_plan is not None and b.grasp_attachment is not None


def test_place_height_reference_grasp_and_transit_rejection_never_move():
    b = Backend()
    b.held_plan.grasp_contract = dict(contact_center_xyz_m=[.4, 0, .3])
    result = prepare(b, height=dict(reference='grasp', value_m=.1),
                     transit_height=dict(reference='grasp', value_m=.3))
    assert result['contact_center_xyz_m'][2] == pytest.approx(.4)
    assert result['transit_z_m'] == pytest.approx(.6)
    with pytest.raises(ValueError, match='must not be below'):
        prepare(b, transit_height=dict(reference='absolute', value_m=.2))
    equal = prepare(b, transit_height=dict(reference='absolute', value_m=.45))
    with pytest.raises(ValueError, match='must not be below'):
        b.explicit_adjust_place(equal['candidate_ref'], 0, 0, 10, 0, 0, 0)
    assert not b.connector.events


def test_place_projection_requires_left_shoulder_instead_of_falling_back_to_wrist(tmp_path):
    b = Backend()
    b.frame.view_id = 'robot0_eye_in_hand'
    b.output_dir = tmp_path
    with pytest.raises(ValueError, match='left shoulder'):
        PlacementMotionMixin._explicit_project_place_hand(b, 'obs0', np.eye(4), caption='test', height_status='test')
