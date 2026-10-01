"""CPU motion contracts: contact-frame heights and ordinary execution guards."""
from dataclasses import replace

import numpy as np
import pytest

from src.tools.motion import planning as motion
from src.tools.grasp.execution import contact_center, plan_explicit_grasp, explicit_grasp_waypoints
from test_graspgen_motion import Connector


OFF = motion.MotionConfig(collision_checks_enabled=False)
# Frozen RoboLab flange/hand frame contract; avoid loading mesh/SciPy tools.
GRASP_TO_EE = np.linalg.inv(np.array([[0., 0., 1., 0.], [-1., 0., 0., 0.],
                                     [0., -1., 0., 0.], [0., 0., 0., 1.]]))
OBJECT = np.array([[.39, -.01, .29], [.41, .01, .32], [.4, 0, .3]])
SCENE = np.array([[1., 1., 0.], [1., 1., .1]])


def hand_pose(*, side=False):
    pose = np.eye(4)
    pose[:3, :3] = ([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]
                       if side else np.diag([1., -1., -1.]))
    pose[:3, 3] = [.4, 0., .436]  # top-down contact centre is Z=.300
    return pose


def build(connector=None, **overrides):
    connector = connector or Connector()
    options = dict(grasp_transform=hand_pose(), grasp_to_ee=GRASP_TO_EE,
        target_points=OBJECT, obstacle_points=SCENE, pre_pick_z_m=.55,
        post_pick_z_m=.75, config=OFF)
    options.update(overrides)
    return connector, plan_explicit_grasp(connector, **options)


def contact(plan, label):
    pose = motion._pose_transform(plan.targets[plan.target_labels.index(label)])
    return contact_center(pose, plan.grasp_to_ee, jaw_offset_m=plan.jaw_offset_m)


def test_top_down_preserves_two_agent_heights_and_flange_calibration():
    connector, plan = build()
    assert plan.target_labels == ('initial_lift', 'high_transit', 'pregrasp', 'grasp', 'lift')
    assert plan.transit_policy == 'explicit'
    assert contact(plan, 'initial_lift')[2] == pytest.approx(.55)
    assert contact(plan, 'high_transit')[2] == pytest.approx(.55)
    assert contact(plan, 'pregrasp')[2] == pytest.approx(.4)
    assert contact(plan, 'grasp')[2] == pytest.approx(.3)
    assert contact(plan, 'lift')[2] == pytest.approx(.75)
    assert plan.targets[-1]['position']['z'] == pytest.approx(.886)
    np.testing.assert_allclose(motion._pose_transform(plan.targets[0])[:3, :3],
                               motion._pose_transform(connector.pose)[:3, :3])
    assert plan.pre_pick_z_m == .55 and plan.post_pick_z_m == .75


def test_explicit_heights_can_be_below_current_and_outgoing_below_incoming():
    _, plan = build(pre_pick_z_m=.49, post_pick_z_m=.45,
                    config=replace(OFF, high_transit_z_m=2., lift_m=1.5))
    assert contact(plan, 'initial_lift')[2] == pytest.approx(.49)
    assert contact(plan, 'high_transit')[2] == pytest.approx(.49)
    assert contact(plan, 'lift')[2] == pytest.approx(.45)


def test_side_grasp_retains_approach_axis_and_aligns_above_pregrasp():
    hand = hand_pose(side=True)
    _, plan = build(grasp_transform=hand, pre_pick_z_m=.65, post_pick_z_m=.8)
    assert plan.target_labels == ('initial_lift', 'high_transit', 'high_pregrasp_align',
                                  'pregrasp', 'grasp', 'lift')
    grasp, pregrasp = contact(plan, 'grasp'), contact(plan, 'pregrasp')
    np.testing.assert_allclose(grasp - pregrasp, .1 * hand[:3, 2], atol=1e-12)
    assert contact(plan, 'high_pregrasp_align')[2] == pytest.approx(.65)
    np.testing.assert_allclose(contact(plan, 'high_pregrasp_align')[:2], pregrasp[:2])
    np.testing.assert_allclose(contact(plan, 'high_transit')[:2], grasp[:2])
    assert contact(plan, 'lift')[2] == pytest.approx(.8)
    for pose in plan.targets[1:]:
        np.testing.assert_allclose(motion._pose_transform(pose)[:3, :3],
                                   (hand @ GRASP_TO_EE)[:3, :3], atol=1e-12)


def test_arbitrary_translating_calibration_and_configurable_contact_offset():
    calibration = GRASP_TO_EE.copy()
    calibration[:3, 3] = [.03, -.02, .041]
    _, plan = build(grasp_to_ee=calibration, jaw_offset_m=.12)
    assert contact(plan, 'grasp')[2] == pytest.approx(.316)
    assert contact(plan, 'high_transit')[2] == pytest.approx(.55)
    assert contact(plan, 'lift')[2] == pytest.approx(.75)


def test_resume_after_correction_uses_measured_pregrasp_and_keeps_outgoing_height():
    connector, first = build()
    connector.pose = first.targets[first.target_labels.index('pregrasp')]
    connector.q = [.02] * 7
    corrected = first.grasp_transform.copy()
    corrected[:3, 3] += [.01, -.015, .02]
    _, plan = build(connector, grasp_transform=corrected,
        pre_pick_z_m=first.pre_pick_z_m, post_pick_z_m=first.post_pick_z_m,
        resume_from_pregrasp=True)
    assert plan.target_labels == ('pregrasp', 'grasp', 'lift')
    assert plan.start_joints == tuple(connector.q)
    assert contact(plan, 'grasp')[2] == pytest.approx(.32)
    assert contact(plan, 'lift')[2] == pytest.approx(.75)
    assert connector.ik.calls[-3][0] == connector.pose
    assert plan.resume_from_pregrasp is True


def test_contact_grasp_has_no_planned_lift():
    _, plan = build(lift_after_grasp=False)
    assert plan.target_labels[-1] == 'grasp'
    assert 'lift' not in plan.target_labels
    _, resumed = build(lift_after_grasp=False, resume_from_pregrasp=True)
    assert resumed.target_labels == ('pregrasp', 'grasp')


@pytest.mark.parametrize('name,value', [('pre_pick_z_m', float('nan')),
    ('post_pick_z_m', float('inf')), ('pre_pick_z_m', True), ('jaw_offset_m', 0.),
    ('jaw_offset_m', -.1), ('open_width_m', .09), ('lift_after_grasp', 1)])
def test_invalid_requests_never_reach_the_planner(name, value):
    connector = Connector()
    with pytest.raises(ValueError):
        build(connector, **{name: value})
    assert not connector.ik.calls and not connector.events


def test_planning_failure_is_labeled_and_never_retried_at_other_heights():
    connector = Connector()
    connector.ik.reject = True
    with pytest.raises(motion.MotionPlanningError) as caught:
        build(connector)
    assert caught.value.planning_feedback['segment'] == 'initial_lift'
    assert len(connector.ik.calls) == 1


def test_existing_executor_guard_and_close_before_independent_lift():
    connector, plan = build()
    result = motion.execute_grasp(connector, plan)
    assert connector.events == ['open', 'trajectory', 'trajectory', 'trajectory',
                                'trajectory', 'close', 'trajectory']
    assert result['success_verified'] is False
    assert contact_center(motion._pose_transform(connector.pose), GRASP_TO_EE)[2] == pytest.approx(.75)
    with pytest.raises(motion.MotionPlanningError, match='consumed'):
        motion.execute_grasp(connector, plan)
    connector, plan = build()
    connector.q = [.1] * 7
    with pytest.raises(motion.MotionPlanningError, match='stale'):
        motion.execute_grasp(connector, plan)
    assert not connector.events


def test_guarded_payload_collision_stops_before_planning():
    connector = Connector()
    with pytest.raises(motion.MotionPlanningError, match='payload'):
        build(connector, config=motion.MotionConfig(), obstacle_points=[[.4, 0., .4]])
    assert not connector.ik.calls


@pytest.mark.parametrize('outgoing', [.3, .295])
def test_guarded_no_rise_uses_ordinary_payload_check_without_a_support_exception(outgoing):
    _, plan = build(post_pick_z_m=outgoing, config=motion.MotionConfig())
    assert contact(plan, 'lift')[2] == pytest.approx(outgoing)


def test_injected_observed_chain_gets_full_labels_and_exact_height_targets():
    received = []
    def planner(connector, targets, obstacles, config, validator, *, target_labels):
        received.append((targets, obstacles.copy(), target_labels, validator))
        return motion._plan(connector, targets, obstacles, config, validator)
    _, plan = build(chain_planner=planner)
    assert received[0][2] == plan.target_labels
    np.testing.assert_array_equal(received[0][1], SCENE)
    assert received[0][3] is None
    assert contact(plan, 'lift')[2] == pytest.approx(.75)


def test_recorded_cgn_roundtrip_difference_does_not_reject_planned_chain():
    # CGN rotation whose quaternion roundtrip differs by
    # 3.05e-8 even though it represents the same intended orientation.
    grasp = hand_pose()
    grasp[:3, :3] = [[.867863406787209, .4923649152510797, -.06625623879059905],
                    [.4615427129854336, -.7497181107842327, .4742374169831973],
                    [.18382436927493673, -.44215338600992515, -.8779003657944981]]
    def planner(connector, targets, obstacles, config, validator, *, target_labels):
        result = motion._plan(connector, targets, obstacles, config, validator)
        assert any(not np.allclose(motion._pose_transform(p), t, rtol=0, atol=1e-8)
                   for p, t in zip(result[1], targets))
        return result
    connector, plan = build(grasp_transform=grasp, chain_planner=planner)
    assert 'grasp' in plan.target_labels and 'lift' in plan.target_labels
    assert not connector.events


@pytest.mark.parametrize('change', ['missing', 'jump', 'start'])
def test_execution_uses_planner_waypoint_count_start_and_continuity(change):
    def planner(connector, targets, obstacles, config, validator, *, target_labels):
        segments, poses, joints, clearance, valid = motion._plan(
            connector, targets, obstacles, config, validator)
        if change == 'missing':
            segments, poses = segments[:-1], poses[:-1]
        elif change == 'jump':
            segments[-1]['waypoints'][-1]['positions'] = [2.] * 7
        else:
            joints = tuple(.1 for _ in joints)
        return segments, poses, joints, clearance, valid
    connector = Connector()
    _, plan = build(connector, chain_planner=planner)
    assert plan is not None
    assert not connector.events


def test_injected_planner_rejection_does_not_fall_back_and_external_validator_runs():
    def reject(*args, **kwargs):
        raise motion.MotionPlanningError('observed route rejected')
    connector = Connector()
    with pytest.raises(motion.MotionPlanningError, match='observed route rejected'):
        build(connector, chain_planner=reject)
    assert not connector.ik.calls
    validators = []
    def planner(connector, targets, obstacles, config, validator, *, target_labels):
        return motion._plan(connector, targets, obstacles, config, validator)
    def validate(segments, joints, obstacles):
        validators.append((segments, joints, obstacles))
        return False
    with pytest.raises(motion.MotionPlanningError, match='external trajectory validator'):
        build(config=motion.MotionConfig(), chain_planner=planner, trajectory_validator=validate)
    assert len(validators) == 1 and len(validators[0][0]) == 5


def test_waypoint_helper_is_pure_and_does_not_mutate_inputs():
    current = motion._pose_transform(Connector().pose)
    grasp, calibration = hand_pose(), GRASP_TO_EE.copy()
    originals = [x.copy() for x in (current, grasp, calibration)]
    targets, labels = explicit_grasp_waypoints(current_ee=current, grasp_transform=grasp,
        grasp_to_ee=calibration, pre_pick_z_m=.55, post_pick_z_m=.75)
    assert len(targets) == len(labels) == 5
    for actual, expected in zip((current, grasp, calibration), originals):
        np.testing.assert_array_equal(actual, expected)
