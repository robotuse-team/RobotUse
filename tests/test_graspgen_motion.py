"""CPU-only contracts; fake planner/executor, not LIBERO success evidence."""
import unittest
import json
from unittest.mock import patch
import numpy as np
from src.tools.motion import planning as m


class FakeIK:
    collision_checks_enabled = False
    trajectory_needs_joint_reverse = False
    def __init__(self):
        self.calls = []
        self.reject = False
        self.jump = False
    def plan_linear(self, start, end, *, seed_joints):
        self.calls.append((start, end, list(seed_joints)))
        if self.reject:
            return None
        q = np.asarray(seed_joints) + (1.0 if self.jump else .01)
        return {"waypoints": [{"positions": list(seed_joints)}, {"positions": q.tolist()}],
                "target": end}


class Connector:
    def __init__(self):
        self.ik = FakeIK()
        self.q = [0.] * 7
        t = np.eye(4)
        t[:3,:3] = np.diag([1,-1,-1])
        t[:3,3] = [.4,0,.6]
        self.pose = m.transform_to_pose(t)
        self.events = []
    def get_ee_pose(self): return self.pose
    def get_observation(self): return {"arms": [{"joint_state": {"positions": self.q}}]}
    def execute_trajectory(self, segment):
        self.events.append("trajectory")
        self.q = segment["waypoints"][-1]["positions"]
        self.pose = segment["target"]
    def open_gripper(self, *, settle_steps): self.events.append("open")
    def close_gripper(self, *, settle_steps): self.events.append("close")
    def get_gripper_fraction(self): return .18


class MotionTests(unittest.TestCase):
    def setUp(self):
        self.c = Connector()
        self.g = np.eye(4)
        self.g[:3,:3] = np.diag([1,-1,-1])
        self.g[:3,3] = [.4,0,.4]
        self.obj = np.array([[.39,-.01,.29],[.41,.01,.32],[.4,0,.3]])
        self.scene = np.array([[1,1,0],[1,1,.1]])
    def plan(self, **kw):
        args = dict(grasp_transform=self.g, target_points=self.obj, obstacle_points=self.scene,
                    grasp_to_ee=m.libero_panda_grasp_to_grip_site())
        args.update(kw)
        return m.plan_grasp(self.c, **args)
    def test_pinned_public_tcp_axes_and_local_offset(self):
        t = m.libero_panda_grasp_to_grip_site()
        np.testing.assert_allclose(t[:3,:3] @ [1,0,0], [1,0,0])
        np.testing.assert_allclose((self.g @ t)[:3,3], [.4,0,.303])
    def test_rotations_round_trip(self):
        for axis in range(3):
            t = np.eye(4)
            t[:3,:3] *= -1
            t[axis,axis] = 1
            np.testing.assert_allclose(m._pose_transform(m.transform_to_pose(t)), t, atol=1e-10)
    def test_reject_invalid_transforms_clouds_frames(self):
        for g in (np.eye(3), np.zeros((4,4)), np.eye(4)*float("nan")):
            with self.assertRaises(ValueError): self.plan(grasp_transform=g)
        with self.assertRaises(ValueError): self.plan(obstacle_points=np.empty((0,3)))
        with self.assertRaises(ValueError): self.plan(frame="camera")
        with self.assertRaises(ValueError): self.plan(frame="base")
        with self.assertRaises(ValueError): m.MotionConfig(clearance_m=-1)
    def test_world_base_equivalence(self):
        a = self.plan()
        w = np.eye(4); w[:3,3] = [.2,.3,.1]
        local = np.linalg.inv(w) @ self.g
        b = self.plan(grasp_transform=local, target_points=self.obj-w[:3,3],
                      obstacle_points=self.scene-w[:3,3],frame="base",world_from_base=w)
        self.assertEqual(a.targets, b.targets)
    def test_chained_real_planner_calls_and_approach_axis(self):
        p = self.plan()
        self.assertEqual(len(self.c.ik.calls), 3)
        self.assertEqual(self.c.ik.calls[1][2], [.01]*7)
        self.assertAlmostEqual(p.targets[0]["position"]["z"], .403)
        self.assertAlmostEqual(p.targets[1]["position"]["z"], .303)
        self.assertFalse(p.trajectory_validated)
    def test_planner_rejection_and_discontinuity(self):
        self.c.ik.reject = True
        with self.assertRaises(m.MotionPlanningError): self.plan()
        self.c.ik.reject = False; self.c.ik.jump = True
        with self.assertRaises(m.MotionPlanningError): self.plan()
    def test_observed_capsule_detects_between_endpoints(self):
        self.assertLess(m.observed_path_clearance([[0,0,0],[1,0,0]], [[.51,.01,0]],radius_m=.02),0)
        with self.assertRaises(m.MotionPlanningError):
            self.plan(obstacle_points=[[.4,0,.38]])
        self.assertEqual(len(self.c.ik.calls),0)
    def test_payload_collision(self):
        with self.assertRaises(m.MotionPlanningError):
            self.plan(obstacle_points=[[.39,-.01,.35]])
    def test_execution_requires_explicit_ack_and_single_use(self):
        p = self.plan()
        with self.assertRaises(m.MotionPlanningError): m.execute_grasp(self.c,p)
        self.assertEqual(self.c.events,[])
        result = m.execute_grasp(self.c,p,allow_partial_safety=True)
        self.assertEqual(self.c.events,["open","trajectory","trajectory","close","trajectory"])
        self.assertEqual(result["gripper_fraction"],.18)
        self.assertFalse(result["success_verified"])
        self.assertFalse(result["full_arm_safety_certified"])
        with self.assertRaises(m.MotionPlanningError): m.execute_grasp(self.c,p,allow_partial_safety=True)
    def test_stale_and_foreign_plans(self):
        p = self.plan(); self.c.q = [.1]*7
        with self.assertRaises(m.MotionPlanningError): m.execute_grasp(self.c,p,allow_partial_safety=True)
        with self.assertRaises(m.MotionPlanningError): m.execute_grasp(Connector(),p,allow_partial_safety=True)
    def test_external_validator_can_reject_actual_segments(self):
        calls=[]
        def reject(segments,joints,obstacles):
            calls.append(segments)
            return False
        with self.assertRaises(m.MotionPlanningError): self.plan(trajectory_validator=reject)
        self.assertEqual(len(calls[0]),3)
    def test_observed_placement_after_grasp_only(self):
        p = self.plan()
        dest = np.array([[.6,0,.2],[.59,-.01,.2],[.61,.01,.2]])
        with self.assertRaises(m.MotionPlanningError):
            m.plan_place(self.c,grasp_plan=p,destination_points=dest,obstacle_points=self.scene)
        m.execute_grasp(self.c,p,allow_partial_safety=True)
        place = m.plan_place(self.c,grasp_plan=p,destination_points=dest,obstacle_points=self.scene)
        self.assertAlmostEqual(place.targets[2]["position"]["x"],.6)
        self.assertAlmostEqual(place.targets[2]["position"]["z"],.303+.2+.015-.29)
        self.c.events=[]
        result=m.execute_place(self.c,place,allow_partial_safety=True)
        self.assertEqual(self.c.events,["trajectory","trajectory","trajectory","open","trajectory"])
        self.assertFalse(result["success_verified"])
    def test_no_unobserved_destination_hole_center(self):
        p = self.plan(); m.execute_grasp(self.c,p,allow_partial_safety=True)
        rim = [[.5,0,.2],[.7,0,.2],[.6,.1,.2],[.6,-.1,.2]]
        with self.assertRaises(m.MotionPlanningError):
            m.plan_place(self.c,grasp_plan=p,destination_points=rim,obstacle_points=self.scene)
    def test_bad_execution_pose_stops_before_close(self):
        p=self.plan()
        original=self.c.execute_trajectory
        def wrong(segment):
            original(segment)
            self.c.pose={"position": {"x": 9, "y": 0, "z": 0},
                         "rotation": {"x": 0, "y": 0, "z": 0, "w": 1}}
        self.c.execute_trajectory=wrong
        with self.assertRaises(m.MotionPlanningError):
            m.execute_grasp(self.c,p,allow_partial_safety=True)
        self.assertNotIn("close",self.c.events)
        self.assertTrue(p.consumed)
    def test_legacy_public_getter_frame_adapter(self):
        # Synthetic known public = planner @ rotation/translation. No sim read.
        planner=m._pose_transform(self.c.get_ee_pose())
        planner_to_public=np.eye(4)
        planner_to_public[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
        planner_to_public[2,3]=-.107
        self.c.pose=m.transform_to_pose(planner @ planner_to_public)
        wrapped=m.PlannerFrameConnector(self.c,public_to_planner=np.linalg.inv(planner_to_public))
        np.testing.assert_allclose(m._pose_transform(wrapped.get_ee_pose()),planner,atol=1e-12)
        self.assertIs(wrapped.ik,self.c.ik)
        self.assertEqual(wrapped.get_observation(),self.c.get_observation())
    def test_measured_robot_calibration_requires_matching_fk(self):
        base=np.eye(4); base[0,3]=-.6
        hand=np.eye(4); hand[:3,3]=[-.15,0,.4]
        grip=hand.copy(); grip[:3,:3]=[[0,1,0],[-1,0,0],[0,0,1]]
        left=hand.copy(); left[1,3]=-.04
        right=hand.copy(); right[1,3]=.04
        public=np.linalg.inv(base) @ hand; public[2,3]-=.010
        fk=np.linalg.inv(base) @ hand; fk[2,3]+=.097
        probe={"public_ee":public,"robot_bodies":{"robot0_base":base,
            "robot0_right_hand":hand,"gripper0_right_gripper":grip,
            "gripper0_leftfinger":left,"gripper0_rightfinger":right}}
        calibration=m.calibrate_libero_panda_from_robot_probe(probe,planner_fk_pose=m.transform_to_pose(fk))
        np.testing.assert_allclose(calibration.grasp_to_ee[:3,:3],[[0,-1,0],[1,0,0],[0,0,1]],atol=1e-12)
        np.testing.assert_allclose(calibration.grasp_to_ee[:3,3],[0,0,.097],atol=1e-12)
        np.testing.assert_allclose(calibration.public_to_planner[:3,:3],np.eye(3),atol=1e-12)
        np.testing.assert_allclose(calibration.public_to_planner[:3,3],[0,0,.107],atol=1e-12)
        fk[0,3]+=.03
        with self.assertRaises(m.MotionPlanningError):
            m.calibrate_libero_panda_from_robot_probe(probe,planner_fk_pose=m.transform_to_pose(fk))
    def test_connector_base_is_explicit_native_identity(self):
        self.assertEqual(self.plan().targets,self.plan(frame="connector_base").targets)
        with self.assertRaises(ValueError): self.plan(frame="connector_base",world_from_base=np.eye(4))
    def test_same_wrapper_corrects_plan_and_execution_endpoints(self):
        correction=np.eye(4); correction[2,3]=.107
        planner_start=m._pose_transform(self.c.pose)
        self.c.pose=m.transform_to_pose(planner_start @ np.linalg.inv(correction))
        original=self.c.execute_trajectory
        def execute_in_planner_frame(segment):
            original(segment)
            self.c.pose=m.transform_to_pose(m._pose_transform(segment["target"]) @ np.linalg.inv(correction))
        self.c.execute_trajectory=execute_in_planner_frame
        wrapped=m.PlannerFrameConnector(self.c,public_to_planner=correction)
        bridge=m.GraspGenMotionBridge(wrapped,grasp_to_ee=m.libero_panda_grasp_to_grip_site(),allow_partial_safety=True)
        plan=bridge.validate(self.g,self.obj,self.scene)
        np.testing.assert_allclose(m._pose_transform(self.c.ik.calls[0][0]),planner_start,atol=1e-12)
        self.assertIs(plan.connector,wrapped)
        self.assertEqual(bridge.execute(plan)["status"],"executed")
        destination=[[.6,0,.2],[.59,-.01,.2],[.61,.01,.2]]
        self.assertEqual(bridge.place(destination,plan,self.scene)["status"],"released")
    def test_initial_resting_support_separates_on_lift(self):
        # Measured lower payload samples touch the table; no scene deletion.
        obj=np.array([[0,0,0],[.01,0,0],[0,0,.03]])
        table=np.array([[0,0,0],[.01,0,-.001],[.02,0,0]])
        m._payload_clearance(obj,[np.zeros(3),np.array([0,0,.15])],table,.008,allow_initial_support=True)
        with self.assertRaises(m.MotionPlanningError):
            m._payload_clearance(obj,[np.zeros(3),np.array([0,0,.15])],table,.008)
    def test_support_exception_never_hides_side_or_new_obstacle(self):
        obj=np.array([[0,0,0],[0,0,.03]])
        for obstacle in ([[0,0,.08]],[[.003,0,.03]],[[0,0,.002]]):
            with self.assertRaises(m.MotionPlanningError):
                m._payload_clearance(obj,[np.zeros(3),np.array([0,0,.15])],np.array(obstacle),.008,allow_initial_support=True)
    def test_support_exception_refuses_nonseparating_or_later_motion(self):
        obj=np.array([[0,0,0]])
        for shift in ([.02,0,.15],[0,0,-.15],[0,0,.001]):
            with self.assertRaises(m.MotionPlanningError):
                m._payload_clearance(obj,[np.zeros(3),np.array(shift)],obj,.008,allow_initial_support=True)
        with self.assertRaises(m.MotionPlanningError):
            m._payload_clearance(obj,[np.array([0,0,.1]),np.array([0,0,.2])],obj,.008,allow_initial_support=True)
    def test_explicit_collision_off_skips_all_filters_and_ack(self):
        config=m.MotionConfig(collision_checks_enabled=False)
        def forbidden(*args,**kwargs): raise AssertionError("collision filter called")
        with patch.object(m,"observed_path_clearance",forbidden), patch.object(m,"_payload_clearance",forbidden):
            plan=self.plan(obstacle_points=self.obj,config=config,trajectory_validator=forbidden)
            self.assertIsNone(plan.clearance_m)
            self.assertFalse(plan.collision_checks_enabled)
            result=m.execute_grasp(self.c,plan)
            self.assertEqual(result["status"],"executed")
            self.assertFalse(result["collision_checks_enabled"])
            dest=[[.6,0,.2],[.59,-.01,.2],[.61,.01,.2]]
            place=m.plan_place(self.c,grasp_plan=plan,destination_points=dest,
                               obstacle_points=dest,config=config,trajectory_validator=forbidden)
            result=m.execute_place(self.c,place)
            self.assertEqual(result["status"],"released")
            self.assertFalse(result["collision_checks_enabled"])
        self.assertEqual(len(self.c.ik.calls),7)
    def test_collision_off_preserves_ik_and_numerical_rejections(self):
        config=m.MotionConfig(collision_checks_enabled=False)
        self.c.ik.reject=True
        with self.assertRaises(m.MotionPlanningError): self.plan(config=config)
        with self.assertRaises(ValueError): self.plan(config=config,grasp_transform=np.zeros((4,4)))
        with self.assertRaises(ValueError): m.MotionConfig(collision_checks_enabled=0)
    def test_collision_off_directed_native_never_calls_self_aware_fallback(self):
        from types import SimpleNamespace
        calls=[]
        def directed(**kwargs):
            calls.append(kwargs)
            return False,None,"no_ik"
        class Native:
            _robot_file="franka.yml"
            def _import_impl(self): return SimpleNamespace(plan_directed_linear=directed)
            def _resolve_seed(self,q): return np.array(q)
            def _pose_for_curobo(self,p,arm): return p,arm
            def plan_linear(self,*a,**k): raise AssertionError("self-aware fallback route")
        Native.__name__="CuRoboBackend"; Native.__module__="gap.connector.ik"
        wrapped=m._collision_disabled_planner(SimpleNamespace(ik=Native()))
        self.assertIsNone(wrapped.ik.plan_linear("start","end",seed_joints=[0.]*7))
        self.assertEqual(calls[0]["start_pose"],("start",0))
        self.assertEqual(calls[0]["target_pose"],("end",0))
        self.assertEqual(calls[0]["orientation_mode"],"TARGET_AT_END")
    def test_off_execution_records_large_errors_without_holding(self):
        config=m.MotionConfig(collision_checks_enabled=False)
        plan=self.plan(config=config)
        original=self.c.execute_trajectory
        def poor_tracking(segment):
            original(segment)
            self.c.q=[v+.2 for v in self.c.q]
            t=m._pose_transform(segment["target"])
            t[0,3]+=.05
            t[:3,:3]=np.eye(3)
            self.c.pose=m.transform_to_pose(t)
        self.c.execute_trajectory=poor_tracking
        result=m.execute_grasp(self.c,plan)
        self.assertEqual(self.c.events,["open","trajectory","trajectory","close","trajectory"])
        self.assertFalse(result["success_verified"])
        self.assertEqual(len(result["execution_diagnostics"]),3)
        self.assertAlmostEqual(result["execution_diagnostics"][0]["joint_max_error_rad"],.2)
        self.assertAlmostEqual(result["execution_diagnostics"][0]["position_error_m"],.05)
        self.assertFalse(result["execution_diagnostics"][0]["thresholds_enforced"])
        # Actual held rotation differs, but off profile must still plan placement.
        place=m.plan_place(self.c,grasp_plan=plan,destination_points=[[.6,0,.2],[.59,0,.2],[.61,0,.2]],
                           obstacle_points=self.scene,config=config)
        self.assertEqual(m.execute_place(self.c,place)["status"],"released")
    def test_off_execution_propagates_real_command_exception(self):
        plan=self.plan(config=m.MotionConfig(collision_checks_enabled=False))
        def broken(segment): raise RuntimeError("real command failed")
        self.c.execute_trajectory=broken
        with self.assertRaisesRegex(RuntimeError,"real command failed"):
            m.execute_grasp(self.c,plan)
        self.assertNotIn("close",self.c.events)
    def test_off_execution_still_rejects_nonfinite_measurements(self):
        plan=self.plan(config=m.MotionConfig(collision_checks_enabled=False))
        def invalid(segment): self.c.q=[float("nan")]*7
        self.c.execute_trajectory=invalid
        with self.assertRaisesRegex(m.MotionPlanningError,"invalid execution joint"):
            m.execute_grasp(self.c,plan)
    def test_missing_calibration_rejected(self):
        with self.assertRaises(m.MotionPlanningError): self.plan(grasp_to_ee=None)
    def test_empty_gripper_is_not_held(self):
        self.c.get_gripper_fraction = lambda: 0.0
        p=self.plan()
        result=m.execute_grasp(self.c,p,allow_partial_safety=True)
        self.assertEqual(result["status"],"failed")
        self.assertEqual(result["held_state"],"not_held")
        self.assertFalse(p.grasp_executed)

    def test_invalid_gripper_after_motion_preserves_raw_sensor_evidence(self):
        for raw in (1.3, -.01, float("nan"), float("inf"), "0.3", True, 10**400):
            with self.subTest(raw=raw):
                self.setUp()
                self.c.get_gripper_fraction = lambda: raw
                plan = self.plan()
                with self.assertRaisesRegex(m.MotionPlanningError, "invalid gripper") as raised:
                    m.execute_grasp(self.c, plan, allow_partial_safety=True)
                evidence = raised.exception.evidence
                measurement = evidence["gripper_measurement"]
                self.assertTrue(evidence["motion_completed"])
                self.assertEqual(evidence["execution_stage"], "post_lift")
                self.assertEqual(len(evidence["execution_diagnostics"]), 3)
                self.assertEqual(measurement["validity"], "invalid")
                self.assertEqual(measurement["source"], "connector.get_gripper_fraction")
                self.assertEqual(measurement["stage"], "post_lift")
                self.assertEqual(measurement["raw_value"], str(raw) if isinstance(raw, float) and not np.isfinite(raw) else raw)
                self.assertIsNone(measurement["value"])
                self.assertFalse(plan.grasp_executed)
                json.dumps(evidence, allow_nan=False)

    def test_gripper_getter_exception_preserves_stage_without_exception_text(self):
        def broken(): raise ValueError("sensitive driver internals")
        self.c.get_gripper_fraction = broken
        with self.assertRaises(m.MotionPlanningError) as raised:
            m.execute_grasp(self.c, self.plan(), allow_partial_safety=True)
        measurement = raised.exception.evidence["gripper_measurement"]
        self.assertEqual(measurement["validity"], "invalid")
        self.assertEqual(measurement["error"], "ValueError")
        self.assertNotIn("sensitive", json.dumps(raised.exception.evidence))

    def test_missing_gripper_is_unknown_and_never_verified(self):
        for getter in (None, lambda: None):
            self.setUp()
            self.c.get_gripper_fraction = getter
            result = m.execute_grasp(self.c, self.plan(), allow_partial_safety=True)
            self.assertEqual(result["held_state"], "unknown")
            self.assertEqual(result["gripper_measurement"]["validity"], "missing")
            self.assertFalse(result["success_verified"])

    def test_empty_boundary_matches_verifier(self):
        for raw in (0.015, 0.02):
            self.setUp()
            self.c.get_gripper_fraction = lambda: raw
            plan = self.plan()
            result = m.execute_grasp(self.c, plan, allow_partial_safety=True)
            self.assertEqual(result["held_state"], "not_held")
            self.assertEqual(result["gripper_measurement"]["validity"], "valid")
            self.assertFalse(plan.grasp_executed)
    def high_config(self, **kw):
        return m.MotionConfig(collision_checks_enabled=False, transit_policy="high", **kw)

    def test_high_policy_validation_and_inert_legacy_height(self):
        for kwargs in ({"transit_policy": "unknown"}, {"high_transit_z_m": 0},
                       {"high_transit_z_m": float("nan")}, {"high_transit_z_m": -1}):
            with self.assertRaises(ValueError): m.MotionConfig(**kwargs)
        legacy = self.plan(config=m.MotionConfig(high_transit_z_m=.9))
        self.assertEqual(len(legacy.targets), 3)
        self.assertEqual(legacy.target_labels, ("pregrasp", "grasp", "lift"))
        self.assertEqual(legacy.segment_labels, legacy.target_labels)
        self.assertEqual(legacy.transit_policy, "legacy")
        self.assertIsNone(legacy.high_transit_z_m)

    def test_high_grasp_up_before_translation_or_rotation(self):
        current = np.eye(4)
        current[:3,3] = [.15, -.2, .2]
        self.c.pose = m.transform_to_pose(current)
        original = self.g.copy()
        plan = self.plan(config=self.high_config(high_transit_z_m=.55))
        self.assertEqual(plan.target_labels,
                         ("initial_lift", "high_transit", "pregrasp", "grasp", "lift"))
        self.assertEqual(plan.segment_labels, plan.target_labels)
        self.assertEqual(plan.high_transit_z_m, .55)
        ts = [m._pose_transform(t) for t in plan.targets]
        np.testing.assert_allclose(ts[0][:2,3], current[:2,3])
        np.testing.assert_allclose(ts[0][:3,:3], current[:3,:3])
        self.assertAlmostEqual(ts[0][2,3], .55)
        np.testing.assert_allclose(ts[1][:2,3], ts[3][:2,3])
        np.testing.assert_allclose(ts[1][:3,:3], ts[3][:3,:3])
        self.assertAlmostEqual(ts[1][2,3], .55)
        np.testing.assert_allclose(ts[2][:2,3], ts[1][:2,3])
        np.testing.assert_allclose(ts[3], original @ plan.grasp_to_ee)
        np.testing.assert_allclose(self.g, original)
        np.testing.assert_allclose(ts[4][:2,3], ts[3][:2,3])
        self.assertAlmostEqual(ts[4][2,3], .55)
        for i, call in enumerate(self.c.ik.calls):
            np.testing.assert_allclose(call[2], [i*.01]*7)
        result = m.execute_grasp(self.c, plan)
        self.assertEqual(self.c.events, ["open"] + ["trajectory"]*4 + ["close", "trajectory"])
        self.assertEqual(len(result["execution_diagnostics"]), 5)

    def test_high_tilted_prediction_keeps_geometry_and_descends_vertically(self):
        angle = .4
        ry = np.array([[np.cos(angle),0,np.sin(angle)], [0,1,0],
                       [-np.sin(angle),0,np.cos(angle)]])
        self.g[:3,:3] = ry @ self.g[:3,:3]
        original = self.g.copy()
        plan = self.plan(config=self.high_config())
        self.assertEqual(len(plan.targets), 6)
        self.assertEqual(plan.target_labels[2], "high_pregrasp_align")
        ts = [m._pose_transform(t) for t in plan.targets]
        np.testing.assert_allclose(ts[2][:2,3], ts[3][:2,3])
        np.testing.assert_allclose(ts[4], original @ plan.grasp_to_ee)
        np.testing.assert_allclose(ts[3][:3,3], ts[4][:3,3]-.1*original[:3,2])
        np.testing.assert_allclose(self.g, original)
        self.assertAlmostEqual(ts[0][2,3], .6)  # never lower an already-high TCP
        self.assertAlmostEqual(ts[1][2,3], .6)
        self.assertAlmostEqual(ts[2][2,3], .6)
        m.execute_grasp(self.c, plan)
        self.assertEqual(self.c.events, ["open"] + ["trajectory"]*5 + ["close", "trajectory"])

    def test_high_place_preserves_actual_orientation_and_destination_formula(self):
        config = self.high_config(high_transit_z_m=.65)
        grasp = self.plan(config=config)
        m.execute_grasp(self.c, grasp)
        current = np.eye(4)
        current[:3,3] = [.4, 0, .3]
        self.c.pose = m.transform_to_pose(current)  # off-policy orientation drift
        dest = [[.6,0,.2],[.59,-.01,.2],[.61,.01,.2]]
        legacy = m.plan_place(self.c, grasp_plan=grasp, destination_points=dest,
                              obstacle_points=self.scene,
                              config=m.MotionConfig(collision_checks_enabled=False))
        place = m.plan_place(self.c, grasp_plan=grasp, destination_points=dest,
                             obstacle_points=self.scene, config=config)
        self.assertEqual(place.target_labels, ("lift", "high_transit", "release", "retreat"))
        self.assertEqual(place.segment_labels, place.target_labels)
        ts = [m._pose_transform(t) for t in place.targets]
        for t in ts: np.testing.assert_allclose(t[:3,:3], current[:3,:3])
        np.testing.assert_allclose(ts[0][:2,3], current[:2,3])
        np.testing.assert_allclose(ts[1][:2,3], ts[2][:2,3])
        np.testing.assert_allclose(ts[3][:2,3], ts[2][:2,3])
        for i in (0,1,3): self.assertAlmostEqual(ts[i][2,3], .65)
        self.assertEqual(place.targets[2]["position"], legacy.targets[2]["position"])
        self.c.events = []
        m.execute_place(self.c, place)
        self.assertEqual(self.c.events, ["trajectory"]*3 + ["open", "trajectory"])

    def test_high_policy_keeps_collision_filters_off(self):
        def forbidden(*args, **kwargs): raise AssertionError("collision filter called")
        with patch.object(m, "observed_path_clearance", forbidden), patch.object(m, "_payload_clearance", forbidden):
            config = self.high_config()
            grasp = self.plan(config=config, obstacle_points=self.obj, trajectory_validator=forbidden)
            m.execute_grasp(self.c, grasp)
            place = m.plan_place(self.c, grasp_plan=grasp,
                destination_points=[[.6,0,.2],[.59,0,.2],[.61,0,.2]],
                obstacle_points=self.obj, config=config, trajectory_validator=forbidden)
            m.execute_place(self.c, place)
        self.assertFalse(place.collision_checks_enabled)
        self.assertIsNone(place.clearance_m)

    def test_semantic_boundary_rejects_invalid_metadata_before_commands(self):
        for labels in (("grasp",), ("grasp", "grasp", "lift")):
            plan = self.plan(config=m.MotionConfig(collision_checks_enabled=False))
            plan.target_labels = labels
            with self.assertRaisesRegex(m.MotionPlanningError, "semantic"):
                m.execute_grasp(self.c, plan)
            self.assertFalse(plan.consumed)
            self.assertEqual(self.c.events, [])

    def test_adapter_requires_fresh_scene(self):
        bridge=m.GraspGenMotionBridge(self.c,allow_partial_safety=True,
                                      grasp_to_ee=m.libero_panda_grasp_to_grip_site())
        p=bridge.validate(self.g,self.obj,self.scene); bridge.execute(p)
        with self.assertRaises(m.MotionPlanningError): bridge.place([[.6,0,.2]],p)


if __name__ == "__main__": unittest.main()
