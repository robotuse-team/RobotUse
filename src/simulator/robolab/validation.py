"""Deterministic native checks; public RGB-D selects geometry, no LLM session."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


def json_value(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot record {type(value).__name__}")


def write_result(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, default=json_value)
        stream.write("\n")


class ValidationRecorder:
    """Append native tool evidence without adding an agent or recording policy."""

    def __init__(self, output):
        self.path = Path(output) / "tool-events.jsonl"

    def event(self, kind, payload):
        row = dict(time_utc=datetime.now(timezone.utc).isoformat(), kind=kind, payload=payload)
        with self.path.open("a") as stream:
            stream.write(json.dumps(row, default=json_value) + "\n")

    @contextmanager
    def active(self, kind, **payload):
        self.event(kind + "_started", payload)
        try:
            yield
        finally:
            self.event(kind + "_finished", payload)

    def register_plan(self, plan, **payload):
        self.event("motion_plan", {**payload, "segment_count": len(plan.segments)})


def check_connector(connector):
    """Use the unchanged connector acceptance thresholds and native robot mesh."""
    from gap_core.types import matrix_to_pose, pose_to_matrix
    from scipy.spatial.transform import Rotation
    from src.simulator.robolab.collision import RoboLabPathCollision

    env = connector.env
    checker = RoboLabPathCollision(connector)
    assert len(checker.meshes) >= 20
    reference = env.joints().copy()
    origin = pose_to_matrix(connector.get_ee_pose())
    fk_checks = []
    for joint, delta in ((6, .7), (6, -.7), (4, .3), (1, .3)):
        q = reference.copy()
        q[joint] += delta
        checker.native.set_joints(q, env.gripper_angle())
        actual = checker.native.body_matrix("base_link")
        predicted = connector.ik.model.fk(q)
        row = dict(joint_index=joint, delta_rad=delta,
            position_error_m=float(np.linalg.norm(actual[:3, 3] - predicted[:3, 3])),
            rotation_error_rad=float(Rotation.from_matrix(actual[:3, :3] @ predicted[:3, :3].T).magnitude()))
        assert row["position_error_m"] < .0001 and row["rotation_error_rad"] < .001, row
        fk_checks.append(row)
    env.refresh_camera_obs()
    fixed = env.camera_optical_matrix("agentview").copy()
    hand_from_camera = np.linalg.inv(origin) @ env.camera_optical_matrix("robot0_eye_in_hand")
    checks = []
    for delta, angle in (([.01, 0, 0], 0), ([0, .05, 0], 0), ([0, 0, -.01], 0),
                         ([0, 0, 0], 15), ([0, 0, 0], -15)):
        before = pose_to_matrix(connector.get_ee_pose())
        target = before.copy()
        target[:3, 3] += delta
        target[:3, :3] = Rotation.from_euler("z", angle, degrees=True).as_matrix() @ before[:3, :3]
        steps = env._sim_step_count
        plan = connector.ik.plan_linear(matrix_to_pose(before), matrix_to_pose(target), seed_joints=env.joints())
        assert plan and len(plan["waypoints"]) >= 2
        assert env._sim_step_count == steps, "Planning advanced physics"
        connector._execute_trajectory(plan, 1, .01, 120)
        measured = pose_to_matrix(connector.get_ee_pose())
        checker.native.set_joints(env.joints(), env.gripper_angle())
        usd = checker.native.body_matrix("base_link")
        env.refresh_camera_obs()
        camera = env.camera_optical_matrix("robot0_eye_in_hand")
        expected = measured @ hand_from_camera
        row = dict(delta_m=delta, rotation_deg=angle,
            position_error_m=float(np.linalg.norm(measured[:3, 3] - target[:3, 3])),
            orientation_error_rad=float(Rotation.from_matrix(measured[:3, :3] @ target[:3, :3].T).magnitude()),
            usd_fk_error_m=float(np.linalg.norm(usd[:3, 3] - measured[:3, 3])),
            usd_fk_rotation_error_rad=float(Rotation.from_matrix(usd[:3, :3] @ measured[:3, :3].T).magnitude()),
            wrist_camera_position_error_m=float(np.linalg.norm(camera[:3, 3] - expected[:3, 3])),
            wrist_camera_rotation_error_rad=float(Rotation.from_matrix(camera[:3, :3] @ expected[:3, :3].T).magnitude()),
            fixed_camera_matrix_error=float(np.max(np.abs(env.camera_optical_matrix("agentview") - fixed))))
        assert row["position_error_m"] <= env.motion_position_tolerance_m, row
        assert row["orientation_error_rad"] <= env.motion_orientation_tolerance_rad, row
        assert row["usd_fk_error_m"] < .0001 and row["usd_fk_rotation_error_rad"] < .001, row
        assert row["wrist_camera_position_error_m"] < .0001 and row["wrist_camera_rotation_error_rad"] < .001, row
        assert row["fixed_camera_matrix_error"] < 1e-6, row
        checks.append(row)
    env.move_to_joints_blocking(reference, tolerance=.001)
    widths = []
    for width in (.085, .04, 0., .085):
        env._set_gripper_width(width)
        for _ in range(30):
            env._step_once()
        actual = env.gripper_width()
        assert abs(actual - width) <= .003, (width, actual)
        widths.append(dict(command_m=width, measured_m=actual))
    return dict(robot_geometry_count=len(checker.meshes), rotated_fk_checks=fk_checks,
        motion_checks=checks, gripper=widths,
        arrival_tolerances=dict(position_m=env.motion_position_tolerance_m,
                                orientation_rad=env.motion_orientation_tolerance_rad))


def check_backend(backend, registry, recorder, *, pick_uv, place_uv, execute):
    """Exercise RobotUse through registered tools, with explicit fixed public inputs."""
    from src.backend.tool_executor import execute_tool

    purpose = "pick"

    def dispatch(name, arguments):
        # The production orchestrator performs these same role-specific bindings.
        if name == "select_region":
            return backend.select_region(**arguments, purpose=purpose)
        name = {"adjust_grasp": "refine_candidate",
                "inspect_place_candidate": "explicit_inspect_place"}.get(name, name)
        return getattr(backend, name)(**arguments)

    def call(name, **arguments):
        recorder.event("registered_tool_call", dict(name=name, arguments=arguments, purpose=purpose))
        result = execute_tool(registry, name, arguments, dispatch=dispatch)
        recorder.event("registered_tool_result", dict(name=name, result=result))
        return result

    def observe():
        # Current observation is an implicit runtime operation, not an LLM tool.
        result = backend.observe()
        recorder.event("implicit_observation", result)
        return result

    before = backend.connector.env._sim_step_count
    observation = observe()
    assert backend.connector.env._sim_step_count == before
    assert len(observation["views"]) == 2
    selected = call("select_region", observation_id=observation["observation_id"],
                    view_id="agentview", u=pick_uv[0], v=pick_uv[1])
    generated = call("grasp_candidates", point_ref=selected["point_ref"], direction="median",
        tolerance_deg=None, azimuth_deg=None, polar_deg=None,
        geometric_height=dict(reference="segment_median", value_m=0.),
        transit=dict(pre=dict(reference="grasp", value_m=.20), post=dict(reference="grasp", value_m=.20)))
    assert generated["candidates"], generated
    candidate = min(generated["candidates"], key=lambda row: abs(row.get("yaw_deg") or 0))
    inspection = call("inspect_candidate", candidate_ref=candidate["candidate_ref"])
    assert inspection["image_refs"]
    adjusted = call("adjust_grasp", candidate_ref=candidate["candidate_ref"],
                    dx_mm=0., dy_mm=0., dz_mm=0., roll_deg=0., pitch_deg=0., yaw_deg=0.)
    assert adjusted.get("candidate_ref"), adjusted
    validation = call("validate_grasp", candidate_ref=adjusted["candidate_ref"])
    assert validation["accepted"], validation
    result = dict(observation=observation, selected_region=selected,
        grasp_candidates=generated, inspection=inspection, adjusted_grasp=adjusted,
        grasp_validation=validation, physical_execution_requested=execute)
    assert backend.connector.env._sim_step_count == before, "Perception or candidate checks advanced physics"
    if not execute:
        return result
    backend.pause_refiner = lambda stage, context: dict(status="continue", source="fixed_no_llm_validation")
    grasp = call("execute_grasp", candidate_ref=adjusted["candidate_ref"],
                 validation_ref=validation["validation_ref"])
    result["grasp_execution"] = grasp
    assert grasp.get("status") == "succeeded", grasp
    assert backend.held_plan is not None and backend.grasp_attachment is not None
    purpose = "place"
    observation = observe()
    destination = call("select_region", observation_id=observation["observation_id"],
                       view_id="agentview", u=place_uv[0], v=place_uv[1])
    saved = call("save_destination", point_ref=destination["point_ref"])
    proposals = call("place_candidates", destination_ref=saved["destination_ref"])
    xy = next(row for row in proposals["xy_candidates"] if row["candidate_id"] == "median")
    prepared = call("prepare_place", destination_ref=saved["destination_ref"], xy_source="median",
        xy_m=xy["observed_xyz_m"][:2], height=dict(reference="observed_max", value_m=.05),
        transit_height=dict(reference="current_tcp", value_m=0.))
    result.update(destination=destination, placement_candidates=proposals, prepared_place=prepared)
    assert prepared["accepted"], prepared
    adjusted_place = call("adjust_place", candidate_ref=prepared["candidate_ref"],
                         dx_mm=0., dy_mm=0., dz_mm=0., roll_deg=0., pitch_deg=0., yaw_deg=0.)
    assert adjusted_place["accepted"], adjusted_place
    placement = call("execute_place", candidate_ref=adjusted_place["candidate_ref"])
    result["placement_execution"] = placement
    assert placement.get("status") == "succeeded", placement
    # Private diagnostic only: never passed to an agent or used to pick a pose.
    env = backend.connector.env
    recorder.event("pre_release_native_state", dict(episode_done=env._current_done,
        episode_truncated=env._current_truncated, simulator_steps=env._sim_step_count,
        measured_gripper_width_m=env.gripper_width()))
    release = call("release")
    result["release"] = release
    assert release.get("status") == "succeeded" and release["release_commanded"], release
    return result
