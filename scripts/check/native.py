#!/usr/bin/env python3
"""Validate RobotUse in actual RoboLab using fixed public inputs and zero LLM calls."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sam-python", type=Path, required=True)
    parser.add_argument("--sam2-snapshot", type=Path, required=True)
    parser.add_argument("--task", default="BananaInBowlTask")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument('--task-score', action=argparse.BooleanOptionalAction, default=True,
                        help='Report native subtask progress separately from reward (default: on)')
    parser.add_argument("--pick-uv", nargs=2, type=float, default=[597., 536.],
                        help="Explicit pick point in the front RGB, normalized to 0..1000")
    parser.add_argument("--place-uv", nargs=2, type=float, default=[508., 571.],
                        help="Explicit destination point in the front RGB, normalized to 0..1000")
    parser.add_argument("--execute", action="store_true", help="Also execute the checked grasp, place, and explicit release")
    args = parser.parse_args()
    from src.simulator.robolab.validation import (
        ValidationRecorder, check_backend, check_connector, write_result)
    from src.simulator.robolab.verifier import task_score
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status="running", started_at_utc=datetime.now(timezone.utc).isoformat(),
        task=args.task, seed=args.seed, llm_calls=0, checks={},
        validation_scope="fixed-input native integration with strict connector defaults; not production runner configuration parity or autonomous agent evaluation")
    connector = None
    try:
        from src.runtime.bootstrap import load_tool_registry
        from src.runtime.configuration import classes
        from src.simulator.robolab.adapter import create_connector
        from src.simulator.robolab.gripper import GRASP_TO_EE, RoboLabGripperAssets
        from src.simulator.robolab.local_planner import plan_observed_transit
        from src.tools.perception.adapter import MultiviewPointRGBDAdapter
        from src.llm.image_context import ImageRegistry
        from src.tools.motion.planning import MotionConfig
        registry = load_tool_registry()
        connector = create_connector(task=args.task, output_dir=output / "native", initial_seed=args.seed)
        connector.reset(seed=args.seed)
        for _ in range(10):
            connector.step_once()
        import robolab
        result["robolab_source"] = str(Path(robolab.__file__).resolve())
        expected = ROOT / "src/simulator/robolab/third_party/robolab"
        assert Path(robolab.__file__).resolve().is_relative_to(expected), result["robolab_source"]
        result["checks"]["connector"] = check_connector(connector)
        # Restore the specified native seed before the fixed-input scenario.
        connector.reset(seed=args.seed)
        for _ in range(10):
            connector.step_once()
        assets = RoboLabGripperAssets(connector.env.robot.cfg.spawn.usd_path, ROOT)
        point = MultiviewPointRGBDAdapter(connector=connector, python=args.sam_python,
            sam2_snapshot=args.sam2_snapshot, output_dir=output, device="cuda:0")
        recorder = ValidationRecorder(output)
        backend_type, _ = classes()
        backend = backend_type(connector=connector, point_adapter=point,
            graspgen=SimpleNamespace(mesh_assets=assets, checkout=assets, libero_adapter=False,
                                      official_clearance_m=.001, calls=0),
            cgn_client=None, images=ImageRegistry(), output_dir=output, grasp_to_ee=GRASP_TO_EE,
            motion_config=MotionConfig(collision_checks_enabled=False), multiview=True,
            active_perception=True, object_cloud_policy="fused", task_grasp_budget=16,
            recorder=recorder, max_gripper_width_m=assets.max_opening_m,
            capture_width_tolerance_m=0., gripper_assets=assets,
            gripper_render_options=dict(finger_tip_z_m=.149, finger_base_z_m=.111,
                                        jaw_center_offset_m=assets.jaw_center_offset_m),
            grasp_opening_policy=partial(assets.contact_opening, padding_per_side_m=.010),
            observed_transit_planner=plan_observed_transit)
        backend.home_joints = connector.env.joints().copy()
        backend.waypoint_path_collision_checks = False
        backend.grasp_path_collision_checks = False
        write_result(output / "configuration.json", dict(
            scope=result["validation_scope"], task=args.task, seed=args.seed,
            randomize_init_pose=False, pick_uv=args.pick_uv, place_uv=args.place_uv,
            place_clearance_m=.05,
            physical_execution_requested=args.execute, task_grasp_budget=backend.task_grasp_budget,
            task_score_enabled=args.task_score,
            grasp_generator="observed_median", selected_yaw="smallest absolute offered yaw",
            paused_refiner="fixed continue without a model", transit_planner="native",
            motion_speed_scale=connector.env.motion_speed_scale,
            motion_position_tolerance_m=connector.env.motion_position_tolerance_m,
            motion_orientation_tolerance_rad=connector.env.motion_orientation_tolerance_rad,
            motion_joint_tolerance_rad=connector.env.motion_joint_tolerance_rad,
            motion_config=asdict(backend.motion_config),
            waypoint_path_collision_checks=backend.waypoint_path_collision_checks,
            grasp_path_collision_checks=backend.grasp_path_collision_checks,
            sam_python=args.sam_python, sam2_snapshot=args.sam2_snapshot,
            production_policy_run=False))
        result["checks"]["robotuse_tools"] = check_backend(backend, registry, recorder,
            pick_uv=args.pick_uv, place_uv=args.place_uv, execute=args.execute)
        result["status"] = "passed"
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        imported = {}
        for name, module in tuple(sys.modules.items()):
            if name.startswith(("src.", "robot_skill_selector", "robolab", "gap.")):
                location = getattr(module, "__file__", None)
                if location and Path(location).is_file():
                    path = Path(location).resolve()
                    imported[name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        write_result(output / "imported_sources.json", imported)
        result["historical_source_imported"] = any("/skills_src/" in row["path"] for row in imported.values())
        if result["historical_source_imported"]:
            result.update(status="failed", source_error="Historical source imported")
        verifier = dict(evaluated=False, task_success=None, reward=None,
                        scope="native task predicate, separate from integration assertions")
        if connector is not None:
            try:
                success, reward = connector.check_success()
                verifier.update(evaluated=True, task_success=bool(success), reward=float(reward))
            except Exception as exc:
                verifier["error"] = repr(exc)
        verifier.update(task_score(connector, enabled=args.task_score))
        write_result(output / "verifier.json", verifier)
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        result["verifier_path"] = str(output / "verifier.json")
        write_result(output / "result.json", result)
        print("ROBOTUSE_NATIVE_VALIDATION", json.dumps(dict(status=result["status"], result=str(output / "result.json"))), flush=True)
        if connector is not None:
            connector.close()
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    from src.simulator.robolab.cli import run_native_cli
    run_native_cli(main)
