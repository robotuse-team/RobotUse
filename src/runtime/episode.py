"""One RobotUse episode: native observation, tool execution, recording and verification."""
from __future__ import annotations
import json
import os
from pathlib import Path
import time
import numpy as np
from src.runtime.paths import REPOSITORY_ROOT as ROOT
from src.runtime.arguments import parse_args, make_grasp_clearance_policy, make_orchestration_budgets
from src.runtime.restart import InitialArmState, RestartLoop
from src.llm.config import ProviderConfig
from src.llm.manager import ImageRegistry, OpenRouterFactory
from src.utils.logging_utils import write_json, append_json, configure_logging, get_logger
from src.tools.perception.adapter import PointRGBDAdapter


def run_episode(argv, configuration):
    args = parse_args(argv)
    grasp_policy = 'agent-choice'
    grasp_motion_policy, grasp_score_tolerance = args.grasp_motion_policy, args.grasp_score_tolerance
    from src.runtime.configuration import configure_args
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('ROBOTUSE_GPU', '0')
    args = configure_args(args)
    if configuration.mandatory_observation:
        args.observe_before_grasp = True
    decision_playbook = configuration.load_playbook()
    from src.tools.grasp.motion_selection import validate_policy
    validate_policy(grasp_motion_policy, grasp_score_tolerance)
    # Fail before simulator creation, reset, or model workers.
    from src.runtime.configuration import validate_runtime
    validate_runtime(args, configuration)
    llm_config = ProviderConfig.resolve(args.model, provider=args.provider, reasoning=args.reasoning_effort)
    args.model = llm_config.model
    grasp_clearance_policy = make_grasp_clearance_policy(args)
    target_intent_mode = True
    args.multiview = True  # Active perception is always available to the skill planner.
    args.prefer_downward_grasps = False
    from src.simulator.robolab.configuration import load_calibration
    transform, public_to_planner = load_calibration(args)
    if transform.shape != (4, 4) or not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1]):
        raise ValueError("invalid grasp-to-EE calibration")
    from src.tools.grasp.prediction import checked_predictions
    checked_predictions(transform[None], [1.0])
    if public_to_planner is not None:
        checked_predictions(public_to_planner[None], [1.0])
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    configure_logging(output / 'runtime.log')
    logger = get_logger('episode')
    logger.info('Starting task %s, seed %s', args.task, args.seed)
    if decision_playbook is not None:
        (output / 'decision_playbook.md').write_text(decision_playbook.text, encoding='utf-8')
        playbook_metadata = decision_playbook.metadata()
        playbook_metadata['version'] = configuration.playbook_version
        write_json(output / 'decision_playbook.json', playbook_metadata)
    write_json(output / "calibration.json", {"grasp_to_ee": transform.tolist(),
        "source": str(args.grasp_to_ee) if args.grasp_to_ee else 'native RoboLab Robotiq grasp-frame mapping',
        "point_cloud_frame": "connector_base",
        "public_to_planner": public_to_planner.tolist() if public_to_planner is not None else None,
        "public_to_planner_source": str(args.public_to_planner) if args.public_to_planner else None})
    write_json(output / "simulation_policy.json", {"simulator": args.environment, "robot_profile": args.robot_profile,
        "pose_dedup": {"enabled": False},
        "manipulation": configuration.metadata(),
        "grasp_policy": 'contact_graspnet_and_observed_centers',
        "grasp_motion_policy": grasp_motion_policy, "grasp_score_tolerance": grasp_score_tolerance,
        "lift_m": None,
        "motion_speed_scale": args.motion_speed_scale,
        "task_score_enabled": args.task_score,
        "motion_position_tolerance_mm": args.motion_position_tolerance_mm,
        "motion_angle_tolerance_deg": args.motion_angle_tolerance_deg,
        "motion_joint_tolerance_deg": args.motion_joint_tolerance_deg,
        "adaptive_grasp_opening": args.adaptive_grasp_opening,
        "grasp_opening_padding_per_side_mm": args.grasp_opening_padding_mm,
        "collision_checks_enabled": False, "physical_safety_checks_enabled": False,
        "graspgen_official_scene_filter": args.grasp_scene_filter,
        "contact_manipulation": args.contact_manipulation,
        "retained_checks": ["numerical JSON", "coordinate calibration", "IK feasibility", "final task verifier"],
        "placement_candidate_checks": ["current observed robot and held-object path geometry",
            "planned route, state freshness and measured arrival; separate explicit release"],
        "placement_visualization": "cyan Refiner mesh projections for measured XY references and agent-selected final contact pose"})
    budgets = make_orchestration_budgets(args, target_intent_mode=target_intent_mode)
    write_json(output / "interface_contract.json", {
        **llm_config.metadata(),
        "object_cloud_policy": args.object_cloud_policy,
        "image_history_policy": args.image_history_policy,
        "append_only_observations": args.append_only_observations,
        "mode": "explicit_geometry",
        "objective_scope": "whole instruction; Prime maintains observable conditions and selects shared grasp, move, release and turn actions",
        "camera_inputs": "synchronized, registered RGBD with intrinsics and camera-to-base transforms",
        "calibration_source": "ideal simulator measurements; physical calibration not validated",
        "point_meaning": "same object, potentially different visible surfaces" if target_intent_mode else "role-dependent legacy",
        "observation_delivery": "front/wrist RGB automatically supplied on every turn; RGBD recaptured after motion, reused while unchanged" if target_intent_mode else "observe tool",
        "view_motion": "Pointer chooses measured front reference and explicit translation with preserved rotation or explicitly proposes calibrated camera-down rotation while hand is open; preview, validate, execute",
        "destination_timing": "agent-selected measured destination; current RGB unchanged assessment before movement",
        "selector_decisions": (['accepted','needs_refinement','needs_observation','failed']),
        "pointer_geometry_review": "RGB point/mask overlays only; backend handles cloud",
        "max_view_requests": None,
        "max_view_requests_per_target": None,
        "view_request_budget_scope": "shared episode tool and delegation budgets; no automatic recovery moves",
        "candidate_reviews_per_target": None,
        "budgets": vars(budgets)})
    write_json(output / 'intent_policy.json', {
        'task_grasp_budget':args.task_grasp_budget,
        'raw_proposals_per_view':None,
        'grasp_clearance_policy':grasp_clearance_policy,
        'grasp_generation':'CGN fused observed scene/segment plus explicit mean/median poses',
        'motion':'Pointer-chosen translation, current rotation retained; no automatic look-at recovery',
        'state':'one Prime and backend for complete instruction; agent rubric is not verifier',
        'handoff':'reason, images, measured geometry and selected contact/transit contract; no sibling transcript or simulator truth',
        'manipulation': configuration.metadata()})
    write_json(output / 'grasp_generation_budget.json', dict(
        max_generated_per_task=args.task_grasp_budget, candidate_limit=4,
        selection='exclusive_direction_or_center',
        sources={'contact_graspnet': 4, 'observed_median': 4, 'observed_mean': 4},
        budget_scope='poses submitted to path planning; raw CGN proposals recorded separately',
        geometric_candidates='explicit mean or median request; agent-selected Z and fixed base yaw -45, 0, 45, 90; no automatic fallback'))
    images = ImageRegistry()
    factory = OpenRouterFactory(images=images, model=args.model, output_dir=output, json_action_fallback=args.json_action_fallback, first_grasp_only=args.first_grasp_only, active_perception=True, provider=args.provider, reasoning=args.reasoning_effort, image_history_policy=args.image_history_policy, append_only_observations=args.append_only_observations)
    factory.target_intent_mode = True
    factory.review_driven = args.review_driven
    factory.intent_driven = args.intent_driven
    factory.grasp_policy = grasp_policy
    factory.place_rotation = False
    factory.explicit_geometry_enabled = True
    from src.simulator.robolab.configuration import tool_options
    from src.runtime.configuration import ACTOR_CONTEXT
    factory.environment_instructions = ACTOR_CONTEXT
    from src.simulator.robolab.adapter import create_connector
    connector = create_connector(task=args.task,
        record_video=not args.no_video and not args.record_agent_interface,
        output_dir=output / 'native', motion_speed_scale=args.motion_speed_scale,
        initial_seed=args.seed, randomize_init_pose=args.randomize_init_pose,
        init_pose_xy_range_m=args.init_pose_xy_range_m,
        gripper_open_settle_steps=args.gripper_open_settle_steps,
        gripper_close_settle_steps=args.gripper_close_settle_steps,
        motion_position_tolerance_m=None if args.motion_position_tolerance_mm is None else args.motion_position_tolerance_mm / 1000.,
        motion_joint_tolerance_rad=None if args.motion_joint_tolerance_deg is None else np.deg2rad(args.motion_joint_tolerance_deg),
        motion_orientation_tolerance_rad=None if args.motion_angle_tolerance_deg is None else np.deg2rad(args.motion_angle_tolerance_deg))
    if args.objective is None:
        args.objective = connector.env.task_language
    graspgen = None
    success, reward, error, result = False, 0.0, None, None
    verifier_error, video_error, close_error = None, None, None
    verifier_evaluated = False
    first_grasp_verification = {"first_grasp_success": False, "evaluated": False}
    backend = None
    restart = None
    recorder = None
    recorder_error = None
    started = time.monotonic()
    try:
        from src.runtime.configuration import make_assets
        graspgen = make_assets(args, connector, output)
        connector.reset(seed=args.seed)
        restart = RestartLoop(initial=InitialArmState.capture(connector),
                              max_restarts=args.max_restarts)
        write_json(output / "restart.json", restart.to_dict())
        adapter_class = PointRGBDAdapter
        if args.multiview:
            from src.tools.perception.adapter import MultiviewPointRGBDAdapter
            adapter_class = MultiviewPointRGBDAdapter
        point = adapter_class(connector=connector, python=args.sam_python,
            sam2_snapshot=args.sam2_snapshot, output_dir=output, device=args.sam_device)
        if args.prewarm_workers:
            import threading
            def _prewarm(name, load):
                warm_started = time.monotonic()
                error = None
                try:
                    load()
                except Exception as exc:  # the real request will surface the failure again
                    error = repr(exc)
                append_json(output / "prewarm.jsonl", dict(worker=name, elapsed_s=time.monotonic()-warm_started,
                                                            error=error, started_at_s=warm_started-started))
            def _prewarm_all():
                _prewarm("sam2", point.prewarm)
            threading.Thread(target=_prewarm_all, daemon=True).start()
        from src.tools.motion.planning import MotionConfig
        motion_config = MotionConfig(collision_checks_enabled=False, transit_policy=args.transit_policy,
            high_transit_z_m=args.high_transit_z_m, lift_m=args.lift_m)
        from src.runtime.configuration import classes
        backend_class, orchestrator_class = classes()
        backend_extra = dict(task_grasp_budget=args.task_grasp_budget,
            grasp_batch_per_view=args.grasp_batch_per_view, downward_pool=args.downward_pool,
            object_cloud_policy=args.object_cloud_policy, moveit_grasps=None, pose_dedup=None,
            grasp_policy=grasp_policy, grasp_motion_policy=grasp_motion_policy,
            grasp_score_tolerance=grasp_score_tolerance, cgn_client=configuration.client())
        backend_extra.update(tool_options(args, graspgen))
        if configuration.observed_transit_planner is not None:
            backend_extra['observed_transit_planner'] = configuration.observed_transit_planner
        def build_backend():
            # Rebuilt per attempt: every cached cloud, plan and candidate ref belongs
            # to the scene before the failure, so none of it may cross a restart.
            return backend_class(connector=connector, point_adapter=point, graspgen=graspgen,
                images=images, output_dir=output, grasp_to_ee=transform, public_to_planner=public_to_planner,
                allow_partial_safety=args.allow_partial_safety, plan_only=args.plan_only,
                motion_config=motion_config, prefer_downward_grasps=args.prefer_downward_grasps,
                downward_weight=args.downward_weight, topk=args.topk, multiview=args.multiview, active_perception=True, first_grasp_only=args.first_grasp_only, contact_manipulation=args.contact_manipulation, **backend_extra)
        backend = build_backend()
        backend.home_joints = restart.initial.joints
        backend.waypoint_path_collision_checks = configuration.waypoint_path_collision_checks
        backend.grasp_path_collision_checks = configuration.grasp_path_collision_checks
        if args.record_agent_interface:
            from src.runtime.recording import AgentInterfaceRecorder
            recorder = AgentInterfaceRecorder(connector=connector, ee_connector=backend.connector,
                output_dir=output / "interface", objective=args.objective, model=args.model,
                every_n_steps=args.interface_every_n_steps,
                preview_dir=os.environ.get("ROBOTUSE_LIVE_PREVIEW_DIR"))
            backend.recorder = recorder
            recorder.install()
        def audit_event(event):
            append_json(output / "events.jsonl", event)
            if recorder is not None:
                recorder.event("actor_tool", event)
        def build_orchestrator(current):
            return orchestrator_class(current, factory,
            observe_before_grasp=args.observe_before_grasp,
            mandatory_observation=configuration.mandatory_observation,
            auto_refine_routes=args.auto_refine_routes,
            budgets=budgets,
            audit_sink=audit_event, debug_reset_on_failed_grasp=args.debug_reset_on_failed_grasp,
            active_perception=True, first_grasp_only=args.first_grasp_only, waypoint_views=args.waypoint_views,
            contact_manipulation=args.contact_manipulation,
            decision_playbook=decision_playbook,
            actor_context=ACTOR_CONTEXT)
        objective = args.objective
        while True:
            failure = None
            notice = restart.notice()
            # Restart notices use a separate stream so orchestrator events retain
            # their {seq, kind, session_id, ...} schema.
            append_json(output / "restart_events.jsonl", {"event": "attempt_started",
                "attempt": restart.attempt, "restart_notice": notice})
            try:
                # The notice leads: a restart is the first thing the actor must know.
                result = build_orchestrator(backend).run(notice + objective)
                if recorder is not None and recorder.error:
                    raise RuntimeError("interface recorder failed: " + recorder.error)
                # run() absorbs FreshAttemptRequired into a status, so the retry
                # decision is a status, never an exception. The benchmark oracle
                # never reaches here: only the actor-visible outcome gates a retry.
                #
                # The session status and the reported outcome are different things:
                # a session that ran to the end reports "completed" while Prime's
                # own finish says {"status": "failed"}. Both mean retry.
                reported = result.result.get("status") if isinstance(result.result, dict) else None
                if result.status in ("error", "fresh_attempt_required"):
                    failure = "orchestrator_status:" + result.status
                elif reported in ("failed", "needs_point"):
                    failure = "reported_status:" + reported
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                # A recorder fault is infrastructure, not an attempt outcome.
                if recorder is not None and recorder.error:
                    raise
                if not restart.may_restart():
                    raise
                failure = f"{type(exc).__name__}: {exc}"
            if failure is None or not restart.may_restart():
                break
            append_json(output / "restart_events.jsonl",
                        restart.restart(connector, reason=failure).to_dict())
            write_json(output / "restart.json", restart.to_dict())
            backend = build_backend()
            backend.recorder = recorder
    except KeyboardInterrupt:
        error = "interrupted"
        logger.warning('Episode interrupted')
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.exception('Episode execution failed')
    finally:
        # Independent post-episode verifier is private: never feed object state to actors.
        try:
            from src.tools.grasp.verifier import verify_first_grasp
            evidence = backend.first_grasp_diagnostic_evidence() if backend else {}
            if (not args.anyplace and result and
                    result.result.get('execution_ref') == evidence.get('execution_ref')):
                evidence["actor_visual_assessment"] = result.result.get("actor_visual_assessment")
            first_grasp_verification = verify_first_grasp(connector=connector,
                evidence=evidence, output_dir=output, model=factory.model, api_key=factory.key, objective=args.objective, provider_config=factory.config)
            first_grasp_verification['execution_ref'] = evidence.get('execution_ref')
            first_grasp_verification['scope'] = 'first grasp attempt only; independent of native task outcome'
        except Exception as exc:
            first_grasp_verification = {"first_grasp_success": False, "evaluated": False, "error": repr(exc)}
        write_json(output / "first_grasp_verifier.json", first_grasp_verification)
        write_json(output / "first_grasp_evidence.json", backend.first_grasp_diagnostic_evidence() if backend else {})
        # Trusted verifier runs even after a controller/model exception, never exposed to agents.
        try:
            success, reward = connector.check_success()
            verifier_evaluated = True
        except Exception as exc:
            verifier_error = f"{type(exc).__name__}: {exc}"
        from src.simulator.robolab.verifier import task_score
        score_result = task_score(connector, enabled=args.task_score)
        write_json(output / "verifier.json", {"evaluated": verifier_evaluated,
            "task_success": bool(success) if verifier_evaluated else None,
            "reward": float(reward) if verifier_evaluated else None, "error": verifier_error,
            **score_result})
        if recorder is not None:
            try:
                recorder.close()
                # Keep the front-video output path without loading every frame into memory.
                import shutil
                if (output / "interface/front.mp4").is_file():
                    shutil.copyfile(output / "interface/front.mp4", output / "rollout.mp4")
            except Exception as exc:
                recorder_error = f"{type(exc).__name__}: {exc}"
        if not args.no_video and not args.record_agent_interface:
            try:
                connector.save_video(str(output / "rollout.mp4"), fps=20, clear=True)
            except Exception as exc:
                video_error = f"{type(exc).__name__}: {exc}"
        try:
            connector.close()
        except Exception as exc:
            close_error = f"{type(exc).__name__}: {exc}"
    summary = {"process_id": os.getpid(), "session_ids": [e["session_id"] for e in result.events if e["kind"] == "session_started"] if result else [],
        "task": args.task, "environment": args.environment, "robot_profile": args.robot_profile,
        "seed": args.seed, **llm_config.metadata(),
        "append_only_observations": args.append_only_observations,
        "debug_reset_on_failed_grasp": args.debug_reset_on_failed_grasp,
        "restart": restart.to_dict() if restart is not None else None,
        "fresh_attempt_required": bool(result and result.status == "fresh_attempt_required"),
        "reset_cause": dict(result.result) if result and result.status == "fresh_attempt_required" else None,
        "transit_policy": args.transit_policy,
        "high_transit_z_m": args.high_transit_z_m if args.transit_policy == 'high' else None,
        "prefer_downward_grasps": args.prefer_downward_grasps,
        "multiview": args.multiview, "active_perception": True,
        "contact_manipulation": args.contact_manipulation,
        "target_intent_mode": target_intent_mode,
        "first_grasp_only": args.first_grasp_only,
        "anyplace_mcp": args.anyplace,
        "task_runner": args.task_runner,
        "placement_agent": None,
        "placement_source": "legacy observed support",
        "first_grasp_success": first_grasp_verification.get("first_grasp_success") is True,
        "first_grasp_verification": first_grasp_verification,
        "actor_visual_assessment": result.result.get("actor_visual_assessment") if result else None,
        "task_success": bool(success) if verifier_evaluated else None,
        "reward": float(reward) if verifier_evaluated else None, "error": error,
        "verifier_evaluated": verifier_evaluated, "verifier_error": verifier_error,
        **score_result,
        "video_error": video_error, "close_error": close_error,
        "record_agent_interface": args.record_agent_interface, "recorder_error": recorder_error,
        "interface_manifest": str(output / "interface/manifest.json") if recorder else None,
        "plan_only": args.plan_only, "validated_grasps": backend.validated_count if backend else 0,
        "orchestrator_status": result.status if result else None,
        "orchestrator_result": dict(result.result) if result else None,
        "tool_calls": result.tool_calls if result else None,
        "delegations": result.delegations if result else None,
        "graspgen_calls": None,
        "intent_driven": args.intent_driven,
        "experimental_collision_fallback": args.experimental_collision_fallback,
        "linear_waypoints": args.linear_waypoints,
        "intent_grasp_budget": backend.grasp_budget() if args.intent_driven and backend else None,
        "elapsed_s": time.monotonic() - started,
        "perception_source": "SAM2 point prompt + calibrated RGBD",
        "grasp_policy": grasp_policy,
        "grasp_motion_policy": grasp_motion_policy, "grasp_score_tolerance": grasp_score_tolerance,
        "grasp_clearance_policy": grasp_clearance_policy,
        "safety_scope": f"grasp candidate paths through grasp: whole robot vs observed scene and configured self pairs, {args.grasp_clearance_m*1000:g}mm initial clearance (effective policy recorded separately), 0.5mm sampling bound; placement: only third release goal closed-gripper/held-object scene geometry, static IK and environment-profile robot self collision with complete four-goal route generation, opening and other-goal/path collisions unchecked; execution tracking follows environment profile and explicit checkpoints",
        "anyplace_local_support_check": False,
        "anyplace_grip_geometry_check": False,
        "anyplace_release_goal_only": args.anyplace,
        "anyplace_max_agent_choices": None,
        "anyplace_pregrasp_compatibility": False,
        "collision_checks_enabled": False, "physical_safety_checks_enabled": False,
        "graspgen_official_scene_filter": args.grasp_scene_filter,
        "contact_manipulation": args.contact_manipulation,
        "allow_partial_safety": True}
    from src.tools.motion.tolerances import cartesian_tolerances, requires_profile_tracking
    summary.update(manipulation=configuration.metadata(), grasp_policy='contact_graspnet_and_observed_centers',
        grasp_source='Contact-GraspNet plus independently selectable observed median and mean poses',
        placement_source='observed click/median/mean XY; agent coordinates and explicit release',
        placement_agent='independent Place LLM session; Prime decides release after movement',
        transit_policy='independent_pre_post', high_transit_z_m=None,
        safety_scope='Observed robot and payload path checks, numerical planning and measured arrival; '
            'occluded geometry and slip remain unknown. Native verifier alone determines task success.')
    from src.runtime.budget import simulation_budget, gripper_settle_steps
    summary['motion_speed_scale'] = args.motion_speed_scale
    summary['simulation_budget'] = simulation_budget(connector)
    summary['gripper_dwell_steps'] = dict(
        grasp_open=gripper_settle_steps(connector, 'open', 40),
        grasp_close=gripper_settle_steps(connector, 'close', 60),
        placement_open=gripper_settle_steps(connector, 'open', 60))
    if requires_profile_tracking(connector):
        position_limit, angle_limit = cartesian_tolerances(connector)
        summary['motion_arrival_tolerances'] = dict(position_m=position_limit,
            orientation_rad=angle_limit, enforced_independently_of_collision_policy=True)
    write_json(output / "summary.json", summary)
    logger.info('Episode finished: verifier evaluated=%s, task success=%s, execution error=%s',
                verifier_evaluated, success if verifier_evaluated else None, error)
    write_json(output / "image_manifest.json", {ref: str(path) for ref, path in images.paths.items()})
    print(json.dumps(summary, allow_nan=False))
    if error == "interrupted" and args.debug_reset_on_failed_grasp:
        raise KeyboardInterrupt
    return 0 if not error and (success or (args.plan_only and backend and backend.validated_count)) else 1
