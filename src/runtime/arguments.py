"""RobotUse episode options and native execution limits."""
import argparse
import os
from pathlib import Path
import numpy as np
from src.core.contracts import Budgets
from src.tools.perception.sam2_adapter import DEFAULT_PYTHON as SAM2_PYTHON, DEFAULT_SNAPSHOT as SAM2_SNAPSHOT


def argument_parser(*, parents=()):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False, parents=parents)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("environment", nargs="?", choices=("robolab",), default="robolab",
        help="RoboLab with the native Franka/Robotiq 2F-85 profile")
    parser.add_argument("--task", default=None)
    parser.add_argument("--objective", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--task-score', action=argparse.BooleanOptionalAction, default=True,
        help='Report the native subtask score separately from reward (default: on); does not change task success.')
    parser.add_argument('--randomize-init-pose', action=argparse.BooleanOptionalAction, default=False,
        help='RoboLab: enable native RandomizeInitPoseUniform for dynamic task objects.')
    parser.add_argument('--init-pose-xy-range-m', type=float, default=.1,
        help='Uniform initial XY offset half-range in metres; Z and rotation unchanged (default: .1).')
    parser.add_argument('--motion-speed-scale', type=float, default=None,
        help='RoboLab trajectory playback speed (default: 3.0; 1.0 restores original speed).')
    parser.add_argument('--gripper-open-settle-steps', type=int, default=None,
        help='RoboLab opening dwell in control ticks (default: 20).')
    parser.add_argument('--gripper-close-settle-steps', type=int, default=None,
        help='RoboLab closing dwell in control ticks (default: 20; verification hold is separate).')
    parser.add_argument('--adaptive-grasp-opening', action=argparse.BooleanOptionalAction, default=False,
        help='RoboLab: size the approach opening from observed points and native Robotiq pads.')
    parser.add_argument('--grasp-opening-padding-mm', type=float, default=10.,
        help='Adaptive RoboLab approach opening: extra clearance per side in mm (default: 10, total: 20).')
    parser.add_argument("--model", default=os.environ.get("ROBOT_LLM_MODEL", "gemini-3.8-flash"))
    parser.add_argument("--provider", choices=("google", "openrouter"), default=os.environ.get("ROBOT_LLM_PROVIDER"),
        help='Provider selection uses ROBOT_LLM_PROVIDER or the configured default.')
    parser.add_argument("--reasoning-effort", default=os.environ.get("ROBOT_LLM_REASONING", "default"))
    parser.add_argument('--task-grasp-budget', type=int, default=12)
    parser.add_argument("--json-action-fallback", action="store_true", help="Explicit non-native JSON compatibility mode")
    parser.add_argument("--sam-python", type=Path, default=SAM2_PYTHON,
        help="Python environment for the pinned official SAM2 implementation")
    parser.add_argument("--sam2-snapshot", type=Path, default=SAM2_SNAPSHOT,
        help="Official sam2.1_hiera_large.pt file or the directory containing it")
    parser.add_argument('--motion-position-tolerance-mm', type=float, default=None)
    parser.add_argument('--motion-joint-tolerance-deg', type=float, default=None)
    parser.add_argument('--motion-angle-tolerance-deg', type=float, default=None)
    parser.add_argument("--sam-device", default="cuda")
    parser.add_argument("--grasp-to-ee", type=Path, default=None,
                        help="Robot-calibrated JSON gripper_T_connector-EE; defaults to runtime calibration")
    parser.add_argument("--public-to-planner", type=Path, default=None,
        help="Measured JSON public-EE_T_planner-EE correction; defaults to runtime calibration; raw RGBD is untouched")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--record-agent-interface", action=argparse.BooleanOptionalAction, default=True, help="Stream synchronized continuous front/wrist plus private plan/event evidence")
    parser.add_argument("--interface-every-n-steps", type=int, default=2, help="Record every N actual CONTROL steps (default2); never adds physics steps")
    parser.add_argument("--grasp-clearance-m", type=float, default=.0006,
        help="Scene and whole-arm clearance in metres with --no-grasp-clearance-batch-decay (default: 0.0006)")
    parser.add_argument("--grasp-clearance-batch-decay", action=argparse.BooleanOptionalAction, default=True,
        help="Use fixed 2 mm clearance (default: on); disable to use --grasp-clearance-m")
    parser.add_argument("--prime-steps", type=int, default=40, help=argparse.SUPPRESS)
    parser.add_argument("--child-steps", type=int, default=24)
    parser.add_argument("--debug-reset-on-failed-grasp", action="store_true",
        help="Preserve diagnostic interrupt propagation; automatic robot resets remain disabled")
    parser.add_argument('--auto-refine-routes', action='store_true')
    parser.add_argument('--observe-before-grasp', action='store_true',
        help='Require one achieved downward observation above the target before the first grasp request.')
    parser.add_argument('--prewarm-workers', action='store_true',
        help='Load the SAM2 worker in a background thread at episode start.')
    parser.add_argument('--append-only-observations', action=argparse.BooleanOptionalAction, default=False,
        help='Preserve prior observation/image inputs in each session for prompt-prefix reuse.')
    # Execution stages are fixed for the supported RoboLab pipeline. Keep the
    # shared backend inputs internal instead of exposing ineffective switches.
    parser.set_defaults(
        intent_driven=True, review_driven=True, task_runner=False,
        object_cloud_policy='fused', image_history_policy='current_turn',
        anyplace=False, grasp_scene_filter=False, decision_playbook=None,
        multiview=True, libero_gripper_adapter=False, waypoint_views=False,
        allow_partial_safety=False, plan_only=False, first_grasp_only=False,
        contact_manipulation=False, max_restarts=0,
        experimental_collision_fallback=True, skill_owned_execution=True,
        reobserve_grasp=True, grasp_checkpoints=True, pause_refine=True,
        world_planning=True, linear_waypoints=False,
        transit_policy='high', high_transit_z_m=.45, lift_m=.15,
        prefer_downward_grasps=False, downward_weight=1., topk=32,
        grasp_batch_per_view=3, downward_pool=0,
        grasp_motion_policy='source-order', grasp_score_tolerance=0.,
    )
    return parser


def parse_args(argv=None):
    parser = argument_parser()
    args = parser.parse_args(argv)
    from src.simulator.robolab.configuration import configure_environment
    configure_environment(args)
    if not np.isfinite(args.init_pose_xy_range_m) or args.init_pose_xy_range_m <= 0:
        parser.error('--init-pose-xy-range-m must be positive and finite')
    if not np.isfinite(args.grasp_opening_padding_mm) or args.grasp_opening_padding_mm < 0:
        parser.error('--grasp-opening-padding-mm must be nonnegative and finite')
    for name in ('motion_position_tolerance_mm', 'motion_angle_tolerance_deg', 'motion_joint_tolerance_deg'):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value <= 0):
            parser.error('--' + name.replace('_', '-') + ' must be positive and finite')
    if args.motion_speed_scale is None:
        args.motion_speed_scale = 1.5
    if not np.isfinite(args.motion_speed_scale) or args.motion_speed_scale <= 0:
        parser.error('--motion-speed-scale must be positive and finite')
    for name in ('gripper_open_settle_steps', 'gripper_close_settle_steps'):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error('--' + name.replace('_', '-') + ' must be a positive integer')
        setattr(args, name, 20 if value is None else value)
    if args.provider is None:
        args.provider = 'openrouter'
    if args.grasp_clearance_batch_decay:
        args.grasp_clearance_m = .002
    if args.task_grasp_budget < 1:
        parser.error('--task-grasp-budget must be positive')
    return args


def make_grasp_clearance_policy(args):
    """Report the fixed clearance actually used by the current grasp backend."""
    return {'mode': 'fixed',
            'initial_clearance_m': args.grasp_clearance_m,
            'floor_clearance_m': args.grasp_clearance_m,
            'applies_to': ['moving_robot_scene', 'configured_self_pairs']}


def make_orchestration_budgets(args, *, target_intent_mode):
    """Retain whole-task orchestration limits for the supported runtime."""
    return Budgets(prime_steps=args.prime_steps, child_steps=args.child_steps,
                   max_tool_calls=256, max_delegations=32)
