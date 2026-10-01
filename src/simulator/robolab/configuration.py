"""Explicit simulator profiles; no implicit gripper or calibration substitution."""
import json
import numpy as np


def configure_environment(args):
    """Resolve the RobotUse native robot profile without selecting another pipeline."""
    if args.environment != 'robolab':
        raise ValueError('RobotUse requires RoboLab')
    args.robot_profile = 'franka_robotiq_2f85'
    args.task = args.task or 'BananaInBowlTask'
    args.intent_driven = True
    args.task_runner = False
    args.object_cloud_policy = 'fused'
    args.libero_gripper_adapter = False
    if args.grasp_scene_filter is None:
        args.grasp_scene_filter = False


def load_calibration(args):
    if args.grasp_to_ee is not None:
        transform = np.asarray(json.loads(args.grasp_to_ee.read_text()), dtype=float)
    elif args.environment == 'robolab':
        from src.simulator.robolab.gripper import GRASP_TO_EE
        transform = GRASP_TO_EE.copy()
    else:
        raise ValueError('LIBERO requires its measured grasp-to-EE calibration')
    correction = None if args.public_to_planner is None else np.asarray(
        json.loads(args.public_to_planner.read_text()), dtype=float)
    return transform, correction


def tool_options(args, graspgen):
    """Inject environment data at construction; shared tools retain legacy defaults."""
    if args.environment == 'libero':
        return {}
    assets = graspgen.mesh_assets
    options = dict(max_gripper_width_m=assets.max_opening_m,
        capture_width_tolerance_m=0., gripper_assets=assets,
        gripper_render_options=dict(finger_tip_z_m=.149, finger_base_z_m=.111,
                                    jaw_center_offset_m=assets.jaw_center_offset_m))
    if getattr(args, 'adaptive_grasp_opening', False):
        from functools import partial
        options['grasp_opening_policy'] = partial(assets.contact_opening,
            padding_per_side_m=args.grasp_opening_padding_mm / 1000.)
    if args.world_planning:
        from src.simulator.robolab.local_planner import plan_observed_transit
        options['observed_transit_planner'] = plan_observed_transit
    return options
