#!/usr/bin/env python3
"""Resolve configuration and run a RobotUse episode.

Playbook v3 is the default; v0, v1 and v2 are explicit
choices. --dry-run validates configuration without simulator/model startup.
"""
import argparse
import json
import os
from pathlib import Path
from dataclasses import replace
import sys
import tempfile

from src.runtime.paths import REPOSITORY_ROOT
from src.runtime.arguments import argument_parser, parse_args
from src.runtime.resolution import configuration_parser, resolve_configuration
from src.runtime.episode import run_episode
from src.utils.gpu import gpu_visibility, physical_gpu_index
DEFAULT_CONFIG = REPOSITORY_ROOT / 'configs/robot.json'
from src.agent.playbook import DEFAULT_VERSION as PLAYBOOK_VERSION
from src.agent.playbook.v0 import PATH as PLAYBOOK_V0_PATH
from src.agent.playbook.v1 import PATH as PLAYBOOK_V1_PATH
from src.agent.playbook.v2 import PATH as PLAYBOOK_V2_PATH
from src.agent.playbook.v3 import PATH as PLAYBOOK_V3_PATH
EXECUTION_FEATURES = dict(planning_failure_stage=True,
    execution_pose_provenance=True, clicked_grasp_candidates=True,
    clicked_seed_yaw_deg=0., clicked_surface_tolerance_m=.01,
    translation_step_limit_mm=None, translation_cumulative_limit_mm=None,
    angular_step_limit_deg=None, angular_cumulative_limit_deg=None,
    motion_validation='existing configured checks; unchanged collision switches')


def resolve(argv=None):
    from src.runtime.bootstrap import load_tool_registry
    load_tool_registry()
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--playbook-version', choices=('v0', 'v1', 'v2', 'v3', '0', '1', '2', '3'),
        default='v' + PLAYBOOK_VERSION,
        help=f'Role-scoped decision policy (default: v{PLAYBOOK_VERSION}).')
    parser.add_argument('--transit-planner', choices=('native', 'curobo'), default='native',
        help='Observed-world transit planner; native preserves existing robot motion.')
    parser.add_argument('--curobo-robot-file', type=Path,
        help='Explicit cuRobo robot YAML matching the native RoboLab robot.')
    parser.add_argument('--curobo-calibration-file', type=Path,
        help='Explicit joint/frame/gripper calibration JSON for that robot YAML.')
    if any(option in ('-h', '--help') for option in argv):
        argument_parser(parents=[parser, configuration_parser()]).parse_args(['--help'])
    options, remaining = parser.parse_known_args(argv)
    transit_planner = None
    if options.transit_planner == 'curobo':
        if options.curobo_robot_file is None or options.curobo_calibration_file is None:
            raise ValueError('cuRobo requires --curobo-robot-file and --curobo-calibration-file; no default robot substitution')
        from src.tools.curobo.adapter import CuroboPlanner
        transit_planner = CuroboPlanner(options.curobo_robot_file, options.curobo_calibration_file)
    elif options.curobo_robot_file is not None or options.curobo_calibration_file is not None:
        raise ValueError('cuRobo configuration requires --transit-planner curobo')
    version = options.playbook_version.removeprefix('v')
    path = {'0': PLAYBOOK_V0_PATH, '1': PLAYBOOK_V1_PATH,
            '2': PLAYBOOK_V2_PATH, '3': PLAYBOOK_V3_PATH}[version]
    resolved, record, dry, configuration = resolve_configuration(
        remaining, configuration_name='RobotUse', default_config=DEFAULT_CONFIG)
    configuration = replace(configuration, pose_preview_editor=True,
                            playbook_path=path, playbook_version=version,
                            observed_transit_planner=transit_planner)
    args = parse_args(resolved)
    reviews = ['paused_pregrasp'] if args.pause_refine else []
    if args.auto_refine_routes:
        reviews = ['grasp_candidate', *reviews, 'place_candidate']
    record['manipulation'] = configuration.metadata()
    record['decision_playbook'] = record['manipulation']['playbook']
    record.update(robotuse=dict(
        observe_before_grasp=args.observe_before_grasp, observation_failure_waiver=True,
        observation_reset='release', pose_preview_editor=True,
        auto_refine_routes=args.auto_refine_routes, refiner_review=reviews,
        requested_candidate_refinement=True, angular_step_limit_deg=None,
        angular_cumulative_limit_deg=None, edited_rotation='arbitrary_finite_local_xyz',
        cgn_preflight_required=True, cgn_autostart=True,
        execution_features=dict(EXECUTION_FEATURES)))
    import os
    gpu = gpu_visibility()
    record['physical_gpu'] = physical_gpu_index(gpu)
    record['cuda_visible_devices'] = gpu
    record['manipulation']['gpu'] = f'CUDA_VISIBLE_DEVICES={gpu}; logical cuda:0'
    record['initial_pose_randomization'] = dict(enabled=args.randomize_init_pose,
        implementation='RandomizeInitPoseUniform', xy_range_m=args.init_pose_xy_range_m,
        z_rotation='unchanged', object_selection='native dynamic task bodies excluding fixed infrastructure',
        seed=args.seed)
    if transit_planner is not None:
        record['robotuse']['transit_planner'] = transit_planner.preflight()
    return resolved, record, dry, configuration


def main(argv=None):
    resolved, record, dry, configuration = resolve(argv)
    if not dry:
        import os
        from src.tools.grasp.service import ensure_cgn
        args = parse_args(resolved)
        if args.output_dir.exists():
            raise FileExistsError(args.output_dir)
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        evidence_dir = Path(tempfile.mkdtemp(prefix=args.output_dir.name + '-cgn-preflight-',
                                             dir=args.output_dir.parent))
        record['cgn_preflight'] = {**ensure_cgn(configuration.cgn_service_url,
            evidence_dir=evidence_dir, timeout_s=configuration.cgn_timeout_s),
            'evidence_dir': str(evidence_dir.resolve())}
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_visibility()
        if configuration.observed_transit_planner is not None:
            record['robotuse']['transit_planner']['runtime_dependencies'] = configuration.observed_transit_planner.validate_runtime()
    return execute(resolved, record, dry, configuration)




def execute(resolved, record, dry, configuration):
    if dry:
        print(json.dumps(record, indent=2))
        return 0
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_visibility()
    output = parse_args(resolved).output_dir
    if output.exists():
        raise FileExistsError(output)
    from src.simulator.robolab.adapter import run_native_cli
    name = record['configuration']
    print(name+'_CONFIGURATION '+json.dumps(record), flush=True)
    def execute():
        try:
            return run_episode(resolved, configuration)
        finally:
            if output.is_dir():
                (output/(name.lower()+'_configuration.json')).write_text(json.dumps(record, indent=2)+'\n')
    return run_native_cli(execute)
