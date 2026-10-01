"""Resolve robot tool policy, task difficulty and execution defaults."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from src.runtime.paths import REPOSITORY_ROOT as ROOT
from src.runtime.arguments import parse_args
from src.runtime.configuration import RuntimeConfiguration, configure_args
DEFAULT_CONFIG = ROOT / 'configs/robot.json'


EXECUTION_DEFAULTS = (
    '--task-grasp-budget', '1000',
    '--motion-position-tolerance-mm', '50',
    '--motion-angle-tolerance-deg', '10',
    '--motion-joint-tolerance-deg', '10',
    '--motion-speed-scale', '3.0',
    '--gripper-open-settle-steps', '20',
    '--gripper-close-settle-steps', '20',
)

def metadata_path():
    from src.simulator.robolab.adapter import ROBOLAB_SOURCE
    return ROBOLAB_SOURCE / 'robolab/tasks/_metadata/task_metadata.json'


def configuration_parser(*, default_config=DEFAULT_CONFIG):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--config', type=Path, default=default_config,
                        help='Runtime presets and Prime turn limits by task difficulty')
    parser.add_argument('--difficulty', choices=('auto', 'simple', 'moderate', 'complex'), default='auto')
    parser.add_argument('--task-metadata', type=Path,
                        help='Native task metadata used to resolve automatic difficulty')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate and print configuration without simulator or model startup')
    parser.add_argument('--cgn-service-url', default=os.environ.get('GRASPNET_SERVICE_URL', 'http://127.0.0.1:8115'))
    parser.add_argument('--cgn-timeout-s', type=float, default=120.)
    parser.add_argument('--waypoint-path-collision-checks', action=argparse.BooleanOptionalAction,
                        default=False, help='Enable post-plan robot/scene, self and held-object waypoint sweep checks (default: off).')
    parser.add_argument('--grasp-path-collision-checks', action=argparse.BooleanOptionalAction,
                        default=False, help='Enable pick approach/grasp/lift post-plan sweep checks, including refinement (default: off; independent of waypoint checks).')
    return parser


def resolve_configuration(argv=None, *, configuration_name='RobotUse', default_config=DEFAULT_CONFIG,
                          mandatory_observation=False):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'robolab':
        argv.pop(0)
    parser = configuration_parser(default_config=default_config)
    options, remaining = parser.parse_known_args(argv)
    config = json.loads(options.config.read_text())
    if config.get('name') != configuration_name or config.get('image_history_policy') != 'current_turn':
        raise ValueError('The configuration requires the expected name and current_turn images')
    limits = config['prime_turns_by_difficulty']
    if set(limits) != {'simple', 'moderate', 'complex'} or any(type(n) is not int or n < 1 for n in limits.values()):
        raise ValueError('Prime turn limits must be positive integers for simple, moderate, complex')
    if any(arg == '--prime-steps' or arg.startswith('--prime-steps=') for arg in remaining):
        raise ValueError(f'Set Prime turns in {options.config}; --prime-steps would bypass the difficulty policy')
    # User values follow presets so explicit overrides retain their precedence.
    base = [*config['runner_options'], 'robolab', *EXECUTION_DEFAULTS, *remaining]
    if mandatory_observation:
        base.append('--observe-before-grasp')
    args = configure_args(parse_args(base))
    label = options.difficulty
    source = None
    if label == 'auto':
        source = options.task_metadata or metadata_path()
        entries = [row for row in json.loads(source.read_text()) if row['task_name'] == args.task]
        if len(entries) != 1:
            raise ValueError(f'Expected one difficulty entry for {args.task}; found {len(entries)}')
        label = entries[0]['difficulty_label']
    if label not in limits:
        raise ValueError(f'Unsupported RoboLab difficulty: {label}')
    steps = limits[label]
    resolved = [*base, '--prime-steps', str(steps)]
    configuration = RuntimeConfiguration(cgn_service_url=options.cgn_service_url, cgn_timeout_s=options.cgn_timeout_s,
        waypoint_path_collision_checks=options.waypoint_path_collision_checks,
        grasp_path_collision_checks=options.grasp_path_collision_checks,
        mandatory_observation=mandatory_observation)
    configuration.client()
    metadata = configuration.metadata()
    record = dict(configuration=configuration_name, task=args.task, native_difficulty=label,
        difficulty=label, prime_steps=steps, image_history_policy=args.image_history_policy,
        difficulty_source=str(source.resolve()) if source else 'explicit --difficulty',
        config_path=str(options.config.resolve()),
        config_sha256=hashlib.sha256(options.config.read_bytes()).hexdigest(), config=config, argv=resolved,
        task_grasp_budget=args.task_grasp_budget, pose_dedup={'enabled': False},
        grasp_policy='contact_graspnet_and_observed_centers',
        grasp_motion_policy=args.grasp_motion_policy, grasp_score_tolerance=args.grasp_score_tolerance,
        manipulation=metadata, decision_playbook=metadata['playbook'],
        observe_before_grasp=args.observe_before_grasp, anyplace=False,
        auto_refine_routes=args.auto_refine_routes, transit_policy='independent_pre_post',
        append_only_observations=args.append_only_observations)
    if source:
        record['metadata_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    from src.core.provenance import _require_value_id
    episode_id = configuration_name.lower()+'-robolab-'+re.sub(r'[^a-z0-9]+', '-', args.task.lower()).strip('-')+'-seed-'+str(args.seed)
    _require_value_id(episode_id, context='episode_id')
    record['episode_id'] = episode_id
    return resolved, record, options.dry_run, configuration
