"""Public CLI options reject unsupported pipelines and preserve native behavior."""
import json
import re

import pytest

from src.runtime import cli
from src.runtime.arguments import make_grasp_clearance_policy, parse_args
from src.runtime.configuration import configure_args
from runtime_test_support import flags


@pytest.fixture(autouse=True)
def forbid_runtime_startup(monkeypatch):
    from src.simulator.robolab import adapter
    from src.tools.grasp import service

    def unexpected_startup(*args, **kwargs):
        pytest.fail('CLI validation must finish before service, simulator, or agent startup')

    monkeypatch.setattr(service, 'ensure_cgn', unexpected_startup)
    monkeypatch.setattr(adapter, 'run_native_cli', unexpected_startup)
    monkeypatch.setattr(adapter, 'create_connector', unexpected_startup)
    monkeypatch.setattr(cli, 'run_episode', unexpected_startup)
    monkeypatch.setenv('ROBOTUSE_GPU', '0')
    monkeypatch.delenv('GRASPNET_SERVICE_URL', raising=False)
    monkeypatch.delenv('ROBOT_LLM_PROVIDER', raising=False)


@pytest.mark.parametrize('options', [
    ['--candidate-pool', '8'],
    ['--candidate-p', '8'],
    ['--grasp-policy', 'agent-choice'],
    ['--grasp-policy=agent-choice'],
    ['--grasp-motion-policy=source-order'],
    ['--grasp-score-tolerance=0'],
    ['--max-restarts=0'],
    ['--anyplace'],
    ['--no-task-runner'],
    ['--world-planning'],
    ['--object-cloud-policy=fused'],
    ['--graspgen-python', '/unused/python'],
    ['--moveit-scene-assets=/unused/assets'],
    ['--decision-playbook=/unused/policy.md'],
])
def test_removed_options_fail_before_runtime_startup(tmp_path, capsys, options):
    with pytest.raises(SystemExit) as exc:
        cli.main([*flags(tmp_path), *options])
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert 'unrecognized arguments' in error
    assert options[0].split('=')[0] in error
    assert not (tmp_path / 'live').exists()
    assert not list(tmp_path.glob('*-cgn-preflight-*'))


def test_custom_config_cannot_restore_removed_options(tmp_path, capsys):
    custom = json.loads(cli.DEFAULT_CONFIG.read_text())
    custom['runner_options'].append('--candidate-pool=8')
    path = tmp_path / 'custom.json'
    path.write_text(json.dumps(custom))
    with pytest.raises(SystemExit) as exc:
        cli.main([*flags(tmp_path), '--config', str(path)])
    assert exc.value.code == 2
    assert '--candidate-pool=8' in capsys.readouterr().err
    assert not (tmp_path / 'live').exists()
    assert not list(tmp_path.glob('*-cgn-preflight-*'))


def test_resolved_defaults_preserve_current_execution_policy(tmp_path):
    resolved, record, dry, configuration = cli.resolve([*flags(tmp_path), '--dry-run'])
    args = configure_args(parse_args(resolved))
    assert dry
    assert args.grasp_clearance_m == pytest.approx(.002)
    assert make_grasp_clearance_policy(args) == {
        'mode': 'fixed', 'initial_clearance_m': .002, 'floor_clearance_m': .002,
        'applies_to': ['moving_robot_scene', 'configured_self_pairs'],
    }
    assert record['grasp_motion_policy'] == 'source-order'
    assert record['grasp_score_tolerance'] == 0
    assert args.task_grasp_budget == record['task_grasp_budget'] == 1000
    assert args.motion_speed_scale == 3.
    assert args.gripper_open_settle_steps == args.gripper_close_settle_steps == 20
    assert args.object_cloud_policy == 'fused'
    assert args.image_history_policy == 'current_turn'
    assert args.max_restarts == 0
    assert args.intent_driven and args.review_driven and args.pause_refine and args.world_planning
    assert not args.task_runner and not args.anyplace and not args.grasp_scene_filter
    assert not args.observe_before_grasp and not args.auto_refine_routes
    assert not args.append_only_observations
    assert args.task_score
    assert not configuration.waypoint_path_collision_checks
    assert not configuration.grasp_path_collision_checks
    assert configuration.observed_transit_planner is None
    assert not (tmp_path / 'live').exists()


def test_effective_runtime_options_override_defaults(tmp_path):
    resolved, record, dry, configuration = cli.resolve([
        *flags(tmp_path), '--dry-run',
        '--observe-before-grasp', '--auto-refine-routes', '--append-only-observations',
        '--no-grasp-clearance-batch-decay', '--grasp-clearance-m', '.004',
        '--task-grasp-budget', '17', '--child-steps', '11',
        '--motion-speed-scale', '2.25',
        '--gripper-open-settle-steps', '13', '--gripper-close-settle-steps', '17',
        '--waypoint-path-collision-checks', '--grasp-path-collision-checks',
        '--no-task-score', '--no-randomize-init-pose', '--no-adaptive-grasp-opening',
    ])
    args = configure_args(parse_args(resolved))
    assert dry
    assert args.observe_before_grasp and args.auto_refine_routes
    assert record['observe_before_grasp'] and record['auto_refine_routes']
    assert args.append_only_observations and record['append_only_observations']
    assert record['robotuse']['refiner_review'] == [
        'grasp_candidate', 'paused_pregrasp', 'place_candidate',
    ]
    assert not args.grasp_clearance_batch_decay
    assert make_grasp_clearance_policy(args)['initial_clearance_m'] == pytest.approx(.004)
    assert args.task_grasp_budget == record['task_grasp_budget'] == 17
    assert args.child_steps == 11
    assert args.motion_speed_scale == 2.25
    assert args.gripper_open_settle_steps == 13
    assert args.gripper_close_settle_steps == 17
    assert configuration.waypoint_path_collision_checks
    assert configuration.grasp_path_collision_checks
    assert not args.task_score and not args.randomize_init_pose and not args.adaptive_grasp_opening
    assert not record['initial_pose_randomization']['enabled']
    assert not (tmp_path / 'live').exists()


def test_explicit_options_override_custom_config_values(tmp_path):
    custom = json.loads(cli.DEFAULT_CONFIG.read_text())
    custom['runner_options'].extend([
        '--task-grasp-budget', '29', '--motion-speed-scale', '2',
        '--append-only-observations', '--no-task-score',
    ])
    path = tmp_path / 'custom.json'
    path.write_text(json.dumps(custom))
    resolved, record, _, _ = cli.resolve([
        *flags(tmp_path), '--config', str(path), '--dry-run',
        '--task-grasp-budget', '17', '--motion-speed-scale', '2.25',
        '--no-append-only-observations', '--task-score',
    ])
    args = configure_args(parse_args(resolved))
    assert args.task_grasp_budget == record['task_grasp_budget'] == 17
    assert args.motion_speed_scale == 2.25
    assert not args.append_only_observations
    assert args.task_score
    assert not (tmp_path / 'live').exists()


def test_help_lists_current_options_without_requiring_output_directory(capsys):
    try:
        code = cli.main(['--help'])
    except SystemExit as exc:
        code = exc.code
    assert code == 0
    captured = capsys.readouterr()
    assert not captured.err
    options = set(re.findall(r'--[a-z][a-z0-9-]*', captured.out))
    assert {
        '--task', '--output-dir', '--dry-run', '--config', '--difficulty',
        '--playbook-version', '--transit-planner', '--curobo-robot-file',
        '--curobo-calibration-file', '--model', '--sam-python',
        '--observe-before-grasp', '--auto-refine-routes', '--append-only-observations',
        '--grasp-clearance-batch-decay', '--waypoint-path-collision-checks',
        '--grasp-path-collision-checks', '--task-score', '--no-task-score',
    } <= options
    assert not {
        '--candidate-pool', '--grasp-policy', '--grasp-motion-policy',
        '--grasp-score-tolerance', '--max-restarts', '--anyplace',
        '--task-runner', '--no-task-runner', '--world-planning',
        '--object-cloud-policy', '--graspgen-python', '--moveit-scene-assets',
        '--decision-playbook',
    } & options
