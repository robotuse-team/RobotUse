"""The standalone RobotUse entrypoint preserves observation/review and pose previews."""
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/run'))
from src.runtime import cli as run_episode
from src.runtime import episode as runner
from runtime_test_support import flags


def test_dry_run_preserves_default_optional_policies(tmp_path):
    argv, record, dry, config = run_episode.resolve([*flags(tmp_path), '--dry-run'])
    args = runner.parse_args(argv)
    assert dry and record['configuration'] == 'RobotUse'
    assert 'base' not in record and record['manipulation']['name'] == 'RobotUse'
    assert record['episode_id'] == 'robotuse-robolab-bananainbowltask-seed-0'
    assert not args.observe_before_grasp and not config.mandatory_observation
    assert not record['observe_before_grasp'] and record['robotuse']['observation_failure_waiver']
    assert not args.auto_refine_routes and not args.append_only_observations
    assert args.pause_refine and record['robotuse']['refiner_review'] == ['paused_pregrasp']
    assert config.pose_preview_editor and record['robotuse']['pose_preview_editor']
    assert not config.waypoint_path_collision_checks and not config.grasp_path_collision_checks
    assert args.sam_device == 'cuda:0' and args.max_restarts == 0
    assert record['decision_playbook'] == record['manipulation']['playbook']
    assert record['decision_playbook']['version'] == '3'
    assert record['decision_playbook']['path'] == str(run_episode.PLAYBOOK_V3_PATH.resolve())
    assert record['decision_playbook']['sha256'] == config.load_playbook().sha256
    assert not (tmp_path / 'live').exists()


@pytest.mark.parametrize('observe', [False, True])
@pytest.mark.parametrize('review', [False, True])
@pytest.mark.parametrize('custom_config', [False, True])
def test_observation_and_review_are_independent_opt_ins(tmp_path, observe, review, custom_config):
    options = (['--observe-before-grasp'] if observe else []) + (['--auto-refine-routes'] if review else [])
    custom = json.loads(run_episode.DEFAULT_CONFIG.read_text())
    if custom_config:
        custom['runner_options'].extend(options)
        path = tmp_path / 'custom.json'
        path.write_text(json.dumps(custom))
        options = ['--config', str(path)]
    argv, record, _, config = run_episode.resolve([*flags(tmp_path), *options, '--dry-run'])
    args = runner.parse_args(argv)
    assert args.observe_before_grasp is observe
    assert record['observe_before_grasp'] is record['robotuse']['observe_before_grasp'] is observe
    assert not config.mandatory_observation and record['robotuse']['observation_failure_waiver']
    assert args.auto_refine_routes is review
    assert record['auto_refine_routes'] is record['robotuse']['auto_refine_routes'] is review
    assert record['robotuse']['refiner_review'] == (
        ['grasp_candidate', 'paused_pregrasp', 'place_candidate'] if review else ['paused_pregrasp'])
    assert args.pause_refine and config.pose_preview_editor


@pytest.mark.parametrize('visible,physical', [('1', 1), ('4,2', 4), ('GPU-assigned', None), ('', None)])
def test_dispatch_records_configuration_and_gpu(tmp_path, monkeypatch, visible, physical):
    import src.simulator.robolab.adapter as cli

    def run(argv, configuration):
        assert os.environ['CUDA_VISIBLE_DEVICES'] == visible
        assert configuration.metadata()['gpu'] == f'CUDA_VISIBLE_DEVICES={visible}; logical cuda:0'
        args = runner.parse_args(argv)
        assert not args.observe_before_grasp and not args.auto_refine_routes
        assert not configuration.mandatory_observation
        args.output_dir.mkdir()
        return 0

    from src.tools.grasp import service as cgn_preflight
    monkeypatch.setattr(cgn_preflight, 'ensure_cgn', lambda *a, **k: {'ready': True})
    monkeypatch.setattr(run_episode, 'run_episode', run)
    monkeypatch.setattr(cli, 'run_native_cli', lambda call: call())
    monkeypatch.delenv('ROBOTUSE_GPU', raising=False)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    assert run_episode.main(flags(tmp_path)) == 0
    record = json.loads((tmp_path / 'live' / 'robotuse_configuration.json').read_text())
    assert record['configuration'] == 'RobotUse' and not record['manipulation']['mandatory_observation']
    assert record['physical_gpu'] == physical
    assert record['cuda_visible_devices'] == visible
    assert record['robotuse']['refiner_review'] == ['paused_pregrasp']


def test_runtime_selects_editor_backend_and_refiner(tmp_path):
    from src.runtime.configuration import classes
    from src.tools.pose_editor.adapter import PoseEditorMixin
    from src.backend.orchestrator import AgentOrchestrator
    _, _, _, config = run_episode.resolve([*flags(tmp_path), '--dry-run'])
    backend, orchestrator = classes(pose_preview_editor=config.pose_preview_editor)
    assert issubclass(backend, PoseEditorMixin)
    assert orchestrator is AgentOrchestrator
    assert issubclass(classes()[0], PoseEditorMixin)
    with pytest.raises(ValueError, match='requires its pose preview editor'):
        classes(pose_preview_editor=False)


@pytest.mark.parametrize('observe', [False, True])
def test_shared_runtime_preserves_optional_observation_before_simulator(tmp_path, monkeypatch, observe):
    import src.runtime.configuration as runtime

    class BeforeSimulator(Exception):
        pass

    def stop(args, configuration):
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
        assert args.observe_before_grasp is observe
        assert not configuration.mandatory_observation
        raise BeforeSimulator

    monkeypatch.setattr(runtime, 'validate_runtime', stop)
    monkeypatch.delenv('ROBOTUSE_GPU', raising=False)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    argv, _, _, config = run_episode.resolve([*flags(tmp_path), *(['--observe-before-grasp'] if observe else [])])
    with pytest.raises(BeforeSimulator):
        runner.run_episode(argv, config)
    assert not (tmp_path / 'live').exists()


@pytest.mark.parametrize('gpu', ['0', '3', '7'])
def test_explicit_queue_gpu_survives_shared_entrypoint(tmp_path, monkeypatch, gpu):
    import src.simulator.robolab.adapter as cli
    from src.tools.grasp import service as cgn_preflight
    monkeypatch.setenv('ROBOTUSE_GPU', gpu)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    def run(argv, configuration):
        assert os.environ['CUDA_VISIBLE_DEVICES'] == gpu
        assert configuration.metadata()['gpu'] == f'CUDA_VISIBLE_DEVICES={gpu}; logical cuda:0'
        runner.parse_args(argv).output_dir.mkdir()
        return 0
    monkeypatch.setattr(cgn_preflight, 'ensure_cgn', lambda *a, **k: {'ready': True})
    monkeypatch.setattr(run_episode, 'run_episode', run)
    monkeypatch.setattr(cli, 'run_native_cli', lambda call: call())
    assert run_episode.main(flags(tmp_path)) == 0
    record = json.loads((tmp_path / 'live' / 'robotuse_configuration.json').read_text())
    assert record['physical_gpu'] == int(gpu)
    assert record['manipulation']['gpu'] == f'CUDA_VISIBLE_DEVICES={gpu}; logical cuda:0'
