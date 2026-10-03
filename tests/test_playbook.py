"""CPU checks for RobotUse policy selection, live snapshots, and actual role prompts."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/run'))
from src.runtime import cli as run_episode
from src.runtime import episode as runner
from src.agent.playbook.loader import ROLES, DecisionPlaybook
from src.runtime.configuration import RuntimeConfiguration
from src.backend.orchestrator import AgentOrchestrator
from test_prime_delegation import Factory, a
from runtime_test_support import flags
from test_orchestrator import Backend


def test_default_configuration_loads_v3_playbook():
    config = RuntimeConfiguration()
    assert config.load_playbook() == DecisionPlaybook.load(run_episode.POLICY_PATHS['3'])
    assert config.playbook_version == '3'
    assert config.load_playbook().path.name == 'v3.md'


@pytest.mark.parametrize('role', [role for role in ROLES if role != 'common'])
@pytest.mark.parametrize('version', ['v0', 'v1', 'v2', 'v3'])
def test_every_role_receives_its_own_section_and_common_guidance(tmp_path, role, version):
    _, record, _, config = run_episode.resolve([*flags(tmp_path), '--playbook-version', version, '--dry-run'])
    book = config.load_playbook()
    factory = Factory(**{role: [[a('finish', status='failed', reason='no_candidates')]]})
    orchestrator = AgentOrchestrator(Backend(), factory, decision_playbook=book)
    status, _ = orchestrator._loop(role, {'instruction': 'Check the current requested object.'})
    assert status == 'completed'
    assert not [event for event in orchestrator._events
                if event['kind'] in ('action_rejected', 'backend_error', 'model_error')]
    prompt = factory.sessions[0][2].inputs[0][0][0]['content']
    assert book.sections['common'] in prompt and book.sections[role] in prompt
    assert f'DECISION PLAYBOOK sha256={book.sha256} role={role}' in prompt
    for other in ROLES:
        if other not in ('common', role):
            assert book.sections[other] not in prompt
    loaded = [event for event in orchestrator._events if event['kind'] == 'decision_playbook_loaded']
    assert len(loaded) == 1 and loaded[0]['role'] == role
    assert loaded[0]['sha256'] == record['decision_playbook']['sha256']


@pytest.mark.parametrize('entrypoint,options', [(run_episode, []),
    (run_episode, ['--playbook-version', 'v0']), (run_episode, ['--playbook-version', 'v1']), (run_episode, ['--playbook-version', 'v2'])])
def test_runtime_persisted_playbook_matches_resolved_configuration(tmp_path, monkeypatch, entrypoint, options):
    import src.simulator.robolab.configuration as profiles
    import src.runtime.configuration as runtime

    class BeforeSimulator(Exception):
        pass

    argv, dry_record, _, config = entrypoint.resolve([*flags(tmp_path), *options, '--dry-run'])
    monkeypatch.setattr(runtime, 'validate_runtime', lambda *args: None)
    monkeypatch.setattr(profiles, 'load_calibration', lambda args: (np.eye(4), None))
    original_write = runner.write_json

    def write_and_stop(path, value):
        original_write(path, value)
        # This executes the real startup snapshot path before any simulator or model worker.
        if path.name == 'calibration.json':
            raise BeforeSimulator

    monkeypatch.setattr(runner, 'write_json', write_and_stop)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    with pytest.raises(BeforeSimulator):
        runner.run_episode(argv, config)
    output = tmp_path / 'live'
    metadata = json.loads((output / 'decision_playbook.json').read_text())
    snapshot = (output / 'decision_playbook.md').read_bytes()
    assert metadata == dry_record['decision_playbook'] == dry_record['manipulation']['playbook']
    assert hashlib.sha256(snapshot).hexdigest() == metadata['sha256']
    assert snapshot == config.load_playbook().text.encode('utf-8')


@pytest.mark.parametrize('content', ['', '## common\nMissing roles'])
def test_invalid_playbook_fails_dry_run_before_startup(tmp_path, monkeypatch, content):
    path = tmp_path / 'invalid.md'
    path.write_text(content)
    monkeypatch.setitem(run_episode.POLICY_PATHS, '3', path)
    with pytest.raises(ValueError, match='decision playbook'):
        run_episode.resolve([*flags(tmp_path), '--dry-run'])
    assert not (tmp_path / 'live').exists()


@pytest.mark.parametrize('review', [False, True])
def test_prime_prompt_distinguishes_optional_candidate_review_from_paused_refiner(review):
    orchestrator = AgentOrchestrator(Backend(), Factory(), auto_refine_routes=review)
    prompt = orchestrator._prompt('prime')
    assert ('The selected grasp candidate is shown to Refiner before validation and execution.' in prompt) is review
    assert ('Only an explicit needs_refinement selection is sent to Refiner before execution.' in prompt) is (not review)
    assert 'separate paused pregrasp Refiner' in prompt
    assert 'Every selected pose is shown to Refiner' not in prompt


@pytest.mark.parametrize('option,version', [('v0', '0'), ('0', '0'), ('v1', '1'), ('1', '1'), ('v2', '2'), ('2', '2'), ('v3', '3'), ('3', '3')])
def test_explicit_playbook_selection_is_consumed_and_recorded(tmp_path, option, version):
    argv, record, dry, config = run_episode.resolve([
        *flags(tmp_path), '--playbook-version', option, '--dry-run'])
    assert dry and '--playbook-version' not in argv
    assert config.playbook_version == record['decision_playbook']['version'] == version
    assert config.playbook_path.name == f'v{version}.md'
    assert record['decision_playbook'] == record['manipulation']['playbook']
    assert config.load_playbook().sha256 == record['decision_playbook']['sha256']
    runner.parse_args(argv)  # The underlying runtime never receives the RobotUse-only option.
    assert not (tmp_path / 'live').exists()






@pytest.mark.parametrize('role', [role for role in ROLES if role != 'common'])
def test_v2_robot_base_convention_reaches_every_actual_role_prompt(tmp_path, role):
    _, _, _, config = run_episode.resolve([*flags(tmp_path), '--playbook-version', 'v2', '--dry-run'])
    book = config.load_playbook()
    factory = Factory(**{role: [[a('finish', status='failed', reason='test_only')]]})
    orchestrator = AgentOrchestrator(Backend(), factory, decision_playbook=book)
    orchestrator._loop(role, {'instruction': 'Check a robot-relative spatial relation.'})
    prompt = factory.sessions[0][2].inputs[0][0][0]['content']
    for rule in ('fixed ROBOT BASE coordinate frame (connector_base)',
                 'not in agentview, the left-shoulder camera, the wrist camera, or screen axes',
                 'toward base -X from that object',
                 'behind means +X, left means +Y, and right means -Y'):
        assert rule in prompt


def test_v2_spatial_handoff_and_checks_are_role_scoped(tmp_path):
    _, _, _, config = run_episode.resolve([*flags(tmp_path), '--playbook-version', 'v2', '--dry-run'])
    sections = config.load_playbook().sections
    assert 'compare the returned clicked base XY with the reference before finishing' in sections['point']
    assert 'reselect rather than describe a diagonal location as directly in front' in sections['point']
    assert 'in the existing finish reason' in sections['point']
    assert "a child's 'directly in front' label alone is not evidence" in sections['prime']
    assert "verify the actual object's relation in the same frame" in sections['prime']
    assert 'Preserve the intended anchor rather than replacing it' in sections['place']
    assert "held object's offset from the jaw contact center" in sections['place']
    assert 'front nearer the robot' not in config.load_playbook().text




@pytest.mark.parametrize('role', ['prime', 'grasp'])
def test_upright_mug_preference_reaches_actual_role_without_forcing_other_shapes(tmp_path, role):
    _, _, _, config = run_episode.resolve([*flags(tmp_path), '--playbook-version', 'v2', '--dry-run'])
    factory = Factory(**{role: [[a('finish', status='failed', reason='test_only')]]})
    orchestrator = AgentOrchestrator(Backend(), factory, decision_playbook=config.load_playbook())
    orchestrator._loop(role, {'instruction': 'Pick up an upright mug.'})
    prompt = factory.sessions[0][2].inputs[0][0][0]['content']
    assert 'visibly upright mug' in prompt
    assert 'median' in prompt
    assert 'sideways/inverted mugs' in prompt
    if role == 'grasp':
        assert 'a CGN call need not precede it' in prompt
        assert 'median success is not necessarily uncorrected success' in prompt
        assert 'Do not copy offsets from earlier scenes' in prompt
        assert 'motion nonconvergence' in prompt




@pytest.mark.parametrize('value', ['v4', '-1', 'improved', 'V1', ''])
def test_invalid_playbook_version_rejects_before_shared_runner(tmp_path, monkeypatch, value):
    monkeypatch.setattr(run_episode, 'resolve_configuration',
        lambda *a, **kw: pytest.fail('invalid version reached configuration resolver'))
    with pytest.raises(SystemExit) as exc:
        run_episode.resolve([*flags(tmp_path), '--playbook-version=' + value, '--dry-run'])
    assert exc.value.code == 2
    assert not (tmp_path / 'live').exists()


@pytest.mark.parametrize('options,expected', [([], '3'), (['--playbook-version=v0'], '0'), (['--playbook-version=v1'], '1'), (['--playbook-version=v2'], '2'), (['--playbook-version=v3'], '3')])
def test_dry_run_selected_playbook_never_probes_or_starts_cgn(tmp_path, monkeypatch, capsys, options, expected):
    from src.tools.grasp import service as cgn_preflight
    monkeypatch.setattr(cgn_preflight, 'ensure_cgn',
        lambda *a, **kw: pytest.fail('dry-run must not call CGN preflight'))
    assert run_episode.main([*flags(tmp_path), *options, '--dry-run']) == 0
    record = json.loads(capsys.readouterr().out)
    assert record['decision_playbook']['version'] == expected
    assert not (tmp_path / 'live').exists()
