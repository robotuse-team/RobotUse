"""Native progress is reported separately from reward and task completion."""
from copy import deepcopy
import builtins
import sys
from types import ModuleType, SimpleNamespace

import pytest

from src.simulator.robolab.verifier import task_score
from src.ui.runner import verifier_status


@pytest.fixture
def native_score(monkeypatch):
    module = ModuleType('robolab.core.events.subtask_recorder')

    class SubtaskCompletionRecorderTerm:
        pass

    module.SubtaskCompletionRecorderTerm = SubtaskCompletionRecorderTerm
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def make(score, *, total=1):
        state = dict(score=score, completed=int(score == 1), total=total)
        term = SubtaskCompletionRecorderTerm()
        term.subtask_state_machines = [SimpleNamespace(get_subtask_state=lambda: state)]
        term.infos = [dict(score=1., completed=1, total=1)]

        def get_term(kind):
            assert kind is SubtaskCompletionRecorderTerm
            return term

        native = SimpleNamespace(recorder_manager=SimpleNamespace(get_term=get_term))
        connector = SimpleNamespace(env=SimpleNamespace(_env=native))
        return connector, term, state

    return make


@pytest.mark.parametrize('value', [0, 0., .375, 1, 1.])
def test_reports_native_normalized_score(native_score, value):
    connector, _, _ = native_score(value)
    result = task_score(connector, enabled=True)
    assert result == dict(score_enabled=True, score_evaluated=True, score=float(value),
                          score_source='robolab.SubtaskStateMachine.get_subtask_state',
                          score_error=None)


def test_disabled_score_does_not_import_native_or_access_connector(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith('robolab'):
            pytest.fail('Disabled score imported the native simulator')
        return original_import(name, *args, **kwargs)

    class UnavailableConnector:
        @property
        def env(self):
            pytest.fail('Disabled score accessed the simulator')

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    assert task_score(UnavailableConnector(), enabled=False) == dict(
        score_enabled=False, score_evaluated=False, score=None,
        score_source=None, score_error=None)


@pytest.mark.parametrize('missing', ['manager', 'term', 'state_machine', 'subtasks'])
def test_missing_native_score_is_unavailable_not_zero(native_score, missing):
    connector, term, state = native_score(0.)
    if missing == 'manager':
        connector.env._env.recorder_manager = None
    elif missing == 'term':
        connector.env._env.recorder_manager.get_term = lambda kind: None
    elif missing == 'state_machine':
        term.subtask_state_machines = []
    else:
        state['total'] = 0
    result = task_score(connector, enabled=True)
    assert result['score_enabled'] is True
    assert result['score_evaluated'] is False
    assert result['score'] is None
    assert result['score_source'] is None
    assert result['score_error'].startswith('ValueError: Native ')


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'),
                                 -.1, 1.1, True, None, '1.0'])
def test_invalid_score_does_not_become_a_numeric_verdict(native_score, value):
    connector, _, _ = native_score(value)
    result = task_score(connector, enabled=True)
    assert result['score_evaluated'] is False
    assert result['score'] is None
    assert result['score_error'] == (
        'ValueError: Native subtask score must be a finite number between 0 and 1')


def test_reset_state_wins_over_stale_recorder_info(native_score):
    connector, term, state = native_score(0.)
    assert term.infos[0]['score'] == 1.
    before = deepcopy((state, term.infos))
    first = task_score(connector, enabled=True)
    second = task_score(connector, enabled=True)
    assert first == second
    assert first['score'] == 0.
    assert (state, term.infos) == before
    # Reading again sees the current state rather than caching an earlier score.
    state['score'] = .5
    assert task_score(connector, enabled=True)['score'] == .5


def test_score_read_failure_keeps_existing_success_and_reward(native_score):
    connector, term, _ = native_score(1.)

    def unavailable():
        raise RuntimeError('Recorder is unavailable')

    term.subtask_state_machines[0].get_subtask_state = unavailable
    verdict = dict(evaluated=True, task_success=True, reward=0.)
    verdict.update(task_score(connector, enabled=True))
    assert verdict['evaluated'] is True
    assert verdict['task_success'] is True
    assert verdict['reward'] == 0.
    assert verdict['score'] is None
    assert verdict['score_error'] == 'RuntimeError: Recorder is unavailable'


@pytest.mark.parametrize(('verdict', 'score', 'expected'), [
    (dict(evaluated=True, task_success=True, reward=0.), 0., 'Passed · Score 0.000'),
    (dict(evaluated=True, task_success=False, reward=0.), 1., 'Failed · Score 1.000'),
    (dict(evaluated=True, task_success=False), .375, 'Failed · Score 0.375'),
    (dict(evaluated=False, task_success=True), 1., 'Not evaluated · Score 1.000'),
    (dict(evaluated=True, task_success='true'), 1., 'Unknown · Score 1.000'),
])
def test_ui_keeps_success_independent_from_score(verdict, score, expected):
    assert verifier_status(dict(verdict, score_enabled=True,
                                score_evaluated=True, score=score)) == expected


@pytest.mark.parametrize(('extra', 'expected'), [
    (dict(score_enabled=True, score_evaluated=False, score=None), 'Passed · Score unavailable'),
    (dict(score_enabled=False, score_evaluated=False, score=None), 'Passed · Score off'),
    ({}, 'Passed'),
])
def test_ui_displays_unavailable_disabled_and_legacy_score(extra, expected):
    assert verifier_status(dict(evaluated=True, task_success=True, **extra)) == expected


@pytest.mark.parametrize('value', [True, '1', float('nan'), float('inf'), -.1, 1.1])
def test_ui_does_not_display_invalid_score(value):
    assert verifier_status(dict(evaluated=True, task_success=False, score_enabled=True,
                                score_evaluated=True, score=value)) == 'Failed · Score unavailable'


@pytest.mark.parametrize(('option', 'enabled'), [([], True), (['--task-score'], True), (['--no-task-score'], False)])
def test_episode_cli_score_defaults_on_and_can_be_disabled(tmp_path, option, enabled):
    from src.runtime.arguments import parse_args
    from src.runtime.cli import resolve

    resolved, _, dry, _ = resolve(['--task', 'BananaInBowlTask', '--difficulty', 'simple',
                                   '--output-dir', str(tmp_path / 'episode'), '--dry-run', *option])
    assert dry is True
    assert parse_args(resolved).task_score is enabled
    assert not (tmp_path / 'episode').exists()


@pytest.mark.parametrize(('option', 'enabled'), [([], True), (['--task-score'], True), (['--no-task-score'], False)])
def test_native_check_uses_the_same_score_default(monkeypatch, tmp_path, option, enabled):
    import argparse
    from pathlib import Path
    import runpy

    namespace = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/check/native.py'))
    original = argparse.ArgumentParser.parse_args

    class ArgumentsChecked(Exception):
        pass

    def inspect(parser):
        args = original(parser, ['--output-dir', str(tmp_path / 'episode'),
                                '--sam-python', 'python', '--sam2-snapshot', 'checkpoint', *option])
        assert args.task_score is enabled
        raise ArgumentsChecked

    monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', inspect)
    with pytest.raises(ArgumentsChecked):
        namespace['main']()
    assert not (tmp_path / 'episode').exists()
