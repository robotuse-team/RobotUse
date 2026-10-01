"""Exercise public entrypoints and startup failures without native execution."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.tools.base_tool import ToolRegistrationError


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_imports_work_before_legacy_imports_without_pythonpath():
    env = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
    result = subprocess.run([sys.executable, '-c',
        'import src.llm.manager; import src.backend.orchestrator; '
        'import src.tools.grasp.backend; import src.tools.place.execution'],
        cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_startup_cannot_load_historical_implementations(tmp_path):
    code = '''
import importlib.abc
import sys
from pathlib import Path

class RejectHistoricalSources(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'robot_skill_selector' or fullname.startswith('robot_skill_selector.'):
            raise AssertionError('Historical implementation requested: ' + fullname)

sys.meta_path.insert(0, RejectHistoricalSources())
from src.runtime.cli import resolve
from src.runtime.configuration import classes
from src.runtime.bootstrap import load_tool_registry
resolved, record, dry, configuration = resolve([
    '--task', 'BananaInBowlTask', '--difficulty', 'simple',
    '--output-dir', sys.argv[1], '--dry-run'])
backend, orchestrator = classes()
assert all(cls.__module__.startswith(('src.', 'builtins')) for cls in backend.__mro__)
assert all(cls.__module__.startswith(('src.', 'builtins')) for cls in orchestrator.__mro__)
assert record['decision_playbook']['version'] == '3'
assert configuration.observed_transit_planner is None
assert not any('skills_src' in path for path in sys.path)
from src.simulator.robolab.adapter import initialize
initialize()
aliases = {name for name in sys.modules if name.startswith('robot_skill_selector')}
assert aliases == {'robot_skill_selector', 'robot_skill_selector.robolab_cli',
                   'robot_skill_selector.robolab_collision'}
for name in aliases - {'robot_skill_selector'}:
    assert sys.modules[name].__name__.startswith('src.simulator.robolab.')
assert not Path(sys.argv[1]).exists()
'''
    env = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path / 'episode')],
        cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_canonical_source_has_no_import_of_historical_runners():
    import ast

    for path in (ROOT / 'src').rglob('*.py'):
        if 'third_party' in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or '']
            else:
                continue
            assert not any(name.startswith('robot_skill_selector') for name in names), path
            assert not any(name.startswith('run_') for name in names), path


def test_shell_entrypoint_works_from_other_directory_without_starting_runtime(tmp_path):
    env = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
    env['ROBOLAB_PYTHON'] = sys.executable
    output = tmp_path / 'episode-not-started'
    result = subprocess.run(['bash', str(ROOT / 'scripts/run/robolab.sh'),
        '--task', 'BananaInBowlTask', '--difficulty', 'simple',
        '--output-dir', str(output), '--dry-run'],
        cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    assert record['decision_playbook']['version'] == '3'
    assert 'transit_planner' not in record['robotuse']
    assert not output.exists()


@pytest.mark.parametrize('group,name', [
    ('gripper', 'release'), ('grasp', 'execute_grasp'),
])
def test_missing_public_or_internal_tool_fails_before_episode_start(monkeypatch, tmp_path, group, name):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts/run'))
    entry = importlib.import_module('episode')
    module = importlib.import_module(f'src.tools.{group}.tool')
    monkeypatch.setattr(module, 'TOOLS', tuple(spec for spec in module.TOOLS if spec.name != name))
    output = tmp_path / 'must-not-start'
    with pytest.raises(ToolRegistrationError):
        entry.main(['--task', 'BananaInBowlTask', '--output-dir', str(output)])
    assert not output.exists()


def test_optional_curobo_never_silently_uses_a_default_robot(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts/run'))
    entry = importlib.import_module('episode')
    with pytest.raises(ValueError, match='no default robot substitution'):
        entry.resolve(['--transit-planner', 'curobo', '--dry-run',
            '--task', 'BananaInBowlTask', '--output-dir', str(tmp_path / 'episode')])
