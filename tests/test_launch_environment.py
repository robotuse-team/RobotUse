"""Launch shell entry points with an offline interpreter stub, never the UI or GPU."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launch_tree(tmp_path):
    for relative in ('scripts/run/ui.sh', 'scripts/run/ui_openrouter.sh', 'scripts/lib/env.sh'):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    return tmp_path


def interpreter_stub(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!{sys.executable}\n' + '''
import json, os, sys
names = ('ROBOTUSE_GPU', 'ROBOTUSE_CGN_GPU', 'CUDA_VISIBLE_DEVICES',
         'ROBOT_LLM_PROVIDER', 'ROBOT_LLM_MODEL', 'OMNI_KIT_ACCEPT_EULA')
print(json.dumps({'interpreter': sys.argv[0], 'args': sys.argv[1:],
                  'environment': {name: os.environ.get(name) for name in names},
                  'key_present': 'OPENROUTER_API_KEY' in os.environ}))
''')
    path.chmod(0o755)


def launch_environment(root):
    env = os.environ.copy()
    for name in ('ROBOTUSE_GPU', 'ROBOTUSE_CGN_GPU', 'CUDA_VISIBLE_DEVICES',
                 'ROBOT_LLM_PROVIDER', 'ROBOT_LLM_MODEL', 'OMNI_KIT_ACCEPT_EULA',
                 'OPENROUTER_API_KEY', 'ROBOTUSE_UI_PYTHON', 'UI_ENVIRONMENT'):
        env.pop(name, None)
    env['ROBOTUSE_RUNTIME_ROOT'] = str(root / 'runtime')
    return env


def launch(root, env, *arguments):
    result = subprocess.run([str(root / 'scripts/run/ui_openrouter.sh'), *arguments],
                            env=env, cwd=root, capture_output=True, text=True,
                            check=True, timeout=20)
    return json.loads(result.stdout)


@pytest.mark.parametrize('selection', ['runtime', 'environment', 'interpreter', 'empty'])
def test_ui_launch_interpreter_precedence_without_credentials(launch_tree, selection):
    env = launch_environment(launch_tree)
    environment = launch_tree / 'custom ui'
    explicit_python = launch_tree / 'explicit python'
    expected = launch_tree / 'runtime/tool-envs/ui/bin/python'
    if selection in ('environment', 'interpreter'):
        env['UI_ENVIRONMENT'] = str(environment)
        expected = environment / 'bin/python'
    if selection == 'interpreter':
        env['ROBOTUSE_UI_PYTHON'] = str(explicit_python)
        expected = explicit_python
    if selection == 'empty':
        env.update(UI_ENVIRONMENT='', ROBOTUSE_UI_PYTHON='')
    interpreter_stub(expected)
    result = launch(launch_tree, env, '--port', '9000')
    assert result['interpreter'] == str(expected)
    assert result['args'] == ['-m', 'src.ui.app', '--port', '9000']
    assert result['key_present'] is False
    assert result['environment'] == dict(
        ROBOTUSE_GPU='0', ROBOTUSE_CGN_GPU='0', CUDA_VISIBLE_DEVICES=None,
        ROBOT_LLM_PROVIDER='openrouter', ROBOT_LLM_MODEL='google/gemini-3.8-flash',
        OMNI_KIT_ACCEPT_EULA=None)


@pytest.mark.parametrize('visibility', ['2,4', 'GPU-assigned-device', ''])
@pytest.mark.parametrize('explicit_gpu', [None, '3'])
def test_preset_preserves_allocations_model_and_eula(launch_tree, visibility, explicit_gpu):
    env = launch_environment(launch_tree)
    env.update(CUDA_VISIBLE_DEVICES=visibility, ROBOT_LLM_PROVIDER='google',
               ROBOT_LLM_MODEL='custom/provider-model', OMNI_KIT_ACCEPT_EULA='N')
    if explicit_gpu is not None:
        env.update(ROBOTUSE_GPU=explicit_gpu, ROBOTUSE_CGN_GPU=explicit_gpu)
    interpreter_stub(launch_tree / 'runtime/tool-envs/ui/bin/python')
    result = launch(launch_tree, env)
    assert result['environment'] == dict(
        ROBOTUSE_GPU=explicit_gpu, ROBOTUSE_CGN_GPU=explicit_gpu,
        CUDA_VISIBLE_DEVICES=visibility, ROBOT_LLM_PROVIDER='openrouter',
        ROBOT_LLM_MODEL='custom/provider-model', OMNI_KIT_ACCEPT_EULA='N')


def test_empty_explicit_gpu_is_preserved_for_runtime_validation(launch_tree):
    env = launch_environment(launch_tree)
    env.update(ROBOTUSE_GPU='', ROBOTUSE_CGN_GPU='')
    interpreter_stub(launch_tree / 'runtime/tool-envs/ui/bin/python')
    result = launch(launch_tree, env)
    assert result['environment']['ROBOTUSE_GPU'] == ''
    assert result['environment']['ROBOTUSE_CGN_GPU'] == ''


@pytest.mark.parametrize('profile', ['cpu', 'ui'])
@pytest.mark.parametrize('selection', ['runtime', 'environment', 'interpreter', 'empty'])
def test_environment_check_uses_launch_interpreter_precedence(tmp_path, monkeypatch, profile, selection):
    spec = importlib.util.spec_from_file_location('launch_setup_check', ROOT / 'scripts/check/setup.py')
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    environment_name = f'{profile.upper()}_ENVIRONMENT'
    interpreter_name = f'ROBOTUSE_{profile.upper()}_PYTHON'
    monkeypatch.delenv(environment_name, raising=False)
    monkeypatch.delenv(interpreter_name, raising=False)
    runtime = tmp_path / 'runtime'
    environment = tmp_path / 'custom environment'
    expected = runtime / 'tool-envs' / profile / 'bin/python'
    if selection in ('environment', 'interpreter'):
        monkeypatch.setenv(environment_name, str(environment))
        expected = environment / 'bin/python'
    if selection == 'interpreter':
        expected = tmp_path / 'explicit python'
        monkeypatch.setenv(interpreter_name, str(expected))
    if selection == 'empty':
        monkeypatch.setenv(environment_name, '')
        monkeypatch.setenv(interpreter_name, '')
    called = []

    def check_interpreter(command, **kwargs):
        called.append(command[0])
        return SimpleNamespace(returncode=0, stdout=json.dumps(dict(
            python='3.11.9', virtual_environment=True, mismatches={})))

    monkeypatch.setattr(checker.subprocess, 'run', check_interpreter)
    checker.check_environment(runtime, profile, gpu=False)
    assert called == [str(expected)]


def test_actual_provider_still_requires_key_without_network():
    from src.llm.config import ProviderConfig

    config = ProviderConfig.resolve(provider='openrouter', environ={})
    with pytest.raises(RuntimeError, match='OPENROUTER_API_KEY is missing'):
        config.key(environ={})
