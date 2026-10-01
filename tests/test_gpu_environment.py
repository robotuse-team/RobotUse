"""GPU selection preserves externally assigned device visibility."""

import os
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.utils.gpu import gpu_visibility, physical_gpu_index


@pytest.mark.parametrize('variable', ['ROBOTUSE_GPU', 'ROBOTUSE_CGN_GPU'])
def test_no_gpu_configuration_keeps_default_zero(variable):
    assert gpu_visibility(variable, {}) == '0'


@pytest.mark.parametrize('visible', ['3', '5,2', 'GPU-assigned', 'MIG-assigned', '', '-1'])
def test_inherited_allocation_is_preserved(visible):
    assert gpu_visibility(environ={'CUDA_VISIBLE_DEVICES': visible}) == visible


@pytest.mark.parametrize('invalid', ['', '-1', '1,2', 'GPU-assigned', 'not-a-device'])
def test_invalid_explicit_index_does_not_silently_fall_back(invalid):
    with pytest.raises(ValueError, match='ROBOTUSE_GPU'):
        gpu_visibility(environ={'ROBOTUSE_GPU': invalid, 'CUDA_VISIBLE_DEVICES': '2'})


def test_explicit_gpu_overrides_inherited_allocation_without_mutating_input():
    env = {'ROBOTUSE_GPU': '7', 'CUDA_VISIBLE_DEVICES': '2,4'}
    assert gpu_visibility(environ=env) == '7'
    assert env['CUDA_VISIBLE_DEVICES'] == '2,4'


@pytest.mark.parametrize('visible,physical', [('3', 3), ('5,2', 5), ('GPU-assigned', None), ('', None), ('-1', None)])
def test_physical_index_is_not_invented_for_unknown_identifiers(visible, physical):
    assert physical_gpu_index(visible) == physical


def test_simulator_gpu_does_not_change_cgn_inherited_allocation(tmp_path, monkeypatch):
    from src.runtime import cli
    from src.tools.grasp import service
    from runtime_test_support import flags

    monkeypatch.setenv('ROBOTUSE_GPU', '7')
    monkeypatch.delenv('ROBOTUSE_CGN_GPU', raising=False)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,4')
    selections = []

    def preflight(*args, **kwargs):
        selections.append(gpu_visibility('ROBOTUSE_CGN_GPU'))
        return {'ready': True}

    def execute(*args):
        selections.append(os.environ['CUDA_VISIBLE_DEVICES'])
        return 0

    monkeypatch.setattr(service, 'ensure_cgn', preflight)
    monkeypatch.setattr(cli, 'execute', execute)
    assert cli.main(flags(tmp_path)) == 0
    assert selections == ['2,4', '7']


@pytest.mark.parametrize('profile', ['robolab', 'sam2', 'cgn', 'curobo'])
@pytest.mark.parametrize('visibility,index', [('4,2', 4), ('GPU-assigned', None), ('', None)])
def test_setup_check_matches_runtime_allocation(tmp_path, monkeypatch, profile, visibility, index):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('gpu_setup_check', root / 'scripts/check/setup.py')
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    monkeypatch.delenv('ROBOTUSE_GPU', raising=False)
    monkeypatch.delenv('ROBOTUSE_CGN_GPU', raising=False)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visibility)
    monkeypatch.setenv('CUDA_DEVICE_ORDER', 'FASTEST_FIRST')

    def check_interpreter(command, **kwargs):
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == visibility
        assert kwargs['env']['CUDA_DEVICE_ORDER'] == 'FASTEST_FIRST'
        return SimpleNamespace(returncode=0, stdout=json.dumps(dict(
            python=checker.PROFILES[profile], virtual_environment=True, mismatches={},
            gpu={'name': 'mock-device'})))

    monkeypatch.setattr(checker.subprocess, 'run', check_interpreter)
    result = checker.check_environment(tmp_path, profile, gpu=True)
    assert result['gpu']['physical_index'] == index
    assert result['gpu']['cuda_visible_devices'] == visibility
