"""Provisioning must preserve sources, existing files and runtime isolation."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from src.tools.perception import sam2_adapter

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('provision_sam2', ROOT / 'scripts/setup/provision_sam2.py')
provisioner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(provisioner)


def test_checkpoint_verified_without_overwriting_or_redownloading(tmp_path):
    original = b'previously downloaded checkpoint'
    destination = tmp_path / 'checkpoint.pt'
    destination.write_bytes(original)
    model = {'sha256': hashlib.sha256(original).hexdigest(), 'url': 'https://invalid.invalid'}
    assert provisioner.provision(destination, model) == destination
    model['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='left unchanged'):
        provisioner.provision(destination, model)
    assert destination.read_bytes() == original


def test_failed_checkpoint_download_leaves_no_partial_final_file(tmp_path):
    origin = tmp_path / 'origin.pt'
    origin.write_bytes(b'wrong checkpoint')
    destination = tmp_path / 'model' / 'checkpoint.pt'
    with pytest.raises(ValueError, match='SHA-256'):
        provisioner.provision(destination, {'sha256': '0' * 64, 'url': origin.as_uri()})
    assert not list(destination.parent.iterdir())


def test_checkpoint_symlink_cannot_overwrite_another_installation(tmp_path):
    original = tmp_path / 'original.pt'
    original.write_bytes(b'preserve')
    linked = tmp_path / 'checkpoint.pt'
    linked.symlink_to(original)
    with pytest.raises(ValueError, match='symlink'):
        provisioner.provision(linked, {'sha256': '0' * 64})
    assert original.read_bytes() == b'preserve'


@pytest.mark.parametrize('profile', ['cpu', 'sam2', 'cgn', 'curobo', 'robolab'])
@pytest.mark.parametrize('protected', ['third_party', 'vendor', 'symlink'])
def test_setup_refuses_protected_runtime_before_calling_uv(tmp_path, profile, protected):
    runtime = tmp_path / protected
    target = tmp_path / 'preserved'
    target.mkdir()
    (target / 'sentinel').write_bytes(b'preserve')
    if protected == 'symlink':
        runtime.symlink_to(target, target_is_directory=True)
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    marker = tmp_path / 'uv-called'
    fake_uv = fake_bin / 'uv'
    fake_uv.write_text('#!/bin/sh\nprintf called > "$UV_TEST_MARKER"\nexit 99\n')
    fake_uv.chmod(0o755)
    env = dict(os.environ, ROBOTUSE_RUNTIME_ROOT=str(runtime), UV_TEST_MARKER=str(marker),
               PATH=str(fake_bin) + os.pathsep + os.environ['PATH'])
    for name in ('CPU_ENVIRONMENT', 'SAM2_ENVIRONMENT', 'SAM2_CHECKPOINT', 'CUROBO_ENVIRONMENT'):
        env.pop(name, None)
    result = subprocess.run(['bash', str(ROOT / 'scripts/setup' / f'{profile}.sh')],
                            env=env, cwd=tmp_path, text=True, capture_output=True, timeout=20)
    assert result.returncode != 0
    assert protected in result.stderr.lower(), result.stdout + result.stderr
    assert not marker.exists(), result.stdout + result.stderr
    assert (target / 'sentinel').read_bytes() == b'preserve'
    assert list(target.iterdir()) == [target / 'sentinel']


@pytest.mark.parametrize('override', ['ROBOTUSE_SAM_PYTHON', 'SAM_RUNTIME_PYTHON'])
def test_sam_launcher_and_preflight_use_same_isolated_override(tmp_path, monkeypatch, override):
    for name in ('ROBOTUSE_SAM_PYTHON', 'SAM_RUNTIME_PYTHON', 'SAM2_ENVIRONMENT', 'ROBOTUSE_RUNTIME_ROOT'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(override, sys.executable)
    monkeypatch.setattr(sam2_adapter, 'ENVIRONMENT_PYTHON', tmp_path / 'unavailable')
    sam2_adapter.validate_python(sam2_adapter.DEFAULT_PYTHON)
    env = dict(os.environ, PYTHONHOME='/unavailable/isaac-python', PYTHONPATH='/unavailable/isaac-path')
    command = [str(sam2_adapter.DEFAULT_PYTHON), '-c',
               'import json, os, sys; print(json.dumps([sys.executable, os.environ.get("PYTHONHOME"), os.environ["PYTHONPATH"]]))']
    result = subprocess.run(command, env=env, text=True, capture_output=True, check=True, timeout=20)
    executable, pythonhome, pythonpath = json.loads(result.stdout)
    assert executable == sys.executable
    assert pythonhome is None
    assert pythonpath == str(ROOT)


@pytest.mark.parametrize('shell', ['bash', 'zsh'])
@pytest.mark.parametrize('gpu', [None, '0', '3'])
def test_env_file_preserves_explicit_public_interpreters_and_gpu(shell, tmp_path, gpu):
    if shutil.which(shell) is None:
        pytest.skip(f'{shell} is not installed')
    env = dict(os.environ, ROBOTUSE_RUNTIME_ROOT=str(tmp_path), ROBOTUSE_CGN_PYTHON='/cgn/python',
               ROBOTUSE_CUROBO_PYTHON='/curobo/python')
    env.pop('CUDA_VISIBLE_DEVICES', None)
    for name in ('ROBOTUSE_GPU', 'ROBOTUSE_CGN_GPU'):
        env.pop(name, None)
        if gpu is not None:
            env[name] = gpu
    command = 'set -e; unset CUDA_VISIBLE_DEVICES; source "$1"; printf "%s\\n" "$ROBOTUSE_CGN_PYTHON" "$ROBOTUSE_CUROBO_PYTHON" "$ROBOTUSE_GPU" "$ROBOTUSE_CGN_GPU"'
    result = subprocess.run([shell, '-f', '-c', command, 'env-test', str(ROOT / 'scripts/lib/env.sh')],
                            cwd=tmp_path, env=env, text=True,
                            capture_output=True, check=True, timeout=20)
    assert result.stdout.splitlines() == ['/cgn/python', '/curobo/python', gpu or '0', gpu or '0']
