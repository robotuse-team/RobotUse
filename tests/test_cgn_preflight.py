from pathlib import Path
import sys
from types import SimpleNamespace
import json

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/run'))
from src.tools.grasp import service as preflight
from src.runtime import cli as run_episode
from runtime_test_support import flags


class Socket:
    def __enter__(self): return self
    def __exit__(self, *args): pass


def test_open_port_is_insufficient_and_does_not_replace_server(tmp_path, monkeypatch):
    monkeypatch.setenv('ROBOTUSE_CGN_LOCK_DIR', str(tmp_path))
    monkeypatch.setenv('ROBOTUSE_CGN_GPU', '3')
    monkeypatch.setattr(preflight.socket, 'create_connection', lambda *a, **k: Socket())
    monkeypatch.setattr(preflight, 'api_ready', lambda *a: None)
    monkeypatch.setattr(preflight.subprocess, 'Popen', lambda *a, **k: pytest.fail('must not replace service'))
    def bad(*a):
        from src.tools.grasp.cgn_client import CGNServiceError
        raise CGNServiceError('server inference failed')
    monkeypatch.setattr(preflight, 'inference_probe', bad)
    with pytest.raises(RuntimeError, match='preflight failed'):
        preflight.ensure_cgn('http://127.0.0.1:8115', evidence_dir=tmp_path)
    record = json.loads((tmp_path / 'preflight.json').read_text())
    assert record['ready'] is False and record['error_details']['phase'] == 'cgn_preflight'
    assert record['requested_gpu'] == '3'
    assert record['physical_gpu'] is None
    assert record['gpu_source'] == 'unknown_existing_service'


@pytest.mark.parametrize('url', ['http://127.0.0.1:8115', 'http://cgn.example.test:8115'])
def test_reused_service_does_not_claim_requested_gpu(tmp_path, monkeypatch, url):
    monkeypatch.setenv('ROBOTUSE_CGN_LOCK_DIR', str(tmp_path))
    monkeypatch.setenv('ROBOTUSE_CGN_GPU', '3')
    monkeypatch.setattr(preflight.socket, 'create_connection', lambda *a, **k: Socket())
    monkeypatch.setattr(preflight.subprocess, 'Popen', lambda *a, **k: pytest.fail('must not replace service'))
    checks = []
    monkeypatch.setattr(preflight, 'api_ready', lambda target: checks.append(('api', target)))
    def probe(target, timeout):
        checks.append(('inference', target, timeout))
        return {'inference_completed': True}
    monkeypatch.setattr(preflight, 'inference_probe', probe)

    result = preflight.ensure_cgn(url, evidence_dir=tmp_path, timeout_s=17)

    assert result['ready'] is True and result['started'] is False
    assert result['requested_gpu'] == '3'
    assert result['physical_gpu'] is None
    assert result['gpu_source'] == 'unknown_existing_service'
    assert checks == [('api', url), ('inference', url, 17)]
    assert not (tmp_path / 'server-process.json').exists()
    assert json.loads((tmp_path / 'preflight.json').read_text()) == result


def test_failure_happens_before_episode_runner(tmp_path, monkeypatch):
    def fail(*a, **k): raise RuntimeError('CGN unavailable')
    monkeypatch.setattr(preflight, 'ensure_cgn', fail)
    monkeypatch.setattr(run_episode, 'execute', lambda *a: pytest.fail('must not enter simulator runner'))
    with pytest.raises(RuntimeError, match='CGN unavailable'):
        run_episode.main(flags(tmp_path))
    assert not (tmp_path / 'live').exists()


def test_dry_run_never_starts_or_probes_service(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, 'ensure_cgn', lambda *a, **k: pytest.fail('offline dry-run'))
    assert run_episode.main([*flags(tmp_path), '--dry-run']) == 0


def test_sibling_episode_preflights_preserve_previous_raw_evidence(tmp_path, monkeypatch):
    directories, records = [], []

    def offline_preflight(url, *, evidence_dir, **kwargs):
        directories.append(evidence_dir)
        (evidence_dir / 'preflight.json').write_text(json.dumps({'attempt': len(directories)}))
        return {'ready': True}

    def do_not_start_episode(argv, record, dry, configuration):
        records.append(record)
        return 0

    monkeypatch.setattr(preflight, 'ensure_cgn', offline_preflight)
    monkeypatch.setattr(run_episode, 'execute', do_not_start_episode)
    assert run_episode.main([*flags(tmp_path), '--output-dir', str(tmp_path / 'first')]) == 0
    first = (directories[0] / 'preflight.json').read_bytes()
    assert run_episode.main([*flags(tmp_path), '--output-dir', str(tmp_path / 'second')]) == 0
    assert directories[0] != directories[1]
    assert all(path.parent == tmp_path for path in directories)
    assert (directories[0] / 'preflight.json').read_bytes() == first
    assert json.loads((directories[1] / 'preflight.json').read_text()) == {'attempt': 2}
    assert [row['cgn_preflight']['evidence_dir'] for row in records] == [str(path) for path in directories]


def test_zero_grasps_is_valid_service_inference(monkeypatch):
    class Client:
        def __init__(self, *a, **k): pass
        def plan_point_clouds(self, full, segment, **kwargs):
            assert full.shape[1] == segment.shape[1] == 3
            assert len(full) > len(segment) > 0
            assert kwargs['max_retries'] == 1
            return SimpleNamespace(poses=[])
    monkeypatch.setattr(preflight, 'ContactGraspNetClient', Client)
    assert preflight.inference_probe('http://localhost:8115')['inference_completed']


@pytest.mark.parametrize('override', [None, '/installed/cgn/runtime'])
def test_start_child_uses_dedicated_gpu_and_gl_runtime_without_simulator_paths(tmp_path, monkeypatch, override):
    source = tmp_path / 'cgn'
    script = source / 'capx/serving/launch_contact_graspnet_server.py'
    script.parent.mkdir(parents=True)
    script.write_text('')
    python = source / '.venv-robolab/bin/python'
    python.parent.mkdir(parents=True)
    python.write_text('')
    library = tmp_path / 'sysroot'
    library.mkdir()
    monkeypatch.setenv('ROBOTUSE_CGN_PYTHON', str(python))
    monkeypatch.setenv('ROBOTUSE_CGN_LOCK_DIR', str(tmp_path))
    monkeypatch.setenv('LD_LIBRARY_PATH', '/bad/simulator/plugins')
    monkeypatch.setenv('PYTHONPATH', '/bad/simulator/python')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setenv('ROBOTUSE_CGN_GPU', '1')
    monkeypatch.setattr(preflight, 'DEFAULT_LIBRARY_PATH', library)
    if override is None:
        monkeypatch.delenv('ROBOTUSE_CGN_LIBRARY_PATH', raising=False)
    else:
        monkeypatch.setenv('ROBOTUSE_CGN_LIBRARY_PATH', override)
    def offline(*args, **kwargs):
        raise ConnectionRefusedError()
    monkeypatch.setattr(preflight.socket, 'create_connection', offline)
    captured = {}
    def spawn(command, **kwargs):
        captured.update(command=command, **kwargs)
        return SimpleNamespace(pid=1234, poll=lambda: None)
    monkeypatch.setattr(preflight.subprocess, 'Popen', spawn)
    monkeypatch.setattr(preflight, 'api_ready', lambda *args: None)
    monkeypatch.setattr(preflight, 'inference_probe', lambda *args: {'inference_completed': True})
    result = preflight.ensure_cgn('http://127.0.0.1:18115', evidence_dir=tmp_path / 'evidence')
    assert result['ready'] and result['started']
    assert captured['env']['CUDA_VISIBLE_DEVICES'] == '1'
    assert captured['command'][1] == str(preflight.DEFAULT_SOURCE / 'service_worker.py')
    assert captured['cwd'] == preflight.DEFAULT_SOURCE
    assert captured['env']['LD_LIBRARY_PATH'] == (override or str(library))
    assert 'PYTHONPATH' not in captured['env']
    assert captured['start_new_session'] is True
    assert result['runtime_library_path'] == captured['env']['LD_LIBRARY_PATH']
