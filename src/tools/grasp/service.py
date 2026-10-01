"""Start a local CGN service if absent; require real inference before RobotUse starts."""
from __future__ import annotations

import datetime
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit
from urllib.request import urlopen

import numpy as np
from src.tools.grasp.cgn_client import ContactGraspNetClient, CAMERA_OPTICAL

DEFAULT_SOURCE = Path(__file__).resolve().parent
DEFAULT_LIBRARY_PATH = None


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def api_ready(url):
    with urlopen(url.rstrip('/') + '/openapi.json', timeout=5) as response:
        paths = json.load(response).get('paths', {})
    if '/plan' not in paths or '/plan_point_clouds' not in paths:
        raise RuntimeError('Endpoint is not the required Contact-GraspNet service')


def inference_probe(url, timeout_s=120):
    # Deterministic optical-frame table + curved object. A valid zero-candidate
    # response is healthy: readiness must not require graspability of this fixture.
    u, v = np.meshgrid(np.linspace(-.3, .3, 64), np.linspace(-.3, .3, 64))
    table = np.column_stack((u.ravel(), v.ravel(), np.full(u.size, .8)))
    u, v = np.meshgrid(np.linspace(-.035, .035, 40), np.linspace(-.035, .035, 40))
    obj = np.column_stack((u.ravel(), v.ravel(), (.73 + 2*(u*u+v*v)).ravel()))
    raw = ContactGraspNetClient(url, timeout_s=timeout_s).plan_point_clouds(
        np.concatenate((table, obj)), obj, input_frame=CAMERA_OPTICAL,
        forward_passes=1, max_retries=1)
    return {'endpoint': '/plan_point_clouds', 'raw_candidates': len(raw.poses),
            'inference_completed': True, 'at': now()}


def ensure_cgn(url, *, evidence_dir, timeout_s=120, startup_timeout_s=180):
    """Never replace an existing server. Serialize starts/probes across workers."""
    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    parsed = urlsplit(url)
    port = parsed.port or 80
    local = parsed.hostname in ('localhost', '127.0.0.1', '::1')
    # Shared lock prevents concurrent RobotUse launches from both starting port 8115.
    lock_dir = Path(os.environ.get('ROBOTUSE_CGN_LOCK_DIR', '/tmp'))
    with (lock_dir / f'robotuse-cgn-{port}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = {'url': url, 'physical_gpu': int(os.environ.get('ROBOTUSE_CGN_GPU', '0')), 'checked_at': now(), 'started': False}
        try:
            try:
                with socket.create_connection((parsed.hostname, port), timeout=2):
                    pass
                listening = True
            except OSError:
                listening = False
            if not listening:
                if not local:
                    raise RuntimeError('Remote CGN is unavailable; cannot start a remote service locally')
                source = DEFAULT_SOURCE
                python = Path(os.environ.get('ROBOTUSE_CGN_PYTHON', sys.executable))
                if not python.is_file() or not (source / 'service_worker.py').is_file():
                    raise FileNotFoundError('Set ROBOTUSE_CGN_PYTHON to the installed CGN runtime interpreter')
                env = {k: v for k, v in os.environ.items() if k not in
                       ('PYTHONPATH', 'PYTHONHOME', 'PYTHONEXE', 'LD_LIBRARY_PATH', 'VIRTUAL_ENV')}
                env.update(CUDA_VISIBLE_DEVICES=os.environ.get('ROBOTUSE_CGN_GPU', '0'), PYTHONUNBUFFERED='1', PYOPENGL_PLATFORM='egl',
                           PYTHONDONTWRITEBYTECODE='1')
                dependency_path = os.environ.get('ROBOTUSE_CGN_PYTHONPATH')
                if dependency_path:
                    env['PYTHONPATH'] = dependency_path
                # Same installed GLU/OpenGL runtime as robolab-local/env.sh.
                # Do not inherit simulator/plugin LD paths into this separate service.
                library_path = os.environ.get('ROBOTUSE_CGN_LIBRARY_PATH')
                if library_path is None and DEFAULT_LIBRARY_PATH is not None and DEFAULT_LIBRARY_PATH.is_dir():
                    library_path = str(DEFAULT_LIBRARY_PATH)
                if library_path:
                    env['LD_LIBRARY_PATH'] = library_path
                command = [str(python), str(source / 'service_worker.py'),
                           '--device', 'cuda:0', '--host', parsed.hostname, '--port', str(port)]
                with (evidence_dir / 'server.log').open('a') as stream:
                    process = subprocess.Popen(command, cwd=source, env=env,
                        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                        start_new_session=True)
                result.update(started=True, pid=process.pid, command=command, source=str(source),
                              runtime_library_path=env.get('LD_LIBRARY_PATH'))
                (evidence_dir / 'server-process.json').write_text(json.dumps(result, indent=2)+'\n')
                deadline = time.monotonic() + startup_timeout_s
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f'CGN server exited during startup ({process.returncode}); see server.log')
                    try:
                        api_ready(url)
                        break
                    except Exception:
                        if time.monotonic() >= deadline:
                            raise TimeoutError('CGN did not become ready before startup deadline; see server.log')
                        time.sleep(1)
            else:
                api_ready(url)
            result.update(inference_probe(url, timeout_s), ready=True)
        except Exception as exc:
            from src.core.errors import structured_error
            result.update(ready=False, error_details=structured_error(exc, phase='cgn_preflight'))
            (evidence_dir / 'preflight.json').write_text(json.dumps(result, indent=2)+'\n')
            raise RuntimeError('CGN preflight failed before simulator/model startup: '+str(exc)) from exc
        (evidence_dir / 'preflight.json').write_text(json.dumps(result, indent=2)+'\n')
        return result
