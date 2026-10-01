"""Keep cuRobo's CUDA/Warp runtime outside the native Isaac process."""
from __future__ import annotations

import atexit
import json
import math
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import threading
import time
from weakref import WeakSet

from src.runtime.paths import REPOSITORY_ROOT

_WORKERS = WeakSet()


def worker_environment():
    excluded = {'PYTHONPATH', 'PYTHONHOME', 'PYTHONEXE', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'VIRTUAL_ENV',
                'OPENAI_API_KEY', 'OPENROUTER_API_KEY', 'GOOGLE_AI_STUDIO_KEY', 'GEMINI_API_KEY', 'GOOGLE_API_KEY'}
    env = {key: value for key, value in os.environ.items() if key not in excluded}
    source = Path(__file__).parent / 'third_party/curobo'
    paths = [str(REPOSITORY_ROOT), str(source)]
    if env.get('ROBOTUSE_CUROBO_PYTHONPATH'):
        paths.append(env['ROBOTUSE_CUROBO_PYTHONPATH'])
    env.update(PYTHONPATH=os.pathsep.join(paths), PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    if env.get('ROBOTUSE_CUROBO_LIBRARY_PATH'):
        env['LD_LIBRARY_PATH'] = env['ROBOTUSE_CUROBO_LIBRARY_PATH']
    return env


class CuroboWorker:
    """One serialized request at a time; original planner caches live in the child."""

    def __init__(self):
        self.timeout = float(os.environ.get('ROBOTUSE_CUROBO_TIMEOUT_S', '600'))
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError('ROBOTUSE_CUROBO_TIMEOUT_S must be positive and finite')
        python = Path(os.environ.get('ROBOTUSE_CUROBO_PYTHON', sys.executable)).expanduser()
        if not python.is_file():
            raise FileNotFoundError('Set ROBOTUSE_CUROBO_PYTHON to the installed cuRobo runtime interpreter')
        directory = os.environ.get('ROBOTUSE_CUROBO_LOG_DIR')
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)
        self.log_path = Path(tempfile.mkdtemp(prefix='robotuse-curobo-', dir=directory)) / 'worker.log'
        self._log = self.log_path.open('xb')
        self._lock = threading.Lock()
        self._buffer = b''
        try:
            self.process = subprocess.Popen(
                [str(python), '-u', str(Path(__file__).with_name('worker.py'))],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                env=worker_environment())
        except BaseException:
            self._log.close()
            raise
        _WORKERS.add(self)
        atexit.register(self.close)

    def call(self, request):
        payload = json.dumps(request, allow_nan=False).encode() + b'\n'
        with self._lock:
            try:
                self.process.stdin.write(payload)
                self.process.stdin.flush()
                deadline = time.monotonic() + self.timeout
                while b'\n' not in self._buffer:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                        raise TimeoutError('cuRobo worker timed out')
                    chunk = os.read(self.process.stdout.fileno(), 65536)
                    if not chunk:
                        raise RuntimeError('cuRobo worker exited before responding')
                    self._buffer += chunk
                line, self._buffer = self._buffer.split(b'\n', 1)
                response = json.loads(line)
                if response.get('ok') is not True:
                    raise RuntimeError(response.get('error', 'cuRobo worker failed'))
                return response['result']
            except BaseException as exc:
                self.close()
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                raise RuntimeError(f'{exc}; cuRobo worker log: {self.log_path}') from exc

    def close(self):
        try:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        finally:
            try:
                for stream in (self.process.stdin, self.process.stdout, self._log):
                    try:
                        stream.close()
                    except OSError:
                        # stdin.close() can flush a pending request into a
                        # crashed child. Close the remaining handles anyway.
                        pass
            finally:
                _WORKERS.discard(self)
                atexit.unregister(self.close)


def close_workers():
    """Native standalone exits skip Python finalizers, so close owned children first."""
    for worker in list(_WORKERS):
        worker.close()
