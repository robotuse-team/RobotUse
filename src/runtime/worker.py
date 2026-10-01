"""Persistent, process-isolated model calls; avoid repeated imports/weight reads."""
import atexit
import threading
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time
from types import SimpleNamespace

_POOL = {}


@contextmanager
def diagnostic_phase(phase, *, request_index=None):
    """Timestamp fixed phase labels on stderr, without request or model data."""
    started = time.monotonic_ns()

    def emit(status):
        now = time.monotonic_ns()
        record = {'schema': 'model-worker.phase.v1', 'phase': phase,
                  'status': status, 'wall_time_unix_s': time.time(),
                  'monotonic_ns': now}
        if request_index is not None:
            record['request_index'] = request_index
        if status != 'started':
            record['elapsed_s'] = (now - started) / 1e9
        try:
            print(json.dumps(record), file=sys.stderr, flush=True)
        except Exception:
            # Diagnostics must not change the model call or mask its exception.
            pass

    emit('started')
    try:
        yield
    except BaseException:
        emit('failed')
        raise
    else:
        emit('completed')


class ModelWorker:
    def __init__(self, python, module, log):
        log = Path(os.environ.get('GUIROBOT_WORKER_LOG_DIR', Path(log).parent)) / Path(log).name
        log.parent.mkdir(parents=True, exist_ok=True)
        self.log = log.open('a')
        env = {k:v for k,v in os.environ.items() if k not in {'OPENROUTER_API_KEY', 'GOOGLE_AI_STUDIO_KEY', 'GEMINI_API_KEY', 'GOOGLE_API_KEY'}}
        self.process = subprocess.Popen([str(python), '-u', str(Path(__file__).resolve()), str(module)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, env=env)
        atexit.register(self.close)
        self.responses = 0
        # One request at a time per worker: a background prewarm and the first
        # real request must not interleave on the line protocol.
        self.lock = threading.Lock()

    def call(self, arguments, timeout, startup_timeout=None):
        with self.lock:
            if self.responses == 0 and startup_timeout is not None:
                timeout = max(timeout, startup_timeout)
            self.process.stdin.write(json.dumps(arguments)+'\n'); self.process.stdin.flush()
            ready, _, _ = select.select([self.process.stdout], [], [], timeout)
            if not ready:
                self.close(); raise TimeoutError('model worker timed out')
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError('model worker exited; inspect worker log')
            result = json.loads(line)
            self.responses += 1
            if not result['ok']:
                raise RuntimeError(result['error'])

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait()
        self.log.close()


_POOL_LOCK = threading.Lock()


def close_workers():
    """Release owned subprocesses before an Isaac standalone fast exit."""
    with _POOL_LOCK:
        workers = list(_POOL.values())
        _POOL.clear()
    for worker in workers:
        worker.close()


def call_worker(python, module, arguments, log, timeout, *, startup_timeout=None):
    key = (str(python), str(module))
    with _POOL_LOCK:
        worker = _POOL.get(key)
        if worker is None or worker.process.poll() is not None:
            worker = _POOL[key] = ModelWorker(python, module, log)
    worker.call(arguments, timeout, startup_timeout=startup_timeout)


def serve(path):
    import contextlib
    import traceback
    with diagnostic_phase('worker.module_import'):
        spec = importlib.util.spec_from_file_location('persistent_model', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    for request_index, line in enumerate(sys.stdin, 1):
        try:
            with diagnostic_phase('worker.request', request_index=request_index), contextlib.redirect_stdout(sys.stderr):
                module._worker(SimpleNamespace(**json.loads(line)))
            result = {'ok':True}
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            result = {'ok':False,'error':f'{type(exc).__name__}: {exc}'}
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    # Model interpreters start from arbitrary working directories.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    serve(sys.argv[1])
