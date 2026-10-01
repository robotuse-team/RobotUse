#!/usr/bin/env bash
# Install the isolated CGN service without writing to either upstream snapshot.
set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cgn_runtime="${ROBOTUSE_RUNTIME_ROOT:-$repository_root/runtime}"
cgn_source="$repository_root/src/tools/grasp/third_party/contact_graspnet"
cgn_revision=da3dcfb2f53e43b186083ee4a9d1e232f73efc98
cgn_requirements="$repository_root/requirements/cgn.txt"
cgn_python="${CGN_BASE_PYTHON:-3.11.9}"
cgn_smoke=false
case "${1:-}" in
  '') ;;
  --smoke-test) cgn_smoke=true ;;
  -h|--help)
    echo 'Usage: scripts/setup/cgn.sh [--smoke-test]'
    echo 'ROBOTUSE_RUNTIME_ROOT: writable runtime destination; CGN_BASE_PYTHON: Python 3.11.9 executable.'
    echo 'Optional GPU smoke uses CGN_SMOKE_GPU (default: ROBOTUSE_CGN_GPU or 0).'
    exit 0 ;;
  *) echo "Unknown option: $1" >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo 'Too many arguments.' >&2; exit 2; }
command -v uv >/dev/null || { echo 'Install uv before running this script.' >&2; exit 1; }
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || {
  echo 'This dependency lock is verified for Linux x86_64.' >&2; exit 1
}
[[ "$(git -C "$cgn_source" rev-parse HEAD)" == "$cgn_revision" ]] || {
  echo 'Initialize the pinned CGN submodule with git submodule update --init --recursive.' >&2; exit 1
}
[[ -z "$(git -C "$cgn_source" status --porcelain --untracked-files=all)" ]] || {
  echo 'The CGN upstream checkout must be clean; setup will not modify or repair it.' >&2; exit 1
}

# Discard simulator/conda import and linker overlays for this separate service.
unset PYTHONPATH PYTHONHOME PYTHONEXE LD_LIBRARY_PATH LD_PRELOAD VIRTUAL_ENV
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYOPENGL_PLATFORM=egl
cgn_runtime="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().absolute())' "$cgn_runtime")"
cgn_environment="$cgn_runtime/tool-envs/cgn"
python3 - "$cgn_runtime" "$cgn_environment" "$cgn_source" "$repository_root" <<'PY'
import hashlib
from pathlib import Path
import sys
runtime = Path(sys.argv[1])
for value in (*sys.argv[1:3], runtime / 'cache', runtime / 'python'):
    path = Path(value).expanduser().absolute()
    if {'third_party', 'vendor'} & set(path.parts) or path.resolve().is_relative_to(Path(sys.argv[4]) / 'src'):
        raise SystemExit('Setup output must be outside src, third_party, and vendor.')
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise SystemExit(f'Refusing setup through a symlink: {parent}')
    if path.exists() and not path.is_dir():
        raise SystemExit(f'Setup destination is not a directory: {path}')
source = Path(sys.argv[3]) / 'checkpoints/contact_graspnet'
for name, expected in {
    'config.yaml': '7ae6cc15c726e8ec2c46b0c799d58ca95caeb9ce9f261aaaf3e467712df76ac5',
    'checkpoints/model.pt': '39fc3439d5814043ba64e0715c127c2e8aca6ea376c22614e11af1bc89317762',
}.items():
    path = source / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f'Pinned CGN asset missing or changed: {path}; initialize git submodules')
PY
mkdir -p -- "$cgn_runtime/cache"
cgn_runtime="$(cd -- "$cgn_runtime" && pwd)"
cgn_environment="$cgn_runtime/tool-envs/cgn"
export UV_CACHE_DIR="$cgn_runtime/cache/uv"
export UV_PYTHON_INSTALL_DIR="$cgn_runtime/python"
export XDG_CACHE_HOME="$cgn_runtime/cache"
export CUDA_CACHE_PATH="$cgn_runtime/cache/cuda"

if [[ -e "$cgn_environment" ]]; then
  [[ -f "$cgn_environment/.robotuse-cgn" && -f "$cgn_environment/pyvenv.cfg" &&
     "$(cat "$cgn_environment/.robotuse-cgn")" == "$cgn_revision" ]] || {
    echo "Refusing to change an existing environment not created by this setup: $cgn_environment" >&2; exit 1
  }
else
  uv venv --python "$cgn_python" "$cgn_environment"
  printf '%s\n' "$cgn_revision" > "$cgn_environment/.robotuse-cgn"
fi
"$cgn_environment/bin/python" - "$cgn_environment" <<'PY'
from pathlib import Path
import sys
if sys.version_info[:3] != (3, 11, 9):
    raise SystemExit('CGN lock requires Python 3.11.9')
if sys.prefix == sys.base_prefix or Path(sys.prefix).resolve() != Path(sys.argv[1]).resolve():
    raise SystemExit('Selected Python does not belong to the requested virtual environment.')
PY

# The original model uses pure PyTorch PointNet operations: no nvcc build or
# installation inside third_party is needed. CUDA libraries come from the lock.
uv pip sync --python "$cgn_environment/bin/python" --index-strategy unsafe-best-match \
  --require-hashes "$cgn_requirements"
uv pip check --python "$cgn_environment/bin/python"

PYTHONPATH="$repository_root" "$cgn_environment/bin/python" - \
  "$cgn_environment" "$cgn_requirements" "$cgn_smoke" "${CGN_SMOKE_GPU:-${ROBOTUSE_CGN_GPU:-0}}" <<'PY'
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time

from src.tools.grasp.service_worker import CGN_SOURCE, SERVICE_SOURCE, verify_service_source

environment, requirements = map(Path, sys.argv[1:3])
manifest = verify_service_source()
sys.path[:0] = [str(SERVICE_SOURCE), str(CGN_SOURCE), str(CGN_SOURCE / 'Pointnet_Pointnet2_pytorch')]
from capx.serving import launch_contact_graspnet_server
from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
import torch

result = {
    'python': platform.python_version(), 'torch': torch.__version__,
    'torch_cuda': torch.version.cuda, 'cgn_revision': manifest['original_cgn_revision'],
    'requirements_sha256': hashlib.sha256(requirements.read_bytes()).hexdigest(),
    'installed': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
}
if sys.argv[3] == 'true':
    from src.tools.grasp.service import api_ready, inference_probe
    evidence = environment.parent.parent / 'setup-checks' / (
        'cgn-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    evidence.mkdir(parents=True, exist_ok=False)
    gpu = sys.argv[4]
    child_env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    child_env.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED='1')
    gpu_check = subprocess.check_output([sys.executable, '-c',
        'import json, torch; assert torch.cuda.is_available(), "CGN smoke requires CUDA"; '
        'print(json.dumps({"name": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0)}))'],
        env=child_env, text=True)
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        port = reservation.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    command = [sys.executable, str(SERVICE_SOURCE.parent.parent / 'service_worker.py'),
               '--device', 'cuda:0', '--host', '127.0.0.1', '--port', str(port)]
    with (evidence / 'server.log').open('x') as stream:
        process = subprocess.Popen(command, env=child_env, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 180
        while True:
            if process.poll() is not None:
                raise RuntimeError(f'CGN startup exited; inspect {evidence / "server.log"}')
            try:
                api_ready(url)
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'CGN startup timed out; inspect {evidence / "server.log"}')
                time.sleep(1)
        smoke = dict(inference_probe(url), physical_gpu=gpu, gpu=json.loads(gpu_check),
                     command=command, python=platform.python_version(), torch=torch.__version__)
        (evidence / 'inference.json').write_text(json.dumps(smoke, indent=2) + '\n')
        result['gpu_smoke'] = str(evidence / 'inference.json')
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)
(environment / 'robotuse-setup.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'installed'}, indent=2))
PY
[[ -z "$(git -C "$cgn_source" status --porcelain --untracked-files=all)" ]] || {
  echo 'CGN upstream checkout changed during setup; inspect before proceeding.' >&2; exit 1
}
printf '\nCGN service environment installed.\nexport ROBOTUSE_CGN_PYTHON=%q\n' "$cgn_environment/bin/python"
