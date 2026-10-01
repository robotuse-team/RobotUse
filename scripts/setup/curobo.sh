#!/usr/bin/env bash
# Provision an optional, isolated cuRobo worker. Never install/build in third_party.
set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
curobo_runtime="${ROBOTUSE_RUNTIME_ROOT:-$repository_root/runtime}"
curobo_environment="${CUROBO_ENVIRONMENT:-$curobo_runtime/tool-envs/curobo}"
curobo_source="$repository_root/src/tools/curobo/third_party/curobo"
curobo_revision=4ea77366ca48ee453e7df139e39fa6532af49f3b
curobo_requirements="$repository_root/requirements/curobo.txt"
curobo_python="${CUROBO_BASE_PYTHON:-3.11.13}"

# Isaac's parent process may carry its own Python and CUDA search paths. Keep
# those out of dependency installation and import checks in this child shell.
unset PYTHONPATH PYTHONHOME PYTHONEXE LD_LIBRARY_PATH LD_PRELOAD VIRTUAL_ENV

command -v uv >/dev/null || { echo 'Install uv before running this script.' >&2; exit 1; }
[[ "$(git -C "$curobo_source" rev-parse HEAD)" == "$curobo_revision" ]] || {
  echo 'Initialize the pinned cuRobo submodule with git submodule update --init --recursive.' >&2
  exit 1
}
[[ -z "$(git -C "$curobo_source" status --porcelain --untracked-files=all)" ]] || {
  echo 'The cuRobo upstream checkout must be clean; this script will not modify or repair it.' >&2
  exit 1
}

# Refuse to follow runtime/environment symlinks into another installation.
python3 - "$repository_root" "$curobo_runtime" "$curobo_environment" "$curobo_runtime/cache" <<'PY'
from pathlib import Path
import sys
repository = Path(sys.argv[1]).resolve()
for value in sys.argv[2:]:
    path = Path(value).expanduser().absolute()
    if {'third_party', 'vendor'} & set(path.parts) or any(
        path.resolve().is_relative_to(repository / name) for name in ('src', 'vendor')):
        raise SystemExit(f'Environment/cache must not be created inside source directories: {path}')
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise SystemExit(f'Refusing setup through a symlink: {parent}')
    if path.exists() and not path.is_dir():
        raise SystemExit(f'Setup destination is not a directory: {path}')
PY
mkdir -p -- "$curobo_runtime/cache"
curobo_runtime="$(cd -- "$curobo_runtime" && pwd)"
export UV_CACHE_DIR="$curobo_runtime/cache/uv"
export XDG_CACHE_HOME="$curobo_runtime/cache"
export TORCH_EXTENSIONS_DIR="$curobo_runtime/cache/torch-extensions"
export WARP_CACHE_PATH="$curobo_runtime/cache/warp"
export CUDA_CACHE_PATH="$curobo_runtime/cache/cuda"
export PYTHONDONTWRITEBYTECODE=1

if [[ -e "$curobo_environment" ]]; then
  [[ -f "$curobo_environment/.robotuse-curobo" && \
     -f "$curobo_environment/pyvenv.cfg" && \
     -x "$curobo_environment/bin/python" && \
     "$(cat -- "$curobo_environment/.robotuse-curobo")" == "$curobo_revision" ]] || {
    echo "Refusing to change an existing environment not created by this setup: $curobo_environment" >&2
    exit 1
  }
else
  uv venv --python "$curobo_python" "$curobo_environment"
  printf '%s\n' "$curobo_revision" > "$curobo_environment/.robotuse-curobo"
fi
curobo_environment="$(cd -- "$curobo_environment" && pwd)"
"$curobo_environment/bin/python" - "$curobo_environment" <<'PY'
from pathlib import Path
import sys
if sys.version_info[:3] != (3, 11, 13):
    raise SystemExit('cuRobo lock requires Python 3.11.13')
if sys.prefix == sys.base_prefix or Path(sys.prefix).resolve() != Path(sys.argv[1]).resolve():
    raise SystemExit('Selected Python does not belong to the requested virtual environment.')
PY

# Torch's CUDA wheel pins the 12.8 libraries; no inherited overlay is needed.
# The source package is imported directly by the isolated worker.
uv pip sync --python "$curobo_environment/bin/python" \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match --require-hashes "$curobo_requirements"

PYTHONPATH="$repository_root:$curobo_source" \
  "$curobo_environment/bin/python" - "$curobo_environment" "$curobo_requirements" <<'PY'
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
from cuda import pathfinder
from src.tools.curobo.adapter import _runtime_source
import torch
import warp

environment, requirements = map(Path, sys.argv[1:])
result = _runtime_source()
headers = pathfinder.find_nvidia_header_directory('nvrtc')
if headers is None or not Path(headers).is_dir():
    raise RuntimeError('CUDA NVRTC headers were not provisioned')
result.update(python=platform.python_version(), torch=torch.__version__, torch_cuda=torch.version.cuda,
              warp=warp.__version__, cuda_headers=headers,
              requirements_sha256=hashlib.sha256(requirements.read_bytes()).hexdigest(),
              installed={d.metadata['Name']: d.version for d in importlib.metadata.distributions()})
(environment / 'robotuse-setup.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'installed'}, indent=2))
PY
[[ -z "$(git -C "$curobo_source" status --porcelain --untracked-files=all)" ]] || {
  echo 'Upstream checkout changed during setup; inspect before proceeding.' >&2
  exit 1
}

printf '\nOptional cuRobo worker installed. Set these only when selecting cuRobo:\n'
printf 'export ROBOTUSE_CUROBO_PYTHON=%q\n' "$curobo_environment/bin/python"
printf 'export XDG_CACHE_HOME=%q\n' "$XDG_CACHE_HOME"
printf 'export TORCH_EXTENSIONS_DIR=%q\n' "$TORCH_EXTENSIONS_DIR"
printf 'export WARP_CACHE_PATH=%q\n' "$WARP_CACHE_PATH"
printf 'export CUDA_CACHE_PATH=%q\n' "$CUDA_CACHE_PATH"
printf 'Robot model and calibration still require explicit --curobo-robot-file and --curobo-calibration-file.\n'
