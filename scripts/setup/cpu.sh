#!/usr/bin/env bash
set -euo pipefail
unset PYTHONPATH PYTHONHOME PYTHONEXE LD_LIBRARY_PATH LD_PRELOAD VIRTUAL_ENV
repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
runtime_root="${ROBOTUSE_RUNTIME_ROOT:-$repository_root/runtime}"
cpu_environment="${CPU_ENVIRONMENT:-$runtime_root/tool-envs/cpu}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
source "$repository_root/scripts/lib/venv.sh"
robotuse_create_venv "$cpu_environment" 3.11.9
uv pip sync --python "$cpu_environment/bin/python" --require-hashes \
  "$repository_root/requirements/cpu.txt"
printf 'CPU environment ready: %s\n' "$cpu_environment/bin/python"
