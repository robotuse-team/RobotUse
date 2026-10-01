#!/usr/bin/env bash
set -euo pipefail
unset PYTHONPATH PYTHONHOME PYTHONEXE LD_LIBRARY_PATH LD_PRELOAD VIRTUAL_ENV
repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
runtime_root="${ROBOTUSE_RUNTIME_ROOT:-$repository_root/runtime}"
ui_environment="${UI_ENVIRONMENT:-$runtime_root/tool-envs/ui}"
export PYTHONDONTWRITEBYTECODE=1
source "$repository_root/scripts/lib/venv.sh"
robotuse_create_venv "$ui_environment" 3.11.9
uv pip sync --python "$ui_environment/bin/python" --require-hashes "$repository_root/requirements/ui.txt"
printf 'UI environment ready: %s\n' "$ui_environment/bin/python"
