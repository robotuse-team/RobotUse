#!/usr/bin/env bash
set -euo pipefail
repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$repository_root/scripts/lib/env.sh"
runtime_root="${ROBOTUSE_RUNTIME_ROOT:-$repository_root/runtime}"
ui_python="${ROBOTUSE_UI_PYTHON:-$runtime_root/tool-envs/ui/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
export GRADIO_ANALYTICS_ENABLED=False
cd "$repository_root"
exec "$ui_python" -m src.ui.app "$@"
