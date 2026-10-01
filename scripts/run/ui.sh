#!/usr/bin/env bash
set -euo pipefail
repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$repository_root/scripts/lib/env.sh"
export PYTHONDONTWRITEBYTECODE=1
export GRADIO_ANALYTICS_ENABLED=False
cd "$repository_root"
exec "$ROBOTUSE_UI_PYTHON" -m src.ui.app "$@"
