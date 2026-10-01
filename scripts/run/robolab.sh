#!/usr/bin/env bash
# Example (configuration only, no simulator/model startup):
# ROBOLAB_PYTHON=/path/to/python scripts/run/robolab.sh \
#   --task BananaInBowlTask --difficulty simple --output-dir runs/example --dry-run
set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$repository_root/scripts/lib/env.sh"
runtime_python="${ROBOLAB_PYTHON:-python}"
export PYTHONDONTWRITEBYTECODE=1
exec "$runtime_python" "$repository_root/scripts/run/episode.py" "$@"
