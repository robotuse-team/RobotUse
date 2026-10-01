#!/usr/bin/env bash
# Keep Isaac's bundled Python/CUDA libraries out of the SAM2 worker.
set -euo pipefail
repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
runtime_root="${ROBOTUSE_RUNTIME_ROOT:-$repository_root/runtime}"
sam_python="${ROBOTUSE_SAM_PYTHON:-${SAM_RUNTIME_PYTHON:-${SAM2_ENVIRONMENT:-$runtime_root/tool-envs/sam2}/bin/python}}"
unset PYTHONHOME LD_PRELOAD LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repository_root"
exec "$sam_python" "$@"
