#!/usr/bin/env bash
# Keep official source read-only; provision only a dedicated env and checkpoint.
set -euo pipefail
unset PYTHONPATH PYTHONHOME PYTHONEXE LD_LIBRARY_PATH LD_PRELOAD VIRTUAL_ENV
repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
runtime_root="${ROBOTUSE_RUNTIME_ROOT:-$repository_root/runtime}"
sam2_environment="${SAM2_ENVIRONMENT:-$runtime_root/tool-envs/sam2}"
sam2_checkpoint="${SAM2_CHECKPOINT:-$runtime_root/model-cache/sam2/sam2.1_hiera_large.pt}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1

test -f "$repository_root/src/tools/perception/third_party/sam2/sam2/build_sam.py" || {
  echo 'Run git submodule update --init --recursive first.' >&2; exit 1;
}
source "$repository_root/scripts/lib/venv.sh"
robotuse_create_venv "$sam2_environment" 3.11.9
uv pip sync --python "$sam2_environment/bin/python" --require-hashes \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match "$repository_root/requirements/sam2.txt"
"$sam2_environment/bin/python" "$repository_root/scripts/setup/provision_sam2.py" "$sam2_checkpoint"
printf 'SAM2 environment ready: %s\nCheckpoint: %s\n' "$sam2_environment/bin/python" "$sam2_checkpoint"
