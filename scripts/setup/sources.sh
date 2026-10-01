#!/usr/bin/env bash
set -euo pipefail
unset PYTHONPATH PYTHONHOME PYTHONEXE LD_LIBRARY_PATH LD_PRELOAD VIRTUAL_ENV
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
git lfs version >/dev/null || { echo 'Install git-lfs first (see SETUP.md).' >&2; exit 1; }
python3 "$repository_root/scripts/check/setup.py" --source-preflight
git -C "$repository_root" submodule sync --recursive
git -C "$repository_root" submodule update --init --recursive --checkout
git -C "$repository_root/src/simulator/robolab/third_party/robolab" lfs pull
python3 "$repository_root/scripts/check/setup.py" --sources-only
