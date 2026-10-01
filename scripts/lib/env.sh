#!/usr/bin/env bash
# Source this file in bash/zsh after setup. It does not enable optional cuRobo.
if [ -n "${BASH_VERSION:-}" ]; then
  robotuse_env_source="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
  robotuse_env_source="${(%):-%N}"
else
  echo 'Source scripts/lib/env.sh from bash or zsh.' >&2
  return 1
fi
robotuse_repository="$(cd -- "$(dirname -- "$robotuse_env_source")/../.." && pwd)"
export ROBOTUSE_RUNTIME_ROOT="${ROBOTUSE_RUNTIME_ROOT:-$robotuse_repository/runtime}"
export ROBOTUSE_CPU_PYTHON="${ROBOTUSE_CPU_PYTHON:-${CPU_ENVIRONMENT:-$ROBOTUSE_RUNTIME_ROOT/tool-envs/cpu}/bin/python}"
export ROBOTUSE_UI_PYTHON="${ROBOTUSE_UI_PYTHON:-${UI_ENVIRONMENT:-$ROBOTUSE_RUNTIME_ROOT/tool-envs/ui}/bin/python}"
export ROBOLAB_PYTHON="${ROBOLAB_PYTHON:-$ROBOTUSE_RUNTIME_ROOT/tool-envs/robolab/bin/python}"
export ROBOTUSE_SAM_PYTHON="${ROBOTUSE_SAM_PYTHON:-${SAM_RUNTIME_PYTHON:-${SAM2_ENVIRONMENT:-$ROBOTUSE_RUNTIME_ROOT/tool-envs/sam2}/bin/python}}"
export ROBOTUSE_CGN_PYTHON="${ROBOTUSE_CGN_PYTHON:-$ROBOTUSE_RUNTIME_ROOT/tool-envs/cgn/bin/python}"
export ROBOTUSE_CUROBO_PYTHON="${ROBOTUSE_CUROBO_PYTHON:-${CUROBO_ENVIRONMENT:-$ROBOTUSE_RUNTIME_ROOT/tool-envs/curobo}/bin/python}"
if [ "${CUDA_VISIBLE_DEVICES+x}" != x ]; then
  export ROBOTUSE_GPU="${ROBOTUSE_GPU-0}"
  export ROBOTUSE_CGN_GPU="${ROBOTUSE_CGN_GPU-0}"
else
  if [ "${ROBOTUSE_GPU+x}" = x ]; then export ROBOTUSE_GPU; fi
  if [ "${ROBOTUSE_CGN_GPU+x}" = x ]; then export ROBOTUSE_CGN_GPU; fi
fi
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROBOTUSE_RUNTIME_ROOT/cache}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$ROBOTUSE_RUNTIME_ROOT/cache/torch-extensions}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$ROBOTUSE_RUNTIME_ROOT/cache/warp}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$ROBOTUSE_RUNTIME_ROOT/cache/cuda}"
unset robotuse_repository robotuse_env_source
