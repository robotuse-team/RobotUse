#!/usr/bin/env bash
# OpenRouter preset using credentials from the launch environment.
set -euo pipefail
repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

export ROBOTUSE_GPU=0 ROBOTUSE_CGN_GPU=0
export ROBOT_LLM_PROVIDER=openrouter
export ROBOT_LLM_MODEL=google/gemini-3.8-flash
export OMNI_KIT_ACCEPT_EULA=Y
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY in the launch environment}"
exec "$repository_root/scripts/run/ui.sh" "$@"
