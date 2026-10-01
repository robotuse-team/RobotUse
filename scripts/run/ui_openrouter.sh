#!/usr/bin/env bash
# OpenRouter preset using credentials from the launch environment.
set -euo pipefail
repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

export ROBOT_LLM_PROVIDER=openrouter
export ROBOT_LLM_MODEL="${ROBOT_LLM_MODEL:-google/gemini-3.8-flash}"
exec "$repository_root/scripts/run/ui.sh" "$@"
