#!/usr/bin/env bash
# Local OpenRouter preset. zsh loads the API key from ~/.zshrc.
set -euo pipefail
repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

exec zsh -ic '
  export ROBOTUSE_GPU=0 ROBOTUSE_CGN_GPU=0
  export ROBOT_LLM_PROVIDER=openrouter
  export ROBOT_LLM_MODEL=google/gemini-3.8-flash
  export OMNI_KIT_ACCEPT_EULA=Y
  : "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY in ~/.zshrc}"
  exec "$@"
' robotuse-ui "$repository_root/scripts/run/ui.sh" "$@"
