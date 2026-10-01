#!/usr/bin/env bash
# Create isolated tool environments without writing into upstream checkouts.
robotuse_create_venv() {
  local destination="$1" python_version="$2"
  command -v uv >/dev/null || { echo 'Install uv 0.12.17 first (see SETUP.md).' >&2; return 1; }
  python3 - "$destination" "$repository_root" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1]).expanduser().absolute()
if {'third_party', 'vendor'} & set(path.parts):
    raise SystemExit('Environment output must be outside third_party and vendor.')
if path.resolve().is_relative_to(Path(sys.argv[2]).resolve() / 'src'):
    raise SystemExit('Environment output must be outside src.')
for parent in (path, *path.parents):
    if parent.is_symlink():
        raise SystemExit(f'Refusing a symlinked environment path: {parent}')
if path.exists() and not (path / 'pyvenv.cfg').is_file():
    raise SystemExit(f'Existing directory is not a virtual environment: {path}')
if (path / 'bin').is_symlink():
    raise SystemExit('Refusing a symlinked virtual-environment bin directory.')
PY
  if [[ ! -x "$destination/bin/python" ]]; then
    uv venv --python "$python_version" "$destination"
  fi
  "$destination/bin/python" - "$python_version" "$destination" <<'PY'
from pathlib import Path
import sys
expected = sys.argv[1]
actual = '.'.join(map(str, sys.version_info[:3]))
if actual != expected:
    raise SystemExit(f'Python {expected} is required; found {actual}. Use a fresh environment directory.')
if sys.prefix == sys.base_prefix or Path(sys.prefix).resolve() != Path(sys.argv[2]).resolve():
    raise SystemExit('Selected Python does not belong to the requested virtual environment.')
PY
}
