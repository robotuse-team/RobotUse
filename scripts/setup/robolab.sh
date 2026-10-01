#!/usr/bin/env bash
# Install an isolated native simulator runtime; never install into upstream sources.
set -euo pipefail
unset PYTHONPATH PYTHONHOME PYTHONEXE LD_LIBRARY_PATH LD_PRELOAD VIRTUAL_ENV
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
runtime_root="${ROBOTUSE_RUNTIME_ROOT:-$repository_root/runtime}"
runtime_root="$(python3 - "$runtime_root" <<'PY'
import os
import sys
print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
)"
native_environment="$runtime_root/tool-envs/robolab"
native_python="$native_environment/bin/python"
base_python="${ROBOLAB_BASE_PYTHON:-3.11.13}"

command -v uv >/dev/null || { echo 'Install uv before running this script.' >&2; exit 1; }
python3 - "$runtime_root" "$native_environment" "$repository_root" <<'PY'
from pathlib import Path
import os
import platform
import sys

if platform.system() != "Linux" or platform.machine() != "x86_64":
    raise SystemExit("The native lock requires Linux x86_64.")
libc, version = platform.libc_ver()
if libc != "glibc" or tuple(map(int, version.split(".")[:2])) < (2, 35):
    raise SystemExit("IsaacSim 5.0 wheels require glibc 2.35 or newer.")
for value in sys.argv[1:3]:
    path = Path(os.path.abspath(os.path.expanduser(value)))
    if any(part in {"third_party", "vendor"} for part in path.parts):
        raise SystemExit("Runtime output must be outside all third_party/vendor directories.")
    if path.resolve().is_relative_to(Path(sys.argv[3]).resolve() / 'src'):
        raise SystemExit("Runtime output must be outside src.")
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise SystemExit(f"Refusing symlink in runtime output path: {ancestor}")
environment = Path(sys.argv[2])
if environment.exists() and not (environment / "pyvenv.cfg").is_file():
    raise SystemExit(f"Refusing an existing non-venv directory: {environment}")
if (environment / "bin").is_symlink():
    raise SystemExit("Refusing a symlinked virtual-environment bin directory.")
source = Path(sys.argv[3]) / "src/simulator/robolab/third_party/robolab"
if not (source / "robolab/__init__.py").is_file():
    raise SystemExit("Initialize the pinned RoboLab submodule before installing its runtime.")
robot = source / "assets/robots/franka_robotiq_2f_85_flattened.usd"
if not robot.is_file():
    raise SystemExit("RoboLab robot asset is missing; prepare its Git LFS assets first.")
with robot.open("rb") as stream:
    if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise SystemExit("RoboLab contains Git LFS pointers; run git lfs pull in its submodule.")
PY

if [[ ! -x "$native_python" ]]; then
  uv venv --python "$base_python" "$native_environment"
fi
"$native_python" - "$native_environment" <<'PY'
from pathlib import Path
import sys
if sys.version_info[:3] != (3, 11, 13):
    raise SystemExit("The native lock requires Python 3.11.13; select it with ROBOLAB_BASE_PYTHON.")
if sys.prefix == sys.base_prefix:
    raise SystemExit("Refusing to install outside a virtual environment.")
if Path(sys.prefix).resolve() != Path(sys.argv[1]).resolve():
    raise SystemExit("Selected Python belongs to a different virtual environment.")
PY

uv pip sync --python "$native_python" --require-hashes \
  --build-constraints "$repository_root/requirements/robolab-build.txt" \
  --index-strategy unsafe-best-match \
  "$repository_root/requirements/robolab.txt"

"$native_python" - <<'PY'
import importlib.metadata as metadata
import json
import torch
import warp

expected = {"isaacsim": "5.0.0.0", "isaaclab": "2.2.0", "torch": "2.7.0+cu128",
            "torchvision": "0.22.0+cu128", "warp-lang": "1.8.1", "numpy": "1.26.0",
            "Pillow": "12.3.0", "websockets": "17.1", "mujoco": "3.13.0",
            "embreex": "4.4.0", "open3d": "0.19.0"}
actual = {name: metadata.version(name) for name in expected}
if actual != expected:
    raise SystemExit(f"Installed native runtime differs from its lock: {actual}")
assert torch.version.cuda == "12.8"
assert warp.__version__ == "1.8.1"
import mujoco
import trimesh
assert trimesh.ray.has_embree, "Native geometry queries require embreex"
print(json.dumps({"status": "installed", "packages": actual,
                  "scope": "package imports only; no simulator or physical task validation"}, indent=2))
PY
