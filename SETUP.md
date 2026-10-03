# Setup

Use Linux x86_64 with glibc 2.35 or later and an NVIDIA RTX GPU. The validation
host runs Ubuntu 22.04 with NVIDIA driver 570.181. PyTorch wheels include CUDA;
the system `nvcc` version is independent of PyTorch's CUDA version.

Before downloading or using dependencies, review the
[dependency and asset licenses](DEPENDENCIES.md#dependency-and-asset-licenses),
including checkpoint scope and terms for exported meshes.

## Sources and system packages

Git LFS and `uv 0.12.17` are required. On Ubuntu, install the following system packages:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-pip ca-certificates git git-lfs curl ffmpeg libglu1-mesa libegl1 libgl1 libgomp1
python3 -m pip install --user uv==0.12.17
export PATH="$HOME/.local/bin:$PATH"

git clone --recurse-submodules https://github.com/robotuse-team/RobotUse.git
cd RobotUse
scripts/setup/sources.sh
```

External sources are pinned submodules and snapshots. RoboLab includes about
6.2 GB of LFS assets; Python and GPU packages and caches also require tens of GB.
Do not run `pip install`, `uv sync`, or builds inside `third_party/`.

`scripts/setup/sources.sh` also works after an ordinary clone. It checks existing
sources before changing any checkout, synchronizes submodule URLs, downloads the
pinned commits recursively, pulls RoboLab LFS assets, and verifies the result.
It never follows the latest upstream branch or resets modified source files.
Commit or undo intentional changes to `.gitmodules` or staged submodule pins
before running it. If upstream source files or preserved snapshots differ, it
stops for inspection instead of repairing them.

To verify sources again without downloading or installing environments:

```bash
python3 scripts/check/setup.py --sources-only
```

## Virtual environments and checkpoints

For simulator episodes, install the three runtime environments:

```bash
scripts/setup/sam2.sh
scripts/setup/cgn.sh
scripts/setup/robolab.sh
```

Add the CPU environment for regression tests and configuration checks. Add the
UI environment only when using the web interface:

```bash
scripts/setup/cpu.sh
scripts/setup/ui.sh
```

| Environment | Python | Key pinned versions |
| --- | --- | --- |
| CPU checks | 3.11.9 | NumPy 2.4.6, pytest 8.4.2 |
| SAM2 | 3.11.9 | PyTorch 2.7.0+cu128, NumPy 1.26.4, official SAM2.1 Hiera Large |
| Contact-GraspNet | 3.11.9 | PyTorch 2.7.0+cu128, NumPy 1.26.4 |
| RoboLab | 3.11.13 | Isaac Sim 5.0.0, Isaac Lab 2.2.0, PyTorch 2.7.0+cu128, Warp 1.8.1 |
| Web UI | 3.11.9 | Gradio 6.29.0 |
| cuRobo (optional) | 3.11.13 | PyTorch 2.7.0+cu128, Warp 1.13.0 |

`requirements/*.txt` lists packages to install, pinning versions and distribution
hashes, including transitive dependencies. The `.in` files are inputs for
regenerating those lists; normal installation uses only the `.txt` files.
Install Isaac and cuRobo in separate environments because their dependencies conflict.

SAM2 is downloaded from the official URL and verified with SHA-256. CGN weights
and configuration are already included in the pinned submodule.
[configs/checkpoints.json](configs/checkpoints.json) records models, source
revisions, download URLs, and hashes. Do not add another copy of the weights to Git.

Environments are installed in `runtime/tool-envs/<environment>` by default, and
models in `runtime/model-cache/`. To use another location, set
`export ROBOTUSE_RUNTIME_ROOT=/absolute/path/runtime` before installation.
Installation through symbolic links is rejected to avoid accidentally modifying
an existing installation.

For separate CPU or UI environment locations, set `CPU_ENVIRONMENT` or
`UI_ENVIRONMENT` before setup and launch. Launchers and environment checks select
the interpreter in this order: `ROBOTUSE_CPU_PYTHON` / `ROBOTUSE_UI_PYTHON`,
the corresponding `*_ENVIRONMENT/bin/python`, then the runtime default above.
The `*_PYTHON` overrides select an existing interpreter; installation uses the
environment directory settings.

The native dependency lock preserves the runtime versions used for validation.
Some differ from Isaac wheel metadata for packages such as Pillow and websockets;
dependency resolution overrides are recorded in `requirements/robolab-overrides.txt`.
Setup does not modify upstream source or metadata. Upgrading Isaac Sim to 5.1 or
later also changes the conditions for reproducing physics results. See the
[Isaac Sim 5.0 installation guide in the Isaac Lab 2.2 documentation](https://isaac-sim.github.io/IsaacLab/v2.2.0/source/setup/installation/pip_installation.html#installing-isaac-sim)
for the upstream installation workflow.

## Runtime environment

```bash
source scripts/lib/env.sh

# Set this after reviewing and accepting the Isaac Sim terms of use.
export OMNI_KIT_ACCEPT_EULA=Y

# Core installation check; use the simulator interpreter already installed.
"$ROBOLAB_PYTHON" scripts/check/setup.py --profiles sam2 cgn robolab --gpu

# Optional CPU checks, after scripts/setup/cpu.sh:
"$ROBOTUSE_CPU_PYTHON" -m pytest
```

The checker verifies source revisions, original file hashes, checkpoints,
LFS assets, package versions, and the GPU without calling an LLM. Existing
`CUDA_VISIBLE_DEVICES` is preserved. Set `ROBOTUSE_GPU` or `ROBOTUSE_CGN_GPU`
to a physical GPU index to override that selection for the simulator or CGN,
respectively; with no selection, both use GPU 0.

To check inference through the actual CGN service, run the following command.
It starts the service on a temporary port and stops it after the check.

```bash
scripts/setup/cgn.sh --smoke-test
```

Agent execution requires provider credentials. Supply keys through environment
variables in the launch terminal, not through the repository or UI.

```bash
export ROBOT_LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY='your-key'
export ROBOT_LLM_MODEL='your-provider-model-id'
# Direct Google access: use ROBOT_LLM_PROVIDER=google and GOOGLE_AI_STUDIO_KEY.

scripts/run/robolab.sh --task BananaInBowlTask --seed 0 --output-dir runs/example --dry-run
```

`--dry-run` checks configuration. Remove it and use a new output directory for
an actual run. Specify a model ID available from your provider.

## Native check without an LLM

```bash
ROBO_RENDER_GPU=0 "$ROBOLAB_PYTHON" scripts/check/native.py \
  --sam-python "$PWD/src/tools/perception/python.sh" \
  --sam2-snapshot "$ROBOTUSE_RUNTIME_ROOT/model-cache/sam2" \
  --output-dir runs/native-check
```

The default check covers movement, observation, geometry, and grasp candidate
planning. Add `--execute` to perform grasp, place, and release as well. Default
pixel coordinates are set for `BananaInBowlTask` with seed 7; use `--pick-uv` and
`--place-uv` for other scenes.

`result.json` reports the integration check, while `verifier.json` reports the
task outcome and native subtask `score` (0–1). Score reporting is enabled by
default; `--no-task-score` disables it without changing simulation or success
criteria. The separate `reward` field is the environment's reward and can remain
zero even when the task score is one.

## Optional cuRobo

```bash
scripts/setup/curobo.sh
source scripts/lib/env.sh
"$ROBOLAB_PYTHON" -m src.tools.curobo.robot_assets \
  --native-config runs/native-check/native/env_cfg.json \
  --output-dir assets/robolab-curobo
```

At runtime, specify `--transit-planner curobo --curobo-robot-file <robot.yaml>
--curobo-calibration-file <calibration.json>`. Installation alone does not enable
cuRobo. Robot, joint, gripper, and coordinate calibration must match the actual
RoboLab assets.
