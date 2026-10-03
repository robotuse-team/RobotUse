# Public release

The main path is `scripts/run/robolab.sh` → `src/runtime/cli.py` →
`src/runtime/episode.py`. It runs a task with the native RoboLab backend,
SAM2 perception, Contact-GraspNet, and a language-model provider.

## Core layout

| Directory | Responsibility |
| --- | --- |
| `src/agent/` | Prime, Point, Grasp, Place, and Refiner roles; conversation sessions and playbooks |
| `src/backend/` | Delegation, tool execution, robot state, and episode control |
| `src/tools/` | Visual target selection, grasp/place, motion, observation, pose editing, and gripper tools |
| `src/simulator/robolab/` | RoboLab integration, calibration, native planning, and task verifier |
| `src/llm/` | OpenRouter and direct Google providers; message and tool serialization |
| `src/runtime/` | CLI options, startup checks, run configuration, budgets, and recording |
| `src/core/`, `src/utils/` | Shared contracts, errors, input provenance, GPU selection, and logging |
| `configs/` | Default robot options and checkpoint sources/hashes |

Tools declare contracts in their own `tool.py`; startup discovers and validates
them automatically. Playbook versions and their policy paths are declared once
in `src/agent/playbook/__init__.py`, shared by the CLI and UI. The default is v3.

## Install only the environments you use

| Use | Setup scripts |
| --- | --- |
| Run simulator episodes | `sources.sh`, `sam2.sh`, `cgn.sh`, `robolab.sh` |
| Add the web UI | `cpu.sh`, `ui.sh` |
| Run CPU regression tests | `cpu.sh` |
| Use cuRobo instead of native planning | `curobo.sh`, plus explicit robot/calibration files |

All setup scripts are in `scripts/setup/`. The UI is a thin client of the same
CLI runner; the CPU environment handles its configuration checks. cuRobo is
optional and disabled by default. Installation details and environment overrides
are in [SETUP.md](../SETUP.md).

Sources in each component's `third_party/` directory retain their pinned
revisions, recorded hashes, and upstream licenses. Adaptation happens outside
those directories. [DEPENDENCIES.md](../DEPENDENCIES.md) records their scope and
licenses; `requirements/*.txt` records each environment's package hashes.

## Checks and results

After provisioning sources, inspect their pins and assets without an LLM call:

```bash
python3 scripts/check/setup.py --sources-only
```

For CPU regression tests:

```bash
scripts/setup/cpu.sh
source scripts/lib/env.sh
"$ROBOTUSE_CPU_PYTHON" -m pytest
```

`--dry-run` resolves an episode's configuration without starting the simulator or
calling an LLM. [Native integration checks](../SETUP.md#native-check-without-an-llm)
exercise the live simulator without an LLM.

Run outputs belong in a fresh `runs/` subdirectory. `verifier.json` records the
native task outcome and score; logs and videos explain the execution. Generated
assets go in `assets/`; these and `runtime/` are excluded from Git.
