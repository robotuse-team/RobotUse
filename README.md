# RobotUse

Robot agents. Prime decides what to do next, while Point, Grasp,
Place, and Refiner handle perception and manipulation. The orchestrator manages
sessions, tool permissions, delegation, and execution budgets.

## Setup and execution

Linux and an NVIDIA RTX GPU are required. The [setup guide](SETUP.md) covers
virtual environments, pinned dependencies and checkpoints, GPU configuration,
and LLM credentials.
Review the [dependency and asset licenses](DEPENDENCIES.md#dependency-and-asset-licenses):
Contact-GraspNet and some RoboLab assets have noncommercial restrictions.

```bash
git clone --recurse-submodules https://github.com/injun-baek/RobotUse.git
cd RobotUse
scripts/setup/sources.sh
```

Install Git LFS first, as described in the setup guide. RoboLab, Contact-GraspNet,
cuRobo, and official SAM2 stay at the commits pinned by this repository, under
their tool or simulator's `third_party/` directory. The source setup downloads
nested submodules and RoboLab LFS assets, then verifies revisions and recorded
file hashes. It stops if existing sources have been modified. cuRobo remains
disabled until explicitly configured and selected.

### Web UI

With `OPENROUTER_API_KEY` exported in `~/.zshrc`, run:

```bash
scripts/run/ui_openrouter.sh
```

This preset uses zsh to load your key, selects the installed virtual environment,
and configures GPU 0 and OpenRouter's `google/gemini-3.8-flash`. For other settings,
configure your environment as described in [SETUP.md](SETUP.md) and use
`scripts/run/ui.sh`.

If the UI runs on a remote server, open a tunnel **on your own computer** and
leave that terminal open. Replace `<ssh-host>` with your server's SSH alias:

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:7860:127.0.0.1:7860 <ssh-host>
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860):

1. Select **Task**, **Seed**, and **Playbook** (for example, `BananaInBowlTask`, `0`, `v0`).
2. Choose **Configuration check** to validate settings without starting the simulator
   or calling an LLM, or **Run episode** for actual simulator execution with LLM calls.
3. Click **Start**. Front and wrist views refresh every second; **Activity** shows
   the current operation. Images stay still while the simulator waits for an LLM.
4. Use **Stop** to cancel the selected run. **Recent episodes** opens earlier runs;
   **Refresh list** reloads that list.
5. After completion, watch the recordings and check **Native verifier** for the task
   outcome. **Process** reports execution status, which does not establish task success.

The **History** tab works for both running episodes and earlier entries in
**Recent episodes**. Use **Previous**, **Next**, or the **Turn** selector to inspect
each agent's input, LLM output, and tool result. Click an image to enlarge it;
**Recorded LLM input** expands the saved request messages. **Follow latest** tracks
new turns, and turns off when you navigate manually. Missing records are shown
as unavailable; viewing history does not call an LLM or alter the saved run.

### Command line

The CLI uses the same runner:

```bash
scripts/run/robolab.sh --task BananaInBowlTask --seed 0 \
  --output-dir runs/example --dry-run
```

Remove `--dry-run` to execute an episode. Use a new output directory for each run.
Run `scripts/run/robolab.sh --help` for the supported options. Execution stages
and candidate selection policies are fixed internally.
The default policy is v0 and the default planner is native. cuRobo requires
separate installation and calibration, and is enabled only when explicitly
selected. For a live RoboLab check without an LLM, see the
[setup guide](SETUP.md#native-check-without-an-llm).

## Files and results

- [ARCHITECTURE.md](ARCHITECTURE.md): Module responsibilities and automatic tool registration.
- [DEPENDENCIES.md](DEPENDENCIES.md): Third-party source provenance and licenses.
- `src/utils/logging_utils.py`: Shared logging and JSON output, with a `runtime.log` for each run.
- `assets/`: Generated robot input assets. `runs/`: Logs, videos, and outcomes for each run. Both are excluded from Git.
- `tests/`: Regression tests. `pyproject.toml`: Python package, dependency, and test configuration.

Use `verifier.json` to determine task success; process termination or an agent's
narration alone is not sufficient. Native subtask `score` (0–1) is reported by
default, separately from `reward`, and shown in the UI. Use `--no-task-score`
to disable score reporting; unavailable scores are `null`, not zero.
