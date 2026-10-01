<div align="center">

<h1>RobotUse</h1>
<h3>Allocating Computation, Context, and Decisions</h3>

<p>
  <a href="https://junhoo.me/">Junhoo Lee</a><sup>1,*</sup> &middot;
  <a href="https://injun-baek.github.io/">Injun Baek</a><sup>1,*</sup> &middot;
  Seungyeon Kim<sup>1</sup> &middot;
  Suhyun Jeon<sup>2</sup><br>
  Minkyu Kim<sup>1</sup> &middot;
  Baekseung Kim<sup>1</sup> &middot;
  Nojun Kwak<sup>1,&dagger;</sup>
</p>

<p><sup>1</sup> Seoul National University (SNU) &nbsp;&nbsp; <sup>2</sup> KAIST</p>

<p><sup>*</sup> Equal contribution &nbsp;&nbsp; <sup>&dagger;</sup> Corresponding author</p>

<p>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/arXiv-Coming_soon-red">
    <img src="https://img.shields.io/badge/arXiv-Coming_soon-red" alt="arXiv: Coming soon" title="Coming soon">
  </picture>
  <a href="https://robotuse-team.github.io/"><img src="https://img.shields.io/badge/Project_Page-RobotUse-green" alt="Project Page: RobotUse"></a>
</p>

</div>

---

<p align="center"><strong>Task:</strong> Put the small red yogurt in the red bowl.</p>

<p align="center">
  <img src="assets/yogurt-in-bowl-comparison.gif" alt="Yogurt in bowl: Ours, CaP-X, and GaP (ORS), shown side by side at 2× speed. Completed clips hold their final frame and show Success or Failure." width="100%">
</p>

<p align="center"><em>2× playback. ORS = Open Robot Skills.</em></p>

---

**RobotUse** is a robot agent harness that lets language-model agents specify and
revise physical actions through visual target selection, grasp selection, and
pose editing. Subagents keep detailed interactions in local contexts and return
outcomes and unresolved constraints to the main agent, while the backend handles
geometry, motion planning, and control.

<p align="center">
  <img src="docs/media/robotuse-overview.png" alt="RobotUse overview: a main agent delegates visual action choices to a subagent and robot backend; a separate loop refines the playbook across episodes." width="100%">
</p>

<p align="center"><em>Overview from the paper. Agents decide in language and on images; the backend plans and controls. The dashed loop shows playbook refinement across episodes.</em></p>

## News

- **2026-10-01:** Added the initial codebase, setup guide, and web UI.

## Setup and execution

Linux and an NVIDIA RTX GPU are required. The [setup guide](SETUP.md) covers
virtual environments, pinned dependencies and checkpoints, GPU configuration,
and LLM credentials.
Review the [dependency and asset licenses](DEPENDENCIES.md#dependency-and-asset-licenses):
Contact-GraspNet and some RoboLab assets have noncommercial restrictions.

```bash
git clone --recurse-submodules https://github.com/robotuse-team/RobotUse.git
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

Launch the UI with the OpenRouter preset:

```bash
scripts/run/ui_openrouter.sh
```

The preset defaults to `google/gemini-3.8-flash` and preserves your model and GPU
settings. Opening the UI, browsing history, and checking configuration do not
require an API key. For actual episodes, export `OPENROUTER_API_KEY` in the launch
environment and complete the [runtime setup](SETUP.md#runtime-environment),
including Isaac Sim terms acceptance. For another provider, configure your
environment as described in [SETUP.md](SETUP.md) and use `scripts/run/ui.sh`.

Open [http://127.0.0.1:7860](http://127.0.0.1:7860):

1. Select **Task** and **Seed**, then choose `v0`, `v1`, `v2`, or `v3` from
   **Playbook**. The default is `v3`.
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
  --playbook-version v3 --output-dir runs/example --dry-run
```

Use `--playbook-version` to select `v0`, `v1`, `v2`, or `v3`.
Omitting this option uses `v3`.

Remove `--dry-run` to execute an episode. Use a new output directory for each run.
Run `scripts/run/robolab.sh --help` for the supported options. Execution stages
and candidate selection policies are fixed internally.
The default planner is native. cuRobo requires
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
