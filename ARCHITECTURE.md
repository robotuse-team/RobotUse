# RobotUse architecture

RobotUse runs with native RoboLab and is organized by function. External source
code lives in each module's `third_party/` directory and is never modified.
Adapters outside those directories handle integration, coordinate calibration,
and process isolation.

```text
src/
  llm/
    manager.py                    # Request, response, and tool serialization; provider selection
    config.py, types.py           # LLM configuration and message contracts
    providers/
      openrouter.py, gemini.py    # OpenRouter and direct Google API calls
  agent/
    base_agent.py, session.py     # Inheritance base and conversations by role
    prime_agent.py                # Goal assessment and next-action decisions
    point_agent.py                # Target selection in images
    grasp_agent.py                # Grasp candidate selection
    place_agent.py                # Placement position and orientation selection
    refiner_agent.py              # Pose review and refinement
    playbook/                     # Policy loading and v0-v3 selection
      policies/                   # Policy text used at runtime
  backend/
    orchestrator.py               # Sessions, delegation, returns, and review order
    tool_executor.py              # Execution boundary for registered tools
    controller.py                 # Budgets and state transitions
    robot.py, robot_base.py       # Robot state and observation access
    *_policy.py, *_routes.py      # Execution and review policies
  runtime/                        # CLI, configuration, startup checks, episodes, and records
  core/                           # Shared types, errors, and input provenance contracts
  utils/logging_utils.py          # Shared logging and JSON records
  ui/                             # Task selection, execution, stopping, and results
  tools/
    base_tool.py, schema.py       # Tool declarations and input contracts
    discovery.py, registry.py     # Automatic discovery, validation, and registration
    list.py                       # Registered tool lookup
    grasp/                        # CGN, geometric candidates, grasp validation and execution
      third_party/                # Original CGN, service, and geometry helper sources
    place/                        # Observation-based placement and release validation
    curobo/                       # Optional transit planning and calibration asset generation
      third_party/                # Original cuRobo and wrapper sources
    perception/                   # Official SAM2, RGB-D, and observation fusion
      third_party/sam2/           # Pinned official SAM2
    observation/                  # Camera movement and observation
    pose_editor/                  # Mesh previews and pose refinement
      mesh_assets.py              # Shared mesh loading, transformation, and export
      third_party/                # Original robosuite assets and TiPToP sources
    motion/                       # Motion planning, coordinates, collision, and arrival checks
    gripper/                      # Grasp evidence, opening, and state
    coordination/                 # Delegation and termination tools
  simulator/robolab/
    adapter.py, calibration.py    # Upstream integration and coordinate calibration
    robot_model.py, gripper.py    # Actual robot and gripper geometry
    local_planner.py, collision.py # Native planning and collision checks
    validation.py                 # Native checks without an LLM
    verifier.py                   # Read native subtask scores for result reporting
    third_party/                  # Original RoboLab and GAP connector sources
configs/robot.json                # Default runtime configuration
configs/checkpoints.json          # Model sources, revisions, and hashes
requirements/                     # Pinned versions and hashes for each environment
scripts/
  setup/                          # Environment, source, and checkpoint setup by component
    sources.sh, cpu.sh, sam2.sh, cgn.sh, robolab.sh, ui.sh, curobo.sh
    provision_sam2.py             # Official checkpoint download and hash verification
  run/                            # Execution entry points
    robolab.sh, episode.py, ui.sh
  check/                          # Installation and native integration checks
    setup.py, native.py
  assets/gripper_mesh.py          # Gripper mesh export CLI
  lib/env.sh, lib/venv.sh         # Shared runtime environment and virtual environment setup
tests/                            # RobotUse regression tests
```

Each tool declares its name, schema, and input validation in its own `tool.py`.
Tools are discovered automatically rather than manually registered in `list.py`.
Missing declarations, duplicates, and invalid contracts cause startup errors.
The orchestrator separates Prime's decisions from execution control.

The default policy is v3 and the default planner is native. cuRobo is enabled
only when explicitly selected. Grasp, place, and release remain separate
operations with explicit input and output contracts.

The adapter provides the `robolab_cli` and `robolab_collision` names required by
the original GAP code as aliases of the current implementation objects. It does
not modify external code. Each run records the selected policy text and hash
alongside its execution configuration.

`assets/` contains generated robot inputs, and `runs/` contains execution outputs.
Both are excluded from Git. The native verifier determines task success;
execution exit codes and agent narration are not sufficient evidence.
