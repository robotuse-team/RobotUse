"""Persistent cuRobo JSON worker with its own CUDA and Warp runtime.

Run by ``process.py`` in the explicitly configured cuRobo interpreter. The
parent retains RoboLab; only this child imports the original GPU planner.
"""

from __future__ import annotations

from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace


def _handle_request(request):
    from src.tools.curobo.adapter import (
        CuroboConfigurationError,
        _load_implementation,
        _runtime_source,
    )

    if not isinstance(request, dict):
        raise ValueError("cuRobo worker request must be a JSON object")
    operation = request.get("operation")
    if operation == "probe":
        return _runtime_source()
    if operation != "plan":
        raise ValueError("cuRobo worker operation must be 'probe' or 'plan'")

    import numpy as np

    def vector(name, length):
        value = np.asarray(request[name], dtype=float)
        if value.shape != (length,) or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain {length} finite values")
        return value

    position = vector("position", 3)
    quaternion = vector("quaternion_wxyz", 4)
    start_joints = vector("start_joints", 7)
    options = request["options"]
    if not isinstance(options, dict):
        raise ValueError("cuRobo planner options must be a JSON object")
    rows = request["world"]["mesh"]
    if not isinstance(rows, list):
        raise ValueError("cuRobo world.mesh must be a list")
    # The unchanged wrapper reads these exact mesh fields and performs its
    # original float32/int32 scene conversion. Keep poses and geometry intact.
    meshes = []
    for row in rows:
        if (not isinstance(row, dict) or row.get("vertices") is None
                or row.get("faces") is None):
            raise ValueError("cuRobo world contains an incomplete mesh")
        meshes.append(SimpleNamespace(**row))
    world = SimpleNamespace(mesh=meshes, observed_points=[])

    implementation = _load_implementation()
    planner = implementation._get_pose_planner(
        **options, with_collision=bool(meshes), mesh_cache=max(len(meshes) + 4, 32))
    joint_names = list(planner.joint_names)
    tool_frames = list(planner.tool_frames)
    if joint_names != request["expected_joint_names"]:
        raise CuroboConfigurationError("loaded cuRobo planner joint order differs from calibration")
    if tool_frames != request["expected_tool_frames"]:
        raise CuroboConfigurationError("loaded cuRobo tool frame differs from calibration")
    success, trajectory = implementation.plan_to_pose(
        position, quaternion, start_joints, **options,
        tcp_offset=None, world_config=world)
    return {
        "success": bool(success),
        "trajectory": None if trajectory is None else np.asarray(trajectory).tolist(),
        "joint_names": joint_names,
        "tool_frames": tool_frames,
    }


def serve(input_stream, output_stream):
    """Return one JSON response per input line, retaining the planner cache."""
    for line in input_stream:
        try:
            with redirect_stdout(sys.stderr):
                result = _handle_request(json.loads(line))
            response = json.dumps({"ok": True, "result": result}, allow_nan=False)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = json.dumps({"ok": False, "error": str(exc)}, allow_nan=False)
        output_stream.write(response + "\n")
        output_stream.flush()


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    # Save a dedicated protocol descriptor before redirecting fd 1. This also
    # keeps native-library stdout and subprocess build logs off the JSON pipe.
    sys.stdout.flush()
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1) as protocol:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        with redirect_stdout(sys.stderr):
            serve(sys.stdin, protocol)


if __name__ == "__main__":
    main()
