"""CPU protocol and isolation checks; these do not verify GPU planning."""

import io
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.tools.curobo import adapter, process, worker


@pytest.fixture
def transport_child(tmp_path, monkeypatch):
    executable = tmp_path / "fake-curobo-python"
    executable.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys, time
if os.environ.get("ROBOTUSE_WORKER_TEST_EXIT_AT_START") == "1":
    raise SystemExit(7)
count = 0
for line in sys.stdin:
    request = json.loads(line)
    count += 1
    mode = request.get("mode")
    if mode == "hang":
        time.sleep(60)
    elif mode == "malformed":
        print("not JSON", flush=True)
    elif mode == "error":
        print(json.dumps({"ok": False, "error": "fixture planning failure"}), flush=True)
    else:
        print("fixture diagnostic", file=sys.stderr, flush=True)
        result = {"pid": os.getpid(), "count": count, "request": request,
                  "env": dict(os.environ)}
        print(json.dumps({"ok": True, "result": result}), flush=True)
''')
    executable.chmod(0o700)
    monkeypatch.setenv("ROBOTUSE_CUROBO_PYTHON", str(executable))
    monkeypatch.setenv("ROBOTUSE_CUROBO_TIMEOUT_S", "2")
    monkeypatch.setenv("ROBOTUSE_CUROBO_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.delenv("ROBOTUSE_CUROBO_LIBRARY_PATH", raising=False)
    return executable


def test_persistent_transport_sanitizes_environment_and_closes(transport_child, monkeypatch):
    excluded = ["PYTHONHOME", "PYTHONEXE", "VIRTUAL_ENV", "LD_LIBRARY_PATH", "LD_PRELOAD",
                "OPENAI_API_KEY", "OPENROUTER_API_KEY", "GOOGLE_AI_STUDIO_KEY",
                "GEMINI_API_KEY", "GOOGLE_API_KEY"]
    for key in excluded:
        monkeypatch.setenv(key, "fixture-parent-only")
    monkeypatch.setenv("PYTHONPATH", "/fixture/isaac/python")
    monkeypatch.setenv("ROBOTUSE_CUROBO_PYTHONPATH", "/fixture/curobo/dependencies")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    before = set(sys.modules)
    child = process.CuroboWorker()
    try:
        first = child.call({"operation": "echo", "value": [1, 2, 3]})
        second = child.call({"operation": "echo", "value": "second"})
        assert first["pid"] == second["pid"] == child.process.pid
        assert (first["count"], second["count"]) == (1, 2)
        assert first["request"]["value"] == [1, 2, 3]
        assert not set(excluded).intersection(first["env"])
        assert "/fixture/isaac/python" not in first["env"]["PYTHONPATH"]
        assert first["env"]["PYTHONPATH"].endswith("/fixture/curobo/dependencies")
        assert first["env"]["CUDA_VISIBLE_DEVICES"] == "0"
        assert not {"torch", "warp", "curobo"}.intersection(set(sys.modules) - before)
    finally:
        child.close()
    child.close()
    assert child.process.poll() is not None
    assert child.process.stdin.closed and child.process.stdout.closed and child._log.closed
    assert child not in process._WORKERS
    assert "fixture diagnostic" in child.log_path.read_text()


def test_only_explicit_curobo_library_path_is_used(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/fixture/isaac/libraries")
    monkeypatch.setenv("ROBOTUSE_CUROBO_LIBRARY_PATH", "/fixture/curobo/libraries")
    assert process.worker_environment()["LD_LIBRARY_PATH"] == "/fixture/curobo/libraries"


@pytest.mark.parametrize("mode,match", [
    ("error", "fixture planning failure"),
    ("malformed", "Expecting value"),
    ("hang", "timed out"),
])
def test_failed_transport_stops_owned_child(transport_child, monkeypatch, mode, match):
    if mode == "hang":
        monkeypatch.setenv("ROBOTUSE_CUROBO_TIMEOUT_S", ".1")
    child = process.CuroboWorker()
    try:
        with pytest.raises(RuntimeError, match=match) as error:
            child.call({"mode": mode})
        assert str(child.log_path) in str(error.value)
        assert child.process.poll() is not None
        assert child not in process._WORKERS
    finally:
        child.close()


def test_dead_child_closes_all_streams_and_reports_its_log(transport_child, monkeypatch):
    monkeypatch.setenv("ROBOTUSE_WORKER_TEST_EXIT_AT_START", "1")
    child = process.CuroboWorker()
    try:
        assert child.process.wait(timeout=2) == 7
        with pytest.raises(RuntimeError, match="Broken pipe") as error:
            child.call({"operation": "probe"})
        assert str(child.log_path) in str(error.value)
        assert child.process.stdin.closed and child.process.stdout.closed and child._log.closed
        assert child not in process._WORKERS
    finally:
        child.close()


def test_next_explicit_request_replaces_failed_child_without_retry(transport_child):
    # Exercise only child ownership; calibrated planning has separate fixtures.
    planner = object.__new__(adapter.CuroboPlanner)
    planner._worker = None
    try:
        first = planner._get_worker()
        with pytest.raises(RuntimeError, match="fixture planning failure"):
            first.call({"mode": "error"})
        assert first.process.poll() is not None
        assert planner._worker is first  # No automatic replay of the failed request.
        second = planner._get_worker()
        assert second is not first and second.process.pid != first.process.pid
        response = second.call({"operation": "probe"})
        assert response["count"] == 1 and response["request"] == {"operation": "probe"}
        assert planner._get_worker() is second
    finally:
        planner.close()
    assert second.process.poll() is not None


def _plan_request():
    return dict(
        operation="plan", options=dict(robot_file="explicit.yml", position_threshold=.005,
            rotation_threshold=.05, num_ik_seeds=32, use_cuda_graph=False),
        expected_joint_names=[f"panda_joint{i}" for i in range(1, 8)],
        expected_tool_frames=["base_link"], position=[.3, .2, .1],
        quaternion_wxyz=[1., 0., 0., 0.], start_joints=[0.] * 7,
        world=dict(mesh=[dict(name="observed_mesh", pose=[.1, .2, .3, 1., 0., 0., 0.],
            vertices=[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], faces=[[0, 1, 2]])]))


def test_worker_preserves_mesh_and_planner_inputs(monkeypatch):
    request = _plan_request()
    calls = []

    def create(**options):
        assert options == {**request["options"], "with_collision": True, "mesh_cache": 32}
        return SimpleNamespace(joint_names=request["expected_joint_names"], tool_frames=["base_link"])

    def plan(position, quaternion, joints, **options):
        calls.append(options)
        assert position.tolist() == request["position"]
        assert quaternion.tolist() == request["quaternion_wxyz"]
        assert joints.tolist() == request["start_joints"]
        assert options["tcp_offset"] is None
        assert vars(options["world_config"].mesh[0]) == request["world"]["mesh"][0]
        assert options["world_config"].observed_points == []
        return True, np.zeros((2, 7))

    monkeypatch.setattr(adapter, "_load_implementation", lambda: SimpleNamespace(
        _get_pose_planner=create, plan_to_pose=plan))
    result = worker._handle_request(request)
    assert result == dict(success=True, trajectory=[[0.] * 7] * 2,
                         joint_names=request["expected_joint_names"], tool_frames=["base_link"])
    assert len(calls) == 1


@pytest.mark.parametrize("field,match", [
    ("joint_names", "joint order"), ("tool_frames", "tool frame"),
])
def test_worker_rejects_loaded_identity_before_planning(monkeypatch, field, match):
    request = _plan_request()
    planner = SimpleNamespace(joint_names=request["expected_joint_names"], tool_frames=["base_link"])
    setattr(planner, field, ["wrong"])
    monkeypatch.setattr(adapter, "_load_implementation", lambda: SimpleNamespace(
        _get_pose_planner=lambda **_: planner,
        plan_to_pose=lambda *_args, **_kwargs: pytest.fail("must reject before planning")))
    with pytest.raises(adapter.CuroboConfigurationError, match=match):
        worker._handle_request(request)


def test_server_recovers_from_bad_request_and_keeps_stdout_clean(monkeypatch, capsys):
    def probe():
        print("third-party diagnostic")
        return {"gpu_initialized": False}

    monkeypatch.setattr(adapter, "_runtime_source", probe)
    output = io.StringIO()
    worker.serve(io.StringIO('{\n{"operation":"probe"}\n'), output)
    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(responses) == 2 and responses[0]["ok"] is False
    assert responses[1] == {"ok": True, "result": {"gpu_initialized": False}}
    captured = capsys.readouterr()
    assert captured.out == "" and "third-party diagnostic" in captured.err


def test_main_redirects_native_stdout_and_imports_no_gpu():
    fixture = '''
import os, sys
from src.tools.curobo import adapter, worker
def probe():
    print("python diagnostic")
    os.write(1, b"native diagnostic\\n")
    return {"gpu_imported": any(name in sys.modules for name in ("torch", "warp", "curobo"))}
adapter._runtime_source = probe
worker.main()
'''
    result = subprocess.run([sys.executable, "-c", fixture],
        input='{"operation":"probe"}\n{"operation":"probe"}\n',
        text=True, capture_output=True, check=True, env=process.worker_environment())
    assert [json.loads(line) for line in result.stdout.splitlines()] == [
        {"ok": True, "result": {"gpu_imported": False}}] * 2
    assert "python diagnostic" in result.stderr and "native diagnostic" in result.stderr
