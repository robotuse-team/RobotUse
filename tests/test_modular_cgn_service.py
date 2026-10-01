"""The standalone service uses tool-local originals and isolated dependencies."""

from pathlib import Path
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.tools.grasp import service, service_worker


def test_service_snapshot_is_unchanged_and_worker_help_is_cpu_only():
    manifest = service_worker.verify_service_source()
    assert manifest["public_upstream"]["repository"] == "https://github.com/Max-Fu/CaP-X"
    assert manifest["public_upstream"]["revision"] == "823fcc5dd3e565b45b414f5785668cf32cba13b4"
    result = subprocess.run([sys.executable, str(Path(service_worker.__file__)), "--help"],
                            check=True, capture_output=True, text=True)
    assert "--host" in result.stdout and "--device" in result.stdout


def test_missing_model_assets_fail_before_importing_service(tmp_path, monkeypatch):
    monkeypatch.setattr(service_worker, "CGN_SOURCE", tmp_path)
    monkeypatch.setattr(service_worker.subprocess, "check_output",
                        lambda *a, **k: "da3dcfb2f53e43b186083ee4a9d1e232f73efc98\n")
    monkeypatch.setattr(service_worker.importlib, "import_module",
                        lambda *a, **k: pytest.fail("must not import GPU service before validating assets"))
    with pytest.raises(FileNotFoundError, match="Pinned CGN asset is missing"):
        service_worker.create_service()


@pytest.mark.parametrize('gpu,inherited,visibility,physical', [
    (None, None, '0', 0),
    ('0', '2,3', '0', 0),
    ('3', '7', '3', 3),
    (None, '2,3', '2,3', 2),
    (None, 'GPU-example', 'GPU-example', None),
    (None, '', '', None),
])
def test_local_start_uses_worker_with_explicit_dependency_paths(tmp_path, monkeypatch,
                                                              gpu, inherited, visibility, physical):
    monkeypatch.setenv("ROBOTUSE_CGN_PYTHON", sys.executable)
    monkeypatch.setenv("ROBOTUSE_CGN_PYTHONPATH", "/runtime/cgn-dependencies")
    monkeypatch.setenv("ROBOTUSE_CGN_LOCK_DIR", str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", "/unrelated/simulator")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    if inherited is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", inherited)
    monkeypatch.delenv("ROBOTUSE_CGN_GPU", raising=False)
    if gpu is not None:
        monkeypatch.setenv("ROBOTUSE_CGN_GPU", gpu)
    def offline(*args, **kwargs):
        raise ConnectionRefusedError()
    monkeypatch.setattr(service.socket, "create_connection", offline)
    captured = {}
    def spawn(command, **kwargs):
        captured.update(command=command, **kwargs)
        return SimpleNamespace(pid=12345, poll=lambda: None)
    monkeypatch.setattr(service.subprocess, "Popen", spawn)
    monkeypatch.setattr(service, "api_ready", lambda *a: None)
    monkeypatch.setattr(service, "inference_probe", lambda *a: {"inference_completed": True})
    result = service.ensure_cgn("http://127.0.0.1:18115", evidence_dir=tmp_path / "evidence")
    assert result["ready"]
    assert captured["command"][1] == str(Path(service_worker.__file__))
    assert captured["env"]["PYTHONPATH"] == "/runtime/cgn-dependencies"
    assert captured["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == visibility
    assert result["requested_gpu"] == visibility
    assert result["physical_gpu"] == physical
    assert result["gpu_source"] == "launch_environment"
    launched = json.loads((tmp_path / "evidence/server-process.json").read_text())
    assert launched["requested_gpu"] == visibility
    assert launched["physical_gpu"] == physical
    assert launched["gpu_source"] == "launch_environment"
    assert json.loads((tmp_path / "evidence/preflight.json").read_text()) == result
