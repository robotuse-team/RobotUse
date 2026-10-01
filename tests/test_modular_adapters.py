"""Adapters preserve canonical implementations and native CLI ownership."""

import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
for relative in ("src/simulator/robolab/third_party/graph_as_policy", "src/simulator/robolab/third_party/graph_as_policy/gap-core/src"):
    sys.path.insert(0, str(ROOT / relative))


@pytest.mark.parametrize("adapter_name, original_name, exports", [
    ("src.simulator.robolab.adapter", "gap.envs.robolab_env", ("RoboLabEnv", "make_env")),
    ("src.simulator.robolab.calibration", "src.simulator.robolab.gripper",
     ("FLANGE_FROM_GRASP", "GRASP_TO_EE", "RoboLabGripperAssets")),
    ("src.simulator.robolab.calibration", "src.simulator.robolab.configuration",
     ("load_calibration",)),
    ("src.simulator.robolab.local_planner", "src.simulator.robolab.local_planner",
     ("plan_observed_transit",)),
    ("src.tools.grasp.adapter", "src.tools.grasp.backend", ("GraspBackend",)),
    ("src.tools.grasp.adapter", "src.tools.grasp.cgn_client", ("ContactGraspNetClient",)),
    ("src.tools.place.adapter", "src.tools.place.execution", ("PlacementMotionMixin",)),
    ("src.tools.motion.adapter", "src.tools.grasp.execution", ("plan_explicit_grasp",)),
    ("src.tools.perception.adapter", "src.tools.perception.rgbd_adapter",
     ("PointGeometry", "PointRGBDAdapter")),
    ("src.tools.perception.adapter", "src.tools.perception.multiview_adapter",
     ("MultiviewPointRGBDAdapter",)),
])
def test_adapter_exposes_same_implementation_objects(adapter_name, original_name, exports):
    adapter = importlib.import_module(adapter_name)
    original = importlib.import_module(original_name)
    for name in exports:
        assert getattr(adapter, name) is getattr(original, name)


def test_factory_and_backend_getters_preserve_identity():
    from src.simulator.robolab.adapter import get_factory
    from src.tools.grasp.adapter import get_backend_class
    from gap.envs.robolab_env import make_env
    from src.tools.grasp.backend import GraspBackend

    assert get_factory() is make_env
    assert get_backend_class() is GraspBackend


def test_cli_delegates_to_canonical_module_ownership(monkeypatch):
    from src.simulator.robolab import adapter
    from src.simulator.robolab import cli as robolab_cli

    main = object()
    calls = []
    monkeypatch.setattr(robolab_cli, "OWNS_PROCESS", False)

    def run(callback):
        robolab_cli.OWNS_PROCESS = True
        calls.append(callback)
        return 7

    monkeypatch.setattr(robolab_cli, "run_native_cli", run)
    assert adapter.run_native_cli(main) == 7
    assert calls == [main]
    assert robolab_cli.OWNS_PROCESS is True
    assert "OWNS_PROCESS" not in vars(adapter)


def test_adapter_imports_do_not_load_simulator_or_model_runtimes():
    code = """
import importlib
import sys
for name in (
    'src.simulator.robolab.adapter',
    'src.simulator.robolab.calibration',
    'src.simulator.robolab.local_planner',
    'src.tools.grasp.adapter',
    'src.tools.place.adapter',
    'src.tools.motion.adapter',
    'src.tools.perception.adapter',
):
    importlib.import_module(name)
assert not any(name.split('.')[0] in (
    'isaaclab', 'isaacsim', 'robolab', 'torch', 'mujoco', 'open3d',
) for name in sys.modules)
"""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(str(ROOT / path) for path in (
        ".", "src/simulator/robolab/third_party/graph_as_policy", "src/simulator/robolab/third_party/graph_as_policy/gap-core/src",
    ))}
    subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT, check=True)


def test_pinned_robolab_wins_when_wrapper_appended_it_after_another_copy(tmp_path):
    package = tmp_path / 'robolab'
    package.mkdir()
    (package / '__init__.py').write_text('raise AssertionError("wrong RoboLab")\n')
    code = '''
import importlib.util
import sys
from pathlib import Path
from src.simulator.robolab.adapter import ROBOLAB_SOURCE, initialize
sys.path.insert(0, sys.argv[1])
sys.path.append(str(ROBOLAB_SOURCE))
initialize()
origin = Path(importlib.util.find_spec('robolab').origin).resolve()
assert origin.is_relative_to(ROBOLAB_SOURCE.resolve()), origin
'''
    env = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
    subprocess.run([sys.executable, '-c', code, str(tmp_path)], env=env, cwd=ROOT, check=True)
