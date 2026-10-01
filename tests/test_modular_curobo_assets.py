"""Native export lifecycle and conservative collision coverage."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.simulator.robolab import cli
from src.tools.curobo.robot_assets import _cover_convex_mesh, export_robot_assets
from src.tools.curobo import robot_assets


def test_convex_cell_spheres_cover_vertices_edges_and_volume():
    vertices = np.array([[0., 0., 0.], [.14, 0., 0.], [0., .08, 0.], [0., 0., .11]])
    spheres = _cover_convex_mesh(vertices, .025)
    centers = np.array([sphere["center"] for sphere in spheres])
    radii = np.array([sphere["radius"] for sphere in spheres])
    rng = np.random.default_rng(1)
    weights = rng.dirichlet(np.ones(4), 2000)
    edges = np.array([a * t + b * (1 - t) for a in vertices for b in vertices
                      for t in np.linspace(0., 1., 25)])
    points = np.concatenate([vertices, edges, weights @ vertices])
    assert np.all(np.min(np.linalg.norm(points[:, None] - centers[None], axis=2) - radii, axis=1) <= 0.)
    assert radii.max() <= np.sqrt(3) * .025 / 2 + 1e-6


@pytest.mark.parametrize("directory", ["third_party", "vendor"])
def test_export_rejects_original_source_tree_before_reading_or_writing(tmp_path, directory):
    original = tmp_path / directory
    original.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(original, target_is_directory=True)
    with pytest.raises(ValueError, match="outside immutable"):
        export_robot_assets(tmp_path / "missing.json", alias / "generated")
    assert list(original.iterdir()) == []


@pytest.mark.parametrize("export_fails", [False, True])
@pytest.mark.parametrize("owns_process", [False, True])
def test_cli_initializes_isaac_and_respects_shutdown_ownership(monkeypatch, tmp_path, export_fails, owns_process):
    events = []
    config, output = tmp_path / "env_cfg.json", tmp_path / "assets"

    def launch(settings):
        assert settings == {"headless": True, "fast_shutdown": False}
        events.append("start")
        return SimpleNamespace(close=lambda: events.append("close"))

    def export(native_config, destination, *, cell_width_m):
        assert events == ["start"]
        assert (native_config, destination, cell_width_m) == (config, output, .025)
        events.append("export")
        if export_fails:
            raise RuntimeError("USD export failed")
        return output

    monkeypatch.setitem(sys.modules, "isaacsim", SimpleNamespace(SimulationApp=launch))
    monkeypatch.setattr(cli, "OWNS_PROCESS", owns_process)
    monkeypatch.setattr(robot_assets, "export_robot_assets", export)
    args = ["--native-config", str(config), "--output-dir", str(output), "--cell-width-m", ".025"]
    if export_fails:
        with pytest.raises(RuntimeError, match="USD export failed"):
            robot_assets.main(args)
    else:
        robot_assets.main(args)
    assert events == (["start", "export"] if owns_process else ["start", "export", "close"])


def test_cli_help_does_not_require_isaac(monkeypatch):
    monkeypatch.setitem(sys.modules, "isaacsim", None)
    with pytest.raises(SystemExit) as result:
        robot_assets.main(["--help"])
    assert result.value.code == 0
