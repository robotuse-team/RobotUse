"""Export a cuRobo model from recorded RoboLab robot configuration and native USD.

All kinematics and meshes come from the existing native robot model/exporter.
Collision cells conservatively cover each native convex hull. The Robotiq uses
an analytic enclosing sphere over native joint ranges because transit receives
no jaw opening.
Generated files belong to the caller's asset directory, never third_party.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull

from .adapter import CuroboPlanner, _gripper_balls, _gripper_radius


def _cover_convex_mesh(vertices, cell_width):
    """Enclose every intersecting grid cell; retain uncertain boundary cells.

    A cell is discarded only when one convex-hull halfspace excludes its entire
    volume. Every retained cell's circumsphere covers the whole cell, including
    mesh interior. This may conservatively retain extra empty cells.
    """
    hull = ConvexHull(vertices)
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    counts = np.maximum(1, np.ceil((upper - lower) / cell_width).astype(int))
    widths = (upper - lower) / counts
    axes = [lower[i] + (np.arange(counts[i]) + .5) * widths[i] for i in range(3)]
    centers = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    normals, offsets = hull.equations[:, :3], hull.equations[:, 3]
    projection_radius = np.abs(normals) @ (widths / 2)
    radius = float(np.linalg.norm(widths / 2)) + 1e-6
    spheres = []
    for chunk in np.array_split(centers, max(1, int(np.ceil(len(centers) / 128)))):
        minimum = chunk @ normals.T + offsets - projection_radius
        for center in chunk[~np.any(minimum > 1e-10, axis=1)]:
            spheres.append(dict(center=center.tolist(), radius=radius))
    return spheres


def export_robot_assets(native_config, output, *, cell_width_m=.04):
    """Export from actual ``native/env_cfg.json``; do not guess robot settings."""
    output = Path(output).expanduser().resolve()
    if any(part in ("third_party", "vendor") for part in output.parts):
        raise ValueError("generated robot assets must be outside immutable third_party/vendor sources")
    if not np.isfinite(cell_width_m) or cell_width_m <= 0:
        raise ValueError("cell_width_m must be positive and finite")
    native_config = Path(native_config).resolve()
    config = json.loads(native_config.read_text())
    robot = config["scene"]["robot"]
    usd = Path(robot["spawn"]["usd_path"]).resolve()
    initial = robot["init_state"]["joint_pos"]
    from src.simulator.robolab.robot_model import RoboLabRobotModel
    from src.simulator.robolab.moveit_model import export_moveit_model, excluded_pair
    native = RoboLabRobotModel(usd)
    output = export_moveit_model(native, output, asset_path=usd)
    # cuRobo's URDF filename handler accepts filesystem paths, whereas the
    # shared native exporter emits file URIs for MoveIt. Preserve its export
    # and produce a planner copy changing only that path representation.
    planner_urdf = ET.parse(output / "robot.urdf")
    for mesh in planner_urdf.getroot().iter("mesh"):
        filename = mesh.get("filename", "")
        if filename.startswith("file://"):
            mesh.set("filename", unquote(urlsplit(filename).path))
    for joint in planner_urdf.getroot().findall("joint"):
        name = joint.get("name", "")
        if re.fullmatch(r"panda_joint[1-7]", name):
            actuators = [entry for entry in robot["actuators"].values()
                         if any(re.fullmatch(pattern, name) for pattern in entry["joint_names_expr"])]
            if len(actuators) != 1:
                raise ValueError(f"native configuration must identify one actuator for {name}")
            limit = joint.find("limit")
            for field, source in (("velocity", "velocity_limit"), ("effort", "effort_limit")):
                value = actuators[0].get(source)
                if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                    raise ValueError(f"native actuator must declare {source} for {name}")
                limit.set(field, str(value))
    planner_urdf.write(output / "planner.urdf", encoding="utf-8", xml_declaration=True)
    arm_names = [f"panda_joint{i}" for i in range(1, 8)]

    def initial_position(name):
        values = [value for pattern, value in initial.items() if re.fullmatch(pattern, name)]
        if len(values) != 1 or not np.isfinite(values[0]):
            raise ValueError(f"native configuration must provide one initial position for {name}")
        return float(values[0])

    parents = {j["child"].rsplit("/", 1)[-1]: j["parent"].rsplit("/", 1)[-1]
               for j in native.joints}
    spheres = {}
    for name, triangles in native.visual_triangles.items():
        body = native.geom_body[name]
        if body.startswith("panda_link"):
            spheres.setdefault(body, []).extend(_cover_convex_mesh(triangles.reshape(-1, 3), cell_width_m))
    balls = _gripper_balls(native)
    centers, radii = np.array([point for point, _ in balls]), np.array([r for _, r in balls])[:, None]
    center = ((centers - radii).min(axis=0) + (centers + radii).max(axis=0)) / 2
    radius = _gripper_radius(native, center)
    spheres["base_link"] = [dict(center=center.tolist(), radius=radius)]
    links = list(spheres)
    ignored = {a: [b for b in links if a != b and excluded_pair(a, b, parents)] for a in links}
    kinematics = dict(format_version=2.0, base_link="panda_link0", tool_frames=["base_link"],
        urdf_path=str(output / "planner.urdf"), asset_root_path=str(output),
        collision_link_names=links, collision_spheres=spheres, collision_sphere_buffer=0.,
        self_collision_ignore=ignored, self_collision_buffer={link: 0. for link in links},
        # Finger joints are downstream of base_link and outside the planner's
        # arm/tool chain. Their geometry is covered over every native angle by the
        # envelope, so no finger joint value is substituted or locked here.
        cspace=dict(joint_names=arm_names,
            default_joint_position=[initial_position(name) for name in arm_names],
            null_space_weight=[1.] * 7, cspace_distance_weight=[1.] * 7))
    import yaml
    (output / "robot.yml").write_text(yaml.safe_dump(dict(robot_cfg=dict(kinematics=kinematics)), sort_keys=False))
    profile = dict(robot_profile="franka_robotiq_2f85", coordinate_frame="connector_base",
        connector_joint_names=arm_names, planner_joint_names=arm_names,
        base_link="panda_link0", tool_link="base_link", flange_from_planner_tool=np.eye(4).tolist(),
        gripper_collision_envelope=dict(link="base_link", center_m=center.tolist(),
                                        radius_m=radius, opening_range_m=[0., .085]))
    (output / "calibration.json").write_text(json.dumps(profile, indent=2) + "\n")
    planner = CuroboPlanner(output / "robot.yml", output / "calibration.json")
    evidence = dict(native_configuration=str(native_config),
        native_configuration_sha256=hashlib.sha256(native_config.read_bytes()).hexdigest(),
        native_usd=str(usd), native_usd_sha256=hashlib.sha256(usd.read_bytes()).hexdigest(),
        cell_width_m=cell_width_m, collision_sphere_count=sum(map(len, spheres.values())),
        collision_coverage="circumspheres of every grid cell intersecting native per-shape convex hulls; analytic Robotiq envelope over native joint and mimic ranges",
        self_collision_policy="existing native graph distance <=2 and internal gripper mechanism exclusions",
        kinematics="existing native USD two-frame URDF export; tool is same native base_link; identity calibration",
        dynamic_limits="arm velocity/effort limits from recorded native actuators; pinned cuRobo acceleration/jerk defaults",
        planner_preflight=planner.preflight(), gpu_validated=False)
    (output / "curobo-assets.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cell-width-m", type=float, default=.04)
    args = parser.parse_args(argv)
    # Isaac must initialize its USD modules before the native model imports pxr.
    from src.simulator.robolab import cli
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "fast_shutdown": False})
    try:
        print(export_robot_assets(args.native_config, args.output_dir, cell_width_m=args.cell_width_m))
    finally:
        # Standalone CLI owns plugin shutdown; embedded callers close the app.
        if not cli.OWNS_PROCESS:
            app.close()


if __name__ == "__main__":
    from src.simulator.robolab.cli import run_native_cli
    run_native_cli(main)
