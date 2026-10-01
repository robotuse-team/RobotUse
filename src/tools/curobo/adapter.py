"""Optional observed-world transit planner for an explicitly calibrated RoboLab model.

This adapter supplies no robot model, calibration, or Panda fallback. Discovery
does not import this module; CUDA imports occur only on the first planning call.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from threading import RLock
import xml.etree.ElementTree as ET
from urllib.parse import unquote, urlsplit

import numpy as np


CUROBO_REVISION = "4ea77366ca48ee453e7df139e39fa6532af49f3b"
_NATIVE_JOINTS = tuple(f"panda_joint{i}" for i in range(1, 8))
_NATIVE_MAX_OPENING_M = .085
_LOCK = RLock()
_IMPLEMENTATION = None


class CuroboConfigurationError(ValueError):
    """Explicit model or calibration does not match the native robot contract."""


def _transform(value, name):
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise CuroboConfigurationError(f"{name} must be a finite rigid 4x4 transform") from exc
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-6, rtol=0)):
        raise CuroboConfigurationError(f"{name} must be a finite rigid 4x4 transform")
    return matrix.copy()


def _file(value, name, *, absolute=False):
    if not isinstance(value, (str, Path)) or not str(value):
        raise CuroboConfigurationError(f"{name} requires an explicit path")
    path = Path(value).expanduser()
    if absolute and not path.is_absolute():
        raise CuroboConfigurationError(f"{name} must be absolute; cuRobo resolves relative assets against its own content")
    if not path.is_file():
        raise CuroboConfigurationError(f"{name} is missing: {path}")
    return path.resolve()


def _names(value, name):
    if (not isinstance(value, list) or not value
            or any(not isinstance(item, str) or not item for item in value)
            or len(set(value)) != len(value)):
        raise CuroboConfigurationError(f"{name} must contain unique nonempty joint/link names")
    return tuple(value)


def _yaml(path):
    try:
        import yaml
    except ImportError as exc:
        raise CuroboConfigurationError("cuRobo configuration preflight requires PyYAML") from exc
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise CuroboConfigurationError(f"cannot read cuRobo YAML: {path}") from exc
    if not isinstance(data, dict):
        raise CuroboConfigurationError(f"cuRobo YAML must be a mapping: {path}")
    return data


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gripper_balls(model):
    """Enclose native gripper geometry over every permitted joint angle.

    Propagate an enclosing ball through the exact native joint frames. At each
    revolute joint, enclose the centre's continuous arc over its native limits
    (or the whole circle if limits are absent). Mimic ranges follow their source
    joint. Independent arc bounds conservatively cover coupled finger motion.
    """
    tree = model.body_paths
    paths = {path.rsplit("/", 1)[-1]: path for path in tree}
    if len(paths) != len(tree) or "base_link" not in paths:
        raise ValueError("native gripper body identities are missing or ambiguous")
    from scipy.spatial.transform import Rotation
    joints = {joint.get("name"): joint for joint in tree.values() if joint is not None}

    def limits(joint, visited=()):
        if joint.get("mimic") is not None:
            source, multiplier = joint["mimic"]
            if source in visited or source not in joints:
                raise ValueError("native gripper mimic joint graph is invalid")
            return sorted(float(multiplier) * value for value in limits(joints[source], (*visited, source)))
        values = np.asarray(joint.get("limits", [-np.pi, np.pi]), dtype=float)
        if values.shape != (2,) or not np.isfinite(values).all() or values[0] > values[1]:
            raise ValueError("native gripper joint limits are invalid")
        return values

    flange, balls, covered = paths["base_link"], [], set()
    for name, triangles in model.visual_triangles.items():
        body = model.geom_body[name]
        points = np.asarray(triangles, dtype=float)
        if points.ndim != 3 or points.shape[1:] != (3, 3) or not len(points) or not np.isfinite(points).all():
            raise ValueError("native gripper triangles are incomplete")
        vertices = points.reshape(-1, 3)
        center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
        bound = float(np.linalg.norm(vertices - center, axis=1).max())
        path, visited = paths[body], set()
        while path != flange:
            if path in visited:
                raise ValueError("native gripper joint graph contains a cycle")
            visited.add(path)
            joint = tree[path]
            if joint is None:
                break
            child_from_joint = _transform(joint["local1"], "local1")
            center = child_from_joint[:3, :3].T @ (center - child_from_joint[:3, 3])
            if joint["revolute"]:
                axis = np.eye(3)[:, joint["axis"]]
                axial_center = axis * np.dot(axis, center)
                radial = center - axial_center
                lo, hi = limits(joint)
                half = (hi - lo) / 2
                if half <= np.pi / 2:
                    bound += float(np.linalg.norm(radial) * np.sin(half))
                    center = axial_center + Rotation.from_rotvec(axis * (lo + hi) / 2).apply(radial) * np.cos(half)
                else:
                    bound += float(np.linalg.norm(radial))
                    center = axial_center
            parent_from_joint = _transform(joint["local0"], "local0")
            center = parent_from_joint[:3, :3] @ center + parent_from_joint[:3, 3]
            path = joint["parent"]
        if path != flange:
            continue
        balls.append((center, bound))
        covered.add(body)
    if not {"base_link", "left_inner_finger", "right_inner_finger"}.issubset(covered):
        raise ValueError("native Robotiq base and finger geometry is incomplete")
    return balls


def _gripper_radius(model, center=None):
    """Radius enclosing permitted native gripper motion about a flange-local centre."""
    center = np.zeros(3) if center is None else np.asarray(center, dtype=float)
    return max(float(np.linalg.norm(point - center)) + radius
               for point, radius in _gripper_balls(model)) + 1e-6


def _native_gripper_radius(env, center=None):
    """Load the same robot-only native geometry used by RoboLab collision checks."""
    try:
        from src.simulator.robolab.robot_model import RoboLabRobotModel
        return _gripper_radius(RoboLabRobotModel(env.robot.cfg.spawn.usd_path), center)
    except Exception as exc:
        raise CuroboConfigurationError(
            "unsupported native gripper geometry: cannot verify all-opening collision envelope") from exc


def _runtime_source():
    """Check package provenance without importing torch or initializing CUDA."""
    spec = importlib.util.find_spec("curobo")
    if spec is None or spec.origin is None:
        raise CuroboConfigurationError(
            "optional cuRobo runtime is not installed; install the pinned checkout " + CUROBO_REVISION)
    checkout = Path(__file__).resolve().parent / "third_party" / "curobo"
    origin = Path(spec.origin).resolve()
    if origin.is_relative_to(checkout.resolve()):
        revision = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    else:
        try:
            direct = importlib.metadata.distribution("nvidia-curobo").read_text("direct_url.json")
            revision = json.loads(direct or "{}").get("vcs_info", {}).get("commit_id")
        except (importlib.metadata.PackageNotFoundError, ValueError):
            revision = None
    if revision != CUROBO_REVISION:
        raise CuroboConfigurationError(
            f"cuRobo import at {origin} is not verified at required revision {CUROBO_REVISION}")
    missing = []
    for dependency in ("torch", "scipy", "trimesh", "warp", "yourdfpy", "cuda.core"):
        try:
            available = importlib.util.find_spec(dependency) is not None
        except (ImportError, ValueError):
            available = False
        if not available:
            missing.append(dependency)
    if missing:
        raise CuroboConfigurationError("optional cuRobo runtime dependencies are missing: " + ", ".join(missing))
    return dict(curobo_import=str(origin), revision=revision,
                dependencies_available=True, gpu_initialized=False)


def _implementation_source():
    """Resolve the unchanged tool-local source without importing CUDA."""
    source = Path(__file__).resolve().parent / "third_party/open_robot_skills/tools/curobo/_curobo_impl.py"
    source = _file(source, "tool-local original cuRobo implementation")
    manifest = json.loads((Path(__file__).resolve().parent / "implementation_source.json").read_text())
    expected = next(record["sha256"] for record in manifest["files"]
                    if record["path"] == "tools/curobo/_curobo_impl.py")
    if _hash(source) != expected:
        raise CuroboConfigurationError("original cuRobo implementation source was modified")
    return source


def _load_implementation():
    """Load unchanged tool-local code only with the pinned cuRobo package."""
    global _IMPLEMENTATION
    if _IMPLEMENTATION is not None:
        return _IMPLEMENTATION
    _runtime_source()
    source = _implementation_source()
    module_name = "_optional_curobo_impl"
    module_spec = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    try:
        module_spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    _IMPLEMENTATION = module
    return module


class CuroboPlanner:
    """Native-order joint trajectory from a public Robotiq flange target.

    Calibration JSON requires robot_profile=franka_robotiq_2f85,
    coordinate_frame=connector_base, connector_joint_names, planner_joint_names,
    base_link, tool_link and flange_from_planner_tool. The final transform is
    ``base_from_public_flange @ flange_from_planner_tool``. Translation is in
    metres; rotations are rigid matrices. No world-frame conversion is hidden.

    ``gripper_collision_envelope`` requires ``link: base_link``, a positive
    ``radius_m``, optional flange-local ``center_m`` (default zero), and ``opening_range_m`` covering
    at least [0, 0.085]. One configured base_link collision sphere (after its
    collision_sphere_buffer) must contain that complete envelope. Finger joint
    locks alone are insufficient: this callback receives no requested opening.
    Before planning, the radius is checked against the native geometry loader's
    conservative bound over native joint limits and mimic ranges, covering all allowed jaw
    openings. Missing/unsupported geometry is rejected. This deliberately
    conservative sphere can reject routes a per-opening model could permit.

    ``implementation`` is an injectable object with ``_get_pose_planner`` and
    ``plan_to_pose`` methods, for CPU contract tests. A fake is not native
    planning verification. Without injection, the pinned GPU runtime is lazy.
    """

    def __init__(self, robot_file, calibration_file, implementation=None):
        self.robot_file = _file(robot_file, "cuRobo robot_file")
        self.calibration_file = _file(calibration_file, "cuRobo calibration_file")
        try:
            profile = json.loads(self.calibration_file.read_text())
        except (OSError, ValueError) as exc:
            raise CuroboConfigurationError("cuRobo calibration_file must contain JSON") from exc
        if not isinstance(profile, dict):
            raise CuroboConfigurationError("cuRobo calibration profile must be a JSON object")
        if profile.get("robot_profile") != "franka_robotiq_2f85":
            raise CuroboConfigurationError("cuRobo requires an explicit franka_robotiq_2f85 calibration profile")
        if profile.get("coordinate_frame") != "connector_base":
            raise CuroboConfigurationError("cuRobo robot base and observed world must use connector_base")
        self.connector_joint_names = _names(profile.get("connector_joint_names"), "connector_joint_names")
        self.planner_joint_names = _names(profile.get("planner_joint_names"), "planner_joint_names")
        if (self.connector_joint_names != _NATIVE_JOINTS
                or set(self.planner_joint_names) != set(self.connector_joint_names)):
            raise CuroboConfigurationError("cuRobo joint mapping must cover exactly the native seven arm joints")
        self.flange_from_planner_tool = _transform(
            profile.get("flange_from_planner_tool"), "flange_from_planner_tool")
        self.flange_from_planner_tool.setflags(write=False)
        self.base_link, self.tool_link = profile.get("base_link"), profile.get("tool_link")
        if not all(isinstance(name, str) and name for name in (self.base_link, self.tool_link)):
            raise CuroboConfigurationError("base_link and tool_link must be explicit link names")
        envelope = profile.get("gripper_collision_envelope")
        if not isinstance(envelope, dict) or envelope.get("link") != "base_link":
            raise CuroboConfigurationError("gripper_collision_envelope must explicitly use native base_link")
        opening_range = envelope.get("opening_range_m")
        radius = envelope.get("radius_m")
        if (not isinstance(opening_range, list) or len(opening_range) != 2
                or any(type(value) not in (int, float) or not np.isfinite(value) for value in opening_range)
                or opening_range[0] != 0 or opening_range[1] < _NATIVE_MAX_OPENING_M
                or type(radius) not in (int, float) or not np.isfinite(radius) or radius <= 0):
            raise CuroboConfigurationError(
                "gripper_collision_envelope requires positive radius_m and opening_range_m covering [0, 0.085]")
        self._gripper_envelope_radius = float(radius)
        try:
            center = np.asarray(envelope.get("center_m", [0., 0., 0.]), dtype=float)
        except (TypeError, ValueError) as exc:
            raise CuroboConfigurationError("gripper center_m must be three finite coordinates") from exc
        if center.shape != (3,) or not np.isfinite(center).all():
            raise CuroboConfigurationError("gripper center_m must be three finite coordinates")
        self._gripper_envelope_center = center.copy()
        self._gripper_envelope_center.setflags(write=False)
        self._gripper_max_opening = float(opening_range[1])
        self._native_gripper_env = None
        self._implementation = implementation
        self._worker = None
        self._assets = self._validate_robot_config()
        self._fingerprints = {path: _hash(path) for path in
                              (self.robot_file, self.calibration_file, *self._assets)}
        self._to_planner = [self.connector_joint_names.index(name) for name in self.planner_joint_names]
        self._to_connector = [self.planner_joint_names.index(name) for name in self.connector_joint_names]
        from src.tools import discover_tools
        self._registry = discover_tools()
        self._registry.require(("curobo_plan",))

    def _validate_robot_config(self):
        config = _yaml(self.robot_file)
        robot = config.get("robot_cfg")
        kin = robot.get("kinematics") if isinstance(robot, dict) else None
        if not isinstance(kin, dict) or not kin:
            raise CuroboConfigurationError("robot_file must declare robot_cfg.kinematics")
        if kin.get("base_link") != self.base_link:
            raise CuroboConfigurationError("robot base_link does not match calibration")
        if kin.get("tool_frames") != [self.tool_link]:
            raise CuroboConfigurationError("robot tool_frames must contain only the calibrated tool_link")
        cspace_config = kin.get("cspace")
        cspace = _names(cspace_config.get("joint_names") if isinstance(cspace_config, dict) else None,
                        "robot cspace.joint_names")
        locked = kin.get("lock_joints") or {}
        if (not isinstance(locked, dict) or any(name in self.connector_joint_names for name in locked)
                or any(type(value) not in (float, int) or not np.isfinite(value) for value in locked.values())):
            raise CuroboConfigurationError("lock_joints must be finite and must not lock an arm joint")
        if tuple(name for name in cspace if name not in locked) != self.planner_joint_names:
            raise CuroboConfigurationError("effective robot cspace order does not match planner_joint_names")
        urdf = _file(kin.get("urdf_path"), "robot urdf_path", absolute=True)
        asset_root = kin.get("asset_root_path")
        if not isinstance(asset_root, str) or not Path(asset_root).is_absolute() or not Path(asset_root).is_dir():
            raise CuroboConfigurationError("robot asset_root_path must be an existing absolute directory")
        asset_root = Path(asset_root)
        try:
            root = ET.parse(urdf).getroot()
        except ET.ParseError as exc:
            raise CuroboConfigurationError("robot URDF is invalid") from exc
        links = {item.get("name") for item in root.findall("link")}
        joints = {item.get("name"): item for item in root.findall("joint")}
        if ("base_link" not in links or "finger_joint" not in joints
                or self.base_link not in links or self.tool_link not in links
                or any(name not in joints for name in self.planner_joint_names)):
            raise CuroboConfigurationError("robot URDF lacks the calibrated arm/Robotiq links and joints; Panda fallback is forbidden")
        if any(joints[name].get("type") != "revolute" for name in self.planner_joint_names):
            raise CuroboConfigurationError("native RoboLab arm mapping requires seven revolute joints")
        assets = [urdf]
        for mesh in root.iter("mesh"):
            filename = mesh.get("filename", "")
            if filename.startswith("file://"):
                location = urlsplit(filename)
                if location.netloc not in ("", "localhost"):
                    raise CuroboConfigurationError("robot mesh file URI must name a local file")
                mesh_path = Path(unquote(location.path))
            else:
                mesh_path = asset_root / filename.removeprefix("package://")
            assets.append(_file(mesh_path, "robot mesh"))
        spheres = kin.get("collision_spheres")
        if isinstance(spheres, str):
            path = _file(spheres, "robot collision_spheres", absolute=True)
            assets.append(path)
            spheres = _yaml(path).get("collision_spheres")
        collision_links = _names(kin.get("collision_link_names"), "collision_link_names")
        if (kin.get("load_collision_spheres") is False or not isinstance(spheres, dict)
                or "base_link" not in collision_links):
            raise CuroboConfigurationError("explicit arm/Robotiq collision spheres are required")
        for link in collision_links:
            values = spheres.get(link)
            if link not in links or not isinstance(values, list) or not values:
                raise CuroboConfigurationError(f"missing robot collision geometry for {link}")
            for sphere in values:
                center = np.asarray(sphere.get("center") if isinstance(sphere, dict) else None, dtype=float)
                radius = sphere.get("radius") if isinstance(sphere, dict) else None
                if (center.shape != (3,) or not np.isfinite(center).all()
                        or type(radius) not in (int, float) or not np.isfinite(radius) or radius <= 0):
                    raise CuroboConfigurationError(f"invalid collision sphere for {link}")
        buffer = kin.get("collision_sphere_buffer", 0.)
        if isinstance(buffer, dict):
            buffer = buffer.get("base_link", 0.)
        if type(buffer) not in (int, float) or not np.isfinite(buffer):
            raise CuroboConfigurationError("base_link collision_sphere_buffer must be finite")
        if "base_link" in (kin.get("extra_collision_spheres") or {}):
            raise CuroboConfigurationError("extra_collision_spheres must not replace the gripper envelope")
        if not any(sphere["radius"] + buffer - np.linalg.norm(
                np.asarray(sphere["center"]) - self._gripper_envelope_center) >= self._gripper_envelope_radius
                   for sphere in spheres["base_link"]):
            raise CuroboConfigurationError("base_link collision spheres do not contain gripper_collision_envelope")
        return tuple(dict.fromkeys(assets))

    def preflight(self):
        """Report configuration evidence only; this does not initialize a GPU."""
        for path, digest in self._fingerprints.items():
            if not path.is_file() or _hash(path) != digest:
                raise CuroboConfigurationError(f"cuRobo configuration asset changed after validation: {path}")
        return dict(planner="curobo", revision=CUROBO_REVISION,
                    robot_file=str(self.robot_file), calibration_file=str(self.calibration_file),
                    robot_profile="franka_robotiq_2f85", coordinate_frame="connector_base",
                    planner_joint_names=list(self.planner_joint_names), tool_link=self.tool_link,
                    gripper_collision_envelope=dict(link="base_link", radius_m=self._gripper_envelope_radius,
                        center_m=self._gripper_envelope_center.tolist(),
                        opening_range_m=[0., self._gripper_max_opening], native_geometry_checked=False),
                    configuration_validated=True, runtime_checked=False,
                    source_hashes={str(path): digest for path, digest in self._fingerprints.items()})

    def validate_runtime(self):
        """Check installed dependency paths/pin before service/simulator startup.

        GPU availability and loaded planner joint/frame checks require actual
        planner initialization and are enforced before returning any trajectory.
        """
        self.preflight()
        try:
            worker = self._get_worker()
            evidence = worker.call({'operation': 'probe'})
        except Exception as exc:
            raise CuroboConfigurationError(
                'Configure ROBOTUSE_CUROBO_PYTHON and ROBOTUSE_CUROBO_PYTHONPATH for the pinned '
                'cuRobo runtime; isolated preflight failed: ' + str(exc)) from exc
        return {**evidence, 'planner_initialized': False, 'process_isolated': True,
                'worker_pid': worker.process.pid, 'worker_log': str(worker.log_path)}

    def _get_worker(self):
        if self._worker is not None and self._worker.process.poll() is not None:
            self._worker.close()
            self._worker = None
        if self._worker is None:
            from .process import CuroboWorker
            self._worker = CuroboWorker()
        return self._worker

    def close(self):
        if self._worker is not None:
            self._worker.close()
            self._worker = None

    def __call__(self, connector, target, start_joints, world):
        from src.tools import ToolExecutionContext
        return self._registry.dispatch("curobo_plan",
            dict(target=target, start_joints=start_joints, world=world),
            context=ToolExecutionContext(dispatch=lambda _name, arguments: self._plan(connector, **arguments)))

    def _plan(self, connector, target, start_joints, world):
        from scipy.spatial.transform import Rotation
        from src.tools.motion.planning import MotionPlanningError

        self.preflight()
        target = _transform(target, "native flange target")
        start = np.asarray(start_joints, dtype=float)
        if start.shape != (7,) or not np.isfinite(start).all():
            raise CuroboConfigurationError("start_joints must contain exactly seven finite native-order values")
        try:
            env = connector.env
            actual_names = tuple(env.robot.joint_names[int(index)] for index in env._arm_ids)
            actual_flange = env.robot.body_names[int(env._eef_id)]
        except (AttributeError, TypeError, IndexError) as exc:
            raise CuroboConfigurationError("connector must expose native RoboLab arm and flange identities") from exc
        if actual_names != self.connector_joint_names or actual_flange != "base_link":
            raise CuroboConfigurationError("live RoboLab joint/flange mapping does not match calibration")
        if not callable(getattr(getattr(connector.ik, "model", None), "fk", None)):
            raise CuroboConfigurationError("native RoboLab FK is required to check the planned endpoint")
        meshes = getattr(world, "mesh", None)
        if not isinstance(meshes, (list, tuple)):
            raise CuroboConfigurationError("observed world must retain its explicit .mesh collection")
        if len(getattr(world, "observed_points", ())) and not meshes:
            raise CuroboConfigurationError("observed points cannot be silently planned without world meshes")
        for mesh in meshes:
            if getattr(mesh, "vertices", None) is None or getattr(mesh, "faces", None) is None:
                raise CuroboConfigurationError("observed world contains an incomplete mesh")
        maximum = getattr(env, "max_gripper_width_m", None)
        if (type(maximum) not in (float, int) or not np.isfinite(maximum) or maximum <= 0
                or maximum > self._gripper_max_opening):
            raise CuroboConfigurationError("gripper collision envelope does not cover the live native opening range")
        if self._native_gripper_env is not env:
            native_radius = _native_gripper_radius(env, self._gripper_envelope_center)
            if not np.isfinite(native_radius) or native_radius <= 0 or native_radius > self._gripper_envelope_radius:
                raise CuroboConfigurationError("gripper collision envelope does not contain all-opening native geometry")
            self._native_gripper_env = env
        planner_target = target @ self.flange_from_planner_tool
        options = dict(robot_file=str(self.robot_file), position_threshold=.005,
                       rotation_threshold=.05, num_ik_seeds=32, use_cuda_graph=False)
        try:
            with _LOCK:
                position = planner_target[:3, 3]
                quaternion = Rotation.from_matrix(planner_target[:3, :3]).as_quat()[[3, 0, 1, 2]]
                if self._implementation is not None:
                    # Injection is reserved for CPU contract tests. Production
                    # imports CUDA/Warp only in the dedicated worker process.
                    impl = self._implementation
                    planner = impl._get_pose_planner(**options, with_collision=bool(meshes),
                                                    mesh_cache=max(len(meshes) + 4, 32))
                    if tuple(planner.joint_names) != self.planner_joint_names:
                        raise CuroboConfigurationError("loaded cuRobo planner joint order differs from calibration")
                    if tuple(planner.tool_frames) != (self.tool_link,):
                        raise CuroboConfigurationError("loaded cuRobo tool frame differs from calibration")
                    success, trajectory = impl.plan_to_pose(position, quaternion, start[self._to_planner],
                        **options, tcp_offset=None, world_config=world)
                else:
                    response = self._get_worker().call(dict(operation='plan', options=options,
                        expected_joint_names=list(self.planner_joint_names), expected_tool_frames=[self.tool_link],
                        position=position.tolist(), quaternion_wxyz=quaternion.tolist(),
                        start_joints=start[self._to_planner].tolist(), world=dict(mesh=[dict(
                            name=getattr(mesh, 'name', None),
                            pose=None if getattr(mesh, 'pose', None) is None else np.asarray(mesh.pose).tolist(),
                            vertices=np.asarray(mesh.vertices).tolist(), faces=np.asarray(mesh.faces).tolist(),
                        ) for mesh in meshes])))
                    if (tuple(response['joint_names']) != self.planner_joint_names
                            or tuple(response['tool_frames']) != (self.tool_link,)):
                        raise CuroboConfigurationError('worker planner joint/tool mapping differs from calibration')
                    success, trajectory = response['success'], response['trajectory']
        except CuroboConfigurationError:
            raise
        except Exception as exc:
            raise MotionPlanningError("optional cuRobo transit planning failed: " + str(exc),
                planning_feedback={"kind": "planning", "planner_reason_code": "planner_exception"}) from exc
        if not success or trajectory is None:
            raise MotionPlanningError("cuRobo found no observed-world transit route",
                planning_feedback={"kind": "planning", "planner_reason_code": "world_route_failed"})
        rows = np.asarray(trajectory, dtype=float)
        if rows.ndim != 2 or rows.shape[1] != 7 or len(rows) < 2 or not np.isfinite(rows).all():
            raise MotionPlanningError("cuRobo returned an invalid seven-joint trajectory",
                planning_feedback={"kind": "planning", "planner_reason_code": "invalid_trajectory"})
        rows = rows[:, self._to_connector]
        endpoint = _transform(connector.ik.model.fk(rows[-1]), "native FK endpoint")
        if (np.linalg.norm(endpoint[:3, 3] - target[:3, 3]) > .005
                or Rotation.from_matrix(endpoint[:3, :3] @ target[:3, :3].T).magnitude() > .05):
            raise MotionPlanningError("cuRobo endpoint does not match native flange calibration",
                planning_feedback={"kind": "planning", "planner_reason_code": "invalid_trajectory"})
        return {"waypoints": [{"positions": row.tolist()} for row in rows], "planner": "curobo"}
