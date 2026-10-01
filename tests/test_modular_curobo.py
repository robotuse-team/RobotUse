"""CPU fakes verify the optional adapter contract, not native cuRobo planning."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from src.tools.curobo.adapter import (
    CuroboConfigurationError, CuroboPlanner, _gripper_radius, _native_gripper_radius,
)


JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
PLANNER_JOINTS = list(reversed(JOINTS))


def test_original_implementation_is_tool_local_and_verified_without_cuda():
    from src.tools.curobo.adapter import _implementation_source
    source = _implementation_source()
    tool_root = Path(__file__).resolve().parents[1] / "src/tools/curobo"
    assert source.is_relative_to(tool_root / "third_party")


def test_native_export_file_uri_mesh_is_validated(configuration):
    mesh = configuration.robot.parent / "native mesh.stl"
    mesh.write_bytes(b"asset bytes are hashed without loading CUDA")
    text = configuration.urdf.read_text().replace("</robot>",
        '<link name="mesh_link"><visual><geometry><mesh filename="' + mesh.as_uri() +
        '"/></geometry></visual></link></robot>')
    configuration.urdf.write_text(text)
    planner = CuroboPlanner(configuration.robot, configuration.profile)
    assert str(mesh) in planner.preflight()["source_hashes"]


@pytest.fixture
def configuration(tmp_path, monkeypatch):
    # Deliberately tiny identity fixture: never supplied as a native robot model.
    # A separate geometry-bound test exercises the native model contract below.
    monkeypatch.setattr("src.tools.curobo.adapter._native_gripper_radius", lambda _env, *_: .02)
    urdf = tmp_path / "cpu-contract-fixture.urdf"
    links = ["panda_link0", "base_link", "test_tool", "finger"]
    urdf.write_text('<robot name="cpu_fixture">' +
        ''.join(f'<link name="{name}"/>' for name in links) +
        ''.join(f'<joint name="{name}" type="revolute"/>' for name in JOINTS) +
        '<joint name="finger_joint" type="revolute"/></robot>')
    robot = tmp_path / "robot.yml"
    profile = tmp_path / "calibration.json"
    kin = dict(base_link="panda_link0", tool_frames=["test_tool"],
               cspace=dict(joint_names=[*PLANNER_JOINTS, "finger_joint"]),
               lock_joints={"finger_joint": .2}, urdf_path=str(urdf),
               asset_root_path=str(tmp_path), collision_link_names=["panda_link0", "base_link"],
               collision_spheres={name: [dict(center=[0., 0., 0.], radius=.03)]
                                  for name in ("panda_link0", "base_link")})
    calibration = dict(robot_profile="franka_robotiq_2f85", coordinate_frame="connector_base",
                       connector_joint_names=JOINTS, planner_joint_names=PLANNER_JOINTS,
                       base_link="panda_link0", tool_link="test_tool",
                       gripper_collision_envelope=dict(link="base_link", radius_m=.025,
                                                       opening_range_m=[0., .085]),
                       flange_from_planner_tool=np.eye(4).tolist())

    def write():
        robot.write_text(yaml.safe_dump(dict(robot_cfg=dict(kinematics=kin))))
        profile.write_text(json.dumps(calibration))

    write()
    return SimpleNamespace(robot=robot, profile=profile, kin=kin, calibration=calibration,
                           urdf=urdf, write=write)


class FakeImplementation:
    def __init__(self, native_rows=None):
        self.calls = []
        self.planner = SimpleNamespace(joint_names=PLANNER_JOINTS, tool_frames=["test_tool"])
        rows = np.array([np.arange(7), np.arange(7) + .1]) if native_rows is None else native_rows
        self.result = True, rows[:, ::-1]

    def _get_pose_planner(self, **kwargs):
        self.calls.append(("create", kwargs))
        return self.planner

    def plan_to_pose(self, position, quaternion, joints, **kwargs):
        self.calls.append(("plan", position, quaternion, joints, kwargs))
        return self.result


def _connector(target=None):
    target = np.eye(4) if target is None else target
    return SimpleNamespace(
        env=SimpleNamespace(robot=SimpleNamespace(joint_names=JOINTS, body_names=["base_link"]),
                            _arm_ids=list(range(7)), _eef_id=0, max_gripper_width_m=.085),
        ik=SimpleNamespace(model=SimpleNamespace(fk=lambda _joints: target)))


def _world():
    mesh = SimpleNamespace(name="observed", pose=None,
                           vertices=[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], faces=[[0, 1, 2]])
    return SimpleNamespace(mesh=[mesh], observed_points=np.array([[0., 0., 0.]]))


def test_configuration_preflight_never_initializes_planner(configuration, monkeypatch):
    def forbidden():
        pytest.fail("CPU preflight must not load CUDA implementation")

    monkeypatch.setattr("src.tools.curobo.adapter._load_implementation", forbidden)
    planner = CuroboPlanner(configuration.robot, configuration.profile)
    result = planner.preflight()
    assert result["configuration_validated"] is True
    assert result["runtime_checked"] is False
    assert result["gripper_collision_envelope"]["native_geometry_checked"] is False
    assert str(configuration.urdf) in result["source_hashes"]


def test_optional_backend_registration_remains_internal():
    from src.runtime.bootstrap import load_tool_registry
    spec = load_tool_registry()["curobo_plan"]
    assert spec.kind == "backend" and spec.visibility == "internal"
    assert spec.required_arguments == ("target", "start_joints", "world")


def test_calibrated_target_joint_reordering_and_world_are_preserved(configuration):
    from scipy.spatial.transform import Rotation

    offset = np.eye(4)
    offset[:3, :3] = Rotation.from_euler("x", 30, degrees=True).as_matrix()
    offset[:3, 3] = [.1, .2, .3]
    configuration.calibration["flange_from_planner_tool"] = offset.tolist()
    configuration.write()
    target = np.eye(4)
    target[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    target[:3, 3] = [.3, -.2, .5]
    impl, world = FakeImplementation(), _world()
    planner = CuroboPlanner(configuration.robot, configuration.profile, implementation=impl)
    result = planner(_connector(target), target, np.arange(7), world)
    _, position, quaternion, start, options = impl.calls[1]
    expected = target @ offset
    np.testing.assert_allclose(position, expected[:3, 3])
    np.testing.assert_allclose(quaternion, Rotation.from_matrix(expected[:3, :3]).as_quat()[[3, 0, 1, 2]])
    np.testing.assert_allclose(start, np.arange(7)[::-1])
    assert options["world_config"] is world
    assert options["tcp_offset"] is None and options["use_cuda_graph"] is False
    assert options["robot_file"] == str(configuration.robot)
    assert impl.calls[0][1]["with_collision"] is True
    assert result["waypoints"][-1]["positions"] == (np.arange(7) + .1).tolist()
    assert result["planner"] == "curobo"


@pytest.mark.parametrize("field,value,match", [
    ("robot_profile", "franka_panda", "franka_robotiq"),
    ("coordinate_frame", "world", "connector_base"),
    ("connector_joint_names", list(reversed(JOINTS)), "native seven"),
    ("planner_joint_names", JOINTS[:-1], "native seven"),
    ("planner_joint_names", [JOINTS[0]] * 7, "unique"),
    ("tool_link", "wrong", "tool_frames"),
    ("base_link", "wrong", "base_link"),
    ("flange_from_planner_tool", [[1., 0.], [0., 1.]], "rigid 4x4"),
    ("flange_from_planner_tool", [[float("nan")] * 4] * 4, "rigid 4x4"),
    ("flange_from_planner_tool", "bad", "rigid 4x4"),
])
def test_invalid_calibration_is_rejected_before_runtime(configuration, field, value, match):
    configuration.calibration[field] = value
    configuration.write()
    with pytest.raises(CuroboConfigurationError, match=match):
        CuroboPlanner(configuration.robot, configuration.profile)


@pytest.mark.parametrize("field,value,match", [
    ("urdf_path", "/missing/robot.urdf", "missing"),
    ("urdf_path", "implicit_panda.urdf", "must be absolute"),
    ("asset_root_path", "/missing/assets", "existing absolute"),
    ("tool_frames", [], "tool_frames"),
    ("cspace", {"joint_names": JOINTS}, "cspace order"),
    ("lock_joints", {"panda_joint1": 0.}, "must not lock an arm"),
    ("collision_link_names", ["panda_link0"], "collision spheres"),
    ("collision_spheres", None, "collision spheres"),
    ("load_collision_spheres", False, "collision spheres"),
])
def test_missing_or_incompatible_robot_assets_fail_preflight(configuration, field, value, match):
    configuration.kin[field] = value
    configuration.write()
    with pytest.raises(CuroboConfigurationError, match=match):
        CuroboPlanner(configuration.robot, configuration.profile)


def test_panda_only_urdf_is_rejected(configuration):
    configuration.urdf.write_text(configuration.urdf.read_text().replace('name="finger_joint"', 'name="panda_finger_joint1"'))
    with pytest.raises(CuroboConfigurationError, match="Panda fallback is forbidden"):
        CuroboPlanner(configuration.robot, configuration.profile)


def test_missing_urdf_mesh_is_rejected(configuration):
    configuration.urdf.write_text(configuration.urdf.read_text().replace('</robot>',
        '<link name="visual"><visual><geometry><mesh filename="missing.obj"/></geometry></visual></link></robot>'))
    with pytest.raises(CuroboConfigurationError, match="robot mesh is missing"):
        CuroboPlanner(configuration.robot, configuration.profile)


def test_configuration_change_after_validation_is_rejected(configuration):
    planner = CuroboPlanner(configuration.robot, configuration.profile, implementation=FakeImplementation())
    configuration.urdf.write_text(configuration.urdf.read_text() + "\n")
    with pytest.raises(CuroboConfigurationError, match="asset changed"):
        planner.preflight()


@pytest.mark.parametrize("field,value", [("joint_names", JOINTS), ("tool_frames", [])])
def test_actual_gpu_planner_mapping_checked_before_planning(configuration, field, value):
    impl = FakeImplementation()
    setattr(impl.planner, field, value)
    planner = CuroboPlanner(configuration.robot, configuration.profile, implementation=impl)
    with pytest.raises(CuroboConfigurationError, match="loaded cuRobo"):
        planner(_connector(), np.eye(4), np.arange(7), _world())
    assert len(impl.calls) == 1


def test_actual_connector_mapping_checked_before_gpu_initialization(configuration):
    impl, connector = FakeImplementation(), _connector()
    connector.env._arm_ids = list(reversed(range(7)))
    planner = CuroboPlanner(configuration.robot, configuration.profile, implementation=impl)
    with pytest.raises(CuroboConfigurationError, match="live RoboLab joint/flange"):
        planner(connector, np.eye(4), np.arange(7), _world())
    assert impl.calls == []


def test_observed_world_never_silently_becomes_free_space(configuration):
    impl, world = FakeImplementation(), _world()
    world.mesh = []
    planner = CuroboPlanner(configuration.robot, configuration.profile, implementation=impl)
    with pytest.raises(CuroboConfigurationError, match="silently planned without"):
        planner(_connector(), np.eye(4), np.arange(7), world)
    assert impl.calls == []


@pytest.mark.parametrize("result", [(False, None), (True, np.zeros((2, 8))),
                                    (True, np.zeros((1, 7))), (True, np.full((2, 7), np.nan))])
def test_failed_or_malformed_trajectories_are_not_executable(configuration, result):
    from src.tools.motion.planning import MotionPlanningError
    impl = FakeImplementation()
    impl.result = result
    planner = CuroboPlanner(configuration.robot, configuration.profile, implementation=impl)
    with pytest.raises(MotionPlanningError):
        planner(_connector(), np.eye(4), np.arange(7), _world())


def test_native_flange_endpoint_mismatch_rejects_nominal_planner_success(configuration):
    from src.tools.motion.planning import MotionPlanningError
    native_endpoint = np.eye(4)
    native_endpoint[0, 3] = .1
    planner = CuroboPlanner(configuration.robot, configuration.profile, implementation=FakeImplementation())
    with pytest.raises(MotionPlanningError, match="native flange calibration"):
        planner(_connector(native_endpoint), np.eye(4), np.arange(7), _world())


def test_validate_runtime_reports_missing_install_without_importing_gpu(configuration, monkeypatch):
    planner = CuroboPlanner(configuration.robot, configuration.profile)
    def missing_runtime(request):
        assert request == {"operation": "probe"}
        raise RuntimeError("optional cuRobo runtime is not installed")
    monkeypatch.setattr(planner, "_get_worker", lambda: SimpleNamespace(call=missing_runtime))
    monkeypatch.setattr("src.tools.curobo.adapter._runtime_source",
                        lambda: pytest.fail("runtime probe must run in the child"))
    with pytest.raises(CuroboConfigurationError, match="runtime is not installed"):
        planner.validate_runtime()


def test_runtime_source_checks_do_not_claim_gpu_initialization(configuration, monkeypatch):
    planner = CuroboPlanner(configuration.robot, configuration.profile)
    evidence = dict(curobo_import="/installed/curobo/__init__.py", dependencies_available=True,
                    revision="pinned-test", gpu_initialized=False)
    def probe(request):
        assert request == {"operation": "probe"}
        return evidence
    child = SimpleNamespace(call=probe, process=SimpleNamespace(pid=42), log_path=Path("worker.log"))
    monkeypatch.setattr(planner, "_get_worker", lambda: child)
    monkeypatch.setattr("src.tools.curobo.adapter._runtime_source",
                        lambda: pytest.fail("runtime probe must run in the child"))
    assert planner.validate_runtime() == {**evidence, "planner_initialized": False,
        "process_isolated": True, "worker_pid": 42, "worker_log": "worker.log"}


@pytest.mark.parametrize("envelope", [None, {},
    dict(link="finger", radius_m=.025, opening_range_m=[0., .085]),
    dict(link="base_link", radius_m=.025, opening_range_m=[.001, .085]),
    dict(link="base_link", radius_m=.025, opening_range_m=[0., .084]),
    dict(link="base_link", radius_m=float("nan"), opening_range_m=[0., .085]),
])
def test_explicit_envelope_must_cover_entire_native_jaw_range(configuration, envelope):
    configuration.calibration["gripper_collision_envelope"] = envelope
    configuration.write()
    with pytest.raises(CuroboConfigurationError, match="gripper_collision_envelope"):
        CuroboPlanner(configuration.robot, configuration.profile)


@pytest.mark.parametrize("change", [
    {"collision_sphere_buffer": -.01},
    {"collision_sphere_buffer": {"base_link": -.01}},
    {"extra_collision_spheres": {"base_link": 2}},
])
def test_curobo_options_cannot_shrink_or_replace_declared_envelope(configuration, change):
    configuration.kin.update(change)
    configuration.write()
    with pytest.raises(CuroboConfigurationError, match="envelope"):
        CuroboPlanner(configuration.robot, configuration.profile)


def test_shifted_collision_sphere_must_contain_whole_flange_envelope(configuration):
    configuration.kin["collision_spheres"]["base_link"][0]["center"] = [.01, 0., 0.]
    configuration.write()
    with pytest.raises(CuroboConfigurationError, match="do not contain"):
        CuroboPlanner(configuration.robot, configuration.profile)


@pytest.mark.parametrize("maximum", [None, .09, float("nan")])
def test_live_jaw_range_checked_before_gpu(configuration, maximum):
    impl, connector = FakeImplementation(), _connector()
    connector.env.max_gripper_width_m = maximum
    planner = CuroboPlanner(configuration.robot, configuration.profile, implementation=impl)
    with pytest.raises(CuroboConfigurationError, match="live native opening range"):
        planner(connector, np.eye(4), np.arange(7), _world())
    assert impl.calls == []


def test_declared_envelope_checked_against_native_geometry_before_gpu(configuration, monkeypatch):
    impl = FakeImplementation()
    monkeypatch.setattr("src.tools.curobo.adapter._native_gripper_radius", lambda _env, *_: .04)
    planner = CuroboPlanner(configuration.robot, configuration.profile, implementation=impl)
    with pytest.raises(CuroboConfigurationError, match="all-opening native geometry"):
        planner(_connector(), np.eye(4), np.arange(7), _world())
    assert impl.calls == []


def test_unsupported_native_geometry_is_explicitly_rejected(configuration, monkeypatch):
    impl = FakeImplementation()
    monkeypatch.setattr("src.tools.curobo.adapter._native_gripper_radius", _native_gripper_radius)
    planner = CuroboPlanner(configuration.robot, configuration.profile, implementation=impl)
    with pytest.raises(CuroboConfigurationError, match="unsupported native gripper geometry"):
        planner(_connector(), np.eye(4), np.arange(7), _world())
    assert impl.calls == []


def test_native_geometry_is_checked_once_per_live_environment(configuration, monkeypatch):
    checked = []

    def radius(env, *_):
        checked.append(env)
        return .02

    monkeypatch.setattr("src.tools.curobo.adapter._native_gripper_radius", radius)
    planner = CuroboPlanner(configuration.robot, configuration.profile, implementation=FakeImplementation())
    connector = _connector()
    for _ in range(2):
        planner(connector, np.eye(4), np.arange(7), _world())
    second = _connector()
    planner(second, np.eye(4), np.arange(7), _world())
    assert len(checked) == 2 and checked[0] is connector.env and checked[1] is second.env


def test_native_geometry_bound_covers_continuous_jaw_motion():
    from scipy.spatial.transform import Rotation

    flange = "/panda/base_link"
    points = np.array([[[0., 0., 0.], [.02, .01, 0.], [0., 0., .04]]])
    local0, local1 = np.eye(4), np.eye(4)
    local0[:3, 3] = [.04, 0., .1]
    local1[:3, 3] = [0., .02, .01]
    tree = {flange: None}
    geometry = {"base": np.zeros((1, 3, 3))}
    bodies = {"base": "base_link"}
    for side in ("left", "right"):
        body = f"{side}_inner_finger"
        tree[f"/panda/{body}"] = dict(parent=flange, local0=local0, local1=local1,
                                     revolute=True, axis=0)
        geometry[side], bodies[side] = points.copy(), body
    model = SimpleNamespace(body_paths=tree, visual_triangles=geometry, geom_body=bodies)
    bound = _gripper_radius(model)
    # Include the full 0..pi/4 native stroke and angles beyond its allowed range.
    for angle in np.r_[np.linspace(0., np.pi / 4, 33), np.linspace(-np.pi, np.pi, 65)]:
        rotation = np.eye(4)
        rotation[:3, :3] = Rotation.from_rotvec([angle, 0., 0.]).as_matrix()
        pose = local0 @ rotation @ np.linalg.inv(local1)
        vertices = points.reshape(-1, 3) @ pose[:3, :3].T + pose[:3, 3]
        assert np.linalg.norm(vertices, axis=1).max() < bound
    del geometry["right"]
    with pytest.raises(ValueError, match="finger geometry is incomplete"):
        _gripper_radius(model)


def test_offset_native_geometry_bound_covers_limited_mimic_motion():
    from scipy.spatial.transform import Rotation

    flange = "/panda/base_link"
    tree = {flange: None}
    points = np.array([[[.02, -.01, 0.], [.04, .01, 0.], [.03, 0., .015]]])
    geometry, bodies = {"base": np.zeros((1, 3, 3))}, {"base": "base_link"}
    for side, ratio in (("left", 1.), ("right", -1.)):
        frame = np.eye(4)
        frame[:3, 3] = [.06, .03 * ratio, 0.]
        body = f"{side}_inner_finger"
        tree[f"/panda/{body}"] = dict(name=side, parent=flange, local0=frame,
            local1=np.eye(4), revolute=True, axis=2, limits=[0., np.pi / 4])
        if side == "right":
            tree[f"/panda/{body}"]["mimic"] = ("left", -1.)
        geometry[side], bodies[side] = points, body
    model = SimpleNamespace(body_paths=tree, visual_triangles=geometry, geom_body=bodies)
    center = np.array([.05, 0., 0.])
    bound = _gripper_radius(model, center)
    for angle in np.linspace(0., np.pi / 4, 301):
        for side, ratio in (("left", 1.), ("right", -1.)):
            frame = tree[f"/panda/{side}_inner_finger"]["local0"]
            vertices = Rotation.from_rotvec([0., 0., angle * ratio]).apply(points.reshape(-1, 3)) + frame[:3, 3]
            assert np.max(np.linalg.norm(vertices - center, axis=1)) < bound


def test_offset_gripper_envelope_requires_matching_sphere(configuration):
    configuration.calibration["gripper_collision_envelope"]["center_m"] = [.1, 0., 0.]
    configuration.write()
    with pytest.raises(CuroboConfigurationError, match="do not contain"):
        CuroboPlanner(configuration.robot, configuration.profile)
    configuration.kin["collision_spheres"]["base_link"][0]["center"] = [.1, 0., 0.]
    configuration.write()
    result = CuroboPlanner(configuration.robot, configuration.profile).preflight()
    assert result["gripper_collision_envelope"]["center_m"] == [.1, 0., 0.]
