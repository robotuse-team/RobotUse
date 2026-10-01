"""Grasp and placement planning with measured clouds and a supplied connector.

The pinned LIBERO public grip_site and GraspGen mesh both close along local X
and approach along +Z. The grasp origin is the mesh hand origin, NOT the TCP:
T_grasp_grip_site translates +0.097 m along Z. The pinned robosuite
panda_gripper.xml has a -90-degree root and +90-degree finger rotation.
The LIBERO public getter additionally shifts/rotates the
site, whereas the planner configuration claims TCP semantics. Explicit live
robot-only getter/IK calibration is required; no default planning map is safe.

Uses the supplied connector's planner and trajectory execution without reading
simulator scene objects. Nominal TCP capsule filtering and translated observed
payload filtering are partial checks: unseen geometry, full arm/self collision,
contact dynamics, mesh calibration error and planner fallback deviations are
NOT certified. Guarded mode leaves existing planner settings unchanged. Explicit
simulation collision-off mode skips these filters and uses native collision-free
IK routing (no self-aware fallback); numerical/endpoint checks remain.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping
import math
import numpy as np

from src.tools.gripper.evidence import read_gripper_fraction


class MotionPlanningError(RuntimeError):
    def __init__(self, message, *, planning_feedback=None):
        super().__init__(message)
        self.planning_feedback = dict(planning_feedback or {})


PLANNER_REASON_CODES = frozenset({
    'no_planner', 'no_usable_route', 'ik_failed', 'planner_failed',
    'invalid_trajectory', 'continuity_rejected', 'planner_exception',
})


def _directed_failure_code(reason):
    """Map known backend results; arbitrary backend text stays private."""
    known = {
        'motion_gen_failed_IK_FAIL': 'ik_failed',
        'motion_gen_failed_MotionGenStatus.IK_FAIL': 'ik_failed',
        'motion_planner_returned_none': 'no_usable_route',
        'no_interpolated_plan': 'no_usable_route',
    }
    return known.get(reason, 'planner_failed') if isinstance(reason, str) else 'planner_failed'


LIMITATIONS = (
    "observed nominal TCP capsule and translated observed payload only",
    "unknown/occluded space and full arm/self collision not certified",
    "plan_linear may fall back to a non-Cartesian path; proxy is not joint-path FK",
    "grasp evidence is provisional gripper proprioception, not benchmark success",
)


COLLISION_DISABLED_LIMITATIONS = (
    "simulation collision checks explicitly disabled: payload, TCP, external validator, native self/world",
    "directed IK only; native self-collision-aware pose fallback disabled",
    "numerical and stale-plan checks remain; tracking follows the environment profile and explicit checkpoints",
    "grasp evidence is provisional proprioception; final task verifier remains separate",
)


@dataclass(frozen=True)
class MotionConfig:
    collision_checks_enabled: bool = True
    approach_m: float = 0.10
    lift_m: float = 0.15
    clearance_m: float = 0.008
    tcp_radius_m: float = 0.015
    release_clearance_m: float = 0.015
    support_radius_m: float = 0.025
    # Absolute planner-frame Z (connector_base for the live calibrated runner).
    # A TCP routing preference, not maximum reach or full-arm collision proof.
    transit_policy: str = "legacy"
    high_transit_z_m: float = 0.45

    def __post_init__(self):
        if type(self.collision_checks_enabled) is not bool:
            raise ValueError("collision_checks_enabled must be a boolean")
        if self.transit_policy not in {"legacy", "high"}:
            raise ValueError("transit_policy must be legacy or high")
        if any(not math.isfinite(v) or v <= 0 for k, v in vars(self).items()
               if k not in {"collision_checks_enabled", "transit_policy"}):
            raise ValueError("motion distances/tolerances must be finite and positive")


def _transform(value: Any) -> np.ndarray:
    t = np.asarray(value, dtype=float)
    if (t.shape != (4, 4) or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-4)):
        raise ValueError("expected finite rigid 4x4 transform")
    return t.copy()


def _points(value: Any, *, minimum: int = 1) -> np.ndarray:
    p = np.asarray(value, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < minimum or not np.isfinite(p).all():
        raise ValueError(f"expected at least {minimum} finite observed Nx3 points")
    return p.copy()


def libero_panda_grasp_to_grip_site() -> np.ndarray:
    """Pinned GraspGen Panda mesh -> raw LIBERO grip_site, meters.

    This is a robot-asset calibration, not a read of any simulator scene object.
    NOT the public getter frame; use only after live getter/IK checks.
    """
    t = np.eye(4)
    t[2, 3] = 0.097
    return t


def _frame(frame: str, world_from_base: Any) -> np.ndarray:
    if frame in {"world", "connector_base"}:
        if world_from_base is not None:
            raise ValueError("world_from_base is only valid for base-frame inputs")
        # connector_base explicitly names LIBERO native coordinates:
        # calibrated camera poses and public planner goals both use robot base.
        # Do NOT add the simulator base translation a second time.
        return np.eye(4)
    if frame != "base" or world_from_base is None:
        raise ValueError("base inputs require explicit world_from_base; camera frame unsupported")
    return _transform(world_from_base)


def _world_points(points: Any, t: np.ndarray) -> np.ndarray:
    p = _points(points)
    return p @ t[:3, :3].T + t[:3, 3]


def transform_to_pose(value: Any) -> dict[str, Any]:
    t = _transform(value)
    r = t[:3, :3]
    # Eigenvector formulation handles half turns without trace singularities.
    k = np.array([
        [r[0,0]-r[1,1]-r[2,2], r[1,0]+r[0,1], r[2,0]+r[0,2], r[2,1]-r[1,2]],
        [r[1,0]+r[0,1], r[1,1]-r[0,0]-r[2,2], r[2,1]+r[1,2], r[0,2]-r[2,0]],
        [r[2,0]+r[0,2], r[2,1]+r[1,2], r[2,2]-r[0,0]-r[1,1], r[1,0]-r[0,1]],
        [r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1], np.trace(r)],
    ]) / 3
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1]
    if q[3] < 0:
        q = -q
    return {"position": dict(zip("xyz", map(float, t[:3,3]))),
            "rotation": dict(zip("xyzw", map(float, q)))}


def _pose_transform(pose: Mapping[str, Any]) -> np.ndarray:
    q = np.array([pose["rotation"][a] for a in "xyzw"], dtype=float)
    if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("invalid proprioceptive quaternion")
    x, y, z, w = q / np.linalg.norm(q)
    t = np.eye(4)
    t[:3,:3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                 [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                 [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    t[:3,3] = [pose["position"][a] for a in "xyz"]
    return _transform(t)


def observed_path_clearance(path: Any, obstacle_points: Any, *, radius_m: float) -> float:
    """Exact point-to-segment capsule clearance, all observed points retained.

    A nominal TCP path proxy only, NOT a robot collision certificate. Obstacles
    must exclude target-mask points, not a simulator box around the target.
    """
    path, points = _points(path, minimum=2), _points(obstacle_points)
    if not math.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("radius must be finite and positive")
    best = float("inf")
    for a, b in zip(path, path[1:]):
        v = b-a
        length2 = float(v @ v)
        for chunk in np.array_split(points, max(1, math.ceil(len(points)/4096))):
            f = np.zeros(len(chunk)) if length2 < 1e-20 else np.clip((chunk-a) @ v / length2, 0, 1)
            best = min(best, float(np.linalg.norm(chunk-a-f[:,None]*v, axis=1).min()))
    return best-radius_m


def _initial_support_lift_clearance(points: np.ndarray, shift: np.ndarray,
                                    obstacles: np.ndarray, margin: float) -> None:
    """Allow only already-touching bottom support pairs separating on +Z lift.

    No scene point is deleted. Exact pairwise sweep distance rejects new
    contacts, side/upper contacts and contacts whose clearance does not grow.
    This exception is not used for transport/place or nonvertical motion.
    """
    if (not np.allclose(shift[:2], 0, atol=1e-10) or shift[2] <= margin):
        raise MotionPlanningError("initial support exception requires separating vertical lift")
    bottom = float(points[:,2].min())
    length2 = float(shift @ shift)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(obstacles)
    except ImportError:
        tree = None
    for point in points:
        if tree is None:
            nearby = obstacles
        else:
            # Broad phase encloses the entire swept capsule, never subsamples.
            ids = tree.query_ball_point(point+shift/2, np.linalg.norm(shift)/2+margin)
            if not ids:
                continue
            nearby = obstacles[ids]
        delta = nearby-point
        fraction = np.clip(delta @ shift / length2, 0, 1)
        minimum2 = np.sum((delta-fraction[:,None]*shift)**2,axis=1)
        collision = minimum2 < margin*margin
        initial2 = np.sum(delta*delta,axis=1)
        final2 = np.sum((delta-shift)**2,axis=1)
        permitted = ((point[2] <= bottom+margin)
                     & (nearby[:,2] <= point[2])
                     & (initial2 < margin*margin)
                     & (delta @ shift <= 0)
                     & (final2 >= margin*margin))
        if np.any(collision & ~permitted):
            raise MotionPlanningError("observed payload translation intersects observed scene")


def _payload_clearance(points: np.ndarray, shifts: list[np.ndarray], obstacles: np.ndarray,
                       margin: float, *, allow_initial_support: bool = False) -> None:
    if allow_initial_support:
        if len(shifts) != 2 or not np.allclose(shifts[0], 0, atol=1e-10):
            raise MotionPlanningError("support exception only valid at initial grasp lift")
        _initial_support_lift_clearance(points, shifts[1], obstacles, margin)
        return
    # Optional existing SciPy accelerates clouds without dropping points.
    # Inflate radius by half translation spacing to cover between samples.
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None
    if cKDTree is not None:
        tree = cKDTree(obstacles)
        for a, b in zip(shifts, shifts[1:]):
            distance = float(np.linalg.norm(b-a))
            count = max(1, math.ceil(distance / 0.004))
            radius = margin + distance / (2*count)
            for fraction in np.linspace(0, 1, count+1):
                shifted = points + a + fraction*(b-a)
                if np.any(tree.query(shifted, workers=1)[0] < radius):
                    raise MotionPlanningError("observed payload translation intersects observed scene")
        return
    # Each sensed payload point sweeps an exact translation segment. No hidden
    # object shape is filled in, and no point subsampling erases narrow hazards.
    for point in points:
        if observed_path_clearance([point+s for s in shifts], obstacles, radius_m=margin) < 0:
            raise MotionPlanningError("observed payload translation intersects observed scene")


@dataclass
class GraspPlan:
    segments: tuple
    targets: tuple
    start_joints: tuple
    grasp_transform: np.ndarray
    grasp_to_ee: np.ndarray
    target_points: np.ndarray
    obstacle_points: np.ndarray
    clearance_m: float | None
    trajectory_validated: bool = False
    limitations: tuple = LIMITATIONS
    consumed: bool = False
    grasp_executed: bool = False
    connector: Any = field(default=None, repr=False)
    collision_checks_enabled: bool = True
    target_labels: tuple = ("pregrasp", "grasp", "lift")
    transit_policy: str = "legacy"
    high_transit_z_m: float | None = None
    open_width_m: float = .08

    @property
    def segment_labels(self):
        """Each segment is labeled by its destination target."""
        return self.target_labels


@dataclass
class PlacePlan:
    segments: tuple
    targets: tuple
    start_joints: tuple
    clearance_m: float | None
    trajectory_validated: bool = False
    limitations: tuple = LIMITATIONS
    consumed: bool = False
    connector: Any = field(default=None, repr=False)
    collision_checks_enabled: bool = True
    target_labels: tuple = ("lift", "high_transit", "release", "retreat")
    transit_policy: str = "legacy"
    high_transit_z_m: float | None = None

    @property
    def segment_labels(self):
        return self.target_labels


class _DirectedOnlyCollisionDisabledIK:
    """Avoid CuRoboBackend's self-collision-aware plan_to_pose fallback."""
    def __init__(self, backend):
        self.backend = backend
        self.planning_failure_code = None

    def plan_linear(self, start_pose, target_pose, *, seed_joints):
        self.planning_failure_code = None
        backend = self.backend
        impl = backend._import_impl()
        success, trajectory, reason = impl.plan_directed_linear(
            start_config=backend._resolve_seed(list(seed_joints)),
            start_pose=backend._pose_for_curobo(start_pose, 0),
            target_pose=backend._pose_for_curobo(target_pose, 0),
            allowed_axes=["X", "Y", "Z"], orientation_mode="TARGET_AT_END",
            robot_file=backend._robot_file,
        )
        if not success or trajectory is None:
            self.planning_failure_code = (_directed_failure_code(reason) if not success
                                          else 'no_usable_route')
            return None
        rows = np.asarray(trajectory, dtype=float)
        if rows.ndim != 2 or rows.shape[1] < 7 or not np.isfinite(rows).all():
            self.planning_failure_code = 'invalid_trajectory'
            return None
        # GaP execute_trajectory consumes simulator-order joint waypoints.
        return {"waypoints": [{"positions": row[:7].tolist()} for row in rows]}


def _collision_disabled_planner(connector):
    from types import SimpleNamespace
    backend = connector.ik
    if type(backend).__name__ == "CuRoboBackend" and type(backend).__module__ == "gap.connector.ik":
        # Pinned _get_directed_planner uses self_collision_check=False, no scene
        # config/cache. Do NOT call backend.plan_linear: its fallback restores self.
        return SimpleNamespace(ik=_DirectedOnlyCollisionDisabledIK(backend))
    if getattr(backend, "collision_checks_enabled", None) is False:
        # Explicit collision-free adapter/test fixture, never inferred by absence.
        return connector
    raise MotionPlanningError("collision-off requires pinned CuRoboBackend or explicit collision-free IK adapter")


def _plan(connector, targets, obstacles, config, trajectory_validator):
    from src.tools.motion.robot_state import _current_robot_state, _plan_chain
    pose, joints = _current_robot_state(connector)
    if len(joints) != 7 or not np.isfinite(joints).all():
        raise MotionPlanningError("bridge requires finite seven-DOF Panda proprioception")
    if getattr(connector.ik, "trajectory_needs_joint_reverse", False):
        raise MotionPlanningError("bridge requires simulator-order planned joints")
    xyz = [_pose_transform(pose)[:3,3], *[t[:3,3] for t in targets]]
    clearance = None
    if config.collision_checks_enabled:
        clearance = observed_path_clearance(xyz, obstacles, radius_m=config.tcp_radius_m+config.clearance_m)
        if clearance < 0:
            raise MotionPlanningError("nominal TCP capsule intersects observed scene")
    poses = tuple(transform_to_pose(t) for t in targets)
    planner_connector = connector if config.collision_checks_enabled else _collision_disabled_planner(connector)
    failure = {}
    segments, detail = _plan_chain(planner_connector, start_pose=pose, start_joints=joints,
                                  targets=poses, failure_details=failure)
    if not segments:
        code = failure.get('planner_reason_code')
        feedback = {'kind': 'planning', 'planner_reason_code':
                    code if isinstance(code, str) and code in PLANNER_REASON_CODES else 'planner_failed'}
        index = failure.get('segment_index')
        if type(index) is int and 0 <= index < len(targets):
            start = _pose_transform(pose) if index == 0 else targets[index-1]
            target = targets[index]
            delta = target[:3, 3] - start[:3, 3]
            angle = math.degrees(math.acos(float(np.clip(
                (np.trace(start[:3, :3].T @ target[:3, :3])-1.)/2., -1., 1.))))
            feedback.update(segment_index=index,
                start_reference='current_ee' if index == 0 else 'previous_planned_waypoint',
                requested_translation_m=dict(zip(('dx_m', 'dy_m', 'dz_m'), map(float, delta))),
                orientation_change_deg=angle)
        raise MotionPlanningError(detail, planning_feedback=feedback)
    validated = False
    if config.collision_checks_enabled and trajectory_validator is not None:
        # Callback must check the actual joint segments (incl planner fallback),
        # complete robot and carried payload against observed geometry. It is
        # caller supplied; this module never labels its own proxy full-arm safe.
        validated = trajectory_validator(segments, tuple(joints), obstacles.copy()) is True
        if not validated:
            raise MotionPlanningError("external trajectory validator rejected path")
    return segments, poses, tuple(joints), clearance, validated


def plan_grasp(connector, *, grasp_transform, target_points, obstacle_points,
               grasp_to_ee=None, frame="world", world_from_base=None,
               config=MotionConfig(), trajectory_validator=None, lift_after_grasp=True, open_width_m=.08,
               max_width_m=.08) -> GraspPlan:
    if not np.isfinite(max_width_m) or max_width_m <= 0:
        raise ValueError('maximum gripper opening must be positive and finite')
    if not np.isfinite(open_width_m) or not 0 < open_width_m <= max_width_m:
        raise ValueError(f'gripper opening must be in (0, {max_width_m}] meters')
    w = _frame(frame, world_from_base)
    grasp = w @ _transform(grasp_transform)
    if grasp_to_ee is None:
        raise MotionPlanningError("explicit grasp_to_ee calibration required: public and IK frames differ")
    calibration = _transform(grasp_to_ee)
    obj, obstacles = _world_points(target_points, w), _world_points(obstacle_points, w)
    ee = grasp @ calibration
    approach, lift = ee.copy(), ee.copy()
    approach[:3,3] -= config.approach_m * grasp[:3,2]
    lift[2,3] += config.lift_m
    targets = (approach, ee, lift)
    labels = ("pregrasp", "grasp", "lift")
    if config.transit_policy == "high":
        current = _pose_transform(connector.get_ee_pose())
        travel_z = max(config.high_transit_z_m, current[2,3], approach[2,3], lift[2,3])
        initial_lift = current.copy()
        initial_lift[2,3] = travel_z  # lift at current XY AND orientation first
        above = ee.copy()
        above[2,3] = travel_z  # reorient only during the high horizontal transit
        high_approach = approach.copy()
        high_approach[2,3] = travel_z
        lift[2,3] = travel_z
        targets, labels = [initial_lift, above], ["initial_lift", "high_transit"]
        # Tilted predictions retain their contact geometry and approach axis.
        # Align the pregrasp XY at height before its strictly vertical descent.
        if not np.allclose(approach[:2,3], ee[:2,3], atol=1e-10, rtol=0):
            targets.append(high_approach)
            labels.append("high_pregrasp_align")
        targets.extend((approach, ee, lift))
        labels.extend(("pregrasp", "grasp", "lift"))
        targets, labels = tuple(targets), tuple(labels)
    if not lift_after_grasp:
        # Articulated/contact tasks must stay at the selected grasp after closing.
        # Remove the lift before planning, so an irrelevant lift cannot reject IK.
        targets, labels = targets[:-1], labels[:-1]
    if config.collision_checks_enabled and lift_after_grasp:
        _payload_clearance(obj, [np.zeros(3), lift[:3,3]-ee[:3,3]], obstacles, config.clearance_m,
                           allow_initial_support=True)
    try:
        segments, poses, joints, clearance, validated = _plan(
            connector, targets, obstacles, config, trajectory_validator)
    except MotionPlanningError as exc:
        index = exc.planning_feedback.get('segment_index')
        if type(index) is int and 0 <= index < len(labels):
            exc.planning_feedback['segment'] = labels[index]
        if lift_after_grasp or config.transit_policy != 'high':
            raise
        # A horizontal handle pose can be reachable at contact height but not
        # above the cabinet. Try the existing direct pregrasp route, without
        # executing either rejected plan or changing the learned contact pose.
        from dataclasses import replace
        return plan_grasp(connector, grasp_transform=grasp_transform,
            target_points=target_points, obstacle_points=obstacle_points,
            grasp_to_ee=grasp_to_ee, frame=frame, world_from_base=world_from_base,
            config=replace(config, transit_policy='legacy'),
            trajectory_validator=trajectory_validator, lift_after_grasp=False,
            open_width_m=open_width_m, max_width_m=max_width_m)
    return GraspPlan(segments, poses, joints, grasp, calibration, obj, obstacles,
                     clearance, validated, connector=connector,
                     target_labels=labels, transit_policy=config.transit_policy,
                     high_transit_z_m=config.high_transit_z_m if config.transit_policy == "high" else None,
                     open_width_m=open_width_m,
                     collision_checks_enabled=config.collision_checks_enabled,
                     limitations=LIMITATIONS if config.collision_checks_enabled else COLLISION_DISABLED_LIMITATIONS)


def _execution_guard(connector, plan, allow_partial_safety):
    from src.tools.motion.robot_state import _current_robot_state
    if plan.connector is not connector or plan.consumed:
        raise MotionPlanningError("foreign or already consumed plan")
    if plan.collision_checks_enabled and not allow_partial_safety:
        # External validator does not remove unknown-space/contact limitations.
        raise MotionPlanningError("execution requires explicit allow_partial_safety acknowledgement")
    _, joints = _current_robot_state(connector)
    if len(joints) != len(plan.start_joints) or not np.isfinite(joints).all() or np.max(np.abs(np.array(joints)-plan.start_joints)) > 0.025:
        raise MotionPlanningError("stale plan: robot moved since planning")
    plan.consumed = True  # fail-stop, also when a physical call throws


def _execute_checked(connector, segments, targets, *, collision_checks_enabled=True,
                     stationary_final=False):
    from src.tools.motion.robot_state import (
        _execute_segments, _pose_execution_error, _current_robot_state, _trajectory_end,
    )
    from src.tools.motion.tolerances import cartesian_tolerances, requires_profile_tracking
    position_limit, angle_limit = cartesian_tolerances(connector)
    enforce = collision_checks_enabled or requires_profile_tracking(connector)
    diagnostics = []
    for index, (segment, target) in enumerate(zip(segments, targets)):
        expected_joints = _trajectory_end(segment)
        guard = getattr(getattr(connector, 'env', None), 'stationary_grasp_arrival', None)
        require_stationary = stationary_final and index == len(segments) - 1 and callable(guard)
        with guard(expected_joints) if require_stationary else nullcontext() as arrival:
            if collision_checks_enabled:
                _execute_segments(connector, (segment,))
            else:
                # Off simulation route bypasses inherited joint-tolerance holds.
                # A real command exception still propagates; never pretend it ran.
                direct = getattr(connector, "execute_trajectory", None)
                if callable(direct):
                    direct(segment)
                else:
                    registry = getattr(connector, "tool_registry", None)
                    if registry is None:
                        raise MotionPlanningError("connector cannot execute planner trajectories")
                    registry.invoke("robot.execute_trajectory", trajectory=segment)
        if arrival is not None and not arrival['accepted']:
            raise MotionPlanningError('grasp endpoint did not confirm stationary arrival before closing')
        actual_pose, actual_joints = _current_robot_state(connector)
        if (len(actual_joints) < len(expected_joints)
                or not np.isfinite(actual_joints).all()
                or not np.isfinite(expected_joints).all()):
            raise MotionPlanningError("invalid execution joint proprioception")
        joint_error = float(np.max(np.abs(np.array(actual_joints[:len(expected_joints)])-expected_joints)))
        # Validate a finite rigid measured pose even when thresholds are off.
        _pose_transform(actual_pose)
        position_error, angle_error = _pose_execution_error(actual_pose, target)
        if not math.isfinite(position_error) or not math.isfinite(angle_error):
            raise MotionPlanningError("invalid execution Cartesian proprioception")
        diagnostics.append({"joint_max_error_rad": joint_error,
                            "position_error_m": position_error,
                            "orientation_error_rad": angle_error,
                            "thresholds_enforced": enforce,
                            "position_tolerance_m": position_limit,
                            "orientation_tolerance_rad": angle_limit})
        if arrival is not None:
            diagnostics[-1]['stationary_arrival'] = dict(arrival)
        if enforce and (position_error > position_limit or angle_error > angle_limit):
            exc = MotionPlanningError("execution endpoint disagrees with public EE target; hold")
            exc.evidence = {'execution_diagnostics': diagnostics, 'execution_stage': 'endpoint'}
            raise exc
    return diagnostics


def _action_boundary(plan, label):
    """Validate one-to-one semantic metadata before any physical command."""
    if (len(plan.segments) != len(plan.targets)
            or len(plan.target_labels) != len(plan.targets)
            or plan.target_labels.count(label) != 1):
        raise MotionPlanningError("invalid semantic target/segment labels")
    return plan.target_labels.index(label) + 1


def execute_grasp(connector, plan: GraspPlan, *, allow_partial_safety=False, on_closed=None) -> dict:
    boundary = _action_boundary(plan, "grasp")
    _execution_guard(connector, plan, allow_partial_safety)
    from src.runtime.budget import gripper_settle_steps
    opening_steps = gripper_settle_steps(connector, 'open', 40)
    connector.open_gripper(settle_steps=opening_steps)
    if plan.open_width_m != getattr(getattr(connector, 'env', None), 'max_gripper_width_m', .08):
        # LIBERO's fractional command integrates opening/closing; it does not
        # set an aperture. Pickup and contact plans need the same width target.
        connector.set_gripper_width(plan.open_width_m, settle_steps=opening_steps)
    checkpoint = getattr(plan, 'execution_checkpoint', None)
    pause = getattr(plan, 'pregrasp_pause', None)
    if checkpoint is None:
        diagnostics = _execute_checked(connector, plan.segments[:boundary], plan.targets[:boundary],
                                       collision_checks_enabled=plan.collision_checks_enabled,
                                       stationary_final=True)
    else:
        pre = _action_boundary(plan, 'pregrasp')
        diagnostics = _execute_checked(connector, plan.segments[:pre], plan.targets[:pre],
                                       collision_checks_enabled=plan.collision_checks_enabled)
        checkpoint('pregrasp', diagnostics[-1])
        if pause is not None:
            # The arm rests at the measured pregrasp. A caller may now replace
            # the remaining descent with a plan made from THIS pose (a small
            # refined correction); None keeps the original descent.
            replacement = pause(plan)
            if replacement is not None:
                _execution_guard(connector, replacement, allow_partial_safety)
                previous_width = plan.open_width_m
                plan = replacement
                boundary = _action_boundary(plan, 'grasp')
                pre = _action_boundary(plan, 'pregrasp')
                if plan.open_width_m != previous_width:
                    connector.set_gripper_width(plan.open_width_m, settle_steps=opening_steps)
                diagnostics += _execute_checked(connector, plan.segments[:pre], plan.targets[:pre],
                                                collision_checks_enabled=plan.collision_checks_enabled)
                checkpoint('pregrasp', diagnostics[-1])
        diagnostics += _execute_checked(connector, plan.segments[pre:boundary], plan.targets[pre:boundary],
                                        collision_checks_enabled=plan.collision_checks_enabled,
                                        stationary_final=True)
        checkpoint('pre_close', diagnostics[-1])
    from src.tools.motion.tolerances import cartesian_tolerances
    position_limit, angle_limit = cartesian_tolerances(connector, position=.02)
    if 'lift' not in plan.target_labels and (diagnostics[-1]['position_error_m'] > position_limit
            or diagnostics[-1]['orientation_error_rad'] > angle_limit):
        exc = MotionPlanningError('contact grasp did not reach the selected pose')
        exc.evidence = {'execution_diagnostics': diagnostics, 'execution_stage': 'pre_close'}
        raise exc
    connector.close_gripper(settle_steps=gripper_settle_steps(connector, 'close', 60))
    if on_closed is not None:
        on_closed()  # sensor-only evidence callback before the lift, never a control policy
    diagnostics += _execute_checked(connector, plan.segments[boundary:], plan.targets[boundary:],
                                    collision_checks_enabled=plan.collision_checks_enabled)
    stage = "post_lift" if "lift" in plan.target_labels else "post_close"
    measurement = read_gripper_fraction(connector, stage=stage)
    fraction = measurement["value"]
    if measurement["validity"] == "invalid":
        exc = MotionPlanningError("invalid gripper proprioception")
        exc.evidence = {"gripper_measurement": measurement,
                        "gripper_fraction": fraction,
                        "execution_stage": stage, "motion_completed": True,
                        "execution_diagnostics": diagnostics}
        raise exc
    empty = measurement["empty"] is True
    plan.grasp_executed = not empty
    return {"status": "failed" if empty else "executed",
            "held_state": "not_held" if empty else "unknown",
            "grasp_evidence": "empty_closed_gripper" if empty else ("provisional" if fraction is not None else "unknown"),
            "gripper_fraction": fraction, "gripper_measurement": measurement,
            "success_verified": False,
            "full_arm_safety_certified": False,
            "execution_diagnostics": diagnostics,
            "collision_checks_enabled": plan.collision_checks_enabled,
            "collision_policy": "guarded_partial" if plan.collision_checks_enabled else "disabled_simulation",
            "limitations": list(plan.limitations)}


def plan_place(connector, *, grasp_plan: GraspPlan, destination_points, obstacle_points,
               frame="world", world_from_base=None, config=MotionConfig(),
               trajectory_validator=None) -> PlacePlan:
    """Keep grasp orientation; center observed payload above local observed support.

    Destination cloud must denote a support surface, not an entire container
    including walls/rim. Uses local high Z, never simulator pose/AABB. This is a
    release-above-surface strategy, not containment or stable-placement proof.
    """
    if grasp_plan.connector is not connector or not grasp_plan.grasp_executed:
        raise MotionPlanningError("placement requires this connector's executed grasp")
    w = _frame(frame, world_from_base)
    destination, obstacles = _world_points(destination_points, w), _world_points(obstacle_points, w)
    current = _pose_transform(connector.get_ee_pose())
    grasp_ee = grasp_plan.grasp_transform @ grasp_plan.grasp_to_ee
    if config.collision_checks_enabled and not np.allclose(current[:3,:3], grasp_ee[:3,:3], atol=0.05):
        raise MotionPlanningError("held orientation changed; re-observe payload before placement")
    center = np.median(destination[:,:2], axis=0)
    local = destination[np.linalg.norm(destination[:,:2]-center, axis=1) <= config.support_radius_m]
    if len(local) < 3:
        raise MotionPlanningError("no observed destination support near intended release center")
    obj = grasp_plan.target_points
    shift = np.r_[center-np.median(obj[:,:2], axis=0),
                  float(local[:,2].max())+config.release_clearance_m-float(obj[:,2].min())]
    release = grasp_ee.copy()
    release[:3,3] += shift
    travel_z = max(current[2,3], release[2,3]) + config.lift_m
    if config.transit_policy == "high":
        # Keep actual held orientation, including off-policy tracking deviations;
        # destination translation remains the original observed-support formula.
        release[:3,:3] = current[:3,:3]
        travel_z = max(config.high_transit_z_m, current[2,3], release[2,3]+config.lift_m)
    lift, above, retreat = current.copy(), release.copy(), release.copy()
    lift[2,3] = above[2,3] = retreat[2,3] = travel_z
    shifts = [t[:3,3]-grasp_ee[:3,3] for t in (current, lift, above, release)]
    if config.collision_checks_enabled:
        _payload_clearance(obj, shifts, obstacles, config.clearance_m)
    segments, poses, joints, clearance, validated = _plan(
        connector, (lift, above, release, retreat), obstacles, config, trajectory_validator)
    return PlacePlan(segments, poses, joints, clearance, validated, connector=connector,
                     transit_policy=config.transit_policy,
                     high_transit_z_m=config.high_transit_z_m if config.transit_policy == "high" else None,
                     collision_checks_enabled=config.collision_checks_enabled,
                     limitations=LIMITATIONS if config.collision_checks_enabled else COLLISION_DISABLED_LIMITATIONS)


def execute_place(connector, plan: PlacePlan, *, allow_partial_safety=False) -> dict:
    from src.runtime.budget import (gripper_settle_steps, require_motion_budget,
                                    admit_place_motion, retreat_fits_budget)
    plan.release_commanded = False
    boundary = _action_boundary(plan, "release")
    _execution_guard(connector, plan, allow_partial_safety)
    opening_steps = gripper_settle_steps(connector, 'open', 60)
    admit_place_motion(connector, plan, boundary, opening_steps)
    diagnostics = _execute_checked(connector, plan.segments[:boundary], plan.targets[:boundary],
                                   collision_checks_enabled=plan.collision_checks_enabled)
    require_motion_budget(connector, (), gripper_steps=opening_steps)
    plan.release_commanded = True
    connector.open_gripper(settle_steps=opening_steps)
    retreat = retreat_fits_budget(connector, plan.segments[boundary:])
    if retreat:
        diagnostics += _execute_checked(connector, plan.segments[boundary:], plan.targets[boundary:],
                                        collision_checks_enabled=plan.collision_checks_enabled)
    return {"status": "released", "success_verified": False,
            "retreat_executed": retreat,
            "retreat_reason_code": None if retreat else 'insufficient_simulation_time',
            "full_arm_safety_certified": False,
            "execution_diagnostics": diagnostics,
            "collision_checks_enabled": plan.collision_checks_enabled,
            "collision_policy": "guarded_partial" if plan.collision_checks_enabled else "disabled_simulation",
            "limitations": list(plan.limitations)}


class GraspGenMotionBridge:
    """Private adapter; scene_points_world means observed NON-target scene points.

    Segmentation must exclude target pixels upstream. Do not remove an oracle
    bounding box or all points near the object (which can erase real obstacles).
    A new destination Point selection is consumed only by place(), after grasp.
    """
    def __init__(self, connector, *, grasp_to_ee=None, allow_partial_safety=False,
                 trajectory_validator: Callable | None = None, config=MotionConfig(), max_width_m=.08):
        self.connector, self.calibration = connector, grasp_to_ee
        self.allow_partial_safety = allow_partial_safety
        self.validator, self.config = trajectory_validator, config
        self.max_width_m = max_width_m

    def validate(self, grasp_pose_world_4x4, object_points_world, scene_points_world):
        return plan_grasp(self.connector, grasp_transform=grasp_pose_world_4x4,
                          target_points=object_points_world, obstacle_points=scene_points_world,
                          grasp_to_ee=self.calibration, config=self.config,
                          trajectory_validator=self.validator, max_width_m=self.max_width_m)

    def execute(self, plan):
        return execute_grasp(self.connector, plan, allow_partial_safety=self.allow_partial_safety)

    def place(self, destination_points_world, held_plan, scene_points_world=None):
        # Refuse old pre-grasp scene geometry: the runner must capture a fresh
        # nontarget cloud with the new destination Point selection.
        if scene_points_world is None:
            raise MotionPlanningError("placement requires a fresh observed scene cloud")
        plan = plan_place(self.connector, grasp_plan=held_plan,
                          destination_points=destination_points_world,
                          obstacle_points=scene_points_world, config=self.config,
                          trajectory_validator=self.validator)
        result = execute_place(self.connector, plan, allow_partial_safety=self.allow_partial_safety)
        held_plan.grasp_executed = False
        return result


class PlannerFrameConnector:
    """Adapt a public getter to the planner's actual commanded EE frame.

    T_public_planner is a robot-only measured constant: returned pose becomes
    T_world_public @ T_public_planner. Planner and execution methods are left
    untouched, so no vendor state or simulator collision settings are changed.
    Supply only after checking the planner FK against that same robot frame.
    Use this SAME wrapper instance for planning and execution (plan ownership).
    """
    def __init__(self, connector, *, public_to_planner):
        self._connector = connector
        self._public_to_planner = _transform(public_to_planner)

    def get_ee_pose(self, arm_id=0):
        if arm_id != 0:
            raise MotionPlanningError("Panda bridge supports arm zero only")
        public = _pose_transform(self._connector.get_ee_pose())
        return transform_to_pose(public @ self._public_to_planner)

    def __getattr__(self, name):
        return getattr(self._connector, name)


@dataclass(frozen=True)
class RobotFrameCalibration:
    grasp_to_ee: np.ndarray
    public_to_planner: np.ndarray
    frame: str = "connector_base"


def calibrate_libero_panda_from_robot_probe(probe: Mapping[str, Any], *,
                                           planner_fk_pose: Mapping[str, Any]) -> RobotFrameCalibration:
    """Derive maps from measured robot bodies/public getter AND planner FK.

    The probe is robot-only: robot_bodies holds base, hand, right_gripper,
    fingers; public_ee is the unwrapped getter matrix. FK is for exactly the
    same seven joint positions, configured with tcp_offset=(0,0,-.097), no
    TCP rotation and zero arm_base_translation. No scene-object data is used.
    Current LIBERO camera/public coordinates are connector-base, not sim world.
    """
    bodies = probe["robot_bodies"]
    base_inverse = np.linalg.inv(_transform(bodies["robot0_base"]))
    hand = base_inverse @ _transform(bodies["robot0_right_hand"])
    grasp_frame = base_inverse @ _transform(bodies["gripper0_right_gripper"])
    public = _transform(probe["public_ee"])
    desired = hand.copy()
    desired[:3,3] += hand[:3,2]*0.097
    fk = _pose_transform(planner_fk_pose)
    position_error = np.linalg.norm(fk[:3,3]-desired[:3,3])
    angle_error = math.acos(float(np.clip((np.trace(fk[:3,:3].T @ desired[:3,:3])-1)/2,-1,1)))
    if position_error > 0.01 or angle_error > 0.10:
        raise MotionPlanningError("robot-only FK disagrees with measured Panda planner TCP")
    # Verify measured jaw separation agrees with GraspGen local X (sign is
    # irrelevant for a symmetric antipodal gripper); approach is hand +Z.
    left = _transform(bodies["gripper0_leftfinger"])[:3,3]
    right = _transform(bodies["gripper0_rightfinger"])[:3,3]
    jaw = base_inverse[:3,:3] @ (left-right)
    if np.linalg.norm(jaw) < 0.01 or abs(float(jaw @ grasp_frame[:3,0])) / np.linalg.norm(jaw) < .99:
        raise MotionPlanningError("measured open Panda jaws disagree with GraspGen mesh X axis")
    grasp_to_ee = _transform(np.linalg.inv(grasp_frame) @ desired)
    public_to_planner = _transform(np.linalg.inv(public) @ desired)
    return RobotFrameCalibration(grasp_to_ee, public_to_planner)
