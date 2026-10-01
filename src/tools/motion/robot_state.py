"""GUI-style finite-candidate manipulation over private sensor geometry.

This module starts after RGB-D perception has selected target and destination
geometry.  The untrusted selector receives only rendered boards and finite
aliases.  OBB coordinates, Cartesian poses, IK trajectories, joint arrays,
connector handles, and verifier internals remain in private Python objects.
"""


from __future__ import annotations


import math


from typing import Any, Callable, Iterable, Mapping, Sequence


from src.tools.perception.geometry import (
    METRIC_DEPTH_MAGIC,
    OrientedBoundingBox,
    PixelBox,
    SensorFramePayload,
    SensorPerceptionError,
    fit_oriented_bounding_box,
    unproject_masked_depth,
)


EXECUTED_SEGMENT_JOINT_TOLERANCE_RAD = 0.025


def _xyz(pose: Mapping[str, Any]) -> tuple[float, float, float]:
    position = pose["position"]
    return tuple(float(position[key]) for key in ("x", "y", "z"))  # type: ignore[return-value]


def _trajectory_end(segment: Mapping[str, Any]) -> list[float]:
    waypoints = segment.get("waypoints")
    if not isinstance(waypoints, list) or not waypoints:
        raise SensorPerceptionError("planner returned an empty trajectory segment")
    positions = waypoints[-1].get("positions")
    if positions is None:
        raise SensorPerceptionError("planner waypoint lacks joint positions")
    return [float(value) for value in positions]


def _segments_safe(
    segments: Sequence[Mapping[str, Any]],
    *,
    start_joints: Sequence[float],
) -> bool:
    previous: list[float] | None = [float(value) for value in start_joints]
    for segment in segments:
        waypoints = segment.get("waypoints")
        if not isinstance(waypoints, list) or not waypoints:
            return False
        for waypoint in waypoints:
            positions = waypoint.get("positions") if isinstance(waypoint, Mapping) else None
            if positions is None:
                return False
            try:
                current = [float(value) for value in positions]
            except (TypeError, ValueError):
                return False
            if not current or not all(math.isfinite(value) for value in current):
                return False
            if previous is not None:
                if len(previous) != len(current):
                    return False
                if max(abs(a - b) for a, b in zip(previous, current)) > 0.75:
                    return False
            previous = current
    return True


def _plan_chain(
    connector: Any,
    *,
    start_pose: Mapping[str, Any],
    start_joints: Sequence[float],
    targets: Sequence[Mapping[str, Any]],
    failure_details: dict[str, Any] | None = None,
) -> tuple[tuple[Mapping[str, Any], ...], str]:
    # Optional per-call diagnostics preserve the existing (segments, detail)
    # return contract. Public consumers must project this private side channel.
    if failure_details is not None:
        failure_details.clear()
    def failed(code, index=None):
        if failure_details is not None:
            failure_details['planner_reason_code'] = code
            if index is not None:
                failure_details['segment_index'] = index
    planner = getattr(getattr(connector, "ik", None), "plan_linear", None)
    if planner is None:
        failed('no_planner')
        return (), "connector has no classical linear IK planner"
    segments: list[Mapping[str, Any]] = []
    pose_cursor = start_pose
    joint_cursor = [float(value) for value in start_joints]
    for index, target in enumerate(targets):
        try:
            segment = planner(pose_cursor, target, seed_joints=joint_cursor)
        except Exception as exc:
            failed('planner_exception', index)
            return (), f"planner exception on segment {index}: {type(exc).__name__}"
        if not isinstance(segment, Mapping) or not segment.get("waypoints"):
            code = getattr(getattr(planner, '__self__', None), 'planning_failure_code', None)
            failed(code if isinstance(code, str) else 'no_usable_route', index)
            return (), f"planner rejected segment {index}"
        segments.append(segment)
        try:
            joint_cursor = _trajectory_end(segment)
        except SensorPerceptionError:
            failed('invalid_trajectory', index)
            return (), f"planner emitted invalid segment {index}"
        pose_cursor = target
    if not _segments_safe(segments, start_joints=start_joints):
        failed('continuity_rejected')
        return (), "joint-path continuity guard rejected planner output"
    return tuple(segments), "classical planner feasible"


def _current_robot_state(connector: Any) -> tuple[Mapping[str, Any], list[float]]:
    pose = connector.get_ee_pose()
    observation = connector.get_observation()
    try:
        positions = observation["arms"][0]["joint_state"]["positions"]
    except (KeyError, IndexError, TypeError) as exc:
        raise SensorPerceptionError("connector lacks arm proprioception") from exc
    return pose, [float(value) for value in positions]


def _execute_segments(connector: Any, segments: Sequence[Mapping[str, Any]]) -> None:
    direct = getattr(connector, "execute_trajectory", None)
    if direct is not None:
        for segment in segments:
            direct(segment)
            expected = _trajectory_end(segment)
            _, actual = _current_robot_state(connector)
            if len(actual) < len(expected) or max(
                abs(left - right) for left, right in zip(actual, expected)
            ) > EXECUTED_SEGMENT_JOINT_TOLERANCE_RAD:
                raise SensorPerceptionError(
                    "executed trajectory endpoint disagrees with arm proprioception"
                )
        return
    registry = getattr(connector, "tool_registry", None)
    if registry is None:
        raise SensorPerceptionError("connector cannot execute planner trajectories")
    for segment in segments:
        registry.invoke("robot.execute_trajectory", trajectory=segment)
        expected = _trajectory_end(segment)
        _, actual = _current_robot_state(connector)
        if len(actual) < len(expected) or max(
            abs(left - right) for left, right in zip(actual, expected)
        ) > EXECUTED_SEGMENT_JOINT_TOLERANCE_RAD:
            raise SensorPerceptionError(
                "executed trajectory endpoint disagrees with arm proprioception"
            )


def _pose_execution_error(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> tuple[float, float]:
    actual_xyz = _xyz(actual)
    expected_xyz = _xyz(expected)
    actual_rotation = actual["rotation"]
    expected_rotation = expected["rotation"]
    left = tuple(float(actual_rotation[key]) for key in ("w", "x", "y", "z"))
    right = tuple(float(expected_rotation[key]) for key in ("w", "x", "y", "z"))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        raise SensorPerceptionError("execution endpoint quaternion is invalid")
    dot = abs(
        sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    )
    return math.dist(actual_xyz, expected_xyz), 2.0 * math.acos(min(1.0, dot))
