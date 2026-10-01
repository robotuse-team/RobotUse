"""CPU wire/frame contracts; synthetic service responses are not inference evidence."""
import base64
import io
import json
import urllib.error

import numpy as np
import pytest

from src.tools.grasp.cgn_client import (
    CAMERA_OPTICAL, ROBOT_BASE, CGNResponseError, CGNServiceError,
    ContactGraspNetClient, RawCGNGrasps, camera_to_robot_base, raw_to_contact_center,
)


def encode(array, *, allow_pickle=False):
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=allow_pickle)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode(value):
    return np.load(io.BytesIO(base64.b64decode(value)), allow_pickle=False)


def response(poses=None, scores=None, points=None):
    return {"grasps_base64": encode(np.eye(4)[None] if poses is None else poses),
            "scores_base64": encode([0.6] if scores is None else scores),
            "contact_pts_base64": encode([[0.1, 0.2, 0.3]] if points is None else points)}


def rgbd():
    return np.ones((2, 3)), np.array([[100, 0, 1], [0, 100, 1], [0, 0, 1]]), np.ones((2, 3), dtype=int)


def test_plan_wire_matches_capx_defaults_and_reads_url_at_creation(monkeypatch):
    monkeypatch.setenv("GRASPNET_SERVICE_URL", "http://cgn.example:8115/")
    calls = []

    def transport(url, payload, timeout):
        calls.append((url, payload, timeout))
        return response()

    raw = ContactGraspNetClient(timeout_s=17, transport=transport).plan(*rgbd())
    url, payload, timeout = calls[0]
    assert url == "http://cgn.example:8115/plan" and timeout == 17
    assert {key: value for key, value in payload.items() if not key.endswith("_base64")} == {
        "segmap_id": 1, "local_regions": True, "filter_grasps": True,
        "skip_border_objects": False, "z_range": [0.2, 2.0], "forward_passes": 2, "max_retries": 10,
    }
    for key, expected in zip(("depth_base64", "cam_K_base64", "segmap_base64"), rgbd()):
        np.testing.assert_array_equal(decode(payload[key]), expected)
    assert raw.frame == CAMERA_OPTICAL
    np.testing.assert_array_equal(raw.poses[0], np.eye(4))  # still raw, no silent TCP shift
    np.testing.assert_allclose(raw.scores, [0.6])
    assert not raw.poses.flags.writeable


def test_point_cloud_wire_keeps_explicit_base_frame_and_options():
    calls = []

    def transport(url, payload, timeout):
        calls.append((url, payload))
        return response()

    full = np.array([[0.5, 0, 0.1], [0.6, 0, 0.2]])
    raw = ContactGraspNetClient(transport=transport).plan_point_clouds(
        full, full[:1], input_frame=ROBOT_BASE, segmap_id=3,
        local_regions=False, filter_grasps=False, forward_passes=4, max_retries=2)
    url, payload = calls[0]
    assert url.endswith("/plan_point_clouds")
    assert set(payload) == {"pc_full_base64", "pc_segment_base64", "segmap_id",
                           "local_regions", "filter_grasps", "forward_passes", "max_retries"}
    assert payload["segmap_id"] == 3 and payload["forward_passes"] == 4
    assert not payload["filter_grasps"] and not payload["local_regions"]
    np.testing.assert_array_equal(decode(payload["pc_full_base64"]), full)
    np.testing.assert_array_equal(decode(payload["pc_segment_base64"]), full[:1])
    assert raw.frame == ROBOT_BASE
    with pytest.raises(ValueError, match="double conversion"):
        camera_to_robot_base(raw_to_contact_center(raw), np.eye(4))


def test_contact_offset_is_local_then_extrinsics_applied_exactly_once():
    pose = np.eye(4)
    pose[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]  # local Z points along camera X
    pose[:3, 3] = [0.1, 0.2, 0.3]
    raw = RawCGNGrasps(pose[None], np.array([0.9]), np.array([[0.2, 0.4, 0.6]]), CAMERA_OPTICAL)
    contact = raw_to_contact_center(raw)
    np.testing.assert_allclose(contact.poses[0, :3, 3], [0.2034, 0.2, 0.3])
    np.testing.assert_array_equal(raw.poses[0], pose)
    transform = np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])
    base = camera_to_robot_base(contact, transform)
    np.testing.assert_allclose(base.poses[0, :3, 3], [0.8, 2.2034, 3.3])
    np.testing.assert_allclose(base.poses[0, :3, 2], [0, 1, 0])
    np.testing.assert_allclose(base.contact_points, [[0.6, 2.2, 3.6]])
    assert base.frame == ROBOT_BASE
    with pytest.raises(TypeError, match="shift TCP twice"):
        raw_to_contact_center(contact)
    with pytest.raises(ValueError, match="double conversion"):
        camera_to_robot_base(base, transform)
    with pytest.raises(TypeError, match="ContactCenterGrasps"):
        camera_to_robot_base(raw, transform)


@pytest.mark.parametrize("poses,points", [(np.array([]), np.array([])),
                                          (np.empty((0, 4, 4)), np.empty((0, 3)))])
def test_empty_candidates_keep_canonical_shapes_through_conversions(poses, points):
    client = ContactGraspNetClient(transport=lambda *args: response(poses, [], points))
    raw = client.plan(*rgbd())
    base = camera_to_robot_base(raw_to_contact_center(raw), np.eye(4))
    assert base.poses.shape == (0, 4, 4)
    assert base.scores.shape == (0,) and base.contact_points.shape == (0, 3)


@pytest.mark.parametrize("field,replacement", [
    ("grasps_base64", "not base64"),
    ("scores_base64", encode([[0.5]])),
    ("scores_base64", encode([float("nan")])),
    ("contact_pts_base64", encode(np.empty((0, 3)))),
    ("grasps_base64", encode(np.diag([-1, 1, 1, 1])[None])),
    ("grasps_base64", encode(np.zeros((1, 4, 4)))),
    ("grasps_base64", encode(np.array([{"do_not_unpickle": True}], dtype=object), allow_pickle=True)),
])
def test_malformed_service_arrays_are_rejected(field, replacement):
    data = response()
    data[field] = replacement
    with pytest.raises(CGNResponseError):
        ContactGraspNetClient(transport=lambda *args: data).plan(*rgbd())


@pytest.mark.parametrize("data", [[], {}, {"grasps_base64": encode(np.eye(4)[None])}])
def test_malformed_response_structure_is_rejected(data):
    with pytest.raises(CGNResponseError):
        ContactGraspNetClient(transport=lambda *args: data).plan(*rgbd())


def test_invalid_input_is_rejected_before_transport():
    def no_transport(*args):
        pytest.fail("invalid input must not reach inference")

    client = ContactGraspNetClient(transport=no_transport)
    depth, intrinsics, labels = rgbd()
    for kwargs in ({"depth": [[float("nan")]]}, {"cam_K": np.zeros((3, 3))},
                   {"segmap": np.zeros((3, 2), dtype=int)}, {"segmap": labels.astype(float)},
                   {"forward_passes": True}, {"z_range": [2, 0.2]}, {"max_retries": -1}):
        arguments = {"depth": depth, "cam_K": intrinsics, "segmap": labels}
        arguments.update(kwargs)
        with pytest.raises(ValueError):
            client.plan(**arguments)
    with pytest.raises(ValueError):
        client.plan_point_clouds([[0, 0, 1]], [], input_frame=ROBOT_BASE)
    with pytest.raises(ValueError):
        client.plan_point_clouds([[0, 0, 1]], [[0, 0, 1]], input_frame="")
    with pytest.raises(ValueError, match="SO"):
        camera_to_robot_base(raw_to_contact_center(
            RawCGNGrasps(np.eye(4)[None], [0.5], [[0, 0, 1]], CAMERA_OPTICAL)), np.diag([2, 1, 1, 1]))


def test_standard_http_transport_sends_json_and_configured_timeout(monkeypatch):
    class Reply(io.BytesIO):
        pass

    def urlopen(request, *, timeout):
        assert request.full_url == "http://localhost:8115/plan"
        assert request.method == "POST" and request.get_header("Content-type") == "application/json"
        assert timeout == 9
        assert json.loads(request.data)["forward_passes"] == 2
        return Reply(json.dumps(response()).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert ContactGraspNetClient("http://localhost:8115", timeout_s=9).plan(*rgbd()).frame == CAMERA_OPTICAL


def test_http_failure_and_non_json_do_not_become_empty_candidates(monkeypatch):
    def unavailable(*args, **kwargs):
        raise urllib.error.URLError("unavailable")

    monkeypatch.setattr("urllib.request.urlopen", unavailable)
    with pytest.raises(CGNServiceError):
        ContactGraspNetClient().plan(*rgbd())
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(b"not JSON"))
    with pytest.raises(CGNResponseError, match="valid JSON"):
        ContactGraspNetClient().plan(*rgbd())
