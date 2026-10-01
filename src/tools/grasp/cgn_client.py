"""Contact-GraspNet wire client and explicit pose-frame conversions.

Matches CaP-X's /plan and /plan_point_clouds service; no model imports, candidate
selection, IK, or collision checks. Raw CGN poses are gripper-base poses. Their
local +Z approaches the object and +X is the jaw closing axis. All units are m.
"""
from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass
import io
import json
import math
import os
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

CAMERA_OPTICAL = "camera_optical"
ROBOT_BASE = "robot_base"
CGN_CONTACT_OFFSET_M = 0.1034
DEFAULT_SERVICE_URL = "http://127.0.0.1:8115"
Transport = Callable[[str, dict[str, Any], float], Mapping[str, Any]]


class CGNServiceError(RuntimeError):
    """The service could not complete an HTTP request (never an empty proposal set)."""

    def __init__(self, message, *, reason_code="cgn_service_unavailable", http_status=None):
        super().__init__(message)
        self.reason_code = reason_code
        self.http_status = http_status


class CGNResponseError(ValueError):
    """The service returned malformed arrays or an invalid pose contract."""


def _numeric(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return np.array(array, dtype=np.float64, copy=True)


def _frame(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError("frame must be a nonempty name without surrounding whitespace")
    return value


def _rigid(value: Any, name: str, *, batch: bool = False) -> np.ndarray:
    array = _numeric(value, name)
    if batch and array.shape == (0,):
        array = array.reshape(0, 4, 4)  # CaP-X emits np.array([]) for no candidates.
    if (batch and (array.ndim != 3 or array.shape[1:] != (4, 4))) or (
        not batch and array.shape != (4, 4)
    ):
        raise ValueError(f"{name} must have shape {'(N, 4, 4)' if batch else '(4, 4)'}")
    if not np.allclose(array[..., 3, :], [0, 0, 0, 1], atol=1e-6, rtol=0):
        raise ValueError(f"{name} must have homogeneous bottom row [0, 0, 0, 1]")
    rotation = array[..., :3, :3]
    if not np.allclose(rotation.swapaxes(-2, -1) @ rotation, np.eye(3),
                       atol=1e-4, rtol=0) or not np.allclose(
        np.linalg.det(rotation), 1, atol=1e-4, rtol=0
    ):
        raise ValueError(f"{name} rotations must be in SO(3)")
    return array


def _checked_results(instance: Any) -> None:
    poses = _rigid(instance.poses, "poses", batch=True)
    scores = _numeric(instance.scores, "scores")
    points = _numeric(instance.contact_points, "contact_points")
    if points.shape == (0,):
        points = points.reshape(0, 3)
    if scores.shape != (len(poses),):
        raise ValueError("scores must have shape (N,) matching poses")
    if points.shape != (len(poses), 3):
        raise ValueError("contact_points must have shape (N, 3) matching poses")
    _frame(instance.frame)
    for name, array in (("poses", poses), ("scores", scores), ("contact_points", points)):
        array.setflags(write=False)
        object.__setattr__(instance, name, array)


@dataclass(frozen=True)
class RawCGNGrasps:
    """Unshifted gripper-base poses and observed contact points in ``frame``."""

    poses: np.ndarray
    scores: np.ndarray
    contact_points: np.ndarray
    frame: str

    def __post_init__(self) -> None:
        _checked_results(self)


@dataclass(frozen=True)
class ContactCenterGrasps:
    """TCP contact-center poses; contact_points remain CGN's surface contacts.

    These are proposals, with reachability and collision status unchecked.
    """

    poses: np.ndarray
    scores: np.ndarray
    contact_points: np.ndarray
    frame: str

    def __post_init__(self) -> None:
        _checked_results(self)


def raw_to_contact_center(raw: RawCGNGrasps) -> ContactCenterGrasps:
    """Shift once by +0.1034 m along each raw pose's local approach (+Z)."""
    if not isinstance(raw, RawCGNGrasps):
        raise TypeError("raw_to_contact_center requires RawCGNGrasps; do not shift TCP twice")
    shift = np.eye(4)
    shift[2, 3] = CGN_CONTACT_OFFSET_M
    return ContactCenterGrasps(raw.poses @ shift, raw.scores, raw.contact_points, raw.frame)


def camera_to_robot_base(
    contact: ContactCenterGrasps, camera_to_base: Any,
) -> ContactCenterGrasps:
    """Apply base-from-optical-camera once, also transforming contact points.

    Point-cloud results already tagged robot_base must bypass this conversion.
    No native flange conversion is performed here.
    """
    if not isinstance(contact, ContactCenterGrasps):
        raise TypeError("camera_to_robot_base requires ContactCenterGrasps")
    if contact.frame != CAMERA_OPTICAL:
        raise ValueError(f"expected {CAMERA_OPTICAL} input, got {contact.frame}; refuse double conversion")
    transform = _rigid(camera_to_base, "camera_to_base")
    points = contact.contact_points @ transform[:3, :3].T + transform[:3, 3]
    return ContactCenterGrasps(transform @ contact.poses, contact.scores, points, ROBOT_BASE)


def _encode(array: np.ndarray) -> str:
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _decode(value: Any, name: str) -> np.ndarray:
    try:
        if not isinstance(value, str):
            raise ValueError("expected base64 text")
        binary = base64.b64decode(value, validate=True)
        if not binary.startswith(b"\x93NUMPY"):
            raise ValueError("expected a single .npy array")
        return np.load(io.BytesIO(binary), allow_pickle=False)
    except Exception as exc:
        raise CGNResponseError(f"invalid {name} numpy payload") from exc


def _http_post(url: str, payload: dict[str, Any], timeout_s: float) -> Mapping[str, Any]:
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        # Neither response bodies nor URLs/headers are agent-visible error messages.
        raise CGNServiceError(f"Contact-GraspNet returned HTTP {exc.code}",
                              reason_code="cgn_http_error", http_status=exc.code) from exc
    except (urllib.error.URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        timed_out = isinstance(reason, TimeoutError)
        raise CGNServiceError("Contact-GraspNet request timed out" if timed_out else
                              "Contact-GraspNet service connection failed",
                              reason_code="cgn_service_timeout" if timed_out else
                              "cgn_service_unavailable") from exc
    try:
        return json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CGNResponseError("Contact-GraspNet response is not valid JSON") from exc


def _integer(value: int, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _options(segmap_id: int, local_regions: bool, filter_grasps: bool,
             forward_passes: int, max_retries: int) -> dict[str, Any]:
    if type(local_regions) is not bool or type(filter_grasps) is not bool:
        raise ValueError("local_regions and filter_grasps must be booleans")
    return {"segmap_id": _integer(segmap_id, "segmap_id", 1),
            "local_regions": local_regions, "filter_grasps": filter_grasps,
            "forward_passes": _integer(forward_passes, "forward_passes", 1),
            "max_retries": _integer(max_retries, "max_retries", 0)}


class ContactGraspNetClient:
    """CPU-only client; transport(url, payload, timeout_s) can be injected.

    Default inference options match CaP-X's client, rather than the server's
    different defaults. No hidden retries, candidate ranking, or fallback occur.
    """

    def __init__(self, service_url: str | None = None, *, timeout_s: float = 120.0,
                 transport: Transport | None = None) -> None:
        url = service_url if service_url is not None else os.environ.get(
            "GRASPNET_SERVICE_URL", DEFAULT_SERVICE_URL)
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("service_url must be an HTTP(S) base URL without query or fragment")
        if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        self.service_url = url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self._transport = transport if transport is not None else _http_post

    def _request(self, endpoint: str, payload: dict[str, Any], frame: str) -> RawCGNGrasps:
        data = self._transport(self.service_url + endpoint, payload, self.timeout_s)
        if not isinstance(data, Mapping):
            raise CGNResponseError("Contact-GraspNet response must be a JSON object")
        try:
            return RawCGNGrasps(_decode(data["grasps_base64"], "grasps"),
                                _decode(data["scores_base64"], "scores"),
                                _decode(data["contact_pts_base64"], "contact_pts"), frame)
        except (KeyError, ValueError) as exc:
            raise CGNResponseError(f"invalid Contact-GraspNet response: {exc}") from exc

    def plan(self, depth: Any, cam_K: Any, segmap: Any, segmap_id: int = 1, *,
             local_regions: bool = True, filter_grasps: bool = True,
             skip_border_objects: bool = False, z_range: Any = None,
             forward_passes: int = 2, max_retries: int = 10) -> RawCGNGrasps:
        """Plan from optical-camera depth (H×W m), intrinsics, and integer labels."""
        depth_array = _numeric(depth, "depth")
        if depth_array.ndim != 2 or not depth_array.size or np.any(depth_array < 0):
            raise ValueError("depth must be a nonempty HxW array of nonnegative metres")
        intrinsics = _numeric(cam_K, "cam_K")
        if intrinsics.shape != (3, 3) or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0 or not np.allclose(
            intrinsics[2], [0, 0, 1], atol=1e-8, rtol=0
        ):
            raise ValueError("cam_K must be 3x3 camera intrinsics with positive focal lengths")
        labels = np.asarray(segmap)
        if labels.shape != depth_array.shape or labels.dtype.kind not in "biu" or np.any(labels < 0):
            raise ValueError("segmap must be an HxW nonnegative integer label map matching depth")
        bounds = _numeric([0.2, 2.0] if z_range is None else z_range, "z_range")
        if bounds.shape != (2,) or not 0 <= bounds[0] < bounds[1]:
            raise ValueError("z_range must contain increasing nonnegative metre bounds")
        if type(skip_border_objects) is not bool:
            raise ValueError("skip_border_objects must be a boolean")
        payload = _options(segmap_id, local_regions, filter_grasps, forward_passes, max_retries)
        payload.update(depth_base64=_encode(depth_array), cam_K_base64=_encode(intrinsics),
                       segmap_base64=_encode(labels), skip_border_objects=skip_border_objects,
                       z_range=bounds.tolist())
        return self._request("/plan", payload, CAMERA_OPTICAL)

    def plan_point_clouds(self, pc_full: Any, pc_segment: Any, *, input_frame: str,
                          segmap_id: int = 1, local_regions: bool = True,
                          filter_grasps: bool = True, forward_passes: int = 2,
                          max_retries: int = 10) -> RawCGNGrasps:
        """Plan from finite Nx3 points; output remains in explicit input_frame."""
        frame = _frame(input_frame)
        full = _numeric(pc_full, "pc_full")
        segment = _numeric(pc_segment, "pc_segment")
        for name, points in (("pc_full", full), ("pc_segment", segment)):
            if points.ndim != 2 or points.shape[1:] != (3,) or not len(points):
                raise ValueError(f"{name} must be a nonempty Nx3 array")
        payload = _options(segmap_id, local_regions, filter_grasps, forward_passes, max_retries)
        payload.update(pc_full_base64=_encode(full), pc_segment_base64=_encode(segment))
        return self._request("/plan_point_clouds", payload, frame)
