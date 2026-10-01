"""Capture measured RGB-D and calibration for the point-prompt SAM2 adapter.

This module never calls a detector, benchmark oracle, or simulator world-state API.
"""


from __future__ import annotations


import hashlib


import json


import math


from pathlib import Path


from typing import Any, Mapping, Sequence


from src.tools.perception.geometry import (
    CALIBRATION_ARTIFACT_SCHEMA,
    METRIC_DEPTH_ARTIFACT_SCHEMA,
    METRIC_DEPTH_MAGIC,
    MaskPayload,
    ObservationProfileId,
    PerceptionToolId,
    PixelBox,
    PrivatePerceptionRegistry,
    RigidTransform,
    SensorActorState,
    SensorFramePayload,
    SensorPerceptionError,
    VisualHypothesisPayload,
    make_hypothesis_payload,
    make_mask_payload,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quaternion_camera_to_base(pose: Mapping[str, Any]) -> RigidTransform:
    """Convert the connector's private wxyz camera pose to a rigid transform."""

    position = pose["position"]
    rotation = pose["rotation"]
    w, x, y, z = (float(rotation[key]) for key in ("w", "x", "y", "z"))
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        raise SensorPerceptionError("camera quaternion is zero")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    matrix = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    return RigidTransform(
        rotation=matrix,
        translation=tuple(float(position[key]) for key in ("x", "y", "z")),
    )


def capture_connector_rgbd(
    *,
    connector: Any,
    output_dir: str | Path,
    capture_revision: int,
    actor_profile: ObservationProfileId,
) -> tuple[SensorFramePayload, ...]:
    """Capture synchronized connector RGB-D without consulting simulator world state."""

    try:
        import imageio.v3 as iio
        import numpy as np
    except ImportError as exc:  # pragma: no cover - runtime-only dependency boundary
        raise SensorPerceptionError("RGB-D runtime dependencies are unavailable") from exc
    if isinstance(capture_revision, bool) or not isinstance(capture_revision, int) or capture_revision < 1:
        raise SensorPerceptionError("capture_revision must be a positive integer")
    profile = ObservationProfileId(actor_profile)
    observation = connector.get_observation()
    cameras = tuple(observation.get("cameras", ()))
    if not cameras:
        raise SensorPerceptionError("connector returned no camera observations")
    if profile is ObservationProfileId.MULTIVIEW_TRACK:
        selected = cameras[:2]
    else:
        selected = (
            next(
                (
                    camera
                    for camera in cameras
                    if "wrist" not in str(camera.get("name", "")).casefold()
                    and "eye_in_hand" not in str(camera.get("name", "")).casefold()
                ),
                cameras[0],
            ),
        )
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    frames: list[SensorFramePayload] = []
    for index, camera in enumerate(selected, start=1):
        required = {"rgb", "depth", "intrinsics", "pose"}
        if not required <= set(camera):
            raise SensorPerceptionError("camera lacks synchronized RGB-D/calibration fields")
        rgb = np.ascontiguousarray(camera["rgb"], dtype=np.uint8)
        depth = np.ascontiguousarray(camera["depth"], dtype=np.float32)
        intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2]:
            raise SensorPerceptionError("camera RGB/depth shapes are invalid")
        if intrinsics.shape != (3, 3):
            raise SensorPerceptionError("camera intrinsics shape is invalid")
        view_id = str(camera.get("name") or f"view{index}").replace("-", "_")
        if not view_id or any(not (character.isalnum() or character in "_.:") for character in view_id):
            view_id = f"view{index}"
        prefix = f"sensor_{capture_revision:03d}_{view_id}"
        rgb_path = destination / f"{prefix}.png"
        depth_path = destination / f"{prefix}_depth_preview.png"
        depth_metric_path = destination / f"{prefix}_depth.f32le"
        calibration_path = destination / f"{prefix}_calibration.json"
        iio.imwrite(rgb_path, rgb)
        finite = np.isfinite(depth) & (depth > 0.02) & (depth < 3.0)
        preview = np.zeros(depth.shape, dtype=np.uint8)
        if finite.any():
            near, far = np.percentile(depth[finite], (2.0, 98.0))
            span = max(float(far - near), 1e-6)
            preview[finite] = np.clip(
                (1.0 - (depth[finite] - near) / span) * 255.0,
                0.0,
                255.0,
            ).astype(np.uint8)
        iio.imwrite(depth_path, preview)
        depth_header = {
            "schema_version": METRIC_DEPTH_ARTIFACT_SCHEMA,
            "dtype": "float32-le",
            "order": "C",
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
        }
        depth_le = np.ascontiguousarray(depth, dtype=np.dtype("<f4"))
        depth_metric_path.write_bytes(
            METRIC_DEPTH_MAGIC
            + json.dumps(
                depth_header,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
            + depth_le.tobytes(order="C")
        )
        camera_to_base = quaternion_camera_to_base(camera["pose"])
        calibration = {
            "schema_version": CALIBRATION_ARTIFACT_SCHEMA,
            "view_id": view_id,
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
            "intrinsics": [[float(value) for value in row] for row in intrinsics],
            "camera_to_base": camera_to_base.to_dict(),
        }
        calibration_path.write_text(
            json.dumps(
                calibration,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        rgb_digest = _sha256(rgb_path)
        depth_preview_digest = _sha256(depth_path)
        depth_digest = _sha256(depth_metric_path)
        calibration_digest = _sha256(calibration_path)
        frame_id = "frame:" + hashlib.sha256(
            json.dumps(
                {
                    "capture_revision": capture_revision,
                    "depth_sha256": depth_digest,
                    "calibration_sha256": calibration_digest,
                    "rgb_sha256": rgb_digest,
                    "view_id": view_id,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:20]
        frames.append(
            SensorFramePayload(
                frame_id=frame_id,
                view_id=view_id,
                width=int(rgb.shape[1]),
                height=int(rgb.shape[0]),
                rgb=rgb,
                depth_m=depth,
                intrinsics=tuple(tuple(float(value) for value in row) for row in intrinsics),
                camera_to_base=camera_to_base,
                rgb_path=str(rgb_path),
                depth_evidence_path=str(depth_path),
                rgb_sha256=rgb_digest,
                depth_sha256=depth_digest,
                depth_metric_path=str(depth_metric_path),
                depth_evidence_sha256=depth_preview_digest,
                calibration_path=str(calibration_path),
                calibration_sha256=calibration_digest,
            )
        )
    return tuple(frames)
