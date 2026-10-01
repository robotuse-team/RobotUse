"""Single-camera gripper pose preview, independent of any calling workflow.

The caller owns view selection, capture, and what a region means. This module
sees one camera, one image point, one camera-relative orientation, and the
region that stops the selected ray. It returns the resolved base-frame pose so
that further cameras show the same pose without re-deriving it. Nothing here
establishes contact, reachability, or a collision-free trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from src.tools.pose_editor.visual_pose import (
    _NEAR, Mat3, Vec3, VisualPoseError, _Camera, _camera, _depth_anchor, _mm, _mv,
    _numbers, orientation_matrix, ray_aabb_intersection,
)
from src.tools.observation.preview import GripperOrientation, NormalizedPoint


def _as_camera(camera: Any):
    """Accept an already-built camera or any view carrying calibration."""
    if hasattr(camera, "origin") and hasattr(camera, "rotation"):
        return camera
    return _camera(camera)


@dataclass(frozen=True)
class BoxAnchor:
    """Axis-aligned region, in base metres, where the selected ray stops."""

    lower: Vec3
    upper: Vec3

    def __post_init__(self) -> None:
        lower = _numbers(self.lower, 3, "anchor box lower")
        upper = _numbers(self.upper, 3, "anchor box upper")
        if any(lo >= hi for lo, hi in zip(lower, upper)):
            raise VisualPoseError("anchor box must have positive extent on every axis")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True)
class DepthAnchor:
    """Registered metric optical Z for this camera, verified inside ``bounds``.

    The selected pixel is read as measured: no neighbor search, and no fallback
    to ``bounds`` when the measurement is absent or lands outside it.
    """

    depth: Any
    bounds: BoxAnchor

    def __post_init__(self) -> None:
        if self.depth is None:
            raise VisualPoseError("depth anchor requires registered metric depth")
        if not isinstance(self.bounds, BoxAnchor):
            raise VisualPoseError("depth anchor requires BoxAnchor bounds")


@dataclass(frozen=True)
class GripperPose:
    """A resolved base-frame gripper pose and how its anchor was obtained."""

    anchor_base: Vec3
    rotation_base: Mat3
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        anchor = _numbers(self.anchor_base, 3, "anchor")
        try:
            rotation = tuple(_numbers(row, 3, "rotation row") for row in self.rotation_base)
        except TypeError as exc:
            raise VisualPoseError("gripper rotation must be a finite 3 by 3 matrix") from exc
        if len(rotation) != 3:
            raise VisualPoseError("gripper rotation must be a finite 3 by 3 matrix")
        for i in range(3):
            for j in range(3):
                product = sum(rotation[k][i] * rotation[k][j] for k in range(3))
                if abs(product - (1. if i == j else 0.)) > 1e-6:
                    raise VisualPoseError("gripper rotation must be orthonormal")
        determinant = sum(
            rotation[0][i] * (rotation[1][(i+1) % 3] * rotation[2][(i+2) % 3]
                              - rotation[1][(i+2) % 3] * rotation[2][(i+1) % 3])
            for i in range(3))
        if abs(determinant - 1.) > 1e-6:
            raise VisualPoseError("gripper rotation must be right-handed")
        object.__setattr__(self, "anchor_base", anchor)
        object.__setattr__(self, "rotation_base", rotation)
        object.__setattr__(self, "provenance", dict(self.provenance))

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_base": list(self.anchor_base),
            "rotation_base_from_gripper": [list(row) for row in self.rotation_base],
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GripperPose":
        try:
            return cls(value["anchor_base"], value["rotation_base_from_gripper"],
                       value.get("provenance", {}))
        except (KeyError, TypeError) as exc:
            raise VisualPoseError(
                "pose requires anchor_base and rotation_base_from_gripper") from exc


CLOSEUP_LONG_SIDE_PX = 768


def _gripper_box(camera: Any, pose: "GripperPose", padding: float,
                 expected_open_width_m=None) -> tuple[int, int, int, int] | None:
    """Pixel bounds of the modelled gripper in this camera, expanded by padding.

    The mesh's own local bounds are projected rather than guessed from the
    anchor, so the crop follows the gripper however it is turned.
    """
    from src.tools.pose_editor.mesh import mesh_source_metadata

    bounds = mesh_source_metadata(expected_open_width_m)["local_bounds_m"]
    rotation, anchor = pose.rotation_base, pose.anchor_base
    pixels = []
    for index in range(8):
        local = tuple((bounds["max"] if index & (1 << axis) else bounds["min"])[axis] for axis in range(3))
        delta = _mv(rotation, local)
        corner = camera.pixel(camera.camera_point(tuple(anchor[i] + delta[i] for i in range(3))))
        if corner is not None:
            pixels.append(corner)
    if not pixels:
        return None
    xs, ys = [p[0] for p in pixels], [p[1] for p in pixels]
    margin = max(padding * max(max(xs) - min(xs), max(ys) - min(ys)), 24.)
    box = (max(0, int(min(xs) - margin)), max(0, int(min(ys) - margin)),
           min(camera.width, int(max(xs) + margin) + 1), min(camera.height, int(max(ys) + margin) + 1))
    return box if box[2] - box[0] >= 8 and box[3] - box[1] >= 8 else None


def _zoomed(camera: Any, rgb: Any, scene_depth: Any, box: tuple[int, int, int, int],
            factor: float):
    """A camera that sees only ``box``, sampled ``factor`` times more finely.

    Enlarging a finished render would only blur it. Re-rendering through scaled
    intrinsics draws the mesh at the closeup's own resolution, so the gripper
    stays sharp; the photographic background is still limited by the sensor.
    """
    import numpy as np
    from PIL import Image

    x0, y0, x1, y1 = box
    width, height = max(1, round((x1 - x0) * factor)), max(1, round((y1 - y0) * factor))
    k = camera.intrinsics
    intrinsics = ((k[0][0] * factor, k[0][1] * factor, (k[0][2] - x0) * factor),
                  (0., k[1][1] * factor, (k[1][2] - y0) * factor),
                  (0., 0., 1.))
    zoomed = _Camera(camera.view_id, width, height, intrinsics, camera.rotation, camera.origin)
    image = (rgb if isinstance(rgb, Image.Image) else Image.fromarray(np.asarray(rgb)))
    image = image.convert("RGB").crop(box).resize((width, height), Image.Resampling.LANCZOS)
    depth = None
    if scene_depth is not None:
        # Metric depth is never interpolated; samples are replicated instead.
        window = np.asarray(scene_depth)[y0:y1, x0:x1].astype("float32")
        depth = np.asarray(Image.fromarray(window, mode="F")
                           .resize((width, height), Image.Resampling.NEAREST))
    return zoomed, image, depth


def resolve_gripper_pose(camera: Any, point: NormalizedPoint,
                         orientation: GripperOrientation,
                         anchor: BoxAnchor | DepthAnchor) -> GripperPose:
    """Lift one image point and a camera-relative orientation into the base frame."""
    cam = _as_camera(camera)
    if not isinstance(point, NormalizedPoint):
        raise VisualPoseError("point must be a NormalizedPoint")
    if not isinstance(orientation, GripperOrientation):
        raise VisualPoseError("orientation must be a GripperOrientation")
    pixel = point.pixels(width=cam.width, height=cam.height)
    if isinstance(anchor, DepthAnchor):
        # Reuse the measured-depth anchor with its containment check intact: it
        # reads only ``depth`` from the view and the box from the target.
        anchored, provenance = _depth_anchor(
            SimpleNamespace(depth=anchor.depth), cam, pixel,
            SimpleNamespace(aabb_lower=anchor.bounds.lower, aabb_upper=anchor.bounds.upper))
        provenance = {"method": "measured_depth", "approximate": False, **provenance}
        provenance["camera_id"] = provenance.pop("view_id")
    elif isinstance(anchor, BoxAnchor):
        k = cam.intrinsics
        y = (pixel[1] - k[1][2]) / k[1][1]
        x = (pixel[0] - k[0][2] - k[0][1] * y) / k[0][0]
        anchored = ray_aabb_intersection(cam.origin, _mv(cam.rotation, (x, y, 1.)),
                                         anchor.lower, anchor.upper)
        provenance = {"method": "ray_box_intersection", "approximate": True,
                      "camera_id": cam.view_id, "selected_pixel_xy": list(pixel),
                      "box_lower": list(anchor.lower), "box_upper": list(anchor.upper)}
    else:
        raise VisualPoseError("anchor must be a BoxAnchor or a DepthAnchor")
    if cam.camera_point(anchored)[2] < _NEAR:
        raise VisualPoseError("resolved anchor lies before the camera near plane")
    provenance["orientation"] = orientation.to_dict()
    return GripperPose(anchored, _mm(cam.rotation, orientation_matrix(orientation)), provenance)


def gripper_pose_preview(camera: Any, rgb: Any, *, point: NormalizedPoint | None = None,
                         orientation: GripperOrientation | None = None,
                         anchor: BoxAnchor | DepthAnchor | None = None,
                         pose: GripperPose | None = None, opacity: float = .88,
                         scene_depth: Any = None, xray: bool = False, show_axes: bool = True,
                         closeup: bool = False, closeup_padding: float = .3,
                         output_path: str | Path | None = None,
                         expected_open_width_m: float | None = None,
                         candidate_axes_rotation_base: Any = None) -> tuple[Any, dict[str, Any]]:
    """Render the modelled gripper onto one camera's RGB; return image and manifest.

    Pass ``point``/``orientation``/``anchor`` to resolve a pose from this camera,
    or a ``pose`` resolved elsewhere to show that same pose from this one. The
    returned manifest carries the pose, so a second camera needs no re-derivation.

    Only the part of the gripper actually in front of the observed scene is drawn,
    so ``scene_depth`` is required. ``xray=True`` opts out of that depth test and
    draws the whole mesh over the image, which shows surfaces the camera cannot see.
    The gripper's own X, Y and Z are drawn by default, because a reviewer cannot
    judge an orientation without seeing which way it will turn; pass
    ``candidate_axes_rotation_base`` to label candidate-local tool axes while
    retaining the renderer's mesh display frame, or
    ``show_axes=False`` for a clean overlay. ``closeup`` crops
    to the drawn gripper with padding and enlarges it; the manifest records the
    crop so every reported pixel can still be mapped back to the full image.
    """
    from src.tools.pose_editor.mesh import mesh_source_metadata, render_mesh_overlay

    cam = _as_camera(camera)
    selection = (point, orientation, anchor)
    if pose is None:
        if any(value is None for value in selection):
            raise VisualPoseError("supply point, orientation and anchor, or a resolved pose")
        pose = resolve_gripper_pose(cam, point, orientation, anchor)
    elif any(value is not None for value in selection):
        raise VisualPoseError(
            "a resolved pose cannot be combined with point, orientation or anchor")
    elif not isinstance(pose, GripperPose):
        raise VisualPoseError("pose must be a GripperPose")
    if type(xray) is not bool:
        raise VisualPoseError("xray must be a boolean")
    if not xray and scene_depth is None:
        raise VisualPoseError("depth-tested rendering requires scene_depth; "
                              "pass xray=True to draw through the scene instead")
    crop = _gripper_box(cam, pose, closeup_padding, expected_open_width_m) if closeup else None
    target, source, measured = cam, rgb, scene_depth
    zoom = 1.
    if crop is not None:
        zoom = max(1., CLOSEUP_LONG_SIDE_PX / max(crop[2] - crop[0], crop[3] - crop[1]))
        target, source, measured = _zoomed(cam, rgb, scene_depth, crop, zoom)
    diagnostics: dict[str, Any] = {}
    try:
        image = render_mesh_overlay(source, target, pose.anchor_base, pose.rotation_base,
                                    opacity=opacity, scene_depth=None if xray else measured,
                                    show_axes=show_axes, diagnostics=diagnostics,
                                    expected_open_width_m=expected_open_width_m,
                                    axes_rotation_base=candidate_axes_rotation_base)
    except ValueError as exc:
        raise VisualPoseError(f"{cam.view_id}: {exc}") from exc
    anchor_pixel = cam.pixel(cam.camera_point(pose.anchor_base))
    geometry = mesh_source_metadata(expected_open_width_m)
    from PIL import ImageDraw
    opening_caption = ('NOMINAL OPEN reference; expected opening UNKNOWN' if expected_open_width_m is None else
                       f'EXPECTED total joint opening {expected_open_width_m*1000:.1f}mm (not measured)')
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, image.height-22, image.width, image.height), fill='white')
    draw.text((6, image.height-17), opening_caption, fill=(40, 55, 75))
    geometry["scene_occlusion"] = not xray
    manifest = {
        "schema_version": 1,
        "camera_id": cam.view_id,
        "image_size": list(image.size),
        "camera_image_size": [cam.width, cam.height],
        "closeup": None if crop is None else {
            "source_box_xyxy": list(crop), "output_size": list(image.size),
            "zoom": zoom, "rendering": "mesh re-rendered through scaled intrinsics; "
                                       "background resampled from the sensor image"},
        "show_axes": show_axes,
        "axes": {"frame": "gripper_display" if candidate_axes_rotation_base is None else "candidate_local",
                 "rotation_base": [list(row) for row in (pose.rotation_base if candidate_axes_rotation_base is None
                                                           else candidate_axes_rotation_base)]},
        "coordinate_spaces": {"anchor_pixel": "full camera image pixels",
                              "closeup.source_box_xyxy": "full camera image pixels"},
        "pose": pose.to_dict(),
        "anchor_pixel": list(anchor_pixel) if anchor_pixel is not None else None,
        "approximate_anchor": bool(pose.provenance.get("approximate", True)),
        "xray": xray,
        "scene_occlusion": not xray,
        "occlusion": None if xray else dict(diagnostics),
        "gripper": geometry,
        "opening_caption": opening_caption,
        "executable_pose": False,
    }
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)
        manifest["output_path"] = str(output)
    return image, manifest
