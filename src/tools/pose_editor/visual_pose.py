"""Calibrated visual gripper annotations with explicit anchor provenance.

The model chooses only normalized image coordinates and a bounded orientation.
The default camera ray/AABB intersection supplies an approximate annotation anchor.
Motion review opts into measured metric depth without an AABB fallback. Neither
mode alone establishes contact, reachability, or a collision-free trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.tools.observation.preview import GripperOrientation, VisualPointSelection, VisualWaypointError

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]
_NEAR = 1e-4
_AXIS_COLORS = {"x": (255, 77, 89), "y": (67, 225, 130), "z": (69, 155, 255)}


class VisualPoseError(VisualWaypointError):
    """The annotation cannot be anchored or projected with valid calibration."""


def _numbers(values: Sequence[Any], size: int, context: str) -> tuple[float, ...]:
    try:
        if len(values) != size or any(isinstance(v, bool) for v in values):
            raise ValueError
        result = tuple(float(v) for v in values)
        if not all(math.isfinite(v) for v in result):
            raise ValueError
        return result
    except (ValueError, TypeError, OverflowError) as exc:
        raise VisualPoseError(f"{context} must contain {size} finite numbers") from exc


def _mv(matrix: Mat3, vector: Sequence[float]) -> Vec3:
    return tuple(sum(matrix[i][j] * vector[j] for j in range(3)) for i in range(3))


def _mm(left: Mat3, right: Mat3) -> Mat3:
    return tuple(tuple(sum(left[i][k] * right[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def orientation_matrix(orientation: GripperOrientation) -> Mat3:
    """Map gripper-local XYZ into selected optical camera XYZ (Rz Ry Rx)."""
    if not isinstance(orientation, GripperOrientation):
        raise VisualPoseError("orientation must be a GripperOrientation")
    roll, pitch, yaw = map(math.radians, (orientation.roll_deg, orientation.pitch_deg, orientation.yaw_deg))
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


@dataclass(frozen=True)
class _Camera:
    view_id: str
    width: int
    height: int
    intrinsics: Mat3
    rotation: Mat3
    origin: Vec3

    def camera_point(self, point: Sequence[float]) -> Vec3:
        delta = tuple(point[i] - self.origin[i] for i in range(3))
        return tuple(sum(self.rotation[j][i] * delta[j] for j in range(3)) for i in range(3))

    def pixel(self, point: Sequence[float]) -> tuple[float, float] | None:
        if point[2] < _NEAR:
            return None
        homogeneous = _mv(self.intrinsics, point)
        result = homogeneous[0] / homogeneous[2], homogeneous[1] / homogeneous[2]
        return result if all(math.isfinite(v) for v in result) else None


def _camera(view: Any) -> _Camera:
    if type(view.width) is not int or type(view.height) is not int or view.width < 2 or view.height < 2:
        raise VisualPoseError(f"{view.view_id}: invalid image dimensions")
    try:
        k = tuple(_numbers(row, 3, "intrinsics row") for row in view.intrinsics)
        if len(k) != 3 or k[0][0] <= 0 or k[1][1] <= 0 or abs(k[1][0]) > 1e-9 or any(abs(k[2][i] - (1.0 if i == 2 else 0.0)) > 1e-9 for i in range(3)):
            raise VisualPoseError("intrinsics must be a calibrated pinhole matrix with positive focal lengths")
        pose = view.pose
        origin = _numbers([pose["position"][axis] for axis in ("x", "y", "z")], 3, "camera position")
        q = _numbers([pose["rotation"][axis] for axis in ("w", "x", "y", "z")], 4, "camera quaternion")
        norm = math.sqrt(sum(v * v for v in q))
        if norm <= 1e-12 or not math.isfinite(norm):
            raise VisualPoseError("camera quaternion must be nonzero and finite")
        w, x, y, z = (v / norm for v in q)
    except (KeyError, TypeError, AttributeError) as exc:
        raise VisualPoseError(f"{view.view_id}: calibrated intrinsics and camera-to-base pose are required") from exc
    rotation = (
        (1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
        (2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
        (2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)),
    )
    return _Camera(view.view_id, view.width, view.height, k, rotation, origin)


def ray_aabb_intersection(origin: Sequence[float], direction: Sequence[float], lower: Sequence[float], upper: Sequence[float]) -> Vec3:
    """Nearest strictly positive ray/bounding-box hit; raises on a miss."""
    origin = _numbers(origin, 3, "ray origin")
    direction = _numbers(direction, 3, "ray direction")
    lower, upper = _numbers(lower, 3, "AABB lower"), _numbers(upper, 3, "AABB upper")
    if any(lo >= hi for lo, hi in zip(lower, upper)):
        raise VisualPoseError("AABB must have positive extent on every axis")
    norm = math.sqrt(sum(v * v for v in direction))
    if norm < 1e-12:
        raise VisualPoseError("ray direction must be nonzero")
    direction = tuple(v / norm for v in direction)
    enter, leave = -math.inf, math.inf
    for i in range(3):
        if abs(direction[i]) < 1e-12:
            if not lower[i] <= origin[i] <= upper[i]:
                raise VisualPoseError("selected pixel ray misses target AABB")
            continue
        a, b = (lower[i] - origin[i]) / direction[i], (upper[i] - origin[i]) / direction[i]
        enter, leave = max(enter, min(a, b)), min(leave, max(a, b))
        if enter > leave:
            raise VisualPoseError("selected pixel ray misses target AABB")
    distance = enter if enter > 1e-10 else leave
    if not math.isfinite(distance) or distance <= 1e-10:
        raise VisualPoseError("selected pixel ray has no positive target AABB intersection")
    return tuple(origin[i] + distance * direction[i] for i in range(3))


def _clip_image(a: Sequence[float], b: Sequence[float], width: int, height: int) -> list[list[float]] | None:
    """Liang-Barsky clip before drawing, including extremely off-screen endpoints."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    lo, hi = 0.0, 1.0
    for p, q in ((-dx, a[0]), (dx, width - 1 - a[0]), (-dy, a[1]), (dy, height - 1 - a[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return None
        elif p < 0:
            lo = max(lo, q / p)
        else:
            hi = min(hi, q / p)
        if lo > hi:
            return None
    return [[max(0., min(width - 1., a[0] + t * dx)), max(0., min(height - 1., a[1] + t * dy))] for t in (lo, hi)]


def _segment(camera: _Camera, a: Sequence[float], b: Sequence[float]) -> list[list[float]] | None:
    a, b = camera.camera_point(a), camera.camera_point(b)
    if a[2] < _NEAR and b[2] < _NEAR:
        return None
    if a[2] < _NEAR or b[2] < _NEAR:
        t = (_NEAR - a[2]) / (b[2] - a[2])
        intersection = tuple(a[i] + t * (b[i] - a[i]) for i in range(3))
        intersection = (intersection[0], intersection[1], _NEAR)
        a, b = (intersection, b) if a[2] < _NEAR else (a, intersection)
    p, q = camera.pixel(a), camera.pixel(b)
    return None if p is None or q is None else _clip_image(p, q, camera.width, camera.height)


def _visible(pixel: Sequence[float] | None, camera: _Camera) -> bool:
    return pixel is not None and 0 <= pixel[0] <= camera.width - 1 and 0 <= pixel[1] <= camera.height - 1


def _font(size: int):
    from PIL import ImageFont
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _depth_anchor(view: Any, camera: _Camera, pixel: Sequence[float], body: Any) -> tuple[Vec3, dict[str, Any]]:
    """Unproject one observed pixel; never search neighbors or substitute a box."""
    import numpy as np

    depth = getattr(view, "depth", None)
    if depth is None:
        raise VisualPoseError("selected view has no metric depth for the approved pose")
    array = np.asarray(depth)
    if array.shape == (camera.height, camera.width, 1):
        array = array[..., 0]
    if array.shape != (camera.height, camera.width) or array.dtype.kind not in "iuf":
        raise VisualPoseError("selected metric depth dimensions/type disagree with calibration")
    # Image-coordinate selections are continuous; depth is measured at pixel
    # centers. Record this explicit <= half-pixel quantization and use that same
    # pixel for unprojection instead of mixing a neighboring depth with a ray.
    px, py = (int(math.floor(value + .5)) for value in pixel)
    if not (0 <= px < camera.width and 0 <= py < camera.height):
        raise VisualPoseError("selected depth pixel is outside the image")
    measured = float(array[py, px])
    if not math.isfinite(measured) or measured <= 0.:
        raise VisualPoseError("selected pixel metric depth must be finite and positive")
    k = camera.intrinsics
    y = (py - k[1][2]) / k[1][1]
    x = (px - k[0][2] - k[0][1] * y) / k[0][0]
    delta = _mv(camera.rotation, (x * measured, y * measured, measured))
    anchor = tuple(camera.origin[i] + delta[i] for i in range(3))
    lower, upper = _numbers(body.aabb_lower, 3, "target AABB lower"), _numbers(body.aabb_upper, 3, "target AABB upper")
    tolerance = .005
    if any(lo >= hi for lo, hi in zip(lower, upper)):
        raise VisualPoseError("target AABB must have positive extent on every axis")
    if any(not lower[i] - tolerance <= anchor[i] <= upper[i] + tolerance for i in range(3)):
        raise VisualPoseError("selected depth point is outside the selected target AABB")
    obb = getattr(body, "obb", None)
    if obb is not None:
        center = _numbers(obb.center, 3, "target OBB center")
        half = _numbers(obb.half_extents, 3, "target OBB half extents")
        axes = tuple(_numbers(axis, 3, "target OBB axis") for axis in obb.axes)
        if len(axes) != 3 or any(value <= 0 for value in half):
            raise VisualPoseError("target OBB requires three positive half extents and axes")
        local = tuple(sum((anchor[j]-center[j]) * axes[i][j] for j in range(3)) for i in range(3))
        if any(abs(local[i]) > half[i] + tolerance for i in range(3)):
            raise VisualPoseError("selected depth point is outside the selected target OBB")
    record = {
        "view_id": camera.view_id, "pixel_xy": [px, py], "selected_pixel_xy": list(pixel),
        "sampling": "nearest_pixel_center_no_search", "depth_m": measured,
        "depth_convention": "metric_optical_camera_z", "target_tolerance_m": tolerance,
        "target_geometry_check": "sensor_obb_and_aabb" if obb is not None else "sensor_aabb",
    }
    for attribute in ("depth_path", "depth_sha256"):
        value = getattr(view, attribute, None)
        if value is not None:
            record[attribute] = str(value)
    return anchor, record


def render_visual_waypoint(scene, selection, body_name, output_path, *, anchor_method="depth", max_image_dimension=1600):
    """Point-only review: validate its backend anchor, without inventing a hand orientation."""
    from PIL import Image, ImageDraw
    if selection.orientation is not None:
        raise VisualPoseError("waypoint stage must not contain an orientation")
    views = {v.view_id: v for v in scene.views}
    if set(views) != {"front", "wrist"}:
        raise VisualPoseError("calibrated front and wrist views are required")
    cameras = {name: _camera(view) for name, view in views.items()}
    camera = cameras[selection.view_id]
    pixel = selection.point.pixels(width=camera.width, height=camera.height)
    body = scene.bodies[body_name]
    if anchor_method == "depth":
        anchor, depth_sample = _depth_anchor(views[selection.view_id], camera, pixel, body)
    elif anchor_method == "aabb":
        k = camera.intrinsics
        y = (pixel[1]-k[1][2])/k[1][1]
        x = (pixel[0]-k[0][2]-k[0][1]*y)/k[0][0]
        anchor = ray_aabb_intersection(camera.origin, _mv(camera.rotation, (x,y,1.)), body.aabb_lower, body.aabb_upper)
        depth_sample = None
    else:
        raise VisualPoseError("unknown waypoint anchor method")
    width = sum(v.width for v in views.values())
    height = max(v.height for v in views.values())
    canvas = Image.new("RGB", (width, height+110), (12,22,34))
    draw = ImageDraw.Draw(canvas)
    offset = 0
    for name in ("front", "wrist"):
        view, cam = views[name], cameras[name]
        image = view.rgb.copy().convert("RGB") if isinstance(view.rgb, Image.Image) else Image.fromarray(view.rgb).convert("RGB")
        if image.size != (cam.width, cam.height):
            raise VisualPoseError("waypoint RGB dimensions disagree with calibration")
        xy = cam.pixel(cam.camera_point(anchor))
        if _visible(xy, cam):
            x,y = xy
            ImageDraw.Draw(image).ellipse((x-11,y-11,x+11,y+11), outline=(255,91,180), width=3)
        canvas.paste(image, (offset,50))
        draw.text((offset+15,12), f"{name.upper()} / WAYPOINT ONLY / {body_name}", font=_font(20), fill="white")
        offset += view.width
    draw.text((15,height+65), f"{selection.view_id} u={selection.point.u:g}, v={selection.point.v:g} | keep: choose orientation next / revise: choose another point", font=_font(18), fill="white")
    canvas.thumbnail((max_image_dimension,max_image_dimension),Image.Resampling.LANCZOS)
    output = Path(output_path)
    output.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(output)
    return {"stage": "waypoint", "selection": selection.to_dict(), "anchor_base": list(anchor),
            "depth_sample": depth_sample, "output_path": str(output), "image_size": list(canvas.size),
            "executable_pose": False, "orientation_selected": False}


def render_visual_pose(scene: Any, selection: VisualPointSelection, body_name: str, output_path: str | Path, *, render_style: str = "solid", max_image_dimension: int | None = None, anchor_method: str = "aabb", scene_occlusion: bool = False, include_xray: bool = False) -> dict[str, Any]:
    """Write the agent's PNG: full-view references above solid pose closeups.

    Per-view files preserve original camera dimensions. Normalized selections
    always refer to full images, never resized crops. The wireframe style is
    retained for diagnostics. Outputs are annotations, not motion commands.
    An optional final-composite size limit preserves all calibrated per-view
    pixels and crop coordinates; only the delivered composite is downscaled.
    ``anchor_method='depth'`` requires valid metric depth at the selected pixel
    inside the selected sensor geometry; failure never falls back to the AABB.
    With scene occlusion, agent images contain references and depth-tested pose
    closeups only. ``include_xray=True`` adds the optional third diagnostic row.
    """
    from PIL import Image, ImageDraw

    if max_image_dimension is not None and (type(max_image_dimension) is not int or max_image_dimension < 256):
        raise VisualPoseError("max_image_dimension must be None or an integer >= 256")
    if render_style not in {"solid", "wireframe"}:
        raise VisualPoseError("render_style must be solid or wireframe")
    if scene_occlusion and render_style != "solid":
        raise VisualPoseError("scene occlusion requires the solid mesh renderer")
    if type(include_xray) is not bool:
        raise VisualPoseError("include_xray must be a boolean")
    show_xray = scene_occlusion and include_xray
    if anchor_method not in {"aabb", "depth"}:
        raise VisualPoseError("anchor_method must be aabb or depth")
    if selection.orientation is None:
        raise VisualPoseError("pose visualization requires an orientation")
    views = {view.view_id: view for view in scene.views if view.view_id in {"front", "wrist"}}
    if set(views) != {"front", "wrist"}:
        raise VisualPoseError("calibrated front and wrist views are required")
    cameras = {name: _camera(view) for name, view in views.items()}
    selected = cameras[selection.view_id]
    pixel = selection.point.pixels(width=selected.width, height=selected.height)
    k = selected.intrinsics
    y = (pixel[1] - k[1][2]) / k[1][1]
    x = (pixel[0] - k[0][2] - k[0][1] * y) / k[0][0]
    try:
        body = scene.bodies[body_name]
    except KeyError as exc:
        raise VisualPoseError(f"selected body is absent: {body_name}") from exc
    depth_sample = None
    if anchor_method == "depth":
        anchor, depth_sample = _depth_anchor(views[selection.view_id], selected, pixel, body)
    else:
        anchor = ray_aabb_intersection(selected.origin, _mv(selected.rotation, (x, y, 1.)), body.aabb_lower, body.aabb_upper)
    if selected.camera_point(anchor)[2] < _NEAR:
        raise VisualPoseError("selected target anchor lies before the camera near plane")
    rotation = _mm(selected.rotation, orientation_matrix(selection.orientation))
    from src.tools.pose_editor.mesh import proxy_geometry
    proxy = proxy_geometry()
    axis_length, jaw_width, finger_length = (
        proxy["axis_length_m"], proxy["jaw_width_m"], proxy["finger_length_m"])

    def world(local: Sequence[float]) -> Vec3:
        delta = _mv(rotation, local)
        return tuple(anchor[i] + delta[i] for i in range(3))

    vertices: list[Vec3] = []
    edges: list[tuple[int, int]] = []
    def box(lower: Sequence[float], upper: Sequence[float]) -> None:
        offset = len(vertices)
        for i in range(8):
            vertices.append(world(tuple(upper[j] if i & (1 << j) else lower[j] for j in range(3))))
        edges.extend((offset + i, offset + (i ^ (1 << j))) for i in range(8) for j in range(3) if not i & (1 << j))
    for lower, upper in proxy["boxes"]:
        box(lower, upper)
    tips = {axis: world(tuple(axis_length if j == i else 0. for j in range(3))) for i, axis in enumerate(("x", "y", "z"))}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered, projections, xrays, depth_stats = {}, {}, {}, {}
    for name in ("front", "wrist"):
        view, camera = views[name], cameras[name]
        if view.rgb is None:
            raise VisualPoseError(f"{name}: RGB evidence is required")
        image = (view.rgb.copy().convert("RGB") if isinstance(view.rgb, Image.Image) else Image.fromarray(view.rgb).convert("RGB"))
        if image.size != (camera.width, camera.height):
            raise VisualPoseError(f"{name}: RGB dimensions disagree with calibration")
        if render_style == "solid":
            from src.tools.pose_editor.mesh import render_mesh_overlay
            scene_depth = getattr(view, "depth", None) if scene_occlusion else None
            if scene_occlusion and scene_depth is None:
                raise VisualPoseError(f"{name}: registered metric depth required for scene occlusion")
            companion, stats = [], {}
            try:
                image = render_mesh_overlay(image, camera, anchor, rotation, opacity=.88,
                    scene_depth=scene_depth, diagnostics=stats, xray_output=companion if show_xray else None)
            except ValueError as exc:
                raise VisualPoseError(f"{name}: {exc}") from exc
            if scene_occlusion:
                xray_path = None
                if show_xray:
                    xrays[name] = companion[0]
                    xray_path = output.with_name(f"{output.stem}_{name}_xray.png")
                    xrays[name].save(xray_path)
                depth_stats[name] = {**stats, "xray_path": str(xray_path) if xray_path else None,
                    "depth_path": getattr(view, "depth_path", None),
                    "depth_sha256": getattr(view, "depth_sha256", None)}
        draw = ImageDraw.Draw(image)
        gripper_segments = [line for i, j in edges if (line := _segment(camera, vertices[i], vertices[j])) is not None]
        for line in gripper_segments if render_style == "wireframe" else ():
            draw.line([tuple(p) for p in line], fill=(12, 17, 24), width=6)
            draw.line([tuple(p) for p in line], fill=(255, 216, 94), width=3)
        anchor_camera = camera.camera_point(anchor)
        anchor_pixel = camera.pixel(anchor_camera)
        axis_records = {}
        for axis, tip in tips.items():
            line = _segment(camera, anchor, tip)
            tip_pixel = camera.pixel(camera.camera_point(tip))
            axis_records[axis] = {"tip_pixel": list(tip_pixel) if tip_pixel is not None else None, "visible_segment": line}
            if line is not None and render_style == "wireframe":
                a, b = line
                draw.line([tuple(a), tuple(b)], fill=(12, 17, 24), width=7)
                draw.line([tuple(a), tuple(b)], fill=_AXIS_COLORS[axis], width=4)
                length = math.dist(a, b)
                # Arrowheads only indicate the real positive tip, never a clipped endpoint.
                if _visible(tip_pixel, camera) and length >= 8:
                    dx, dy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
                    draw.polygon([tuple(b), (b[0] - dx*10 - dy*5, b[1] - dy*10 + dx*5), (b[0] - dx*10 + dy*5, b[1] - dy*10 - dx*5)], fill=_AXIS_COLORS[axis])
                    tx, ty = max(2, min(camera.width - 18, b[0] + 5)), max(2, min(camera.height - 22, b[1] - 22))
                    draw.text((tx, ty), axis.upper(), font=_font(18), fill=_AXIS_COLORS[axis], stroke_width=2, stroke_fill=(0, 0, 0))
        if _visible(anchor_pixel, camera):
            ax, ay = anchor_pixel
            draw.ellipse((ax-4, ay-4, ax+4, ay+4), fill=(255, 255, 255), outline=(12, 17, 24), width=2)
        if name == selection.view_id:
            px, py = pixel
            draw.ellipse((px-11, py-11, px+11, py+11), outline=(255, 91, 180), width=3)
        if not _visible(anchor_pixel, camera):
            draw.text((12, 12), "Anchor outside this view / behind camera", font=_font(15), fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        per_view = output.with_name(f"{output.stem}_{name}{output.suffix or '.png'}")
        image.save(per_view)
        rendered[name] = image
        projections[name] = {
            "image_path": str(per_view),
            "width": camera.width, "height": camera.height,
            "anchor_pixel": list(anchor_pixel) if anchor_pixel is not None else None,
            "anchor_visible": _visible(anchor_pixel, camera),
            "anchor_depth_m": anchor_camera[2],
            "selected_pixel": list(pixel) if name == selection.view_id else None,
            "axes": axis_records, "gripper_segments": gripper_segments,
        }
    width = sum(image.width for image in rendered.values())
    height = max(image.height for image in rendered.values())
    orientation = selection.orientation
    crop_records = {}
    if render_style == "solid":
        header, detail_header, detail_height, footer = 70, 52, max(160, int(height * .88)), 105
        extra_height = detail_header + detail_height if show_xray else 0
        canvas = Image.new("RGB", (width, header + height + detail_header + detail_height + extra_height + footer), (12, 22, 34))
        draw = ImageDraw.Draw(canvas)
        offset = 0
        for name in ("front", "wrist"):
            view, camera = views[name], cameras[name]
            source = view.rgb.copy().convert("RGB") if isinstance(view.rgb, Image.Image) else Image.fromarray(view.rgb).convert("RGB")
            marker = ImageDraw.Draw(source)
            xy = projections[name]["anchor_pixel"]
            if _visible(xy, camera):
                x, y = xy
                marker.ellipse((x-10,y-10,x+10,y+10), outline=(255,91,180), width=3)
            draw.text((offset + 15, 10), f"{name.upper()} / FULL-VIEW REFERENCE", font=_font(20), fill=(225,239,249))
            draw.text((offset + 15, 40), f"{body_name} | " + ("selected point" if name == selection.view_id else "projected anchor"), font=_font(14), fill=(143,172,190))
            canvas.paste(source, (offset, header))
            detail_y = header + height
            detail_label = "SCENE DEPTH / PROPOSED PANDA POSE" if scene_occlusion else "PROPOSED PANDA POSE"
            if scene_occlusion and depth_stats[name]["unknown_depth_samples"]:
                stats = depth_stats[name]
                missing = 100 * stats["unknown_depth_samples"] / max(1, stats["mesh_samples"])
                detail_label = f"SCENE DEPTH / {missing:.1f}% HAND DEPTH UNKNOWN" + (" - CHECK XRAY" if show_xray else " - HIDDEN")
            draw.text((offset + 15, detail_y + 16), f"{name.upper()} DETAIL / {detail_label}", font=_font(18), fill=(135,236,208))
            if _visible(xy, camera):
                # Square-ish target crop, then preserve aspect ratio in the panel.
                crop_w = min(view.width, max(100, view.width * .44))
                crop_h = min(view.height, crop_w * detail_height / view.width)
                crop_w = crop_h * view.width / detail_height
                left = max(0, min(view.width - crop_w, xy[0] - crop_w / 2))
                top = max(0, min(view.height - crop_h, xy[1] - crop_h / 2))
                crop = (int(left), int(top), int(left + crop_w), int(top + crop_h))
                detail = rendered[name].crop(crop).resize((view.width, detail_height), Image.Resampling.LANCZOS)
                crop_records[name] = {"source_box_xyxy": list(crop), "output_size": [view.width, detail_height]}
                canvas.paste(detail, (offset, detail_y + detail_header))
                if show_xray:
                    xray_y = detail_y + detail_header + detail_height
                    draw.text((offset + 15, xray_y + 16), f"{name.upper()} / XRAY - FULL HAND, IGNORES SCENE DEPTH", font=_font(17), fill=(255,205,133))
                    canvas.paste(xrays[name].crop(crop).resize((view.width, detail_height), Image.Resampling.LANCZOS),
                                 (offset, xray_y + detail_header))
            else:
                draw.text((offset + 20, detail_y + detail_header + 30), "Anchor outside view: no enlarged pose", font=_font(17), fill=(183,199,211))
            offset += view.width
        base_y = header + height + detail_header + detail_height + extra_height + 13
        draw.text((15, base_y), f"POINT + POSE | {selection.view_id} (u={selection.point.u:g}, v={selection.point.v:g}) | Roll {orientation.roll_deg:g} / Pitch {orientation.pitch_deg:g} / Yaw {orientation.yaw_deg:g} deg", font=_font(20), fill=(232,244,251))
        draw.text((15, base_y + 31), "Mint: Panda hand / dark: fingers / pink: anchor. Angles: selected camera. Coordinates: FULL views, not crops.", font=_font(15), fill=(160,188,207))
        anchor_note = "Measured depth TCP anchor. Review exact point AND orientation; reachability/collision checks are separate." if depth_sample else "Approximate AABB anchor. No scene-depth occlusion or motion validation. Review the point AND orientation."
        if scene_occlusion:
            anchor_note = "Scene depth applied (2 mm tolerance). " + ("XRAY reveals hidden hand. " if show_xray else "") + "Original camera views; not predicted wrist view or collision proof."
        draw.text((15, base_y + 58), anchor_note, font=_font(14), fill=(222,193,144))
    else:
        header, footer = 70, 98
        canvas = Image.new("RGB", (width, height + header + footer), (16, 22, 33))
        draw = ImageDraw.Draw(canvas)
        offset = 0
        for name in ("front", "wrist"):
            image = rendered[name]
            draw.text((offset + 14, 10), f"{name.upper()} | {'selected camera' if name == selection.view_id else 'same 3D pose'}", font=_font(19), fill=(242,245,250))
            draw.text((offset + 14, 39), f"Target: {body_name}", font=_font(14), fill=(174,187,206))
            canvas.paste(image, (offset, header)); offset += image.width
        base_y = header + height + 10
        draw.text((14, base_y), f"Roll {orientation.roll_deg:g} deg / Pitch {orientation.pitch_deg:g} deg / Yaw {orientation.yaw_deg:g} deg | selected camera", font=_font(17), fill=(245,245,250))
        draw.text((14, base_y + 29), "X red / Y green: jaw separation / Z blue: approach / gold: diagnostic gripper", font=_font(14), fill=(205,216,232))
        anchor_note = "Measured depth TCP anchor. Diagnostic wireframe; reachability/collision checks are separate." if depth_sample else "Approximate AABB anchor. Diagnostic wireframe; not an executable pose."
        draw.text((14, base_y + 54), anchor_note, font=_font(14), fill=(255,209,117))
    natural_image_size = canvas.size
    if max_image_dimension is not None:
        canvas.thumbnail((max_image_dimension, max_image_dimension), Image.Resampling.LANCZOS)
    canvas.save(output)
    if render_style == "solid":
        from src.tools.pose_editor.mesh import mesh_source_metadata
        geometry = mesh_source_metadata()
        geometry["scene_occlusion"] = scene_occlusion
    else:
        geometry = {"axis_length_m": axis_length, "jaw_width_m": jaw_width,
                    "finger_length_m": finger_length, "dimensions": "illustrative wireframe"}
    return {
        "schema_version": 2, "output_path": str(output),
        "render_style": render_style,
        "agent_image_layout": ("full_reference_depth_and_xray_crops" if show_xray else "full_reference_depth_crops" if scene_occlusion else "full_reference_top_pose_crops_bottom") if render_style == "solid" else "two_views",
        "include_xray": show_xray,
        "scene_occlusion": scene_occlusion and render_style == "solid", "occlusion_views": depth_stats,
        "image_size": list(canvas.size),
        "natural_image_size": list(natural_image_size),
        "image_resize": {
            "max_image_dimension": max_image_dimension,
            "scale_x": canvas.width / natural_image_size[0],
            "scale_y": canvas.height / natural_image_size[1],
            "method": "lanczos_thumbnail" if canvas.size != natural_image_size else "none",
        },
        "coordinate_spaces": {
            "projections": "original_per_view_pixels",
            "detail_crops.source_box_xyxy": "original_per_view_pixels",
            "detail_crops.output_size": "natural_composite_pixels",
            "image_size": "final_composite_pixels",
        },
        "detail_crops": crop_records,
        "axes_visible": render_style == "wireframe",
        "selection": selection.to_dict(), "body_name": body_name,
        "anchor_base": list(anchor),
        "rotation_base_from_gripper": [list(row) for row in rotation],
        "anchor_method": "selected_camera_metric_depth" if depth_sample else "selected_camera_ray_nearest_positive_aabb_intersection",
        "depth_sample": depth_sample,
        "approximate_anchor": depth_sample is None, "executable_pose": False,
        "pose_convention": {
            "frame": "selected_camera", "units": "degrees", "rotation_order": "Rz(yaw) Ry(pitch) Rx(roll)",
            "camera_axes": "+X right, +Y down, +Z into scene",
            "gripper_axes": "+Z approach, +Y jaw separation",
            "occlusion": (("registered metric optical Z" + ("; paired xray ignores scene depth" if show_xray else "")) if scene_occlusion else "mesh self-occlusion only; scene depth is not used for occlusion") if render_style == "solid" else "wireframe overlay; no depth test",
        },
        "geometry": geometry,
        "projections": projections,
    }
