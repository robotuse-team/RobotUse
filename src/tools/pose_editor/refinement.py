"""Bounded adjustment of a proposed gripper pose, about its own jaw centre.

A refinement never edits a candidate in place. It derives a new pose that the
same generation-time checks must accept before anything can be chosen, so a
refined candidate is evidence of the same kind as an originally proposed one.

Rotations default to fine adjustments; RobotUse explicitly removes angular magnitude
budgets while retaining finite transforms and motion validation. They are applied in the gripper's own frame
about the point between its jaws. That keeps one meaning for roll/pitch/yaw:
the numbers mean the same thing for a grasp, for a placement, and in the pose
preview the reviewer is shown.
"""
from __future__ import annotations

from typing import Any, Sequence
import math

import numpy as np

from src.tools.pose_editor.visual_pose import orientation_matrix
from src.tools.observation.preview import GripperOrientation, VisualWaypointError

STEP_LIMIT_DEG = 10.
CUMULATIVE_LIMIT_DEG = 30.
AXES = ("roll_deg", "pitch_deg", "yaw_deg")


def rotation_limits(tool, object_cloud_policy='source_view'):
    """Fused-cloud placement permits reorientation; other tools use fine limits."""
    if tool == 'adjust_place' and object_cloud_policy == 'fused':
        return 180., 360.
    return STEP_LIMIT_DEG, CUMULATIVE_LIMIT_DEG


class RefinementError(VisualWaypointError):
    """The requested adjustment leaves the fine-adjustment budget."""


def checked_adjustment(step: Sequence[float], applied: Sequence[float] = (0., 0., 0.),
                       *, step_limit_deg=STEP_LIMIT_DEG, cumulative_limit_deg=CUMULATIVE_LIMIT_DEG,
                       unrestricted=False):
    """Check finite angles, optionally imposing per-step and cumulative budgets."""
    try:
        requested = tuple(float(value) for value in step)
        already = tuple(float(value) for value in applied)
    except (TypeError, ValueError) as exc:
        raise RefinementError("adjustment angles must be numeric degrees") from exc
    if len(requested) != 3 or len(already) != 3:
        raise RefinementError("adjustment needs roll, pitch and yaw in degrees")
    total = tuple(a + b for a, b in zip(already, requested))
    for axis, value, accumulated in zip(AXES, requested, total):
        if not np.isfinite(value) or not np.isfinite(accumulated):
            raise RefinementError(f"{axis} and its accumulated value must be finite degrees")
        if not unrestricted and abs(value) > step_limit_deg:
            raise RefinementError(
                f"{axis} must be an adjustment within {step_limit_deg:g} degrees")
        if not unrestricted and abs(accumulated) > cumulative_limit_deg:
            raise RefinementError(
                f"{axis} would leave the {cumulative_limit_deg:g} degree budget for this candidate; "
                "choose a different candidate instead")
    return requested, total


def _local_rotation(step: Sequence[float]) -> np.ndarray:
    # The preview orientation type is canonical -180..180. Normalize only the
    # equivalent transform; retain the requested angles in the edit audit.
    roll, pitch, yaw = (math.remainder(float(value), 360.) for value in step)
    return np.asarray(orientation_matrix(
        GripperOrientation(roll_deg=roll, pitch_deg=pitch, yaw_deg=yaw)), dtype=float)


def _about(pivot: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Homogeneous transform rotating the base frame about ``pivot``."""
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = pivot - rotation @ pivot
    return matrix


def jaw_centre(hand_pose: Any, tcp_offset_z_m: float) -> np.ndarray:
    """The point between the jaws: the hand origin advanced along its approach."""
    hand = np.asarray(hand_pose, dtype=float)
    return hand[:3, 3] + hand[:3, :3] @ np.array([0., 0., float(tcp_offset_z_m)])


def _orthonormal(rotation: np.ndarray) -> np.ndarray:
    """The nearest true rotation to a pose that is only orthonormal to its own precision.

    A planner hand pose arrives off by roughly its float32 epsilon. Conjugating
    the step through that pose multiplies the error by three, so a third
    refinement inside the budget would no longer be a rotation the preview
    accepts. Conjugating through the nearest true rotation instead keeps the
    error where it started, however many steps are chained.
    """
    left, _, right = np.linalg.svd(rotation)
    nearest = left @ right
    if np.linalg.det(nearest) < 0:  # a reflection is never a gripper pose
        raise RefinementError("grasp pose rotation is not right-handed")
    return nearest


def _pivoted(hand: np.ndarray, step: Sequence[float], tcp_offset_z_m: float) -> np.ndarray:
    # Conjugating the local rotation into the base frame keeps the new hand
    # rotation exactly ``hand @ local`` while the jaw centre stays put.
    rotation = _orthonormal(hand[:3, :3])
    about = rotation @ _local_rotation(step) @ rotation.T
    return _about(jaw_centre(hand, tcp_offset_z_m), about)


def refined_grasp_pose(pose: Any, step: Sequence[float], *, tcp_offset_z_m: float) -> np.ndarray:
    """Rotate a grasp hand pose about the point between its jaws."""
    hand = np.asarray(pose, dtype=float)
    if hand.shape != (4, 4) or not np.isfinite(hand).all():
        raise RefinementError("grasp pose must be a finite 4 by 4 transform")
    return _pivoted(hand, step, tcp_offset_z_m) @ hand


def refined_object_transform(transform: Any, step: Sequence[float], *, closed_ee: Any,
                             grasp_to_ee: Any, release_clearance_m: float,
                             tcp_offset_z_m: float) -> np.ndarray:
    """Rotate a released placement about the jaw centre that will be holding it.

    The placement pipeline rebuilds the release pose as ``C(delta @ closed_ee)``,
    where ``C`` raises it by the release clearance. Returning
    ``C^-1 A C delta`` therefore makes that rebuild reproduce ``A`` applied to
    the original release pose exactly, with no clearance drift to absorb.
    """
    delta = np.asarray(transform, dtype=float)
    closed = np.asarray(closed_ee, dtype=float)
    if delta.shape != (4, 4) or closed.shape != (4, 4):
        raise RefinementError("placement transforms must be 4 by 4")
    clearance = np.eye(4)
    clearance[2, 3] = float(release_clearance_m)
    released = clearance @ delta @ closed
    hand = released @ np.linalg.inv(np.asarray(grasp_to_ee, dtype=float))
    return np.linalg.inv(clearance) @ _pivoted(hand, step, tcp_offset_z_m) @ clearance @ delta


def tcp_offset_from(grasp_to_ee: Any) -> float:
    """The jaw-centre offset carried by the runtime grasp/EE calibration."""
    matrix = np.asarray(grasp_to_ee, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise RefinementError("grasp-to-EE calibration must be a finite 4 by 4 transform")
    return float(matrix[2, 3])


def preview_refined_pose(frames: Sequence[Any], hand_pose: Any, output_dir: Any, *,
                         tcp_offset_z_m: float, show_axes: bool = True,
                         closeup: bool = True, expected_open_width_m: float | None = None,
                         mesh_source=None) -> list[str]:
    """Render one hand pose onto every calibrated frame of an observation.

    The preview tool is deliberately ignorant of this pipeline, so the frame
    adaptation lives here: a sensor payload becomes a camera, the native grip
    frame becomes the renderer's display frame, and measured depth hides the
    parts of the gripper the camera could not see.

    A refinement is judged on how the gripper turned, so by default the crop
    follows the gripper and its own axes are drawn: the reviewer can see which
    way roll, pitch and yaw will move it before asking for the next step.
    """
    from pathlib import Path

    if mesh_source is not None:
        from PIL import Image, ImageDraw
        from src.tools.grasp.input_cards import camera_project, mesh_draw
        from src.tools.pose_editor.inspection import transform_points
        parts, _ = mesh_source.load_gripper_mesh(expected_open_width_m)
        world = {name: transform_points(triangles, hand_pose) for name, triangles in parts.items()}
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for frame in frames:
            selected_view = getattr(mesh_source, 'preview_view_id', None)
            if selected_view is not None and frame.view_id != selected_view:
                continue
            image = Image.fromarray(np.asarray(frame.rgb).copy())
            pixel_scale = np.ones(2)
            pixel_offset = np.zeros(2)
            padding = getattr(mesh_source, 'preview_padding_fraction', None)
            if closeup and padding is not None:
                vertices = np.concatenate([part.reshape(-1, 3) for part in world.values()])
                xy, depth = camera_project(vertices, frame)
                valid = np.isfinite(xy).all(axis=1) & (depth > 0)
                if valid.any():
                    lo = xy[valid].min(axis=0)
                    hi = xy[valid].max(axis=0)
                    # Only magnify the visible intersection; never wrap a crop
                    # around an offscreen proposal or fabricate missing pixels.
                    lo = np.maximum(lo, [0, 0])
                    hi = np.minimum(hi, image.size)
                    if np.all(hi > lo):
                        margin = np.maximum((hi - lo) * padding, 32.)
                        lo = np.maximum(np.floor(lo - margin), [0, 0]).astype(int)
                        hi = np.minimum(np.ceil(hi + margin), image.size).astype(int)
                        crop = image.crop((*lo, *hi))
                        # Letterbox instead of stretching the geometry.
                        from PIL import ImageOps
                        content_size = ImageOps.contain(crop, image.size).size
                        pixel_scale = np.asarray(content_size) / (hi - lo)
                        pixel_offset = (np.round((np.asarray(image.size) - content_size) / 2)
                                        - lo * pixel_scale)
                        image = ImageOps.pad(crop, image.size, method=Image.Resampling.LANCZOS,
                                             color=(24, 24, 24))
            # Project into the final crop, then antialias the mesh layer only.
            # Resizing an already rasterized mesh magnifies its original pixels.
            def project(points):
                xy, depth = camera_project(points, frame)
                return xy * pixel_scale + pixel_offset, depth
            mesh_draw(image, world, project, translucent=True,
                      supersampling=getattr(mesh_source, 'preview_supersampling', 1))
            ImageDraw.Draw(image).text((8, 8), 'Gripper proposal mesh overlay (occlusion not tested)', fill='yellow')
            path = directory / f'preview-{frame.view_id}.png'
            image.save(path)
            paths.append(str(path))
        return paths

    from src.tools.pose_editor.gripper_preview import GripperPose, gripper_pose_preview
    from src.tools.pose_editor.visual_pose import _Camera
    from src.tools.pose_editor.mesh import mesh_source_metadata

    display_from_grip = np.asarray(mesh_source_metadata()["display_from_grip_site_rotation"], float)
    hand = np.asarray(hand_pose, dtype=float)
    pose = GripperPose(tuple(jaw_centre(hand, tcp_offset_z_m)),
                       tuple(map(tuple, hand[:3, :3] @ display_from_grip.T)),
                       {"method": "refined_candidate_pose", "approximate": False})
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for frame in frames:
        camera = _Camera(frame.view_id, int(frame.width), int(frame.height),
                         tuple(map(tuple, frame.intrinsics)),
                         tuple(map(tuple, frame.camera_to_base.rotation)),
                         tuple(frame.camera_to_base.translation))
        path = directory / f"preview-{frame.view_id}.png"
        gripper_pose_preview(camera, frame.rgb, pose=pose, scene_depth=frame.depth_m,
                             show_axes=show_axes, closeup=closeup, output_path=path,
                             expected_open_width_m=expected_open_width_m,
                             candidate_axes_rotation_base=hand[:3, :3])
        paths.append(str(path))
    return paths
