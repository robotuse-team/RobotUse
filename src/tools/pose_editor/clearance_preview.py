"""Read-only RGB localization of a saved observed-scene clearance witness."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def visible_witness_pixel(frame, witness):
    """Return a calibrated pixel only when saved depth supports visibility.

    The point is already resolved by the checker. A local point_index
    cannot be used here. Missing depth and an occluding raster footprint are
    deliberately inconclusive; no new scene capture or geometric inference.
    """
    if not isinstance(witness, dict) or witness.get('frame') != 'connector_base':
        return None
    try:
        point = np.asarray(witness['scene_point_xyz_m'], dtype=float)
        rotation = np.asarray(frame.camera_to_base.rotation, dtype=float)
        translation = np.asarray(frame.camera_to_base.translation, dtype=float)
        intrinsics = np.asarray(frame.intrinsics, dtype=float)
        depth = np.asarray(frame.depth_m)
        rgb = np.asarray(frame.rgb)
        if (point.shape != (3,) or rotation.shape != (3, 3) or translation.shape != (3,)
                or intrinsics.shape != (3, 3) or depth.shape != rgb.shape[:2]
                or not all(np.isfinite(a).all() for a in (point, rotation, translation, intrinsics))):
            return None
        camera = rotation.T @ (point - translation)
        if camera[2] <= 0:
            return None
        homogeneous = intrinsics @ camera
        if not np.isfinite(homogeneous).all() or homogeneous[2] <= 0:
            return None
        u, v = homogeneous[:2] / homogeneous[2]
        height, width = depth.shape
        if not (0 <= u <= width - 1 and 0 <= v <= height - 1):
            return None
        # Saved scene points/depth are float32. A back-projected sensor pixel
        # can return just across its integer boundary after camera transforms.
        # Do not treat that roundoff as a genuine four-pixel footprint.
        pixel = np.array([u, v])
        nearest = np.rint(pixel)
        pixel_epsilon = 8 * np.finfo(np.float32).eps * np.maximum(1., np.abs(pixel))
        u, v = np.where(np.abs(pixel-nearest) <= pixel_epsilon, nearest, pixel)
        # Check every pixel touched by the projection's raster footprint. This
        # conservatively rejects depth edges instead of marking foreground.
        xs = sorted({int(np.floor(u)), int(np.ceil(u))})
        ys = sorted({int(np.floor(v)), int(np.ceil(v))})
        observed = depth[np.ix_(ys, xs)]
        if not np.isfinite(observed).all() or (observed <= 0).any():
            return None
        # Numerical tolerance for saved float32 sensor depth, not clearance.
        epsilon = 8 * np.finfo(np.float32).eps * max(1., abs(float(camera[2])))
        if camera[2] > float(observed.min()) + epsilon:
            return None
        return float(u), float(v)
    except (AttributeError, KeyError, TypeError, ValueError, IndexError):
        return None


def compose_witness_preview(candidate_path, frame, witness, output_path):
    """Keep the endpoint preview and actual witness in distinctly named panels.

    Returns None when localization is unsupported, preserving the original
    diagnostic. Input RGB and candidate images are never modified.
    """
    pixel = visible_witness_pixel(frame, witness)
    if pixel is None:
        return None
    with Image.open(candidate_path) as source:
        candidate = source.convert('RGB')
    observed = Image.fromarray(np.asarray(frame.rgb).copy()).convert('RGB')
    draw = ImageDraw.Draw(observed)
    u, v = pixel
    draw.ellipse((u-7, v-7, u+7, v+7), outline='magenta', width=2)
    draw.line((u-10, v, u+10, v), fill='magenta', width=1)
    draw.line((u, v-10, u, v+10), fill='magenta', width=1)
    # Headers sit outside sensor pixels so a near-edge witness stays visible.
    canvas = Image.new('RGB', (candidate.width+observed.width, max(candidate.height, observed.height)+48), 'black')
    canvas.paste(candidate, (0, 48))
    canvas.paste(observed, (candidate.width, 48))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 16)
    except OSError:
        font = ImageFont.load_default()
    draw.text((5, 5), 'REJECTED: candidate endpoint preview', fill='orange', font=font)
    draw.text((candidate.width+5, 5), 'First sampled clearance witness (measured point)', fill='magenta', font=font)
    view = getattr(frame, 'view_id', '')
    view_label = {'agentview':'front', 'robot0_eye_in_hand':'wrist'}.get(view, 'camera')
    draw.text((candidate.width+5, 23), f'Saved {view_label} RGB; margin proximity, not physical contact', fill='white', font=font)
    output = Path(output_path)
    canvas.save(output)
    return output
