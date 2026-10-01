"""Deterministic, calibrated grasp review cards from recorded sensor geometry.

RGB overlays are explicitly translucent X-ray proposals, not future images.
Virtual views use a target-bound scale shared by all candidates for that target.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLORS = {'hand': (88, 108, 143), 'left_finger': (29, 167, 207),
          'right_finger': (202, 83, 155)}
ORANGE = (235, 156, 35)


def font(size):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default()


def camera_project(points, frame):
    """Inverse of sensor_perception.unproject_masked_depth; optical +Z forward."""
    transform = frame.camera_to_base
    k = np.asarray(frame.intrinsics, dtype=float)
    if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError('finite calibrated intrinsics with positive focal lengths required')
    camera = (np.asarray(points) - np.asarray(transform.translation)) @ np.asarray(transform.rotation)
    z = camera[..., 2]
    projected = camera @ k.T
    return projected[..., :2] / np.where(z > 1e-5, z, np.nan)[..., None], z


def arrow(draw, a, b, color, width=4):
    a, b = np.asarray(a), np.asarray(b)
    if not np.isfinite([a, b]).all():
        return
    delta = b - a
    length = np.linalg.norm(delta)
    if length < 3:
        return
    direction = delta / length
    side = np.array([-direction[1], direction[0]])
    draw.line([tuple(a), tuple(b)], fill=color, width=width)
    size = min(13, length * .4)
    draw.polygon([tuple(b), tuple(b-size*direction+size*.45*side),
                  tuple(b-size*direction-size*.45*side)], fill=color)


def mesh_draw(image, parts, project, *, translucent=False, supersampling=1):
    """Composite a projected mesh, optionally antialiasing only its RGBA layer."""
    if type(supersampling) is not int or supersampling < 1:
        raise ValueError('supersampling must be a positive integer')
    layer = Image.new('RGBA', tuple(size * supersampling for size in image.size))
    draw = ImageDraw.Draw(layer)
    faces = []
    for name, triangles in parts.items():
        xy, depth = project(triangles)
        xy = xy * supersampling
        for tri, dep, world in zip(xy, depth, triangles):
            if not np.isfinite(tri).all():
                continue
            normal = np.cross(world[1]-world[0], world[2]-world[0])
            normal /= max(np.linalg.norm(normal), 1e-12)
            shade = .65 + .35 * abs(normal @ np.array([.3, -.4, .866]))
            color = tuple(int(c*shade) for c in COLORS.get(name, COLORS['hand']))
            alpha = (75 if name == 'hand' else 210) if translucent else 255
            faces.append((float(np.mean(dep)), tri, (*color, alpha)))
    for _, tri, color in sorted(faces, key=lambda x: -x[0]):
        draw.polygon([tuple(p) for p in tri], fill=color)
    if supersampling > 1:
        layer = layer.resize(image.size, Image.Resampling.LANCZOS)
    image.paste(Image.alpha_composite(image.convert('RGBA'), layer).convert('RGB'))


def rgb_panel(frame, obj, parts, center, span, *, crop_points=None):
    rgb = Image.open(frame.rgb_path).convert('RGB')
    uv, _ = camera_project(obj if crop_points is None else crop_points, frame)
    valid = uv[np.isfinite(uv).all(axis=1)]
    if not len(valid):
        raise ValueError('target has no positive camera depth')
    target_center = (valid.min(0) + valid.max(0)) / 2
    _, z = camera_project(np.array([center]), frame)
    # Same target/calibration => same crop for every candidate, including rejected ones.
    side = max(80., float(frame.intrinsics[0][0]) * span / max(float(z[0]), .02))
    side = min(side, max(rgb.size)*1.25)
    # Preserve equal X/Y scale: rectangular crop matches the panel aspect ratio.
    height = side * 312 / 448
    box = tuple(int(round(v)) for v in [target_center[0]-side/2, target_center[1]-height/2,
                                        target_center[0]+side/2, target_center[1]+height/2])
    panel = rgb.crop(box).resize((448, 312), Image.Resampling.LANCZOS)
    scale = np.array([448/(box[2]-box[0]), 312/(box[3]-box[1])])
    def project(points):
        xy, depth = camera_project(points, frame)
        return (xy-np.array(box[:2]))*scale, depth
    # Small measured-target dots tie the RGB to the partial cloud without filling the bowl.
    draw = ImageDraw.Draw(panel)
    points, _ = project(obj[::max(1, len(obj)//500)])
    for x, y in points[np.isfinite(points).all(axis=1)]:
        draw.ellipse((x-1, y-1, x+1, y+1), fill=ORANGE)
    mesh_draw(panel, parts, project, translucent=True)
    return panel, {'view_id': frame.view_id, 'crop_xyxy': box,
                   'rgb_path': str(frame.rgb_path), 'overlay': 'translucent X-ray; occlusion not predicted'}


def virtual_panel(obj, scene, parts, pose, center, span, closing=False, *, fit_points=None, show_directions=True):
    from src.tools.pose_editor.inspection import transform_points
    # Fixed oblique view across candidates; second view aligns with the finger closing plane.
    if closing:
        right, up = pose[:3, 0], -pose[:3, 2]
    else:
        right = np.array([1., -1., 0.]); right /= np.linalg.norm(right)
        up = np.array([.5, .5, 1.]); up /= np.linalg.norm(up)
    basis = np.array([right, up])
    forward = -np.cross(right, up)
    fit = obj if fit_points is None else fit_points
    scale = min(830/span, 345/(float(np.max(np.ptp(fit, axis=0)))+.12))
    # Fixed target-relative center, independent of candidate orientation/ranking.
    display_center = center + np.array([0., 0., .035]) if not closing else center - pose[:3, 2]*.035
    def project(points):
        relative = np.asarray(points)-display_center
        return relative @ basis.T * [scale, -scale] + [448, 195], relative @ forward
    panel = Image.new('RGB', (896, 398), (242, 246, 250))
    draw = ImageDraw.Draw(panel)
    near = scene[np.linalg.norm(scene-center, axis=1) < span*.8]
    near = near[::max(1, len(near)//4500)]
    for x, y in project(near)[0]:
        draw.point((x, y), fill=(192, 202, 214))
    for x, y in project(obj)[0]:
        draw.ellipse((x-.8, y-.8, x+.8, y+.8), fill=ORANGE)
    mesh_draw(panel, parts, project, translucent=True)
    # Axis glyphs are directions only, not contact predictions or a planned trajectory.
    guides = transform_points([[0, 0, 0], [0, 0, .06],
                               [-.07, 0, .10], [-.022, 0, .10],
                               [.07, 0, .10], [.022, 0, .10]], pose)
    xy = project(guides)[0]
    if show_directions:
        arrow(draw, xy[0], xy[1], (52, 100, 218))
        arrow(draw, xy[2], xy[3], COLORS['right_finger'])
        arrow(draw, xy[4], xy[5], COLORS['left_finger'])
    draw.line([(24, 365), (24+scale*.05, 365)], fill=(40, 55, 75), width=3)
    draw.text((24, 372), '50 mm', font=font(14), fill=(40, 55, 75))
    return panel, {'basis_rows': basis.tolist(), 'pixels_per_m': scale,
                   'center_m': display_center.tolist(), 'span_m': span,
                   'target_rendering': 'X-ray measured surface; no inferred contacts'}


def render_grasp_cards(geometry, prediction, output_dir, *, candidate_ref,
                       graspgen_root=None, expected_open_width_m=None, views=2):
    from src.tools.pose_editor.inspection import load_panda_mesh, transform_points, checked_transform, _cloud
    pose = checked_transform(prediction.pose)
    obj, scene = _cloud(geometry.object_points), _cloud(geometry.scene_points)
    if not len(obj):
        raise ValueError('review card needs observed target points')
    parts, meta = load_panda_mesh(graspgen_root, expected_open_width_m,
        libero_adapter=bool(getattr(prediction, "gripper_adapter", None)))
    parts = {k: transform_points(v, pose) for k, v in parts.items()}
    center = (obj.min(0)+obj.max(0))/2
    span = max(.32, float(np.linalg.norm(np.ptp(obj, axis=0)))+.24)
    frames = [g.frame for g in getattr(geometry, 'per_view', (geometry,))]
    frames.sort(key=lambda f: {'agentview': 0, 'robot0_eye_in_hand': 1}.get(f.view_id, 2))
    rgb_panels = [rgb_panel(f, obj, parts, center, span) for f in frames[:2]]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths, view_meta = [], []
    for index in range(views):
        card = Image.new('RGB', (960, 960), (255, 255, 255))
        draw = ImageDraw.Draw(card)
        draw.text((32, 18), 'GRASP CANDIDATE  /  '+candidate_ref[:12], font=font(26), fill=(24, 38, 58))
        draw.text((32, 55), f"Proposed pose | nominal open {meta.get('opening_m', .08)*1000:g} mm" if expected_open_width_m is None else
                  f'Candidate pose | expected opening {expected_open_width_m*1000:.0f} mm', font=font(18), fill=(78, 91, 110))
        for col, (panel, provenance) in enumerate(rgb_panels):
            x = 32+col*448
            label = {'agentview': 'CURRENT FRONT', 'robot0_eye_in_hand': 'CURRENT WRIST'}.get(provenance['view_id'], provenance['view_id'])
            draw.text((x, 92), label+' / crop', font=font(20), fill=(24, 38, 58))
            card.paste(panel, (x, 123))
        draw.text((32, 443), 'Current RGB + projected candidate; NOT the future wrist-camera view', font=font(18), fill=(78, 91, 110))
        draw.text((32, 480), 'CLOSING PLANE / finger gap and depth' if index else
                  'OBLIQUE / target and finger relationship', font=font(22), fill=(24, 38, 58))
        panel, vm = virtual_panel(obj, scene, parts, pose, center, span, bool(index))
        card.paste(panel, (32, 517))
        for x, color, label in [(32, COLORS['left_finger'], 'Finger A'), (188, COLORS['right_finger'], 'Finger B'),
                                 (344, ORANGE, 'Observed target'), (588, (52,100,218), 'Approach / inward = close')]:
            draw.rectangle((x, 920, x+12, 932), fill=color)
            draw.text((x+18, 916), label, font=font(16), fill=(40, 55, 75))
        draw.text((32, 940), 'X-ray proposal. Contact, hidden shape and collision remain unverified.', font=font(14), fill=(78,91,110))
        path = output / f'{candidate_ref}-review-{index+1}.png'
        card.save(path)
        paths.append(str(path)); view_meta.append(vm)
    meta.update(schema='grasp-review-card.v1', candidate_ref=candidate_ref,
                candidate_pose=pose.tolist(), observation_id=geometry.observation_id,
                rgb_views=[p[1] for p in rgb_panels], views=view_meta,
                actual_contact='unknown; no contact points predicted',
                camera_semantics='current capture pose; wrist camera moves with hand during execution',
                required_inputs='registered metric RGBD, calibrated intrinsics, capture-time camera-to-base pose, same-frame candidate and known gripper mesh',
                rendering='recorded RGB + calibrated proposed mesh; measured partial target X-ray')
    manifest = output / f'{candidate_ref}-review.json'
    manifest.write_text(json.dumps(meta, indent=2)+'\n')
    return {'image_paths': paths, 'metadata': meta, 'metadata_path': str(manifest)}
