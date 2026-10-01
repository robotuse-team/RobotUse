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
SCENE_GREY = (150, 162, 178)


def part_role(name):
    """Colour role of a mesh part; native RoboLab parts are keyed 'body:robot_mesh_N'."""
    key = str(name).lower()
    if key in COLORS:
        return key
    if 'finger' in key or 'pad' in key or 'knuckle' in key:
        if 'left' in key:
            return 'left_finger'
        if 'right' in key:
            return 'right_finger'
    return 'hand'


def font(size):
    for face in ('DejaVuSans.ttf', '/System/Library/Fonts/Supplemental/Arial.ttf'):
        try:
            return ImageFont.truetype(face, size)
        except OSError:
            pass
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


def mesh_draw(image, parts, project, *, translucent=False):
    layer = Image.new('RGBA', image.size)
    draw = ImageDraw.Draw(layer)
    faces = []
    for name, triangles in parts.items():
        xy, depth = project(triangles)
        for tri, dep, world in zip(xy, depth, triangles):
            if not np.isfinite(tri).all():
                continue
            normal = np.cross(world[1]-world[0], world[2]-world[0])
            normal /= max(np.linalg.norm(normal), 1e-12)
            shade = .65 + .35 * abs(normal @ np.array([.3, -.4, .866]))
            role = part_role(name)
            color = tuple(int(c*shade) for c in COLORS[role])
            alpha = (75 if role == 'hand' else 210) if translucent else 255
            faces.append((float(np.mean(dep)), tri, (*color, alpha)))
    for _, tri, color in sorted(faces, key=lambda x: -x[0]):
        draw.polygon([tuple(p) for p in tri], fill=color)
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


def fitted_projection(points, basis, center, viewport, *, requested_scale=None):
    """Equal metric scale fitting only supplied geometry into a pixel viewport."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    projected = (points - center) @ basis.T
    lo, hi = projected.min(0), projected.max(0)
    left, top, right, bottom = viewport
    fit_scale = min((right-left)/max(hi[0]-lo[0], .001),
                    (bottom-top)/max(hi[1]-lo[1], .001))
    scale = fit_scale if requested_scale is None else min(requested_scale, fit_scale)
    origin = np.array([(left+right)/2, (top+bottom)/2]) - (lo+hi)/2 * [scale, -scale]
    return scale, origin, fit_scale


def view_basis(view, pose):
    """Right/up rows for a named view of a pose in the base frame (z up).

    oblique: fixed three-quarter view shared by every candidate. closing: along
    the finger closing plane, in the gripper's own frame. side: a level view with
    world up, fingers left and right, so depth against the table reads directly.
    top: straight down, so yaw against the object's long axis reads directly.
    """
    if view == 'closing':
        right, up = pose[:3, 0], -pose[:3, 2]
    elif view == 'side':
        right = np.array([pose[0, 0], pose[1, 0], 0.])
        if np.linalg.norm(right) < .2:
            right = np.array([1., 0., 0.])
        right = right / np.linalg.norm(right)
        up = np.array([0., 0., 1.])
    elif view == 'top':
        right, up = np.array([1., 0., 0.]), np.array([0., 1., 0.])
    else:
        right = np.array([1., -1., 0.]); right /= np.linalg.norm(right)
        up = np.array([.5, .5, 1.]); up /= np.linalg.norm(up)
    return np.asarray(right, float), np.asarray(up, float)


def virtual_panel(obj, scene, parts, pose, center, span, closing=False, *, fit_points=None, show_directions=True,
                  view=None, support_z=None, support_label='estimated support height', edit_cues=None):
    from src.tools.pose_editor.inspection import transform_points
    view = view or ('closing' if closing else 'oblique')
    right, up = view_basis(view, pose)
    if edit_cues and view == 'side':
        right, up = np.array([1.,0.,0.]), np.array([0.,0.,1.])
    basis = np.array([right, up])
    forward = -np.cross(right, up)
    guides = transform_points([[0, 0, 0], [0, 0, .06],
                               [-.07, 0, .10], [-.022, 0, .10],
                               [.07, 0, .10], [.022, 0, .10]], pose)
    fit = np.concatenate([obj, obj if fit_points is None else fit_points,
                          *[v.reshape(-1, 3) for v in parts.values()],
                          guides if show_directions else np.empty((0, 3))])
    if edit_cues:
        pivot = np.asarray(edit_cues['pivot'])
        axes = pose[:3, :3] if edit_cues['translation_frame'] == 'local' else np.eye(3)
        from src.tools.pose_editor.refinement import _local_rotation
        cue_points = [pivot, *[pivot + .06*axes[:, i] for i in range(3)]]
        for i in range(3):
            for angle in (0, 10):
                cue_points.append(pivot + pose[:3,:3] @
                    (_local_rotation(np.eye(3)[i]*angle) @ (np.eye(3)[:,(i+1)%3]*.07)))
        fit = np.concatenate([fit, np.asarray(cue_points)])
    display_center = np.asarray(center)
    scale, origin, _ = fitted_projection(fit, basis, display_center,
        (65, 65, 790, 765) if edit_cues else (48, 40, 790, 315))
    def project(points):
        relative = np.asarray(points)-display_center
        return relative @ basis.T * [scale, -scale] + origin, relative @ forward
    panel = Image.new('RGB', (896, 896) if edit_cues else (896, 398), (242, 246, 250))
    draw = ImageDraw.Draw(panel)
    near = scene[np.linalg.norm(scene-center, axis=1) < span*.8]
    near = near[::max(1, len(near)//(900 if edit_cues else 4500))]
    for x, y in project(near)[0]:
        draw.point((x, y), fill=(211,219,228) if edit_cues else SCENE_GREY)
    if view == 'side' and (support_z is not None or len(obj)):
        # The surface the object rests on (or, for a placement, the destination
        # top the caller supplies), so "under it" is a line the fingers or the
        # carried object are above or below, not something inferred from dots.
        support = float(support_z) if support_z is not None else float(np.percentile(np.asarray(obj)[:, 2], 2))
        ends = np.array([display_center + right*span*.45, display_center - right*span*.45])
        ends[:, 2] = support
        (x0, y0), (x1, y1) = project(ends)[0]
        draw.line([(x0, y0), (x1, y1)], fill=(96, 72, 40), width=3)
        draw.text((max(8, min(x0, x1)+4), max(8, min(790 if edit_cues else 330, min(y0, y1)+4))), support_label, font=font(25 if edit_cues else 13), fill=(96, 72, 40))
    for x, y in project(obj)[0]:
        draw.ellipse((x-.8, y-.8, x+.8, y+.8), fill=ORANGE)
    mesh_draw(panel, parts, project, translucent=True)
    # Axis glyphs are directions only, not contact predictions or a planned trajectory.
    xy = project(guides)[0]
    if show_directions and not edit_cues:
        arrow(draw, xy[0], xy[1], (52, 100, 218))
        arrow(draw, xy[2], xy[3], COLORS['right_finger'])
        arrow(draw, xy[4], xy[5], COLORS['left_finger'])
    if edit_cues:
        # These glyphs use the SAME projection as the mesh. Translation arrows
        # show positive tool inputs; place is base XYZ while grasp is local XYZ.
        pivot = np.asarray(edit_cues['pivot'])
        axes = pose[:3, :3] if edit_cues['translation_frame'] == 'local' else np.eye(3)
        origin_xy = project(pivot)[0]
        for index, color in enumerate(((204, 58, 58), (32, 142, 93), (50, 98, 215))):
            end = project(pivot + .06*axes[:, index])[0]
            if np.linalg.norm(end-origin_xy) > 12:
                arrow(draw, origin_xy, end, color, 7)
                label = '+'+'XYZ'[index]
                pos = tuple(end + [8, -32])
                box = draw.textbbox(pos, label, font=font(36))
                draw.rectangle((box[0]-3,box[1]-3,box[2]+3,box[3]+3), fill='white')
                draw.text(pos, label, fill=color, font=font(36))
        target = (np.asarray(edit_cues['target_center'])
                  if edit_cues['kind'] == 'grasp' else np.asarray(edit_cues['destination_center']))
        start = pivot if edit_cues['kind'] == 'grasp' else np.asarray(edit_cues['object_center'])
        for point, color in ((start, (25, 45, 75)), (target, ORANGE)):
            x, y = project(point)[0]
            draw.ellipse((x-12,y-12,x+12,y+12), fill='white', outline=color, width=3)
            draw.line([(x-9,y),(x+9,y)], fill=color, width=4)
            draw.line([(x,y-9),(x,y+9)], fill=color, width=4)
        if edit_cues['kind'] == 'place' and view == 'top' and fit_points is not None:
            from scipy.spatial import ConvexHull, QhullError
            try:
                footprint = np.asarray(fit_points)
                hull = ConvexHull(footprint[:, :2])
                outline = project(footprint[hull.vertices])[0]
                draw.line([tuple(p) for p in np.vstack((outline, outline[:1]))], fill=(32,142,93), width=3)
            except QhullError:
                pass
    ruler_y = 840 if edit_cues else 365
    draw.line([(24, ruler_y), (24+scale*.05, ruler_y)], fill=(40, 55, 75), width=3)
    draw.text((24, ruler_y+7), '50 mm', font=font(25 if edit_cues else 14), fill=(40, 55, 75))
    return panel, {'basis_rows': basis.tolist(), 'pixels_per_m': scale,
                   'center_m': display_center.tolist(), 'span_m': span,
                   'pixel_origin': origin.tolist(), 'framing': 'measured target, full proposed gripper and direction guides',
                   'target_rendering': 'X-ray measured surface; no inferred contacts'}


SLIDER_AXES = (('roll_deg', -180, 180), ('pitch_deg', -180, 180), ('yaw_deg', -180, 180),
               ('dx_mm', -150, 150), ('dy_mm', -150, 150), ('dz_mm', -150, 150))


def slider_strip(sliders, width=896, *, axes=SLIDER_AXES):
    """The pose editor as sliders: cumulative offset per axis from the generator's pose."""
    strip = Image.new('RGB', (width, 22 + 26*len(axes)), (250, 250, 252))
    draw = ImageDraw.Draw(strip)
    draw.text((12, 4), 'POSE EDITOR / command totals from the generated draft', font=font(14), fill=(24, 38, 58))
    for row, (name, lo, hi) in enumerate(axes):
        y = 30 + row*26
        value = float((sliders or {}).get(name, 0.) or 0.)
        draw.text((12, y-2), name, font=font(13), fill=(40, 55, 75))
        x0, x1 = 100, width-110
        draw.line([(x0, y+6), (x1, y+6)], fill=(200, 205, 214), width=3)
        mid = (x0+x1)/2
        draw.line([(mid, y), (mid, y+12)], fill=(160, 166, 178), width=1)
        px = x0 + (min(max(value, lo), hi)-lo)/(hi-lo)*(x1-x0)
        draw.ellipse((px-7, y-1, px+7, y+13), fill=(52, 100, 218) if value else (150, 162, 178))
        draw.text((x1+12, y-2), f'{value:+.0f}', font=font(13), fill=(24, 38, 58))
    return strip


def rotation_example(pose, pivot, axis, degrees=30.):
    """Same local-axis, jaw-pivot transform as the edit tool; visual sample only."""
    from src.tools.pose_editor.refinement import _local_rotation, _orthonormal
    rotation = _orthonormal(np.asarray(pose)[:3,:3])
    world = rotation @ _local_rotation(np.eye(3)[axis]*degrees) @ rotation.T
    pivot = np.asarray(pivot)
    return world, pivot - world @ pivot


def rotation_panel(parts, obj, pose, cues, axis):
    """Look along the positive edited axis so its rotation cannot be foreshortened."""
    pivot = np.asarray(cues['pivot'])
    rotation, translation = rotation_example(pose, pivot, axis)
    ghost = {key: points @ rotation.T + translation for key,points in parts.items()}
    right, up = pose[:3,(axis+1)%3], pose[:3,(axis+2)%3]
    basis = np.array([right,up]); forward = -pose[:3,axis]
    fit = np.concatenate([v.reshape(-1,3) for v in [*parts.values(),*ghost.values()]]+[pivot[None]])
    if cues['kind'] == 'place':
        fit = np.concatenate([fit,obj,obj @ rotation.T + translation])
    scale, origin, _ = fitted_projection(fit,basis,pivot,(55,40,595,440))
    def project(points):
        relative = np.asarray(points)-pivot
        return relative @ basis.T * [scale,-scale]+origin, relative @ forward
    panel = Image.new('RGB',(650,560),(245,247,251))
    mesh_draw(panel,parts,project)
    layer = Image.new('RGBA',panel.size)
    overlay = ImageDraw.Draw(layer)
    for triangles in ghost.values():
        for tri in project(triangles)[0]:
            overlay.polygon([tuple(point) for point in tri],fill=(133,69,198,90))
    panel = Image.alpha_composite(panel.convert('RGBA'),layer).convert('RGB')
    draw = ImageDraw.Draw(panel)
    if cues['kind'] == 'place':
        for cloud,color in ((obj,ORANGE),(obj @ rotation.T+translation,(133,69,198))):
            for x,y in project(cloud[::max(1,len(cloud)//1800)])[0]:
                draw.ellipse((x-1,y-1,x+1,y+1),fill=color)
    x,y=project(pivot)[0]
    draw.ellipse((x-12,y-12,x+12,y+12),fill='white',outline=(24,38,58),width=3)
    draw.ellipse((x-4,y-4,x+4,y+4),fill=(24,38,58))
    # Fixed large sign legend: looking from +axis toward the pivot, + is CCW.
    arc=np.array([[525+70*np.cos(t),485-70*np.sin(t)] for t in np.linspace(0,np.pi/2,25)])
    draw.line([tuple(p) for p in arc],fill=(133,69,198),width=6)
    arrow(draw,arc[-4],arc[-1],(133,69,198),6)
    draw.text((30,480),'NOW + purple +30 deg example',font=font(24),fill=(24,38,58))
    draw.text((30,515),'View from +local '+'XYZ'[axis]+' toward jaw centre',font=font(22),fill=(78,91,110))
    return panel


def render_pose_card(obj, scene, parts, pose, center, span, path, *, title, sliders=None,
                     fit_points=None, show_directions=True, note=None, support_z=None,
                     support_label='estimated support height', side_label=None, edit_cues=None):
    """One image per edit: the sliders, then the pose in the cloud from the side, the top and the closing plane."""
    strip = slider_strip(sliders)
    panels = [(label, virtual_panel(obj, scene, parts, pose, center, span, view=view,
                                    fit_points=fit_points, show_directions=show_directions,
                                    support_z=support_z, support_label=support_label, edit_cues=edit_cues)[0])
              for label, view in ((side_label or 'SIDE / level view, world up: depth against the table', 'side'),
                                  ('TOP / straight down: yaw against the object', 'top'),
                                  ('CLOSING PLANE / between the fingers: opening vs thickness', 'closing'))]
    if edit_cues:
        # Three large square projections keep geometry legible in a single image.
        card = Image.new('RGB', (1440, 1280), 'white')
        draw = ImageDraw.Draw(card)
        ink = (24,38,58)
        draw.text((24, 14), title[:90], font=font(28), fill=ink)
        for index, line in enumerate(edit_cues['lines'][:2]):
            draw.text((24, 58+32*index), line, font=font(23), fill=ink)
        for (label,panel), x in zip(panels, (24,496,968)):
            draw.text((x,150), label.split(' / ')[0], font=font(27), fill=ink)
            card.paste(panel.resize((448,448), Image.Resampling.LANCZOS), (x,190))
        for axis,x in enumerate((24,496,968)):
            label = ('ROLL / local X','PITCH / local Y','YAW / local Z')[axis]
            draw.text((x,662),label,font=font(27),fill=ink)
            preview = rotation_panel(parts,obj,pose,edit_cues,axis)
            card.paste(preview.resize((448,386),Image.Resampling.LANCZOS),(x,702))
        frame = edit_cues['translation_frame']
        unit = edit_cues.get('translation_unit', 'mm' if edit_cues['kind'] == 'grasp' else 'm')
        draw.text((24,1110),f'MOVE: {frame.upper()} XYZ in {unit}  |  ROTATION EXAMPLES ONLY: no edit applied',font=font(24),fill=ink)
        totals = '   '.join(f'{key} {float(value):+.3g}' for key,value in (sliders or {}).items()) or 'No edits yet'
        draw.text((24,1152),'Command totals: '+totals,font=font(20),fill=(78,91,110))
        draw.text((24,1194),'DRAFT  >  validate this candidate  >  execute with a fresh token',font=font(24),fill=ink)
        draw.text((24,1240),edit_cues['scope'],font=font(20),fill=(78,91,110))
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); card.save(path)
        path.with_suffix('.cues.json').write_text(json.dumps(edit_cues, indent=2)+'\n')
        return path
    height = 44 + strip.height + 8 + sum(28 + p.height + 10 for _, p in panels) + 30 + (24 if note else 0)
    card = Image.new('RGB', (960, height), (255, 255, 255))
    draw = ImageDraw.Draw(card)
    draw.text((32, 12), title, font=font(24), fill=(24, 38, 58))
    y = 44
    card.paste(strip, (32, y)); y += strip.height + 8
    for label, panel in panels:
        draw.text((32, y+4), label, font=font(17), fill=(24, 38, 58)); y += 28
        card.paste(panel, (32, y)); y += panel.height + 10
    for x, color, text in [(32, COLORS['left_finger'], 'Finger A'), (150, COLORS['right_finger'], 'Finger B'),
                           (270, ORANGE, 'Measured object'), (440, SCENE_GREY, 'Scene / table points'),
                           (640, (96, 72, 40), 'Support surface (side view)')]:
        draw.rectangle((x, y+4, x+12, y+16), fill=color)
        draw.text((x+18, y), text, font=font(14), fill=(40, 55, 75))
    if note:
        draw.text((32, y+18), note, font=font(13), fill=(78, 91, 110))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    card.save(path)
    return path
