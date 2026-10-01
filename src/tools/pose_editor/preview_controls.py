"""Agent-controlled orbit/zoom of measured geometry and an immutable grasp pose."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw

VIEW_LIMITS = {'azimuth_deg': (-180., 180.), 'elevation_deg': (-85., 85.), 'zoom': (.5, 3.)}


def checked_view(azimuth_deg, elevation_deg, zoom):
    values = dict(azimuth_deg=azimuth_deg, elevation_deg=elevation_deg, zoom=zoom)
    for key, value in values.items():
        lo, hi = VIEW_LIMITS[key]
        if type(value) not in (int, float) or not np.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f'{key} must be a finite number in [{lo}, {hi}]')
    return values


def render_preview(geometry, prediction, output_dir, *, candidate_ref, azimuth_deg=-45.,
                   elevation_deg=30., zoom=1., graspgen_root=None, expected_open_width_m=None):
    from src.tools.pose_editor.inspection import load_panda_mesh, transform_points, checked_transform, _cloud
    from src.tools.grasp.input_cards import mesh_draw, arrow, font, ORANGE
    from src.tools.pose_editor.tiptop import observed_frame_points, REVISION
    view = checked_view(azimuth_deg, elevation_deg, zoom)
    pose = checked_transform(prediction.pose)
    obj, scene = _cloud(geometry.object_points), _cloud(geometry.scene_points)
    if not len(obj):
        raise ValueError('preview requires measured target points')
    # Include measured camera surfaces as context, without adding any to planning.
    frames = [g.frame for g in getattr(geometry, 'per_view', (geometry,)) if hasattr(g, 'frame')]
    measured = [observed_frame_points(f) for f in frames if hasattr(f, 'depth_m')]
    if measured:
        scene = np.concatenate([scene, *measured])
    parts, source = load_panda_mesh(graspgen_root, expected_open_width_m,
        libero_adapter=bool(getattr(prediction, 'gripper_adapter', None)))
    parts = {name: transform_points(vertices, pose) for name, vertices in parts.items()}
    center = (obj.min(0)+obj.max(0))/2
    az, el = np.radians([azimuth_deg, elevation_deg])
    right = np.array([-np.sin(az), np.cos(az), 0.])
    up = np.array([-np.sin(el)*np.cos(az), -np.sin(el)*np.sin(az), np.cos(el)])
    forward = -np.cross(right, up)
    basis = np.array([right, up])
    fit = np.concatenate([obj, *[v.reshape(-1, 3) for v in parts.values()]])
    span = max(.2, float(np.linalg.norm(np.ptp(fit, axis=0)))+.08)
    scale = 580/span*zoom
    def project(points):
        relative = np.asarray(points)-center
        return relative @ basis.T * [scale, -scale] + [480, 395], relative @ forward
    image = Image.new('RGB', (960, 800), (242, 246, 250))
    draw = ImageDraw.Draw(image)
    near = scene[np.linalg.norm(scene-center, axis=1) < span]
    for x, y in project(near[::max(1, len(near)//8000)])[0]:
        draw.point((x, y), fill=(176, 187, 202))
    for x, y in project(obj[::max(1, len(obj)//6000)])[0]:
        draw.ellipse((x-1, y-1, x+1, y+1), fill=ORANGE)
    mesh_draw(image, parts, project, translucent=True)
    draw = ImageDraw.Draw(image)
    # Local hand axes have the same meaning as refine_candidate roll/pitch/yaw.
    origin = project(pose[:3, 3])[0]
    for i, (name, color) in enumerate(zip(('roll X', 'pitch Y', 'yaw Z'),
                                        ((220, 55, 55), (40, 145, 65), (45, 85, 220)))):
        endpoint = project(pose[:3, 3]+pose[:3, i]*.06)[0]
        arrow(draw, origin, endpoint, color)
        draw.text(tuple(endpoint), name, font=font(16), fill=color)
    draw.rectangle((0, 0, 960, 92), fill='white')
    viewing = candidate_ref.startswith('view_')
    draw.text((24, 14), ('VIEWING POSE' if viewing else 'GRASP')+' PREVIEW / orbit and zoom', font=font(26), fill=(24, 38, 58))
    draw.text((24, 50), f'{candidate_ref[:20]}  azimuth {azimuth_deg:g}  elevation {elevation_deg:g}  zoom {zoom:g}', font=font(18), fill=(40, 55, 75))
    draw.rectangle((0, 700, 960, 800), fill='white')
    draw.text((24, 714), 'Virtual X-ray view of measured surfaces + proposed gripper; hidden surfaces unknown.', font=font(17), fill=(40, 55, 75))
    caption = ('Observation pose only; approve/refine, then backend validates before motion.' if viewing else
               'Orbit/zoom changes only the view. Pose correction requires a new route check.')
    draw.text((24, 744), caption, font=font(17), fill=(40, 55, 75))
    opening_caption = (f"NOMINAL OPEN {source['opening_m']*1000:g}mm; expected opening UNKNOWN" if expected_open_width_m is None else
                       f'EXPECTED total joint opening {source["opening_m"]*1000:.1f}mm (not measured)')
    draw.text((24, 774), opening_caption, font=font(16), fill=(40, 55, 75))
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory/'preview.png'
    image.save(path)
    metadata = dict(candidate_ref=candidate_ref, view=view, pose=pose.tolist(), mesh=source,
        observation_id=str(geometry.observation_id), preview_only=True, robot_motion=False,
        basis_rows=basis.tolist(), tiptop_revision=REVISION, measured_frame_count=len(measured),
        opening_caption=opening_caption)
    (directory/'preview.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    return {'image_paths': [str(path)], 'metadata': metadata}
