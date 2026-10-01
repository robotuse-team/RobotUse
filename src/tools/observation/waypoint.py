"""Two calibrated image rays to a reviewed, free-space EE waypoint."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw


# Pinned Panda finger STL at LIBERO z=52.4 mm, relative to the 97 mm EE site.
PANDA_FINGERTIP_EXTENSION_M = .009249034011364
PUSH_SUPPORT_CLEARANCE_M = .002


def measured_surface_point(frame, pixel):
    """Unproject one explicitly selected visible surface using measured depth."""
    uv = np.asarray(pixel, float)
    if uv.shape != (2,) or not np.isfinite(uv).all() or np.any(uv < 0) or np.any(uv > 1000):
        raise ValueError('normalized surface point required')
    px = uv * [(frame.width-1)/1000, (frame.height-1)/1000]
    x, y = np.rint(px).astype(int)
    depth = float(frame.depth_m[y, x])
    if not np.isfinite(depth) or not .05 < depth < 3:
        raise ValueError('selected surface has no valid measured depth')
    rotation = np.asarray(frame.camera_to_base.rotation)
    origin = np.asarray(frame.camera_to_base.translation)
    surface = origin + rotation @ (np.linalg.solve(frame.intrinsics, [*px, 1.]) * depth)
    return surface, px


def measured_push_point(frame, pixel, stage):
    """A visible support point plus the calibrated downward fingertip clearance."""
    if stage not in ('approach', 'contact'):
        raise ValueError('approach/contact stage required')
    surface, px = measured_surface_point(frame, pixel)
    xyz = surface + [0, 0, .18 if stage == 'approach' else PANDA_FINGERTIP_EXTENSION_M + PUSH_SUPPORT_CLEARANCE_M]
    camera = np.asarray(frame.camera_to_base.rotation).T @ (xyz-np.asarray(frame.camera_to_base.translation))
    projected = np.asarray(frame.intrinsics) @ camera
    return xyz, {'requested_pixels': [px.tolist()], 'projected_pixels': [(projected[:2]/projected[2]).tolist()],
        'method': 'selected measured RGBD support + declared Panda fingertip clearance',
        'surface_point_base': surface.tolist(), 'stage': stage,
        'fingertip_extension_m': PANDA_FINGERTIP_EXTENSION_M, 'support_clearance_m': PUSH_SUPPORT_CLEARANCE_M, 'clearance_m': float(xyz[2]-surface[2])}


def horizontal_contact_point(frame, pixel, height_m):
    """One measured camera ray intersected with the current robot EE height.

    A held horizontal contact supplies the plane that free-space stereo lacks.
    No depth image or simulator object state supplies the destination.
    """
    uv = np.asarray(pixel, float)
    if uv.shape != (2,) or not np.isfinite(uv).all() or np.any(uv < 0) or np.any(uv > 1000) or not np.isfinite(height_m):
        raise ValueError('finite normalized image point and contact height required')
    px = uv * [(frame.width-1)/1000, (frame.height-1)/1000]
    origin = np.asarray(frame.camera_to_base.translation)
    ray = np.asarray(frame.camera_to_base.rotation) @ np.linalg.solve(frame.intrinsics, [*px, 1.])
    if abs(ray[2]) < 1e-6:
        raise ValueError('camera ray is parallel to contact plane')
    distance = (height_m-origin[2])/ray[2]
    if distance <= 0:
        raise ValueError('contact destination is behind camera')
    xyz = origin + distance*ray
    return xyz, {'requested_pixels': [px.tolist()], 'projected_pixels': [px.tolist()],
        'method': 'calibrated camera ray / current EE horizontal contact plane',
        'contact_height_m': float(height_m), 'height_source': 'current robot EE proprioception'}


def object_destination_point(frame, source_pixel, destination_pixel, current_ee, *, contact_height_m=None):
    """Translate the hand by a clicked object's horizontal displacement.

    Only the source uses measured depth. The destination ray intersects that
    source height, so clicking free space never substitutes background depth.
    The hand/object offset is retained instead of moving the EE to the object centre.
    """
    source, source_px = measured_surface_point(frame, source_pixel)
    destination, _ = horizontal_contact_point(frame, destination_pixel, source[2])
    current = np.asarray(current_ee, float)
    if current.shape != (3,) or not np.isfinite(current).all():
        raise ValueError('finite measured EE position required')
    xyz = current + destination - source
    if contact_height_m is not None:
        if not np.isfinite(contact_height_m):
            raise ValueError('finite contact height required')
        xyz[2] = contact_height_m
    from src.tools.grasp.input_cards import camera_project
    projected, _ = camera_project(np.asarray([xyz]), frame)
    if not np.isfinite(projected).all():
        raise ValueError('EE destination cannot be projected in selected view')
    uv = np.asarray(destination_pixel) * [(frame.width-1)/1000, (frame.height-1)/1000]
    return xyz, {'method': 'clicked object source RGBD / destination ray at source height; preserve hand offset',
        'requested_pixels': [uv.tolist()], 'projected_pixels': projected.tolist(),
        'object_source_pixels': [source_px.tolist()], 'object_source_base': source.tolist(),
        'object_destination_base': destination.tolist(), 'object_displacement_base': (destination-source).tolist(),
        'current_ee_base': current.tolist(), 'contact_height_m': float(xyz[2])}


def triangulate(frames, pixels, *, max_reprojection_px=12., min_ray_angle_deg=3.):
    if len(frames) != 2 or len(pixels) != 2:
        raise ValueError('two calibrated views and corresponding points required')
    origins, rays, image_points = [], [], []
    for frame, uv in zip(frames, pixels):
        uv = np.asarray(uv, float)
        if uv.shape != (2,) or not np.isfinite(uv).all() or np.any(uv < 0) or np.any(uv > 1000):
            raise ValueError('image points must be finite normalized coordinates in [0,1000]')
        pixel = uv * [(frame.width-1)/1000, (frame.height-1)/1000]
        ray = np.asarray(frame.camera_to_base.rotation) @ np.linalg.solve(frame.intrinsics, [*pixel, 1.])
        rays.append(ray / np.linalg.norm(ray))
        origins.append(np.asarray(frame.camera_to_base.translation))
        image_points.append(pixel)
    cosine = np.clip(rays[0] @ rays[1], -1., 1.)
    angle = np.degrees(np.arccos(abs(cosine)))
    if angle < min_ray_angle_deg:
        raise ValueError('rays nearly parallel; choose a better constrained corresponding point')
    distances = np.linalg.lstsq(np.column_stack((rays[0], -rays[1])), origins[1]-origins[0], rcond=None)[0]
    if np.any(distances <= 0):
        raise ValueError('point is behind a camera; choose a location visible in both views')
    closest = [o+t*d for o,t,d in zip(origins, distances, rays)]
    xyz = np.mean(closest, axis=0)
    projected, errors = [], []
    for frame, pixel in zip(frames, image_points):
        camera = np.asarray(frame.camera_to_base.rotation).T @ (xyz-np.asarray(frame.camera_to_base.translation))
        if camera[2] <= .01:
            raise ValueError('waypoint too near or behind camera')
        p = np.asarray(frame.intrinsics) @ camera
        projected.append(p[:2]/p[2]); errors.append(float(np.linalg.norm(projected[-1]-pixel)))
    if max(errors) > max_reprojection_px:
        raise ValueError(f'points disagree across views (reprojection error {max(errors):.1f}px); correct the pair')
    return xyz, dict(ray_angle_deg=float(angle), ray_gap_m=float(np.linalg.norm(closest[0]-closest[1])),
                     reprojection_error_px=errors, projected_pixels=[p.tolist() for p in projected],
                     requested_pixels=[p.tolist() for p in image_points], method='closest calibrated rays; no depth-surface substitution')


def render_waypoint(frames, audit, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    paths = []
    for i, frame in enumerate(frames):
        im = Image.fromarray(np.asarray(frame.rgb).copy()); draw = ImageDraw.Draw(im)
        for pixel,color in [(audit['requested_pixels'][i], 'yellow'), (audit['projected_pixels'][i], 'cyan')]:
            x,y=pixel; draw.ellipse((x-7,y-7,x+7,y+7),outline=color,width=3)
        draw.text((8,8),'Waypoint: yellow=requested; cyan=EE target',fill='white',stroke_fill='black',stroke_width=1)
        if 'object_source_pixels' in audit:
            from src.tools.grasp.input_cards import arrow
            x, y = audit['object_source_pixels'][i]
            draw.ellipse((x-7,y-7,x+7,y+7), outline='lime', width=3)
            arrow(draw, (x,y), audit['requested_pixels'][i], 'lime', width=3)
            draw.text((8,42),'OBJECT: green=now -> yellow=destination; cyan=hand target',
                      fill='lime', stroke_fill='black', stroke_width=1)
        if 'current_ee_base' in audit:
            from src.tools.grasp.input_cards import camera_project, arrow
            current, _ = camera_project(np.asarray([audit['current_ee_base']]), frame)
            if np.isfinite(current).all():
                x, y = current[0]
                draw.ellipse((x-8,y-8,x+8,y+8), outline='orange', width=3)
                arrow(draw, current[0], audit['projected_pixels'][i], 'orange', width=3)
                draw.text((8,25),'Orange: CURRENT EE -> requested motion (robot measurement)', fill='orange',stroke_fill='black',stroke_width=1)
        path = output / (frame.view_id+'.png'); im.save(path); paths.append(str(path))
    (output/'triangulation.json').write_text(json.dumps(audit,indent=2)+'\n')
    return paths
