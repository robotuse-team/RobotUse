"""CPU rendering of actual Panda visual meshes onto calibrated camera evidence.

The z-buffer handles self-occlusion and optional registered optical-Z scene depth.
The overlay is a pose proposal, never a contact or motion check.
NumPy and Pillow are sufficient; no simulator, browser or model call is made.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.runtime.paths import REPOSITORY_ROOT as ROOT
from src.tools.pose_editor import mesh_assets
_NEAR = .001
# The single place the modelled gripper is named. Every other asset path,
# actuator endpoint and proxy dimension below is derived from this file.
GRIPPER_XML = ROOT / 'src/tools/pose_editor/third_party/robosuite/robosuite/models/assets/grippers/panda_gripper.xml'


@lru_cache(maxsize=1)
def open_gripper_joint_positions():
    """Use the actual XML actuator endpoints commanded by open_gripper()."""
    xml = ET.parse(GRIPPER_XML)
    joints = {}
    for actuator in xml.findall('actuator/position'):
        if actuator.get('joint') in ('finger_joint1', 'finger_joint2'):
            limits = tuple(map(float, actuator.attrib['ctrlrange'].split()))
            joints[actuator.attrib['joint']] = max(limits, key=abs)
    if set(joints) != {'finger_joint1', 'finger_joint2'}:
        raise ValueError('Panda open-gripper actuator calibration is missing')
    return joints


def open_gripper_width_m():
    joints = open_gripper_joint_positions()
    return joints['finger_joint1'] - joints['finger_joint2']


def proxy_geometry():
    """Illustrative wireframe stand-in for the visual mesh, in the gripper frame.

    A rectangular palm and two rectangular fingers, +Z toward the contact. It
    lives beside the mesh loader so one GRIPPER_XML drives both representations.
    """
    jaw, finger, axis = open_gripper_width_m(), .06, .09
    boxes = [((-.014, -jaw / 2 - .006, -.075), (.014, jaw / 2 + .006, -.06))]
    for side in (-1, 1):
        center = side * jaw / 2
        boxes.append(((-.010, center - .006, -finger), (.010, center + .006, .012)))
    return {'axis_length_m': axis, 'jaw_width_m': jaw, 'finger_length_m': finger,
            'boxes': tuple(boxes)}


@lru_cache(maxsize=32, typed=True)
def _load_mesh_parts(expected_open_width_m=None):
    """Read source assets, applying the same XML transforms as the WebGL export."""
    assets, xml_path = GRIPPER_XML.parent, GRIPPER_XML
    xml = ET.parse(xml_path).getroot()
    mesh_paths = {e.attrib['name']: assets / e.attrib['file'] for e in xml.findall('asset/mesh')}
    sites, geoms = {}, []
    joints = open_gripper_joint_positions()
    if expected_open_width_m is not None:
        if (isinstance(expected_open_width_m, bool) or not math.isfinite(expected_open_width_m)
                or not 0 <= expected_open_width_m <= open_gripper_width_m()):
            raise ValueError('expected total Panda joint opening outside calibrated range')
        joints = {'finger_joint1': expected_open_width_m/2,
                  'finger_joint2': -expected_open_width_m/2}

    def visit(body, parent):
        frame = mesh_assets.compose(parent, mesh_assets.local_transform(body))
        for joint in body.findall('joint'):
            axis = tuple(map(float, joint.get('axis', '0 0 1').split()))
            displacement = tuple(joints.get(joint.get('name'), 0.)*x for x in axis)
            frame = mesh_assets.compose(frame, (mesh_assets.IDENTITY, displacement))
        for site in body.findall('site'):
            sites[site.attrib['name']] = mesh_assets.compose(frame, mesh_assets.local_transform(site))
        for geom in body.findall('geom'):
            if geom.get('group') == '1' and geom.get('mesh'):
                geoms.append((geom, mesh_assets.compose(frame, mesh_assets.local_transform(geom))))
        for child in body.findall('body'):
            visit(child, frame)

    for body in xml.findall('worldbody/body'):
        visit(body, (mesh_assets.IDENTITY, (0., 0., 0.)))
    transform = mesh_assets.compose((mesh_assets.DISPLAY_FROM_GRIP, (0., 0., 0.)), mesh_assets.inverse(sites['grip_site']))
    parts, paths = [], {xml_path, Path(mesh_assets.__file__).resolve()}
    for geom, frame in geoms:
        source = mesh_paths[geom.attrib['mesh']]
        paths.add(source)
        source_transform = mesh_assets.compose(transform, frame)
        mesh = mesh_assets.indexed_mesh(mesh_assets.binary_stl(source), source_transform)
        parts.append({
            'name': geom.attrib['name'],
            'source_to_display': {'rotation': source_transform[0], 'translation': source_transform[1]},
            'positions': np.asarray(mesh['positions'], dtype=np.float64).reshape(-1, 3),
            'normals': np.asarray(mesh['normals'], dtype=np.float64).reshape(-1, 3),
            'indices': np.asarray(mesh['indices'], dtype=np.int32).reshape(-1, 3),
            'color': np.array((.25, .91, .78) if geom.attrib['mesh'] == 'hand_vis' else (.29, .41, .45)),
        })
    all_vertices = np.concatenate([p['positions'] for p in parts])
    metadata = {
        'geometry': 'robosuite Panda visual STL; XML transforms; configured finger joint positions',
        'renderer': 'CPU depth-buffered triangles, perspective-correct interpolation, smooth lighting, 2x supersampling',
        'frame': 'grip_site origin; +Z approach; +Y jaw opening; native grip_site rotated -90 degrees about Z',
        'scene_occlusion': False,
        'nominal_jaw_separation_m': open_gripper_width_m(),
        'expected_joint_opening_m': expected_open_width_m,
        'finger_joint_positions_m': dict(joints),
        'jaw_state': ('nominal fully open reference; expected opening unknown' if expected_open_width_m is None else
                      'expected total finger joint-axis spacing, not measured or inner surface gap'),
        'local_bounds_m': {'min': all_vertices.min(axis=0).tolist(), 'max': all_vertices.max(axis=0).tolist()},
        'dimensions_m': (all_vertices.max(axis=0)-all_vertices.min(axis=0)).tolist(),
        'display_from_grip_site_rotation': mesh_assets.DISPLAY_FROM_GRIP,
        'parts': [{'name': p['name'], 'source_to_display': p['source_to_display'],
                   'triangle_count': len(p['indices'])} for p in parts],
        'sources': [{'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(paths)],
    }
    return parts, metadata


def mesh_source_metadata(expected_open_width_m=None):
    """Expose asset provenance and rendering limits for preview manifests."""
    return dict(_load_mesh_parts(expected_open_width_m)[1])


def mesh_metadata():
    """Return default-opening mesh transforms, bounds and source hashes."""
    return mesh_source_metadata()


def _clip_near(points, normals):
    """Clip one triangle with interpolated normals against positive camera Z."""
    polygon = list(zip(points, normals))
    output = []
    for index, (current, normal) in enumerate(polygon):
        previous, previous_normal = polygon[index-1]
        inside, was_inside = current[2] >= _NEAR, previous[2] >= _NEAR
        if inside != was_inside:
            t = (_NEAR-previous[2]) / (current[2]-previous[2])
            point = previous + t*(current-previous)
            point[2] = _NEAR
            output.append((point, previous_normal + t*(normal-previous_normal)))
        if inside:
            output.append((current, normal))
    return [(np.asarray([output[0][0], output[i][0], output[i+1][0]]),
             np.asarray([output[0][1], output[i][1], output[i+1][1]]))
            for i in range(1, len(output)-1)]


def _rasterize(points, normals, color, intrinsics, depth, layer):
    projected = points @ intrinsics.T
    pixels = projected[:, :2] / projected[:, 2:3]
    height, width = depth.shape
    lower = np.maximum(np.floor(pixels.min(axis=0)).astype(int), [0, 0])
    upper = np.minimum(np.ceil(pixels.max(axis=0)).astype(int), [width-1, height-1])
    if np.any(lower > upper):
        return
    a, b, c = pixels
    denominator = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
    if abs(denominator) < 1e-10:
        return
    xs, ys = np.meshgrid(np.arange(lower[0], upper[0]+1)+.5, np.arange(lower[1], upper[1]+1)+.5)
    u = ((b[1]-c[1])*(xs-c[0]) + (c[0]-b[0])*(ys-c[1])) / denominator
    v = ((c[1]-a[1])*(xs-c[0]) + (a[0]-c[0])*(ys-c[1])) / denominator
    w = 1-u-v
    inside = (u >= -1e-7) & (v >= -1e-7) & (w >= -1e-7)
    reciprocal = u/points[0, 2] + v/points[1, 2] + w/points[2, 2]
    z = np.divide(1., reciprocal, out=np.full_like(reciprocal, np.inf), where=reciprocal > 0)
    region = depth[lower[1]:upper[1]+1, lower[0]:upper[0]+1]
    visible = inside & (z < region)
    if not visible.any():
        return
    weights = np.stack([u[visible]/points[0, 2], v[visible]/points[1, 2], w[visible]/points[2, 2]], axis=-1)
    weights *= z[visible, None]
    normal = weights @ normals
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    position = weights @ points
    view = -position / np.maximum(np.linalg.norm(position, axis=1, keepdims=True), 1e-12)
    # Double-sided shading keeps clipped surfaces legible without invented geometry.
    normal *= np.where(np.sum(normal*view, axis=1, keepdims=True) < 0, -1., 1.)
    light = np.array([-.45, -.65, -1.]); light /= np.linalg.norm(light)
    half_vector = view + light
    half_vector /= np.maximum(np.linalg.norm(half_vector, axis=1, keepdims=True), 1e-12)
    diffuse = np.maximum(normal @ light, 0.)
    specular = np.maximum(np.sum(normal*half_vector, axis=1), 0.)**42
    rim = (1-np.clip(np.sum(normal*view, axis=1), 0, 1))**3
    shaded = color[None, :]*(.34 + .66*diffuse[:, None]) + .40*specular[:, None] + .10*rim[:, None]
    region[visible] = z[visible]
    layer[lower[1]:upper[1]+1, lower[0]:upper[0]+1][visible] = np.clip(shaded*255, 0, 255)


@lru_cache(maxsize=8)
def _axis_font(size):
    """A real sized face where one exists; the bitmap default is unreadable."""
    for path in ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def scene_visibility(mesh_depth, scene_depth, *, tolerance_m=.002):
    """Classify calibrated optical-Z samples; unknown scene depth cannot certify visibility."""
    if not math.isfinite(tolerance_m) or not 0 <= tolerance_m <= .01:
        raise ValueError('depth tolerance must be between zero and .01 metres')
    observed = np.asarray(scene_depth)
    if observed.shape != mesh_depth.shape or observed.dtype.kind not in 'fiu':
        raise ValueError('scene depth dimensions/type must match camera calibration')
    mesh = np.isfinite(mesh_depth) & (mesh_depth > 0)
    valid = np.isfinite(observed) & (observed > 0)
    visible = mesh & valid & (mesh_depth <= observed + tolerance_m)
    hidden = mesh & valid & ~visible
    unknown = mesh & ~valid
    return visible, {'mesh_samples': int(mesh.sum()), 'visible_samples': int(visible.sum()),
                     'occluded_samples': int(hidden.sum()), 'unknown_depth_samples': int(unknown.sum()),
                     'tolerance_m': tolerance_m, 'depth_convention': 'metres_optical_z',
                     'unknown_depth_policy': 'hidden_in_depth_panel_visible_in_xray'}


def render_mesh_overlay(rgb, camera, anchor_base, rotation_base_from_gripper, *, opacity=.88, show_axes=False,
                        scene_depth=None, depth_tolerance_m=.002, diagnostics=None, xray_output=None,
                        expected_open_width_m=None, axes_rotation_base=None):
    """Return an antialiased solid-gripper overlay, without modifying ``rgb``.

    ``camera`` accepts visual_pose._Camera or a view with intrinsics and a
    camera-to-base pose. Orientation is a proper gripper-to-base 3x3 rotation.
    The caller supplies the anchor in base metres.
    Optional scene_depth is registered RGB-sized metric optical Z, not normalized
    depth or ray distance. The xray companion intentionally ignores scene depth.
    ``axes_rotation_base`` optionally selects the coordinate glyph frame only;
    mesh vertices, normals, depth and the anchor retain the supplied mesh pose.
    """
    from src.tools.pose_editor.visual_pose import _camera
    if not hasattr(camera, 'origin') or not hasattr(camera, 'rotation'):
        camera = _camera(camera)
    if not math.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError('opacity must be between zero and one')
    image = rgb.convert('RGB').copy() if isinstance(rgb, Image.Image) else Image.fromarray(np.asarray(rgb)).convert('RGB')
    if image.size != (camera.width, camera.height):
        raise ValueError('RGB dimensions must match camera calibration')
    anchor = np.asarray(anchor_base, dtype=float)
    rotation = np.asarray(rotation_base_from_gripper, dtype=float)
    if anchor.shape != (3,) or rotation.shape != (3, 3) or not np.isfinite(anchor).all() or not np.isfinite(rotation).all():
        raise ValueError('anchor and orientation must have finite 3D coordinates')
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6):
        raise ValueError('orientation must be a proper rotation matrix')
    axes_rotation = rotation if axes_rotation_base is None else np.asarray(axes_rotation_base, dtype=float)
    if (axes_rotation.shape != (3, 3) or not np.isfinite(axes_rotation).all()
            or not np.allclose(axes_rotation.T @ axes_rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(axes_rotation), 1., atol=1e-6)):
        raise ValueError('axis orientation must be a proper rotation matrix')
    scale = 2
    width, height = image.width*scale, image.height*scale
    depth = np.full((height, width), np.inf)
    layer = np.zeros((height, width, 3), dtype=np.float64)
    intrinsics = np.asarray(camera.intrinsics, dtype=float).copy()
    intrinsics[:2] *= scale
    base_from_camera = np.asarray(camera.rotation, dtype=float)
    camera_from_gripper = base_from_camera.T @ rotation
    camera_from_axes = base_from_camera.T @ axes_rotation
    camera_anchor = base_from_camera.T @ (anchor-np.asarray(camera.origin))
    parts, _ = _load_mesh_parts(expected_open_width_m)
    for part in parts:
        positions = part['positions'] @ camera_from_gripper.T + camera_anchor
        normals = part['normals'] @ camera_from_gripper.T
        for triangle in part['indices']:
            points, ns = positions[triangle], normals[triangle]
            if np.all(points[:, 2] < _NEAR):
                continue
            clipped = [(points, ns)] if np.all(points[:, 2] >= _NEAR) else _clip_near(points, ns)
            for p, n in clipped:
                _rasterize(p, n, part['color'], intrinsics, depth, layer)
    def composite(mask):
        overlay = Image.fromarray(np.uint8(np.clip(layer, 0, 255)), mode='RGB')
        overlay.putalpha(Image.fromarray(np.uint8(mask.astype(float)*255*opacity), mode='L'))
        overlay = overlay.resize(image.size, Image.Resampling.LANCZOS)
        return Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
    mask = np.isfinite(depth)
    if xray_output is not None:
        xray_output.append(composite(mask))
    if scene_depth is not None:
        observed = np.asarray(scene_depth)
        if observed.shape != (image.height, image.width) or observed.dtype.kind not in 'fiu':
            raise ValueError('scene depth dimensions/type must match camera calibration')
        # Nearest replication avoids inventing depth at object edges.
        observed = np.repeat(np.repeat(observed, scale, axis=0), scale, axis=1)
        mask, stats = scene_visibility(depth, observed, tolerance_m=depth_tolerance_m)
        if diagnostics is not None:
            diagnostics.update(stats, supersampling=scale, scene_occlusion=True)
    result = composite(mask)
    if show_axes:
        draw = ImageDraw.Draw(result)
        k = np.asarray(camera.intrinsics)
        if camera_anchor[2] >= _NEAR:
            p = k @ camera_anchor; p = p[:2]/p[2]
            # Sized against the drawing, so a closeup gets proportionate axes.
            span = max(result.width, result.height)
            casing, stroke = max(2, round(span/110)), max(1, round(span/260))
            width, label = max(1, round(span/200)), _axis_font(max(13, round(span/26)))
            # Y avoids the mesh's own mint; every mark is cased in near-black so
            # it stays legible over both the gripper and the photograph.
            for i, color in enumerate(('#ff4d5e', '#d7ff2e', '#4ea8ff')):
                tip = camera_anchor + .065*camera_from_axes[:, i]
                if tip[2] >= _NEAR:
                    q = k @ tip; q = q[:2]/q[2]
                    draw.line([tuple(p), tuple(q)], fill='#0b1016', width=casing)
                    draw.line([tuple(p), tuple(q)], fill=color, width=width)
                    draw.text(tuple(q), 'XYZ'[i], fill=color, font=label, anchor='mm',
                              stroke_width=stroke, stroke_fill='#0b1016')
    return result
