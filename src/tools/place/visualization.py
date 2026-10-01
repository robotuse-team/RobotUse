"""Place review cards sharing Grasp's calibrated RGB and Panda mesh renderer."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from src.tools.grasp.input_cards import rgb_panel, virtual_panel, font, COLORS, ORANGE
from src.tools.pose_editor.inspection import load_panda_mesh, transform_points
from src.tools.motion.planning import _pose_transform


def containment_metrics(placed, destination_points):
    """Numbers the reviewer cannot read off an x-ray overlay: is the object inside, and how deep."""
    from scipy.spatial import Delaunay, QhullError

    footprint = destination_points[:, :2]
    try:
        inside = float((Delaunay(footprint).find_simplex(placed[:, :2]) >= 0).mean())
    except QhullError:  # a degenerate footprint has no hull; the bounding box still bounds it
        inside = float(((placed[:, :2] >= footprint.min(0)) & (placed[:, :2] <= footprint.max(0))).all(1).mean())
    center = (footprint.min(0) + footprint.max(0)) / 2
    return {
        'inside_xy_fraction': inside,
        'bottom_below_destination_top_m': float(destination_points[:, 2].max() - placed[:, 2].min()),
        'center_offset_xy_m': float(np.linalg.norm(placed[:, :2].mean(0) - center)),
        'drop_height_m': float(placed[:, 2].min() - destination_points[:, 2].min()),
        'scope': 'measured destination cloud only; occluded surfaces unknown',
    }


def _metrics_line(metrics):
    below = metrics['bottom_below_destination_top_m'] * 1000
    return ('Containment | XY inside destination {:.0%} | bottom {:.0f} mm {} destination top'
            ' | center offset {:.0f} mm | drop {:.0f} mm').format(
        metrics['inside_xy_fraction'], abs(below), 'below' if below >= 0 else 'above',
        metrics['center_offset_xy_m'] * 1000, metrics['drop_height_m'] * 1000)


def render_placement(destination, object_points, transform, plan, directory, *, grasp_to_ee, mesh_source=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    memory = json.loads((destination.directory / 'memory.json').read_text())
    snapshot = json.loads(Path(memory['calibrated_rgbd_snapshot']).read_text())
    capture_timing = memory.get('acquisition', 'saved destination observation')
    placed = transform_points(object_points, transform)
    placed[:, 2] += plan.release_clearance_m
    hand = _pose_transform(plan.targets[plan.target_labels.index('release')]) @ np.linalg.inv(grasp_to_ee)
    parts, mesh_meta = load_panda_mesh(mesh_source, expected_open_width_m=plan.jaw_width_m)
    parts = {name: transform_points(triangles, hand) for name, triangles in parts.items()}
    metrics = containment_metrics(placed, destination.points)
    center = (destination.points.min(0) + destination.points.max(0)) / 2
    # All choices use one destination-bound crop and scale, as Grasp uses one target-bound crop.
    span = max(.32, float(np.linalg.norm(np.ptp(destination.points, axis=0))) + .24)
    views = sorted(snapshot['views'], key=lambda v: {'agentview': 0, 'robot0_eye_in_hand': 1}[v['view_id']])
    if len(views) != 2 or {v['view_id'] for v in views} != {'agentview', 'robot0_eye_in_hand'}:
        raise ValueError('place cards require the saved front/wrist pair')
    panels = []
    for view in views:
        calibration = json.loads(Path(view['calibration_path']).read_text())
        frame = SimpleNamespace(view_id=view['view_id'], rgb_path=view['rgb_path'],
            intrinsics=calibration['intrinsics'], camera_to_base=SimpleNamespace(**calibration['camera_to_base']))
        with Image.open(frame.rgb_path) as source:
            if source.size != (calibration['width'], calibration['height']):
                raise ValueError('saved RGB and calibration dimensions differ')
        panel, evidence = rgb_panel(frame, placed, parts, center, span, crop_points=destination.points)
        evidence.update(source_rgb_sha256=hashlib.sha256(Path(frame.rgb_path).read_bytes()).hexdigest(),
                        calibration_path=view['calibration_path'])
        panels.append((panel, evidence))
    paths, virtual_views = [], []
    scene = np.vstack((destination.scene, destination.points))
    for index in range(2):
        card = Image.new('RGB', (960, 960), 'white')
        draw = ImageDraw.Draw(card)
        title = getattr(plan, 'validation_label', 'PLACE CANDIDATE / ' + directory.name[:18])
        draw.text((32, 18), title, font=font(26), fill=(24, 38, 58))
        draw.text((32, 52), f'Predicted release | measured jaw {plan.jaw_width_m*1000:.0f} mm | offset {plan.release_clearance_m*1000:.0f} mm',
                  font=font(17), fill=(78, 91, 110))
        draw.text((32, 73), _metrics_line(metrics), font=font(15), fill=(24, 38, 58))
        for col, (panel, _) in enumerate(panels):
            draw.text((32 + col*448, 92), ('SAVED FRONT', 'SAVED WRIST')[col] + ' / capture', font=font(20), fill=(24, 38, 58))
            card.paste(panel, (32 + col*448, 123))
        draw.text((32, 443), 'Saved RGB + predicted release overlay; NOT a future camera image', font=font(18), fill=(78, 91, 110))
        draw.text((32, 480), 'RELEASE / closing plane' if index else 'RELEASE / object and destination context', font=font(22), fill=(24, 38, 58))
        panel, evidence = virtual_panel(placed, scene, parts, hand, center, span, bool(index),
                                       fit_points=destination.points, show_directions=False)
        card.paste(panel, (32, 517))
        for x, color, label in [(32, COLORS['left_finger'], 'Finger A'), (180, COLORS['right_finger'], 'Finger B'),
                                (330, ORANGE, 'Predicted object'), (560, (192, 202, 214), 'Saved scene / destination')]:
            draw.rectangle((x, 920, x+12, 932), fill=color)
            draw.text((x+18, 916), label, font=font(16), fill=(40, 55, 75))
        draw.text((32, 940), 'X-ray release proposal. Lift, transit, opening and retreat/path collisions unchecked.', font=font(14), fill=(78, 91, 110))
        path = directory / f'preview-{index}.png'
        card.save(path)
        paths.append(path)
        evidence['target_rendering'] = 'predicted transformed measured object; not observed at release'
        virtual_views.append(evidence)
    (directory / 'preview-metadata.json').write_text(json.dumps({
        'schema': 'place-review-card.v1', 'renderer': 'grasp_input_cards.rgb_panel/virtual_panel',
        'observation_id': snapshot['observation_id'], 'views': [p[1] for p in panels],
        'virtual_views': virtual_views, 'gripper': mesh_meta, 'release_hand_pose': hand.tolist(),
        'release_clearance_m': plan.release_clearance_m,
        'containment': metrics,
        'object_display_pose': 'at release, same instant as attached gripper',
        'camera_semantics': 'saved capture; overlays are predictions, including hidden surfaces',
        'capture_timing': capture_timing,
        'validation_label': getattr(plan, 'validation_label', 'validated candidate'),
        'path_description': 'lift -> transit -> release -> retreat; stored planned route, not depicted as a straight path',
    }, indent=2) + '\n')
    return paths


def candidate_overviews(entries, directory):
    """One complete paired-camera/scene card per candidate fits the model image budget."""
    if len(entries) > 4:
        raise ValueError('sample at most four passing candidates before rendering')
    return [Path(directory) / entry['candidate_ref'] / 'preview-0.png' for entry in entries]
