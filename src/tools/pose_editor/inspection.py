"""Audited metric Panda orientation/space context, never collision prediction.

CPU-only; no GraspGen import, inference, SAM, IK, collision gate, or asset download.
PointGeometry clouds and GraspPrediction.pose MUST share the connector frame.
Mesh origin is the GraspGen HAND origin, NOT the public TCP or planner EE.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import struct
from typing import TYPE_CHECKING, Mapping

from src.runtime.paths import REPOSITORY_ROOT

import numpy as np
from PIL import Image, ImageDraw

if TYPE_CHECKING:
    from src.tools.perception.rgbd_adapter import PointGeometry
    from src.tools.grasp.prediction import GraspPrediction

PINNED_REVISION = "2dd8852e1be60f5f9d277fafcc621835cdf59110"
ASSET_HASHES = {
    "assets/panda_gripper/hand.stl": "65983ed489e673599d05b048dda3c883a0bd59823caed149b1d561525f30f52a",
    "assets/panda_gripper/finger.stl": "b69c173da7c8986281307b51f6181a8d0827c8727eaf87f6a959fa3098567ee2",
    "config/grippers/franka_panda.py": "2b895f760bd5d7e172fe9c30dba48a3838905e0c021ad7f9affa8383dc0f637f",
    "config/grippers/franka_panda.yaml": "91321c3a7de51e0d1a1c9ed57cddf75d0749fe29c2b6c32f051581c227c480c2",
}
GRASP_TO_EE = np.array([[0., -1., 0., 0.], [1., 0., 0., 0.],
                        [0., 0., 1., .097], [0., 0., 0., 1.]])
LIMITATIONS = (
    "Static mesh preview: orientation/space context, NOT collision prediction.",
    "Observed surface points only; holes/occlusion are UNKNOWN, not free space.",
    "No future whole-arm pose, IK, reachability, or execution is predicted.",
    "No angle-based rejection; side approaches may be reasonable.",
)


def checked_transform(value):
    t = np.asarray(value, dtype=float)
    if (t.shape != (4, 4) or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-4)):
        raise ValueError("expected finite rigid hand-origin transform")
    return t


def transform_points(points, pose):
    """Apply exactly one hand-origin transform; never append TCP calibration."""
    t = checked_transform(pose)
    return np.asarray(points) @ t[:3, :3].T + t[:3, 3]


def _stl_triangles(data):
    if len(data) < 84:
        raise ValueError("invalid binary STL")
    n = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + 50 * n:
        raise ValueError("expected audited binary STL")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
    return np.frombuffer(data, dtype=dtype, offset=84, count=n)["vertices"].astype(float)


def load_panda_mesh(graspgen_root=None, expected_open_width_m=None, *, libero_adapter=False):
    """Reproduce pinned GripperModel assembly from raw assets (already meters).

    The source constructor places finger joint origins at +/-40mm and z58.4mm.
    Do NOT call its set_offset: that applies another z translation. A supplied
    width is the expected total finger joint opening, not a measured object size.
    """
    provider = getattr(graspgen_root, 'load_gripper_mesh', None)
    if callable(provider):
        if libero_adapter:
            raise ValueError('LIBERO Panda retargeting cannot be applied to a native gripper')
        return provider(expected_open_width_m)
    root = Path(graspgen_root) if graspgen_root is not None else REPOSITORY_ROOT / "runtime/graspgen"
    data = {}
    for relative, digest in ASSET_HASHES.items():
        raw = (root / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"unaudited Panda asset/source: {relative}")
        data[relative] = raw
    nominal = expected_open_width_m is None
    width = .08 if nominal else float(expected_open_width_m)
    if isinstance(expected_open_width_m, bool) or not np.isfinite(width) or not 0 <= width <= .08:
        raise ValueError("expected Panda joint opening in [0, .08] meters")
    hand = _stl_triangles(data["assets/panda_gripper/hand.stl"])
    finger = _stl_triangles(data["assets/panda_gripper/finger.stl"])
    left = finger * [-1, -1, 1] + [width / 2, 0, .0584]
    right = finger + [-width / 2, 0, .0584]
    parts = {"hand": hand, "left_finger": left, "right_finger": right}
    adaptation = None
    if libero_adapter:
        from src.tools.grasp.generator_adapter import GraspGenLiberoAdapter
        parts = GraspGenLiberoAdapter.adapt_mesh_parts(parts)
        adaptation = GraspGenLiberoAdapter.metadata()
    vertices = np.concatenate(list(parts.values())).reshape(-1, 3)
    return parts, {
        "asset_root": str(root), "pinned_revision": PINNED_REVISION,
        "gripper_adapter": adaptation,
        "source_sha256": ASSET_HASHES, "source_units": "meters; scale=1 (pinned source loads STL without scaling)",
        "asset_to_graspgen": "identity (audited YAML)", "mesh_origin": "GraspGen hand origin, NOT TCP",
        "axes": {"approach": "+Z", "closing": "local X"},
        "opening_m": width, "opening_role": "nominal_OPEN_reference_unknown_actual_width" if nominal else "caller_expected_joint_opening_not_measured",
        "local_mesh_bounds_m": [vertices.min(0).tolist(), vertices.max(0).tolist()],
        "grasp_to_public_ee": GRASP_TO_EE.tolist(),
        "public_to_planner_z_m": .107,
        "calibration_role": "verified runtime calibration context; neither offset applied to mesh",
    }


def _pose_matrix(value):
    if not isinstance(value, Mapping):
        return checked_transform(value)
    q = np.array([value["rotation"][a] for a in "xyzw"], dtype=float)
    if not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q), 1, atol=1e-4):
        raise ValueError("invalid plan quaternion")
    x, y, z, w = q / np.linalg.norm(q)
    t = np.eye(4)
    t[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                 [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                 [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    t[:3, 3] = [value["position"][a] for a in "xyz"]
    return checked_transform(t)


def approach_context(pose, plan=None, nominal_approach_m=.10):
    pose = checked_transform(pose)
    if not np.isfinite(nominal_approach_m) or not 0 < nominal_approach_m <= .5:
        raise ValueError("nominal approach must be in (0, .5] meters")
    reason = "no validated matching plan supplied"
    if plan is not None and getattr(plan, "trajectory_validated", False):
        try:
            if not np.allclose(checked_transform(plan.grasp_transform), pose, atol=1e-6):
                raise ValueError("plan frame/pose does not match candidate")
            calibration = checked_transform(plan.grasp_to_ee)
            if not np.allclose(calibration, GRASP_TO_EE, atol=1e-6):
                raise ValueError("plan calibration differs from configured runtime calibration")
            labels = list(plan.target_labels)
            if len(labels) != len(plan.targets) or labels.count("pregrasp") != 1 or labels.count("grasp") != 1:
                raise ValueError("ambiguous plan target labels")
            a, b = labels.index("pregrasp"), labels.index("grasp")
            if b != a + 1:
                raise ValueError("nonadjacent pregrasp/grasp targets")
            targets = [_pose_matrix(p) @ np.linalg.inv(calibration) for p in plan.targets[a:b+1]]
            if not np.allclose(targets[-1], pose, atol=1e-5):
                raise ValueError("plan grasp endpoint mismatch")
            return targets[0], {"role": "validated_plan_target_chord_not_executed_path",
                "hand_origin_targets_m": [p[:3, 3].tolist() for p in targets],
                "note": "Endpoint chord/corridor only, not interpolated robot trajectory or swept volume."}
        except (ValueError, AttributeError, KeyError, TypeError, IndexError) as exc:
            reason = str(exc)
    pre = pose.copy()
    pre[:3, 3] -= nominal_approach_m * pose[:3, 2]
    return pre, {"role": "candidate_local_nominal_approach_not_executed_path",
        "nominal_length_m": nominal_approach_m, "fallback_reason": reason,
        "hand_origin_targets_m": [pre[:3, 3].tolist(), pose[:3, 3].tolist()]}


def orthographic_project(points, basis, center, pixels_per_m, origin=(480., 350.)):
    """Virtual metric view (not a calibrated RGB overlay); equal XY scale."""
    xy = (np.asarray(points) - center) @ np.asarray(basis).T
    return np.asarray(origin) + xy * [pixels_per_m, -pixels_per_m]


def _cloud(value):
    p = np.asarray(value, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("expected finite Nx3 observed points in candidate frame")
    return p


def _sample(p, n):
    return p[np.linspace(0, len(p)-1, min(len(p), n), dtype=int)] if len(p) else p


def render_candidate_inspection(geometry: PointGeometry, prediction: GraspPrediction,
        output_dir, *, candidate_ref="candidate", plan=None, graspgen_root=None,
        expected_open_width_m=None, views=2, nominal_approach_m=.10, static_pose=False):
    """Return 1-2 deterministic PNG paths + measured/inferred provenance metadata.

    Caller supplies same-frame observations and optionally an audited matching
    GraspPlan. No robot connection is used. Pair these with actual current RGB
    front/wrist images at the caller; this helper invents no whole-arm geometry.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", candidate_ref):
        raise ValueError("invalid candidate reference")
    if type(views) is not int or views not in (1, 2):
        raise ValueError("one or two views per candidate")
    if static_pose and getattr(getattr(geometry, 'frame', None), 'rgb_path', None):
        from src.tools.grasp.input_cards import render_grasp_cards
        return render_grasp_cards(geometry, prediction, output_dir, candidate_ref=candidate_ref,
            graspgen_root=graspgen_root, expected_open_width_m=expected_open_width_m, views=views)
    pose = checked_transform(prediction.pose)
    obj, scene = _cloud(geometry.object_points), _cloud(geometry.scene_points)
    parts, meta = load_panda_mesh(graspgen_root, expected_open_width_m,
        libero_adapter=bool(getattr(prediction, "gripper_adapter", None)))
    pre, approach = approach_context(pose, plan, nominal_approach_m)
    if static_pose:
        pre = pose.copy()
        approach = {'role': 'static_grasp_pose_only_no_trajectory'}
    local = np.concatenate(list(parts.values())).reshape(-1, 3)
    mesh, ghost = transform_points(local, pose), transform_points(local, pre)
    center = np.median(obj, axis=0) if len(obj) else pose[:3, 3]
    # Full scene context in view 1; local observed scene subset in view 2.
    bases = [np.array([[1., -1., 0.], [.5, .5, 1.]]), np.array([pose[:3, 0], pose[:3, 2]])]
    paths, view_meta = [], []
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for index in range(views):
        basis = bases[index] / np.linalg.norm(bases[index], axis=1)[:, None]
        displayed_scene = scene if index == 0 else scene[np.linalg.norm(scene-center, axis=1) <= .30]
        cloud = _sample(displayed_scene, 14000)
        target = _sample(obj, 6000)
        fit = np.concatenate([mesh, ghost, target, cloud])
        projected = (fit-center) @ basis.T
        lo, hi = projected.min(0), projected.max(0)
        scale = min(860 / max(hi[0]-lo[0], .15), 410 / max(hi[1]-lo[1], .15))
        origin = np.array([480, 355]) - (lo+hi)/2 * [scale, -scale]
        project = lambda p: orthographic_project(p, basis, center, scale, origin)
        image = Image.new("RGB", (960, 720), "white")
        draw = ImageDraw.Draw(image)
        opening = f"NOMINAL OPEN {meta['opening_m']*1000:g}mm; actual width UNKNOWN" if expected_open_width_m is None else f"EXPECTED opening {meta['opening_m']*1000:.1f}mm (not measured)"
        lines = [f"{candidate_ref} | {'observed scene context' if index == 0 else 'local X/Z closing-approach view'} | metric virtual view",
                 opening, "Blue: selected robot gripper mesh   Orange: observed target   Gray: observed scene",
                 "Purple: approach endpoint chord/corridor; pale mesh: pregrasp reference",
                 approach["role"]]
        if static_pose:
            lines[3] = "Static candidate pose only; current scene RGB supplied separately"
        for i, text in enumerate(lines):
            draw.text((16, 12+18*i), text, fill=(25, 30, 40))
        for p in project(cloud):
            draw.point(tuple(p), fill=(158, 165, 173))
        for p in project(target):
            draw.ellipse((p[0]-1, p[1]-1, p[0]+1, p[1]+1), fill=(205, 128, 15))
        # Exact asset triangle wireframe, no proxy dimensions or mesh rescaling.
        for vertices, color in ([(mesh, (38,104,167))] if static_pose else [(ghost,(194,202,219)),(mesh,(38,104,167))]):
            triangles = project(vertices).reshape(-1, 3, 2)
            for triangle in triangles:
                draw.line([tuple(p) for p in triangle] + [tuple(triangle[0])], fill=color, width=1)
        if not static_pose:
            # Four corresponding mesh-bound corners: endpoint-space context ONLY.
            bounds = np.array(meta["local_mesh_bounds_m"])
            corners = np.array([[x, y, bounds[0, 2]] for x in bounds[:, 0] for y in bounds[:, 1]])
            for a, b in zip(project(transform_points(corners, pre)), project(transform_points(corners, pose))):
                draw.line([tuple(a), tuple(b)], fill=(170, 145, 185), width=1)
            a, b = project(np.array([pre[:3, 3], pose[:3, 3]]))
            draw.line([tuple(a), tuple(b)], fill=(130, 62, 165), width=3)
            d = b-a
            if np.linalg.norm(d) > 1:
                d = d/np.linalg.norm(d); side = np.array([-d[1], d[0]])
                draw.polygon([tuple(b), tuple(b-12*d+5*side), tuple(b-12*d-5*side)], fill=(130, 62, 165))
        origin_px, tcp_px = project(transform_points([[0, 0, 0], [0, 0, .097]], pose))
        draw.text(tuple(origin_px), "HAND origin", fill=(10, 35, 70))
        draw.text(tuple(tcp_px), "TCP +97mm", fill=(10, 35, 70))
        bar = .05 * scale
        draw.line([(25, 590), (25+bar, 590)], fill=(20, 20, 20), width=3)
        draw.text((25, 597), "50mm", fill=(20, 20, 20))
        for i, text in enumerate(LIMITATIONS):
            draw.text((16, 630+18*i), text, fill=(40, 40, 40))
        path = output / f"{candidate_ref}-inspection-{index+1}.png"
        image.save(path)
        paths.append(str(path))
        view_meta.append({"basis_rows": basis.tolist(), "center_m": center.tolist(),
            "pixels_per_m": scale, "pixel_origin": origin.tolist(),
            "scene_points_rendered": len(cloud), "target_points_rendered": len(target),
            "scene_crop": "none" if index == 0 else "within 0.30m of target median; outside omitted, NOT free space"})
    meta.update({"candidate_ref": candidate_ref, "candidate_pose_role": "inferred GraspGen hand pose",
        "candidate_pose": pose.tolist(), "score": float(prediction.score),
        "observation_id": str(geometry.observation_id), "view_id": str(geometry.view_id),
        "source_views": list(getattr(geometry, "source_views", ())),
        "cloud_role": "observed partial metric surfaces; caller guarantees shared candidate frame",
        "observed_point_counts": {"target": len(obj), "scene": len(scene)},
        "approach": approach, "views": view_meta, "limitations": list(LIMITATIONS),
        "robot_arm_context": "not rendered; use actual current front/wrist RGB separately"})
    manifest = output / f"{candidate_ref}-inspection.json"
    manifest.write_text(json.dumps(meta, indent=2, allow_nan=False)+"\n")
    return {"image_paths": paths, "metadata": meta, "metadata_path": str(manifest)}


class CandidateInspector:
    """ID-only on-demand access to ANY registered candidate, separate from previews.

    Budget counts unique emitted inspection images (cached refs cost no more).
    Caller must reserve its actual RGB/other previews in the total model budget.
    """
    def __init__(self, geometry, candidates: Mapping, output_dir, *, max_images=8,
                 views=2, plans=None, expected_open_widths=None, graspgen_root=None):
        if type(max_images) is not int or not 1 <= max_images <= 8:
            raise ValueError("inspection image budget must be in [1,8]")
        if type(views) is not int or views not in (1, 2):
            raise ValueError("one or two views per candidate")
        self.geometry, self.candidates, self.output_dir = geometry, dict(candidates), output_dir
        self.max_images, self.views = max_images, views
        self.plans, self.widths = plans or {}, expected_open_widths or {}
        self.root, self.cache = graspgen_root, {}

    def inspect_candidate(self, candidate_ref):
        if candidate_ref not in self.candidates:
            raise KeyError("unknown candidate reference")
        if candidate_ref not in self.cache:
            if sum(len(r["image_paths"]) for r in self.cache.values()) + self.views > self.max_images:
                raise ValueError("inspection image budget exhausted; do not exceed model total budget")
            self.cache[candidate_ref] = render_candidate_inspection(
                self.geometry, self.candidates[candidate_ref], self.output_dir,
                candidate_ref=candidate_ref, plan=self.plans.get(candidate_ref),
                expected_open_width_m=self.widths.get(candidate_ref),
                graspgen_root=self.root, views=self.views)
        return self.cache[candidate_ref]
