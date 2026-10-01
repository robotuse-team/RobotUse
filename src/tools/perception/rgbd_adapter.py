"""Point-prompt SAM2 on fresh RGB, then calibrated observed-depth unprojection.

The model chooses only a normalized image point. No body lookup, simulator
segmentation, detector box, or object-name geometry is consulted.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

# Script worker can run inside an independent official SAM2 environment.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.tools.perception.geometry import ObservationProfileId, unproject_masked_depth
from src.tools.perception.runtime import capture_connector_rgbd


@dataclass(frozen=True)
class PointGeometry:
    observation_id: str
    view_id: str
    role: str
    object_points: np.ndarray
    scene_points: np.ndarray
    mask_path: Path
    overlay_path: Path
    frame: Any
    epoch: Any = None
    source_views: tuple[str, ...] = ()


def normalized_pixel(u: float, v: float, width: int, height: int) -> tuple[int, int]:
    if isinstance(u, bool) or isinstance(v, bool) or not np.isfinite([u, v]).all() or not (0 <= u <= 1000 and 0 <= v <= 1000):
        raise ValueError("point coordinates must be finite in [0,1000]")
    return round(u * (width - 1) / 1000), round(v * (height - 1) / 1000)


def refine_point_depth_mask(depth_m: Any, source_mask: Any, *, pixel: tuple[int, int],
                            depth_band_m: float = 0.06, neighbor_step_m: float = 0.012,
                            neighborhood_radius: int = 2) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep the clicked surface's four-connected, bounded-depth component.

    Only removes pixels from the supplied mask. The seed is the selected point,
    never the mask's modal depth, which may be dominated by background.
    A hard seed-relative band prevents
    gradual depth bridges; an adjacent-depth bound rejects sharp discontinuities
    even inside that band. No points are invented and no scene pixels erased.
    This is a conservative partial surface, not object completion/segmentation proof.
    """
    from collections import deque
    depth = np.asarray(depth_m, dtype=np.float64)
    source = np.asarray(source_mask, dtype=bool)
    if depth.ndim != 2 or source.shape != depth.shape:
        raise ValueError("depth and SAM mask shape mismatch")
    x, y = pixel
    height, width = depth.shape
    if not (0 <= x < width and 0 <= y < height) or not source[y, x]:
        raise ValueError("SAM mask does not contain the requested point")
    if not (np.isfinite(depth_band_m) and np.isfinite(neighbor_step_m) and
            0 < neighbor_step_m <= depth_band_m <= 0.10):
        raise ValueError("depth consistency bounds must satisfy 0 < step <= band <= 0.10m")
    if type(neighborhood_radius) is not int or not 1 <= neighborhood_radius <= 4:
        raise ValueError("seed neighborhood radius must be bounded in [1,4]")
    valid = source & np.isfinite(depth) & (depth > 0.02) & (depth < 3.0)
    if not valid[y, x]:
        raise ValueError("clicked pixel lacks valid metric depth; re-point required")
    seed_depth = float(depth[y, x])
    r = neighborhood_radius
    window = np.zeros_like(source)
    window[max(0, y-r):min(height, y+r+1), max(0, x-r):min(width, x+r+1)] = True
    local = depth[window & valid & (np.abs(depth-seed_depth) <= neighbor_step_m)]
    if len(local) < 3:
        raise ValueError("clicked depth is isolated from its local neighborhood; re-point required")
    center = float(np.median(local))
    gated = valid & (np.abs(depth-center) <= depth_band_m)
    component = np.zeros_like(source)
    component[y, x] = True
    queue = deque([(y, x)])
    while queue:
        row, column = queue.popleft()
        for rr, cc in ((row-1, column), (row+1, column), (row, column-1), (row, column+1)):
            if not (0 <= rr < height and 0 <= cc < width) or component[rr, cc] or not gated[rr, cc]:
                continue
            if abs(float(depth[rr, cc] - depth[row, column])) > neighbor_step_m:
                continue
            component[rr, cc] = True
            queue.append((rr, cc))
    stats = {"method": "point-seeded-four-connected-bounded-depth.v1", "pixel_xy": [x, y],
             "raw_mask_pixels": int(source.sum()), "raw_valid_pixels": int(valid.sum()),
             "refined_pixels": int(component.sum()), "rejected_valid_pixels_to_scene": int((valid & ~component).sum()),
             "seed_depth_m": seed_depth, "local_center_depth_m": center,
             "depth_band_m": depth_band_m, "neighbor_step_m": neighbor_step_m,
             "refined_depth_range_m": [float(depth[component].min()), float(depth[component].max())]}
    if not component[y, x] or np.any(component & ~source):
        raise AssertionError("depth refinement violated remove-only point binding")
    return component, stats


def geometry_from_point_mask(frame: Any, mask: Any, *, pixel: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    depth = np.asarray(frame.depth_m)
    if mask.shape != depth.shape or not mask[pixel[1], pixel[0]]:
        raise ValueError("SAM mask does not contain the requested point")
    valid = np.isfinite(depth) & (depth > 0.02) & (depth < 3.0)
    if np.count_nonzero(mask & valid) < 100:
        raise ValueError("point-selected mask has fewer than 100 valid depth points")
    if np.count_nonzero(mask & valid) / np.count_nonzero(mask) < 0.5:
        raise ValueError("point-selected mask has insufficient metric depth coverage")
    kwargs = dict(depth_m=depth, intrinsics=frame.intrinsics,
                  camera_to_base=frame.camera_to_base)
    obj = np.asarray(unproject_masked_depth(mask=mask & valid, **kwargs), dtype=np.float32)
    scene = np.asarray(unproject_masked_depth(mask=~mask & valid, **kwargs), dtype=np.float32)
    if len(scene) == 0:
        raise ValueError("no observed non-target scene points for collision checks")
    return obj, scene


PREWARM_TIMEOUT_S = 3600.  # a timeout kills the worker; never kill a model that is still loading


class PointRGBDAdapter:
    def __init__(self, *, connector: Any, python: str | Path, sam2_snapshot: str | Path,
                 output_dir: str | Path, device: str = "cuda", timeout_s: float = 300):
        self.connector, self.python = connector, str(python)
        self.snapshot = Path(sam2_snapshot).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.device, self.timeout_s = device, timeout_s
        self.revision = 0
        self.point_count = 0
        self.frames: dict[str, tuple[Any, ...]] = {}
        self.latest: str | None = None

    def observe(self) -> dict[str, Any]:
        self.revision += 1
        frames = capture_connector_rgbd(connector=self.connector,
            output_dir=self.output_dir / "rgbd", capture_revision=self.revision,
            actor_profile=ObservationProfileId.MULTIVIEW_TRACK)
        observation_id = f"obs_{self.revision:04d}"
        self.frames[observation_id] = frames
        self.latest = observation_id
        return {"observation_id": observation_id, "images": [
            {"view_id": frame.view_id, "image_path": frame.rgb_path} for frame in frames]}

    def prewarm(self):
        """Load SAM2 in the persistent worker before the first real selection.

        A synthetic image is segmented once under output_dir/prewarm; the real
        point_NNNN artifacts are untouched.
        """
        from PIL import Image
        directory = self.output_dir / "prewarm" / "sam2"
        directory.mkdir(parents=True, exist_ok=True)
        image = np.full((128, 128, 3), 90, dtype=np.uint8)
        image[40:88, 40:88] = 220
        image_path, mask_path = directory / "image.png", directory / "sam_mask.png"
        Image.fromarray(image).save(image_path)
        from src.runtime.worker import call_worker
        call_worker(self.python, Path(__file__).resolve(), dict(snapshot=str(self.snapshot), image=str(image_path),
            output=str(mask_path), x=64, y=64, device=self.device), self.output_dir/'sam2-worker.log',
            PREWARM_TIMEOUT_S)

    def point(self, observation_id: str, view_id: str, u: float, v: float,
              role: str = "pick") -> PointGeometry:
        if observation_id != self.latest:
            raise ValueError("point must reference the latest RGBD observation")
        if role not in ("pick", "place"):
            raise ValueError("point role must be pick or place")
        frame = next((f for f in self.frames[observation_id] if f.view_id == view_id), None)
        if frame is None:
            raise ValueError("unknown RGBD view")
        x, y = normalized_pixel(u, v, frame.width, frame.height)
        self.point_count += 1
        directory = self.output_dir / f"point_{self.point_count:04d}"
        directory.mkdir(parents=True, exist_ok=False)
        mask_path, overlay_path = directory / "mask.png", directory / "overlay.png"
        raw_mask_path = directory / "sam_mask.png"
        from src.runtime.worker import call_worker
        call_worker(self.python, Path(__file__).resolve(), dict(snapshot=str(self.snapshot), image=frame.rgb_path,
            output=str(raw_mask_path), x=x, y=y, device=self.device), self.output_dir/'sam2-worker.log', self.timeout_s,
            startup_timeout=900)
        raw_mask = np.asarray(Image.open(raw_mask_path).convert("L")) > 0
        mask, depth_refinement = refine_point_depth_mask(frame.depth_m, raw_mask, pixel=(x, y))
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
        (directory / "depth_refinement.json").write_text(json.dumps(depth_refinement, indent=2) + "\n")
        obj, scene = geometry_from_point_mask(frame, mask, pixel=(x, y))
        rgb = np.asarray(frame.rgb).copy()
        rgb[mask] = (0.55 * rgb[mask] + 0.45 * np.array([0, 255, 255])).astype(np.uint8)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        draw.line((x - 8, y, x + 8, y), fill="red", width=2)
        draw.line((x, y - 8, x, y + 8), fill="red", width=2)
        image.save(overlay_path)
        np.savez_compressed(directory / "observed_points.npz", object_points=obj, scene_points=scene)
        (directory / "provenance.json").write_text(json.dumps({
            "observation_id": observation_id, "view_id": view_id, "role": role,
            "source": "SAM2 positive point + point-connected depth refinement + synchronized RGBD",
            "depth_refinement": depth_refinement,
            "raw_mask_sha256": hashlib.sha256(raw_mask_path.read_bytes()).hexdigest(),
            "frame": "connector_base (SensorFramePayload.camera_to_base)",
            "rgb_sha256": frame.rgb_sha256, "depth_sha256": frame.depth_sha256,
            "calibration_sha256": frame.calibration_sha256,
            "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
            "collision_scope": "selected-view observed non-target points; occluded space unknown",
        }, indent=2) + "\n")
        return PointGeometry(observation_id, view_id, role, obj, scene, mask_path, overlay_path, frame)


_MODEL_CACHE = {}

def _worker(args: argparse.Namespace) -> None:
    from src.runtime.worker import diagnostic_phase
    with diagnostic_phase('sam.imports'):
        from src.tools.perception.sam2_adapter import create_predictor, predict_masks, resolve_checkpoint
    snapshot = Path(args.snapshot)
    checkpoint = resolve_checkpoint(snapshot)
    key = (str(checkpoint), args.device)
    if key not in _MODEL_CACHE:
        with diagnostic_phase('sam.model_load'):
            _MODEL_CACHE[key] = create_predictor(checkpoint, args.device)
    predictor = _MODEL_CACHE[key]
    with diagnostic_phase('sam.input_preparation'):
        image = Image.open(args.image).convert("RGB")
    with diagnostic_phase('sam.inference'):
        masks, scores = predict_masks(predictor, image, args.x, args.y)
    with diagnostic_phase('sam.postprocess'):
        masks = masks.reshape(-1, image.height, image.width)
        eligible = [i for i in np.argsort(-scores) if masks[i, args.y, args.x]]
        if not eligible:
            raise RuntimeError("SAM2 returned no mask containing the point")
        selected = eligible[0]
        Image.fromarray(masks[selected].astype(np.uint8) * 255).save(args.output)
        Path(args.output).with_suffix(".json").write_text(json.dumps({
            "model": str(snapshot), "prompt": "positive-point", "score": float(scores[selected]),
        }) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    for key in ("snapshot", "image", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--x", type=int, required=True)
    parser.add_argument("--y", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    _worker(parser.parse_args())
