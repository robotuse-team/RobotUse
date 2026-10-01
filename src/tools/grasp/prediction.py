"""Process-isolated NVlabs GraspGen diffusion inference on observed metric points.

No oracle, OBB proposal generator, heuristic grasp fallback, or implicit download.
Mesh/scene checks concern observed surfaces, not a complete safety certificate;
Collision filtering defaults ON for standalone use; the simulation runner
explicitly disables it. Numerical pose validation and learned ranking remain.
"""


from __future__ import annotations


from dataclasses import dataclass


import json


from pathlib import Path


from typing import Any


import numpy as np


@dataclass(frozen=True)
class GraspPrediction:
    pose: np.ndarray  # connector_base_T_gripper; inference preserves input frame
    score: float
    gripper_adapter: str | None = None
    gripper_name: str = "franka_panda"


def checked_points(value: Any, *, minimum: int = 1) -> np.ndarray:
    points = np.asarray(value, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < minimum:
        raise ValueError(f"expected at least {minimum} Nx3 metric points")
    if not np.isfinite(points).all():
        raise ValueError("point cloud contains nonfinite values")
    return np.ascontiguousarray(points)


def checked_predictions(poses: Any, scores: Any, *, gripper_adapter=None) -> tuple[GraspPrediction, ...]:
    poses = np.asarray(poses, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if poses.shape != (len(scores), 4, 4) or scores.ndim != 1:
        raise ValueError("invalid GraspGen prediction shapes")
    if not np.isfinite(poses).all() or not np.isfinite(scores).all():
        raise ValueError("nonfinite GraspGen predictions")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("invalid discriminator confidence")
    for pose in poses:
        rotation = pose[:3, :3]
        if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-4):
            raise ValueError("invalid homogeneous grasp pose")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-3):
            raise ValueError("grasp rotation is not SO(3)")
    return tuple(GraspPrediction(poses[i].copy(), float(scores[i]), gripper_adapter)
        for i in np.argsort(-scores))


PREWARM_TIMEOUT_S = 3600.  # a timeout kills the worker; never kill a model that is still loading


class GraspGenBackend:
    def __init__(self, *, python: str | Path, checkout: str | Path,
                 gripper_config: str | Path, output_dir: str | Path,
                 num_grasps: int = 128, topk: int = 24, threshold: float = 0.5,
                 collision_margin_m: float = 0.008, timeout_s: float = 600,
                 collision_check: bool = True, official_scene_filter: bool = False,
                 libero_adapter: bool = False, target_candidates: int = 0,
                 max_generated: int = 500, official_clearance_m: float = .002,
                 clearance_batch_floor_m: float | None = None, open_width_m: float | None = None):
        self.python = str(python)
        self.checkout = Path(checkout).resolve()
        # Preserve snapshot symlinks: upstream resolves sibling weights relative to the YAML path.
        self.config = Path(gripper_config).absolute()
        self.output_dir = Path(output_dir).resolve()
        self.num_grasps, self.topk = num_grasps, topk
        self.threshold, self.margin = threshold, collision_margin_m
        if type(collision_check) is not bool:
            raise ValueError("collision_check must be boolean")
        if open_width_m is not None and (not np.isfinite(open_width_m) or not 0 <= open_width_m <= .08):
            raise ValueError("invalid expected Panda opening")
        self.open_width_m = open_width_m
        self.adaptive_contact_opening = False
        self.collision_check = collision_check
        self.official_scene_filter = official_scene_filter
        self.initial_official_clearance_m = official_clearance_m
        self.official_clearance_m = official_clearance_m
        self.clearance_batch_floor_m = clearance_batch_floor_m
        if clearance_batch_floor_m is not None and (not np.isfinite(clearance_batch_floor_m)
                or not 0 < clearance_batch_floor_m <= official_clearance_m):
            raise ValueError("invalid collision clearance retry floor")
        if not np.isfinite(official_clearance_m) or official_clearance_m < 0:
            raise ValueError("invalid official collision clearance")
        self.libero_adapter = libero_adapter
        self.target_candidates, self.max_generated = target_candidates, max_generated
        if target_candidates < 0 or max_generated < 1 or target_candidates > topk:
            raise ValueError("invalid candidate accumulation budget")
        self.timeout_s = timeout_s
        self.calls = 0
        if not self.config.is_file() or not (self.checkout / "grasp_gen/grasp_server.py").is_file():
            raise FileNotFoundError("GraspGen checkout/config missing; install explicitly before running")
        if num_grasps < 1 or topk < 1 or not 0 <= threshold <= 1 or collision_margin_m <= 0:
            raise ValueError("invalid GraspGen configuration")

    def predict(self, object_points: Any, scene_points: Any) -> tuple[GraspPrediction, ...]:
        obj = checked_points(object_points, minimum=100)
        scene = checked_points(scene_points)
        self.calls += 1
        # Each new observed-cloud request restarts the internal batch schedule.
        self.official_clearance_m = self.initial_official_clearance_m
        directory = self.output_dir / f"graspgen_{self.calls:03d}"
        directory.mkdir(parents=True, exist_ok=False)
        source, result = directory / "observed_points.npz", directory / "predictions.npz"
        np.savez_compressed(source, object_points=obj, scene_points=scene)
        from src.runtime.worker import call_worker
        call_worker(self.python, Path(__file__).resolve(), dict(checkout=str(self.checkout), config=str(self.config),
            input=str(source), output=str(result), num_grasps=self.num_grasps, topk=self.topk,
            threshold=self.threshold, margin=self.margin, collision_check=self.collision_check,
            official_scene_filter=self.official_scene_filter, libero_adapter=self.libero_adapter,
            target_candidates=self.target_candidates, max_generated=self.max_generated,
            official_clearance_m=self.official_clearance_m,
            clearance_batch_floor_m=self.clearance_batch_floor_m, open_width_m=self.open_width_m,
            adaptive_contact_opening=self.adaptive_contact_opening),
            self.output_dir/'graspgen-worker.log', self.timeout_s)
        if self.clearance_batch_floor_m is not None:
            # Whole-arm checks use the final batch's margin; earlier survivors
            # already passed the same or a stricter scene margin.
            self.official_clearance_m = json.loads(result.with_suffix(".json").read_text()).get("effective_clearance_m", self.initial_official_clearance_m)
        with np.load(result, allow_pickle=False) as data:
            adapter_name = str(data["gripper_adapter"].item()) if "gripper_adapter" in data else ""
            return checked_predictions(data["poses"], data["scores"],
                gripper_adapter=adapter_name or None)


    def prewarm(self):
        """Load the model in the persistent worker before the first real request.

        The prewarm timeout is deliberately long: a timeout kills the worker,
        and killing it while it is still importing under I/O contention breaks
        the first real request that is queued behind the same worker lock.

        A throwaway synthetic blob is inferred once under output_dir/prewarm so
        the worker's imports, checkpoint load and CUDA warm-up overlap with the
        agent's first turns. Never touches the numbered graspgen_NNN artifacts.
        """
        directory = self.output_dir / "prewarm" / "graspgen"
        directory.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        obj = np.array([.5, 0., .05]) + rng.normal(scale=.015, size=(200, 3))
        scene = np.array([.5, .3, 0.]) + rng.normal(scale=.05, size=(50, 3))
        source, result = directory / "observed_points.npz", directory / "predictions.npz"
        np.savez_compressed(source, object_points=obj, scene_points=scene)
        from src.runtime.worker import call_worker
        call_worker(self.python, Path(__file__).resolve(), dict(checkout=str(self.checkout), config=str(self.config),
            input=str(source), output=str(result), num_grasps=1, topk=1,
            threshold=self.threshold, margin=self.margin, collision_check=self.collision_check,
            official_scene_filter=self.official_scene_filter, libero_adapter=self.libero_adapter,
            target_candidates=0, max_generated=1,
            official_clearance_m=self.initial_official_clearance_m,
            clearance_batch_floor_m=None, open_width_m=self.open_width_m,
            adaptive_contact_opening=self.adaptive_contact_opening),
            self.output_dir/'graspgen-worker.log', PREWARM_TIMEOUT_S)

    def predict_with_path_filter(self, object_points, scene_points, accept_candidate, *, rank_batch=None):
        """Collect candidates only after the caller's existing path checks pass.

        Model inference stays in its worker; robot planning stays in Prime.
        Each worker request generates exactly one batch, including a short
        final batch at the configured raw-proposal cap.
        """
        import copy
        obj, scene = checked_points(object_points, minimum=100), checked_points(scene_points)
        self.calls += 1
        directory = self.output_dir / f"graspgen_{self.calls:03d}"
        directory.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(directory / "observed_points.npz", object_points=obj, scene_points=scene)
        generated, batch_count, accepted = 0, 0, []
        self.official_clearance_m = self.initial_official_clearance_m
        while generated < self.max_generated and len(accepted) < self.target_candidates:
            count = min(self.num_grasps, self.max_generated - generated)
            floor = self.clearance_batch_floor_m
            margin = (max(floor, self.initial_official_clearance_m * .5 ** batch_count)
                      if floor is not None else self.initial_official_clearance_m)
            batch = copy.copy(self)
            batch.output_dir = directory / "batches" / f"batch-{batch_count+1:03d}"
            batch.calls = 0
            batch.num_grasps = batch.max_generated = count
            # Let every confidence/scene survivor reach the existing path check.
            # The final six-candidate limit belongs after accept_candidate.
            batch.topk = count
            batch.target_candidates = 1
            batch.initial_official_clearance_m = margin
            batch.clearance_batch_floor_m = None
            predictions = batch.predict(obj, scene)
            source = batch.output_dir / "graspgen_001" / "predictions.json"
            meta = json.loads(source.read_text())
            if meta["generated_count"] != count:
                raise RuntimeError("single batch did not honor the raw proposal budget")
            self.official_clearance_m = margin
            generated += count
            batch_count += 1
            for prediction in (rank_batch(predictions) if rank_batch else predictions):
                if accept_candidate(prediction):
                    accepted.append(prediction)
                    if len(accepted) >= self.target_candidates:
                        break
        poses = np.asarray([p.pose for p in accepted]).reshape(-1, 4, 4)
        scores = np.asarray([p.score for p in accepted])
        np.savez_compressed(directory / "predictions.npz", poses=poses, scores=scores,
                            gripper_adapter=np.asarray(accepted[0].gripper_adapter or "" if accepted else ""))
        meta.update(generated_count=generated, accepted_count=len(accepted))
        (directory / "predictions.json").write_text(json.dumps(meta, indent=2)+"\n")
        return tuple(accepted)
