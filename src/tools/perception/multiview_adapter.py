"""Epoch-bound front/wrist measured-cloud capture, fusion and comparison."""
from dataclasses import replace
from typing import Any
import json
from src.tools.perception.rgbd_adapter import PointRGBDAdapter
from src.tools.perception.multiview import capture_fresh_pair, simulation_epoch, fuse_selected, FusedGeometry, voxel_merge
from src.tools.perception.cloud_inspection import inspect_fusion

class MultiviewPointRGBDAdapter(PointRGBDAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epochs = {}
        self.sim_epochs = {}
        self.selections = []
        self.inspection_count = 0
        self.fusions = []
        self.cloud_comparisons = []

    def observe(self, *, epoch=None) -> dict[str, Any]:
        self.revision += 1
        frames, sim_epoch = capture_fresh_pair(connector=self.connector,
            output_dir=self.output_dir / "rgbd", capture_revision=self.revision)
        observation_id = f"obs_{self.revision:04d}"
        self.frames[observation_id] = frames
        self.latest = observation_id
        self.epochs[observation_id] = str(epoch) if epoch is not None else f"sim:{sim_epoch[0]}:{sim_epoch[1]:.17g}"
        self.sim_epochs[observation_id] = sim_epoch
        (self.output_dir / "rgbd" / f"{observation_id}_snapshot.json").write_text(json.dumps({
            "observation_id": observation_id, "epoch": self.epochs[observation_id],
            "simulation_epoch": sim_epoch, "frame": "connector_base",
            "capture": "direct same-state renders; robot/camera calibration only; no physics steps",
            "views": [{"view_id": f.view_id, "frame_id": f.frame_id,
                       "rgb_path": f.rgb_path, "depth_metric_path": f.depth_metric_path,
                       "calibration_path": f.calibration_path, "rgb_sha256": f.rgb_sha256,
                       "depth_sha256": f.depth_sha256, "calibration_sha256": f.calibration_sha256}
                      for f in frames]}, indent=2) + "\n")
        return {"observation_id": observation_id, "epoch": self.epochs[observation_id],
                "simulation_epoch": sim_epoch, "images": [
            {"view_id": frame.view_id, "image_path": frame.rgb_path,
             "camera_to_base": {"rotation": frame.camera_to_base.rotation,
                                "translation": frame.camera_to_base.translation},
             "frame_id": frame.frame_id, "calibration_path": frame.calibration_path}
            for frame in frames], "cloud_update_policy": "fresh_pair_replace_no_temporal_merge"}

    def _check_current(self, observation_id):
        if self.latest is None or observation_id != self.latest or simulation_epoch(self.connector) != self.sim_epochs[observation_id]:
            raise ValueError("stale observation/epoch; observe again after motion")

    def fuse_points(self, geometries, *, same_object: str, epoch=None, voxel_size_m=.002, role="pick"):
        self._check_current(self.latest)
        selected = tuple(geometries)
        if any(not any(g is known for known in self.selections) for g in selected):
            raise ValueError("fusion requires this adapter independently clicked selections")
        bound_epoch = self.epochs[self.latest]
        if epoch is not None and str(epoch) != bound_epoch:
            raise ValueError("mismatched epoch")
        fusion = fuse_selected(selected, observation_id=self.latest, epoch=bound_epoch,
                               same_object=same_object, voxel_size_m=voxel_size_m, role=role)
        self.fusions.append(fusion)
        return fusion

    def compare_cloud_update(self, previous, current, *, same_object, distance_threshold_m=.005):
        """Explicit same-object fresh replacement comparison; old cloud is not reused."""
        from src.tools.perception.multiview import compare_measured_clouds
        self._check_current(current.observation_id)
        if not isinstance(same_object, str) or not same_object.strip():
            raise ValueError("explicit same_object assertion after reselection required")
        known = self.selections + self.fusions
        if any(not any(g is k for k in known) for g in (previous, current)):
            raise ValueError("comparison requires adapter-owned selections/fusions")
        if previous.observation_id == current.observation_id:
            raise ValueError("replacement comparison requires a fresh observation")
        if current.epoch != self.epochs[self.latest]:
            raise ValueError("mismatched comparison epoch")
        report = compare_measured_clouds(previous, current, distance_threshold_m=distance_threshold_m)
        report["same_object_assertion"] = same_object
        self.cloud_comparisons.append(report)
        path = self.output_dir / f"cloud_update_{len(self.cloud_comparisons):04d}.json"
        report["comparison_path"] = str(path)
        path.write_text(json.dumps(report, indent=2) + "\n")
        return report

    def inspect_point_cloud(self, fusion):
        self._check_current(fusion.observation_id)
        if fusion.epoch != self.epochs[self.latest]:
            raise ValueError("mismatched inspection epoch")
        if not isinstance(fusion, FusedGeometry):
            if not any(fusion is g for g in self.selections):
                raise ValueError("inspection requires an adapter selection")
            fusion = FusedGeometry(fusion.observation_id, fusion.epoch, "single-view agent selection", (fusion,),
                voxel_merge({fusion.view_id: fusion.object_points}),
                voxel_merge({fusion.view_id: fusion.scene_points}), .002)
        self.inspection_count += 1
        return inspect_fusion(fusion, self.output_dir / f"inspection_{self.inspection_count:04d}")


    def point(self, observation_id, view_id, u, v, role="pick"):
        self._check_current(observation_id)
        geometry = super().point(observation_id, view_id, u, v, role)
        self._check_current(observation_id)
        geometry = replace(geometry, epoch=self.epochs[observation_id], source_views=(view_id,))
        self.selections.append(geometry)
        return geometry
