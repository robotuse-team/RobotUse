"""Agent-visible metric point scatter: unknown surfaces remain unknown."""
from pathlib import Path
import json
import numpy as np

def inspect_fusion(fusion, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    fig = plt.figure(figsize=(18, 9))
    colors = ("#00aacc", "#ee7733")
    obj, scene = fusion.object_points, fusion.scene_points
    for slot, title, cloud in ((121, "Observed scene context", np.concatenate([obj, scene])),
                              (122, "OBJECT ZOOM — measured points only", obj)):
        ax = fig.add_subplot(slot, projection="3d")
        if slot == 121 and len(scene):
            stride = max(1, int(np.ceil(len(scene)/20000)))
            p = scene[::stride]
            ax.scatter(*p.T, c="#aaaaaa", s=.4, alpha=.25, rasterized=True)
        for i, view in enumerate(fusion.object_cloud.view_ids):
            p = obj[fusion.object_cloud.source_view == i]
            if len(p): ax.scatter(*p.T, c=colors[i], s=3, label=view)
        lo, hi = cloud.min(axis=0), cloud.max(axis=0)
        center, half = (lo+hi)/2, max(float(np.max(hi-lo))*.55, .01)
        for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center): setter(c-half, c+half)
        ax.set_box_aspect((1, 1, 1))
        ax.set_proj_type("ortho")
        ax.view_init(elev=55, azim=-60)
        ax.set(xlabel="base X (m)", ylabel="base Y (m)", zlabel="base Z (m)", title=title)
        ax.legend(loc="upper left")
    fig.suptitle(f"{fusion.observation_id} | epoch {fusion.epoch} | {fusion.same_object}\n"
                 "Equal metric axes in each panel. No mesh/completion. Unseen is UNKNOWN, not free.")
    png = destination / "cloud_inspection.png"
    fig.savefig(png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    comparison = destination / "source_comparison.png"
    comparison_fig = plt.figure(figsize=(18, 6))
    center = (obj.min(axis=0)+obj.max(axis=0))/2
    half = max(float(np.max(np.ptp(obj, axis=0)))*.55, .01)
    panels = [(g.view_id, g.object_points, colors[fusion.object_cloud.view_ids.index(g.view_id)])
              for g in fusion.per_view] + [("FUSED measurements" if len(fusion.per_view) > 1 else "VOXELIZED measurements (single view)", obj, None)]
    for i, (title, p, color) in enumerate(panels, 1):
        ax = comparison_fig.add_subplot(1, len(panels), i, projection="3d")
        if color is None:
            for j, view in enumerate(fusion.object_cloud.view_ids):
                part = p[fusion.object_cloud.source_view == j]
                if len(part): ax.scatter(*part.T, c=colors[j], s=3, label=view)
        else:
            ax.scatter(*p.T, c=color, s=3)
        for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center): setter(c-half, c+half)
        ax.set_box_aspect((1, 1, 1)); ax.set_proj_type("ortho"); ax.view_init(elev=55, azim=-60)
        ax.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)", title=title)
    comparison_fig.suptitle("Same connector_base limits and equal metric axes; independent masks; unknown surfaces UNKNOWN")
    comparison_fig.savefig(comparison, dpi=130, bbox_inches="tight")
    plt.close(comparison_fig)
    arrays = {}
    for name, cloud in (("object", fusion.object_cloud), ("scene", fusion.scene_cloud)):
        for key in ("points", "view_bits", "view_counts", "sample_counts", "source_view", "source_index"):
            arrays[name+"_"+key] = getattr(cloud, key)
    for g in fusion.per_view:
        arrays[g.view_id+"_object_points"] = g.object_points
        arrays[g.view_id+"_scene_points"] = g.scene_points
    npz = destination / "measured_clouds.npz"
    np.savez_compressed(npz, **arrays)
    metadata = dict(schema="measured-multiview.v1", observation_id=fusion.observation_id,
        epoch=fusion.epoch, same_object=fusion.same_object, frame="connector_base", units="meters",
        image_path=str(png), image_paths=[str(png), str(comparison)], cloud_path=str(npz), voxel_size_m=fusion.voxel_size_m,
        same_object_policy="explicit agent assertion, not ground-truth identity proof",
        scene_object_overlap_policy="retain contradictory scene evidence; no hidden projection or erasure",
        source_views=list(fusion.source_views), provenance=str(destination / "inspection.json"),
        view_ids=fusion.object_cloud.view_ids, object_points=len(obj), scene_points=len(scene),
        unknown_surfaces="unknown; no inferred points, no free-space claim",
        representative="lexicographically smallest measured point; colors show representative source",
        per_view=[dict(view_id=g.view_id, mask_path=str(g.mask_path), overlay_path=str(g.overlay_path),
                       frame_id=g.frame.frame_id) for g in fusion.per_view])
    (destination / "inspection.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return metadata
