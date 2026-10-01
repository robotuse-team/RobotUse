"""A weak, measured-only disagreement check for paired same-target clicks."""
from __future__ import annotations

import numpy as np


INTENT_PROVENANCE = (
    "The front and wrist clicks indicate the same intended object, not matching "
    "3D surface points. Independent SAM2 masks and calibrated measured depth are "
    "fused without hidden-surface completion. Only gross cloud separation is "
    "rejected; nearby wrong-object selections and object identity remain unverified."
)


def measured_target_consistency(front_points, wrist_points):
    """Reject only gross spatial disagreement, never demand overlapping faces.

    The gap between *full* measured bounding boxes is a lower bound on all
    cross-view sample distances. Require it to exceed both 30 cm and either
    cloud's measured diagonal before rejecting. This is a deliberately weak
    tabletop pickup heuristic, not an identity test: opposite faces of a large
    object can still exceed that bound, while adjacent wrong objects can pass.
    No midpoint, inferred surface, correspondence, or modified cloud is emitted.
    """
    clouds = [np.asarray(points, dtype=float) for points in (front_points, wrist_points)]
    if any(points.ndim != 2 or points.shape[1] != 3 or not len(points)
           or not np.isfinite(points).all() for points in clouds):
        raise ValueError("paired target selections require finite measured clouds")
    bounds = [(points.min(axis=0), points.max(axis=0)) for points in clouds]
    (front_min, front_max), (wrist_min, wrist_max) = bounds
    gap = np.maximum(0., np.maximum(front_min - wrist_max, wrist_min - front_max))
    gap_m = float(np.linalg.norm(gap))
    threshold_m = max(.30, *(float(np.linalg.norm(high - low)) for low, high in bounds))
    return {"status": "disagreement" if gap_m > threshold_m else "not_rejected",
            "measured_box_gap_m": gap_m, "rejection_threshold_m": threshold_m,
            "object_identity_verified": False, "hidden_surface_completion": False,
            "limitations": INTENT_PROVENANCE}
