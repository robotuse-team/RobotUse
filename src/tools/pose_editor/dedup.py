"""Stable pose suppression within one generated pool, before feasibility checks."""
from dataclasses import dataclass, asdict
import math
import time
import numpy as np
from src.tools.observation.views import rigid_matrix


@dataclass(frozen=True)
class PoseDedup:
    angle_deg: float = 1.
    translation_mm: float = 3.
    width_mm: float = .001

    def __post_init__(self):
        if not math.isfinite(self.angle_deg) or not 0 < self.angle_deg <= 180:
            raise ValueError('pose dedup angle must be finite and in (0, 180] degrees')
        if not math.isfinite(self.translation_mm) or self.translation_mm <= 0:
            raise ValueError('pose dedup translation must be finite and positive in mm')
        if not math.isfinite(self.width_mm) or self.width_mm < 0:
            raise ValueError('pose dedup width must be finite and nonnegative in mm')

    def metadata(self):
        return dict(enabled=True, **asdict(self), stage='before_collision_ik_and_path',
                    policy='first representative in source order; no chained or jaw-symmetry merging; no failed-representative fallback')

    def select(self, poses, *, ids, widths=None):
        start = time.perf_counter()
        matrices = [rigid_matrix(p) for p in poses]
        if len(ids) != len(matrices) or (widths is not None and len(widths) != len(matrices)):
            raise ValueError('pose dedup metadata length mismatch')
        keep, removed = [], []
        if matrices:
            a = np.asarray(matrices)
            xyz, rot = a[:, :3, 3], a[:, :3, :3]
            for i in range(len(a)):
                representative = None
                if keep:
                    close = np.linalg.norm(xyz[keep]-xyz[i], axis=1) <= self.translation_mm/1000+1e-12
                    close &= (np.einsum('kij,ij->k', rot[keep], rot[i])-1)/2 >= math.cos(math.radians(self.angle_deg))-1e-12
                    if widths is not None:
                        close &= np.abs(np.asarray(widths)[keep]-widths[i]) <= self.width_mm/1000+1e-12
                    matches = np.flatnonzero(close)
                    if len(matches):
                        representative = keep[int(matches[0])]
                if representative is None:
                    keep.append(i)
                else:
                    removed.append(dict(source_id=ids[i], representative_id=ids[representative]))
        return keep, dict(**self.metadata(), input_count=len(matrices), retained_count=len(keep),
                         kept_source_ids=[ids[i] for i in keep], removed=removed,
                         elapsed_s=time.perf_counter()-start)
