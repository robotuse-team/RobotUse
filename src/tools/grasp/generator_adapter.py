"""Retarget GraspGen Panda hand poses and meshes to the LIBERO Panda.

GraspGen attaches the same finger meshes at hand-local z=58.4 mm; LIBERO
attaches them at 52.4 mm. Executing the unmodified hand pose therefore leaves
the real fingers 6 mm short along the approach axis. After confidence filtering,
translate the hand by +6 mm along its LOCAL +Z and use LIBERO finger origins
for subsequent existing collision checks and previews. This preserves the
predicted finger surfaces in world space; the palm moves with the real hand.

This is output retargeting, not retraining or a GraspGen source modification.
Scores still describe the original predictions. Do not fold this displacement
into hand-to-EE calibration, apply it twice, or add a world-Z displacement.
No arrival gate, candidate rejection rule, or controller change lives here.
"""
from __future__ import annotations

import numpy as np


class GraspGenLiberoAdapter:
    name = "graspgen_libero_panda"
    graspgen_finger_z_m = .0584
    libero_finger_z_m = .0524
    approach_offset_m = graspgen_finger_z_m - libero_finger_z_m

    @classmethod
    def adapt_poses(cls, poses):
        """Copy one pose or a batch, retaining rotations and candidate order."""
        adapted = np.array(poses, dtype=np.float64, copy=True)
        adapted[..., :3, 3] += cls.approach_offset_m * adapted[..., :3, 2]
        return adapted

    @classmethod
    def finger_points(cls, points):
        """Express GraspGen finger geometry at the real hand-local origin."""
        return np.asarray(points, dtype=float) - [0., 0., cls.approach_offset_m]

    @classmethod
    def adapt_mesh_parts(cls, parts):
        return {name: cls.finger_points(triangles) if name in ("left_finger", "right_finger")
                else np.array(triangles, copy=True) for name, triangles in parts.items()}

    @classmethod
    def collision_mesh(cls, graspgen_root, *, expected_open_width_m=None):
        """Supply the official scene-filter API with the retargeted mesh."""
        import trimesh
        from src.tools.pose_editor.inspection import load_panda_mesh
        parts, _ = load_panda_mesh(graspgen_root, expected_open_width_m, libero_adapter=True)
        # Preserve upstream's fingers-then-palm face order for seeded sampling.
        triangles = np.concatenate([parts[k] for k in ("left_finger", "right_finger", "hand")])
        return trimesh.Trimesh(vertices=triangles.reshape(-1, 3),
            faces=np.arange(triangles.size // 3).reshape(-1, 3), process=True)

    @classmethod
    def metadata(cls):
        return {"name": cls.name, "approach_offset_m": cls.approach_offset_m,
                "translation_frame": "grasp hand local +Z",
                "graspgen_finger_z_m": cls.graspgen_finger_z_m,
                "libero_finger_z_m": cls.libero_finger_z_m,
                "score_role": "original GraspGen confidence; corrected pose not rescored"}
