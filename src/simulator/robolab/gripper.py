"""Native 2F-85 mesh source and grasp-frame calibration."""
from pathlib import Path
import numpy as np
from src.simulator.robolab.robot_model import RoboLabRobotModel

# RoboLab droid.py EEF_OFFSET_ROT; grasp +Z approaches, +X closes the jaws.
FLANGE_FROM_GRASP = np.array([[0.,0.,1.,0.],[-1.,0.,0.,0.],[0.,-1.,0.,0.],[0.,0.,0.,1.]])
GRASP_TO_EE = np.linalg.inv(FLANGE_FROM_GRASP)


class RoboLabGripperAssets:
    gripper_name = 'robotiq_2f_85'
    max_opening_m = .085
    jaw_center_offset_m = .136

    def __init__(self, asset_path, source_checkout):
        self.asset_path = Path(asset_path)
        self.source_checkout = Path(source_checkout)
        self._model = None
        self._cache = {}
        self._closing_bounds = None

    def __fspath__(self):
        return str(self.source_checkout)

    def _closing_volume_bounds(self):
        """Native fingertip Y/Z envelope throughout the 2F-85 linkage stroke.

        Robotiq pads move forward as they close. Using only the fully open
        fingers (or Panda's finger depths) misses observed contact surfaces.
        This is robot-only geometry; no simulator object geometry is read.
        """
        if self._closing_bounds is None:
            from gap.envs.robolab_control import HOME_JOINTS
            parts, _ = self.load_gripper_mesh(self.max_opening_m)
            pads = []
            for body in ('left_inner_finger', 'right_inner_finger'):
                names = [name for name in parts if self._model.geom_body[name] == body]
                if not names:
                    raise ValueError(f'native Robotiq fingertip geometry missing: {body}')
                # Each native inner finger owns a linkage mesh and a distal pad.
                pads.append(max(names, key=lambda name: parts[name][..., 2].max()))
            bounds = []
            for angle in np.linspace(0., np.pi / 4, 33):
                self._model.set_joints(HOME_JOINTS, angle)
                grasp_from_base = np.linalg.inv(self._model.body_matrix('base_link') @ FLANGE_FROM_GRASP)
                for name in pads:
                    transform = grasp_from_base @ self._model.body_matrix(self._model.geom_body[name])
                    vertices = self._model.visual_triangles[name].reshape(-1, 3)
                    local = vertices @ transform[:3, :3].T + transform[:3, 3]
                    bounds.extend([local[:, 1:].min(0), local[:, 1:].max(0)])
            # 57.15mm native linkage, <=1.41-degree samples: <0.71mm motion
            # from the nearest sample. Guard Y/Z by 1mm; X padding is separate.
            self._closing_bounds = (np.min(bounds, axis=0) - .001,
                                    np.max(bounds, axis=0) + .001)
        return tuple(bound.copy() for bound in self._closing_bounds)

    def contact_opening(self, pose, object_points, *, padding_per_side_m=.010):
        """Observed aperture at a fixed grasp centre, with per-side clearance."""
        from gap.envs.robolab_control import rigid_pose
        pose = rigid_pose(pose)
        points = np.asarray(object_points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('finite observed Nx3 cloud required')
        if not np.isfinite(padding_per_side_m) or padding_per_side_m < 0:
            raise ValueError('grasp opening padding must be nonnegative and finite')
        lower, upper = self._closing_volume_bounds()
        local = (points - pose[:3, 3]) @ pose[:3, :3]
        captured = local[np.all((local[:, 1:] >= lower) & (local[:, 1:] <= upper), axis=1)]
        # Do not discard points outside the maximum X aperture: their presence
        # must saturate the opening, never make a wide target look narrow.
        required = float(2 * np.max(np.abs(captured[:, 0]))) if len(captured) >= 3 else None
        requested = required + 2 * padding_per_side_m if required is not None else self.max_opening_m
        opening = float(np.clip(np.ceil(requested * 1000) / 1000, .001, self.max_opening_m))
        return dict(open_width_m=opening, required_width_m=required,
            padding_per_side_m=float(padding_per_side_m),
            effective_padding_per_side_m=(opening-required)/2 if required is not None else None,
            clamped_to_maximum=requested > self.max_opening_m,
            observed_capture_points=len(captured),
            closing_volume_yz_bounds_m=[lower.tolist(), upper.tolist()],
            method='observed points in native Robotiq pad sweep; fixed grasp centre; 1mm upward rounding'
                if required is not None else 'insufficient observed capture points; use maximum opening')

    def remove_captured_points(self, scene, captured_hand, captured_width_m):
        """Remove captured visual surfaces without filling gaps with collision hulls."""
        import trimesh
        from src.tools.motion.planning import _transform
        from src.simulator.robolab.visual_query import remove_visual_points
        scene = np.asarray(scene)
        parts, _ = self.load_gripper_mesh(captured_width_m)
        meshes = {name: trimesh.Trimesh(vertices=triangles.reshape(-1, 3),
            faces=np.arange(triangles.size//3).reshape(-1, 3), process=True)
            for name, triangles in parts.items()}
        pose = _transform(captured_hand)
        poses = {name: (pose[:3, 3], pose[:3, :3]) for name in meshes}
        retained, audit = remove_visual_points(scene, meshes, poses, margin_m=.002)
        return retained, {**audit, 'removed_points': len(scene)-len(retained),
            'retained_points': len(retained), 'tolerance_m': .002,
            'policy': 'captured native visual gripper only; target points preserved; arm and unknown space not cleared'}

    def load_gripper_mesh(self, expected_open_width_m=None):
        from gap.envs.robolab_control import HOME_JOINTS, robotiq_width_to_angle
        width = self.max_opening_m if expected_open_width_m is None else float(expected_open_width_m)
        angle = min(np.pi/4, robotiq_width_to_angle(width))
        if width not in self._cache:
            if self._model is None: self._model = RoboLabRobotModel(self.asset_path)
            self._model.set_joints(HOME_JOINTS,angle)
            base_from_grasp = self._model.body_matrix('base_link') @ FLANGE_FROM_GRASP
            parts={}
            for name,triangles in self._model.visual_triangles.items():
                body=self._model.geom_body[name]
                if body.startswith('panda_link'):continue
                transform=np.linalg.inv(base_from_grasp) @ self._model.body_matrix(body)
                parts[name]=triangles @ transform[:3,:3].T + transform[:3,3]
            vertices=np.concatenate(list(parts.values())).reshape(-1,3)
            self._cache[width]=(parts,dict(gripper_name=self.gripper_name,opening_m=width,
                nominal_open=expected_open_width_m is None,source=str(self.asset_path),
                frame='GraspGenX canonical Robotiq base',units='metres',
                local_mesh_bounds_m=[vertices.min(0).tolist(),vertices.max(0).tolist()],
                grasp_to_public_ee=GRASP_TO_EE.tolist(),
                geometry_policy='native RoboLab robot meshes; no Panda retargeting'))
        parts,meta=self._cache[width]
        return {name:part.copy() for name,part in parts.items()},dict(meta)
