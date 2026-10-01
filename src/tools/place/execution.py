"""Agent-chosen placement, explicit release, and target-free robot motion.

Placement moves a retained grasp to a chosen contact-centre pose and stops.
Opening is a later agent action. Scene data always comes from current RGB-D;
no target/contact exemption is granted for these transport operations.
"""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

import numpy as np

from src.core.action_feedback import ActionPreconditionError
from src.tools.motion import planning as motion
from src.tools.grasp.geometry import observed_cloud_statistics, xy_candidates

_PLACE_CANDIDATE_PREVIEW_Z_OFFSET_M = .05


class _CyanGripperMesh:
    """Preserve native triangles; use the shared renderer's cyan palette."""
    preview_view_id = 'agentview'
    preview_padding_fraction = .5
    preview_supersampling = 4
    def __init__(self, assets):
        self.assets = assets

    def load_gripper_mesh(self, width):
        parts, metadata = self.assets.load_gripper_mesh(width)
        triangles = [np.asarray(part, dtype=float) for part in parts.values() if len(part)]
        if not triangles:
            raise ValueError('native gripper mesh has no triangles')
        # mesh_draw's established left_finger palette is cyan. The key controls
        # color/opacity only; all hand/finger geometry remains in this mesh.
        return {'left_finger': np.concatenate(triangles)}, metadata


def _real_vector(value, size, name):
    array = np.asarray(value)
    if array.shape != (size,) or array.dtype.kind not in 'iuf' or not np.isfinite(array).all():
        raise ValueError(f'{name} requires {size} finite real values')
    return array.astype(float, copy=True)


def observed_scene(frames):
    """Unproject all current measured views in their calibrated base frame."""
    clouds = []
    for frame in frames:
        depth = np.asarray(frame.depth_m, dtype=float)
        intrinsics = np.asarray(frame.intrinsics, dtype=float)
        if (depth.ndim != 2 or intrinsics.shape != (3, 3)
                or not np.isfinite(intrinsics).all() or np.any(np.diag(intrinsics)[:2] <= 0)):
            raise ValueError('finite calibrated RGB-D intrinsics required')
        y, x = np.nonzero(np.isfinite(depth) & (depth > .02) & (depth < 3.))
        z = depth[y, x]
        camera = np.column_stack(((x-intrinsics[0, 2])*z/intrinsics[0, 0],
                                  (y-intrinsics[1, 2])*z/intrinsics[1, 1], z))
        calibration = np.eye(4)
        calibration[:3, :3] = np.asarray(frame.camera_to_base.rotation)
        calibration[:3, 3] = np.asarray(frame.camera_to_base.translation)
        calibration = motion._transform(calibration)
        clouds.append(camera @ calibration[:3, :3].T + calibration[:3, 3])
    if not clouds or not any(len(cloud) for cloud in clouds):
        raise ValueError('current RGB-D has no valid measured scene points')
    return np.vstack(clouds)


class PlacementMotionMixin:
    def _explicit_current_scene(self):
        observation_id = self.latest_observation_id
        if observation_id is None:
            observation_id = self.observe()['observation_id']
        self.point_adapter._check_current(observation_id)
        return observation_id, observed_scene(self.point_adapter.frames[observation_id])

    def _explicit_destination(self, destination_ref):
        memory = self.destinations.get(destination_ref)
        if memory is None:
            raise ValueError('registered observed destination required')
        return memory

    def explicit_place_candidates(self, destination_ref):
        from src.tools.grasp.backend import robot_geometry_context
        destination = self._explicit_destination(destination_ref)
        click = self.clicked_points.get(destination.point_ref, destination.anchor)
        result = dict(destination_ref=destination_ref, point_ref=destination.point_ref,
            observation_id=destination.observation_id,
            statistics=observed_cloud_statistics(destination.points),
            xy_candidates=xy_candidates(destination.points, clicked_xyz_m=click),
            frame='connector_base', position_reference='jaw_contact_center', units='metres',
            **robot_geometry_context(self),
            release_policy='move to selected pose, inspect fresh RGB, then explicitly release',
            geometry_limitations='recorded observed destination surfaces; displacement is not tracked')
        previews = self._explicit_candidate_overlays(result)
        result['image_refs'] = [ref for preview in previews for ref in preview['image_refs']]
        result['candidate_previews'] = previews
        result['height_decision_required'] = True
        self._record('explicit_place_candidates', result)
        return result

    def _explicit_candidate_overlays(self, public):
        from src.tools.grasp.backend import contact_to_hand, hand_to_contact
        current = motion._pose_transform(self.connector.get_ee_pose())
        current_contact = hand_to_contact(current @ np.linalg.inv(self.grasp_to_ee), self.jaw_offset_m)
        previews = []
        for candidate in public['xy_candidates']:
            contact = current_contact.copy()
            contact[:3, 3] = candidate['observed_xyz_m']
            contact[2, 3] += _PLACE_CANDIDATE_PREVIEW_Z_OFFSET_M
            hand = contact_to_hand(contact, self.jaw_offset_m)
            label = candidate['candidate_id']
            xyz = contact[:3, 3]
            refs = self._explicit_project_place_hand(public['observation_id'], hand,
                caption=f'{label}: X={xyz[0]:.3f} Y={xyz[1]:.3f} preview Z={xyz[2]:.3f} m',
                height_status='PREVIEW: surface Z + 0.05 m; execution height not selected')
            previews.append(dict(xy_source=label,
                reference_contact_center_xyz_m=list(candidate['observed_xyz_m']),
                preview_contact_center_xyz_m=xyz.tolist(),
                preview_z_offset_m=_PLACE_CANDIDATE_PREVIEW_Z_OFFSET_M,
                height_status='reference_surface_plus_offset_not_execution_height', executable=False,
                image_refs=refs))
        return previews

    def _explicit_project_place_hand(self, observation_id, hand, *, caption, height_status):
        """Use the actual Refiner projector and native mesh, uniformly cyan."""
        from pathlib import Path
        from PIL import Image, ImageDraw
        from src.tools.pose_editor.refinement import preview_refined_pose
        # RoboLab agentview is over_shoulder_left_camera; do not substitute wrist.
        frames = [frame for frame in self.point_adapter.frames[observation_id]
                  if frame.view_id == 'agentview']
        if not frames:
            raise ValueError('left shoulder agentview required for place projection')
        directory = self.output_dir / ('place_projection_'+uuid4().hex)
        paths = preview_refined_pose(frames, hand, directory,
            tcp_offset_z_m=self.jaw_offset_m,
            expected_open_width_m=(self.max_gripper_width_m if self.grasp_jaw_width_m is None
                                   else self.grasp_jaw_width_m),
            mesh_source=_CyanGripperMesh(self.gripper_assets))
        if not paths:
            raise ValueError('placement contact pose preview unavailable')
        refs = []
        for path in paths:
            path = Path(path)
            with Image.open(path) as source:
                image = source.convert('RGB')
            draw = ImageDraw.Draw(image)
            draw.text((8, 26), caption, fill='cyan', stroke_width=1, stroke_fill='black')
            draw.text((8, 44), height_status, fill='cyan', stroke_width=1, stroke_fill='black')
            image.save(path)
            refs.append(self.images.add(path))
        return refs

    def _explicit_require_attachment(self):
        if (self.held_plan is None or self.grasp_attachment is None
                or not getattr(self.held_plan, 'grasp_executed', False)):
            raise ValueError('placement requires an executed grasp and measured attachment')

    def _explicit_save_motion_pose(self, pose, observation_id, scene, *, operation, instruction,
                             orientation_policy='preserve_current'):
        ref = 'waypoint_'+uuid4().hex
        self.view_proposals[ref] = dict(epoch=self.epoch, observation_id=observation_id,
            target_ref=None, pose=motion._transform(pose), purpose='transport',
            instruction=deepcopy(instruction), scene=scene.copy(), target_points=None,
            start=self._robot_state(), orientation_policy=orientation_policy,
            ee_from_optical=None, motion_operation=operation)
        self._record('waypoint_proposed', dict(waypoint_ref=ref, operation=operation,
            observation_id=observation_id, epoch=self.epoch, ee_pose=pose.tolist(),
            instruction=instruction, geometry_source='current calibrated RGB-D full scene',
            contact_exemptions=False))
        return ref

    def _explicit_place_preview(self, waypoint_ref):
        p = self._view(waypoint_ref)
        hand = p['pose'] @ np.linalg.inv(self.grasp_to_ee)
        xyz = motion._transform(hand)[:3, 3] + self.jaw_offset_m * hand[:3, 2]
        refs = self._explicit_project_place_hand(p['observation_id'], hand,
            caption=f'Agent placement: X={xyz[0]:.3f} Y={xyz[1]:.3f} Z={xyz[2]:.3f} m',
            height_status='SELECTED CONTACT POSE; movement and release are separate actions')
        if not refs:
            raise ValueError('placement contact pose preview unavailable')
        return refs

    def _explicit_publish_place(self, waypoint_ref, metadata):
        previews = self._explicit_place_preview(waypoint_ref)
        validation = self.validate_view(waypoint_ref, motion='planned')
        candidate_ref = 'place_'+uuid4().hex
        if not hasattr(self, '_explicit_place_proposals'):
            self._explicit_place_proposals = {}
        self._explicit_place_proposals[candidate_ref] = dict(waypoint_ref=waypoint_ref,
            epoch=self.epoch, validation=deepcopy(validation), **deepcopy(metadata))
        result = dict(candidate_ref=candidate_ref, waypoint_ref=waypoint_ref,
            accepted=bool(validation['accepted']), executable=bool(validation['accepted']),
            image_refs=previews, validation=validation, **deepcopy(metadata),
            robot_motion=False, release_commanded=False)
        self._record('place_prepared', result)
        return result

    def explicit_prepare_place(self, destination_ref, xy_source, xy_m, height, transit_height):
        from src.tools.grasp.backend import contact_to_hand, hand_to_contact, height_from_geometry
        self._explicit_require_attachment()
        public = self.explicit_place_candidates(destination_ref)
        choices = {row['candidate_id']: row for row in public['xy_candidates']}
        if xy_source not in choices:
            raise ValueError('xy_source must identify an offered measured candidate')
        xy = _real_vector(xy_m, 2, 'xy_m')
        clicked = choices.get('clicked', {}).get('observed_xyz_m')
        context = dict(current_tcp_z_m=public['current_tcp_z_m'],
                       grasp_z_m=public['last_grasp_target_tcp_z_m'])
        z = height_from_geometry(height, public['statistics'], clicked, **context)
        travel_z = height_from_geometry(transit_height, public['statistics'], clicked, **context)
        current = motion._pose_transform(self.connector.get_ee_pose())
        contact = hand_to_contact(current @ np.linalg.inv(self.grasp_to_ee), self.jaw_offset_m)
        contact[:3, 3] = [*xy, z]
        pose = contact_to_hand(contact, self.jaw_offset_m) @ self.grasp_to_ee
        observation_id, scene = self._explicit_current_scene()
        metadata = dict(destination_ref=destination_ref, xy_source=xy_source,
            xy_m=xy.tolist(), height=deepcopy(height), contact_center_xyz_m=contact[:3, 3].tolist(),
            transit_height=deepcopy(transit_height), transit_z_m=travel_z,
            height_reference_context=context,
            frame='connector_base', position_reference='jaw_contact_center',
            translation_mm=[0., 0., 0.], rotation_deg=[0., 0., 0.])
        ref = self._explicit_save_place_route(pose, observation_id, scene, metadata)
        return self._explicit_publish_place(ref, metadata)

    def _explicit_save_place_route(self, pose, observation_id, scene, metadata):
        from src.tools.grasp.backend import contact_to_hand, hand_to_contact
        current = motion._pose_transform(self.connector.get_ee_pose())
        contact = hand_to_contact(pose @ np.linalg.inv(self.grasp_to_ee), self.jaw_offset_m)
        travel_z = metadata['transit_z_m']
        if not np.isfinite(travel_z) or travel_z < contact[2, 3]:
            raise ValueError('place transit Z must not be below final target Z; no automatic raising')
        initial = hand_to_contact(current @ np.linalg.inv(self.grasp_to_ee), self.jaw_offset_m)
        initial[2, 3] = travel_z
        above = contact.copy()
        above[2, 3] = travel_z
        targets = tuple(contact_to_hand(t, self.jaw_offset_m) @ self.grasp_to_ee
                        for t in (initial, above, contact))
        ref = self._explicit_save_motion_pose(pose, observation_id, scene, operation='place',
            instruction=metadata, orientation_policy='place_transit')
        self.view_proposals[ref]['place_targets'] = targets
        return ref

    def _plan_waypoint(self, proposal, scene, checker):
        targets = proposal.get('place_targets')
        if targets is None:
            return super()._plan_waypoint(proposal, scene, checker)
        if proposal.get('motion') == 'linear':
            raise ValueError('place transit requires planned motion through all three targets')
        # Existing validate_view checks the entire returned robot/payload path.
        return self._explicit_chain_planner(self.connector, targets, scene, self.motion_config, None,
            target_labels=('initial_lift', 'transit', 'waypoint'))

    def _explicit_place_proposal(self, candidate_ref):
        value = getattr(self, '_explicit_place_proposals', {}).get(candidate_ref)
        if value is None or value['epoch'] != self.epoch:
            raise ValueError('current placement candidate required')
        self._view(value['waypoint_ref'])
        return value

    def explicit_inspect_place(self, candidate_ref):
        value = self._explicit_place_proposal(candidate_ref)
        return dict(candidate_ref=candidate_ref, image_refs=self._explicit_place_preview(value['waypoint_ref']),
            contact_center_xyz_m=value['contact_center_xyz_m'], validation=deepcopy(value['validation']),
            transit_z_m=value['transit_z_m'], transit_height=deepcopy(value['transit_height']),
            executable=bool(value['validation']['accepted']), robot_motion=False,
            release_commanded=False)

    def explicit_adjust_place(self, candidate_ref, dx_mm, dy_mm, dz_mm,
                        roll_deg, pitch_deg, yaw_deg):
        from src.tools.grasp.backend import hand_to_contact
        from src.tools.pose_editor.refinement import checked_adjustment, refined_grasp_pose
        from src.tools.pose_editor.geometry import checked_translation
        self._explicit_require_attachment()
        original = self._explicit_place_proposal(candidate_ref)
        shift, total_shift = checked_translation((dx_mm, dy_mm, dz_mm), original['translation_mm'],
            step_limit_mm=30, cumulative_limit_mm=90,
            unrestricted=getattr(self, 'unrestricted_pose_translation', False))
        angles, total_angles = checked_adjustment((roll_deg, pitch_deg, yaw_deg), original['rotation_deg'],
            unrestricted=getattr(self, 'unrestricted_pose_rotation', False))
        proposal = self._view(original['waypoint_ref'])
        hand = proposal['pose'] @ np.linalg.inv(self.grasp_to_ee)
        hand = refined_grasp_pose(hand, angles, tcp_offset_z_m=self.jaw_offset_m)
        pose = hand @ self.grasp_to_ee
        pose[:3, 3] += np.asarray(shift)/1000.
        contact = hand_to_contact(pose @ np.linalg.inv(self.grasp_to_ee), self.jaw_offset_m)
        metadata = {k: deepcopy(v) for k, v in original.items()
                    if k not in ('waypoint_ref', 'epoch', 'validation')}
        metadata.update(origin_candidate_ref=candidate_ref,
            contact_center_xyz_m=contact[:3, 3].tolist(), translation_mm=list(total_shift),
            rotation_deg=list(total_angles), translation_frame='connector_base_xyz',
            rotation_frame='gripper_local_axes_about_contact_center')
        ref = self._explicit_save_place_route(pose, proposal['observation_id'], proposal['scene'], metadata)
        return self._explicit_publish_place(ref, metadata)

    def _explicit_invalidate_release_arrival(self):
        previous = getattr(self, 'last_place', None)
        if previous and previous.get('arrival_current'):
            self._explicit_place_release_blocked = True
            self.last_place = {**previous, 'arrival_current': False}

    @contextmanager
    def _explicit_preserve_command(self):
        """Retain commanded aperture even through the inherited hold reminder."""
        env = getattr(self.connector, 'env', None)
        previous = getattr(self, '_explicit_motion_gripper_command', None)
        command = dict(fraction=getattr(self.connector, '_gripper_fraction', None),
                       width=getattr(env, '_width_target', None))
        self._explicit_motion_gripper_command = command
        try:
            yield
        finally:
            self._explicit_restore_command(command)
            self._explicit_motion_gripper_command = previous

    def _explicit_restore_command(self, command):
        env = getattr(self.connector, 'env', None)
        if command['width'] is not None and callable(getattr(env, '_set_gripper_width', None)):
            env._set_gripper_width(command['width'])
            if command['fraction'] is not None:
                self.connector._gripper_fraction = command['fraction']
        elif command['fraction'] is not None:
            self.connector.set_gripper(command['fraction'])

    def _execute_waypoint(self, plan):
        command = getattr(self, '_explicit_motion_gripper_command', None)
        if command is not None:
            self._explicit_restore_command(command)
        return super()._execute_waypoint(plan)

    def execute_view(self, waypoint_ref, validation_ref):
        # Any waypoint motion invalidates the previously reached placement,
        # including calls through the shared waypoint tools.
        before_epoch = self.epoch
        try:
            return super().execute_view(waypoint_ref, validation_ref)
        finally:
            if self.epoch != before_epoch:
                self._explicit_invalidate_release_arrival()

    def _explicit_execute_waypoint(self, waypoint_ref, validation):
        if not validation.get('accepted'):
            return dict(status='not_executed', view_status='not_executed',
                accepted=False, reason_code=validation.get('reason_code', 'waypoint_path_rejected'),
                validation=validation, release_commanded=False, executed=False, state_changed=False)
        self._view(waypoint_ref)
        token = self.view_validations.get(validation['validation_ref'])
        if token is None or token[0] != waypoint_ref or token[1] != self._robot_state():
            raise ValueError('waypoint validation stale or mismatched')
        before_epoch = self.epoch
        if not self.plan_only:
            self._explicit_invalidate_release_arrival()
        execution_ref = 'motion_'+uuid4().hex
        try:
            with self._explicit_preserve_command():
                result = self.execute_view(waypoint_ref, validation['validation_ref'])
        except Exception as exc:
            if self.epoch == before_epoch:
                raise
            result = dict(view_status='requested_view_failed', error=repr(exc))
            try:
                result.update(self.observe())
            except Exception as observation_error:
                result['observation_error'] = repr(observation_error)
        changed = self.epoch != before_epoch
        public = {**result, 'executed': changed, 'state_changed': changed,
            'status': 'succeeded' if result.get('view_status') == 'achieved' else (
                'failed' if changed else 'not_executed')}
        if changed:
            public['execution_ref'] = execution_ref
        if result.get('observation_id'):
            public['observation'] = {key: result[key] for key in ('observation_id', 'views') if key in result}
        self._record('waypoint_execution', dict(waypoint_ref=waypoint_ref, **public))
        return public

    def explicit_execute_place(self, candidate_ref):
        self._explicit_require_attachment()
        value = self._explicit_place_proposal(candidate_ref)
        self._explicit_place_release_blocked = True
        try:
            result = self._explicit_execute_waypoint(value['waypoint_ref'], value['validation'])
        except Exception as exc:
            self._record('place_move_failed', dict(candidate_ref=candidate_ref, error=repr(exc)))
            raise
        achieved = result.get('view_status') == 'achieved'
        self._explicit_place_release_blocked = not achieved
        public = {**result, 'candidate_ref': candidate_ref, 'destination_ref': value['destination_ref'],
            'release_required': True, 'release_commanded': False, 'arrival_current': achieved,
            'placement_verified': False, 'contact_center_xyz_m': value['contact_center_xyz_m']}
        self.last_place = deepcopy(public)
        self._record('place_moved', public)
        return public

    def move_vertical(self, dz_m):
        dz = _real_vector([dz_m], 1, 'dz_m')[0]
        observation_id, scene = self._explicit_current_scene()
        pose = motion._pose_transform(self.connector.get_ee_pose())
        pose[2, 3] += dz
        ref = self._explicit_save_motion_pose(pose, observation_id, scene, operation='vertical',
            instruction=dict(base_displacement_m=[0., 0., float(dz)]))
        validation = self.validate_view(ref, motion='linear')
        result = self._explicit_execute_waypoint(ref, validation)
        public = {**result, 'operation': 'move_vertical', 'requested_dz_m': float(dz),
                  'release_commanded': False}
        self._record('vertical_motion', public)
        return public

    def _explicit_make_checker(self):
        from src.tools.motion.path_collision import make_candidate_path_collision
        return make_candidate_path_collision(self.connector, clearance_m=.0005)

    def _explicit_home_scene(self, checker, scene):
        scene, removal = checker.remove_captured_robot(scene)
        if self.held_plan is not None:
            if self.grasp_attachment is None:
                raise ValueError('measured attachment required for held home motion')
            from scipy.spatial import cKDTree
            relative = motion._pose_transform(self.connector.get_ee_pose()) @ np.linalg.inv(self.grasp_attachment)
            held = self.held_plan.target_points @ relative[:3, :3].T + relative[:3, 3]
            distances, _ = cKDTree(held).query(scene)
            keep = distances > .012
            removal['removed_held_samples'] = int((~keep).sum())
            scene = scene[keep]
        return scene, removal

    def _explicit_advance_motion_epoch(self):
        self._explicit_invalidate_release_arrival()
        self.epoch += 1
        self.validations.clear()
        self.view_validations.clear()
        self.latest_fused_ref = self.latest_observation_id = self.point_adapter.latest = None

    def goto_home_joint_position(self):
        from src.tools.motion.robot_state import _current_robot_state, _segments_safe
        from src.tools.motion.tolerances import cartesian_tolerances
        home = getattr(self, 'home_joints', None)
        if home is None:
            home = getattr(self.connector, '_home_joints', None)
        if home is None:
            raise ValueError('configured native or recorded reset home joints required')
        target = _real_vector(home, 7, 'home joints')
        validator = getattr(getattr(self.connector, 'env', None), '_validate_joints', None)
        if callable(validator):
            target = np.asarray(validator(target), dtype=float)
        observation_id, scene = self._explicit_current_scene()
        _, start_joints = _current_robot_state(self.connector)
        start = _real_vector(start_joints, 7, 'measured starting joints')
        target_pose = motion._transform(self.connector.ik.model.fk(target))
        count = max(1, int(np.ceil(np.max(np.abs(target-start))/.025)))
        segment = {'waypoints': [{'positions': row.tolist()} for row in np.linspace(start, target, count+1)]}
        plan = SimpleNamespace(segments=(segment,), targets=(motion.transform_to_pose(target_pose),),
            start_joints=tuple(start), target_labels=('waypoint',), segment_labels=('waypoint',),
            transit_policy='home_joints', high_transit_z_m=None)
        if not _segments_safe(plan.segments, start_joints=start):
            raise motion.MotionPlanningError('home joint path violates continuity')
        checker = self._explicit_make_checker()
        checked_scene, removal = self._explicit_home_scene(checker, scene)
        width = self.grasp_jaw_width_m if self.held_plan is not None else (
            0. if self.closed_push else self.max_gripper_width_m)
        evidence = checker.check(plan, checked_scene, stop_label='waypoint', jaw_width_m=width,
                                 ignore_initial_contacts=True)
        if evidence.get('accepted') and self.held_plan is not None:
            evidence['held_object'] = self._check_held_path(checker, plan, checked_scene)
            evidence['accepted'] = bool(evidence['held_object']['accepted'])
        evidence['capture_removal'] = removal
        self._record('home_validation', dict(observation_id=observation_id,
            home_joints_rad=target.tolist(), evidence=evidence, contact_exemptions=False))
        if not evidence.get('accepted'):
            return dict(status='not_executed', accepted=False, reason_code='home_path_rejected',
                        validation=evidence, release_commanded=False, executed=False, state_changed=False)
        if self.plan_only:
            return dict(status='not_executed', accepted=True, reason_code='plan_only',
                        validation=evidence, release_commanded=False, executed=False, state_changed=False)
        _, actual_start = _current_robot_state(self.connector)
        if not np.allclose(actual_start, start, atol=1e-10, rtol=0):
            raise motion.MotionPlanningError('home plan became stale before execution')
        self._explicit_advance_motion_epoch()
        execution_ref = 'home_'+uuid4().hex
        diagnostics, error = [], None
        try:
            if self.recorder:
                self.recorder.register_plan(plan, kind='goto_home_joint_position')
            context = self.recorder.active('goto_home_joint_position', execution_ref=execution_ref) if self.recorder else nullcontext()
            with context:
                diagnostics = motion._execute_checked(self.connector, plan.segments, plan.targets,
                    collision_checks_enabled=self.motion_config.collision_checks_enabled)
        except Exception as exc:
            error = repr(exc)
            diagnostics = getattr(exc, 'evidence', {}).get('execution_diagnostics', diagnostics)
        pose, joints = _current_robot_state(self.connector)
        measured = motion._pose_transform(pose)
        joint_error = float(np.max(np.abs(_real_vector(joints, 7, 'measured joints')-target)))
        position_error = float(np.linalg.norm(measured[:3, 3]-target_pose[:3, 3]))
        angle_error = float(np.arccos(np.clip((np.trace(measured[:3, :3].T @ target_pose[:3, :3])-1)/2, -1., 1.)))
        joint_limit = getattr(getattr(self.connector, 'env', None), 'motion_joint_tolerance_rad', None) or .01
        position_limit, angle_limit = cartesian_tolerances(self.connector)
        achieved = error is None and joint_error <= joint_limit and position_error <= position_limit and angle_error <= angle_limit
        observation = self.observe()
        result = dict(**observation, status='succeeded' if achieved else 'failed',
            reason_code='home_arrived' if achieved else 'home_execution_incomplete',
            execution_ref=execution_ref, home_joints_rad=target.tolist(),
            joint_max_error_rad=joint_error, position_error_m=position_error,
            orientation_error_rad=angle_error, execution_diagnostics=diagnostics,
            error=error, release_commanded=False, success_verified=False,
            executed=True, state_changed=True, observation=deepcopy(observation),
            hold_status='unverified; unchanged gripper command is not proof of holding')
        self._record('home_executed', result)
        return result

    def release(self):
        from src.runtime.budget import gripper_settle_steps
        from src.simulator.robolab.adapter import is_successful_episode_end
        if getattr(self, '_explicit_place_release_blocked', False):
            raise ActionPreconditionError('release_requires_placement_retry')
        if self.plan_only:
            return dict(status='not_executed', reason_code='plan_only', release_commanded=False,
                        executed=False, state_changed=False)
        execution_ref = 'release_'+uuid4().hex
        self._explicit_advance_motion_epoch()
        episode_ended = False
        try:
            context = self.recorder.active('release', execution_ref=execution_ref) if self.recorder else nullcontext()
            with context:
                try:
                    self.connector.open_gripper(settle_steps=gripper_settle_steps(self.connector, 'open', 40))
                except RuntimeError as exc:
                    if not is_successful_episode_end(self.connector, exc):
                        raise
                    episode_ended = True
        except Exception as exc:
            result = dict(status='failed', execution_ref=execution_ref,
                reason_code='release_execution_incomplete', release_commanded=None,
                success_verified=False, error=repr(exc), executed=True, state_changed=True)
            self._explicit_release_observation(result)
            self.placement_execution = deepcopy(result)
            self._record('release', result)
            return result
        self.held_plan = self.grasp_attachment = self.grasp_jaw_width_m = None
        self.closed_push = self.grasp_attempted = False
        self.push_contact_height_m = None
        self.destinations.clear()
        self.selected_destination_ref = None
        self._explicit_place_release_blocked = False
        self.last_place = None
        getattr(self, '_explicit_place_proposals', {}).clear()
        result = dict(status='succeeded', execution_ref=execution_ref,
            release_commanded=True, success_verified=False, executed=True, state_changed=True,
            execution_feedback=dict(release_commanded=True, task_success=None,
                interpretation='explicit opening command completed; placement/task success requires observed/native verification'))
        if episode_ended:
            result.update(terminal=True, reason_code='episode_terminated')
            result['execution_feedback']['interpretation'] = (
                'opening command issued; episode ended during settling')
        self._explicit_release_observation(result)
        self.placement_execution = deepcopy(result)
        self._record('release', result)
        return result

    def _explicit_release_observation(self, result):
        try:
            observation = self.observe()
            result.update(observation=deepcopy(observation), **observation)
        except Exception as exc:
            result['observation_error'] = repr(exc)
