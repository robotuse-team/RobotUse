"""Optional placement rotation and held-object observation capabilities.

When disabled, nudge_place accepts translation only and observation requires
an empty, open hand. Global tool signatures remain unchanged.
"""
from dataclasses import dataclass
from uuid import uuid4
import numpy as np

from src.core.action_feedback import ActionPreconditionError


@dataclass(frozen=True)
class InteractionFeatures:
    place_rotation: bool = False
    held_observation: bool = False

    def __post_init__(self):
        if type(self.place_rotation) is not bool or type(self.held_observation) is not bool:
            raise ValueError('Interaction features must be booleans')

    def validate_runtime(self, args):
        if not (self.place_rotation or self.held_observation):
            return
        if args.environment != 'robolab' or not args.intent_driven or args.task_runner:
            raise ValueError('Interaction features require RoboLab intent execution without task-runner')
        if self.place_rotation and not args.pause_refine:
            raise ValueError('--place-rotation requires --pause-refine')

    def metadata(self):
        return dict(place_rotation=self.place_rotation, held_observation=self.held_observation,
            rotation_frame='gripper local axes about jaw centre', translation_frame='robot base XYZ',
            rotation_step_limit_deg=10, rotation_cumulative_limit_deg=30,
            translation_step_limit_mm=30, translation_cumulative_limit_mm=90,
            observation_scope='transport grasp with measured attachment; pusher/contact grasp excluded')


def configure_classes(backend, orchestrator, features):
    if not (features.place_rotation or features.held_observation):
        return backend, orchestrator
    return (type('Interaction'+backend.__name__, (InteractionBackend, backend), {'interaction_features': features}),
            type('Interaction'+orchestrator.__name__, (InteractionOrchestrator, orchestrator), {'interaction_features': features}))


class InteractionBackend:
    """Mixin for placement rotation and observation while holding an object."""
    def _check_observation_command(self):
        if not self.interaction_features.held_observation or self.held_plan is None:
            return super()._check_observation_command()
        if self.closed_push or self.grasp_mode != 'transport':
            raise ActionPreconditionError('held_observation_requires_transport_grasp')
        if self.grasp_attachment is None or not getattr(self.held_plan, 'grasp_executed', False):
            raise ActionPreconditionError('held_observation_requires_measured_attachment')
        # No gripper release or fabricated hold verification. The ordinary
        # validate_view checks the robot AND the measured held-object sweep.

    def validate_view(self, waypoint_ref, *args, **kwargs):
        if self._view(waypoint_ref)['purpose'] == 'observe':
            self._check_observation_command()
        return super().validate_view(waypoint_ref, *args, **kwargs)

    def execute_view(self, waypoint_ref, validation_ref):
        if self._view(waypoint_ref)['purpose'] == 'observe':
            self._check_observation_command()
        return super().execute_view(waypoint_ref, validation_ref)

    def execute_place_candidate(self, candidate_ref, validation_ref):
        if not self.interaction_features.place_rotation:
            return super().execute_place_candidate(candidate_ref, validation_ref)
        # Use the same measured, target-excluded scene as this place candidate.
        # It is only evidence of known surfaces, not a claim of a complete world.
        self._release_scene = None
        try:
            _, destination_ref, _, _, _ = self._place_candidate(candidate_ref)
            self._release_scene = self._placement_scene(self.held_plan, self.destinations[destination_ref])
        except (KeyError, ValueError, AttributeError):
            pass  # A nudge will explicitly reject missing geometry.
        try:
            return super().execute_place_candidate(candidate_ref, validation_ref)
        finally:
            self._release_scene = None

    def _place_previews(self, flight, target, delta, width):
        from PIL import Image, ImageDraw
        from src.tools.grasp.input_cards import camera_project
        hand = target @ np.linalg.inv(np.asarray(self.grasp_to_ee, dtype=float))
        refs = self._inflight_preview(flight['observation'], hand, width, 'inflight_'+uuid4().hex)
        if not refs:
            return []
        payload = self.held_plan.target_points @ delta[:3, :3].T + delta[:3, 3]
        for frame in self.point_adapter.frames[flight['observation']['observation_id']]:
            im = Image.fromarray(np.asarray(frame.rgb).copy()); draw = ImageDraw.Draw(im)
            xy, depth = camera_project(payload, frame)
            valid = np.isfinite(xy).all(axis=1) & (depth > 0)
            valid &= (xy[:, 0] >= 0) & (xy[:, 0] < im.width) & (xy[:, 1] >= 0) & (xy[:, 1] < im.height)
            visible = xy[valid]
            for x, y in visible[::max(1, len(visible)//1500)]:
                draw.ellipse((x-1, y-1, x+1, y+1), fill='orange')
            draw.text((8, 8), 'PENDING held-surface projection (orange); rigid/no-slip prediction, not new RGB',
                      fill='yellow', stroke_fill='black', stroke_width=1)
            path = self.output_dir/('inflight_payload_'+uuid4().hex+'.png')
            im.save(path); refs.append(self.images.add(path))
        return refs

    def nudge_inflight_place(self, dx_mm=0., dy_mm=0., dz_mm=0., roll_deg=0., pitch_deg=0., yaw_deg=0.):
        if not self.interaction_features.place_rotation:
            if any(v != 0 for v in (roll_deg, pitch_deg, yaw_deg)):
                raise ActionPreconditionError('place_rotation_disabled')
            return super().nudge_inflight_place(dx_mm, dy_mm, dz_mm)
        from src.tools.pose_editor.refinement import checked_adjustment, refined_grasp_pose, tcp_offset_from
        from src.tools.pose_editor.geometry import checked_translation
        from src.tools.motion.planning import _plan, _pose_transform
        from src.tools.motion.robot_state import _trajectory_end
        from src.tools.place.release_validation import check_release_geometry
        flight = self._inflight_place
        if flight is None:
            raise ValueError('no paused place execution')
        try:
            shift, total_shift = checked_translation((dx_mm, dy_mm, dz_mm), flight['translation'],
                                                     step_limit_mm=30, cumulative_limit_mm=90)
            angles, total_angles = checked_adjustment((roll_deg, pitch_deg, yaw_deg), flight.get('rotation', (0., 0., 0.)))
        except ValueError:
            return dict(accepted=False, reason_code='adjustment_budget_exhausted')
        scene = getattr(self, '_release_scene', None)
        if scene is None or self.held_plan is None or self.grasp_attachment is None:
            return dict(accepted=False, reason_code='release_geometry_unavailable')
        calibration = np.asarray(self.grasp_to_ee, dtype=float)
        hand = flight['current'] @ np.linalg.inv(calibration)
        pivot = getattr(getattr(self, 'gripper_assets', None), 'jaw_center_offset_m', tcp_offset_from(calibration))
        hand = refined_grasp_pose(hand, angles, tcp_offset_z_m=pivot)
        target = hand @ calibration
        target[:3, 3] += np.asarray(shift)/1000.
        delta = target @ np.linalg.inv(self.grasp_attachment)
        retreat = _pose_transform(flight['plan'].targets[3]).copy()
        retreat[:3, 3] += target[:3, 3] - flight['release'][:3, 3]
        retreat[:3, :3] = target[:3, :3]
        attempt = dict(step_mm=list(shift), cumulative_mm=list(total_shift),
                       step=dict(zip(('roll_deg','pitch_deg','yaw_deg'), angles)),
                       cumulative=dict(zip(('roll_deg','pitch_deg','yaw_deg'), total_angles)))
        width = self.grasp_jaw_width_m if self.grasp_jaw_width_m is not None else getattr(self, 'max_gripper_width_m', .08)
        reason = 'nudged_release_rejected'
        try:
            geometry = check_release_geometry(self.held_plan.target_points, scene, delta,
                self.grasp_attachment, calibration, jaw_width_m=width, release_clearance_m=0.,
                mesh_source=getattr(self, 'gripper_assets', None))
            attempt['path_check'] = {k:geometry[k] for k in ('compatible','reason_code','scope') if k in geometry}
            if not geometry['compatible']:
                accepted = False
            else:
                segments, poses, *_ = _plan(self.connector, (target, retreat), np.empty((0, 3)), self.motion_config, None)
                native = getattr(self.connector.ik, 'check_release_self_collision', None)
                if callable(native):
                    check = native([_trajectory_end(segments[0])], jaw_width_m=width)[0]
                else:
                    from src.tools.place.self_collision import check_release_self_collision
                    check = check_release_self_collision(self.connector.ik._robot_file,
                        [_trajectory_end(segments[0])], jaw_width_m=width)[0]
                attempt['self_collision'] = {'accepted':bool(check['accepted'])}
                accepted = bool(check['accepted'])
            if accepted:
                previews = self._place_previews(flight, target, delta, width)
                if not previews:
                    accepted, reason = False, 'refinement_preview_unavailable'
        except Exception as exc:
            accepted = False
            attempt['error'] = type(exc).__name__
        attempt['accepted'] = accepted
        flight['attempts'].append(attempt)
        self._record('inflight_nudge', dict(stage='release', **attempt, release_ee=target.tolist(),
                                           rotation_frame='gripper_local_jaw_centre', translation_frame='base_xyz'))
        if not accepted:
            return {**attempt, 'reason_code':reason}
        flight.update(current=target, translation=total_shift, rotation=total_angles,
            replacement=dict(segments=(segments[0],), targets=(poses[0],),
                retreat=((segments[1],), (poses[1],)), shift_mm=list(total_shift),
                rotation_deg=list(total_angles)))
        return {**attempt, 'image_refs':previews}


class InteractionOrchestrator:
    def _optional_arguments(self, tool):
        if tool == 'nudge_place' and self.interaction_features.place_rotation:
            return ('roll_deg', 'pitch_deg', 'yaw_deg')
        return super()._optional_arguments(tool)

    def _request_context(self, role, task, step, limit):
        return {**super()._request_context(role, task, step, limit),
                'interaction_features': self.interaction_features.metadata()}

    def _prompt(self, role):
        prompt = super()._prompt(role)
        if self.interaction_features.place_rotation and role in ('prime', 'place', 'refiner'):
            prompt += (' PLACE ROTATION ENABLED: paused release nudge_place accepts the required dx_mm, dy_mm, dz_mm '
                'and optional roll_deg, pitch_deg, yaw_deg (omitted angles are zero). Translation is base XYZ; '
                'rotations are local gripper axes about the jaw centre, 10 degrees per axis per step and 30 cumulative. '
                'This extends the translation-only paused-release description above. Inspect the updated gripper and '
                'orange predicted held-surface previews before continue. They assume rigid retention, not measured new contact. '
                'A rejected nudge leaves the previous pending pose unchanged. The corrected pose must be reached before opening. '
                'Prime delegates the desired orientation; only the paused release Refiner calls nudge_place.')
        if self.interaction_features.held_observation and role in ('prime', 'point', 'refiner'):
            prompt = prompt.replace('with a free/open hand', 'with a free/open hand or a transport-held object')
            prompt += (' HELD OBSERVATION ENABLED: an executed transport grasp with a measured attachment may retain '
                'the closed command during observe-purpose propose_waypoint/shift_waypoint or propose_downward_waypoint. '
                'This replaces the empty/open-only observation restriction for that state, not for contact grasps or empty pushers. '
                'Pointer proposes from current RGB and Prime validates/executes; planned camera-down may rotate the held object. '
                'Use motion=linear only for orientation-preserving translation. Robot and measured held-object paths must pass '
                'validation; no teleport or arbitrary joint path is provided. After motion use fresh RGB to assess retention '
                'and reselect targets/destinations; do not release just to get a new view. Movement of a wrist-mounted camera '
                'does not independently orbit an object rigidly attached to that wrist.')
        return prompt
