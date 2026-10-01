"""Pose previews and edit history at checked pose boundaries."""
from copy import deepcopy
from uuid import uuid4

import numpy as np

from src.tools.pose_editor.cues import grasp_cues, place_cues
from src.tools.pose_editor.editor import PoseEditor
from src.tools.pose_editor.cards import render_pose_card
from src.backend.controller import BoundaryError
from src.backend.interaction import InteractionFeatures


class PoseEditorFeatures(InteractionFeatures):
    """Publish pose-tool capabilities while preserving inherited feature switches."""

    def metadata(self):
        result = super().metadata()
        # This flag describes paused nudge_place rather than adjust_place.
        # Keep the runtime boolean on the object, and disambiguate its public name.
        result['legacy_nudge_place_enabled'] = result.pop('place_rotation')
        result.update(place_pose_rotation=True,
            rotation_step_limit_deg=None, rotation_cumulative_limit_deg=None,
            rotation_limit_scope=['adjust_grasp', 'nudge_grasp', 'adjust_place'],
            rotation_values='finite local roll/pitch/yaw; resulting pose and motion must validate',
            translation_limit_scope=['adjust_grasp', 'nudge_grasp', 'adjust_place'],
            translation_step_limit_mm=None, translation_cumulative_limit_mm=None,
            grasp_candidate_translation_step_limit_mm=None,
            grasp_candidate_translation_cumulative_limit_mm=None,
            clicked_grasp_candidates=True,
            clicked_grasp_scope='explicit measured surface XY, chosen height, one top-down yaw-zero seed; checked like other candidates')
        if self.place_rotation:
            # The internal command is absent from the placement and Refiner tools.
            result['legacy_nudge_place'] = dict(agent_tool_exposed=False,
                rotation_step_limit_deg=10, rotation_cumulative_limit_deg=30)
        return result


class PoseEditorMixin:
    interaction_features = PoseEditorFeatures(place_rotation=True, held_observation=True)
    unrestricted_pose_rotation = True
    unrestricted_pose_translation = True
    clicked_grasp_candidates = True

    def _editor(self, kind):
        if not hasattr(self, '_editors'):
            self._editors, self._panels, self._details = {}, {}, {}
            self._grasp_previews = {}
        if kind not in self._editors:
            # Edit magnitudes are unrestricted; route validation and turn budgets remain.
            self._editors[kind] = PoseEditor(None, translation_frame='base')
        return self._editors[kind]

    def _record_edit(self, kind, origin, ref, arguments):
        editor = self._editor(kind)
        panel = editor.record(origin, ref, arguments, used=0)
        panel.update(translation_frame='base', translation_unit='mm',
                     budget_policy='finite translation and rotation have no magnitude cap; route validation and agent turn budget remain')
        self._panels[ref] = panel

    def _pose_card(self, kind, ref, points, scene, pose, opening, *, destination=None,
                   sliders=None, observation_id=None):
        editor = self._editor(kind)
        points, scene, pose = np.asarray(points), np.asarray(scene), np.asarray(pose)
        if not len(points):
            raise ValueError('RobotUse pose preview needs measured target points')
        parts, _ = self.gripper_assets.load_gripper_mesh(opening)
        parts = {key: np.asarray(triangles) @ pose[:3, :3].T + pose[:3, 3]
                 for key, triangles in parts.items() if len(triangles)}
        if not parts:
            raise ValueError('RobotUse pose preview needs native gripper triangles')
        if destination is None:
            cues = grasp_cues(points, pose, jaw_offset=self.jaw_offset_m, opening=opening,
                              translation_frame='base')
            fit = points
            support_z = float(np.percentile(points[:, 2], 2))
            support_label = 'observed target lower surface'
        else:
            fit = np.asarray(destination)
            cues = place_cues(points, fit, pose, jaw_offset=self.jaw_offset_m)
            shift = cues['center_shift_m']
            cues['lines'][0] = f'BASE footprint offset: dx_mm {shift[0]*1000:+.0f}   dy_mm {shift[1]*1000:+.0f}'
            support_z = float(np.percentile(fit[:, 2], 98))
            support_label = 'measured destination top (98th pct)'
        cues.update(translation_frame='base', translation_unit='mm', observation_id=observation_id,
                    support_z_m=support_z, support_label=support_label,
                    rotation_frame='gripper_local_axes_about_contact_center')
        if kind == 'paused':
            cues['scope'] = ('Pre-approach target cloud (not tracked); orange cross is NOT a grasp target; '
                             'span is NOT contact thickness')
        elif destination is not None:
            cues['scope'] = 'recorded measured payload, rigid/no-slip prediction; no support, collision or retention verdict'
        center = (fit.min(0) + fit.max(0)) / 2
        span = max(.32, float(np.linalg.norm(np.ptp(fit, axis=0))) + .24)
        path = render_pose_card(points, scene, parts, pose, center, span,
            self.output_dir / 'pose_editor' / (ref + '.png'), title=f'RobotUse {kind.upper()} / {ref[:36]}',
            sliders=editor.cumulative(ref) if sliders is None else sliders,
            fit_points=fit, edit_cues=cues, support_z=support_z, support_label=support_label)
        image_ref = self.images.add(path)
        self._details[ref] = dict(geometry_cues=cues, image_refs=[image_ref])
        return image_ref

    def pose_editor_feedback(self, candidate_ref):
        self._editor('grasp')
        ref = candidate_ref
        if ref in getattr(self, '_explicit_place_proposals', {}):
            ref = self._explicit_place_proposals[ref]['waypoint_ref']
        details = deepcopy(self._details.get(ref, {}))
        if candidate_ref in self._panels:
            details['pose_editor'] = deepcopy(self._panels[candidate_ref])
        if candidate_ref in self._details and candidate_ref in getattr(self, 'candidates', {}):
            details['candidate_geometry'] = dict(candidate_ref=candidate_ref,
                **self._explicit_candidate_metadata(candidate_ref),
                executable=candidate_ref not in getattr(self, 'diagnostic_candidates', {}))
        return details

    def _explicit_pick_previews(self, source, prediction, ref, opening):
        paths = super()._explicit_pick_previews(source, prediction, ref, opening)
        self._editor('grasp')
        self._grasp_previews[ref] = (source, prediction, opening)
        pending = getattr(self, '_pending_edit', None)
        if pending:
            self._record_edit('grasp', pending[0], ref, pending[1])
        # The inherited candidate cache owns only the camera projections.
        return paths

    def _ensure_grasp_card(self, candidate_ref):
        self._editor('grasp')
        if candidate_ref not in self._details:
            source, prediction, opening = self._grasp_previews[candidate_ref]
            self._pose_card('grasp', candidate_ref, source.object_points, source.scene_points,
                            prediction.pose, opening, observation_id=source.observation_id)

    def _with_pose_feedback(self, result):
        details = self.pose_editor_feedback(result.get('candidate_ref'))
        refs = self.images.unique_refs_by_path([
            *details.pop('image_refs', []), *result.get('image_refs', [])])
        return {**result, **details, 'image_refs': refs}

    def refine_candidate(self, candidate_ref, **arguments):
        self._pending_edit = (candidate_ref, arguments)
        try:
            result = super().refine_candidate(candidate_ref, **arguments)
        finally:
            self._pending_edit = None
        if result.get('candidate_ref'):
            axes = ('roll_deg', 'pitch_deg', 'yaw_deg')
            result['adjustment_deg'] = {key: float(arguments.get(key, 0.)) for key in axes}
            result['cumulative_deg'] = dict(zip(axes, self.candidate_adjustments[result['candidate_ref']]))
            self._ensure_grasp_card(result['candidate_ref'])
        return self._with_pose_feedback(result)

    def inspect_candidate(self, candidate_ref):
        result = super().inspect_candidate(candidate_ref)
        self._ensure_grasp_card(candidate_ref)
        return self._with_pose_feedback(result)

    def _ensure_place_card(self, candidate_ref):
        waypoint_ref = self._explicit_place_proposal(candidate_ref)['waypoint_ref']
        self._editor('place')
        if waypoint_ref in self._details:
            return
        proposal = self._view(waypoint_ref)
        metadata = proposal['instruction']
        hand = proposal['pose'] @ np.linalg.inv(self.grasp_to_ee)
        delta = proposal['pose'] @ np.linalg.inv(self.grasp_attachment)
        payload = self.held_plan.target_points @ delta[:3, :3].T + delta[:3, 3]
        destination = self._explicit_destination(metadata['destination_ref'])
        sliders = dict(zip(('dx_mm', 'dy_mm', 'dz_mm', 'roll_deg', 'pitch_deg', 'yaw_deg'),
                           [*metadata['translation_mm'], *metadata['rotation_deg']]))
        self._pose_card('place', waypoint_ref, payload, proposal['scene'], hand,
            self.grasp_jaw_width_m, destination=destination.points, sliders=sliders,
            observation_id=proposal['observation_id'])

    def explicit_inspect_place(self, candidate_ref):
        result = super().explicit_inspect_place(candidate_ref)
        self._ensure_place_card(candidate_ref)
        return self._with_pose_feedback(result)

    def explicit_adjust_place(self, candidate_ref, **arguments):
        result = super().explicit_adjust_place(candidate_ref, **arguments)
        if result.get('candidate_ref'):
            self._record_edit('place', candidate_ref, result['candidate_ref'], arguments)
            self._ensure_place_card(result['candidate_ref'])
        return self._with_pose_feedback(result)

    def _pregrasp_pause(self, plan):
        self._editor('paused').reset()
        self._paused_ref = 'paused_' + uuid4().hex
        self.inflight_feedback = {}
        replacement = super()._pregrasp_pause(plan)
        self._inflight['resume_authorized'] = True
        return replacement

    def execute_grasp(self, candidate_ref, validation_ref):
        from src.tools.grasp.backend import hand_to_contact
        self.last_execution_pose = None
        if candidate_ref in getattr(self, 'diagnostic_candidates', {}):
            raise ValueError('diagnostic candidate is not executable; refine and validate a new candidate')
        validation = self.validations.get(validation_ref)
        original = validation[3] if validation is not None else None
        before_epoch = self.epoch
        result = super().execute_grasp(candidate_ref, validation_ref)
        flight = getattr(self, '_inflight', None) or {}
        replacement = flight.get('replacement')
        resumed = flight.get('resume_authorized', False)
        used = replacement if replacement is not None and resumed else original
        def contact_pose(plan):
            return hand_to_contact(plan.grasp_transform, self.jaw_offset_m).tolist() if plan is not None else None
        details = dict(reference_only=True, candidate_ref=candidate_ref, execution_ref=result.get('execution_ref'),
            execution_started=self.epoch != before_epoch,
            selected_target_pose_base=contact_pose(original),
            execution_target_pose_base=contact_pose(used),
            pending_edited_target_pose_base=contact_pose(replacement),
            paused_edit_applied=replacement is not None and resumed,
            grasp_plan_marked_executed=bool(getattr(used, 'grasp_executed', False)),
            frame='connector_base', position_reference='jaw_contact_center', units='metres',
            scope='commanded grasp target, not measured arrival, object pose or success')
        self.last_execution_pose = deepcopy(details)
        self._record('grasp_execution_pose', details)
        return {**result, 'execution_pose': details}

    def _inflight_preview(self, observation, hand_pose, open_width_m, name):
        refs = super()._inflight_preview(observation, hand_pose, open_width_m, name)
        if not refs:
            return refs
        from src.tools.place.execution import observed_scene
        flight = self._inflight
        axes = ('dx_mm', 'dy_mm', 'dz_mm', 'roll_deg', 'pitch_deg', 'yaw_deg')
        sliders = dict(zip(axes, [*flight['translation'], *flight['adjustment']]))
        for key, value in getattr(self, '_pending_nudge', {}).items():
            sliders[key] += value
        card = self._pose_card('paused', name, flight['original'].target_points,
            observed_scene(self.point_adapter.frames[observation['observation_id']]),
            hand_pose, open_width_m, sliders=sliders, observation_id=observation['observation_id'])
        self.inflight_feedback = deepcopy(self._details[name])
        return self.images.unique_refs_by_path([card, *refs])

    def nudge_inflight_grasp(self, **arguments):
        self._pending_nudge = arguments
        try:
            result = super().nudge_inflight_grasp(**arguments)
        finally:
            self._pending_nudge = {}
        if result.get('accepted'):
            ref = 'paused_' + uuid4().hex
            self._record_edit('paused', self._paused_ref, ref, arguments)
            self._paused_ref = ref
            self.inflight_feedback['pose_editor'] = deepcopy(self._panels[ref])
        return result
