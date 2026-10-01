"""Observed centers and Contact-GraspNet, with agent-owned coordinates.

Observation, attachment, inspection and execution use shared backend helpers.
This backend calls no additional grasp or placement predictor.
Agent-visible positions locate the contact center between the jaws in base metres.
"""
from copy import deepcopy
from contextlib import contextmanager
from types import MappingProxyType
from uuid import uuid4
import numpy as np

from src.core.action_feedback import ActionPreconditionError
from src.tools.grasp.prediction import GraspPrediction
from src.tools.motion.planning import MotionPlanningError, _pose_transform, _transform
from src.backend.interaction import InteractionBackend, InteractionFeatures
from src.tools.motion.world_planning import WorldPlanIntentBackend
from src.tools.grasp.geometry import observed_cloud_statistics, xy_candidates, resolve_height, top_down_contact_pose


_MANIPULATION_REASONS = MappingProxyType({
    'requires_explicit_grasp_geometry_and_transit':
        'Grasp generation requires an explicit direction, geometric height and yaw, '
        'and independent pre-pick and post-pick transit heights.',
    'requires_explicit_place_coordinates':
        'Placement requires an observed XY candidate and an agent-selected contact height.',
    'placement_predictor_unavailable':
        'Placement uses agent-selected observed coordinates; no placement predictor is available.',
    'geometric_grasp_requires_yaw_only':
        'Median and mean grasps retain a downward approach. Refine base XYZ and yaw only; '
        'roll and pitch must remain zero.',
    'requires_checked_agent_selected_pose':
        'The agent-selected pose must pass the normal planning and geometry checks.',
    'place_movement_and_release_are_separate_actions':
        'Move to the selected placement pose first, then make a separate agent release decision.',
})


class ManipulationPreconditionError(ActionPreconditionError):
    """Fixed grasp feedback compatible with shared ActionPreconditionError catches."""
    def __init__(self, reason_code):
        if reason_code not in _MANIPULATION_REASONS:
            raise ValueError('Unknown action precondition reason code')
        self._reason_code = reason_code
        ValueError.__init__(self, _MANIPULATION_REASONS[reason_code])

    def public_feedback(self):
        return dict(error='action_precondition_failed', executed=False, state_changed=False,
                    reason_code=self.reason_code, reason=_MANIPULATION_REASONS[self.reason_code])


def contact_to_hand(pose, jaw_offset_m):
    result = _transform(pose).copy()
    if not np.isfinite(jaw_offset_m) or jaw_offset_m <= 0:
        raise ValueError('positive calibrated jaw offset required')
    result[:3, 3] -= result[:3, 2] * jaw_offset_m
    return result


def hand_to_contact(pose, jaw_offset_m):
    result = _transform(pose).copy()
    if not np.isfinite(jaw_offset_m) or jaw_offset_m <= 0:
        raise ValueError('positive calibrated jaw offset required')
    result[:3, 3] += result[:3, 2] * jaw_offset_m
    return result


def height_from_geometry(spec, statistics, clicked_xyz=None, *, current_tcp_z_m=None, grasp_z_m=None):
    """Resolve height references once, accepting both supported input schemas."""
    if not isinstance(spec, dict):
        raise ValueError('height requires reference and value_m')
    if set(spec) == {'mode', 'value_m', 'reference'}:
        # Normalize the mode/reference schema to the tools' reference/value schema.
        mode, ref = spec['mode'], spec['reference']
        if mode == 'absolute' and ref == 'none':
            spec = dict(reference='absolute', value_m=spec['value_m'])
        elif mode == 'surface_relative' and ref in ('clicked', 'median', 'mean', 'min', 'max'):
            aliases = dict(clicked='clicked_point', median='segment_median', mean='segment_mean',
                           min='observed_min', max='observed_max')
            spec = dict(reference=aliases[ref], value_m=spec['value_m'])
        else:
            raise ValueError('invalid height mode/reference')
    if set(spec) != {'value_m', 'reference'}:
        raise ValueError('height requires exactly reference and value_m')
    reference = spec['reference']
    if reference == 'absolute':
        return resolve_height(spec['value_m'], mode='absolute')
    bases = dict(current_tcp=current_tcp_z_m, grasp=grasp_z_m,
        clicked_point=None if clicked_xyz is None else clicked_xyz[2],
        segment_median=statistics['median_xyz_m'][2], segment_mean=statistics['mean_xyz_m'][2],
        observed_min=statistics['min_xyz_m'][2], observed_max=statistics['max_xyz_m'][2])
    if reference not in bases or bases[reference] is None:
        raise ValueError('requested height reference is unavailable: ' + str(reference))
    return resolve_height(spec['value_m'], mode='surface_relative', reference_z_m=bases[reference])


def robot_geometry_context(backend):
    """Measured robot context shared by grasp and place numeric decisions."""
    from src.tools.grasp.arguments import HEIGHT_REFERENCES
    hand = _pose_transform(backend.connector.get_ee_pose()) @ np.linalg.inv(backend.grasp_to_ee)
    contact = hand_to_contact(hand, backend.jaw_offset_m)
    held = getattr(backend, 'held_plan', None)
    contract = getattr(held, 'grasp_contract', None)
    grasp_z = None
    if held is not None:
        if contract and contract.get('contact_center_xyz_m') is not None:
            grasp_z = float(contract['contact_center_xyz_m'][2])
        elif getattr(held, 'grasp_transform', None) is not None:
            grasp_z = float(hand_to_contact(held.grasp_transform, backend.jaw_offset_m)[2, 3])
    return dict(current_tcp_pose_base=contact.tolist(), contact_center_xyz_m=contact[:3, 3].tolist(),
        current_tcp_z_m=float(contact[2, 3]), last_grasp_target_tcp_z_m=grasp_z,
        grasp_axis_preapproach_distance_m=backend.motion_config.approach_m,
        gripper_max_opening_m=backend.max_gripper_width_m, height_references=list(HEIGHT_REFERENCES),
        transit_constraint='approach transit TCP Z must cover grasp and axis-pregrasp Z; lift TCP Z must be at or above grasp Z',
        height_reference_scope='current_tcp is measured at planning; grasp is candidate Z for pick transit, otherwise held grasp Z')


class GraspBackend(InteractionBackend, WorldPlanIntentBackend):
    interaction_features = InteractionFeatures(place_rotation=True, held_observation=True)
    attachment_release_equation = (
        'agent_contact_center_pose @ contact_to_hand_offset @ grasp_to_ee; '
        'opening requires a separate explicit agent release action')

    def __init__(self, *, cgn_client, **kwargs):
        # ObservedPlacementBackend contains shared observed attachment state; its
        # predictor is deliberately absent and all inference entrypoints reject.
        kwargs['anyplace'] = None
        kwargs['moveit_grasps'] = None
        kwargs['grasp_policy'] = 'agent-choice'
        kwargs['grasp_motion_policy'] = 'source-order'
        super().__init__(**kwargs)
        self.cgn_client = cgn_client
        self.grasp_contracts = {}
        self.clicked_points = {}
        self._active_grasp_contract = None
        self.last_place = None

    def _execution_failure(self, operation, execution_ref, exc):
        from src.core.errors import structured_error
        result = super()._execution_failure(operation, execution_ref, exc)
        result['error_details'] = structured_error(exc, phase='motion_execution', tool=operation)
        return result

    def _public_grasp_feedback(self, public, raw):
        # This reports the existing sensor finding only; no native task truth or
        # full private execution diagnostics cross the agent boundary.
        if raw.get('grasp_evidence') == 'empty_closed_gripper' and raw.get('held_state') == 'not_held':
            from src.core.errors import add_result_error
            public.update(grasp_evidence='empty_closed_gripper', held_state='not_held')
            add_result_error(public, phase='grasp_execution', tool='execute_grasp')
        return public

    @property
    def jaw_offset_m(self):
        return self.gripper_assets.jaw_center_offset_m

    def select_region(self, observation_id, view_id, u, v, **kwargs):
        result = super().select_region(observation_id, view_id, u, v, **kwargs)
        from src.tools.place.observed_placement import measured_anchor
        frame = next(f for f in self.point_adapter.frames[observation_id] if f.view_id == view_id)
        self.clicked_points[result['point_ref']] = measured_anchor(frame, u, v)
        result['geometry'] = self.explicit_geometry(result['point_ref'])
        return result

    def explicit_geometry(self, point_ref):
        geometry = self._point(point_ref)
        click = self.clicked_points.get(point_ref)
        stats = observed_cloud_statistics(geometry.object_points)
        result = dict(point_ref=point_ref, observation_id=geometry.observation_id,
            statistics=stats, xy_candidates=xy_candidates(geometry.object_points, clicked_xyz_m=click),
            **robot_geometry_context(self),
            frame='connector_base', position_reference='jaw_contact_center', units='metres',
            rotation_convention='approach=local +Z, jaw closing=local +X; yaw about base +Z',
            statistic_source='fused_observed_SAM_surface',
            per_view_statistics=[dict(view_id=item.view_id, statistics=observed_cloud_statistics(item.object_points))
                for item in getattr(geometry, 'per_view', ())],
            geometry_limitations='SAM-selected visible RGB-D surfaces; not full object or center of mass')
        self._record('observed_geometry', result)
        return result

    def grasp_budget(self):
        return dict(limit=self.task_grasp_budget, reserved=self.grasp_reserved,
            remaining=self.task_grasp_budget-self.grasp_reserved,
            published_candidates=self.grasp_published,
            published_diagnostics=self.grasp_diagnostics_published,
            execution_attempts=len(self.grasp_execution_evidence),
            policy='candidate poses submitted to planning; CGN raw proposal counts recorded separately')

    @contextmanager
    def _contract(self, value):
        previous = self._active_grasp_contract
        self._active_grasp_contract = value
        try:
            yield
        finally:
            self._active_grasp_contract = previous

    def grasp_candidates(self, *args, **kwargs):
        raise ManipulationPreconditionError('requires_explicit_grasp_geometry_and_transit')

    def place_candidates(self, *args, **kwargs):
        raise ManipulationPreconditionError('requires_explicit_place_coordinates')

    def _placement_pool(self, *args, **kwargs):
        raise ManipulationPreconditionError('placement_predictor_unavailable')

    def _raw_cgn(self, geometry):
        """Use the same observed full/segment point clouds in one explicit frame."""
        from src.tools.grasp.cgn_client import raw_to_contact_center, ROBOT_BASE, CGNResponseError
        # The scene cloud excludes the selected target for collision checks.
        # CGN instead requires the full observed scene, including that target.
        full = np.concatenate((geometry.scene_points, geometry.object_points), axis=0)
        raw = self.cgn_client.plan_point_clouds(
            full, geometry.object_points, input_frame=ROBOT_BASE)
        if raw.frame != ROBOT_BASE:
            raise CGNResponseError('CGN point-cloud result changed the explicit input frame')
        return raw_to_contact_center(raw)

    def explicit_grasp_candidates(self, point_ref, direction, tolerance_deg, azimuth_deg,
                            polar_deg, geometric_height, transit):
        from src.tools.grasp.direction import filter_cgn_directions
        from src.tools.grasp.cgn_client import CGNResponseError
        if self.held_plan is not None or self.closed_push:
            raise ActionPreconditionError('grasp_requires_release')
        geometry = self._point(point_ref)
        if point_ref != self.latest_fused_ref or geometry.role != 'pick':
            raise ValueError('current selected pick segment required')
        self.point_adapter._check_current(geometry.observation_id)
        public_geometry = self.explicit_geometry(point_ref)
        statistics = public_geometry['statistics']
        click = self.clicked_points.get(point_ref)
        if not isinstance(transit, dict) or set(transit) != {'pre', 'post'}:
            raise ValueError('transit requires independent pre and post height decisions')
        current_z = public_geometry['current_tcp_z_m']
        clicked = direction == 'clicked' and getattr(self, 'clicked_grasp_candidates', False)
        geometric = direction in ('mean', 'median') or clicked
        if geometric:
            if any(value is not None for value in (tolerance_deg, azimuth_deg, polar_deg)):
                raise ValueError('geometric candidates require null direction angles and tolerance')
            if clicked:
                from scipy.spatial import cKDTree
                if click is None or np.asarray(click).shape != (3,) or not np.isfinite(click).all():
                    raise ValueError('clicked grasp requires a valid registered depth point')
                # Reject background clicks even when SAM returns a nearby object.
                if cKDTree(geometry.object_points).query(click)[0] > .01:
                    raise ValueError('clicked grasp point must lie on the measured selected surface (10 mm tolerance)')
            z = height_from_geometry(geometric_height, statistics, click,
                current_tcp_z_m=current_z, grasp_z_m=public_geometry['last_grasp_target_tcp_z_m'])
        else:
            if geometric_height is not None:
                raise ValueError('CGN requests require null geometric_height')
            filter_cgn_directions(np.empty((0, 4, 4)), np.empty(0), direction=direction,
                tolerance_deg=tolerance_deg, azimuth_deg=azimuth_deg, polar_deg=polar_deg)
        remaining = self.task_grasp_budget - self.grasp_reserved
        generator = direction if geometric else 'contact_graspnet'
        required_slots = 1 if clicked else 4 if geometric else 1
        if remaining < required_slots:
            result = dict(candidates=[], diagnostic_candidates=[], generator=generator,
                geometry=public_geometry, source_reports=[], task_budget=self.grasp_budget(),
                candidate_limit=1 if clicked else 4, reason_code='grasp_budget_exhausted',
                reason='Mean/median require four planning slots; clicked and CGN require at least one')
            self._record('explicit_grasp_candidates', result)
            return result
        self._capture_grasp_scene(geometry)
        source = getattr(geometry, 'per_view', (geometry,))[0]
        accepted, diagnostics, source_reports = [], [], []
        proposals = []
        cgn_report = None
        if geometric:
            xy = click[:2] if clicked else statistics[direction+'_xyz_m'][:2]
            proposals = [(top_down_contact_pose(xy, z, yaw),
                          0., direction, True, index)
                         for index, yaw in enumerate((0.,) if clicked else (-45., 0., 45., 90.))]
        else:
            try:
                cgn = self._raw_cgn(geometry)
                selection = filter_cgn_directions(cgn.poses, cgn.scores, direction=direction,
                    tolerance_deg=tolerance_deg, azimuth_deg=azimuth_deg, polar_deg=polar_deg)
                proposals = [(cgn.poses[i], float(cgn.scores[i]), 'contact_graspnet', False, int(i))
                             for i in selection.indices]
                cgn_report = dict(generator='contact_graspnet', raw_count=len(cgn.poses),
                    direction_survivors=len(selection.indices), direction_filter=selection.metadata)
                source_reports.append(cgn_report)
            except (TimeoutError, ConnectionError, RuntimeError, CGNResponseError) as exc:
                from src.core.errors import structured_error
                details = structured_error(exc, phase='cgn_generation', tool='explicit_grasp_candidates')
                source_reports.append(dict(generator='contact_graspnet', error_type=type(exc).__name__,
                    reason_code=details['code'], reason=details['message'], error_details=details))
                self._record('cgn_error', dict(error_type=type(exc).__name__, error_details=details))
        cgn_checked = cgn_accepted = 0
        deferred = []
        for contact, score, kind, top_down, source_index in proposals:
            if self.grasp_reserved >= self.task_grasp_budget or (not top_down and cgn_accepted >= 4):
                break
            self.grasp_reserved += 1
            ref = 'g_'+uuid4().hex
            heights = dict(pre_pick_z_m=height_from_geometry(transit['pre'], statistics, click,
                current_tcp_z_m=current_z, grasp_z_m=float(contact[2, 3])),
                post_pick_z_m=height_from_geometry(transit['post'], statistics, click,
                current_tcp_z_m=current_z, grasp_z_m=float(contact[2, 3])))
            contract = dict(**heights, source=kind, top_down_only=top_down,
                height_request=deepcopy(geometric_height) if top_down else None,
                transit_request=deepcopy(transit), direction=direction,
                tolerance_deg=tolerance_deg, azimuth_deg=azimuth_deg, polar_deg=polar_deg,
                contact_center_xyz_m=contact[:3, 3].tolist(),
                source_candidate_index=source_index, source_score=None if top_down else score)
            self.grasp_contracts[ref] = contract
            prediction = GraspPrediction(contact_to_hand(contact, self.jaw_offset_m), score,
                                         None, 'robotiq_2f_85')
            with self._contract(contract):
                entry = self._publish_grasp_pose(ref, source, prediction, point_ref,
                                             defer_diagnostic=not top_down)
            if not top_down:
                cgn_checked += 1
                cgn_accepted += bool(entry.get('executable'))
            if entry.get('executable'):
                accepted.append(entry)
            elif top_down:
                diagnostics.append(entry)
            else:
                deferred.append((ref, source, prediction, point_ref, entry['_evidence'], entry['_open_width_m']))
        # Rejected CGN poses do not hide later feasible ones. Only now allocate
        # their previews to the unfilled four CGN slots; total publication <= 4.
        for ref, source, prediction, point_ref, evidence, opening in deferred[:4-cgn_accepted]:
            diagnostics.append(self._publish_grasp_diagnostic(
                ref, source, prediction, point_ref, evidence, opening))
        if cgn_report is not None:
            stop = ('four_feasible_candidates' if cgn_accepted >= 4 else
                    'direction_candidates_exhausted' if cgn_checked == cgn_report['direction_survivors'] else
                    'task_planning_budget_exhausted')
            cgn_report.update(planning_checked_count=cgn_checked, planning_accepted_count=cgn_accepted,
                planning_rejected_count=cgn_checked-cgn_accepted, search_stop_reason=stop)
        self.grasp_published += len(accepted)
        self.grasp_diagnostics_published += len(diagnostics)
        result = dict(candidates=accepted, diagnostic_candidates=diagnostics,
            generator=generator, geometry=public_geometry,
            source_reports=source_reports, task_budget=self.grasp_budget(), candidate_limit=1 if clicked else 4,
            reason_code='candidate_available' if accepted else 'no_feasible_candidate')
        if not accepted:
            from src.core.errors import error_detail
            failure = next((report['error_details'] for report in source_reports if 'error_details' in report), None)
            if failure is not None:
                # No plan was attempted: a failed service is not geometric infeasibility.
                result.update(reason_code=failure['code'], error_details=deepcopy(failure))
            elif cgn_report is not None and not cgn_report['raw_count']:
                result['reason_code'] = 'no_cgn_proposals'
                result['error_details'] = error_detail('generation', 'no_cgn_proposals', 'cgn_generation',
                    'CGN completed successfully and returned zero proposals for this observed input.',
                    'Reassess the observed selection or explicitly choose another supported generator.',
                    tool='explicit_grasp_candidates')
            elif cgn_report is not None and not cgn_report['direction_survivors']:
                result['reason_code'] = 'direction_filter_empty'
                result['error_details'] = error_detail('generation', 'direction_filter_empty', 'direction_filter',
                    'CGN returned proposals but none matched the requested direction filter.',
                    'Reassess the requested approach and tolerance; this is not a CGN service failure.',
                    tool='explicit_grasp_candidates')
            else:
                result['error_details'] = error_detail('planning', 'no_feasible_candidate', 'candidate_planning',
                    'The candidates checked for this request did not pass the configured planning checks.',
                    'Inspect candidate diagnostics, then change the pose or route and validate again.',
                    tool='explicit_grasp_candidates')
        self._record('explicit_grasp_candidates', result)
        return result

    def _publish_grasp_pose(self, ref, source, prediction, point_ref, *, defer_diagnostic=False):
        checker, scene = self._candidate_path_scene(point_ref)
        options = self._grasp_options(prediction.pose, self._point(point_ref), prediction, ref)
        state = self._robot_state()
        try:
            plan, evidence = self._plan_grasp_candidate(prediction.pose,
                self._point(point_ref).object_points, self._point(point_ref).scene_points,
                checker, scene, options)
        except MotionPlanningError as exc:
            plan = None
            evidence = dict(getattr(exc, 'planning_feedback', {}))
            evidence.update(accepted=False, kind='planning', error=str(exc))
        contract = self.grasp_contracts[ref]
        self._record('candidate_path_check', dict(candidate_ref=ref, evidence=evidence,
            source=contract['source'], source_candidate_index=contract['source_candidate_index'],
            source_score=contract['source_score'],
            contact_center_pose_base=hand_to_contact(prediction.pose, self.jaw_offset_m).tolist()))
        if not evidence['accepted']:
            if defer_diagnostic:
                return dict(candidate_ref=ref, executable=False, _evidence=evidence,
                            _open_width_m=options['open_width_m'])
            return self._publish_grasp_diagnostic(ref, source, prediction, point_ref, evidence,
                                              options['open_width_m'])
        paths = self._explicit_pick_previews(source, prediction, ref, plan.open_width_m)
        self._candidate_inspection_cache[ref] = paths
        self.candidate_routes[ref] = (state, plan)
        self.candidates[ref] = (self.epoch, source, prediction)
        self.candidate_point_refs[ref] = point_ref
        self.candidate_sources[ref] = source.view_id
        self.candidate_open_widths[ref] = plan.open_width_m
        return dict(candidate_ref=ref, source_view=source.view_id, executable=True,
                    image_refs=[self.images.add(path) for path in paths], **self._explicit_candidate_metadata(ref))

    def _explicit_pick_previews(self, source, prediction, ref, opening):
        from src.tools.pose_editor.refinement import preview_refined_pose
        from src.tools.place.execution import _CyanGripperMesh
        paths = preview_refined_pose(self.point_adapter.frames[source.observation_id], prediction.pose,
            self.output_dir / (ref+'_cyan_projection'), tcp_offset_z_m=self.jaw_offset_m,
            expected_open_width_m=opening, mesh_source=_CyanGripperMesh(self.gripper_assets))
        if not paths:
            raise ValueError('pick gripper mesh projection unavailable')
        return paths

    def _inflight_preview(self, observation, hand_pose, open_width_m, name):
        from src.tools.pose_editor.refinement import preview_refined_pose
        from src.tools.place.execution import _CyanGripperMesh
        try:
            paths = preview_refined_pose(self.point_adapter.frames[observation['observation_id']],
                hand_pose, self.output_dir / name, tcp_offset_z_m=self.jaw_offset_m,
                expected_open_width_m=open_width_m, mesh_source=_CyanGripperMesh(self.gripper_assets))
            return [self.images.add(path) for path in paths]
        except Exception as exc:
            self._record('inflight_preview_error', dict(name=name, error=repr(exc)))
            return []

    def _publish_grasp_diagnostic(self, ref, source, prediction, point_ref, evidence, opening):
        entry = self._publish_diagnostic(ref, source, prediction, point_ref, evidence,
                                        open_width_m=opening,
                                        paths=self._explicit_pick_previews(source, prediction, ref, opening))
        if entry is None:
            raise ValueError('candidate diagnostic preview unavailable')
        entry.update(self._explicit_candidate_metadata(ref))
        self.diagnostic_candidates[ref] = deepcopy(entry)
        return entry

    def _explicit_candidate_metadata(self, ref):
        contract = self.grasp_contracts[ref]
        prediction = self.candidates[ref][2]
        contact = hand_to_contact(prediction.pose, self.jaw_offset_m)
        return dict(source=contract['source'], source_candidate_index=contract['source_candidate_index'],
            score=contract['source_score'], contact_center_xyz_m=contact[:3, 3].tolist(),
            contact_center_pose_base=contact.tolist(), approach_direction_base=contact[:3, 2].tolist(),
            top_down_only=contract['top_down_only'],
            **({'source_top_down_only': contract['source_top_down_only']}
               if 'source_top_down_only' in contract else {}),
            yaw_deg=(float(np.degrees(np.arctan2(contact[1, 0], contact[0, 0])))
                     if contract['top_down_only'] else None), frame='connector_base',
            position_reference='jaw_contact_center', units='metres',
            transit={key: contract[key] for key in ('pre_pick_z_m', 'post_pick_z_m')})

    def inspect_candidate(self, candidate_ref):
        result = super().inspect_candidate(candidate_ref)
        return {**result, **self._explicit_candidate_metadata(candidate_ref)}

    def preview_candidate(self, candidate_ref, azimuth_deg, elevation_deg, zoom):
        result = super().preview_candidate(candidate_ref, azimuth_deg, elevation_deg, zoom)
        return {**result, **self._explicit_candidate_metadata(candidate_ref)}

    def _explicit_chain_planner(self, connector, targets, obstacles, config, validator, *, target_labels):
        """Observed-world planner with the exact agent-requested waypoints.

        Only transit segments may detour. Descent, approach and lift stay directed.
        Failure never substitutes a new height or automatic route.
        """
        from src.tools.motion.planning import transform_to_pose, _collision_disabled_planner
        from src.tools.motion.robot_state import _current_robot_state, _trajectory_end
        from src.core.planning_feedback import segment_planning_feedback
        pose, joints = _current_robot_state(connector)
        seed = tuple(joints)
        current = pose
        world, summary = self._obstacle_world(obstacles, joints=joints)
        linear = _collision_disabled_planner(connector).ik
        segments = []
        for index, (target, label) in enumerate(zip(targets, target_labels)):
            goal = transform_to_pose(target)
            world_segment = label in ('high_transit', 'high_pregrasp_align', 'transit', 'pregrasp_align')
            try:
                segment = (self._plan_world_segment(target, seed, world) if world_segment else
                           linear.plan_linear(current, goal, seed_joints=seed))
            except Exception as exc:
                reason = (exc.planning_feedback.get('planner_reason_code', 'planner_failed')
                          if isinstance(exc, MotionPlanningError) else 'planner_exception')
                raise MotionPlanningError('requested segment planning failed', planning_feedback=
                    segment_planning_feedback(index, label, _pose_transform(current), target, reason)) from exc
            if not segment:
                reason = (getattr(linear, 'planning_failure_code', None) if not world_segment else None)
                raise MotionPlanningError('requested transit segment has no route',
                    planning_feedback=segment_planning_feedback(index, label, _pose_transform(current),
                        target, reason or 'no_usable_route'))
            segments.append(segment)
            try:
                seed = _trajectory_end(segment)
            except Exception as exc:
                raise MotionPlanningError('segment returned an invalid trajectory', planning_feedback=
                    segment_planning_feedback(index, label, _pose_transform(current),
                        target, 'invalid_trajectory')) from exc
            current = goal
        self._record('world_transit', dict(labels=list(target_labels), world=summary))
        return tuple(segments), tuple(transform_to_pose(t) for t in targets), tuple(joints), None, False

    def _plan_grasp_candidate(self, pose, target_points, obstacle_points, checker, scene, options):
        from src.tools.grasp.execution import plan_explicit_grasp
        contract = self._active_grasp_contract
        if contract is None:
            raise ValueError('an explicit transit contract is required')
        plan = plan_explicit_grasp(self.connector, grasp_transform=pose, grasp_to_ee=self.grasp_to_ee,
            target_points=target_points, obstacle_points=scene, config=self.motion_config,
            pre_pick_z_m=contract['pre_pick_z_m'], post_pick_z_m=contract['post_pick_z_m'],
            jaw_offset_m=self.jaw_offset_m, max_width_m=self.max_gripper_width_m,
            chain_planner=self._explicit_chain_planner if self.observed_transit_planner else None, **options)
        plan.grasp_contract = deepcopy(contract)
        return plan, self._explicit_check_grasp_path(checker, plan, scene)

    def _explicit_check_grasp_path(self, checker, plan, scene):
        stop = 'lift' if 'lift' in plan.target_labels else 'grasp'
        enabled = getattr(self, 'grasp_path_collision_checks', True)
        if enabled:
            evidence = checker.check(plan, scene, stop_label=stop, jaw_width_m=plan.open_width_m)
        else:
            evidence = dict(accepted=True, checks=[], collision_check_status='disabled_by_configuration',
                            scope='pick_approach_grasp_lift', requested_stop_label=stop)
        evidence = dict(evidence, grasp_path_collision_checks=enabled)
        self._record('grasp_path_validation', evidence)
        return evidence

    def _refinement_contract(self, contract, angles):
        updated = deepcopy(contract)
        if getattr(self, 'unrestricted_pose_rotation', False) and any(angles[:2]):
            # Keep the generator provenance, but the checked edited proposal is
            # no longer constrained to the generator's initial downward approach.
            updated.setdefault('source_top_down_only', contract['top_down_only'])
            updated['top_down_only'] = False
        return updated

    def _checked_refinement(self, pose, contract, translation, angles):
        from src.tools.pose_editor.refinement import refined_grasp_pose
        if contract['top_down_only'] and (angles[0] != 0 or angles[1] != 0):
            raise ManipulationPreconditionError('geometric_grasp_requires_yaw_only')
        result = refined_grasp_pose(pose, angles, tcp_offset_z_m=self.jaw_offset_m)
        result[:3, 3] += np.asarray(translation, dtype=float)/1000.
        if contract['top_down_only']:
            from src.tools.grasp.geometry import validate_top_down_contact_pose
            validate_top_down_contact_pose(hand_to_contact(result, self.jaw_offset_m))
        return result

    def refine_candidate(self, candidate_ref, roll_deg, pitch_deg, yaw_deg, dx_mm=0., dy_mm=0., dz_mm=0.):
        from src.tools.pose_editor.refinement import checked_adjustment
        from src.tools.pose_editor.geometry import checked_translation
        source, prediction, point_ref = self._candidate_context(candidate_ref)
        contract = self.grasp_contracts[candidate_ref]
        try:
            angles, total_angles = checked_adjustment((roll_deg, pitch_deg, yaw_deg),
                self.candidate_adjustments.get(candidate_ref, (0., 0., 0.)),
                unrestricted=getattr(self, 'unrestricted_pose_rotation', False))
            shift, total_shift = checked_translation((dx_mm, dy_mm, dz_mm),
                getattr(self, 'candidate_translations', {}).get(candidate_ref, (0., 0., 0.)),
                unrestricted=getattr(self, 'unrestricted_pose_translation', False))
        except ValueError as exc:
            from src.core.errors import structured_error
            return dict(accepted=False, reason_code='adjustment_budget_exhausted',
                        error_details=structured_error(exc, phase='pose_adjustment', tool='adjust_grasp'))
        updated = self._refinement_contract(contract, angles)
        pose = self._checked_refinement(prediction.pose, updated, shift, angles)
        new_ref = 'g_'+uuid4().hex
        updated['contact_center_xyz_m'] = hand_to_contact(pose, self.jaw_offset_m)[:3, 3].tolist()
        self.grasp_contracts[new_ref] = updated
        with self._contract(updated):
            entry = self._publish_grasp_pose(new_ref, source,
                GraspPrediction(pose, prediction.score, None, 'robotiq_2f_85'), point_ref)
        self.candidate_adjustments[new_ref] = total_angles
        if not hasattr(self, 'candidate_translations'):
            self.candidate_translations = {}
        self.candidate_translations[new_ref] = total_shift
        entry.update(accepted=entry['executable'], top_down_only=updated['top_down_only'],
            contact_center_xyz_m=updated['contact_center_xyz_m'])
        self._record('grasp_refinement', dict(origin_candidate_ref=candidate_ref, **entry,
            translation_frame='connector_base', rotation_frame='local axes about contact center'))
        return entry

    def nudge_inflight_grasp(self, dx_mm=0., dy_mm=0., dz_mm=0., roll_deg=0., pitch_deg=0., yaw_deg=0.):
        from src.tools.pose_editor.refinement import checked_adjustment
        from src.tools.pose_editor.geometry import checked_translation
        from src.tools.grasp.execution import plan_explicit_grasp
        flight = self._inflight
        if flight is None:
            raise ValueError('no paused grasp')
        original = flight['original']
        contract = (flight.get('replacement') or original).grasp_contract
        try:
            angles, total_angles = checked_adjustment((roll_deg, pitch_deg, yaw_deg), flight['adjustment'],
                unrestricted=getattr(self, 'unrestricted_pose_rotation', False))
            shift, total_shift = checked_translation((dx_mm, dy_mm, dz_mm), flight['translation'],
                step_limit_mm=30, cumulative_limit_mm=90,
                unrestricted=getattr(self, 'unrestricted_pose_translation', False))
        except ValueError as exc:
            from src.core.errors import structured_error
            return dict(accepted=False, reason_code='adjustment_budget_exhausted',
                        error_details=structured_error(exc, phase='pose_adjustment', tool='nudge_grasp'))
        contract = self._refinement_contract(contract, angles)
        pose = self._checked_refinement(flight['current_pose'], contract, shift, angles)
        checker_scene = (self._inflight_context or {}).get('checker_scene')
        if checker_scene is None:
            return dict(accepted=False, reason_code='captured_grasp_scene_unavailable')
        options = self._grasp_options(pose, type('Geometry', (), {'object_points': original.target_points})(),
            None, (self._inflight_context or {}).get('candidate_ref'))
        try:
            plan = plan_explicit_grasp(self.connector, grasp_transform=pose, grasp_to_ee=self.grasp_to_ee,
                target_points=original.target_points, obstacle_points=original.obstacle_points,
                pre_pick_z_m=contract['pre_pick_z_m'], post_pick_z_m=contract['post_pick_z_m'],
                config=self.motion_config, jaw_offset_m=self.jaw_offset_m,
                resume_from_pregrasp=True, max_width_m=self.max_gripper_width_m, **options)
            checker, scene = checker_scene
            evidence = self._explicit_check_grasp_path(checker, plan, scene)
        except MotionPlanningError as exc:
            evidence = {**exc.planning_feedback, 'accepted': False, 'kind': 'planning', 'error': str(exc)}
        attempt = dict(accepted=bool(evidence['accepted']), path_check=self._validation_feedback(evidence),
            step_mm=list(shift), cumulative_mm=list(total_shift),
            step=dict(zip(('roll_deg', 'pitch_deg', 'yaw_deg'), angles)),
            cumulative=dict(zip(('roll_deg', 'pitch_deg', 'yaw_deg'), total_angles)))
        reason = 'nudged_path_rejected'
        if attempt['accepted']:
            previews = self._inflight_preview(flight['observation'], pose, plan.open_width_m, 'grasp_nudge_'+uuid4().hex)
            if not previews:
                attempt['accepted'], reason = False, 'refinement_preview_unavailable'
        flight['attempts'].append(attempt)
        self._record('inflight_nudge', attempt)
        if not attempt['accepted']:
            return {**attempt, 'reason_code': reason}
        plan.grasp_contract = deepcopy(contract)
        plan.grasp_contract['contact_center_xyz_m'] = hand_to_contact(pose, self.jaw_offset_m)[:3, 3].tolist()
        for name in ('scene_observation_id', 'scene_point_ref'):
            if hasattr(original, name):
                setattr(plan, name, getattr(original, name))
        flight.update(replacement=plan, current_pose=pose, adjustment=total_angles, translation=total_shift)
        return {**attempt, 'image_refs': previews}

    def _execute_grasp(self, candidate_ref, validation_ref):
        result = super()._execute_grasp(candidate_ref, validation_ref)
        flight = self._inflight
        if result.get('status') == 'succeeded' and flight is not None and flight.get('replacement') is not None:
            # The attachment archive is written by an inner inherited wrapper,
            # before PauseRefineIntentBackend.execute_grasp returns. Select the
            # executed correction so that archive and later held motion agree.
            self.held_plan = flight['replacement']
        return result

    def relax_candidate(self, *args, **kwargs):
        raise ManipulationPreconditionError('requires_checked_agent_selected_pose')

    def execute_place_candidate(self, *args, **kwargs):
        raise ManipulationPreconditionError('place_movement_and_release_are_separate_actions')
