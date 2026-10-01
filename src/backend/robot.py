"""Small, evidence-returning primitives for the intent-led Prime policy.

No task names, success history, look-at search, or automatic recovery direction
enter this backend. Motion offsets and contact intent belong to the caller.
"""
from contextlib import nullcontext
from copy import copy, deepcopy
import json
import time
from types import SimpleNamespace
from uuid import uuid4

import numpy as np

from src.tools.place.observed_placement import measured_anchor
from src.core.action_feedback import ActionPreconditionError
from src.backend.robot_base import render_candidate
from src.backend.candidate_review import ReviewDrivenBackend


def translation_pose(current, *, delta=None, anchor=None, height_offset_m=0.,
                     optical_from_ee=None):
    """Translate a measured EE pose; preserve its rotation exactly.

    For observation, intersect the CURRENT optical axis with the selected
    surface at a caller-chosen camera height. Calibration fixes the camera/hand
    offset. A non-downward axis is returned as an error, never secretly rotated.
    """
    from src.tools.observation.views import rigid_matrix
    pose = rigid_matrix(current)
    if (delta is None) == (anchor is None):
        raise ValueError('provide either a relative displacement or a surface anchor')
    if delta is not None:
        delta = np.asarray(delta, float)
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise ValueError('finite base-frame displacement required')
        pose[:3, 3] += delta
        return pose
    anchor = np.asarray(anchor, float)
    if anchor.shape != (3,) or not np.isfinite(anchor).all() or not np.isfinite(height_offset_m):
        raise ValueError('finite anchor and explicit height offset required')
    if optical_from_ee is None:
        pose[:3, 3] = anchor + [0., 0., height_offset_m]
        return pose
    calibration = rigid_matrix(optical_from_ee)
    direction = pose[:3, :3] @ calibration[:3, 2]
    if direction[2] >= -1e-3:
        raise ActionPreconditionError('current_optical_axis_not_downward')
    if height_offset_m <= 0:
        raise ActionPreconditionError('positive_optical_height_required')
    camera = anchor - direction * (height_offset_m / -direction[2])
    pose[:3, 3] = camera - pose[:3, :3] @ calibration[:3, 3]
    return pose


def downward_camera_pose(current, ee_from_optical, *, anchor, height_offset_m, delta=(0., 0., 0.)):
    """Minimum camera-axis rotation to base -Z, then calibrated placement.

    The antiparallel case has equally short solutions; rotate about the current
    calibrated camera X axis. No alternate viewpoints or route search occur.
    """
    from src.tools.observation.views import rigid_matrix
    pose, calibration = rigid_matrix(current), rigid_matrix(ee_from_optical)
    anchor = np.asarray(anchor, dtype=float)
    if anchor.shape != (3,) or not np.isfinite(anchor).all() or not np.isfinite(height_offset_m):
        raise ValueError('finite anchor and explicit height offset required')
    if height_offset_m <= 0:
        raise ActionPreconditionError('positive_optical_height_required')
    camera = pose @ calibration
    axis = camera[:3, 2] / np.linalg.norm(camera[:3, 2])
    down = np.array([0., 0., -1.])
    cross = np.cross(axis, down)
    cosine = float(np.clip(axis @ down, -1., 1.))
    sine = float(np.linalg.norm(cross))
    if sine <= 1e-12:
        x = camera[:3, 0] / np.linalg.norm(camera[:3, 0])
        rotation = np.eye(3) if cosine > 0 else 2*np.outer(x, x)-np.eye(3)
    else:
        x, y, z = cross
        skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        rotation = np.eye(3)+skew+skew@skew*((1.-cosine)/(sine*sine))
    pose[:3, :3] = rotation @ pose[:3, :3]
    pose[:3, 3] = anchor + [0., 0., height_offset_m] - pose[:3, :3] @ calibration[:3, 3]
    return rigid_matrix(translation_pose(pose, delta=delta))


class IntentBackend(ReviewDrivenBackend):
    require_release_tracking = True
    def __init__(self, *, task_grasp_budget=12, grasp_batch_per_view=1, downward_pool=0,
                 moveit_grasps=None, object_cloud_policy="source_view", grasp_policy="agent-choice",
                 grasp_motion_policy="source-order", grasp_score_tolerance=0., pose_dedup=None, **kwargs):
        if (type(task_grasp_budget) is not int or task_grasp_budget < 1 or
                type(grasp_batch_per_view) is not int or grasp_batch_per_view < 1):
            raise ValueError('positive integer task and per-view grasp budgets required')
        if type(downward_pool) is not int or downward_pool < 0:
            raise ValueError('downward_pool is a non-negative integer raw proposal pool per view')
        if object_cloud_policy not in ('source_view', 'fused'):
            raise ValueError('unknown object cloud policy')
        if grasp_policy not in ('agent-choice', 'moveit-top', 'ensemble'):
            raise ValueError('unknown grasp policy')
        self.pose_dedup = pose_dedup
        self.grasp_policy = grasp_policy
        from src.tools.grasp.motion_selection import validate_policy
        validate_policy(grasp_motion_policy, grasp_score_tolerance)
        self.grasp_motion_policy = grasp_motion_policy
        self.grasp_score_tolerance = grasp_score_tolerance
        self.object_cloud_policy = object_cloud_policy
        kwargs['contact_manipulation'] = True
        super().__init__(**kwargs)
        self.grasp_mode = 'transport'
        self.task_grasp_budget = task_grasp_budget
        self.grasp_batch_per_view = grasp_batch_per_view
        # 0 keeps one raw proposal per view. N>count samples N raw proposals per
        # view, orders them by downward alignment plus score, and path-checks in
        # that order until `count` are accepted. Budget still counts proposals.
        self.downward_pool = downward_pool
        self.moveit_grasps = moveit_grasps
        self.grasp_reserved = 0
        self.grasp_published = 0
        self.grasp_diagnostics_published = 0
        self.candidate_sources = {}
        self.diagnostic_candidates = {}

    def _planning_target_points(self, source, point_ref):
        """Keep proposal provenance separate from the measured attachment geometry."""
        if getattr(self, 'object_cloud_policy', 'source_view') == 'source_view':
            return source.object_points
        geometry = self._point(point_ref)
        if point_ref != self.latest_fused_ref or geometry.observation_id != source.observation_id:
            raise ValueError('current synchronized fused target required')
        self.point_adapter._check_current(geometry.observation_id)
        views = getattr(geometry, 'per_view', ())
        points = np.array(geometry.object_points, copy=True)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ValueError('invalid fused target points')
        self._record('grasp_object_geometry', dict(policy='fused', point_ref=point_ref,
            observation_id=geometry.observation_id, proposal_view=source.view_id,
            source_views=[view.view_id for view in views], object_point_count=len(points),
            proposal_point_count=len(source.object_points)))
        return points

    def grasp_budget(self):
        return dict(limit=self.task_grasp_budget, reserved=self.grasp_reserved,
                    remaining=self.task_grasp_budget-self.grasp_reserved,
                    per_view_batch=self.grasp_batch_per_view,
                    per_view_batch_scope='graspgen_only; not MoveIt',
                    moveit_pool_slots=6,
                    published_candidates=self.grasp_published,
                    published_diagnostics=self.grasp_diagnostics_published,
                    execution_attempts=len(getattr(self, 'grasp_execution_evidence', {})),
                    policy='episode lifetime candidate slots; failures are not refunded; raw generation counts recorded separately')

    def review_observation(self, observation_id):
        """Read only RGB already captured and registered by this backend instance."""
        from pathlib import Path
        from src.tools.perception.multiview import CAMERAS
        if (not isinstance(observation_id, str) or observation_id not in self.observation_views
                or observation_id not in self.point_adapter.frames):
            raise ValueError('observation is not available in this episode')
        captured = {frame.view_id: frame for frame in self.point_adapter.frames[observation_id]}
        registered = {view['view_id']: view for view in self.observation_views[observation_id]}
        views = []
        for view_id in CAMERAS:
            frame, view = captured.get(view_id), registered.get(view_id)
            if frame is None or view is None:
                raise ValueError('saved observation does not contain both RGB views')
            image_ref = view['image_ref']
            if self.images.paths.get(image_ref) != Path(frame.rgb_path):
                raise ValueError('saved RGB does not match this observation registration')
            views.append(dict(view_id=view_id, image_ref=image_ref))
        return dict(observation_id=observation_id, views=views, reference_only=True,
                    historical=observation_id != self.latest_observation_id,
                    current_observation_id=self.latest_observation_id,
                    scope='saved RGB from this episode; no recapture or state change')

    def save_destination(self, point_ref):
        # Resolve current geometry before reporting its role; stale refs retain
        # their existing rejection and cannot be reinterpreted as destinations.
        if self._point(point_ref).role != 'place':
            raise ActionPreconditionError('destination_requires_place_selection')
        return super().save_destination(point_ref)

    def set_grasp_mode(self, mode):
        if mode not in ('transport', 'contact'):
            raise ValueError('grasp mode must be transport or contact')
        if self.held_plan is not None or self.closed_push:
            raise ActionPreconditionError('grasp_mode_requires_open')
        changed = mode != self.grasp_mode
        if changed:
            # A contact route does not lift; a transport route does. Never reuse
            # the cached plan of the other mode under an unchanged candidate ID.
            for name in ('candidates', 'candidate_routes', 'candidate_point_refs',
                         'candidate_open_widths',
                         'candidate_adjustments', 'candidate_sources', 'diagnostic_candidates', 'validations',
                         '_candidate_inspection_cache'):
                getattr(self, name).clear()
            getattr(self, 'candidate_translations', {}).clear()
        self.grasp_mode = mode
        result = dict(grasp_mode=mode, lift_after_grasp=mode == 'transport', candidates_invalidated=changed)
        self._record('grasp_mode', result)
        return result

    def _capture_grasp_scene(self, geometry):
        """Preserve the inherited placement attachment's sensor provenance."""
        from src.tools.motion.planning import _pose_transform
        from src.llm.manager import write_json
        if geometry.observation_id in self.scene_grippers:
            return
        from src.tools.gripper.state import measured_opening
        tolerance = getattr(self, 'capture_width_tolerance_m', .0005)
        width, raw_qpos, _ = measured_opening(self.connector,
            max_width_m=getattr(self, 'max_gripper_width_m', .08), tolerance_m=tolerance)
        captured = dict(observation_id=geometry.observation_id,
            hand_pose=(_pose_transform(self.connector.get_ee_pose()) @ np.linalg.inv(self.grasp_to_ee)).tolist(),
            raw_jaw_qpos=raw_qpos, modeled_jaw_width_m=width,
            width_clamp_tolerance_m=tolerance, frame='connector_base',
            source='same capture epoch robot proprioception; no scene object state')
        self.point_adapter._check_current(geometry.observation_id)
        self.scene_grippers[geometry.observation_id] = captured
        write_json(self.output_dir/('scene_gripper_'+geometry.observation_id+'.json'), captured)

    def grasp_candidates(self, point_ref, preferred_direction=None, grasp_type=None, batch_size=None):
        if self.grasp_policy == 'ensemble':
            if any(value is not None for value in (preferred_direction, grasp_type, batch_size)):
                raise ValueError('ensemble has fixed sources and two slots per source')
            from src.tools.grasp.ensemble import generate_ensemble
            return generate_ensemble(self, point_ref)
        return self._grasp_candidates_single(point_ref, preferred_direction, grasp_type, batch_size)

    def _grasp_candidates_single(self, point_ref, preferred_direction=None, grasp_type=None, batch_size=None):
        """Route generation, then share all publication, planning and refinement."""
        from src.tools.motion.planning import plan_grasp
        from src.tools.grasp.preference import MAX_GRASP_CANDIDATES, normalize_preference, normalize_grasp_type
        if self.grasp_policy == 'moveit-top':
            preferred_direction, grasp_type = 'vertical', 'face'
        preference = normalize_preference(preferred_direction)
        family = normalize_grasp_type(grasp_type)
        use_moveit = preference is not None or family is not None
        if batch_size is not None and (type(batch_size) is not int or not 1 <= batch_size <= MAX_GRASP_CANDIDATES):
            raise ValueError('batch_size must be an integer from 1 to 6')
        pool_slots = batch_size if batch_size is not None else MAX_GRASP_CANDIDATES
        generator_name = 'moveit_grasps' if use_moveit else getattr(self.graspgen, 'generator_name', 'graspgen')
        geometry = self._point(point_ref)
        self.point_adapter._check_current(geometry.observation_id)
        if point_ref != self.latest_fused_ref or geometry.role != 'pick':
            raise ValueError('current selected pick region required')
        if self.held_plan is not None or self.closed_push:
            raise ActionPreconditionError('grasp_requires_release')
        moveit = self.moveit_grasps
        if use_moveit:
            if moveit is None:
                from src.tools.grasp.moveit import MoveItGraspsBackend
                moveit = MoveItGraspsBackend(robot_profile=
                    'robolab_robotiq' if getattr(self, 'gripper_assets', None) is not None else 'libero_panda')
            # Test/custom generators may implement only the prediction protocol.
            if hasattr(moveit, 'preflight'):
                moveit.preflight()
        self._capture_grasp_scene(geometry)
        images = self.saved_views(point_ref)['image_refs']
        result, views, diagnostics = [], [], []
        request_reserved = 0
        # Source order is deterministic, not a task-conditioned preference.
        geometries = ([geometry] if use_moveit else
                      sorted(getattr(geometry, 'per_view', (geometry,)), key=lambda g: g.view_id))
        for source in geometries:
            count = min(pool_slots-len(result),
                        MAX_GRASP_CANDIDATES if use_moveit else self.grasp_batch_per_view,
                        self.task_grasp_budget-self.grasp_reserved,
                        pool_slots-request_reserved if batch_size is not None else MAX_GRASP_CANDIDATES)
            report = dict(source_view=source.view_id, generated_budget=count,
                          reserved_slots=0, raw_proposals_generated=None,
                          image_refs=[self.images.add(source.overlay_path)],
                          measured_target_points=len(source.object_points), path_rejections=[])
            views.append(report)
            if not count:
                report.update(reason_code=('task_grasp_budget_exhausted' if self.grasp_reserved >= self.task_grasp_budget
                                           else 'pool_slot_limit_reached'), accepted_count=0)
                continue
            if len(source.object_points) < 100:
                report.update(reason_code='insufficient_measured_depth', accepted_count=0)
                continue
            # Reserve before crossing the model-worker boundary: timeout may have
            # occurred after inference, so retrying must not replenish this budget.
            self.grasp_reserved += count
            request_reserved += count
            report['reserved_slots'] = count
            self._record('grasp_budget_reserved', dict(point_ref=point_ref,
                         source_view=source.view_id, batch_count=count, **self.grasp_budget()))
            original = self.graspgen
            if use_moveit:
                batch = copy(moveit)
                batch.last_generation = {}
                batch.refinement_only_rows = []
                batch.expose_partial_diagnostics = self.grasp_policy == 'ensemble'
                batch.preferred_direction = preference
                batch.grasp_type = family or "face"
                batch.motion_policy = self.grasp_motion_policy
                batch.motion_score_tolerance = self.grasp_score_tolerance
                if self.grasp_motion_policy == 'low-motion':
                    from src.tools.motion.planning import _pose_transform
                    batch.current_ee = _pose_transform(self.connector.get_ee_pose())
                    batch.grasp_to_ee = self.grasp_to_ee.copy()
                if getattr(batch, 'scene_filter_enabled', False):
                    from src.tools.grasp.scene_filter import MoveItSceneFilter
                    batch.scene_filter = MoveItSceneFilter(self, source, point_ref,
                        prefix=batch.prefix, ros_master_uri=batch.ros_master_uri,
                        executable=getattr(batch, 'scene_executable', None),
                        assets=getattr(batch, 'scene_assets', None),
                        robot_profile=getattr(batch, 'robot_profile', 'libero_panda'))
                batch.checkout = getattr(original, 'checkout', None)  # same environment-specific preview meshes
                batch.official_clearance_m = getattr(original, 'official_clearance_m', .002)
            else:
                batch = copy(original)
            directory = self.output_dir / ('intent_grasp_'+uuid4().hex)
            batch.output_dir, batch.calls = directory, 0
            pool = max(count, self.downward_pool)
            batch.num_grasps = batch.max_generated = batch.topk = pool
            batch.target_candidates = count
            batch.clearance_batch_floor_m = None
            batch.open_width_m, batch.adaptive_contact_opening = None, True
            rejected, accepted = [], []
            motion_metadata = {}
            approach_meta = {}
            rank_batch = None
            if self.downward_pool and not preference:
                from src.backend.robot_base import rank_predictions
                def rank_batch(predictions):
                    ranked = rank_predictions(list(predictions), downward_weight=self.downward_weight,
                                              topk=max(1, len(predictions)))
                    for prediction, meta in ranked:
                        approach_meta[id(prediction)] = meta
                    return [prediction for prediction, _ in ranked]
                report['selection_rule'] = dict(pool=pool, ranking='source_score + downward_weight * dot(local_+Z, base_-Z)',
                                                downward_weight=float(self.downward_weight))

            def accept(prediction):
                if len(accepted) >= count or len(result)+len(accepted) >= MAX_GRASP_CANDIDATES:
                    return False
                ref = 'g_'+uuid4().hex
                checker, scene = self._candidate_path_scene(point_ref)
                state = self._robot_state()
                started = time.monotonic()
                plan = None
                grasp_options = None
                try:
                    grasp_options = self._grasp_options(prediction.pose, source, prediction, ref)
                    self.candidate_open_widths[ref] = grasp_options['open_width_m']
                    if getattr(self, '_relax_grasp_generation', False):
                        plan, evidence = self._plan_relaxed_grasp(source, prediction, point_ref, ref)
                    else:
                        plan, evidence = self._plan_grasp_candidate(prediction.pose, self._planning_target_points(source, point_ref),
                            geometry.scene_points, checker, scene, grasp_options)
                    if evidence.get('collision_witness'):
                        evidence['collision_witness'] = {**evidence['collision_witness'],
                            'observation_id': geometry.observation_id, 'point_ref': point_ref}
                except Exception as exc:
                    evidence = dict(accepted=False, kind='planning', error=repr(exc))
                    from src.core.planning_feedback import public_planning_feedback
                    evidence.update(public_planning_feedback(getattr(exc, 'planning_feedback', None)))
                elapsed = time.monotonic()-started
                self._record('intent_grasp_path_check', dict(candidate_ref=ref,
                             source_view=source.view_id, elapsed_s=elapsed, evidence=evidence))
                if not evidence['accepted']:
                    rejected.append(dict(kind=evidence.get('kind', 'path'),
                                         segment=evidence.get('segment'), elapsed_s=elapsed))
                    diagnostic = (self._publish_diagnostic(ref, source, prediction, point_ref, evidence,
                        open_width_m=grasp_options['open_width_m'] if grasp_options is not None else None)
                        if len(diagnostics) < MAX_GRASP_CANDIDATES else None)
                    if diagnostic is not None:
                        diagnostics.append(diagnostic)
                    return False
                path = self.output_dir / (ref+'.png')
                render_candidate(source, prediction, ref, path, expected_open_width_m=plan.open_width_m,
                    max_width_m=getattr(self, 'max_gripper_width_m', .08),
                    **getattr(self, 'gripper_render_options', {}))
                self.candidate_routes[ref] = (state, plan)
                self.candidates[ref] = (self.epoch, source, prediction)
                # The target identity remains the same while proposal geometry
                # retains its own view. Validation/refinement keep existing guards.
                self.candidate_point_refs[ref] = point_ref
                self.candidate_sources[ref] = source.view_id
                entry = dict(candidate_ref=ref, source_view=source.view_id,
                             image_refs=[self.images.add(path)])
                if getattr(self, '_relax_grasp_generation', False):
                    self.relaxed_refs.add(ref)
                    entry.update(executable=True, collision_relaxed=True)
                meta = approach_meta.get(id(prediction))
                if meta is not None:
                    entry['downward_angle_deg'] = round(float(meta['downward_angle_deg']), 1)
                    entry['pool_rank'] = int(meta['published_rank'])
                accepted.append(entry)
                if use_moveit and self.grasp_motion_policy == 'low-motion':
                    from src.tools.grasp.motion_selection import motion_cost, estimated_motion_seconds
                    cost = motion_cost(prediction.pose, batch.current_ee, self.grasp_to_ee)
                    duration = estimated_motion_seconds(plan, getattr(self.connector, 'env', None))
                    if duration is not None:
                        cost['estimated_motion_s'] = duration
                    motion_metadata[ref] = dict(source_score=float(prediction.score), **cost)
                self._record('intent_grasp_candidate', dict(**entry, point_ref=point_ref,
                             generator=generator_name,
                             preferred_direction=preference, grasp_type=(family or "face") if use_moveit else None,
                             grasp_transform=prediction.pose, source_score=prediction.score,
                             image_path=str(path)))
                return True

            started = time.monotonic()
            self.graspgen = batch
            try:
                # This existing method emits one exact raw batch and verifies its
                # generated_count. Setting max_generated=count forbids inner retry.
                if rank_batch is None:
                    batch.predict_with_path_filter(source.object_points, source.scene_points, accept)
                else:
                    batch.predict_with_path_filter(source.object_points, source.scene_points, accept,
                                                   rank_batch=rank_batch)
                if use_moveit and self.grasp_motion_policy == 'low-motion':
                    self._record('grasp_motion_selection', dict(policy=self.grasp_motion_policy,
                        selection_scope='equivalent_180_degree_jaw_symmetries_only',
                        score_tolerance=self.grasp_score_tolerance, candidates=motion_metadata,
                        ordered_candidate_refs=[e['candidate_ref'] for e in accepted],
                        time_estimate_scope='planned streaming ticks; excludes settling, gripper holds and model latency'))
                if use_moveit and ((self.grasp_policy == 'ensemble' and len(accepted)+len(diagnostics) < count)
                                   or (not accepted and not diagnostics)):
                    for row in getattr(batch, 'refinement_only_rows', ()):
                        if self.grasp_policy == 'ensemble' and len(accepted)+len(diagnostics) >= count:
                            break
                        prediction = batch.prediction(row)
                        ref = 'g_'+uuid4().hex
                        options = self._grasp_options(prediction.pose, source, prediction, ref)
                        evidence = dict(accepted=False, kind='moveit_prefilter',
                            error=row.get('moveit_scene_result', {}).get('status'),
                            original_rank=row['original_rank'])
                        diagnostic = self._publish_diagnostic(ref, source, prediction, point_ref,
                            evidence, open_width_m=options['open_width_m'])
                        if diagnostic is not None:
                            diagnostics.append(diagnostic)
                            self._record('moveit_refinement_only_candidate', dict(candidate_ref=ref, **evidence))
                report['reason_code'] = ('candidate_available' if accepted else
                    'path_rejected' if rejected else 'model_or_scene_filter_rejected')
                metadata_path = directory/'graspgen_001'/'batches'/'batch-001'/'graspgen_001'/'predictions.json'
                if use_moveit:
                    report['generation'] = dict(getattr(batch, 'last_generation', {}))
                elif metadata_path.is_file():
                    metadata = json.loads(metadata_path.read_text())
                    report['model_filter'] = dict(generated_count=metadata.get('generated_count'),
                        confidence_survivors=metadata.get('raw_count'), scene_survivors=metadata.get('accepted_count'))
                    if not accepted and not rejected:
                        report['reason_code'] = ('confidence_rejected' if metadata.get('raw_count') == 0
                                                 else 'gripper_scene_rejected')
            except Exception as exc:
                report['reason_code'] = 'inference_or_geometry_error'
                report['error_type'] = type(exc).__name__
                self._record('intent_grasp_error', dict(source_view=source.view_id, error=repr(exc)))
            finally:
                self.graspgen = original
                generation = getattr(batch, 'last_generation', {}) if use_moveit else report.get('model_filter', {})
                raw_count = generation.get('raw_count' if use_moveit else 'generated_count')
                report['raw_proposals_generated'] = raw_count if type(raw_count) is int else None
                report['reserved_slots'] = count
                report.update(elapsed_s=time.monotonic()-started, accepted_count=len(accepted),
                              path_rejections=rejected)
                self._record('intent_grasp_source', dict(**report, inference_artifact=str(directory)))
            result.extend(accepted)
        # Accepted poses take priority; every published diagnostic still follows
        # the existing nonexecutable inspection/refinement contract.
        shown_diagnostics = diagnostics[:pool_slots-len(result)]
        self.grasp_published += len(result)
        self.grasp_diagnostics_published += len(shown_diagnostics)
        output = dict(candidates=result, diagnostic_candidates=shown_diagnostics, image_refs=images, source_reports=views,
                      generator=generator_name, preferred_direction=preference, grasp_type=(family or "face") if use_moveit else None,
                      candidate_limit=pool_slots,
                      task_budget=self.grasp_budget(),
                      reason_code='candidate_available' if result else 'no_feasible_candidate',
                      uncertainty='Missing observed surfaces remain unknown; rejection alone does not prove poor visibility.')
        self._record('grasp_candidates', dict(point_ref=point_ref, **output))
        return output

    def inspect_candidate(self, candidate_ref):
        if candidate_ref in self.diagnostic_candidates:
            self._candidate_context(candidate_ref)
            return deepcopy(self.diagnostic_candidates[candidate_ref])
        result = super().inspect_candidate(candidate_ref)
        result['source_view'] = self.candidate_sources.get(candidate_ref, self.candidates[candidate_ref][1].view_id)
        return result

    @staticmethod
    def _validation_feedback(evidence):
        """Small geometric diagnosis; private exceptions and scene coordinates stay private."""
        from src.core.planning_feedback import public_planning_feedback
        feedback = {}
        for key in ('kind', 'segment'):
            value = evidence.get(key)
            if isinstance(value, str) and value.replace('_', '').isalnum():
                feedback[key] = value
        for key in ('sample_count', 'clearance_m', 'colliding_points', 'geom_scene_clearance_m'):
            value = evidence.get(key)
            if type(value) in (int, float) and np.isfinite(value):
                feedback[key] = value
        if type(evidence.get('geom_world_stationary')) is bool:
            feedback['geom_world_stationary'] = evidence['geom_world_stationary']
        geom = evidence.get('geom', evidence.get('geoms'))
        names = [geom] if isinstance(geom, str) else geom
        if (isinstance(names, list) and names and all(isinstance(n, str)
                and n.startswith(('robot0_', 'gripper0_')) and n.replace('_', '').isalnum() for n in names)):
            feedback['robotgeom'] = names[0] if isinstance(geom, str) else list(names)
        if evidence.get('kind') == 'planning':
            feedback.pop('segment', None)
        feedback.update(public_planning_feedback(evidence))
        return feedback

    def _candidate_context(self, candidate_ref):
        self._latest_candidate(candidate_ref)
        epoch, geometry, prediction = self.candidates[candidate_ref]
        point_ref = self.candidate_point_refs[candidate_ref]
        if epoch != self.epoch or point_ref != self.latest_fused_ref:
            raise ValueError('stale candidate or selected target')
        self._point(point_ref)
        self.point_adapter._check_current(geometry.observation_id)
        return geometry, prediction, point_ref

    @staticmethod
    def _diagnostic_images(paths):
        """Publish new labelled images; never overwrite an earlier preview artifact."""
        from pathlib import Path
        from PIL import Image, ImageDraw
        result = []
        for path in paths:
            path = Path(path)
            output = path.with_name(path.stem+'_diagnostic'+path.suffix)
            with Image.open(path) as source:
                image = source.convert('RGB')
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, image.width, 24), fill='black')
            draw.text((5, 5), 'DIAGNOSTIC: NOT EXECUTABLE - path rejected', fill='orange')
            image.save(output)
            result.append(output)
        return result

    def _witness_diagnostic_images(self, paths, geometry, point_ref, evidence, candidate_ref):
        """Best-effort localization must never discard an endpoint diagnostic."""
        result = list(paths)
        try:
            witness = evidence.get('collision_witness', {})
            if not (evidence.get('kind') == 'environment' and isinstance(witness, dict) and witness
                    and witness.get('observation_id') == geometry.observation_id
                    and witness.get('point_ref') == point_ref
                    and geometry.observation_id == self.latest_observation_id):
                return result
            from src.tools.pose_editor.clearance_preview import compose_witness_preview, visible_witness_pixel
            frames = self.point_adapter.frames.get(geometry.observation_id, [])
            self.point_adapter._check_current(geometry.observation_id)
            visible = [f for f in frames if visible_witness_pixel(f, witness) is not None]
            for i, path in enumerate(paths):
                try:
                    frame = next((f for f in visible if f'preview-{f.view_id}' in path.name), None)
                    if frame is None:
                        frame = next((f for f in visible if f.view_id == geometry.view_id), None)
                    if frame is None and visible:
                        frame = visible[0]
                    if frame is not None:
                        composed = compose_witness_preview(path, frame, witness,
                            path.with_name(path.stem+'_witness.png'))
                        if composed is not None:
                            result[i] = composed
                except Exception as exc:
                    self._record('clearance_witness_preview_error', dict(candidate_ref=candidate_ref, error=repr(exc)))
        except Exception as exc:
            self._record('clearance_witness_preview_error', dict(candidate_ref=candidate_ref, error=repr(exc)))
        return result

    def _publish_diagnostic(self, ref, geometry, prediction, point_ref, evidence, *,
                            open_width_m=None, paths=None, adjustment=(0., 0., 0.), translation=(0., 0., 0.)):
        try:
            if paths is None:
                path = self.output_dir/(ref+'.png')
                render_candidate(geometry, prediction, ref, path, expected_open_width_m=open_width_m,
                    max_width_m=getattr(self, 'max_gripper_width_m', .08),
                    **getattr(self, 'gripper_render_options', {}))
                paths = [path]
            paths = self._diagnostic_images(paths)
            # Localize a saved checker witness, never reconstruct a local
            # point_index or re-query geometry. Keep the existing image count.
            paths = self._witness_diagnostic_images(paths, geometry, point_ref, evidence, ref)
            if not paths:
                raise ValueError('diagnostic preview unavailable')
        except Exception as exc:
            self._record('diagnostic_preview_error', dict(candidate_ref=ref, error=repr(exc)))
            return None
        entry = dict(candidate_ref=ref, source_view=geometry.view_id,
            image_refs=[self.images.add(p) for p in paths], executable=False,
            reason_code='candidate_path_rejected', validation_feedback=self._validation_feedback(evidence))
        self.candidates[ref] = (self.epoch, geometry, prediction)
        self.candidate_open_widths[ref] = open_width_m
        self.candidate_point_refs[ref] = point_ref
        self.candidate_sources[ref] = geometry.view_id
        self.candidate_adjustments[ref] = adjustment
        if not hasattr(self, 'candidate_translations'):
            self.candidate_translations = {}
        self.candidate_translations[ref] = translation
        self.diagnostic_candidates[ref] = deepcopy(entry)
        self._record('diagnostic_grasp_candidate', {**entry, 'point_ref':point_ref})
        return entry

    def preview_candidate(self, candidate_ref, azimuth_deg, elevation_deg, zoom):
        if candidate_ref in self.diagnostic_candidates:
            self._candidate_context(candidate_ref)
        result = super().preview_candidate(candidate_ref, azimuth_deg, elevation_deg, zoom)
        if candidate_ref in self.diagnostic_candidates:
            paths = self._diagnostic_images([self.images.paths[r] for r in result['image_refs']])
            result = {**deepcopy(self.diagnostic_candidates[candidate_ref]),
                      'image_refs':[self.images.add(p) for p in paths]}
            self._record('diagnostic_candidate_preview', result)
        return result

    def validate_grasp(self, candidate_ref):
        if candidate_ref in self.diagnostic_candidates:
            self._candidate_context(candidate_ref)
            return {**deepcopy(self.diagnostic_candidates[candidate_ref]), 'accepted':False}
        return super().validate_grasp(candidate_ref)

    def execute_grasp(self, candidate_ref, validation_ref):
        if candidate_ref in self.diagnostic_candidates:
            self._candidate_context(candidate_ref)
            raise ValueError('diagnostic candidate is not executable; refine and validate a new candidate')
        return super().execute_grasp(candidate_ref, validation_ref)

    def refine_candidate(self, candidate_ref, roll_deg, pitch_deg, yaw_deg, dx_mm=0., dy_mm=0., dz_mm=0.):
        from src.tools.motion.planning import plan_grasp
        from src.tools.grasp.prediction import GraspPrediction
        from src.tools.pose_editor.refinement import checked_adjustment, preview_refined_pose, refined_grasp_pose, tcp_offset_from
        from src.tools.pose_editor.geometry import checked_translation
        geometry, prediction, point_ref = self._candidate_context(candidate_ref)
        try:
            step, total = checked_adjustment((roll_deg,pitch_deg,yaw_deg),
                self.candidate_adjustments.get(candidate_ref,(0.,0.,0.)))
            translation, total_translation = checked_translation((dx_mm,dy_mm,dz_mm),
                getattr(self,'candidate_translations',{}).get(candidate_ref,(0.,0.,0.)))
        except ValueError:
            return {'accepted':False, 'reason_code':'adjustment_budget_exhausted'}
        offset = getattr(getattr(self, 'gripper_assets', None), 'jaw_center_offset_m', tcp_offset_from(self.grasp_to_ee))
        pose = refined_grasp_pose(prediction.pose,step,tcp_offset_z_m=offset)
        pose[:3,3] += prediction.pose[:3,:3] @ (np.asarray(translation)/1000.)
        refined = GraspPrediction(pose, prediction.score, getattr(prediction,'gripper_adapter',None),
                                  getattr(prediction, 'gripper_name', 'franka_panda'))
        ref = 'g_'+uuid4().hex
        grasp_options = self._grasp_options(pose,geometry,refined,ref)
        self.candidate_open_widths[ref] = grasp_options['open_width_m']
        paths = preview_refined_pose(self.point_adapter.frames[geometry.observation_id],pose,
                                    self.output_dir/ref,tcp_offset_z_m=offset,
                                    expected_open_width_m=grasp_options['open_width_m'],
                                     mesh_source=getattr(self, 'gripper_assets', None))
        if not paths:
            return {'accepted':False, 'reason_code':'refinement_preview_unavailable'}
        try:
            checker, scene = self._candidate_path_scene(point_ref)
            state = self._robot_state()
            plan, evidence = self._plan_grasp_candidate(pose, self._planning_target_points(geometry, point_ref),
                self._point(point_ref).scene_points, checker, scene, grasp_options)
            if evidence.get('collision_witness'):
                evidence['collision_witness'] = {**evidence['collision_witness'],
                    'observation_id': geometry.observation_id, 'point_ref': point_ref}
        except Exception as exc:
            evidence = {'accepted':False, 'kind':'planning', 'error':repr(exc)}
            from src.core.planning_feedback import public_planning_feedback
            evidence.update(public_planning_feedback(getattr(exc, 'planning_feedback', None)))
        self._record('intent_refinement_path_check', dict(candidate_ref=ref,origin_candidate_ref=candidate_ref,evidence=evidence))
        if not evidence['accepted']:
            diagnostic = self._publish_diagnostic(ref,geometry,refined,point_ref,evidence,paths=paths,
                                                  open_width_m=grasp_options['open_width_m'],
                                                  adjustment=total,translation=total_translation)
            self._record('candidate_refinement_rejected', dict(candidate_ref=candidate_ref,
                diagnostic_candidate_ref=ref if diagnostic else None,
                adjustment_deg=step, cumulative_deg=total,
                translation_local_mm=translation, cumulative_translation_local_mm=total_translation,
                grasp_transform=pose, path_check=evidence))
            return {'accepted':False, **(diagnostic or {'reason_code':'refinement_preview_unavailable'})}
        self.candidates[ref] = (self.epoch,geometry,refined)
        self.candidate_point_refs[ref] = point_ref
        self.candidate_sources[ref] = geometry.view_id
        self.candidate_routes[ref] = (state,plan)
        self.candidate_adjustments[ref] = total
        if not hasattr(self,'candidate_translations'):
            self.candidate_translations = {}
        self.candidate_translations[ref] = total_translation
        result = dict(accepted=True,candidate_ref=ref,source_view=geometry.view_id,
            image_refs=[self.images.add(p) for p in paths],adjustment_deg=dict(zip(('roll_deg','pitch_deg','yaw_deg'),step)),
            cumulative_deg=dict(zip(('roll_deg','pitch_deg','yaw_deg'),total)))
        self._record('candidate_refinement', {**result, 'origin_candidate_ref':candidate_ref,
            'cumulative_translation_local_mm':total_translation,'grasp_transform':pose})
        return result

    def _plan_grasp_candidate(self, pose, target_points, obstacle_points, checker, scene, options):
        from src.tools.motion.planning import plan_grasp
        plan = plan_grasp(self.connector, grasp_transform=pose,
            max_width_m=getattr(self, 'max_gripper_width_m', .08),
            target_points=target_points, obstacle_points=obstacle_points,
            grasp_to_ee=self.grasp_to_ee, frame='connector_base', config=self.motion_config, **options)
        return plan, checker.check(plan, scene, stop_label='grasp', jaw_width_m=plan.open_width_m)

    def _motion_geometry(self, observation_id, target_ref):
        """Current observed scene, including target, never a pre-motion cloud."""
        from src.tools.perception.multiview import voxel_merge
        self.point_adapter._check_current(observation_id)
        if target_ref not in self.region_tracks:
            raise ValueError('known target reference required')
        clouds = {}
        for frame in self.point_adapter.frames[observation_id]:
            depth = np.asarray(frame.depth_m)
            y, x = np.nonzero(np.isfinite(depth) & (depth > .02) & (depth < 3.))
            k = np.asarray(frame.intrinsics)
            rays = np.column_stack(((x-k[0, 2])/k[0, 0], (y-k[1, 2])/k[1, 1], np.ones(len(x))))
            camera = rays*depth[y, x, None]
            clouds[frame.view_id] = camera @ np.asarray(frame.camera_to_base.rotation).T + np.asarray(frame.camera_to_base.translation)
        return voxel_merge(clouds).points

    def _check_observation_command(self):
        # Held-object observation can override this gate while retaining path validation.
        if self.held_plan is not None or self.closed_push:
            raise ActionPreconditionError('observe_requires_open_command')

    def _save_waypoint(self, observation_id, target_ref, pose, purpose, instruction,
                       *, orientation_policy='preserve_current', ee_from_optical=None):
        if purpose not in ('observe', 'transport', 'contact'):
            raise ValueError('explicit observe, transport, or contact purpose required')
        if purpose == 'observe':
            self._check_observation_command()
        if purpose == 'contact' and self.held_plan is None and not self.closed_push:
            raise ActionPreconditionError('contact_requires_closed_command')
        if purpose == 'transport' and self.closed_push:
            raise ActionPreconditionError('pusher_requires_contact_purpose')
        ref = 'wp_'+uuid4().hex
        track = self.region_tracks.get(target_ref, {})
        selected = self.points.get(track.get('point_ref'))
        target_points = (selected[1].object_points if selected and selected[0] == self.epoch
                         and selected[1].observation_id == observation_id else None)
        if self.held_plan is None and self.closed_push and purpose == 'contact' and target_points is None:
            raise ActionPreconditionError('contact_requires_current_target')
        self.view_proposals[ref] = dict(epoch=self.epoch, observation_id=observation_id,
            target_ref=target_ref, pose=pose, purpose=purpose, instruction=instruction,
            scene=self._motion_geometry(observation_id, target_ref), target_points=target_points,
            start=self._robot_state(), orientation_policy=orientation_policy,
            ee_from_optical=None if ee_from_optical is None else ee_from_optical.copy())
        output = self.preview_view(ref)
        self._record('intent_waypoint', {**output, 'instruction': instruction, 'ee_pose': pose})
        return output

    def propose_waypoint(self, observation_id, u, v, target_ref, purpose, height_offset_m,
                         dx_m=0., dy_m=0., dz_m=0.):
        from src.tools.observation.views import fresh_wrist_state
        from src.tools.motion.planning import _pose_transform
        from src.tools.perception.multiview import CAMERAS
        self.point_adapter._check_current(observation_id)
        frame = next(f for f in self.point_adapter.frames[observation_id] if f.view_id == CAMERAS[0])
        anchor = measured_anchor(frame, u, v)
        if purpose == 'observe':
            state = fresh_wrist_state(self.connector)
            pose = translation_pose(state['ee'], anchor=anchor, height_offset_m=height_offset_m,
                                    optical_from_ee=state['ee_from_optical'])
        else:
            pose = translation_pose(_pose_transform(self.connector.get_ee_pose()),
                                    anchor=anchor, height_offset_m=height_offset_m)
        pose = translation_pose(pose, delta=(dx_m, dy_m, dz_m))
        return self._save_waypoint(observation_id, target_ref, pose, purpose,
            dict(front_pixel=[u, v], height_offset_m=height_offset_m,
                 base_offset_m=[dx_m, dy_m, dz_m]))

    def shift_waypoint(self, observation_id, dx_m, dy_m, dz_m, target_ref, purpose):
        from src.tools.motion.planning import _pose_transform
        self.point_adapter._check_current(observation_id)
        pose = translation_pose(_pose_transform(self.connector.get_ee_pose()), delta=(dx_m, dy_m, dz_m))
        return self._save_waypoint(observation_id, target_ref, pose, purpose,
                                   dict(base_displacement_m=[dx_m, dy_m, dz_m]))

    def propose_downward_waypoint(self, observation_id, u, v, target_ref, height_offset_m,
                                  dx_m=0., dy_m=0., dz_m=0.):
        from src.tools.observation.views import fresh_wrist_state
        from src.tools.perception.multiview import CAMERAS
        self.point_adapter._check_current(observation_id)
        self._check_observation_command()
        frame = next(f for f in self.point_adapter.frames[observation_id] if f.view_id == CAMERAS[0])
        anchor = measured_anchor(frame, u, v)
        state = fresh_wrist_state(self.connector)
        pose = downward_camera_pose(state['ee'], state['ee_from_optical'], anchor=anchor,
            height_offset_m=height_offset_m, delta=(dx_m, dy_m, dz_m))
        return self._save_waypoint(observation_id, target_ref, pose, 'observe',
            dict(front_pixel=[u, v], height_offset_m=height_offset_m,
                 base_offset_m=[dx_m, dy_m, dz_m]), orientation_policy='camera_down',
                 ee_from_optical=state['ee_from_optical'])

    def preview_view(self, waypoint_ref, azimuth_deg=None, elevation_deg=None, zoom=None):
        """Saved RGB arrows describe a proposed translation, not a future image."""
        from PIL import Image, ImageDraw
        from src.tools.grasp.input_cards import camera_project, arrow
        from src.tools.motion.planning import _pose_transform
        p = self._view(waypoint_ref)
        current = _pose_transform(self.connector.get_ee_pose())
        orientation_policy = p.get('orientation_policy', 'preserve_current')
        refs = []
        for frame in self.point_adapter.frames[p['observation_id']]:
            path = self.output_dir / (waypoint_ref+'_'+frame.view_id+'_'+uuid4().hex+'.png')
            im = Image.fromarray(np.asarray(frame.rgb).copy())
            draw = ImageDraw.Draw(im)
            xy, _ = camera_project(np.array([current[:3, 3], p['pose'][:3, 3]]), frame)
            arrow(draw, xy[0], xy[1], 'cyan')
            caption = 'EE translation proposal; orientation unchanged'
            if orientation_policy == 'camera_down':
                optical = p['pose'] @ p['ee_from_optical']
                camera_xy, _ = camera_project(np.array([optical[:3, 3],
                    optical[:3, 3]+.08*optical[:3, 2]]), frame)
                arrow(draw, camera_xy[0], camera_xy[1], 'lime')
                caption = 'Camera down proposal; rotation + translation; not executed'
            draw.text((8, 8), caption, fill='yellow',
                      stroke_fill='black', stroke_width=1)
            im.save(path)
            refs.append(self.images.add(path))
        return dict(waypoint_ref=waypoint_ref, observation_id=p['observation_id'], image_refs=refs,
                    purpose=p['purpose'], orientation_policy=orientation_policy, robot_motion=False)

    def refine_view(self, *args, **kwargs):
        raise ValueError('translation proposals preserve orientation; request a new shift_waypoint instead')

    def validate_view(self, waypoint_ref):
        from src.tools.motion.path_collision import make_candidate_path_collision as CandidatePathCollision
        from src.tools.motion.planning import _plan, _pose_transform
        from scipy.spatial import cKDTree
        p = self._view(waypoint_ref)
        token = 'mv_'+uuid4().hex
        try:
            if p['start'] != self._robot_state():
                raise ValueError('robot moved after waypoint proposal')
            if (self.held_plan is None and self.closed_push and p['purpose'] == 'contact'
                    and p.get('target_points') is None):
                raise ActionPreconditionError('contact_requires_current_target')
            checker = CandidatePathCollision(self.connector, clearance_m=.0005)
            scene, removal = checker.remove_captured_robot(p['scene'])
            if self.held_plan is not None:
                if self.grasp_attachment is None:
                    raise ValueError('measured grasp attachment unavailable')
                current = _pose_transform(self.connector.get_ee_pose())
                relative = current @ np.linalg.inv(self.grasp_attachment)
                held = self.held_plan.target_points @ relative[:3, :3].T + relative[:3, 3]
                distances, _ = cKDTree(held).query(scene)
                scene = scene[distances > .012]
                removal['removed_held_samples'] = int((distances <= .012).sum())
            elif self.closed_push and p['purpose'] == 'contact':
                # Contact was explicitly requested. Permit touching only the
                # current measured target, not all environment geometry.
                distances, _ = cKDTree(p['target_points']).query(scene)
                scene = scene[distances > .002]
                removal['explicit_contact_target_samples'] = int((distances <= .002).sum())
            segments, targets, *_ = self._plan_waypoint(p, scene, checker)
            plan = SimpleNamespace(segments=segments, targets=targets, target_labels=tuple(f'waypoint_step_{i}' for i in range(len(targets)-1))+('waypoint',),
                segment_labels=tuple(f'waypoint_step_{i}' for i in range(len(targets)-1))+('waypoint',), transit_policy=('explicit_camera_down'
                    if p.get('orientation_policy') == 'camera_down' else 'explicit_translation'), high_transit_z_m=None,
                start_joints=json.loads(self._robot_state())['joints'])
            width = self.grasp_jaw_width_m if self.held_plan is not None else 0. if self.closed_push else getattr(self, 'max_gripper_width_m', .08)
            # A waypoint move may start from a pre-existing contact (e.g. an object
            # the hand just pushed); moving away from it is not a new collision.
            if getattr(self, 'waypoint_path_collision_checks', True):
                evidence = checker.check(plan, scene, stop_label='waypoint', jaw_width_m=width,
                                         ignore_initial_contacts=True)
                if evidence['accepted'] and self.held_plan is not None and p['purpose'] != 'contact':
                    evidence['held_object'] = self._check_held_path(checker, plan, scene)
                    evidence['accepted'] = evidence['held_object']['accepted']
            else:
                evidence = dict(accepted=True, checks=[], collision_check_status='disabled_by_configuration')
            evidence['capture_removal'] = removal
            evidence['contact_policy'] = ('waypoint collision sweep disabled by configuration'
                if not getattr(self, 'waypoint_path_collision_checks', True) else
                'agent explicitly permits target/environment contact; robot/self checks retained'
                if p['purpose'] == 'contact' else 'observed scene collision checks')
            accepted = bool(evidence['accepted'])
            if accepted:
                self.view_validations[token] = (waypoint_ref, self._robot_state(), plan)
        except ActionPreconditionError:
            raise
        except Exception as exc:
            from src.core.planning_feedback import public_planning_feedback
            accepted, evidence = False, {
                **public_planning_feedback(getattr(exc, 'planning_feedback', None)),
                'kind': 'planning', 'error': repr(exc)}
        self._record('view_validation', dict(waypoint_ref=waypoint_ref, validation_ref=token,
                     accepted=accepted, evidence=evidence))
        return dict(waypoint_ref=waypoint_ref, validation_ref=token, accepted=accepted,
                    waypoint_path_collision_checks=getattr(self, 'waypoint_path_collision_checks', True),
                    reason_code='accepted' if accepted else 'waypoint_path_rejected',
                    rejection_kind=None if accepted else evidence.get('kind', 'held_object_collision'),
                    validation_feedback={} if accepted else self._validation_feedback(evidence),
                    image_refs=self.observation_image_refs.get(p['observation_id'], []),
                    limitations=('Waypoint path collision checks disabled; planning and execution tracking remain active.'
                        if not getattr(self, 'waypoint_path_collision_checks', True) else
                        'Only measured geometry checked; contact and slip require post-motion observation.'))

    def _check_held_path(self, checker, plan, scene):
        """Sweep the measured rigid attachment using private robot FK only."""
        import mujoco
        from scipy.spatial import cKDTree
        from src.tools.motion.path_collision import path_samples
        from src.tools.motion.planning import _pose_transform
        current = _pose_transform(self.connector.get_ee_pose())
        native = getattr(checker, 'native', None)
        if native is not None:
            capture_hand = current
        else:
            hand = next(g for g, name in checker.names.items() if name == 'gripper0_hand_collision')
            position, rotation = checker.capture_poses[hand]
            capture_hand = np.eye(4)
            capture_hand[:3, :3], capture_hand[:3, 3] = rotation, position
        hand_from_object = np.linalg.inv(capture_hand) @ current @ np.linalg.inv(self.grasp_attachment)
        local = self.held_plan.target_points @ hand_from_object[:3, :3].T + hand_from_object[:3, 3]
        # Add the held radius to every joint's displacement bound.
        radii = checker.radii + float(np.linalg.norm(local, axis=1).max())
        tree = cKDTree(scene)
        count = 0
        for count, (q, label) in enumerate(path_samples(plan, radii, 'waypoint'), 1):
            checker.data.qpos[checker.addresses] = q
            mujoco.mj_kinematics(checker.model, checker.data)
            if native is not None:
                pose = native.body_matrix('base_link')
                position, rotation = pose[:3, 3], pose[:3, :3]
            else:
                position, rotation = checker._pose(hand)
            moved = local @ rotation.T + position
            distances, _ = tree.query(moved, distance_upper_bound=.002)
            if np.isfinite(distances).any():
                return dict(accepted=False, kind='held_object_scene', segment=label, sample_count=count)
        return dict(accepted=True, sample_count=count,
                    limitation='measured held surface with rigid no-slip assumption; unknown interior/occlusions absent')

    def _plan_waypoint(self, proposal, scene, checker):
        from src.tools.motion.planning import _plan
        return _plan(self.connector, (proposal['pose'],), scene, self.motion_config, None)

    def _execute_waypoint(self, plan):
        from src.tools.motion.planning import _execute_checked
        return _execute_checked(self.connector, plan.segments, plan.targets,
                                collision_checks_enabled=self.motion_config.collision_checks_enabled)

    def execute_view(self, waypoint_ref, validation_ref):
        from src.tools.motion.planning import _execute_checked, _pose_transform
        p = self._view(waypoint_ref)
        ref, state, plan = self.view_validations.pop(validation_ref)
        if ref != waypoint_ref or state != self._robot_state():
            raise ValueError('waypoint validation stale or mismatched')
        if self.plan_only:
            return dict(view_status='not_executed', status='unknown', observation_id=self.latest_observation_id)
        before = _pose_transform(self.connector.get_ee_pose())
        self.epoch += 1
        self.validations.clear()
        self.view_validations.clear()
        self.latest_fused_ref = self.latest_observation_id = self.point_adapter.latest = None
        status, error = 'achieved', None
        try:
            if self.recorder:
                self.recorder.register_plan(plan, kind=p['purpose'], point_ref=waypoint_ref)
            with self.recorder.active('execute_view', point_ref=waypoint_ref) if self.recorder else nullcontext():
                if self.held_plan is not None or self.closed_push:
                    self.connector.set_gripper(0.)
                self._execute_waypoint(plan)
        except Exception as exc:
            status, error = 'requested_view_failed', repr(exc)
        actual = _pose_transform(self.connector.get_ee_pose())
        distance = float(np.linalg.norm(actual[:3, 3]-p['pose'][:3, 3]))
        from src.tools.motion.tolerances import cartesian_tolerances, requires_profile_tracking
        position_limit, angle_limit = cartesian_tolerances(self.connector, position=.02)
        if distance > position_limit:
            status = 'requested_view_failed'
        orientation_feedback = {}
        if p.get('orientation_policy') == 'camera_down' or requires_profile_tracking(self.connector):
            cosine = float(np.clip((np.trace(actual[:3, :3].T @ p['pose'][:3, :3])-1.)/2., -1., 1.))
            angle = float(np.arccos(cosine))
            orientation_feedback = dict(orientation_error_rad=angle, endpoint_orientation_tolerance_rad=angle_limit)
            if angle > angle_limit:
                status = 'requested_view_failed'
        observation = self.observe()
        # No automatic resegmentation or hidden direction search. Pointer can
        # confirm the target in these fresh RGBs before generating new grasps.
        result = dict(**observation, view_status=status, purpose=p['purpose'],
            translation_m=float(np.linalg.norm(actual[:3, 3]-before[:3, 3])), endpoint_error_m=distance,
            endpoint_position_tolerance_m=position_limit,
            holding_command=self.held_plan is not None or self.closed_push,
            target_update=dict(target_ref=p['target_ref'], tracking_status='needs_pointer'),
            observation_improvement='unassessed; agent must compare fresh RGB',
            hold_status='unverified; closed command is not proof of holding', **orientation_feedback)
        self._record('view_executed', dict(**result, waypoint_ref=waypoint_ref,
                     validation_ref=validation_ref, error=error))
        return result
