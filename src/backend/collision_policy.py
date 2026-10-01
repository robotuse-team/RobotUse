"""Placement refinement and explicit simulation collision fallback.

Normal candidates are tried first. Fallback creates a NEW candidate/route after
a refinement attempt, records the ignored scene check, and retains state, IK,
robot self-collision and execution tracking checks. Raw candidates are immutable.
"""
from copy import deepcopy
from dataclasses import replace
from uuid import uuid4
import numpy as np

from src.backend.robot import IntentBackend
from src.backend.intent import IntentOrchestrator
from src.backend.controller import ARGUMENTS, BoundaryError

ARGUMENTS.update(adjust_place=('candidate_ref', 'dx_m', 'dy_m', 'dz_m',
                              'roll_deg', 'pitch_deg', 'yaw_deg'),
                 relax_candidate=('candidate_ref',))


def translated_place(transform, delta):
    values = np.asarray(delta, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all() or np.max(np.abs(values)) > .15:
        raise ValueError('placement translation must be finite and within 0.15 m per axis')
    result = np.array(transform, dtype=float, copy=True)
    result[:3, 3] += values
    return result


class RelaxedIntentBackend(IntentBackend):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.motion_config.collision_checks_enabled:
            raise ValueError('collision fallback requires the explicit simulation collision-disabled configuration')
        self.place_diagnostics = {}
        self.place_entries = {}
        self.refined_groups = set()
        self.normal_groups = set()
        self.relaxed_refs = set()
        self.place_translation = {}

    def _group(self, ref):
        if ref in self.place_predictions:
            value = self._place_candidate(ref)
            return ('place', self.epoch, value[1])
        _, _, point = self._candidate_context(ref)
        return ('grasp', self.epoch, point)

    def grasp_candidates(self, point_ref, preferred_direction=None, grasp_type=None, batch_size=None):
        result = super().grasp_candidates(point_ref, preferred_direction=preferred_direction, grasp_type=grasp_type,
                                         **({'batch_size': batch_size} if batch_size is not None else {}))
        if result['candidates']:
            self.normal_groups.add(('grasp', self.epoch, point_ref))
        return result

    def adjust_grasp(self, **args):
        group = self._group(args['candidate_ref'])
        result = super().adjust_grasp(**args)
        self.refined_groups.add(group)
        if result.get('accepted'):
            self.normal_groups.add(group)
        return result

    def place_candidates(self, **args):
        result = super().place_candidates(**args)
        destination_ref = args['destination_ref']
        group = ('place', self.epoch, destination_ref)
        if result['candidates']:
            self.normal_groups.add(group)
            for item in result['candidates']:
                self.place_entries[item['candidate_ref']] = dict(item, executable=True)
            return result
        # Expose a few rejected predictions as REFINE-ONLY references. The
        # inherited pipeline has already checked the complete original pool.
        predictions, _ = self._placement_pool(self.grasped_point_ref, destination_ref,
                                             self.held_plan.target_points)
        from src.tools.place.release_validation import check_release_geometry
        scene = self._placement_scene(self.held_plan, self.destinations[destination_ref])
        checked = [(p, check_release_geometry(self.held_plan.target_points, scene, p.transform,
                    self.grasp_attachment, self.held_plan.grasp_to_ee,
                    jaw_width_m=self.grasp_jaw_width_m,
                    release_clearance_m=self.motion_config.release_clearance_m,
                    mesh_source=getattr(self, 'gripper_assets', None))) for p in predictions]
        checked.sort(key=lambda item: (item[1].get('colliding_points', 0), item[0].source_index))
        diagnostics = [self._publish_place(destination_ref, p, None, g) for p, g in checked[:4]]
        return {**result, 'refinement_candidates': diagnostics}

    def _publish_place(self, destination_ref, prediction, plan, geometry, *, origin=None, relaxed=False):
        from src.tools.place.planning import preview_anyplace
        from src.tools.place.visualization import render_placement
        ref = 'place_' + uuid4().hex
        self.place_predictions[ref] = (self.epoch, destination_ref, prediction, self._robot_state(), plan)
        preview = plan or preview_anyplace(self.connector, relative_transform=prediction.transform,
                          closed_ee_pose=self.grasp_attachment, config=self.motion_config)
        preview.jaw_width_m = self.grasp_jaw_width_m
        preview.validation_label = ('RELAXED / scene collision ignored' if relaxed else
                                    'REJECTED / refine only' if plan is None else 'validated refined candidate')
        paths = render_placement(self.destinations[destination_ref], self.held_plan.target_points,
            prediction.transform, preview, self.output_dir / ref, grasp_to_ee=self.held_plan.grasp_to_ee,
            mesh_source=getattr(self, 'gripper_assets', None))
        item = dict(candidate_ref=ref, image_refs=[self.images.add(p) for p in paths],
                    executable=plan is not None, collision_relaxed=relaxed,
                    reason_code=geometry.get('reason_code', 'accepted' if plan else 'route_rejected'))
        if plan is None:
            self.place_diagnostics[ref] = deepcopy(geometry)
        if relaxed:
            self.relaxed_refs.add(ref)
        self.place_entries[ref] = item
        self._record('placement_refinement_candidate', {**item, 'origin_candidate_ref': origin,
            'destination_ref': destination_ref, 'transform': prediction.transform,
            'scene_check': geometry, 'route_generated': plan is not None})
        return item

    def inspect_place_candidate(self, candidate_ref):
        self._place_candidate(candidate_ref)
        return deepcopy(self.place_entries.get(candidate_ref) or
                        super().inspect_place_candidate(candidate_ref))

    def _plan_place(self, destination_ref, prediction, *, relaxed=False):
        from src.tools.place.release_validation import check_release_geometry, release_goal_ik
        from src.tools.place.planning import plan_anyplace
        from src.tools.motion.planning import MotionPlanningError
        scene = self._placement_scene(self.held_plan, self.destinations[destination_ref])
        geometry = check_release_geometry(self.held_plan.target_points,
            scene, prediction.transform,
            self.grasp_attachment, self.held_plan.grasp_to_ee, jaw_width_m=self.grasp_jaw_width_m,
            release_clearance_m=self.motion_config.release_clearance_m,
            mesh_source=getattr(self, 'gripper_assets', None))
        can_ignore = geometry.get('reason_code') in ('gripper_scene_collision', 'payload_scene_collision')
        if not geometry['compatible'] and not (relaxed and can_ignore):
            return None, geometry
        ik = release_goal_ik(self.connector.ik, [geometry['release_ee_pose']],
                             jaw_width_m=self.grasp_jaw_width_m)[0]
        if not ik['accepted']:
            return None, {**geometry, 'reason_code': 'release_ik_or_self_collision_rejected', 'ik': ik}
        try:
            plan = plan_anyplace(self.connector, grasp_plan=self.held_plan,
                relative_transform=prediction.transform, closed_ee_pose=self.grasp_attachment,
                config=self.motion_config, jaw_width_m=self.grasp_jaw_width_m)
        except MotionPlanningError as exc:
            return None, {**geometry, 'reason_code': 'route_generation_failed', 'error': str(exc)}
        return plan, geometry

    def adjust_place(self, candidate_ref, dx_m, dy_m, dz_m, roll_deg, pitch_deg, yaw_deg):
        from src.tools.pose_editor.refinement import checked_adjustment, refined_object_transform, tcp_offset_from, rotation_limits
        group = self._group(candidate_ref)
        _, destination, prediction, _, _ = self._place_candidate(candidate_ref)
        step_limit, cumulative_limit = rotation_limits('adjust_place',
            getattr(self, 'object_cloud_policy', 'source_view'))
        angles, cumulative = checked_adjustment((roll_deg, pitch_deg, yaw_deg),
            self.place_adjustments.get(candidate_ref, (0., 0., 0.)),
            step_limit_deg=step_limit, cumulative_limit_deg=cumulative_limit)
        total = np.asarray(self.place_translation.get(candidate_ref, (0., 0., 0.))) + [dx_m, dy_m, dz_m]
        if np.max(np.abs(total)) > .3:
            raise ValueError('cumulative placement displacement exceeds 0.3 m')
        transform = refined_object_transform(prediction.transform, angles,
            closed_ee=self.grasp_attachment, grasp_to_ee=self.held_plan.grasp_to_ee,
            release_clearance_m=self.motion_config.release_clearance_m,
            tcp_offset_z_m=getattr(getattr(self, 'gripper_assets', None), 'jaw_center_offset_m',
                                  tcp_offset_from(self.held_plan.grasp_to_ee)))
        transform = translated_place(transform, (dx_m, dy_m, dz_m))
        derived = replace(prediction, transform=transform)
        plan, geometry = self._plan_place(destination, derived)
        self.refined_groups.add(group)
        item = self._publish_place(destination, derived, plan, geometry, origin=candidate_ref)
        self.place_adjustments[item['candidate_ref']] = cumulative
        self.place_translation[item['candidate_ref']] = total.tolist()
        if plan is not None:
            self.normal_groups.add(group)
        return dict(accepted=plan is not None, **item)

    def relax_candidate(self, candidate_ref):
        group = self._group(candidate_ref)
        if group in self.normal_groups:
            return dict(accepted=False, reason_code='normal_candidate_available')
        if group not in self.refined_groups:
            return dict(accepted=False, reason_code='try_refinement_before_collision_fallback')
        if group[0] == 'place':
            _, destination, prediction, _, _ = self._place_candidate(candidate_ref)
            plan, geometry = self._plan_place(destination, prediction, relaxed=True)
            item = self._publish_place(destination, prediction, plan, geometry,
                                       origin=candidate_ref, relaxed=plan is not None)
        else:
            from src.tools.motion.planning import MotionPlanningError
            from src.backend.robot_base import render_candidate
            geometry, prediction, point = self._candidate_context(candidate_ref)
            ref = 'g_' + uuid4().hex
            try:
                plan, self_check = self._plan_relaxed_grasp(geometry, prediction, point, ref)
            except MotionPlanningError:
                return dict(accepted=False, reason_code='route_generation_failed')
            if not self_check['accepted']:
                return dict(accepted=False, reason_code='robot_self_collision_or_path_rejected')
            self.candidates[ref] = (self.epoch, geometry, prediction)
            self.candidate_point_refs[ref] = point
            self.candidate_sources[ref] = geometry.view_id
            self.candidate_routes[ref] = (self._robot_state(), plan)
            path = self.output_dir / (ref + '.png')
            render_candidate(geometry, prediction, ref, path, expected_open_width_m=plan.open_width_m,
                max_width_m=getattr(self, 'max_gripper_width_m', .08),
                **getattr(self, 'gripper_render_options', {}))
            self.candidate_open_widths[ref] = plan.open_width_m
            self.relaxed_refs.add(ref)
            item = dict(candidate_ref=ref, executable=True, collision_relaxed=True,
                         image_refs=[self.images.add(path)])
        self._record('collision_fallback', dict(origin_candidate_ref=candidate_ref,
            group=group, **item, policy='scene geometry ignored; IK, robot self checks and tracking retained'))
        return dict(accepted=item['executable'], **item)

    def _plan_relaxed_grasp(self, geometry, prediction, point, ref):
        from src.tools.motion.planning import plan_grasp
        plan = plan_grasp(self.connector, grasp_transform=prediction.pose,
            max_width_m=getattr(self, 'max_gripper_width_m', .08),
            target_points=self._planning_target_points(geometry, point), obstacle_points=self._point(point).scene_points,
            grasp_to_ee=self.grasp_to_ee, frame='connector_base', config=self.motion_config,
            **self._grasp_options(prediction.pose, geometry, prediction, ref))
        checker, scene = self._candidate_path_scene(point)
        return plan, checker.check(plan, scene, stop_label='grasp',
                                   jaw_width_m=plan.open_width_m, skip_scene=True)

    def validate_place(self, candidate_ref):
        value = self._place_candidate(candidate_ref)
        if value[4] is None:
            return dict(candidate_ref=candidate_ref, accepted=False, reason_code='refine_only_no_route')
        return super().validate_place(candidate_ref)


class RelaxedIntentOrchestrator(IntentOrchestrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Advertise exactly the backend's version-specific limits to the model.
        self.factory.object_cloud_policy = getattr(self.backend, 'object_cloud_policy', 'source_view')

    def _tool_argument(self, tool, key, value):
        if (tool == 'adjust_place' and key in ('roll_deg', 'pitch_deg', 'yaw_deg')
                and getattr(self.backend, 'object_cloud_policy', 'source_view') == 'fused'):
            from src.tools.pose_editor.refinement import rotation_limits
            limit, _ = rotation_limits(tool, self.backend.object_cloud_policy)
            if type(value) not in (int, float) or not np.isfinite(value) or abs(value) > limit:
                raise BoundaryError(f'placement rotation must be finite and within {limit:g} degrees per axis')
            return float(value)
        return super()._tool_argument(tool, key, value)

    def _tools(self, role, task):
        if role == 'refiner' and task.get('placement_refinement'):
            return ('inspect_place_candidate', 'adjust_place', 'relax_candidate', 'finish')
        tools = super()._tools(role, task)
        return (*tools[:-1], 'relax_candidate', tools[-1]) if role == 'refiner' else tools

    def _prompt(self, role):
        prompt = super()._prompt(role)
        if role in ('prime', 'place', 'refiner'):
            prompt += (' Placement candidates, including refine-only rejected candidates, can be sent to '
                'delegate_refiner. Refiner can adjust_place: XYZ are BASE-frame metres (up to 0.15 per step), '
                'angles are local gripper degrees. Raising the release pose enables a deliberate drop. '
                'The adjustment produces a new preview and replanned lift/transit/release/retreat route. '
                'If the whole normal pool has no executable candidate and a refinement has been attempted, '
                'Refiner may call relax_candidate on the most promising supplied grasp OR place candidate. '
                'For grasp fallback, relax_candidate keeps the selected pose and replans its route with '
                'scene collision relaxed; it does not regenerate grasps or consume generation budget. '
                'This simulation collision fallback permits scene contact; fallback retains IK, self checks and '
                'tracking. A fallback failure is not executable. Inspect then return a passing candidate. '
                'Changing placement instruction alone does not resample. For several objects, continue '
                'selection and pick/place after each release until the original whole instruction is satisfied.')
            if getattr(self.backend, 'object_cloud_policy', 'source_view') == 'fused':
                prompt += (' For placement refinement, choose the rotation axis from the displayed local '
                    'gripper axes and use large angles (e.g. 90 or 180 degrees) when needed. '
                    'Inspect and validate the resulting object pose.')
        return prompt

    def _finish(self, role, args, scope):
        ref = args.get('candidate_ref')
        if ref in self.backend.place_diagnostics and role in ('place', 'refiner'):
            if scope['candidates'].get(ref) != self._epoch:
                raise BoundaryError('current supplied placement required')
            return dict(status='failed', reason=args.get('reason', 'Placement requires refinement.'),
                        related_refs={'candidate_ref': ref}, evidence_image_refs=args.get('evidence_image_refs', []))
        return super()._finish(role, args, scope)

    def _intent_dispatch(self, role, tool, args, scope, task, sid):
        ref = args.get('candidate_ref')
        if tool == 'delegate_refiner':
            if scope['candidates'].get(ref) != self._epoch and ref not in self._diagnostic_candidates:
                raise BoundaryError('current supplied candidate required')
            placement = ref in self.backend.place_predictions
            self.backend._group(ref)
            output = self._delegate('refiner', {**args, 'refinement_task': True,
                'placement_refinement': placement}, scope, sid)
            selected = output['result'].get('candidate_ref')
            if selected and not placement:
                scope.setdefault('grasp_decisions', {})[selected] = 'accepted'
            return output
        if role == 'place' and tool == 'place_candidates':
            if any(args[k] != task[k] for k in ('destination_ref','hold_assessment','destination_assessment')):
                raise BoundaryError('use delegated current destination')
            if 'place_bundle' not in scope:
                scope['place_bundle'] = self.backend.place_candidates(**args)
            result = scope['place_bundle']
            for candidate in [*result['candidates'], *result.get('refinement_candidates', [])]:
                scope['candidates'][candidate['candidate_ref']] = self._epoch
            if self._place_generation_stack:
                self._place_generation_stack[-1].update(result.get('candidate_generation', {}))
            return result
        if tool == 'delegate_place':
            result = super()._intent_dispatch(role, tool, args, scope, task, sid)
            refs = [r for r, v in self.backend.place_predictions.items()
                    if v[0] == self._epoch and v[1] == args['destination_ref'] and r in self.backend.place_diagnostics]
            result['refinement_candidates'] = [self.backend.inspect_place_candidate(r) for r in refs[-4:]]
            for r in refs[-4:]:
                scope['candidates'][r] = self._epoch
            return result
        if role == 'refiner' and tool in ('inspect_place_candidate', 'adjust_place', 'relax_candidate'):
            if scope['candidates'].get(ref) != self._epoch:
                raise BoundaryError('candidate outside Refiner session')
            output = getattr(self.backend, tool)(**args)
            new = output.get('candidate_ref')
            if new:
                scope['candidates'][new] = self._epoch
                scope['inspected_candidates'][new] = self._epoch
            return output
        if tool == 'inspect_place_candidate' and ref in self.backend.place_diagnostics:
            if scope['candidates'].get(ref) != self._epoch:
                raise BoundaryError('candidate outside Place session')
            scope['inspected_candidates'][ref] = self._epoch
            return self.backend.inspect_place_candidate(ref)
        return super()._intent_dispatch(role, tool, args, scope, task, sid)
