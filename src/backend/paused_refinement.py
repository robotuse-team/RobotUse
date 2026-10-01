"""Opt-in paused refinement at the measured pregrasp and release goals.

The executor stops with the arm resting at the measured pregrasp (gripper open)
or at the measured release goal (object still held), captures fresh RGB-D, and
lets Refiner make a small bounded correction from THAT pose before the gripper
closes or opens. A corrected descent is replanned from the current joints; the
original candidate identity is not reused after motion. Nothing here relaxes
the arrival thresholds of the checkpoint layer.

ObserveFirstIntentOrchestrator additionally requires one downward observation
above the target before the first grasp request of an object.
"""
from copy import deepcopy
from dataclasses import replace
from uuid import uuid4
import numpy as np

from src.core.action_feedback import ActionPreconditionError
from src.backend.checkpoints import CheckpointIntentBackend, CheckpointIntentOrchestrator
from src.tools.motion.planning import MotionPlanningError
from src.backend.controller import ARGUMENTS, BoundaryError, _text

DELTAS = ('dx_mm', 'dy_mm', 'dz_mm', 'roll_deg', 'pitch_deg', 'yaw_deg')
ARGUMENTS['nudge_grasp'] = DELTAS
ARGUMENTS['nudge_place'] = ('dx_mm', 'dy_mm', 'dz_mm')
INFLIGHT_TURN_LIMIT = 6


class GraspAbortedBeforeClose(MotionPlanningError):
    """Refiner judged the paused pregrasp pose unsuitable; the gripper never closed."""


def _public_delta(step, total):
    return dict(step=dict(zip(DELTAS, [float(v) for v in step])),
                cumulative=dict(zip(DELTAS, [float(v) for v in total])))


class PauseRefineIntentBackend(CheckpointIntentBackend):
    pause_refiner = None  # orchestrator callback(stage, context) -> {'status': 'continue'|'abort', ...}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._inflight = None
        self._inflight_place = None
        self._inflight_context = None

    # ---- grasp -----------------------------------------------------------
    def execute_grasp(self, candidate_ref, validation_ref):
        self._inflight = None
        point_ref = self.candidate_point_refs.get(candidate_ref)
        context = dict(candidate_ref=candidate_ref, point_ref=point_ref,
                       relaxed=candidate_ref in getattr(self, 'relaxed_refs', ()))
        try:
            # Captured now: after motion the point reference is stale by design.
            context['checker_scene'] = self._candidate_path_scene(point_ref) if point_ref else None
        except Exception:
            context['checker_scene'] = None
        self._inflight_context = context
        validation = self.validations.get(validation_ref)
        if validation is not None:
            validation[3].pregrasp_pause = self._pregrasp_pause
        result = super().execute_grasp(candidate_ref, validation_ref)
        flight = self._inflight
        if flight is not None:
            if result.get('status') == 'succeeded' and flight.get('replacement') is not None:
                # The executed (corrected) plan is the held plan; the original
                # never closed and carries no executed-grasp state.
                self.held_plan = flight['replacement']
            if flight.get('refinement') is not None:
                result['inflight_refinement'] = deepcopy(flight['refinement'])
        return result

    def _execution_failure(self, operation, execution_ref, exc):
        result = super()._execution_failure(operation, execution_ref, exc)
        if isinstance(exc, GraspAbortedBeforeClose):
            result['reason_code'] = 'grasp_aborted_before_close'
            if self._inflight and self._inflight.get('refinement') is not None:
                result['inflight_refinement'] = deepcopy(self._inflight['refinement'])
        return result

    def _inflight_preview(self, observation, hand_pose, open_width_m, name):
        try:
            from src.tools.pose_editor.refinement import preview_refined_pose, tcp_offset_from
            frames = self.point_adapter.frames[observation['observation_id']]
            paths = preview_refined_pose(frames, hand_pose, self.output_dir / name,
                                         tcp_offset_z_m=getattr(getattr(self, 'gripper_assets', None), 'jaw_center_offset_m', tcp_offset_from(self.grasp_to_ee)),
                                         expected_open_width_m=open_width_m, mesh_source=getattr(self, 'gripper_assets', None))
            return [self.images.add(p) for p in paths]
        except Exception as exc:
            self._record('inflight_preview_error', dict(name=name, error=repr(exc)))
            return []

    def _pregrasp_pause(self, plan):
        observation = getattr(self, 'last_checkpoint_observation', None) or self.observe()
        context = self._inflight_context or {}
        self._inflight = dict(original=plan, current_pose=np.array(plan.grasp_transform, dtype=float),
                              replacement=None, adjustment=(0., 0., 0.), translation=(0., 0., 0.),
                              attempts=[], observation=observation, refinement=None)
        previews = self._inflight_preview(observation, plan.grasp_transform, plan.open_width_m,
                                          'inflight_' + uuid4().hex)
        public = dict(stage='pregrasp', observation=observation, image_refs=previews,
                      origin_candidate_ref=context.get('candidate_ref'), open_width_m=float(plan.open_width_m),
                      checkpoint={k: v for k, v in (getattr(self, 'last_checkpoint_evidence', None) or {}).items()
                                  if k in ('position_error_m', 'orientation_error_rad', 'stage')})
        self._record('inflight_pause', dict(**{k: v for k, v in public.items() if k != 'observation'},
                                            observation_id=observation.get('observation_id')))
        if self.pause_refiner is None:
            return None
        decision = self.pause_refiner('pregrasp', public)
        flight = self._inflight
        flight['refinement'] = dict(stage='pregrasp', decision=deepcopy(decision),
                                    attempts=deepcopy(flight['attempts']),
                                    corrected=flight['replacement'] is not None,
                                    cumulative=_public_delta((0., 0., 0.), (*flight['translation'], *flight['adjustment']))['cumulative'])
        self._record('inflight_refinement', flight['refinement'])
        if decision.get('status') == 'abort':
            exc = GraspAbortedBeforeClose('Refiner judged the paused pregrasp pose unsuitable; gripper not closed')
            exc.evidence = dict(execution_stage='pregrasp', motion_completed=False,
                                refinement=deepcopy(flight['refinement']))
            raise exc
        return flight['replacement']

    def nudge_inflight_grasp(self, dx_mm=0., dy_mm=0., dz_mm=0., roll_deg=0., pitch_deg=0., yaw_deg=0.):
        from src.tools.pose_editor.refinement import checked_adjustment, refined_grasp_pose, tcp_offset_from
        from src.tools.pose_editor.geometry import checked_translation
        from src.tools.motion.planning import plan_grasp
        flight = self._inflight
        if flight is None:
            raise ValueError('no paused grasp execution')
        try:
            step, total = checked_adjustment((roll_deg, pitch_deg, yaw_deg), flight['adjustment'])
            translation, total_translation = checked_translation((dx_mm, dy_mm, dz_mm), flight['translation'], step_limit_mm=30, cumulative_limit_mm=90)
        except ValueError:
            return dict(accepted=False, reason_code='adjustment_budget_exhausted')
        base = flight['current_pose']
        pose = refined_grasp_pose(base, step, tcp_offset_z_m=getattr(getattr(self, 'gripper_assets', None), 'jaw_center_offset_m', tcp_offset_from(self.grasp_to_ee)))
        pose[:3, 3] += base[:3, :3] @ (np.asarray(translation, dtype=float) / 1000.)
        original = flight['original']
        context = self._inflight_context or {}
        attempt = dict(**_public_delta((*translation, *step), (*total_translation, *total)))
        try:
            opening = original.open_width_m
            policy = getattr(self, 'grasp_opening_policy', None)
            if policy is not None:
                aperture = policy(pose, original.target_points)
                opening = aperture['open_width_m']
                attempt['grasp_opening'] = aperture
            plan = plan_grasp(self.connector, grasp_transform=pose, target_points=original.target_points,
                max_width_m=getattr(self, 'max_gripper_width_m', .08),
                              obstacle_points=original.obstacle_points, grasp_to_ee=self.grasp_to_ee,
                              frame='connector_base', config=replace(self.motion_config, transit_policy='legacy'),
                              lift_after_grasp='lift' in original.target_labels,
                              open_width_m=opening)
            checker_scene = context.get('checker_scene')
            if checker_scene is None:
                evidence = dict(accepted=True, kind='unchecked',
                                note='no captured scene for this candidate; self-collision only through planner')
            else:
                checker, scene = checker_scene
                evidence = checker.check(plan, scene, stop_label='grasp', jaw_width_m=plan.open_width_m,
                                         skip_scene=bool(context.get('relaxed')))
        except MotionPlanningError as exc:
            plan, evidence = None, dict(accepted=False, kind='planning', error=str(exc),
                                        **getattr(exc, 'planning_feedback', {}))
        attempt['accepted'] = bool(evidence.get('accepted'))
        attempt['path_check'] = self._validation_feedback(evidence)
        flight['attempts'].append(attempt)
        self._record('inflight_nudge', dict(stage='pregrasp', **attempt, grasp_transform=pose.tolist()))
        if not attempt['accepted']:
            return {**attempt, 'accepted': False, 'reason_code': 'nudged_path_rejected'}
        flight.update(replacement=plan, current_pose=pose, adjustment=total, translation=total_translation)
        previews = self._inflight_preview(flight['observation'], pose, plan.open_width_m,
                                          'inflight_' + uuid4().hex)
        return {**attempt, 'accepted': True, 'image_refs': previews}

    # ---- place -----------------------------------------------------------
    def execute_place_candidate(self, candidate_ref, validation_ref):
        self._inflight_place = None
        try:
            plan = self._place_candidate(candidate_ref)[4]
        except Exception:
            plan = None
        if plan is not None:
            plan.release_pause = self._release_pause
        result = super().execute_place_candidate(candidate_ref, validation_ref)
        flight = self._inflight_place
        if flight is not None and flight.get('refinement') is not None:
            result['inflight_refinement'] = deepcopy(flight['refinement'])
        return result

    def _release_pause(self, plan):
        from src.tools.motion.planning import _pose_transform
        observation = self.observe()
        release = _pose_transform(plan.targets[2])
        self._inflight_place = dict(plan=plan, release=release, current=release.copy(),
                                    translation=(0., 0., 0.), replacement=None, attempts=[],
                                    observation=observation, refinement=None)
        hand = release @ np.linalg.inv(np.asarray(self.grasp_to_ee, dtype=float))
        width = self.grasp_jaw_width_m if getattr(self, 'grasp_jaw_width_m', None) is not None else getattr(self, 'max_gripper_width_m', .08)
        previews = self._inflight_preview(observation, hand, width, 'inflight_' + uuid4().hex)
        public = dict(stage='release', observation=observation, image_refs=previews,
                      release_clearance_m=float(getattr(plan, 'release_clearance_m', 0.) or 0.))
        self._record('inflight_pause', dict(**{k: v for k, v in public.items() if k != 'observation'},
                                            observation_id=observation.get('observation_id')))
        if self.pause_refiner is None:
            return None
        decision = self.pause_refiner('release', public)
        flight = self._inflight_place
        flight['refinement'] = dict(stage='release', decision=deepcopy(decision),
                                    attempts=deepcopy(flight['attempts']),
                                    corrected=flight['replacement'] is not None,
                                    cumulative_translation_mm=[float(v) for v in flight['translation']])
        self._record('inflight_refinement', flight['refinement'])
        # An abort keeps the object held at the release goal without opening;
        # the tracking result then reports an incomplete release to Prime.
        if decision.get('status') == 'abort':
            return dict(segments=(), targets=(), retreat=None, aborted=True)
        return flight['replacement']

    def nudge_inflight_place(self, dx_mm=0., dy_mm=0., dz_mm=0.):
        from src.tools.pose_editor.geometry import checked_translation
        from src.tools.motion.planning import _plan, _pose_transform
        from src.tools.motion.robot_state import _trajectory_end
        flight = self._inflight_place
        if flight is None:
            raise ValueError('no paused place execution')
        try:
            translation, total = checked_translation((dx_mm, dy_mm, dz_mm), flight['translation'], step_limit_mm=30, cumulative_limit_mm=90)
        except ValueError:
            return dict(accepted=False, reason_code='adjustment_budget_exhausted')
        plan = flight['plan']
        target = flight['current'].copy()
        target[:3, 3] += np.asarray(translation, dtype=float) / 1000.  # base-frame shift
        retreat = _pose_transform(plan.targets[3]).copy()
        retreat[:3, 3] += target[:3, 3] - flight['release'][:3, 3]
        attempt = dict(step_mm=[float(v) for v in translation], cumulative_mm=[float(v) for v in total])
        try:
            segments, poses, *_ = _plan(self.connector, (target, retreat), np.empty((0, 3)),
                                        self.motion_config, None)
            from src.tools.place.self_collision import check_release_self_collision
            width = self.grasp_jaw_width_m if getattr(self, 'grasp_jaw_width_m', None) is not None else getattr(self, 'max_gripper_width_m', .08)
            native = getattr(self.connector.ik, 'check_release_self_collision', None)
            if callable(native):
                self_check = native([_trajectory_end(segments[0])], jaw_width_m=width)[0]
            else:
                self_check = check_release_self_collision(self.connector.ik._robot_file,
                    [_trajectory_end(segments[0])], jaw_width_m=width)[0]
            accepted = bool(self_check.get('accepted'))
            attempt['self_collision'] = {'accepted': accepted}
        except Exception as exc:
            accepted, segments, poses = False, None, None
            attempt['error'] = type(exc).__name__
        attempt['accepted'] = accepted
        flight['attempts'].append(attempt)
        self._record('inflight_nudge', dict(stage='release', **attempt, release_ee=target.tolist()))
        if not accepted:
            return {**attempt, 'accepted': False, 'reason_code': 'nudged_release_rejected'}
        flight['replacement'] = dict(segments=(segments[0],), targets=(poses[0],),
                                     retreat=((segments[1],), (poses[1],)), shift_mm=attempt['cumulative_mm'])
        flight.update(current=target, translation=total)
        hand = target @ np.linalg.inv(np.asarray(self.grasp_to_ee, dtype=float))
        width = self.grasp_jaw_width_m if getattr(self, 'grasp_jaw_width_m', None) is not None else getattr(self, 'max_gripper_width_m', .08)
        previews = self._inflight_preview(flight['observation'], hand, width, 'inflight_' + uuid4().hex)
        return {**attempt, 'accepted': True, 'image_refs': previews}


class PauseRefineIntentOrchestrator(CheckpointIntentOrchestrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inflight_call = None
        self.backend.pause_refiner = self._inflight_refinement

    def _tool_argument(self, tool, key, value):
        if tool in ('nudge_grasp', 'nudge_place') and key in ('dx_mm', 'dy_mm', 'dz_mm'):
            from src.tools.pose_editor.geometry import checked_translation
            try:
                checked_translation((value, 0., 0.), step_limit_mm=30)
            except ValueError as exc:
                raise BoundaryError(str(exc)) from None
            return float(value)
        return super()._tool_argument(tool, key, value)

    def _tools(self, role, task):
        if role == 'refiner' and task.get('inflight_refinement'):
            return ('nudge_grasp' if task.get('stage') == 'pregrasp' else 'nudge_place', 'finish')
        return super()._tools(role, task)

    def _turn_limit(self, role, task):
        limit = super()._turn_limit(role, task)
        return min(limit, INFLIGHT_TURN_LIMIT) if task.get('inflight_refinement') else limit

    def _prompt(self, role):
        prompt = super()._prompt(role)
        if role == 'refiner':
            prompt += (
                ' PAUSED EXECUTION MODE (task has inflight_refinement): the arm rests at the measured '
                'pregrasp with the gripper open, or at the measured release goal while '
                'still holding the object. Supplied images are the fresh front/wrist RGB from this pose plus '
                'a preview of the pending gripper pose. nudge_grasp(dx_mm, dy_mm, dz_mm, roll_deg, pitch_deg, '
                'yaw_deg) or nudge_place(dx_mm, dy_mm, dz_mm) applies one small correction; supply every field, 0 for '
                'unchanged axes (translation at most 30 mm per axis per step and 90 mm cumulative; '
                'rotation at most 10 degrees per axis per step and 30 degrees cumulative); grasp offsets are in the gripper local frame, place '
                'offsets in robot base XYZ. An accepted nudge replans the remaining motion from the current '
                'pose and returns a new preview; a rejected nudge keeps the previous pose. finish '
                'status=continue closes or releases with the current pose; finish status=abort stops without '
                'closing or releasing when the pose is clearly wrong (wrong object, fingers would strike a '
                'neighbour, release clearly outside the destination). Prefer continue for small imperfections. '
                'If the current pose is unsuitable but the target is identifiable and a useful correction within '
                'the allowed bounds is visible, use nudge_grasp or nudge_place and assess the new preview. '
                'Abort if correction is unlikely to resolve the problem or the target cannot be identified. '
                'An unsuitable current pose alone does not establish that bounded correction cannot help. '
                'No candidate_ref is needed in this mode.')
        if role in ('grasp', 'place', 'prime'):
            prompt += (
                ' Execution pauses at the measured pregrasp and release goals for a brief Refiner check '
                'from fresh RGB; reason_code grasp_aborted_before_close means the gripper never closed '
                'and Prime may request another view or grasp.')
        if role == 'grasp' and getattr(self.backend, 'downward_pool', 0):
            prompt += (
                ' Candidates carry downward_angle_deg (0 = approaching straight down). Prefer the smaller '
                'angle when candidates are otherwise comparable; in a cluttered scene a top-down approach '
                'keeps the fingers away from neighbours. A side approach is acceptable only when the object '
                'shape requires it.')
        return prompt

    def _skill_call(self, tool, args, scope, task, sid, owner):
        if tool in ('execute_grasp', 'execute_place_candidate'):
            self._inflight_call = (scope, task, sid, owner)
            try:
                return super()._skill_call(tool, args, scope, task, sid, owner)
            finally:
                self._inflight_call = None
        return super()._skill_call(tool, args, scope, task, sid, owner)

    def _inflight_refinement(self, stage, context):
        if self._inflight_call is None:
            return {'status': 'continue', 'reason': 'no active skill session'}
        scope, task, sid, owner = self._inflight_call
        if self._delegations >= self.budgets.max_delegations:
            return {'status': 'continue', 'reason': 'delegation budget exhausted'}
        self._delegations += 1
        if stage == 'pregrasp':
            instruction = ('The arm is paused at the measured pregrasp with the gripper open. '
                           'Check the fresh RGB and the pending gripper preview against the original goal below. '
                           'If the target is identifiable and a useful correction within the allowed bounds is visible, '
                           'use nudge_grasp and assess the new preview, repeating if useful within the cumulative budget. '
                           'Finish status=continue when suitable; abort if correction is unlikely to resolve the problem '
                           'or the target cannot be identified.')
        else:
            instruction = ('The arm is paused at the measured release goal while holding the object. Check the '
                           'fresh RGB and the pending gripper preview against the destination. Make at most a '
                           'small base-frame correction with nudge_place if useful, then finish status=continue '
                           'to release, or status=abort to keep holding if release here is clearly wrong.')
        original_instruction = task.get('instruction')
        if original_instruction:
            instruction += '\nOriginal delegated goal: ' + str(original_instruction)
        child = dict(instruction=instruction, original_instruction=original_instruction, inflight_refinement=True, stage=stage,
                     observation=deepcopy(context.get('observation')),
                     image_refs=list(context.get('image_refs', ())),
                     checkpoint=deepcopy(context.get('checkpoint')),
                     origin_candidate_ref=context.get('origin_candidate_ref'))
        if context.get('observation'):
            self._remember_observation(context['observation'], scope)
        self._event('inflight_refinement', sid, role=owner, stage=stage, state='started')
        try:
            output = self._delegate('refiner', child, scope, sid)
            result = output.get('result', {})
            decision = {'status': 'abort' if result.get('status') == 'abort' else 'continue',
                        'reason': result.get('reason')}
        except (BoundaryError,) as exc:
            decision = {'status': 'continue', 'reason': f'refiner session boundary error: {exc}'}
        self._event('inflight_refinement', sid, role=owner, stage=stage, state='completed', decision=decision)
        return decision

    def _finish(self, role, args, scope):
        task = self._task_stack[-1] if self._task_stack else {}
        if role == 'refiner' and task.get('inflight_refinement'):
            # Accept the generic child aliases too: a provider schema that only
            # knows success/failed must still be able to end the pause.
            status = {'success': 'continue', 'failed': 'abort'}.get(args.get('status'), args.get('status'))
            if status not in ('continue', 'abort') or not set(args) <= {'status', 'reason', 'evidence_image_refs'}:
                raise BoundaryError('paused refinement finishes with status continue or abort and an optional reason')
            result = {'status': status, 'reason': _text(args['reason']) if args.get('reason') else 'No reason supplied.'}
            if 'evidence_image_refs' in args:
                result['evidence_image_refs'] = self._evidence(args['evidence_image_refs'], scope)
            return result
        return super()._finish(role, args, scope)

    def _intent_dispatch(self, role, tool, args, scope, task, sid):
        if tool in ('nudge_grasp', 'nudge_place'):
            if role != 'refiner' or not task.get('inflight_refinement'):
                raise BoundaryError('nudge tools belong to the paused-execution Refiner session')
            if (tool == 'nudge_grasp') != (task.get('stage') == 'pregrasp'):
                raise BoundaryError('use the nudge tool matching the paused stage')
            raw = getattr(self.backend, 'nudge_inflight_grasp' if tool == 'nudge_grasp' else 'nudge_inflight_place')(**args)
            output = {k: deepcopy(v) for k, v in raw.items()
                      if k in ('accepted', 'reason_code', 'image_refs', 'step', 'cumulative', 'step_mm',
                               'cumulative_mm', 'path_check', 'self_collision')}
            output['accepted'] = bool(raw.get('accepted'))
            return output
        return super()._intent_dispatch(role, tool, args, scope, task, sid)


class ObserveFirstIntentOrchestrator(PauseRefineIntentOrchestrator):
    """Require one achieved downward observation above the target before grasping."""
    OVERHEAD_ATTEMPT_LIMIT = 2

    def __init__(self, *args, observe_before_grasp=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.observe_before_grasp = observe_before_grasp
        self._overhead_done = False
        self._overhead_attempts = 0

    def _prompt(self, role):
        prompt = super()._prompt(role)
        if role == 'prime' and self.observe_before_grasp:
            prompt += (
                ' OBSERVE-FIRST POLICY: first delegate_point to select the intended object; a target is '
                'required before delegate_waypoint. Before the first grasp request for an object, ask Pointer for a '
                'downward observation waypoint above the target (propose_downward_waypoint, about 0.30 m), '
                'validate_view and execute_view it, then re-select the target in the returned RGB and request '
                'the grasp. Grasp requests before that observation are refused with observe_from_above_first. '
                'After two rejected or failed overhead attempts the requirement is waived.')
        return prompt

    def _intent_dispatch(self, role, tool, args, scope, task, sid):
        if (self.observe_before_grasp and role == 'prime' and tool == 'delegate_grasp' and not self._overhead_done
                and self._overhead_attempts < self.OVERHEAD_ATTEMPT_LIMIT):
            raise ActionPreconditionError('observe_from_above_first')
        output = super()._intent_dispatch(role, tool, args, scope, task, sid)
        if role == 'prime' and tool == 'validate_view' and not output.get('accepted'):
            self._overhead_attempts += 1
        if role == 'prime' and tool == 'execute_view':
            if output.get('view_status') == 'achieved' and output.get('purpose') == 'observe':
                self._overhead_done = True
            elif output.get('view_status') != 'achieved':
                self._overhead_attempts += 1
        if role == 'prime' and (tool == 'release' or
                                (tool == 'execute_place_candidate' and output.get('status') == 'succeeded')):
            self._overhead_done, self._overhead_attempts = False, 0
        return output
