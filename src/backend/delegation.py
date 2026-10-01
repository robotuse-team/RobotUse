"""Agent-selected geometry, grasp review, and separate release."""
from copy import deepcopy

from src.core.action_feedback import ActionPreconditionError
from src.agent.intent_prompts import COMMON, FEEDBACK, intent_prompt
from src.backend.intent import public_operation_result
from src.backend.paused_refinement import ObserveFirstIntentOrchestrator
from src.backend.controller import ARGUMENTS, BoundaryError, BudgetExceeded
from src.backend.interaction import InteractionFeatures, InteractionOrchestrator
from src.tools.grasp.arguments import property_schema, validate_argument
from src.tools.coordination.tool import DELEGATE_REFINER
from src.tools.grasp.tool import EXECUTE_GRASP, VALIDATE_GRASP
from src.tools.place.tool import EXECUTE_PLACE, SAVE_DESTINATION


ARGUMENTS.update({
    'explicit_grasp_candidates': ('point_ref', 'direction', 'tolerance_deg', 'azimuth_deg',
                            'polar_deg', 'geometric_height', 'transit'),
    'explicit_place_candidates': ('destination_ref',),
    'explicit_prepare_place': ('destination_ref', 'xy_source', 'xy_m', 'height', 'transit_height'),
    'explicit_adjust_place': ('candidate_ref', 'dx_mm', 'dy_mm', 'dz_mm', 'roll_deg', 'pitch_deg', 'yaw_deg'),
    'explicit_execute_place': ('candidate_ref',),
    'move_vertical': ('dz_m',),
    'goto_home_joint_position': (),
})

from src.agent.prompts import _HEIGHT_GUIDANCE



class DelegationOrchestrator(InteractionOrchestrator, ObserveFirstIntentOrchestrator):
    interaction_features = InteractionFeatures(place_rotation=False, held_observation=True)
    structured_tool_errors = True
    requested_refine_routes = False
    unrestricted_pose_rotation = False
    unrestricted_pose_translation = False

    def __init__(self, *args, observe_before_grasp=False, mandatory_observation=False,
                 auto_refine_routes=False, **kwargs):
        super().__init__(*args, observe_before_grasp=observe_before_grasp or mandatory_observation, **kwargs)
        self.mandatory_observation = mandatory_observation
        self.auto_refine_routes = auto_refine_routes
        self._previous_grasp_feedback = None

    def _optional_arguments(self, tool):
        if tool == 'delegate_grasp':
            return ()
        return super()._optional_arguments(tool)

    def _tool_argument(self, tool, key, value):
        if property_schema(key, tool=tool) is not None:
            return validate_argument(tool, key, value)
        return super()._tool_argument(tool, key, value)

    def _tools(self, role, task):
        if role == 'prime':
            return ('review_observation', 'delegate_point', 'delegate_waypoint', 'validate_view',
                    'execute_view', 'set_grasp_mode', 'delegate_grasp', 'delegate_destination',
                    'delegate_place', 'move_vertical',
                    'goto_home_joint_position', 'close_for_push', 'release', 'turn', 'finish')
        if role == 'grasp':
            return ('explicit_grasp_candidates', 'inspect_candidate', 'preview_candidate', 'finish')
        if role == 'place':
            return ('explicit_place_candidates', 'explicit_prepare_place', 'explicit_adjust_place', 'inspect_place_candidate', 'finish')
        if role == 'refiner' and not task.get('inflight_refinement') and not task.get('waypoint_ref'):
            return ('inspect_candidate', 'preview_candidate', 'adjust_grasp', 'finish')
        return super()._tools(role, task)

    def _prompt(self, role):
        from src.agent import create_agent
        try:
            agent = create_agent(role)
        except KeyError:
            raise BoundaryError('unknown agent role') from None
        return agent.base_prompt(self)

    def _register_bundle(self, raw, scope, task):
        bundle = deepcopy(raw)
        if not hasattr(self, '_geometry_handoffs'):
            self._geometry_handoffs = {}
        for candidate in bundle.get('candidates', []):
            scope['candidates'][candidate['candidate_ref']] = self._epoch
        for candidate in bundle.get('diagnostic_candidates', []):
            ref = candidate['candidate_ref']
            self._diagnostic_candidates[ref] = (self._epoch, deepcopy(candidate))
            scope['candidates'][ref] = self._epoch
        for candidate in [*bundle.get('candidates', []), *bundle.get('diagnostic_candidates', [])]:
            self._candidate_handoffs[candidate['candidate_ref']] = dict(
                epoch=self._epoch, point_ref=task.get('point_ref'),
                observation_id=(self._current_observation or {}).get('observation_id'),
                model_intent=dict(instruction=task.get('instruction')),
                measured_geometry=deepcopy(bundle.get('geometry', {})),
                candidate_geometry=deepcopy(candidate))
            self._geometry_handoffs[candidate['candidate_ref']] = deepcopy(
                self._candidate_handoffs[candidate['candidate_ref']])
        return bundle

    def _delegate(self, role, task, scope, sid):
        if role == 'grasp' and self._previous_grasp_feedback is not None:
            task = {**task, 'previous_grasp_feedback': deepcopy(self._previous_grasp_feedback)}
        output = super()._delegate(role, task, scope, sid)
        # The Grasp handoff was captured before the child generated
        # any candidates. Retain the geometry actually returned inside it.
        selected = output.get('result', {}).get('candidate_ref')
        if role == 'grasp' and selected in getattr(self, '_geometry_handoffs', {}):
            self._candidate_handoffs[selected] = deepcopy(self._geometry_handoffs[selected])
        return output

    def _grasp_selection_feedback(self, output):
        """Copy only the child's answer and candidate fields already shown to it."""
        result = output.get('result', {})
        selection = {key: deepcopy(result[key]) for key in
                     ('candidate_ref', 'status', 'decision', 'reason', 'limitations') if key in result}
        candidate = getattr(self, '_geometry_handoffs', {}).get(
            result.get('candidate_ref'), {}).get('candidate_geometry', {})
        selection['candidate'] = {key: deepcopy(candidate[key]) for key in (
            'candidate_ref', 'source', 'source_candidate_index', 'contact_center_xyz_m',
            'contact_center_pose_base', 'approach_direction_base', 'top_down_only',
            'yaw_deg', 'frame', 'position_reference', 'units', 'transit',
            'executable', 'reason_code') if key in candidate}
        return selection

    def _grasp_feedback_result(self, output, selection, stage, args, *, selected=None):
        feedback = dict(reference_only=True, instruction=args['instruction'],
            point_ref=args['point_ref'], selection=deepcopy(selection), stage=stage,
            selection_scope='original Grasp choice before optional Refiner; not the executed pose',
            original_selection=deepcopy(selection), selected_candidate=deepcopy(selected),
            executed=bool(output.get('executed')),
            tool_feedback=public_operation_result(output.get('result', output)))
        execution_pose = output.get('result', {}).get('execution_pose')
        if isinstance(execution_pose, dict):
            feedback['execution_pose'] = deepcopy(execution_pose)
        self._previous_grasp_feedback = deepcopy(feedback)
        return {**output, 'grasp_feedback': feedback}

    def _dispatch(self, role, tool, args, scope, task, sid):
        output = super()._dispatch(role, tool, args, scope, task, sid)
        if role == 'grasp' and tool == 'inspect_candidate':
            # The inherited loop otherwise replaces all Grasp tools after three
            # inspections with a refinement tool. All four alternatives
            # stay reviewable; only the separate Refiner may change their poses.
            # The loop increments this counter once after dispatch returns.
            scope['candidate_review_calls'] = 0
        return output

    @staticmethod
    def _refinement_requested(output):
        choice = output.get('result', {})
        return (isinstance(choice, dict) and
                (choice.get('status') == 'needs_refinement' or choice.get('decision') == 'needs_refinement'))

    @staticmethod
    def _approved_choice(output, *, refinement=False):
        if output.get('status') != 'completed' or not isinstance(output.get('result'), dict):
            return None
        choice = output['result']
        if choice.get('status') in ('failed', 'needs_observation', 'unknown'):
            return None
        if choice.get('decision') in ('failed', 'needs_observation'):
            return None
        if refinement and (choice.get('status') not in (None, 'success')
                           or choice.get('decision') == 'needs_refinement'):
            return None
        return choice.get('candidate_ref')

    def _explicit_grasp(self, args, scope, task, sid):
        if (self.observe_before_grasp and not self._overhead_done
                and (self.mandatory_observation or self._overhead_attempts < self.OVERHEAD_ATTEMPT_LIMIT)):
            raise ActionPreconditionError('observe_from_above_first')
        ref = args['point_ref']
        if self._point_refs.get(ref) != self._epoch:
            raise BoundaryError('use current Pointer-approved target')
        output = self._delegate('grasp', {**args, 'target_evidence': self._target_evidence.get(ref, {}),
            'measured_geometry': deepcopy(self.backend.explicit_geometry(point_ref=ref))}, scope, sid)
        selection = self._grasp_selection_feedback(output)
        selected_feedback = deepcopy(selection)
        selected = self._approved_choice(output)
        if not selected:
            return self._grasp_feedback_result({**output, 'executed': False}, selection, 'selection', args)
        if scope['candidates'].get(selected) != self._epoch:
            raise BoundaryError('Grasp selection must belong to the current delegation')
        requested = self._refinement_requested(output)
        if self.auto_refine_routes or (self.requested_refine_routes and requested):
            original_ref = selected
            review = self._skill_call(DELEGATE_REFINER.dispatch_name, dict(candidate_ref=selected,
                instruction=args['instruction'] + ' Inspect this selected pose before execution; preserve explicit '
                            'transit heights. Selection reason: ' + str(output['result'].get('reason', 'not supplied'))
                            + ('' if self.unrestricted_pose_rotation else
                               ' Geometric mean/median grasps remain downward, yaw-only.')),
                scope, task, sid, 'grasp')
            selected = self._approved_choice(review, refinement=True)
            self._event('candidate_refinement', sid, role='grasp',
                trigger='requested' if requested else 'automatic', original_candidate_ref=original_ref,
                selected_candidate_ref=selected, result=deepcopy(review.get('result', {})))
            if not selected or selected in self._diagnostic_candidates:
                return self._grasp_feedback_result(
                    {**review, 'executed': False, 'original_selection': output.get('result')},
                    selection, 'refinement', args)
            if scope['candidates'].get(selected) != self._epoch:
                raise BoundaryError('Refiner selection must be current')
            selected_feedback = self._grasp_selection_feedback(review)
        elif selected in self._diagnostic_candidates or requested:
            return self._grasp_feedback_result({**output, 'executed': False}, selection, 'selection', args)
        scope.setdefault('grasp_decisions', {})[selected] = 'accepted'
        validation = self._skill_call(VALIDATE_GRASP.dispatch_name, {'candidate_ref': selected}, scope, task, sid, 'grasp')
        if not validation.get('accepted') or not validation.get('validation_ref'):
            return self._grasp_feedback_result(dict(result=validation, executed=False), selection, 'validation', args,
                                               selected=selected_feedback)
        execution = self._skill_call(EXECUTE_GRASP.dispatch_name, dict(candidate_ref=selected,
            validation_ref=validation['validation_ref']), scope, {**task, 'instruction': args['instruction']}, sid, 'grasp')
        return self._grasp_feedback_result(
            dict(skill='grasp', result=execution, executed=bool(execution.get('execution_ref')),
                 observation=execution.get('observation'), robot_state=execution.get('robot_state')),
            selection, 'execution', args, selected=selected_feedback)

    def _motion_result(self, method, args, scope, sid):
        try:
            output = getattr(self.backend, method)(**args)
        finally:
            if self.backend.epoch != self._epoch:
                self._current_observation = None
            self._epoch = self.backend.epoch
        if not isinstance(output, dict):
            raise TypeError('Motion must return an operation result')
        observation = output.get('observation')
        if observation is not None:
            self._remember_observation(observation, scope, invalidate=True)
        else:
            observation = self._environment_observation(scope, sid, refresh=self._current_observation is None)
        return {**output, 'observation': observation, 'robot_state': dict(
            grasp_mode=getattr(self.backend, 'grasp_mode', None),
            grasp_command_active=getattr(self.backend, 'held_plan', None) is not None,
            closed_pusher_command=bool(getattr(self.backend, 'closed_push', False)),
            observation_id=observation.get('observation_id'))}

    def _intent_dispatch(self, role, tool, args, scope, task, sid):
        if tool == 'explicit_geometry':
            raise BoundaryError('explicit_geometry is internal; use supplied measured_geometry')
        if tool == 'delegate_destination' and role == 'prime':
            output = super()._intent_dispatch(role, tool, args, scope, task, sid)
            choice = output.get('result', {})
            if (output.get('status') != 'completed' or not isinstance(choice, dict)
                    or choice.get('status') in ('failed', 'needs_observation', 'unknown')
                    or not choice.get('point_ref')):
                return output
            saved = self._skill_call(SAVE_DESTINATION.dispatch_name, {'point_ref': choice['point_ref']},
                                     scope, task, sid, 'prime')
            if not saved.get('destination_ref'):
                return dict(status='failed', result=choice, destination_save=saved)
            return {**output, 'destination_ref': saved['destination_ref'],
                    'result': {**choice, **saved}, 'destination_saved': True}
        if tool == 'delegate_grasp' and role == 'prime':
            return self._explicit_grasp(args, scope, task, sid)
        if tool == 'explicit_grasp_candidates':
            clicked = (args.get('direction') == 'clicked'
                       and getattr(self, 'clicked_grasp_candidates', False))
            if not clicked:
                validate_argument(tool, 'direction', args.get('direction'))
            if role != 'grasp' or args['point_ref'] != task.get('point_ref'):
                raise BoundaryError('use the current delegated grasp target')
            if self._point_refs.get(args['point_ref']) != self._epoch:
                raise BoundaryError('grasp target is stale')
            if tool == 'explicit_grasp_candidates':
                angles = (args.get('azimuth_deg'), args.get('polar_deg'))
                if args.get('direction') == 'custom':
                    if any(value is None for value in angles):
                        raise BoundaryError('custom direction requires azimuth_deg and polar_deg')
                elif any(value is not None for value in angles):
                    raise BoundaryError('non-custom direction requires null azimuth_deg and polar_deg')
                if args.get('direction') in ('mean', 'median') or clicked:
                    if args.get('tolerance_deg') is not None or args.get('geometric_height') is None:
                        modes = 'clicked' if clicked else 'mean/median'
                        raise BoundaryError(modes + ' require geometric_height and null tolerance_deg')
                elif args.get('geometric_height') is not None or args.get('tolerance_deg') is None:
                    raise BoundaryError('CGN requires null geometric_height and explicit tolerance_deg')
            raw = getattr(self.backend, tool)(**args)
            return self._register_bundle(raw, scope, task) if tool == 'explicit_grasp_candidates' else deepcopy(raw)
        if tool in ('explicit_place_candidates', 'explicit_prepare_place'):
            if role != 'place' or args['destination_ref'] != task.get('destination_ref'):
                raise BoundaryError('use the delegated placement destination')
            raw = getattr(self.backend, tool)(**args)
            if tool == 'explicit_prepare_place' and raw.get('candidate_ref') and raw.get('accepted', True):
                scope['candidates'][raw['candidate_ref']] = self._epoch
                if raw.get('image_refs'):
                    scope['inspected_candidates'][raw['candidate_ref']] = self._epoch
            return deepcopy(raw)
        if tool == 'explicit_adjust_place' or (tool == 'inspect_place_candidate' and role == 'place'):
            ref = args['candidate_ref']
            if role != 'place' or scope['candidates'].get(ref) != self._epoch:
                raise BoundaryError('review or adjust a current Place candidate')
            method = 'explicit_inspect_place' if tool == 'inspect_place_candidate' else tool
            raw = getattr(self.backend, method)(**args)
            if tool == 'inspect_place_candidate' and raw.get('candidate_ref') != ref:
                raise BoundaryError('placement inspection changed the candidate identity')
            if raw.get('accepted', True) and raw.get('candidate_ref'):
                scope['candidates'][raw['candidate_ref']] = self._epoch
                if raw.get('image_refs'):
                    scope['inspected_candidates'][raw['candidate_ref']] = self._epoch
            return deepcopy(raw)
        if tool == 'delegate_place' and role == 'prime':
            if args['destination_ref'] not in self.destinations:
                raise BoundaryError('use a saved measured destination')
            if args['hold_assessment'] != 'held' or args['destination_assessment'] != 'unchanged':
                raise BoundaryError('placement needs current held and unchanged assessments')
            output = self._delegate('place', args, scope, sid)
            selected = self._approved_choice(output)
            if not selected:
                return {**output, 'executed': False}
            if scope['candidates'].get(selected) != self._epoch:
                raise BoundaryError('Place selection must be current')
            result = self._skill_call(EXECUTE_PLACE.dispatch_name, {'candidate_ref': selected}, scope,
                                      {**task, 'instruction': args['instruction']}, sid, 'place')
            return dict(skill='place', result=result, executed=bool(result.get('execution_ref')),
                        observation=result.get('observation'), release_requires_agent_decision=True)
        if tool == 'explicit_execute_place':
            if role != 'prime' or scope['candidates'].get(args['candidate_ref']) != self._epoch:
                raise BoundaryError('execute only the current Place selection')
            return self._motion_result(tool, args, scope, sid)
        if tool in ('move_vertical', 'goto_home_joint_position'):
            if role != 'prime':
                raise BoundaryError('only Prime chooses robot motion')
            return self._motion_result(tool, args, scope, sid)
        return super()._intent_dispatch(role, tool, args, scope, task, sid)

    def _inflight_refinement(self, stage, context):
        if stage != 'pregrasp' or self._inflight_call is None:
            return dict(status='abort', reason='An explicit agent decision is required')
        if self._delegations >= self.budgets.max_delegations:
            return dict(status='abort', reason='Refiner delegation budget exhausted')
        scope, task, sid, owner = self._inflight_call
        child = dict(inflight_refinement=True, stage=stage,
            instruction='Inspect the fresh open-hand pregrasp and pending pose. Correct if useful, then explicitly '
                        'continue to close or abort. '
                        + ('All candidate sources allow finite local roll/pitch/yaw without angular magnitude limits; '
                           + ('translation magnitudes are also unrestricted; validate the changed route. '
                              if self.unrestricted_pose_translation else
                              'retain translation bounds and validate the changed route. ')
                           if self.unrestricted_pose_rotation else
                           'Mean/median grasps allow translation and yaw only. ')
                        + 'Original delegated goal: ' + str(task.get('instruction', '')),
            observation=deepcopy(context.get('observation')), image_refs=list(context.get('image_refs', [])),
            checkpoint=deepcopy(context.get('checkpoint')), origin_candidate_ref=context.get('origin_candidate_ref'))
        if child['observation']:
            self._remember_observation(child['observation'], scope)
        self._event('inflight_refinement', sid, role=owner, stage=stage, state='started')
        try:
            output = self._delegate('refiner', child, scope, sid)
            approved = output.get('status') == 'completed' and output.get('result', {}).get('status') == 'continue'
            decision = dict(status='continue' if approved else 'abort',
                            reason=output.get('result', {}).get('reason', 'No explicit Refiner approval'))
        except (BoundaryError, BudgetExceeded) as exc:
            decision = dict(status='abort', reason=str(exc))
        self._event('inflight_refinement', sid, role=owner, stage=stage, state='completed', decision=decision)
        return decision
