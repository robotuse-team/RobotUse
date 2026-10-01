"""Grasp/Place hand off execution outcomes, refining only when needed."""
from src.backend.refinement_routes import AutoRefineIntentOrchestrator
from src.backend.waypoint_policy import LinearIntentOrchestrator
from src.agent.intent_prompts import COMMON, FEEDBACK, GRASP_APPROACH_PROMPT
from src.backend.controller import BudgetExceeded


class SkillOwnedIntentOrchestrator(AutoRefineIntentOrchestrator):
    def _finish(self, role, args, scope):
        if role == 'place' and args.get('status') == 'needs_refinement':
            # Retain a checked explanatory reference, without granting execution.
            clean = {**args, 'status': 'failed'}
            result = super()._finish(role, clean, scope)
            return {**result, 'status': 'needs_refinement'}
        return super()._finish(role, args, scope)

    def _tools(self, role, task):
        tools = super()._tools(role, task)
        if role == 'prime':
            hidden = {'delegate_refiner', 'validate_grasp', 'execute_grasp',
                      'validate_place', 'execute_place_candidate'}
            tools = tuple(t for t in tools if t not in hidden)
        return tools

    def _prompt(self, role):
        if role == 'prime':
            return COMMON + (
                'You are Prime. Follow the original objective directly. Pointer selects targets, destinations '
                'and waypoints. delegate_grasp and delegate_place select candidates, ask Refiner only when '
                'needed, validate and execute internally. Their result is an execution outcome, not a pose '
                'for you to adjust or execute. Assess the returned current RGB before deciding what to do next. '
                'A succeeded command does not prove an object is held or the task succeeded. '
                'grasp_command_active is command state: if the pickup failed visually, release before a new '
                'grasp request. Do not change mode while a grasp command is active. Choose transport mode '
                'for free objects or contact mode for handles/knobs before requesting a grasp. '
                'Placement uses Pointer destination selection, save_destination, then delegate_place with '
                'hold_assessment=held and destination_assessment=unchanged when current RGB supports them. '
                'Describe useful placement changes in the delegation instruction; Place/Refiner own pose changes. '
                'turn is only for rotating a contact-held knob about local tool Z at fixed TCP, never for '
                'repairing object transport or placement. Waypoints support validate_view with motion=linear '
                'for straight translation at current orientation, or planned, then execute_view. Use contact '
                'waypoints for pushing/pulling; close_for_push prepares an empty closed pusher. '
                'No fixed observation, goal-decomposition or retry sequence is required. A rejected no-motion '
                'request preserves the current observation. After motion use returned fresh observation. '
                'Finish completed, failed or unknown with your assessment; native verifier decides actual success. '
            ) + GRASP_APPROACH_PROMPT + FEEDBACK
        prompt = LinearIntentOrchestrator._prompt(self, role)
        if role in ('grasp', 'place'):
            prompt = prompt.replace('Prime validates and executes a selected candidate.',
                'The skill validates and executes your selected candidate before returning to Prime.')
            prompt += (' Select a suitable candidate directly. Ask for refinement with status=needs_refinement '
                'and candidate_ref when correction or approach repair is useful. A rejected pose can be '
                'returned with status=failed and candidate_ref. The skill invokes Refiner when needed and executes '
                'a passing selection; do not require Prime to manipulate pose angles.')
        if role == 'refiner':
            prompt += (' You are called because this skill needs pose or route repair. For grasps, '
                'replan_grasp can try auto, high or legacy without changing the pose. Return a passing '
                'candidate or a concrete limitation. No approval or mandatory modification is needed.')
        return prompt

    def _skill_call(self, tool, args, scope, task, sid, owner):
        if self._calls >= self.budgets.max_tool_calls:
            raise BudgetExceeded('tool-call budget exhausted inside skill')
        self._calls += 1
        self._event('tool_called', sid, role=owner, tool=tool, arguments=args, internal_skill=True)
        result = self._dispatch('prime', tool, args, scope, task, sid)
        self._event('tool_result', sid, role=owner, tool=tool, result=result, internal_skill=True)
        return result

    def _intent_dispatch(self, role, tool, args, scope, task, sid):
        if role != 'prime' or tool not in ('delegate_grasp', 'delegate_place'):
            return super()._intent_dispatch(role, tool, args, scope, task, sid)
        # Dispatch the child selection before applying optional pose review.
        output = LinearIntentOrchestrator._intent_dispatch(self, role, tool, args, scope, task, sid)
        # A failed child session has made no valid selection. Never turn its
        # error (or exhausted budget) into an automatic diagnostic-pose choice.
        if output.get('status') != 'completed':
            return {**output, 'executed': False}
        choice = output.get('result', {})
        if choice.get('status', choice.get('decision')) == 'needs_observation':
            # An explanatory rejected candidate is not a request to refine it.
            return {**output, 'executed': False}
        ref = choice.get('candidate_ref') or choice.get('related_refs', {}).get('candidate_ref')
        owner = 'grasp' if tool == 'delegate_grasp' else 'place'
        needs = choice.get('decision') == 'needs_refinement' or choice.get('status') == 'needs_refinement'
        if not ref and choice.get('status', choice.get('decision')) != 'needs_observation':
            candidates = (self._cached_grasp_bundle(args).get('diagnostic_candidates', [])
                          if owner == 'grasp' else output.get('refinement_candidates', []))
            if candidates: ref = candidates[0]['candidate_ref']; needs = True
        if not ref:
            return {**output, 'executed': False}
        diagnostic = ref in self._diagnostic_candidates or ref in self.backend.place_diagnostics
        needs = needs or diagnostic
        if not needs and not choice.get('candidate_ref'):
            return {**output, 'executed': False}
        scope['candidates'][ref] = self._epoch
        validate = 'validate_grasp' if owner == 'grasp' else 'validate_place'
        validation = None
        if not needs:
            validation = self._skill_call(validate, {'candidate_ref': ref}, scope, task, sid, owner)
            needs = not validation.get('accepted', False)
        if needs:
            self._event('automatic_refinement', sid, role=owner, stage='started', candidate_ref=ref)
            review = self._skill_call('delegate_refiner', {
                'candidate_ref': ref,
                'instruction': args['instruction'] + ' Repair the selected pose/approach if useful. '
                    'Keep a suitable pose; after normal refinement fails try the available scene relaxation. '
                    'Return an executable candidate or the concrete limitation.'}, scope, task, sid, owner)
            if review.get('status') != 'completed':
                return {**review, 'executed': False,
                        'candidate_generation': output.get('candidate_generation')}
            choice = review.get('result', {})
            self._event('automatic_refinement', sid, role=owner, stage='completed', result=choice)
            ref = choice.get('candidate_ref')
            if not ref or ref in self._diagnostic_candidates or ref in self.backend.place_diagnostics:
                return {**output, 'result': choice, 'executed': False}
            validation = self._skill_call(validate, {'candidate_ref': ref}, scope, task, sid, owner)
        if not validation.get('accepted'):
            return {**output, 'result': {'status': 'failed', 'validation': validation}, 'executed': False}
        # Paused Refiner needs this skill's current object/relation/recovery intent.
        # Keep Prime's original task unchanged; the task stack still supplies the
        # original objective independently to delegated sessions.
        execution_task = {**task, 'instruction': args['instruction']}
        execution = self._skill_call('execute_grasp' if owner == 'grasp' else 'execute_place_candidate',
            {'candidate_ref': ref, 'validation_ref': validation['validation_ref']}, scope, execution_task, sid, owner)
        return {'skill': owner, 'result': execution, 'executed': bool(execution.get('execution_ref')),
                'observation': execution.get('observation'), 'robot_state': execution.get('robot_state'),
                'candidate_generation': output.get('candidate_generation')}
