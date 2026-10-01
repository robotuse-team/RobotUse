"""Opt-in automatic refinement and high-to-direct grasp route recovery."""
from copy import deepcopy
from dataclasses import replace

from src.backend.waypoint_policy import LinearIntentBackend, LinearIntentOrchestrator
from src.backend.controller import ARGUMENTS, BoundaryError

ARGUMENTS['replan_grasp'] = ('candidate_ref', 'transit_policy')


class AutoRefineIntentBackend(LinearIntentBackend):
    def _routes(self, pose, target, obstacles, checker, scene, options, *, relaxed=False):
        from src.tools.motion.planning import plan_grasp, MotionPlanningError
        policy = getattr(self, '_route_policy_override', 'auto')
        policies = ('high', 'legacy') if policy == 'auto' else (policy,)
        attempts = []
        plan = None
        for policy in policies:
            try:
                plan = plan_grasp(self.connector, grasp_transform=pose,
                    max_width_m=getattr(self, 'max_gripper_width_m', .08),
                    target_points=target, obstacle_points=obstacles,
                    grasp_to_ee=self.grasp_to_ee, frame='connector_base',
                    config=replace(self.motion_config, transit_policy=policy), **options)
                evidence = checker.check(plan, scene, stop_label='grasp',
                    jaw_width_m=plan.open_width_m, **({'skip_scene': True} if relaxed else {}))
            except MotionPlanningError as exc:
                evidence = {'kind': 'planning', 'error': str(exc),
                    **getattr(exc, 'planning_feedback', {}), 'accepted': False}
            attempts.append(dict(policy=policy, relaxed=relaxed, evidence=deepcopy(evidence)))
            self._record('grasp_route_attempt', attempts[-1])
            if evidence['accepted']:
                return plan, {**evidence, 'route_attempts': attempts, 'transit_policy': policy}
        return None, {**evidence, 'route_attempts': attempts}

    def _plan_grasp_candidate(self, pose, target_points, obstacle_points, checker, scene, options):
        return self._routes(pose, target_points, obstacle_points, checker, scene, options)

    def _plan_relaxed_grasp(self, geometry, prediction, point, ref):
        from src.tools.motion.planning import MotionPlanningError
        checker, scene = self._candidate_path_scene(point)
        plan, evidence = self._routes(prediction.pose, geometry.object_points,
            self._point(point).scene_points, checker, scene,
            self._grasp_options(prediction.pose, geometry, prediction, ref), relaxed=True)
        if plan is None:
            raise MotionPlanningError('Both scene-relaxed grasp routes failed', planning_feedback=evidence)
        return plan, evidence


class AutoRefineIntentOrchestrator(LinearIntentOrchestrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._auto_reviews = {}

    def _tools(self, role, task):
        tools = super()._tools(role, task)
        if role == 'refiner' and not task.get('placement_refinement'):
            return (*tools[:-1], 'replan_grasp', tools[-1])
        return tools

    def _prompt(self, role):
        prompt = super()._prompt(role)
        if role in ('prime', 'grasp', 'place', 'refiner'):
            prompt += (' Candidate handoffs automatically include one Refiner review before execution. '
                'Refiner is a pose/approach helper, not an approval reviewer: inspect and keep a suitable '
                'candidate unchanged, or adjust it when useful. No mandatory adjustment or goal breakdown. '
                'With world planning enabled, automatic grasp routes try world, high, then legacy after planning OR scene '
                'rejection. Refiner can replan_grasp(candidate_ref, transit_policy=auto|high|legacy) '
                'without changing the grasp pose. After ordinary refinement fails and no normal candidate remains, '
                'explicit relax_candidate keeps the selected pose and replans with scene collision relaxed, retaining IK, self checks and tracking. '
                'Prime may also delegate_refiner to ask whether a pose/approach is suitable. '
                'Distinguish planning rejection, tool error, and an actual failed execution; a tool error '
                'does not prove a target is impossible. Return candidate_ref to keep/use a pose or '
                'status=failed with a short reason when no usable candidate is available.')
        return prompt

    def _intent_dispatch(self, role, tool, args, scope, task, sid):
        if tool == 'replan_grasp':
            if role != 'refiner' or args['transit_policy'] not in ('auto', 'high', 'legacy'):
                raise BoundaryError('Refiner selects auto, high, or legacy')
            previous = getattr(self.backend, '_route_policy_override', 'auto')
            self.backend._route_policy_override = args['transit_policy']
            try:
                return super()._intent_dispatch(role, 'adjust_grasp', dict(
                    candidate_ref=args['candidate_ref'], dx_mm=0., dy_mm=0., dz_mm=0.,
                    roll_deg=0., pitch_deg=0., yaw_deg=0.), scope, task, sid)
            finally:
                self.backend._route_policy_override = previous
        output = super()._intent_dispatch(role, tool, args, scope, task, sid)
        if role != 'prime' or tool not in ('delegate_grasp', 'delegate_place'):
            return output
        choice = output.get('result', {})
        ref = choice.get('candidate_ref') or choice.get('related_refs', {}).get('candidate_ref')
        if not ref:
            if tool == 'delegate_grasp':
                bundle = self._cached_grasp_bundle(args)
                candidates = bundle.get('candidates', []) + bundle.get('diagnostic_candidates', [])
            else:
                candidates = output.get('refinement_candidates', [])
            if candidates:
                ref = candidates[0]['candidate_ref']
        if not ref:
            return output
        key = (self._epoch, ref)
        if key in self._auto_reviews:
            return {**output, 'result': deepcopy(self._auto_reviews[key]), 'automatic_refinement': 'reused'}
        scope['candidates'][ref] = self._epoch
        self._event('automatic_refinement', sid, role='prime', candidate_ref=ref, stage='started')
        review = super()._intent_dispatch('prime', 'delegate_refiner', dict(candidate_ref=ref,
            instruction='Check whether this pose and approach suit the current scene. Keep it if suitable; '
                        'otherwise repair pose/approach. Try scene relaxation after normal refinement '
                        'fails. Return a usable candidate or the concrete limitation.'), scope, task, sid)
        result = deepcopy(review['result'])
        selected = result.get('candidate_ref')
        if selected and tool == 'delegate_grasp':
            result['decision'] = 'accepted'
            scope.setdefault('grasp_decisions', {})[selected] = 'accepted'
        # An optional review that simply declines to change a validated original
        # must not become a new approval gate. Execution still needs validation.
        if not selected and choice.get('candidate_ref') and choice.get('decision') != 'needs_refinement':
            result = deepcopy(choice)
            selected = result['candidate_ref']
        self._auto_reviews[key] = deepcopy(result)
        if selected:
            self._auto_reviews[(self._epoch, selected)] = deepcopy(result)
        self._event('automatic_refinement', sid, role='prime', stage='completed',
                    original_candidate_ref=ref, result=result, review_result=review['result'])
        return {**output, 'result': result, 'automatic_refinement': review,
                'original_selection': choice}
