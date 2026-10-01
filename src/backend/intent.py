"""Persistent whole-task reasoning with isolated, evidence-bearing specialists.

Prime follows the original instruction and current observations directly.
Specialists return selections or limitations; native verification stays separate.
"""
from copy import deepcopy
import math

from src.backend.review import ReviewDrivenOrchestrator
from src.backend.controller import ARGUMENTS, BoundaryError, BudgetExceeded, AuditError, EpisodeEnded, public_result, _text
from src.core.planning_feedback import public_planning_feedback
from src.core.action_feedback import ActionPreconditionError

ARGUMENTS.update({
    'review_observation': ('observation_id',),
    'propose_waypoint': ('observation_id', 'u', 'v', 'purpose', 'height_offset_m', 'dx_m', 'dy_m', 'dz_m'),
    'propose_downward_waypoint': ('observation_id', 'u', 'v', 'height_offset_m', 'dx_m', 'dy_m', 'dz_m'),
    'shift_waypoint': ('observation_id', 'purpose', 'dx_m', 'dy_m', 'dz_m'),
    'set_grasp_mode': ('mode',),
})


def image_refs(value):
    refs = []
    def walk(item):
        if isinstance(item, dict):
            for key, val in item.items():
                if key == 'image_ref' and isinstance(val, str): refs.append(val)
                elif key in ('image_refs', 'evidence_image_refs') and isinstance(val, list): refs.extend(val)
                else: walk(val)
        elif isinstance(item, (list, tuple)):
            for val in item: walk(val)
    walk(value)
    return list(dict.fromkeys(refs))


_OPERATION_TOOLS = frozenset(('validate_view', 'execute_view', 'validate_grasp', 'execute_grasp',
    'validate_place', 'execute_place_candidate', 'close_for_push', 'release', 'turn', 'set_grasp_mode'))

_GRASP_BUDGET_FIELDS = ('limit', 'reserved', 'remaining', 'per_view_batch', 'policy',
    'per_view_batch_scope', 'moveit_pool_slots', 'published_candidates', 'published_diagnostics', 'execution_attempts')


def public_operation_result(raw):
    """Small result summary, without geometry, validation tokens or arbitrary nesting."""
    output = {}
    planning = public_planning_feedback(raw.get('validation_feedback'))
    if not planning and isinstance(raw.get('validation'), dict):
        planning = public_planning_feedback(raw['validation'].get('validation_feedback'))
    if planning:
        output['planning_feedback'] = planning
    for key in ('status', 'view_status', 'reason_code', 'rejection_kind', 'error', 'exception_type',
                'reason', 'limitations', 'purpose', 'hold_status', 'observation_improvement', 'grasp_mode'):
        value = raw.get(key)
        if isinstance(value, str) and value.strip() and len(value) <= 8192: output[key] = value
    for key in ('accepted', 'holding_command', 'candidates_invalidated', 'executed', 'state_changed', 'terminal'):
        if type(raw.get(key)) is bool: output[key] = raw[key]
    for key in ('translation_m', 'endpoint_error_m', 'orientation_error_rad', 'endpoint_orientation_tolerance_rad',
                'required_simulation_s'):
        value = raw.get(key)
        if type(value) in (int, float) and math.isfinite(value): output[key] = value
    if 'error_details' in raw:
        from src.core.errors import public_error_details
        details = public_error_details(raw['error_details'])
        if details is not None:
            output['error_details'] = details
    tracking = raw.get('target_update', {})
    if isinstance(tracking, dict) and isinstance(tracking.get('tracking_status'), str):
        output['target_update'] = {'tracking_status': tracking['tracking_status'][:8192]}
    execution = raw.get('execution_feedback', {})
    if isinstance(execution, dict):
        feedback = {}
        if type(execution.get('release_commanded')) is bool:
            feedback['release_commanded'] = execution['release_commanded']
        if type(execution.get('retreat_executed')) is bool:
            feedback['retreat_executed'] = execution['retreat_executed']
        if execution.get('retreat_reason_code') == 'insufficient_simulation_time':
            feedback['retreat_reason_code'] = 'insufficient_simulation_time'
        check = execution.get('release_tracking_check', {})
        if isinstance(check, dict):
            clean = {}
            if type(check.get('accepted')) is bool: clean['accepted'] = check['accepted']
            for key in ('position_error_m', 'orientation_error_rad', 'position_tolerance_m', 'orientation_tolerance_rad'):
                value = check.get(key)
                if type(value) in (int, float) and math.isfinite(value): clean[key] = value
            if clean: feedback['release_tracking_check'] = clean
        if feedback: output['execution_feedback'] = feedback
    return output


def public_bundle(tool, raw):
    """Keep measured geometry/private files out of the evidence handoff."""
    output = public_result(tool, raw)
    sources = {c['candidate_ref']: c for c in raw.get('candidates', [])}
    for candidate in output.get('candidates', []):
        entry = sources.get(candidate['candidate_ref'], {})
        source = entry.get('source_view')
        if source: candidate['source_view'] = _text(source)
        if entry.get('ensemble_source'): candidate['ensemble_source'] = _text(entry['ensemble_source'])
        for key in ('downward_angle_deg', 'pool_rank'):
            value = entry.get(key)
            if type(value) in (int, float) and math.isfinite(value):
                candidate[key] = value
    if tool == 'grasp_candidates':
        output['generator'] = raw.get('generator', 'graspgen')
        output['preferred_direction'] = raw.get('preferred_direction')
        output['grasp_type'] = raw.get('grasp_type')
        output['candidate_limit'] = raw.get('candidate_limit', 6)
        if raw.get('generator') == 'ensemble':
            output['ensemble_sources'] = [{k: deepcopy(r[k]) for k in
                ('ensemble_source', 'requested_slots', 'executable_count', 'diagnostic_count',
                 'missing_count', 'reason_codes') if k in r} for r in raw.get('ensemble_sources', [])]
        output['diagnostic_candidates'] = [public_diagnostic(c) for c in raw.get('diagnostic_candidates', [])]
        output['source_reports'] = [{k: deepcopy(r[k]) for k in
            ('source_view','ensemble_source','generated_budget','reason_code','path_rejections','model_filter',
             'accepted_count','raw_proposals_generated','reserved_slots','error_type','elapsed_s','image_refs') if k in r}
            for r in raw.get('source_reports', [])]
        output['task_budget'] = {k: deepcopy(raw.get('task_budget', {}).get(k)) for k in
            _GRASP_BUDGET_FIELDS if k in raw.get('task_budget', {})}
    elif 'validation_feedback' in raw:
        output['validation_feedback'] = {k: deepcopy(v) for k,v in raw['validation_feedback'].items() if k in
            ('generated_count','geometry_passed_count','ik_checked_count','route_passed_count',
             'rejection_counts','diagnostic_image_refs','diagnostic_images_executable','scope')}
    if tool == 'place_candidates':
        generation = raw.get('candidate_generation', {})
        if (isinstance(generation, dict) and generation.get('status') in ('requested', 'reused')
                and isinstance(generation.get('destination_ref'), str)):
            output['candidate_generation'] = dict(status=generation['status'],
                destination_ref=_text(generation['destination_ref']), instruction_scope='place_evaluation')
    return output


def public_diagnostic(raw):
    """A rejected pose is inspectable evidence, never an execution capability."""
    if raw.get('executable') is not False:
        raise ValueError('diagnostic candidate must be explicitly nonexecutable')
    output = {key: _text(raw[key]) for key in ('candidate_ref', 'source_view', 'reason_code')}
    output.update(executable=False, image_refs=[_text(ref) for ref in raw['image_refs']])
    if raw.get('ensemble_source'): output['ensemble_source'] = _text(raw['ensemble_source'])
    feedback = {}
    for key, value in raw.get('validation_feedback', {}).items():
        if key in ('kind', 'segment') and isinstance(value, str):
            feedback[key] = _text(value)
        elif key == 'robotgeom':
            if isinstance(value, str): feedback[key] = _text(value)
            elif isinstance(value, list): feedback[key] = [_text(item) for item in value]
        elif key in ('sample_count', 'clearance_m', 'colliding_points', 'geom_scene_clearance_m'):
            if type(value) in (int, float) and math.isfinite(value): feedback[key] = value
        elif key == 'geom_world_stationary' and type(value) is bool:
            feedback[key] = value
    if raw.get('validation_feedback', {}).get('kind') == 'planning':
        feedback.pop('segment', None)
    feedback.update(public_planning_feedback(raw.get('validation_feedback')))
    output['validation_feedback'] = feedback
    return output


class IntentOrchestrator(ReviewDrivenOrchestrator):
    def _optional_arguments(self, tool):
        return ('preferred_direction', 'batch_size') if tool == 'delegate_grasp' else super()._optional_arguments(tool)

    @staticmethod
    def _grasp_bundle_key(args):
        from src.tools.grasp.preference import resolve_approach
        direction, family = resolve_approach(args.get('preferred_direction'))
        if args.get('batch_size') is not None:
            return (args['point_ref'], direction, family, args['batch_size'])
        if family is not None:
            return (args['point_ref'], direction, family)
        return (args['point_ref'], direction) if direction else args['point_ref']

    def _cached_grasp_bundle(self, args):
        return self._candidate_bundles.get(self._grasp_bundle_key(args), {})

    def __init__(self, *args, **kwargs):
        kwargs['contact_manipulation'] = True
        super().__init__(*args, **kwargs)
        self.last_feedback = None
        self.last_operation_feedback = None
        self.approached_targets = set()
        self._task_stack = []
        self._diagnostic_handoff_stack = []
        self._precondition_handoff_stack = []
        self._place_generation_stack = []
        self._diagnostic_candidates = {}
        self.target_reference = None
        self._candidate_handoffs = {}
        self._candidate_inspections = {}
        self._selection_instructions = {}
        self._last_tool_failure = None
        self._last_grasp_execution_ref = None
        self._last_execution_ref = None
        self.place_agent.prompt = self.place_agent.prompt.replace(
            'Photos were captured BEFORE pickup', 'Photos were captured AFTER pickup')
        self.place_agent.prompt += (
            ' candidate_generation reports whether the underlying AnyPlace prediction pool was requested or reused. '
            'A changed placement instruction changes your evaluation, not the AnyPlace model input. '
            'Repeated delegation with the same held-object selection and destination can reuse predictions even '
            'when candidate references are new; route checks may run again. Requested does not guarantee a valid '
            'candidate. Explain unsuitable geometry to Prime; do not describe an instruction-only retry as resampling.')

    def _tools(self, role, task):
        if role == 'prime':
            return ('review_observation', 'delegate_point', 'delegate_waypoint', 'validate_view', 'execute_view',
                    'set_grasp_mode', 'delegate_grasp', 'delegate_refiner', 'validate_grasp', 'execute_grasp',
                    'delegate_destination', 'save_destination', 'delegate_place', 'validate_place',
                    'execute_place_candidate', 'close_for_push', 'release', 'turn', 'finish')
        if role == 'point' and task.get('waypoint_task'):
            return ('propose_waypoint', 'propose_downward_waypoint', 'shift_waypoint', 'finish')
        return super()._tools(role, task)

    def _prompt(self, role):
        from src.agent.intent_prompts import intent_prompt
        return intent_prompt(role) + (
            ' runtime_context contains recorded command state, not proof of physical retention. '
            'delegated_intent and selection_context.model_intent preserve model instructions, not verified identity. '
            'Keep the delegated target; compare prior_candidate_inspection with current RGB. '
            'A prior failure is reference evidence, not proof the current target is impossible. '
            'delegate_grasp accepts optional batch_size 1–6 for a fresh pool; omission keeps the existing default. '
            'The task slot budget, raw proposals, published candidates and execution attempts are different counts.')

    def _loop(self, role, task, parent_id=None):
        diagnostics = {}
        if role in ('grasp', 'refiner'):
            initial = task.get('candidate_bundle', {}).get('diagnostic_candidates', [])
            if task.get('diagnostic_candidate'):
                initial = [*initial, task['diagnostic_candidate']]
            for raw in initial:
                clean = public_diagnostic(raw)
                diagnostics[clean['candidate_ref']] = (self._epoch, clean)
        self._task_stack.append(deepcopy(task))
        self._diagnostic_handoff_stack.append(diagnostics)
        self._precondition_handoff_stack.append({})
        try:
            status, result = super()._loop(role, task, parent_id)
            selected = result.get('candidate_ref')
            if role in ('grasp', 'refiner') and (not selected or selected in diagnostics):
                current = {ref: value for ref, (epoch, value) in diagnostics.items()
                           if epoch == self._epoch}
                refs = ([selected] if selected in current else [])
                refs += [ref for ref in reversed(current) if ref not in refs]
                if refs:
                    result = {**result, 'backend_diagnostics':
                              [deepcopy(current[ref]) for ref in refs[:2]]}
            rejection = self._precondition_handoff_stack[-1]
            if (role == 'point' and task.get('waypoint_task') and not result.get('waypoint_ref')
                    and rejection.get('epoch') == self._epoch):
                result = {**result, 'backend_preconditions': [deepcopy(rejection['feedback'])]}
            return status, result
        finally:
            self._precondition_handoff_stack.pop()
            self._diagnostic_handoff_stack.pop()
            self._task_stack.pop()

    @staticmethod
    def _argument(key, value):
        if key == 'batch_size':
            if type(value) is not int or not 1 <= value <= 6:
                raise BoundaryError('batch_size must be an integer from 1 to 6')
            return value
        if key in ('height_offset_m', 'dx_m', 'dy_m', 'dz_m'):
            if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > .5:
                raise BoundaryError('waypoint offsets must be finite metres within 0.5 m')
            return float(value)
        if key == 'evidence_image_refs':
            if not isinstance(value, list) or not 1 <= len(value) <= 8:
                raise BoundaryError('supply one to eight evidence image references')
            return [_text(x) for x in value]
        return ReviewDrivenOrchestrator._argument(key, value)

    def _evidence(self, refs, scope):
        refs = self._argument('evidence_image_refs', refs)
        allowed = set(scope.get('evidence_images', ()))
        allowed.update(image_refs(self._current_observation or {}))
        if self._task_stack: allowed.update(image_refs(self._task_stack[-1]))
        if not set(refs) <= allowed:
            raise BoundaryError('evidence must reference images supplied to this role')
        return refs

    def _finish(self, role, args, scope):
        if role == 'prime':
            if (args.get('status') not in ('completed', 'failed', 'unknown')
                    or not set(args) <= {'status', 'reason', 'evidence_image_refs'}):
                raise BoundaryError('Prime finish requires status and optional reason/evidence_image_refs')
            output = {'status': args['status'], 'verified': False}
            if 'reason' in args: output['reason'] = _text(args['reason'])
            if 'evidence_image_refs' in args:
                output['evidence_image_refs'] = self._evidence(args['evidence_image_refs'], scope)
            return output
        clean = dict(args)
        evidence_raw = clean.pop('evidence_image_refs', None)
        evidence = self._evidence(evidence_raw, scope) if evidence_raw is not None else []
        recommendation = _text(clean.pop('recommendation')) if 'recommendation' in clean else 'none'
        reason = _text(clean['reason']) if clean.get('reason') else 'No reason supplied.'
        clean['reason'] = reason
        if role == 'grasp':
            if 'status' not in clean and 'decision' not in clean:
                clean['decision'] = 'accepted' if clean.get('candidate_ref') else 'failed'
            status = clean.pop('status', None)
            decision = 'accepted' if status == 'success' else status
            if decision is not None:
                if clean.get('decision', decision) != decision:
                    raise BoundaryError('status and decision disagree; return one assessment')
                clean['decision'] = decision
        if role != 'grasp' and 'status' not in clean and not any(clean.get(k) for k in ('point_ref', 'waypoint_ref', 'candidate_ref')):
            clean['status'] = 'failed'
        assessment = clean.get('decision') if role == 'grasp' else clean.get('status')
        if assessment in ('failed', 'needs_observation'):
            # Reporting a problem is not selection approval. Preserve explanatory
            # references separately so inherited delegation cannot register them.
            related = {key: clean.pop(key) for key in ('candidate_ref', 'point_ref', 'waypoint_ref')
                       if key in clean}
            if role != 'grasp': clean['status'] = 'failed'
            result = super()._finish(role, clean, scope)
            return {**result, 'reason': reason, 'evidence_image_refs': evidence,
                    'recommendation': recommendation, **({'related_refs': related} if related else {})}
        ref = clean.get('candidate_ref')
        if ref in self._diagnostic_candidates and (
                role == 'refiner' or (role == 'grasp' and clean.get('decision') == 'accepted')):
            raise BoundaryError('diagnostic pose is nonexecutable; request refinement or report failed')
        if role in ('point', 'place', 'refiner'):
            keys = ('point_ref', 'waypoint_ref') if role == 'point' else ('candidate_ref', 'waypoint_ref') if role == 'refiner' else ('candidate_ref',)
            refs = [key for key in keys if key in clean]
            status = clean.get('status')
            if status == 'success':
                if len(refs) != 1:
                    raise BoundaryError('selection success requires exactly one current selection reference')
                # This alias approves a selection only. It establishes neither
                # physical execution nor task success; all reference guards remain.
                clean.pop('status')
            elif status is not None and status != 'failed':
                raise BoundaryError('selection status must be success or failed; omit it for canonical success')
            if role == 'point' and refs:
                if len(refs) != 1 or set(clean) != {refs[0], 'reason'}:
                    raise BoundaryError('Pointer success needs one point_ref or waypoint_ref; explanation fields are optional')
                key = refs[0]
                records = scope['points'] if key == 'point_ref' else scope['waypoints']
                if records.get(clean[key]) != self._epoch:
                    raise BoundaryError(key + ' must be the current opaque selection reference returned by its tool, not an image_ref')
        if role == 'point' and ('point_ref' in clean or 'waypoint_ref' in clean): clean.pop('reason')
        result = super()._finish(role, clean, scope)
        if role == 'point' and result.get('point_ref'):
            task = self._task_stack[-1] if self._task_stack else {}
            if not task.get('destination_task') and not task.get('waypoint_task'):
                # Store only the image evidence actually approved by Pointer.
                # It survives motion solely as historical identity context.
                selection = self._target_evidence.get(result['point_ref'], {})
                self.target_reference = None
                refs = selection.get('image_refs', [])[:2]
                if refs and selection.get('observation_id'):
                    self.target_reference = dict(observation_id=_text(selection['observation_id']),
                        image_refs=[_text(ref) for ref in refs], reference_only=True,
                        scope='prior Pointer-approved pick selection; compare with current RGB, not an executable selection')
        return {**result, 'reason': reason, 'evidence_image_refs': evidence, 'recommendation': recommendation}

    def _request_context(self, role, task, step, limit):
        if role != 'prime': return super()._request_context(role, task, step, limit)
        def counter(used, maximum):
            return {'used': used, 'limit': maximum, 'remaining': max(0, maximum - used)}
        budget = {
            'tool_calls': counter(self._calls, self.budgets.max_tool_calls),
            'delegations': counter(self._delegations, self.budgets.max_delegations),
            'prime_turns': {**counter(step, limit), 'includes_current_response': True}}
        grasp_budget = getattr(self.backend, 'grasp_budget', None)
        if callable(grasp_budget):
            raw = grasp_budget()
            budget['grasp_generation'] = {key: deepcopy(raw[key]) for key in
                _GRASP_BUDGET_FIELDS if key in raw}
        simulation_budget = getattr(self.backend, 'simulation_budget', None)
        if callable(simulation_budget):
            current = simulation_budget()
            if current is not None:
                budget['simulation_time'] = deepcopy(current)
        return {'execution_budget': budget}

    def _delegate(self, role, task, scope, sid):
        task = deepcopy(task)
        current = self._current_observation or {}
        context = dict(epoch=self._epoch, observation_id=current.get('observation_id'),
            role_stage=('destination_selection' if task.get('destination_task') else
                        'waypoint_selection' if task.get('waypoint_task') else task.get('stage', role)),
            grasp_mode=getattr(self.backend, 'grasp_mode', None),
            grasp_command_active=getattr(self.backend, 'held_plan', None) is not None)
        original = self._task_stack[0].get('instruction') if self._task_stack else None
        intent = dict(source='model_delegation; intent, not verified object identity',
                      instruction=task.get('instruction'))
        point = task.get('point_ref')
        if point in self._selection_instructions:
            intent['selection_instruction'] = self._selection_instructions[point]
        if original:
            task['original_objective'] = original
        ref = task.get('candidate_ref') or task.get('origin_candidate_ref')
        inherited = self._candidate_handoffs.get(ref)
        if inherited and (inherited['epoch'] == self._epoch or task.get('inflight_refinement')):
            task['selection_context'] = {**deepcopy(inherited), 'reference_only': True,
                                        'current_epoch_matches': inherited['epoch'] == self._epoch}
        if role == 'grasp':
            inherited = dict(epoch=self._epoch, observation_id=current.get('observation_id'),
                             point_ref=point, model_intent=intent)
            for candidate in [*task.get('candidate_bundle', {}).get('candidates', []),
                              *task.get('candidate_bundle', {}).get('diagnostic_candidates', [])]:
                self._candidate_handoffs[candidate['candidate_ref']] = deepcopy(inherited)
        inspection = self._candidate_inspections.get(ref)
        if inspection and (inspection['epoch'] == self._epoch or task.get('inflight_refinement')):
            task['prior_candidate_inspection'] = {**deepcopy(inspection), 'reference_only': True,
                                                 'current_epoch_matches': inspection['epoch'] == self._epoch}
        task['runtime_context'] = context
        task['delegated_intent'] = intent
        if self._last_tool_failure is not None:
            failure = deepcopy(self._last_tool_failure)
            failure['current_epoch_matches'] = failure['epoch'] == self._epoch
            task['last_tool_failure'] = failure
        if (role == 'point' and not task.get('waypoint_task') and not task.get('destination_task')
                and self.target_reference is not None):
            task = {**task, 'target_reference': deepcopy(self.target_reference)}
        ref = task.get('candidate_ref')
        if role == 'refiner' and ref in self._diagnostic_candidates:
            epoch, diagnostic = self._diagnostic_candidates[ref]
            if epoch != self._epoch: raise BoundaryError('diagnostic pose is stale')
            task = {**task, 'diagnostic_candidate': deepcopy(diagnostic)}
        task = {**task, 'previous_feedback': deepcopy(self.last_feedback)}
        if self.last_operation_feedback is not None:
            record = deepcopy(self.last_operation_feedback)
            current = self._current_observation or {}
            matches = (record['epoch'] == self._epoch and record['observation_id'] is not None
                       and record['observation_id'] == current.get('observation_id'))
            record.update(current_observation_id=current.get('observation_id'), observation_matches_current=matches)
            record['evidence_image_refs'] = ([ref for ref in record['evidence_image_refs']
                                             if ref in image_refs(current)] if matches else [])
            task['last_operation_feedback'] = record
        generation = {}
        if role == 'place': self._place_generation_stack.append(generation)
        try:
            output = super()._delegate(role, task, scope, sid)
        finally:
            if role == 'place': self._place_generation_stack.pop()
        if generation: output['candidate_generation'] = deepcopy(generation)
        if role == 'point' and output['result'].get('point_ref'):
            self._selection_instructions[output['result']['point_ref']] = task.get('instruction')
        selected = output['result'].get('candidate_ref')
        if selected and inherited:
            self._candidate_handoffs[selected] = deepcopy(inherited)
        self.last_feedback = deepcopy(output['result'])
        self._event('agent_handoff', sid, from_role=role, to_role='prime', assessment=self.last_feedback)
        return output

    def _dispatch(self, role, tool, args, scope, task, sid):
        input_observation_id = (self._current_observation or {}).get('observation_id')
        # Record an existing model assertion, even if the requested placement is
        # later refused. This neither verifies a hold nor adds a model call.
        if role == 'prime' and tool == 'delegate_place' and args.get('hold_assessment') in ('held', 'empty', 'uncertain'):
            assessment = dict(source='agent_visual_assessment', role=role, tool=tool,
                assessment=args['hold_assessment'], observation_id=input_observation_id,
                evidence_image_refs=image_refs(self._current_observation or {}),
                grasp_execution_ref=self._last_grasp_execution_ref,
                latest_execution_ref=self._last_execution_ref)
            self._event('agent_hold_assessment', sid, **assessment)
            directory = getattr(self.backend, 'output_dir', None)
            if directory is not None:
                from pathlib import Path
                from src.utils.logging_utils import append_json
                append_json(Path(directory)/'agent_hold_assessments.jsonl', dict(session_id=sid, **assessment))
        structured_failure = None
        try:
            output = self._intent_dispatch(role, tool, args, scope, task, sid)
        except (BoundaryError, BudgetExceeded, AuditError, EpisodeEnded):
            raise
        except ActionPreconditionError as exc:
            if getattr(self, 'structured_tool_errors', False):
                from src.core.errors import structured_error
                structured_failure = structured_error(exc, phase='action_precondition', tool=tool)
            output = {**exc.public_feedback(), 'operation': tool,
                      'evidence_image_refs': image_refs(self._current_observation or {})}
            if structured_failure is not None:
                output['error_details'] = structured_failure
            if self._precondition_handoff_stack:
                self._precondition_handoff_stack[-1].update(epoch=self._epoch, feedback=deepcopy(output))
            self._event('action_precondition_rejected', sid, role=role, tool=tool,
                        reason_code=output['reason_code'])
        except Exception as exc:
            if getattr(self, 'structured_tool_errors', False):
                from src.core.errors import structured_error
                structured_failure = structured_error(exc, phase='tool_dispatch', tool=tool)
            self._backend_failure(sid, role, tool, exc)
            self._event('backend_error', sid, role=role, tool=tool, exception_type=type(exc).__name__)
            contract_error = isinstance(exc, (TypeError, AttributeError, NameError))
            output = {'error':'backend_contract_error' if contract_error else 'tool_timeout' if isinstance(exc, TimeoutError) else 'tool_operation_failed',
                      'exception_type':type(exc).__name__,
                      'operation':tool, 'evidence_image_refs':image_refs(self._current_observation or {}),
                      'reason':('The backend interface failed. Report the operation and exception type to Prime; '
                                'changing object selection or geometry is not an established repair.' if contract_error else
                                'The tool did not complete. This does not establish that the object or view is unsuitable.')}
        if getattr(self, 'structured_tool_errors', False):
            from src.core.errors import add_result_error
            if structured_failure is not None:
                output['error_details'] = structured_failure
            add_result_error(output, phase='tool_result', tool=tool)
        if role == 'prime' and (tool in _OPERATION_TOOLS or output.get('error')):
            output['robot_state'] = {
                'grasp_mode': getattr(self.backend, 'grasp_mode', None),
                'grasp_command_active': getattr(self.backend, 'held_plan', None) is not None,
                'closed_pusher_command': bool(getattr(self.backend, 'closed_push', False)),
                'observation_id': (self._current_observation or {}).get('observation_id'),
            }
        if role == 'prime' and tool in _OPERATION_TOOLS:
            current = self._current_observation or {}
            subject = {}
            for key in ('waypoint_ref', 'candidate_ref', 'execution_ref'):
                value = args.get(key, output.get(key))
                if isinstance(value, str) and value.strip() and len(value) <= 8192: subject[key] = value
            target_update = output.get('target_update', {})
            if isinstance(target_update, dict):
                target = target_update.get('target_ref')
                if isinstance(target, str) and target.strip() and len(target) <= 8192:
                    subject['target_ref'] = target
            self.last_operation_feedback = dict(operation=tool, epoch=self._epoch,
                input_observation_id=input_observation_id, observation_id=current.get('observation_id'),
                provenance='backend_operation_result', reference_only=True, subject=subject,
                result=public_operation_result(output), evidence_image_refs=image_refs(current))
        scope.setdefault('evidence_images', set()).update(image_refs(output))
        if tool in ('inspect_candidate', 'preview_candidate') and args.get('candidate_ref') and not output.get('error'):
            prior = self._candidate_inspections.get(args['candidate_ref'], {})
            refs = image_refs(output)
            if tool == 'preview_candidate' and prior.get('epoch') == self._epoch:
                refs = list(dict.fromkeys([*prior['image_refs'], *refs]))
            self._candidate_inspections[args['candidate_ref']] = dict(epoch=self._epoch,
                candidate_ref=args['candidate_ref'], source='public_tool_result',
                image_refs=refs[:6])
        if output.get('error') or output.get('accepted') is False or output.get('status') == 'failed':
            self._last_tool_failure = dict(epoch=self._epoch, role=role, operation=tool,
                candidate_ref=args.get('candidate_ref'), source='public_tool_result',
                result=public_operation_result(output))
        if tool in _OPERATION_TOOLS and output.get('execution_ref'):
            self._last_execution_ref = output['execution_ref']
            if tool == 'execute_grasp': self._last_grasp_execution_ref = output['execution_ref']
            feedback = output.get('execution_feedback') or {}
            facts = {key: output[key] for key in ('state_changed', 'terminal') if type(output.get(key)) is bool}
            if type(feedback.get('release_commanded')) is bool:
                facts['release_commanded'] = feedback['release_commanded']
            elif tool == 'release' and output.get('status') == 'succeeded':
                facts['release_commanded'] = True
            post = output.get('observation') or {}
            if post.get('observation_id'):
                facts.update(observation_id=post['observation_id'], image_refs=image_refs(post))
            self._event('execution_diagnostic', sid, operation=tool,
                execution_ref=output['execution_ref'], candidate_ref=args.get('candidate_ref'),
                command_status=output.get('status', 'unknown'),
                scope='runtime command result and post-execution observation; no physical hold or native outcome inference',
                **facts)
        if (role in ('grasp', 'refiner') and self._diagnostic_handoff_stack
                and isinstance(output, dict) and output.get('executable') is False
                and output.get('candidate_ref') in self._diagnostic_candidates):
            clean = public_diagnostic(output)
            records = self._diagnostic_handoff_stack[-1]
            records.pop(clean['candidate_ref'], None)
            records[clean['candidate_ref']] = (self._epoch, clean)
        return output

    def _intent_dispatch(self, role, tool, args, scope, task, sid):
        if tool == 'save_destination':
            # Preserve the inherited current Pointer-approval guard, while
            # allowing trusted typed feedback to reach Intent's handler.
            return super()._dispatch_checked(role, tool, args, scope, task, sid)
        if tool == 'review_observation':
            if role != 'prime': raise BoundaryError('saved observation review is Prime-only')
            raw = self.backend.review_observation(**args)
            output = public_result('observe', raw)
            if (output['observation_id'] != args['observation_id']
                    or raw.get('reference_only') is not True
                    or raw.get('current_observation_id') != scope.get('latest_observation')):
                raise BoundaryError('historical review must preserve the current observation')
            output.update(reference_only=True,
                historical=output['observation_id'] != scope['latest_observation'],
                current_observation_id=scope['latest_observation'],
                scope='saved RGB from this episode; no recapture or state change')
            # Do not remember this as a new/current observation or grant stale
            # observation IDs to any action or child selection scope.
            return output
        if role == 'grasp':
            # The base loop seeds normal candidates. Diagnostics remain a separate
            # public list, with ownership granted only inside this delegation.
            for diagnostic in task.get('candidate_bundle', {}).get('diagnostic_candidates', []):
                ref = diagnostic['candidate_ref']
                record = self._diagnostic_candidates.get(ref)
                if record and record[0] == self._epoch:
                    scope['candidates'][ref] = self._epoch
        ref = args.get('candidate_ref')
        diagnostic = self._diagnostic_candidates.get(ref)
        if diagnostic and tool in ('validate_grasp', 'execute_grasp'):
            raise BoundaryError('diagnostic pose has no executable route; refine to a passing candidate first')
        if diagnostic and tool in ('inspect_candidate', 'preview_candidate'):
            if diagnostic[0] != self._epoch or scope['candidates'].get(ref) != self._epoch:
                raise BoundaryError('diagnostic candidate outside current session')
            raw = getattr(self.backend, tool)(**args)
            output = public_diagnostic({**diagnostic[1], **raw})
            if output['candidate_ref'] != ref: raise BoundaryError('inspection changed diagnostic identity')
            scope['inspected_candidates'][ref] = self._epoch
            return output
        if tool == 'adjust_grasp':
            if role != 'refiner' or scope['candidates'].get(ref) != self._epoch:
                raise BoundaryError('candidate outside Refiner session')
            if diagnostic and diagnostic[0] != self._epoch:
                raise BoundaryError('diagnostic pose is stale')
            raw = self.backend.adjust_grasp(**args)
            if not raw.get('accepted', True):
                if raw.get('candidate_ref'):
                    output = public_diagnostic(raw)
                    new_ref = output['candidate_ref']
                    if new_ref == ref or new_ref in self._diagnostic_candidates or new_ref in scope['candidates']:
                        raise BoundaryError('failed refinement must return a new diagnostic identity')
                    self._diagnostic_candidates[new_ref] = (self._epoch, deepcopy(output))
                    scope['candidates'][new_ref] = self._epoch
                    return dict(accepted=False, **output)
                output = {'accepted': False, 'reason_code': _text(raw.get('reason_code', 'refinement_rejected'))}
                if 'error_details' in raw:
                    from src.core.errors import public_error_details
                    details = public_error_details(raw['error_details'])
                    if details is not None:
                        output['error_details'] = details
                return output
            if raw.get('executable') is False or raw.get('candidate_ref') in self._diagnostic_candidates:
                raise BoundaryError('refinement did not produce an executable candidate')
            output = public_result('refine_candidate', raw)
            output.update(accepted=True, executable=True)
            scope['candidates'][output['candidate_ref']] = self._epoch
            scope['inspected_candidates'][output['candidate_ref']] = self._epoch
            return output
        if tool == 'delegate_waypoint':
            if self.target_ref is None: raise BoundaryError('identify a visible target first')
            return self._delegate('point', {**args, 'waypoint_task': True, 'target_ref': self.target_ref}, scope, sid)
        if tool in ('propose_waypoint', 'propose_downward_waypoint', 'shift_waypoint'):
            if role != 'point' or not task.get('waypoint_task') or args['observation_id'] != scope['latest_observation']:
                raise BoundaryError('waypoint requires delegated current observation')
            output = getattr(self.backend, tool)(**args, target_ref=task['target_ref'])
            scope['waypoints'][output['waypoint_ref']] = self._epoch
            return output
        if tool in ('validate_view', 'execute_view'):
            if self._waypoint_refs.get(args['waypoint_ref']) != self._epoch:
                raise ActionPreconditionError('waypoint_requires_current_state')
            if tool == 'validate_view':
                output = self.backend.validate_view(**args)
                if output['accepted']: scope.setdefault('view_validations', {})[output['validation_ref']] = args['waypoint_ref']
                return output
            if scope.get('view_validations', {}).pop(args['validation_ref'], None) != args['waypoint_ref']:
                raise BoundaryError('execute exactly the validated waypoint')
            target = self.target_ref
            try: output = self.backend.execute_view(**args)
            finally:
                if self.backend.epoch != self._epoch:
                    self._current_observation = None
                self._epoch = self.backend.epoch
            self._remember_observation(output, scope, invalidate=True)
            if output.get('view_status') == 'achieved': self.approached_targets.add(target)
            update = output.get('target_update', {})
            if update.get('tracking_status') == 'retained':
                self.current_point = update['point_ref']; self._point_refs[self.current_point] = self._epoch
                self._target_evidence[self.current_point] = update
            return output
        if tool == 'set_grasp_mode':
            output = self.backend.set_grasp_mode(**args)
            if output.get('candidates_invalidated'):
                self._candidate_bundles.clear()
                self._diagnostic_candidates.clear()
                scope['candidates'].clear(); scope['validations'].clear()
                scope.pop('grasp_decisions', None)
            return output
        if tool == 'delegate_grasp':
            ref = args['point_ref']
            if self._point_refs.get(ref) != self._epoch: raise BoundaryError('use current Pointer-approved target')
            from src.tools.grasp.preference import resolve_approach
            direction, family = resolve_approach(args.get('preferred_direction'))
            key = self._grasp_bundle_key(args)
            generation = 'reused' if key in self._candidate_bundles else 'requested'
            if generation == 'requested':
                options = {'preferred_direction': direction} if direction else {}
                if 'batch_size' in args: options['batch_size'] = args['batch_size']
                if family is not None: options['grasp_type'] = family
                self._candidate_bundles[key] = public_bundle('grasp_candidates',
                    self.backend.grasp_candidates(point_ref=ref, **options))
            bundle = deepcopy(self._candidate_bundles[key])
            for diagnostic in bundle.get('diagnostic_candidates', []):
                self._diagnostic_candidates[diagnostic['candidate_ref']] = (self._epoch, deepcopy(diagnostic))
            output = self._delegate('grasp', {**args, 'candidate_bundle': bundle,
                'target_evidence': self._target_evidence.get(ref, {})}, scope, sid)
            output['candidate_generation'] = dict(status=generation, point_ref=ref,
                preferred_direction=args.get('preferred_direction'), grasp_type=family or ('face' if direction else None),
                generator=bundle.get('generator', 'unknown'),
                candidate_limit=bundle.get('candidate_limit', 6), instruction_scope='selector_evaluation')
            choice = output['result']
            if choice.get('decision') == 'needs_refinement': self.refinement_choices.add(choice['candidate_ref'])
            scope.setdefault('grasp_decisions', {})[choice.get('candidate_ref')] = choice.get('decision')
            return output
        if role == 'place' and tool == 'place_candidates':
            if any(args[k] != task[k] for k in ('destination_ref', 'hold_assessment', 'destination_assessment')):
                raise BoundaryError('use delegated destination and current assessments')
            if 'place_bundle' not in scope:
                raw = self.backend.place_candidates(**args)
                scope['place_bundle'] = public_bundle('place_candidates', raw)
                scope['candidates'].update({c['candidate_ref']: self._epoch for c in raw.get('candidates', [])})
            if self._place_generation_stack:
                self._place_generation_stack[-1].update(scope['place_bundle'].get('candidate_generation', {}))
            return scope['place_bundle']
        if tool in ('close_for_push', 'release', 'turn'):
            if (tool == 'release' and role == 'prime' and getattr(self, '_release_refused', False)
                    and getattr(self.backend, 'held_plan', None) is not None):
                # The tracking check refused to open away from the release goal;
                # a plain release must not become the way around that refusal.
                raise ActionPreconditionError('release_requires_placement_retry')
            try: output = getattr(self.backend, tool)(**args)
            finally:
                if self.backend.epoch != self._epoch:
                    self._current_observation = None
                self._epoch = self.backend.epoch
            output['observation'] = self._environment_observation(scope, sid, refresh=self._current_observation is None)
            if tool == 'release':
                self.approached_targets.clear()
                if output.get('status') == 'succeeded': self.target_reference = None
            return output
        if tool == 'execute_place_candidate':
            ref = args['candidate_ref']
            if scope['candidates'].get(ref) != self._epoch or scope['validations'].pop(args['validation_ref'], None) != ref:
                raise BoundaryError('execute the current Place choice with its matching validation')
            try:
                raw = self.backend.execute_place_candidate(**args)
                output = public_result(tool, raw)
                for key in ('execution_feedback',):
                    if key in raw: output[key] = deepcopy(raw[key])
            finally:
                if self.backend.epoch != self._epoch:
                    self._current_observation = None
                self._epoch = self.backend.epoch
            output['observation'] = self._environment_observation(scope, sid, refresh=self._current_observation is None)
            feedback = output.get('execution_feedback') or {}
            self._release_refused = (output.get('status') != 'succeeded'
                                     and feedback.get('release_commanded') is False)
        else:
            output = super()._dispatch(role, tool, args, scope, task, sid)
        if tool == 'delegate_place' and role == 'prime':
            self._release_refused = False
        if tool == 'execute_place_candidate' and output.get('status') == 'succeeded':
            # Preserve the world and task rubric; permit next-object selection or repair.
            self.backend.grasp_attempted = False
            self.backend.held_plan = None
            self.approached_targets.clear()
            self.target_reference = None
            for name in ('destinations', 'destination_anchors', 'place_predictions', 'place_validations', 'placement_pools'):
                getattr(self.backend, name).clear()
            self.destinations.clear()
        return output
