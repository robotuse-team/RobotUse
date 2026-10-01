"""RobotUse role orchestration and checked delegation boundaries."""
from copy import deepcopy
import math

from src.backend.controller import BoundaryError
from src.backend.delegation import DelegationOrchestrator
from src.tools.names import TOOL_SCHEMA_ALIASES, public_tool_name
from src.tools.pose_editor.adapter import PoseEditorFeatures


class AgentOrchestrator(DelegationOrchestrator):
    interaction_features = PoseEditorFeatures(place_rotation=False, held_observation=True)
    requested_refine_routes = True
    unrestricted_pose_rotation = True
    unrestricted_pose_translation = True
    clicked_grasp_candidates = True

    def __init__(self, *args, **kwargs):
        from src.runtime.bootstrap import load_tool_registry
        self.tool_registry = load_tool_registry()
        super().__init__(*args, **kwargs)
        self.factory.tool_registry = self.tool_registry
        # Per-factory schemas keep grasp/place contracts isolated between sessions.
        self.factory.tool_schema_aliases = dict(self.tool_registry.aliases)
        self.factory.requested_refine_routes = self.requested_refine_routes
        self.factory.unrestricted_pose_rotation = self.unrestricted_pose_rotation
        self.factory.unrestricted_pose_translation = self.unrestricted_pose_translation
        self.factory.clicked_grasp_candidates = self.clicked_grasp_candidates

    def _required_arguments(self, tool):
        return self.tool_registry.required_arguments(tool)

    def _optional_arguments(self, tool):
        return self.tool_registry.optional_arguments(tool, context=self)

    def _tool_argument(self, tool, key, value):
        return self.tool_registry.validate_argument(tool, key, value, context=self)

    def _finish(self, role, args, scope):
        if role == 'place' and args.get('status') == 'needs_refinement':
            # Reuse the normal inspected/current-reference checks. This status
            # remains a request, and _delegate must obtain Refiner approval.
            result = super()._finish(role, {**args, 'status': 'success'}, scope)
            return {**result, 'status': 'needs_refinement'}
        return super()._finish(role, args, scope)

    def _dispatch(self, role, tool, args, scope, task, sid):
        from .tool_executor import execute_tool
        inherited_dispatch = super()._dispatch
        output = execute_tool(self.tool_registry, tool, args,
            dispatch=lambda name, arguments: inherited_dispatch(role, name, arguments, scope, task, sid))
        if isinstance(output.get('operation'), str):
            output = {**output, 'operation': public_tool_name(output['operation'])}
        from src.core.planning_feedback import public_planning_feedback
        planning = public_planning_feedback(output.get('validation_feedback'))
        if not planning and isinstance(output.get('validation'), dict):
            planning = public_planning_feedback(output['validation'].get('validation_feedback'))
        if not planning:
            planning = public_planning_feedback(output.get('planning_feedback'))
        if planning and getattr(self, '_planning_stack', None):
            record = dict(tool=public_tool_name(tool), reference_only=True,
                candidate_ref=output.get('candidate_ref'), waypoint_ref=output.get('waypoint_ref'),
                planning_feedback=planning)
            records = self._planning_stack[-1]
            if not records or records[-1] != record:
                records.append(record)
                del records[:-5]
        return output

    def _event(self, kind, session_id, **fields):
        # Includes execution calls owned internally by the inherited skill.
        if 'tool' in fields:
            fields['tool'] = public_tool_name(fields['tool'])
        return super()._event(kind, session_id, **fields)

    def _tools(self, role, task):
        if role == 'refiner' and task.get('place_refinement'):
            names = ('inspect_place_candidate', 'adjust_place', 'finish')
        else:
            names = tuple(public_tool_name(tool) for tool in super()._tools(role, task))
        return self.tool_registry.require(names)

    def _prompt(self, role):
        from src.agent.prompts import extend_pose_prompt
        return extend_pose_prompt(role, super()._prompt(role))

    def _delegate(self, role, task, scope, sid):
        if role == 'refiner' and task.get('inflight_refinement'):
            details = deepcopy(self.backend.inflight_feedback)
            task = {**task, **{k: v for k, v in details.items() if k != 'image_refs'}}
        if not hasattr(self, '_planning_stack'):
            self._planning_stack = []
        records = []
        self._planning_stack.append(records)
        try:
            output = super()._delegate(role, task, scope, sid)
        finally:
            self._planning_stack.pop()
        if records:
            output['planning_diagnostics'] = deepcopy(records)
        requested = self._refinement_requested(output)
        if role == 'place' and (self.auto_refine_routes or requested):
            selected = self._approved_choice(output)
            if selected:
                if scope['candidates'].get(selected) != self._epoch:
                    raise BoundaryError('review a current delegated placement')
                inspection = self.backend.explicit_inspect_place(candidate_ref=selected)
                review = self._delegate('refiner', dict(
                    instruction=task['instruction'] + ' Review the pending placement before movement. Selection reason: '
                                + str(output['result'].get('reason', 'not supplied')),
                    candidate_ref=selected, destination_ref=task['destination_ref'],
                    refinement_task=True, place_refinement=True,
                    image_refs=inspection['image_refs'],
                    **{k: v for k, v in self.backend.pose_editor_feedback(selected).items() if k != 'image_refs'}),
                    scope, sid)
                self._event('candidate_refinement', sid, role='place',
                    trigger='requested' if requested else 'automatic', original_candidate_ref=selected,
                    selected_candidate_ref=self._approved_choice(review, refinement=True),
                    result=deepcopy(review.get('result', {})))
                if not self._approved_choice(review, refinement=True):
                    return dict(status='failed', result=review.get('result', {}),
                        planning_diagnostics=[*records, *review.get('planning_diagnostics', [])][-5:])
                if records:
                    review['planning_diagnostics'] = [*records, *review.get('planning_diagnostics', [])][-5:]
                return review
        return output

    def _intent_dispatch(self, role, tool, args, scope, task, sid):
        if tool == 'execute_grasp':
            self.backend.last_execution_pose = None
        if role == 'refiner' and task.get('place_refinement') and tool in (
                'inspect_place_candidate', 'explicit_adjust_place'):
            ref = args['candidate_ref']
            if scope['candidates'].get(ref) != self._epoch:
                raise BoundaryError('review a current delegated placement')
            method = 'explicit_inspect_place' if tool == 'inspect_place_candidate' else tool
            output = getattr(self.backend, method)(**args)
            new_ref = output.get('candidate_ref')
            if tool == 'inspect_place_candidate' and new_ref != ref:
                raise BoundaryError('placement inspection changed candidate identity')
            if output.get('accepted', output.get('executable', True)) and new_ref:
                scope['candidates'][new_ref] = self._epoch
                scope['inspected_candidates'][new_ref] = self._epoch
        else:
            output = super()._intent_dispatch(role, tool, args, scope, task, sid)
        if tool in ('inspect_candidate', 'preview_candidate', 'adjust_grasp',
                    'explicit_prepare_place', 'explicit_adjust_place', 'inspect_place_candidate'):
            details = self.backend.pose_editor_feedback(output.get('candidate_ref'))
            # Backend results exclusively own image attachment; restore only
            # metadata removed by the inherited public-result whitelist.
            details.pop('image_refs', None)
            output = {**output, **details}
            geometry = details.get('candidate_geometry')
            if geometry:
                if not hasattr(self, '_geometry_handoffs'):
                    self._geometry_handoffs = {}
                inherited = self._geometry_handoffs.get(geometry['candidate_ref'],
                    self._geometry_handoffs.get(args.get('candidate_ref'), {}))
                self._geometry_handoffs[geometry['candidate_ref']] = {
                    **deepcopy(inherited), 'candidate_geometry': deepcopy(geometry)}
        elif tool == 'nudge_grasp' and output.get('accepted'):
            details = self.backend.inflight_feedback
            output.update({k: deepcopy(v) for k, v in details.items() if k != 'image_refs'})
        elif tool == 'execute_grasp':
            details = getattr(self.backend, 'last_execution_pose', None)
            if isinstance(details, dict) and details.get('candidate_ref') == args['candidate_ref']:
                output['execution_pose'] = deepcopy(details)
        return output
