"""Shared agent turn loop and role policy interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, ClassVar, Mapping
from uuid import uuid4

from src.tools.pose_editor.tool import REFINE_CANDIDATE

from src.core.contracts import (
    Action, BoundaryError, BudgetExceeded, EpisodeEnded, FirstGraspFinished,
)


class BaseAgent(ABC):
    """A role owns its turn loop; the orchestrator owns shared execution state."""

    role: ClassVar[str]

    @abstractmethod
    def base_prompt(self, context):
        """Render this role's policy from the execution settings."""
        raise NotImplementedError

    def run(self, orchestrator, task, parent_id=None):
        return run_agent_loop(orchestrator, self.role, task, parent_id)


def run_agent_loop(controller, role: str, task: dict[str, Any], parent_id: str | None = None) -> tuple[str, dict[str, Any]]:
    sid = uuid4().hex
    controller._check_episode(sid)
    controller._event('session_started', sid, role=role, parent_id=parent_id)
    try:
        session = controller.factory.new_session(role, sid)
    except Exception:
        controller._event('session_error', sid, role=role, reason='factory_error')
        controller._event('session_closed', sid, role=role, status='error')
        return 'error', {'error': 'factory_error'}
    if any(session is previous for previous in controller._sessions):
        controller._event('session_rejected', sid, reason='reused_session')
        controller._event('session_closed', sid, role=role, status='error')
        return 'error', {'error': 'reused_session'}
    controller._sessions.append(session)
    prompt = controller._prompt(role)
    if controller.waypoint_views and role == 'prime':
        prompt += (
            ' WAYPOINT OBSERVATION MODE: all observation motion uses delegate_waypoint(instruction), '
            'then move(waypoint_ref). move_to_view is unavailable. Point selects the desired EE position '
            'as corresponding front/wrist image points; orientation stays at the current EE orientation. '
            'Choose the view from current visual evidence; no automatic above-target move. '
            'Destination and pick object each allow at most two view requests. After every motion reselect '
            'using fresh Point evidence. After two moves, delegate Grasp to orbit/zoom the current candidate '
            'and make small pose refinements if nearly suitable, or finish unknown if evidence is inadequate.')
    if controller.waypoint_views and role == 'point' and task.get('waypoint_task'):
        prompt = (
            'You are Point selecting a viewing waypoint for Prime. Current calibrated front/wrist RGB is '
            'provided with the delegation or automatically each turn. If observe is available and no current '
            'pair was supplied, call it first. Select the SAME desired gripper/EE position in both images with '
            'waypoint(observation_id, front_u, front_v, wrist_u, wrist_v, reason), coordinates [0,1000]. '
            'This is an EE position, not a camera position; orientation is retained. The two rays triangulate '
            'free space without using background depth. Inspect returned requested/triangulated overlays, '
            'correct mismatched points or report {status:failed, reason:...}. On success finish exactly '
            '{waypoint_ref:...}. You do not move or execute. Never invent coordinates in metres or references.')
    if controller.decision_playbook is not None:
        prompt += controller.decision_playbook.render(role)
        controller._event('decision_playbook_loaded', sid, role=role,
                    sha256=controller.decision_playbook.sha256)
    messages: list[dict[str, Any]] = [
        {'role': 'system', 'content': prompt + controller._actor_context(role)},
        {'role': 'user', 'content': deepcopy(task)},
    ]
    scope: dict[str, Any] = {'observations': {}, 'points': {},
                             'candidates': {}, 'validations': {}, 'executions': {},
                             'execution_epochs': {}, 'latest_observation': None,
                             'targeted_reobserve': task.get('targeted_reobserve', False),
                             'waypoints': {}, 'inspected_candidates': {}}
    if task.get('observation'):
        observation = task['observation']
        scope['latest_observation'] = observation['observation_id']
        scope['observations'][observation['observation_id']] = (controller._epoch, {v['view_id'] for v in observation['views']})
    if controller.target_intent_mode and role == 'grasp':
        scope['candidates'].update({c['candidate_ref']: controller._epoch for c in task.get('candidate_bundle', {}).get('candidates', [])})
    if task.get('refinement_task'):
        if task.get('candidate_ref'):
            scope['candidates'][task['candidate_ref']] = controller._epoch
        if task.get('waypoint_ref'):
            scope['waypoints'][task['waypoint_ref']] = controller._epoch
    available_tools = controller._tools(role, task)
    limit = controller._turn_limit(role, task)
    status = 'budget_exhausted'
    result: dict[str, Any] = {'error': 'step_budget'}
    consecutive_rejections = 0
    try:
        for step in range(limit):
            controller._check_episode(sid)
            turn_messages = deepcopy(messages)
            turn_tools = available_tools
            request_context = controller._request_context(role, task, step, limit)
            if request_context:
                turn_messages.append({'role': 'user', 'content': deepcopy(request_context)})
            if controller.contact_manipulation and role == 'grasp' and scope.get('candidate_review_calls', 0) >= 3:
                turn_tools = (REFINE_CANDIDATE.name, 'finish')
                turn_messages.append({'role': 'user', 'content':
                    'Three static candidate inspections are complete. Repeating them provides no new geometry. '
                    'Refine a nearly suitable pose, finish with an inspected candidate and visual reason, or needs_point if none is suitable.'})
            if controller.target_intent_mode:
                try:
                    observation = controller._environment_observation(scope, sid)
                except Exception:
                    controller._event('observation_error', sid, role=role)
                    status, result = 'error', {'error': 'environment_observation_unavailable'}
                    break
                # A fresh request always contains current paired RGB. This is
                # environment input, not a model action or growing transcript.
                delivered = controller._agent_observation(role, task, observation)
                turn_messages.append({'role': 'user', 'content': {'current_observation': delivered}})
                registry = getattr(controller.factory, 'images', None)
                controller._event('agent_input', sid, role=role, step=step, instruction=task.get('instruction'),
                    observation=delivered, image_refs=registry.refs(turn_messages) if registry else [],
                    **deepcopy(request_context))
            try:
                action = session.next_action(turn_messages, turn_tools)
            except Exception:
                controller._event('model_error', sid, role=role)
                status, result = 'error', {'error': 'model_error'}
                break
            try:
                if not isinstance(action, Action) or action.tool not in turn_tools:
                    raise BoundaryError('tool not allowed')
                if not isinstance(action.arguments, Mapping):
                    raise BoundaryError('arguments must be a mapping')
                args = dict(action.arguments)
                if action.tool == 'finish':
                    result = controller._finish(role, args, scope)
                    if (controller.debug_reset_on_failed_grasp and not controller.active_perception and role == 'grasp'
                            and result.get('status') in ('failed', 'needs_point')):
                        controller._fresh_attempt(sid, result.get('reason', 'failed'), next(reversed(scope['executions']), None))
                    status = 'completed'
                    controller._event('finished', sid, role=role, result=result)
                    break
                required = set(controller._required_arguments(action.tool))
                optional = set(controller._optional_arguments(action.tool))
                if not required <= set(args) <= required | optional:
                    raise BoundaryError('invalid tool arguments')
                args = {key: controller._tool_argument(action.tool, key, value) for key, value in args.items()}
                if controller._calls >= controller.budgets.max_tool_calls:
                    raise BudgetExceeded()
                controller._calls += 1
                controller._event('tool_called', sid, role=role, tool=action.tool, arguments=args)
                output = controller._dispatch(role, action.tool, args, scope, task, sid)
                if action.tool == 'inspect_candidate' and 'error' not in output:
                    scope['candidate_review_calls'] = scope.get('candidate_review_calls', 0) + 1
                consecutive_rejections = 0
                controller._event('tool_result', sid, role=role, tool=action.tool, result=output)
                controller._check_episode(sid)
                messages.extend([{'role': 'assistant', 'tool': action.tool, 'arguments': args},
                                 {'role': 'tool', 'tool': action.tool, 'content': output}])
                if controller.debug_reset_on_failed_grasp and action.tool == 'execute_grasp' and 'execution_ref' in output:
                    if controller.target_intent_mode:
                        controller._environment_observation(scope, sid, refresh=True)
                        continue
                    # Driver guarantees observation even if the next model turn crashes/exhausts.
                    controller._calls += 1
                    controller._event('tool_called', sid, role=role, tool='observe', arguments={})
                    observed = controller._dispatch(role, 'observe', {}, scope, task, sid)
                    controller._event('tool_result', sid, role=role, tool='observe', result=observed)
                    messages.extend([{'role': 'assistant', 'tool': 'observe', 'arguments': {}},
                                     {'role': 'tool', 'tool': 'observe', 'content': observed}])
            except BudgetExceeded:
                status, result = 'budget_exhausted', {'error': 'tool_or_delegation_budget'}
                break
            except (BoundaryError, TypeError) as exc:
                rejection_details = {}
                if getattr(controller, 'structured_tool_errors', False):
                    from src.core.errors import structured_error
                    rejection_details['error_details'] = structured_error(exc, phase='tool_validation',
                        tool=action.tool if isinstance(action, Action) else None)
                controller._event('action_rejected', sid, role=role, reason='boundary_error', **rejection_details)
                if controller.active_perception:
                    # Preserve the rejected call so the model can correct its own arguments.
                    if isinstance(action, Action) and isinstance(action.arguments, Mapping):
                        messages.append({'role':'assistant','tool':action.tool,'arguments':dict(action.arguments)})
                    detail = str(exc) if isinstance(exc, BoundaryError) else 'invalid argument types'
                    rejection = {'error': 'boundary_error', 'detail': detail, **rejection_details}
                    messages.append({'role': 'tool', 'content': rejection})
                else:
                    messages.append({'role': 'tool', 'content': {'error': 'boundary_error'}})
                consecutive_rejections += 1
                if controller.target_intent_mode and consecutive_rejections >= 3:
                    status, result = 'error', {'error': 'repeated_invalid_action'}
                    break
        return status, result
    except FirstGraspFinished:
        status = 'first_grasp_finished'
        raise
    except EpisodeEnded:
        status = 'episode_ended'
        raise
    finally:
        try:
            session.close()
        except Exception:
            controller._event('close_error', sid, role=role)
        controller._event('session_closed', sid, role=role, status=status)
