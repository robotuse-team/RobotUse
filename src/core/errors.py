"""Small public error contract; errors never establish native success.

Only bounded, sanitized exception text is exposed. Raw traceback/body/evidence
remain in private artifacts. Retry guidance describes a condition, not a retry loop.
"""
from __future__ import annotations

from copy import deepcopy
import re


_URL = re.compile(r'https?://[^\s\]\[<>"\']+', re.I)
_SECRET = re.compile(
    r'\b(authorization|api[-_ ]?key|access[-_ ]?token|refresh[-_ ]?token|token|password|passwd|secret)'
    r'[\s"\']*[:=][\s"\']*(?:Bearer\s+)?[^\s,;"\'}]+', re.I)


def safe_error_message(value, *, limit=400):
    """Bound a single public line without URLs, credentials or control characters."""
    message = str(value)
    message = _URL.sub('<service-url>', message)
    message = _SECRET.sub(lambda match: match.group(1) + '=<redacted>', message)
    message = re.sub(r'\bBearer\s+\S+', 'Bearer <redacted>', message, flags=re.I)
    message = re.sub(r'\b(?:sk|key)-[A-Za-z0-9_-]{12,}', '<redacted>', message)
    message = ' '.join(''.join(char if char.isprintable() else ' ' for char in message).split())
    return message[:limit] or 'No public exception message was supplied.'


def error_detail(category, code, phase, message, action, *, tool=None, exception_type=None):
    result = dict(schema_version=1, category=category, code=code, phase=phase,
                  message=safe_error_message(message), retry=dict(same_request=False, action=action))
    if tool is not None:
        result['tool'] = tool
    if exception_type is not None:
        result['exception_type'] = exception_type
    return result


def structured_error(exc, *, phase, tool=None):
    """Describe observed failure only, preserving service/contract/motion distinctions."""
    from src.core.action_feedback import ActionPreconditionError
    from src.tools.motion.planning import MotionPlanningError
    from src.backend.controller import BoundaryError
    from src.runtime.budget import MotionBudgetExceeded
    from src.tools.grasp.cgn_client import CGNResponseError, CGNServiceError

    name, message = type(exc).__name__, safe_error_message(exc)
    category, code = 'backend', 'backend_operation_failed'
    action = 'Report this operation and error to Prime; inspect current state before choosing recovery.'
    if isinstance(exc, CGNServiceError) or (phase.startswith('cgn') and isinstance(exc, (TimeoutError, ConnectionError))):
        category = 'service'
        code = getattr(exc, 'reason_code', 'cgn_service_timeout' if isinstance(exc, TimeoutError) else 'cgn_service_unavailable')
        action = ('Restore and pass CGN readiness before retrying this request. This service failure does not '
                  'establish geometric infeasibility; do not repeat calls without recovery evidence.')
    elif isinstance(exc, CGNResponseError):
        category, code = 'service', 'cgn_invalid_response'
        action = 'Report the CGN response contract error and repair the service/client contract before retrying.'
    elif isinstance(exc, ActionPreconditionError):
        category, code = 'state', exc.reason_code
        action = exc.public_feedback()['reason']
    elif isinstance(exc, MotionBudgetExceeded):
        category, code = 'budget', 'insufficient_simulation_time'
        action = 'Respect the remaining simulation budget; do not retry a motion that cannot fit.'
    elif isinstance(exc, MotionPlanningError):
        if 'execution' in phase:
            category, code = 'execution', 'motion_execution_failed'
            action = 'Inspect fresh robot/object state; a motion execution error does not prove an empty grasp or geometric impossibility.'
        else:
            category, code = 'planning', 'motion_plan_rejected'
            action = 'Use the candidate-specific planning feedback to change the pose or route, then validate again.'
    elif isinstance(exc, (BoundaryError, ValueError)) and any(text in str(exc).lower() for text in (
            'stale', 'current pointer-approved', 'current selected', 'current delegated',
            'current selection', 'outside current session', 'validation is stale')):
        category, code = 'state', 'stale_or_unowned_reference'
        action = 'Obtain the required current observation, selection or validation and use its returned reference.'
    elif isinstance(exc, BoundaryError):
        category, code = 'input', 'invalid_tool_contract'
        action = 'Correct the tool arguments or role/reference ownership using the reported message before retrying.'
    elif isinstance(exc, TypeError) and phase == 'tool_validation':
        category, code = 'input', 'invalid_argument_type'
        action = 'Correct the tool argument types using its schema before retrying.'
    elif isinstance(exc, (TypeError, AttributeError, NameError, KeyError)):
        category, code = 'contract', 'backend_contract_error'
        action = 'Report the operation and exception to Prime; changing object geometry is not an established repair.'
    elif isinstance(exc, ValueError):
        category, code = 'input', 'invalid_or_unavailable_input'
        action = 'Correct the reported input or unavailable reference; this is not evidence that the object is ungraspable.'
    elif isinstance(exc, TimeoutError):
        category, code = 'timeout', 'tool_timeout'
        action = 'Check whether the tool is still running and inspect current state before retrying.'
    result = error_detail(category, code, phase, message, action, tool=tool, exception_type=name)
    status = getattr(exc, 'http_status', None)
    if isinstance(status, int):
        result['http_status'] = status
    return result


def add_result_error(output, *, phase, tool=None):
    """Annotate nonexception tool rejections without inferring a physical cause."""
    if not isinstance(output, dict) or 'error_details' in output:
        return output
    category = code = action = message = None
    reason = output.get('reason_code')
    reason = reason if isinstance(reason, str) else ''
    if output.get('grasp_evidence') == 'empty_closed_gripper' and output.get('held_state') == 'not_held':
        category, code = 'physical', 'empty_grasp_observed'
        message = 'The completed grasp reported an empty closed gripper; physical retention was not established.'
        action = 'Inspect current RGB and command state; release the closed command before a fresh selection and regrasp.'
    elif output.get('accepted') is False or reason in ('adjustment_budget_exhausted', 'grasp_budget_exhausted', 'insufficient_simulation_time'):
        if reason in ('adjustment_budget_exhausted', 'grasp_budget_exhausted', 'insufficient_simulation_time'):
            category, code = 'budget', reason
            action = 'Use an operation within the remaining budget; repeating the same request does not restore it.'
        elif reason.endswith('_unavailable'):
            category, code = 'state', reason
            action = 'Obtain the missing reported geometry or state before retrying; do not infer object impossibility.'
        else:
            category, code = 'planning', reason or 'candidate_validation_rejected'
            action = 'Read the reported candidate-specific validation feedback, change the pose or route and validate again.'
        message = output.get('reason') or code
    elif output.get('status') == 'failed' and output.get('execution_ref'):
        category, code = 'execution', reason or 'execution_failed_unknown_cause'
        message = 'Execution did not complete successfully; the returned status alone does not identify a physical cause.'
        action = 'Inspect current RGB and robot state before recovery; do not assume the gripper is empty.'
    elif output.get('error'):
        category, code = 'backend', reason or 'tool_operation_failed'
        message = output.get('reason') or output.get('error')
        action = 'Report this tool failure to Prime and inspect current state before recovery.'
    if category is not None:
        output['error_details'] = error_detail(category, code, phase, message, action, tool=tool)
    return output


def public_error_details(value):
    """Keep the compact contract in handoffs, excluding arbitrary backend nesting."""
    if not isinstance(value, dict) or value.get('schema_version') != 1:
        return None
    result = {key: safe_error_message(value[key]) for key in
              ('category', 'code', 'phase', 'message', 'tool', 'exception_type') if isinstance(value.get(key), str)}
    result['schema_version'] = 1
    if isinstance(value.get('http_status'), int):
        result['http_status'] = value['http_status']
    retry = value.get('retry', {})
    if isinstance(retry, dict) and retry.get('same_request') is False and isinstance(retry.get('action'), str):
        result['retry'] = dict(same_request=False, action=safe_error_message(retry['action']))
    return deepcopy(result)
