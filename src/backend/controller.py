"""Two-layer, tool-driven Prime / Point / Grasp orchestration (no simulator imports).

Integration contract
--------------------
Backend owns images, geometry, simulator state, candidate poses and safety checks.
Only the public fields in PUBLIC_FIELDS cross the model/audit boundary. References
must be unique opaque, non-semantic tokens; image references must name sensor-only images,
not ground-truth overlays. Public strings must never contain BDDL, hidden geometry,
object-state annotations or private paths. Projection is structural protection, not
a content classifier: the trusted backend is responsible for these semantics.

Factory must create a genuinely new provider conversation for each new_session;
no shared provider thread, memory, history or sibling transcript is permitted.
The runner additionally rejects reused Python session objects and assigns unique
session IDs. The same child session iterates until finish or its budget expires.
No child receives Prime history or another child's messages. Prime chooses every
next tool; Point -> Grasp -> Point is just as valid as any other sequence.

Budgets bound calls/turns, not wall time. Adapters must enforce network/device
timeouts. execute_grasp must atomically recheck validation freshness and safety;
validation in this module is a capability gate, not a physical safety proof.
Legacy Grasp must observe after execution before finishing with an execution reference.
In target-intent pickup mode, every role gets automatic current paired RGB and
Prime owns execution; Point and Grasp receive only bounded delegated context.
Execution status denotes the backend command outcome, never physical grasp or task
success; the final result remains verified=False even after visual inspection.
Model finish reports are not independent task-success verification.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import math
from typing import Any, Callable, Mapping, Protocol, Sequence
from uuid import uuid4


from src.core.contracts import (
    Backend,
    Action,
    AgentSession,
    SessionFactory,
    Budgets,
    RunResult,
    BoundaryError,
    BudgetExceeded,
    EpisodeEnded,
    FreshAttemptRequired,
    FirstGraspFinished,
    AuditError,
)


PUBLIC_FIELDS = {
    'observe': ('observation_id', 'views'),
    'point': ('point_ref', 'observation_id', 'image_refs'),
    'select_target': ('point_ref', 'observation_id', 'image_refs', 'source_views', 'provenance', 'reason_code'),
    'select_view_target': ('point_ref', 'observation_id', 'image_refs', 'source_views', 'provenance', 'reason_code'),
    'move_to_view': ('observation_id', 'views', 'view_status', 'reason_code'),
    'observe_target': ('observation_id', 'views'),
    'waypoint': ('waypoint_ref', 'observation_id', 'image_refs'),
    'contact_waypoint': ('waypoint_ref', 'observation_id', 'image_refs'),
    'contact_destination': ('waypoint_ref', 'observation_id', 'image_refs'),
    'push_waypoint': ('waypoint_ref', 'observation_id', 'image_refs'),
    'move': ('observation_id', 'views', 'view_status'),
    'turn': ('observation_id', 'views', 'execution_ref', 'status'),
    'inspect_candidate': ('candidate_ref', 'image_refs'),
    'preview_candidate': ('candidate_ref', 'image_refs'),
    'refine_candidate': ('candidate_ref', 'image_refs', 'adjustment_deg', 'cumulative_deg'),
    'inspect_point_cloud': ('point_ref', 'observation_id', 'image_refs', 'source_views', 'provenance'),
    'fuse_points': ('point_ref', 'observation_id', 'image_refs', 'source_views', 'provenance'),
    'grasp_candidates': ('candidates', 'image_refs'),
    'validate_grasp': ('candidate_ref', 'validation_ref', 'accepted', 'reason_code', 'image_refs'),
    'execute_grasp': ('execution_ref', 'status', 'reason_code', 'image_refs'),
    'place': ('execution_ref', 'status'),
    'release': ('execution_ref', 'status'),
    'close_for_push': ('execution_ref', 'status'),
    'save_destination': ('destination_ref', 'observation_id', 'image_refs', 'memory_policy'),
    'place_candidates': ('candidates', 'image_refs', 'candidate_policy'),
    'inspect_place_candidate': ('candidate_ref', 'image_refs'),
    'refine_place_candidate': ('candidate_ref', 'image_refs', 'adjustment_deg', 'cumulative_deg'),
    'validate_place': ('candidate_ref', 'validation_ref', 'accepted', 'reason_code'),
    'execute_place_candidate': ('execution_ref', 'status', 'reason_code'),
}
TOOLS = {
    'prime': ('observe', 'delegate_point', 'delegate_grasp', 'place', 'finish'),
    'point': ('observe', 'point', 'inspect_point_cloud', 'fuse_points', 'observe_target', 'finish'),
    'grasp': ('observe', 'grasp_candidates', 'validate_grasp', 'execute_grasp', 'finish'),
}
ARGUMENTS = {
    'observe': (), 'point': ('observation_id', 'view_id', 'u', 'v'),
    'select_target': ('observation_id', 'front_u', 'front_v', 'wrist_u', 'wrist_v'),
    'select_view_target': ('observation_id', 'view_id', 'u', 'v'),
    'move_to_view': ('point_ref',),
    'inspect_point_cloud': ('point_ref',), 'fuse_points': ('point_refs', 'same_object'),
    'observe_target': ('point_ref',),
    'waypoint': ('observation_id', 'front_u', 'front_v', 'wrist_u', 'wrist_v', 'reason'),
    'contact_waypoint': ('observation_id', 'view_id', 'u', 'v', 'reason'),
    'contact_destination': ('observation_id', 'view_id', 'source_u', 'source_v', 'u', 'v', 'reason'),
    'push_waypoint': ('observation_id', 'view_id', 'u', 'v', 'stage', 'reason'),
    'delegate_waypoint': ('instruction',), 'move': ('waypoint_ref',),
    'turn': ('angle_deg',),
    'inspect_candidate': ('candidate_ref',),
    'preview_candidate': ('candidate_ref', 'azimuth_deg', 'elevation_deg', 'zoom'),
    'refine_candidate': ('candidate_ref', 'roll_deg', 'pitch_deg', 'yaw_deg'),
    'grasp_candidates': ('point_ref',), 'validate_grasp': ('candidate_ref',),
    'execute_grasp': ('candidate_ref', 'validation_ref'), 'place': ('point_ref',),
    'delegate_point': ('instruction',), 'delegate_grasp': ('instruction', 'point_ref'),
    'release': (),
    'close_for_push': (),
    'delegate_destination': ('instruction',),
    'point_destination': ('observation_id', 'view_id', 'u', 'v'),
    'select_destination': ('observation_id', 'front_u', 'front_v', 'wrist_u', 'wrist_v'),
    'save_destination': ('point_ref',),
    'delegate_place': ('instruction', 'destination_ref', 'hold_assessment', 'destination_assessment'),
    'place_candidates': ('destination_ref', 'hold_assessment', 'destination_assessment'),
    'inspect_place_candidate': ('candidate_ref',), 'validate_place': ('candidate_ref',),
    'refine_place_candidate': ('candidate_ref', 'roll_deg', 'pitch_deg', 'yaw_deg'),
    'execute_place_candidate': ('candidate_ref', 'validation_ref'),
}


REFINE_GUIDANCE = (
    'Use preview_candidate to orbit the saved measured 3D scene and zoom, like camera sliders: '
    'azimuth_deg [-180,180], elevation_deg [-85,85], zoom [0.5,3]. It moves only the virtual view. '
    'Inspect finger enclosure from different angles; this is an X-ray proposal, not predicted physical contact. '
    'If a candidate is nearly right, refine_candidate rotates it by at most 10 degrees per axis about the point '
    'between its jaws, replans it, and returns a NEW candidate_ref with fresh overlays. Use it for small corrections '
    'only; a pose that is clearly wrong should be replaced by another candidate, not nudged. After refining you have '
    'the same choices as before: preview/refine again within the 30 degree budget, or finish on whichever reference you judged best. '
    'After two observation moves, use current geometry and preview/refine a nearly suitable candidate instead of another move. '
    'If geometry is inadequate or no acceptable candidate remains, report needs_point/failed with the reason; never force a grasp. '
)


def _text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 8192:
        raise BoundaryError('expected bounded nonempty string')
    return value


def _images(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)) or len(value) > 64:
        raise BoundaryError('invalid image references')
    return [_text(item) for item in value]


def public_result(tool: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    """Copy allowlisted typed fields only; never serialize arbitrary backend data."""
    if not isinstance(raw, Mapping):
        raise BoundaryError('expected backend mapping')
    out: dict[str, Any] = {}
    if 'error_details' in raw:
        from src.core.errors import public_error_details
        details = public_error_details(raw['error_details'])
        if details is not None:
            out['error_details'] = details
    if raw.get('terminal') is True:
        out['terminal'] = True
    if raw.get('reason_code') in ('simulation_time_limit', 'episode_terminated', 'insufficient_simulation_time'):
        out['reason_code'] = raw['reason_code']
    required = raw.get('required_simulation_s')
    if type(required) in (int, float) and math.isfinite(required) and required >= 0:
        out['required_simulation_s'] = required
    for key in PUBLIC_FIELDS[tool]:
        if key not in raw and (key == 'reason_code' or (tool in ('validate_grasp', 'execute_grasp') and key == 'image_refs')
                              or (tool in ('select_target', 'select_view_target') and key in ('point_ref', 'source_views', 'provenance'))):
            continue
        if tool == 'move_to_view' and raw.get('reason_code') == 'observation_refresh_failed':
            if key == 'observation_id' and raw.get(key) is None:
                out[key] = None
                continue
            if key == 'views' and raw.get(key) == []:
                out[key] = []
                continue
        if key == 'image_refs' and tool in ('point', 'grasp_candidates') and key not in raw:
            continue
        if key in ('view_change', 'view_status') and key not in raw:
            continue
        if key not in raw:
            raise BoundaryError('missing public field')
        value = raw[key]
        if key == 'views':
            if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 64:
                raise BoundaryError('invalid views')
            out[key] = []
            seen_views = set()
            for view in value:
                if not isinstance(view, Mapping):
                    raise BoundaryError('invalid view')
                view_id = _text(view.get('view_id'))
                if view_id in seen_views:
                    raise BoundaryError('duplicate view')
                seen_views.add(view_id)
                out[key].append({'view_id': view_id, 'image_ref': _text(view.get('image_ref'))})
        elif key in ('image_refs', 'source_views'):
            out[key] = _images(value)
        elif key == 'candidates':
            if not isinstance(value, (list, tuple)) or len(value) > 128:
                raise BoundaryError('invalid candidates')
            out[key] = []
            seen: set[str] = set()
            for candidate in value:
                if not isinstance(candidate, Mapping):
                    raise BoundaryError('invalid candidate')
                ref = _text(candidate.get('candidate_ref'))
                if ref in seen:
                    raise BoundaryError('duplicate candidate reference')
                seen.add(ref)
                item = {'candidate_ref': ref}
                if 'image_refs' in candidate:
                    item['image_refs'] = _images(candidate['image_refs'])
                if tool == 'place_candidates' and 'path_description' in candidate:
                    item['path_description'] = _text(candidate['path_description'])
                if 'approach' in candidate:
                    meta = candidate['approach']
                    fields = ('source_score', 'original_rank', 'published_rank', 'downward_alignment',
                              'downward_angle_deg', 'preference_score', 'downward_weight')
                    if not isinstance(meta, Mapping) or any(k not in meta for k in fields):
                        raise BoundaryError('invalid approach metadata')
                    clean = {}
                    for name in fields:
                        val = meta[name]
                        if type(val) not in (int, float) or not math.isfinite(val):
                            raise BoundaryError('invalid approach number')
                        clean[name] = val
                    direction = meta.get('direction_base')
                    if (not isinstance(direction, (list, tuple)) or len(direction) != 3
                            or any(type(v) not in (int, float) or not math.isfinite(v) for v in direction)
                            or not math.isclose(sum(v*v for v in direction), 1.0, abs_tol=1e-5)):
                        raise BoundaryError('invalid approach direction')
                    clean['direction_base'] = list(direction)
                    item['approach'] = clean
                out[key].append(item)
        elif key in ('adjustment_deg', 'cumulative_deg'):
            if (not isinstance(value, Mapping) or set(value) != {'roll_deg', 'pitch_deg', 'yaw_deg'}
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in value.values())):
                raise BoundaryError('adjustment must carry finite roll, pitch and yaw degrees')
            out[key] = {name: float(value[name]) for name in ('roll_deg', 'pitch_deg', 'yaw_deg')}
        elif key == 'accepted':
            if type(value) is not bool:
                raise BoundaryError('accepted must be boolean')
            out[key] = value
        elif key == 'status':
            if value not in ('succeeded', 'failed', 'unknown'):
                raise BoundaryError('invalid execution status')
            out[key] = value
        else:
            out[key] = _text(value)
    if tool in ('select_target', 'select_view_target') and not out.get('point_ref') and not out.get('reason_code'):
        raise BoundaryError('selection requires a point reference or failure reason')
    return out


class PrimeOrchestrator:
    """Synchronous, single-use runner with injectable model and backend adapters.

    Audit sink receives detached JSON-safe events. Sink failure stops execution
    (fail closed). Events omit raw backend values and exception messages.
    """

    def __init__(self, backend: Backend, factory: SessionFactory, *,
                 budgets: Budgets | None = None,
                 audit_sink: Callable[[Mapping[str, Any]], None] | None = None,
                 debug_reset_on_failed_grasp: bool = False,
                 actor_context: str = "", active_perception: bool = False,
                 first_grasp_only: bool = False, contact_manipulation: bool = False,
                 waypoint_views: bool = False, decision_playbook=None):
        if contact_manipulation and (not active_perception or first_grasp_only):
            raise ValueError('contact manipulation requires active perception and full-task evaluation')
        self.contact_manipulation = contact_manipulation
        self.waypoint_views = waypoint_views
        self.active_perception = active_perception
        self.first_grasp_only = first_grasp_only
        self.target_intent_mode = active_perception and first_grasp_only
        self.debug_reset_on_failed_grasp = debug_reset_on_failed_grasp or first_grasp_only
        self.actor_context = actor_context
        self.decision_playbook = decision_playbook
        self.backend = backend
        self.factory = factory
        self.budgets = budgets or Budgets()
        self.audit_sink = audit_sink
        self._events: list[dict[str, Any]] = []
        self._sessions: list[AgentSession] = []  # retain identities against id reuse
        self._point_refs: dict[str, int] = {}
        self._view_only_refs: dict[str, int] = {}
        self._waypoint_refs: dict[str, int] = {}
        self._epoch = 0
        self._calls = 0
        self._delegations = 0
        self._used = False
        self._debug_execution_ref = None
        self._current_observation = None
        self._target_evidence = {}
        self._candidate_bundles = {}
        self._rejected_candidates = {}
        self._reviews_per_target = {}
        self._view_requests = {}
        self._current_view_target = 'pick_object'

    def _event(self, kind: str, session_id: str, **fields: Any) -> None:
        event = deepcopy(dict(seq=len(self._events), kind=kind, session_id=session_id, **fields))
        json.dumps(event, allow_nan=False)
        if self.audit_sink is not None:
            try:
                self.audit_sink(deepcopy(event))
            except Exception:
                raise AuditError('audit unavailable') from None
        self._events.append(event)

    def run(self, instruction: str) -> RunResult:
        if self._used:
            raise RuntimeError('runner is single-use; create a fresh runner')
        self._used = True
        instruction = _text(instruction)
        try:
            status, result = self._loop('prime', {'instruction': instruction})
        except FreshAttemptRequired as exc:
            status, result = 'fresh_attempt_required', exc.result
        except FirstGraspFinished as exc:
            status, result = 'first_grasp_finished', exc.result
        except EpisodeEnded as exc:
            status, result = 'episode_ended', exc.result
        return RunResult(status, deepcopy(result), tuple(deepcopy(self._events)),
                         self._calls, self._delegations)

    def _turn_limit(self, role, task):
        return self.budgets.prime_steps if role == 'prime' else self.budgets.child_steps

    def _actor_context(self, role):
        return self.actor_context

    def _request_context(self, role, task, step, limit):
        """Fresh request-only metadata; legacy roles have no extra context."""
        return {}

    def _check_episode(self, sid):
        read = getattr(self.backend, 'simulation_budget', None)
        budget = read() if callable(read) else None
        if budget is not None and budget['terminal']:
            self._event('episode_ended', sid, reason_code=budget['reason_code'], simulation_budget=budget)
            raise EpisodeEnded(budget)

    def _loop(self, role: str, task: dict[str, Any], parent_id: str | None = None) -> tuple[str, dict[str, Any]]:
        from src.agent import run_role
        return run_role(self, role, task, parent_id)

    def _required_arguments(self, tool):
        return ARGUMENTS[tool]

    def _optional_arguments(self, tool):
        return ()

    def _tool_argument(self, tool, key, value):
        return self._argument(key, value)

    @staticmethod
    def _argument(key: str, value: Any) -> Any:
        if key == 'grasp_type':
            raise BoundaryError('grasp_type is no longer a tool argument; use preferred_direction '
                                'with vertical, horizontal or null')
        if key == 'preferred_direction':
            from src.tools.grasp.preference import normalize_approach
            try:
                return normalize_approach(value)
            except ValueError as exc:
                raise BoundaryError(str(exc)) from None
        if key == 'angle_deg':
            from src.tools.motion.contact import turn_angle
            try:
                return turn_angle(value)
            except ValueError as exc:
                raise BoundaryError(str(exc)) from None
        from src.tools.pose_editor.preview_controls import VIEW_LIMITS
        if key in VIEW_LIMITS:
            lo, hi = VIEW_LIMITS[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
                raise BoundaryError(f'{key} must be a finite number in [{lo}, {hi}]')
            return float(value)
        if key in ('dx_mm', 'dy_mm', 'dz_mm'):
            if type(value) not in (int, float) or not math.isfinite(value) or abs(value)>10:
                raise BoundaryError('translation must be finite and within 10 mm')
            return float(value)
        if key in ('u', 'v', 'front_u', 'front_v', 'wrist_u', 'wrist_v', 'source_u', 'source_v'):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1000:
                raise BoundaryError('coordinates must be finite numbers in [0,1000]')
            return value
        if key in ('roll_deg', 'pitch_deg', 'yaw_deg'):
            from src.tools.pose_editor.refinement import STEP_LIMIT_DEG
            if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > STEP_LIMIT_DEG:
                raise BoundaryError(f'a fine adjustment stays within {STEP_LIMIT_DEG:g} degrees per axis')
            return float(value)
        if key == 'point_refs':
            if not isinstance(value, (list, tuple)) or len(value) != 2 or len(set(map(str, value))) != 2:
                raise BoundaryError('fusion requires two distinct explicit same-object point references')
            return [_text(ref) for ref in value]
        return _text(value)

    def _tools(self, role, task):
        if self.target_intent_mode:
            if self.waypoint_views and role == 'point' and task.get('waypoint_task'):
                return ('waypoint', 'finish')
            if self.waypoint_views and role == 'prime':
                return ('delegate_point', 'delegate_grasp', 'delegate_waypoint', 'move', 'validate_grasp', 'execute_grasp', 'finish')
            return {'prime': ('delegate_point', 'delegate_grasp', 'move_to_view', 'validate_grasp', 'execute_grasp', 'finish'),
                    'point': ('select_target', 'select_view_target', 'finish'),
                    'grasp': ('inspect_candidate', 'preview_candidate', 'refine_candidate', 'finish')}[role]
        tools = TOOLS[role]
        if self.first_grasp_only and role == 'prime':
            tools = tuple(t for t in tools if t != 'place')
        if self.contact_manipulation and role == 'prime':
            tools = tuple(t for t in tools if t != 'place') + ('release', 'close_for_push', 'turn', 'move_to_view')
        if self.active_perception:
            if role == 'prime':
                tools = (*tools, 'delegate_waypoint', 'move', 'validate_grasp', 'execute_grasp')
            elif role == 'grasp':
                tools = ('grasp_candidates', 'inspect_candidate', 'preview_candidate', 'refine_candidate', 'finish')
            elif task.get('waypoint_task'):
                tools = ('observe', 'waypoint', 'finish')
                if self.contact_manipulation and task.get('contact_plane_available'):
                    tools = (*tools, 'contact_waypoint', 'contact_destination', 'push_waypoint')
            elif task.get('point_mode') == 'placement':
                tools = ('observe', 'point', 'inspect_point_cloud', 'finish')
            else:
                tools = tuple(t for t in tools if t != 'observe_target')
                if self.contact_manipulation:
                    tools = (*tools, 'select_view_target')
        return tools

    def _prompt(self, role: str) -> str:
        if self.target_intent_mode:
            common = ('Visual intention interface: identify the requested object through image points and judge gripper overlays. '
                      'Only supplied front/wrist RGB and measured RGBD are available. SAM3D is not used. '
                      'The environment automatically supplies the current front/wrist observation on EVERY turn; no observe call is needed. '
                      'No code, metric coordinates, joints, hidden geometry or invented references. '
                      'A tool command is not proof of holding the object. ')
            return common + {
                'prime': ('You are Prime, the planner. Goal: pick the specified object, lift it and retain it; no placement. '
                    'delegate_point identifies the same target in both views; the backend segments and fuses automatically. '
                    'delegate_grasp gives an independent visual judge the target and current candidate overlays. '
                    'Validate a returned candidate before execution. On rejection another review reuses remaining candidates; '
                    'never revalidate a rejected candidate. If poses or visibility are unsuitable, move_to_view(point_ref) '
                    'requests the existing planner to move above that selected target with a downward viewing orientation. '
                    'This is a viewing operation, not an arbitrary spatial waypoint. Do not move if current evidence is adequate. '
                    'After actual motion request a fresh target selection. A planning rejection leaves the previous view valid. '
                    'At most two view requests and three candidate reviews per target are available. '
                    'If only one view shows the target, Point can return a view_only reference from select_view_target. '
                    'Use move_to_view with that reference, then request fresh paired target selection before grasping. '
                    'A view_only reference cannot be delegated to Grasp. If neither view shows the target, finish unknown. '
                    'After a physical grasp the driver supplies fresh RGB; finish exactly '
                    '{status: completed|failed|unknown, execution_ref, actor_visual_assessment: held|empty|uncertain}. '
                    'Judge the actual hold, not controller status. Before any physical grasp, finish exactly {status: failed|unknown} '
                    'when evidence or feasible actions are exhausted. Only one physical grasp attempt is allowed.'),
                'point': ('You are Point, the visual target selector. Use the automatically supplied current observation. '
                    'Call select_target(observation_id, front_u, front_v, wrist_u, wrist_v), normalized to [0,1000]. '
                    'The two clicks identify the SAME OBJECT; they may land on DIFFERENT visible surfaces. '
                    'If only one camera shows the target, call select_view_target(observation_id, view_id, u, v) '
                    'on that visible target instead. Its measured RGBD point_ref is for viewing motion only; '
                    'finish with {point_ref} so Prime can move and request fresh selection. '
                    'You do not solve stereo correspondence or free-space depth. Inspect the returned mask overlays and cloud; '
                    'correct clicks if the selected object is wrong. Never click an invisible/occluded target by guessing. '
                    'Finish exactly {point_ref} on success, or {status: failed, reason: concrete visibility or selection problem}. '
                    'The planner alone decides movement and grasping.'),
                'grasp': ('You are Grasp, a visual pose judge. The target description, current front/wrist RGB, and generated '
                    'candidate gripper overlays are supplied. The backend has already generated the candidate pool. '
                    'Review cards project the proposed gripper onto CURRENT calibrated RGB crops; the wrist image is not '
                    'the future view after moving to that pose. Cyan/magenta mark fingers, orange marks observed target '
                    'surfaces, and arrows mark approach/closing directions. Overlays are X-ray proposals, not verified contacts. '
                    'Compare candidates and inspect_candidate for promising ones. Assess enclosure, centering, contact depth, '
                    'palm/finger interference and orientation in the nearby scene. A static overlay cannot establish full-arm '
                    'collision freedom, joint configuration or reachability; the planner validates those it can check. '
                    'Do not enforce a blanket top-down rule. Excluded candidates already failed planning and cannot be chosen. '
                    + REFINE_GUIDANCE +
                    'Finish exactly {candidate_ref: an inspected candidate, reason: visual basis} or '
                    '{status: needs_point, reason: why all remaining poses seem unsuitable or which visibility is missing}. '
                    'You do not generate poses, move, execute or delegate.')
            }[role]
        common = ('Use only supplied sensor evidence and opaque references. Never infer hidden '
                  'geometry or BDDL. Choose one tool per turn; tools may be called repeatedly. '
                  'Finish does not verify task success. Children may finish with {status: failed, '
                  'reason: no_candidates|wrong_target|unreachable|needs_reobserve|no_valid_mask|empty_grip}; '
                  'Grasp may use status needs_point with those same reasons. ')
        if self.active_perception:
            common = ('Point-based robot interface: no code, metric coordinates, joints or trajectories. '
                       'Current synchronized front/wrist measurements only; SAM3D is not used. ')
            if role == 'prime':
                contact_policy = ('This is a push/pull/turn contact task. Select the movable handle or graspable object rim using the existing '
                    'Point and Grasp agents. execute_grasp approaches and closes WITHOUT lifting. Observe the actual contact, then '
                    'delegate_waypoint to click the OBJECT DESTINATION using contact_destination: mark a visible point on the object '
                    'and where that SAME point should move in the fresh front image. Choose the final object destination that completes the task and request one continuous move to it. '
                    'Call move with that waypoint_ref: it keeps the gripper closed and preserves its current orientation. '
                    'contact_destination translates the hand by the clicked object displacement, retaining its offset, height and orientation. '
                    'It does not move the hand onto the object centre. The source supplies measured height; '
                    'the destination may be free space at that height and need not appear in the wrist view. '
                    'For a rotary knob, choose a grasp aligned with its visible rotation axis. After observing the held contact, '
                    'turn(angle_deg) rotates about the current tool local Z through the TCP, keeping TCP position fixed. '
                    'Each call accepts an angle from -180 to 180 degrees, excluding zero. Choose the angle needed from visual evidence. Positive follows the right-hand rule about tool +Z, '
                    'which points from the wrist toward the fingers; its direction in an image depends on the view. '
                    'This tool does not locate a knob pivot or align an off-axis grasp. Use it only when the tool and knob axes are aligned. '
                    'Inspect the returned fresh RGB after every increment; continue only if the knob actually moved toward the goal and contact remains. '
                    'If the object slips, fails to turn, or alignment is uncertain, release and regrasp or finish failed/unknown. '
                    'A succeeded turn reports robot tracking only, never that the stove is on. Do not use turn with an empty pusher or transport grasp. '
                    'For a push-only task where a thin object on its support has no feasible grasp, close_for_push closes '
                    'the fingers without claiming to hold an object. Observe, then use waypoints to bring the closed fingertips '
                    'to an exposed edge of the object and slide it along the support toward the goal. '
                    'For the initial push, delegate a push_waypoint on visible TABLE just outside an exposed edge, stage approach, '
                    'then a fresh push_waypoint at the same support location, stage contact. This accounts for fingertip length. '
                    'A push needs contact on the side OPPOSITE the desired object motion. For diagonal right-and-back travel, '
                    'stage outside the exposed FRONT-LEFT arc and push through the object toward the goal. '
                    'A pusher on the front-right arc moving farther right just leaves or skirts the object; it cannot push it right. '
                    'Use the open foreground arc, avoiding the narrow gap directly beside furniture. '
                    'To restage an empty pusher, use push_waypoint stage approach on visible nearby TABLE to raise before repositioning. '
                    'A low horizontal foreground retreat can exceed reach; an upward free-space point may be behind the wrist camera. '
                    'Use this existing measured-support approach instead of repeatedly attempting either unavailable retreat. '
                    'This fallback is for pushing, never for pulling a handle. '
                    'Use both views for palm clearance: overlap in one RGB projection alone is not proof of physical collision. '
                    'If an edge approach is blocked, choose another exposed edge '
                    'and push diagonally. contact_waypoint can first shift the closed hand horizontally at its CURRENT height '
                    'away from the obstruction; use the ordinary paired waypoint to descend from a clear position. '
                    'Keep the contact height and follow the visible drawer slide or tabletop direction. Prefer one full stroke to the final destination '
                    'and inspect the returned fresh RGB afterward. Use an intermediate destination only when the final target exceeds the 30 cm per-move limit, '
                    'the route is obstructed or rejected by planning, or contact is lost and must be restored. State the specific reason for splitting. '
                    'For a drawer, continue pulling until it is visibly well extended and its interior is exposed. '
                    'For a drawer, target the handle or moving front position when the drawer is well open, within its visible slide direction; do not request tiny increments by default. '
                    'Use contact_destination for moving the object; ordinary waypoint/contact_waypoint are direct EE targets for repositioning. '
                    'Do not use overhead observation moves while holding a handle. '
                    'Release opens the gripper in place. Complete this contact goal, release, and assess fresh RGB before finishing. '
                    'If fresh RGB shows the object was not retained, call release before requesting another grasp, '
                    'even when execute_grasp returned succeeded: its holding evidence is provisional. '
                    'Finish only after visually assessing the entire requested final state; '
                    'the independent benchmark verifier determines success. ') if self.contact_manipulation else ''
                return common + contact_policy + ('You are the planner. Own all decisions about what to do next. '
                    'delegate_point asks an independent Point agent to select the target in both current views and fuse its measured cloud. '
                    'delegate_grasp gives that point_ref and target description to an independent visual grasp judge. '
                    'The judge returns a candidate_ref with reason or status needs_point with reason; it never moves the robot. '
                    'You may validate_grasp then execute_grasp on a returned candidate. If IK rejects it, delegate another candidate review. '
                    + ('If visibility or grasp candidates are unsuitable, ask delegate_point to select the object you want to see, '
                    'then call move_to_view(point_ref). It adjusts height first, then moves above that object facing down; '
                    'the selected point is the viewing subject, not the EE destination. If only one camera sees the object, '
                    'Point can return a view_only reference using select_view_target; use it only for move_to_view, never Grasp. '
                    'At most two view requests are available. View motion requires an empty, open gripper; release contact first. '
                    'Use move(waypoint_ref) for a specified EE destination or contact stroke; it preserves orientation and adds no viewing height. '
                    'For better visibility use move_to_view, not a waypoint on the object. '
                    if self.contact_manipulation else
                    'If candidates look unsuitable, decide what observation would help, delegate_waypoint with a visual instruction '
                    'for the next gripper position, then move using its waypoint_ref. Point specifies one spatial waypoint as two corresponding '
                    'image points; the calibrated planner handles execution. ') +
                    'After ANY movement request fresh Point perception before new grasp proposals. '
                    'Do not force reobservation or top-down grasps when current evidence is adequate. '
                    'After execute_grasp observe actual RGB, never confuse command completion with holding the object. '
                    'In first-grasp-only mode finish with status, execution_ref and actor_visual_assessment held|empty|uncertain; do not place. '
                    + ('' if self.contact_manipulation else 'Otherwise, if visibly held, delegate_point to select the destination support surface, then place. ')
                    +
                    'Finish with status completed|failed|unknown. Fail explicitly if no feasible meaningful action remains. '
                    'Do not repeat a physical grasp in the same first-grasp attempt.')
            if role == 'point':
                return common + ('You are Point. For an object task, select the SAME requested object separately in agentview and '
                    'robot0_eye_in_hand using point(observation_id, view_id, u,v), inspect mask overlays and correct if needed. '
                    'Inspect clouds and fuse_points with an explicit same_object description; finish with that fused point_ref. '
                    + ('If only one camera shows the target, use select_view_target(observation_id, view_id, u, v) on the visible '
                    'object and finish with its point_ref. This view_only reference lets Prime move_to_view above the object '
                    'and request fresh selection; it cannot supply a grasp. Never guess a click on an occluded target. '
                    if self.contact_manipulation else '') +
                    'The task includes point_mode. If placement, select the destination SUPPORT surface from fresh front RGB and finish its point_ref; do not fuse. '
                    'For a waypoint_task, observe both views, then use waypoint with front_u,front_v,wrist_u,wrist_v in [0,1000] and reason. '
                    'Both points must denote the SAME intended 3D location, potentially in free space, NOT two different visible surfaces. '
                    'The waypoint is the desired gripper/EE position, not the camera position. Review returned reprojection overlays, '
                    'correct the pair if misplaced, then finish with waypoint_ref. Camera/EE orientation is retained during move. '
                    'Never infer free-space depth from a background surface; triangulation uses the two calibrated image rays. '
                    'If correspondence is impossible or rejected, retry the pixels or finish status failed with reason. '
                    + ('For a push/pull stroke, use contact_destination(observation_id, view_id, source_u, source_v, u, v, reason). '
                       'Click a visible point ON the movable object or handle as source, then click where that SAME point should end up when the task is complete. '
                       'Do not shorten the requested final displacement to a fraction of the object width by default. Use an intermediate point only for a stated route, contact or 30 cm limit constraint. '
                       'Both clicks use one current image and normalized [0,1000] pixels. Do not click the gripper or fixed cabinet as source. '
                       'Review the green object-source to yellow object-destination arrow and the cyan hand target; '
                       'the hand retains its offset from the object. Correct misplaced clicks before finishing waypoint_ref. '
                       'The destination is on the horizontal plane of the measured source, not the depth of the background behind it. '
                       'If the source is occluded, use another visible point on the SAME rigid object, or request a better view. '
                       'For an empty pusher first reach stage contact, then select the object destination. '
                       'Only for direct hand repositioning, use contact_waypoint(observation_id, view_id, u, v, reason). '
                       'Click the desired NEXT EE position in the visible front image, on the horizontal plane through '
                       'the current grasp. Height and orientation are retained; the camera ray determines XY. '
                       'You do not need a wrist correspondence. Select the repositioning endpoint requested by the planner. '
                       'For a closed-finger push approach, use push_waypoint(observation_id, view_id, u, v, stage, reason): '
                       'click visible bare TABLE just outside the chosen exposed object edge in front RGB; stage approach '
                       'places the EE above that support, stage contact puts the fingertips just above it. '
                       'Do not click the plate top or furniture. Choose the exposed arc OPPOSITE the desired push direction: '
                       'for rightward-and-back motion this is the front-left arc, not the front-right arc. '
                       'The fingertips must start outside that rim, then move into the object toward the goal, not circle around it. '
                       'Review the projected EE target; orange marks the CURRENT EE and the arrow shows the actual requested direction. '
                       'Correct a waypoint whose arrow moves opposite the requested slide or push BEFORE finishing. '
                       'For an empty-pusher retreat/restage, use stage approach on nearby visible TABLE to raise before lateral repositioning. '
                       'After contact use contact_destination to point where the OBJECT should move, keeping the established support height. '
                       if self.contact_manipulation else '') +
                    'You only point; the planner decides when to move or grasp.')
            return common + ('You are Grasp, a visual candidate judge. Target description, current front/wrist RGB, measured cloud '
                'and actual-size gripper pose overlays are supplied. Call grasp_candidates, compare them, and inspect_candidate '
                'for candidates of interest. Choose based on target enclosure, centering, contact depth, orientation and nearby scene. '
                'A static gripper overlay cannot certify whole-arm collision or reachability. Do not apply a fixed downward rule. '
                + ('For contact grasps, compare a promising alternative before choosing when several candidates are available. '
                   'Prefer jaws around the exposed handle/rim with the palm outside the cabinet, using the measured cloud and both views. '
                   'A high grasp score does not justify palm penetration into furniture. Single-view projected overlap alone is not penetration. '
                   'Do not invent forbidden grasp angles; use existing planner checks and actual 3D evidence. '
                   if self.contact_manipulation else '')
                + REFINE_GUIDANCE +
                'finish {candidate_ref: inspected candidate, reason: concrete visual basis} or '
                '{status: needs_point, reason: why no candidate is suitable and what evidence is missing}. '
                'You may regenerate or inspect alternatives, but cannot move, execute, or delegate. The planner decides what to do with your judgment.')
        if self.first_grasp_only:
            common += 'FIRST GRASP ONLY: no placement. Stop after observed close/lift/hold; command success is not pickup evidence. '
        if role == 'prime':
            return common + ('You are Prime. Dynamically delegate to independent Point or Grasp '
                             'children in any order, including Point -> Grasp -> Point. '
                             'Delegate only the minimal task and required point_ref. '
                             'finish arguments: {status: completed|failed|unknown}.')
        if role == 'point':
            return common + ('You are Point. Observe RGB views, select normalized u,v in [0,1000], then call point with observation_id, view_id, u, v. Inspect returned segmentation images and re-point until acceptable. '
                             'Use inspect_point_cloud to examine actual observed 3D clouds; correct wrong masks by re-pointing. '
                             'When multiview mode is enabled: first select the object in front, call observe_target '
                             'to rise then look down from high above that cloud, discarding all old references. '
                             'In the returned synchronized observation explicitly click the SAME object in front AND wrist, '
                             'inspect both clouds, fuse_points(point_refs=[front_ref, wrist_ref], same_object=your explicit '
                             'same-object description/confirmation), inspect the fused cloud, then finish '
                             'with the latest fused reference. Occlusion or inconsistent geometry requires reobservation, '
                             'never a guessed wrist click or silent single-view fallback. While holding, select placement '
                             'from fresh RGB only; never observe_target or change held orientation. '
                             'finish arguments: {point_ref: a reference produced in this session}.')
        return common + ('You are Grasp. Generate candidates, validate and execute iteratively. '
                         'Prefer top-down claw grasps: downward approach with fingers straddling opposite object sides, '
                         'not an oblique side swipe. Compare actual 3D cloud context with candidate previews and '
                         'downward angle; never rotate a predicted pose or treat rank as success. '
                         'If no suitable actual prediction is reachable, request new Point evidence. '
                         'Only execute an accepted validation for your selected candidate. '
                         'After execute_grasp you MUST observe fresh RGB and inspect the gripper '
                         'before finish(execution_ref). Command succeeded is not physical grasp success. '
                         'If the gripper is empty, finish with status needs_point and reason empty_grip. '
                         'finish arguments: {execution_ref: a reference produced in this session}.')

    def _finish(self, role: str, args: dict[str, Any], scope: dict[str, Any]) -> dict[str, Any]:
        if self.active_perception:
            if role == 'grasp':
                if set(args) == {'status', 'reason'} and args['status'] in ('needs_point', 'failed'):
                    return {'status': args['status'], 'reason': _text(args['reason'])}
                if set(args) != {'candidate_ref', 'reason'} or scope['inspected_candidates'].get(args.get('candidate_ref')) != self._epoch:
                    raise BoundaryError('finish requires an inspected current candidate and visual reason')
                return {'candidate_ref': args['candidate_ref'], 'reason': _text(args['reason'])}
            if role == 'point':
                if set(args) == {'waypoint_ref'} and scope['waypoints'].get(args['waypoint_ref']) == self._epoch:
                    self._waypoint_refs[args['waypoint_ref']] = self._epoch
                    return dict(args)
                if set(args) == {'status', 'reason'} and args['status'] == 'failed':
                    return {'status': 'failed', 'reason': _text(args['reason'])}
                if ((self.target_intent_mode or self.contact_manipulation) and set(args) == {'point_ref'}
                        and self._view_only_refs.get(args['point_ref']) == self._epoch
                        and scope['points'].get(args['point_ref']) == self._epoch
                        and args['point_ref'] == getattr(self.backend, 'latest_view_ref', None)):
                    self._point_refs[args['point_ref']] = self._epoch
                    return {**args, 'selection_scope': 'view_only'}
                if ('point_ref' in args and getattr(self.backend, 'held_plan', None) is None
                        and args['point_ref'] != self.backend.latest_fused_ref):
                    raise BoundaryError('Point must return current front+wrist fusion')
            if role == 'prime' and self.first_grasp_only and scope['executions']:
                ref = args.get('execution_ref')
                if (ref not in scope['executions'] or args.get('actor_visual_assessment') not in ('held', 'empty', 'uncertain')
                        or scope['latest_observation'] is None
                        or scope['observations'][scope['latest_observation']][0] != self._epoch
                        or args.get('status') not in ('completed', 'failed', 'unknown')
                        or set(args) != {'status', 'execution_ref', 'actor_visual_assessment'}):
                    raise BoundaryError('observe execution and report explicit hold assessment')
                return {**args, 'verified': False}
        if role == 'prime':
            if set(args) != {'status'} or args['status'] not in ('completed', 'failed', 'unknown'):
                raise BoundaryError('invalid prime finish')
            return {'status': args['status'], 'verified': False}
        if set(args) == {'status', 'reason'}:
            statuses = ('failed',) if role == 'point' else ('failed', 'needs_point')
            if args['status'] not in statuses or args['reason'] not in (
                    'no_candidates', 'wrong_target', 'unreachable', 'needs_reobserve', 'no_valid_mask', 'empty_grip'):
                raise BoundaryError('invalid child recovery result')
            if self.debug_reset_on_failed_grasp and role == 'grasp' and scope['executions']:
                latest = scope['latest_observation']
                if latest is None or scope['observations'][latest][0] != self._epoch:
                    raise BoundaryError('observe after execution before reporting grip failure')
            return {'status': args['status'], 'reason': args['reason']}
        same_object = None
        if role == 'point' and scope.get('targeted_reobserve'):
            args = dict(args)
            same_object = _text(args.pop('same_object', None))
        assessment = None
        if role == 'grasp' and self.first_grasp_only and 'execution_ref' in args:
            args = dict(args)
            assessment = args.pop('actor_visual_assessment', None)
            if assessment not in ('held', 'empty', 'uncertain'):
                raise BoundaryError('explicit visual hold assessment required')
        key = 'point_ref' if role == 'point' else 'execution_ref'
        allowed = scope['points'] if role == 'point' else scope['executions']
        if set(args) != {key} or _text(args[key]) not in allowed:
            raise BoundaryError(f'finish requires exactly {{{key}: current session reference}}; omit status/reason on success')
        if role == 'point':
            if scope['points'][args[key]] != self._epoch:
                raise BoundaryError('stale point')
            if (not self.active_perception and getattr(self.backend, 'multiview', False)
                    and getattr(self.backend, 'held_plan', None) is None):
                if (args[key] != self.backend.latest_fused_ref
                        or args[key] not in self.backend.inspected_cloud_refs):
                    raise BoundaryError('finish requires inspected latest fused cloud')
            self._point_refs[args[key]] = self._epoch
            return {key: args[key], **({'same_object': same_object} if same_object else {})}
        observation_id = scope['latest_observation']
        execution_epoch = scope['execution_epochs'][args[key]]
        if (execution_epoch != self._epoch or observation_id is None
                or scope['observations'][observation_id][0] != execution_epoch):
            raise BoundaryError('observe after execution before finishing')
        result = {key: args[key], 'status': scope['executions'][args[key]],
                  'observation_id': observation_id, 'verified': False}
        if assessment is not None:
            result['actor_visual_assessment'] = assessment
        return result

    def _view_request_target(self, point_ref):
        # Reselection of the same object never resets its view budget.
        return 'pick_object'

    def _dispatch(self, role: str, tool: str, args: dict[str, Any], scope: dict[str, Any],
                  task: dict[str, Any], sid: str) -> dict[str, Any]:
        if self.target_intent_mode and self.first_grasp_only and self._debug_execution_ref and tool != 'observe':
            raise BoundaryError('physical attempt ended; assess fresh hold images and finish')
        if (self.debug_reset_on_failed_grasp and role == 'grasp' and scope['executions']
                and tool not in ('observe',)):
            raise BoundaryError('debug attempt requires observation and finish, not another grasp')
        if tool.startswith('delegate_'):
            if tool == 'delegate_point':
                self._current_view_target = 'pick_object'
            if self._delegations >= self.budgets.max_delegations:
                raise BudgetExceeded()
            if self.debug_reset_on_failed_grasp and tool == 'delegate_grasp' and self._debug_execution_ref:
                self._fresh_attempt(sid, 'second_grasp_delegation', self._debug_execution_ref)
            if tool == 'delegate_grasp' and self._point_refs.get(args['point_ref']) != self._epoch:
                raise BoundaryError('point reference not returned by Point')
            if tool == 'delegate_grasp' and args['point_ref'] in self._view_only_refs:
                raise BoundaryError('view_only target requires move_to_view and fresh paired selection before grasping')
            self._delegations += 1
            child_task = deepcopy(args)
            if self.active_perception and tool == 'delegate_grasp':
                child_task['preview_context'] = {
                    'view_requests_used': self._view_requests.get(self._view_request_target(args['point_ref']), 0),
                    'view_request_limit': 2,
                    'guidance': 'Orbit/zoom and refine a nearly suitable current candidate; finish unknown if none is adequate.'}
            if self.target_intent_mode and tool == 'delegate_point' and self._current_observation:
                child_task['observation'] = deepcopy(self._current_observation)
            if self.active_perception and tool == 'delegate_point':
                # This is the backend operation mode, not a claim of verified physical holding.
                child_task['point_mode'] = 'placement' if getattr(self.backend, 'held_plan', None) is not None else 'target'
            if self.target_intent_mode and tool == 'delegate_grasp':
                ref = args['point_ref']
                reviews = self._reviews_per_target.get(ref, 0)
                if reviews >= 3:
                    return {'status': 'needs_point', 'reason': 'candidate_review_limit; request a new view or finish unknown'}
                self._reviews_per_target[ref] = reviews + 1
                try:
                    cached = ref in self._candidate_bundles
                    if not cached:
                        self._candidate_bundles[ref] = public_result('grasp_candidates', self.backend.grasp_candidates(point_ref=ref))
                    bundle = deepcopy(self._candidate_bundles[ref])
                    bundle['candidates'] = [c for c in bundle['candidates'] if c['candidate_ref'] not in self._rejected_candidates]
                    child_task['candidate_bundle'] = bundle
                    child_task['observed_cloud'] = deepcopy(self._target_evidence.get(ref, {}))
                    child_task['excluded_candidates'] = [
                        {'candidate_ref': c['candidate_ref'], 'reason_code': self._rejected_candidates[c['candidate_ref']]}
                        for c in self._candidate_bundles[ref]['candidates'] if c['candidate_ref'] in self._rejected_candidates]
                    self._event('candidate_bundle_prepared', sid, point_ref=ref, cached=cached,
                                available=len(bundle['candidates']), excluded=len(child_task['excluded_candidates']))
                    if not bundle['candidates']:
                        return {'status': 'needs_point', 'reason': 'no_remaining_candidates; request a new view or finish unknown'}
                except Exception:
                    self._event('backend_error', sid, role=role, tool='grasp_candidates')
                    return {'error': 'backend_error', 'reason_code': 'candidate_generation_failed', 'recovery': 'needs_reobserve'}
            elif self.active_perception and tool == 'delegate_grasp':
                try:
                    child_task['observed_cloud'] = public_result('inspect_point_cloud', self.backend.inspect_point_cloud(args['point_ref']))
                except Exception:
                    return {'error': 'backend_error', 'recovery': 'needs_reobserve'}
            child_role = tool.removeprefix('delegate_')
            if tool == 'delegate_waypoint':
                child_role = 'point'
                child_task['waypoint_task'] = True
                if self.contact_manipulation:
                    child_task['contact_plane_available'] = (getattr(self.backend, 'held_plan', None) is not None
                        or getattr(self.backend, 'closed_push', False))
                if self._current_observation:
                    child_task['observation'] = deepcopy(self._current_observation)
            if self.active_perception and tool == 'delegate_grasp':
                child_task['current_scene'] = self.backend.saved_views(args['point_ref'])
            status, result = self._loop(child_role, child_task, sid)
            if self.active_perception and tool == 'delegate_grasp' and 'candidate_ref' in result:
                scope['candidates'][result['candidate_ref']] = self._epoch

            if self.debug_reset_on_failed_grasp and tool == 'delegate_grasp' and status != 'completed':
                self._fresh_attempt(sid, 'grasp_' + status, result.get('execution_ref'))
            if self.first_grasp_only and not self.active_perception and tool == 'delegate_grasp':
                self._event('tool_result', sid, role=role, tool=tool, result={'status': status, 'result': result})
                raise FirstGraspFinished(result)
            return {'status': status, 'result': result}
        if tool == 'move_to_view':
            if (not (self.target_intent_mode or self.contact_manipulation) or role != 'prime'
                    or self._point_refs.get(args['point_ref']) != self._epoch):
                raise BoundaryError('view movement requires a current target returned by Point')
            target = self._view_request_target(args['point_ref'])
            used = self._view_requests.get(target, 0)
            if used >= 2:
                raise BoundaryError('view request limit reached; use current evidence or finish unknown')
            self._view_requests[target] = used + 1
            try:
                output = public_result(tool, self.backend.move_to_view(**args))
            except Exception:
                self._current_observation = None
                self._point_refs.clear()
                scope['latest_observation'] = None
                return {'error': 'backend_error', 'reason_code': 'view_state_unavailable', 'recovery': 'Obtain a fresh observation and reselect the target before continuing'}
            finally:
                self._epoch = self.backend.epoch
            if output['observation_id'] is None:
                self._current_observation = None
                self._point_refs.clear()
                scope['latest_observation'] = None
            else:
                changed = not self._current_observation or output['observation_id'] != self._current_observation['observation_id']
                self._remember_observation(output, scope, invalidate=changed)
            return output
        if tool == 'turn':
            observed = scope['observations'].get(scope['latest_observation'])
            if not self.contact_manipulation or role != 'prime' or not observed or observed[0] != self._epoch:
                raise BoundaryError('turn requires a fresh contact observation; observe after grasping or moving')
            try:
                output = public_result('turn', self.backend.turn(**args))
            except Exception:
                scope['latest_observation'] = None
                self._current_observation = None
                return {'error': 'turn_motion_unavailable', 'recovery': 'Observe actual contact before continuing; the turn may be partial.'}
            finally:
                self._epoch = self.backend.epoch
            self._remember_observation(output, scope, invalidate=True)
            return output
        if tool == 'move':
            if self.waypoint_views:
                used = self._view_requests.get(self._current_view_target, 0)
                if used >= 2:
                    raise BoundaryError('two waypoint moves used; delegate current candidate preview/refinement or finish unknown')
                if role != 'prime' or self._waypoint_refs.get(args['waypoint_ref']) != self._epoch:
                    raise BoundaryError('waypoint must be returned by current Point session')
                self._view_requests[self._current_view_target] = used + 1
            if not self.active_perception or role != 'prime' or self._waypoint_refs.get(args['waypoint_ref']) != self._epoch:
                raise BoundaryError('move requires a current waypoint returned by Point')
            try:
                output = public_result('move', self.backend.move(**args))
            except Exception:
                if self.target_intent_mode:
                    self._current_observation = None
                    self._point_refs.clear()
                return {'error':'waypoint_motion_unavailable', 'recovery':'Choose another waypoint or refresh observation; the requested motion was not established.'}
            finally:
                self._epoch = self.backend.epoch
            self._remember_observation(output, scope, invalidate=self.target_intent_mode)
            return output
        if tool in ('waypoint', 'contact_waypoint', 'contact_destination', 'push_waypoint'):
            observed = scope['observations'].get(args['observation_id'])
            if role != 'point' or not observed or observed[0] != self._epoch:
                raise BoundaryError('waypoint needs current session observation')
            if tool in ('contact_waypoint', 'contact_destination', 'push_waypoint') and args['view_id'] not in observed[1]:
                raise BoundaryError('unknown contact view')
        if tool in ('inspect_point_cloud', 'observe_target', 'fuse_points'):
            refs = args['point_refs'] if tool == 'fuse_points' else [args['point_ref']]
            if any(scope['points'].get(ref) != self._epoch for ref in refs):
                if self.active_perception and role == 'point' and tool == 'fuse_points':
                    return {'error': 'stale_or_foreign_point_refs',
                            'recovery': 'Reselect the same target in the NEW observation front/agentview and wrist views. Fuse ONLY references created by this Point session from that observation; never historical refs. If both views cannot support the target, report failed with the reason.',
                            'current_point_refs': [ref for ref, epoch in scope['points'].items() if epoch == self._epoch]}
                raise BoundaryError('point outside current Point session')
        if tool in ('point', 'select_target', 'select_view_target'):
            observation = scope['observations'].get(args['observation_id'])
            if (observation is None or observation[0] != self._epoch
                    or (tool in ('point', 'select_view_target') and args['view_id'] not in observation[1])):
                raise BoundaryError('observation or view outside current session')
        if tool == 'grasp_candidates' and (args['point_ref'] != task.get('point_ref') or self._point_refs.get(args['point_ref']) != self._epoch):
            raise BoundaryError('point outside delegation')
        if tool in ('inspect_candidate', 'preview_candidate', 'refine_candidate', 'validate_grasp', 'execute_grasp') and scope['candidates'].get(args['candidate_ref']) != self._epoch:
            raise BoundaryError('candidate outside session')
        if self.target_intent_mode and tool in ('inspect_candidate', 'preview_candidate', 'refine_candidate', 'validate_grasp', 'execute_grasp') and args['candidate_ref'] in self._rejected_candidates:
            raise BoundaryError('candidate already rejected; review an alternative or request a new view')
        if tool == 'execute_grasp':
            if self.debug_reset_on_failed_grasp and self._calls >= self.budgets.max_tool_calls:
                raise BudgetExceeded()  # reserve one post-execution diagnostic observation
            if scope['validations'].get(args['validation_ref']) != args['candidate_ref']:
                raise BoundaryError('missing accepted validation')
            # Consume before side effects: retries need a new validation even on failure.
            del scope['validations'][args['validation_ref']]
        if tool == 'place' and self._point_refs.get(args['point_ref']) != self._epoch:
            raise BoundaryError('placement point not returned by Point')
        if tool in ('execute_grasp', 'place', 'observe_target', 'release', 'close_for_push'):
            # Motion attempts invalidate all earlier visual evidence, even on error.
            self._epoch += 1
        try:
            raw = getattr(self.backend, tool)(**args)
        except Exception as exc:
            if (self.contact_manipulation and tool == 'grasp_candidates'
                    and (getattr(self.backend, 'held_plan', None) is not None or getattr(self.backend, 'closed_push', False))):
                return {'error': 'release_required', 'recovery':
                    'A previous grasp or closed-pusher command is still active. Prime must release before a new grasp if the fresh images show the object was not retained.'}
            if tool == 'contact_destination':
                return {'error': 'invalid_contact_destination', 'recovery': 'Establish grasp or achieve push contact first. In a fresh image click a visible point ON the movable object and a nearby destination for that SAME point, not the gripper or background.'}
            if tool in ('contact_waypoint', 'push_waypoint'):
                return {'error': 'invalid_contact_point', 'recovery': 'Choose a nearby visible front-image point on the current grasp height plane.'}
            if tool == 'waypoint':
                return {'error':'invalid_point_pair', 'recovery':'Use matching points in both views; rays must meet in front of both cameras with low reprojection error. Correct the pair or report failed.'}
            if tool == 'observe_target' and type(getattr(self.backend, 'epoch', None)) is int:
                # Backend consumes epoch before motion, not on preflight rejection.
                self._epoch = self.backend.epoch
            self._event('backend_error', sid, role=role, tool=tool)
            if tool in ('observe_target', 'inspect_point_cloud', 'fuse_points'):
                return {'error': 'backend_error', 'recovery': 'needs_reobserve'}
            return {'error': 'backend_error'}  # never expose private exception text
        finally:
            if self.active_perception and tool in ('execute_grasp', 'place', 'observe_target', 'release', 'close_for_push'):
                self._epoch = self.backend.epoch
        if tool == 'refine_candidate' and not raw.get('accepted', True):
            return {'error': 'refinement_rejected', 'reason_code': raw.get('reason_code', 'rejected'),
                    'recovery': 'Keep the original candidate, try a smaller adjustment, or inspect another candidate.'}
        output = public_result(tool, raw)
        if tool in ('observe', 'observe_target'):
            self._remember_observation(output, scope, invalidate=self.target_intent_mode)
        elif tool in ('waypoint', 'contact_waypoint', 'contact_destination', 'push_waypoint'):
            scope['waypoints'][output['waypoint_ref']] = self._epoch
        elif tool in ('inspect_candidate', 'preview_candidate'):
            scope['inspected_candidates'][output['candidate_ref']] = self._epoch
        elif tool == 'refine_candidate':
            # A refinement arrives with its own overlay; it is reviewed evidence already.
            scope['candidates'][output['candidate_ref']] = self._epoch
            scope['inspected_candidates'][output['candidate_ref']] = self._epoch
        elif tool in ('point', 'select_target', 'select_view_target'):
            if output['observation_id'] != args['observation_id']:
                raise BoundaryError('point observation mismatch')
            if output.get('point_ref'):
                scope['points'][output['point_ref']] = self._epoch
                if tool in ('select_target', 'select_view_target'):
                    self._point_refs.clear()
                    self._target_evidence[output['point_ref']] = deepcopy(output)
                if tool == 'select_view_target':
                    self._view_only_refs[output['point_ref']] = self._epoch
        elif tool == 'fuse_points':
            scope['points'][output['point_ref']] = self._epoch
        elif tool == 'grasp_candidates':
            scope['candidates'].update({item['candidate_ref']: self._epoch for item in output['candidates']})
        elif tool == 'validate_grasp':
            if output['candidate_ref'] != args['candidate_ref']:
                raise BoundaryError('validation candidate mismatch')
            scope['validations'].pop(output['validation_ref'], None)
            if output['accepted']:
                scope['validations'][output['validation_ref']] = output['candidate_ref']
            elif self.target_intent_mode:
                self._rejected_candidates[output['candidate_ref']] = output.get('reason_code', 'planning_failed')
        elif tool == 'execute_grasp':
            self._debug_execution_ref = output['execution_ref']
            scope['executions'][output['execution_ref']] = output['status']
            scope['execution_epochs'][output['execution_ref']] = self._epoch
        if (tool == 'execute_grasp' and self.debug_reset_on_failed_grasp
                and output['status'] == 'failed' and not self.target_intent_mode):
            # Completed execution failure still gets fresh sensor evidence before reset.
            self._event('tool_result', sid, role=role, tool=tool, result=output)
            self._calls += 1
            self._event('tool_called', sid, role=role, tool='observe', arguments={})
            observed = self._dispatch(role, 'observe', {}, scope, task, sid)
            self._event('tool_result', sid, role=role, tool='observe', result=observed)
            self._fresh_attempt(sid, 'execution_failed', output['execution_ref'])
        return output

    def _environment_observation(self, scope, sid, *, refresh=False):
        """Capture after motion; otherwise reuse the still-current RGBD pair."""
        if refresh or self._current_observation is None:
            output = public_result('observe', self.backend.observe())
            self._remember_observation(output, scope, invalidate=True)
            self._event('environment_observed', sid, observation=output, epoch=self._epoch)
        observation = self._current_observation
        if len(observation['views']) != 2:
            raise BoundaryError('paired front/wrist environment images are required')
        scope['latest_observation'] = observation['observation_id']
        scope['observations'][observation['observation_id']] = (self._epoch, {v['view_id'] for v in observation['views']})
        return deepcopy(observation)

    def _agent_observation(self, role, task, observation):
        return observation

    def _remember_observation(self, output, scope, *, invalidate=False):
        observation = {key: deepcopy(output[key]) for key in ('observation_id', 'views')}
        self._current_observation = observation
        scope['latest_observation'] = observation['observation_id']
        scope['observations'][observation['observation_id']] = (self._epoch, {v['view_id'] for v in observation['views']})
        if invalidate:
            self._point_refs.clear()
            self._candidate_bundles.clear()
            self._target_evidence.clear()
            scope['candidates'].clear()
            scope['validations'].clear()

    def _fresh_attempt(self, sid, cause, execution_ref):
        self._event('fresh_attempt_required', sid, cause=cause, execution_ref=execution_ref,
                    reset_scope=['simulator', 'prime', 'point', 'grasp'])
        raise FreshAttemptRequired(cause, execution_ref)
