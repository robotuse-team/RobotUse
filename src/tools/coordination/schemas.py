"""Role-aware completion contracts owned by coordination tools."""
from types import SimpleNamespace
from ..schema import object_schema

_EVIDENCE = {'type':'array','items':{'type':'string'},'minItems':1,'maxItems':8}


def _finish_contract(*, context=None, role=None, tools=()):
    context = context or SimpleNamespace(active_perception=True, intent_driven=True,
        review_driven=True, target_intent_mode=True, explicit_geometry_enabled=True,
        requested_refine_routes=True)
    name = 'finish'
    if name == "finish":
        if role == "prime":
            properties = {"status": {"type": "string", "enum": ["completed", "failed", "unknown"]}}
            required = ["status"]
        else:
            key = "point_ref" if role == "point" else "execution_ref"
            properties = {key: {"type": "string"}, "status": {"type": "string", "enum": ["failed", "needs_point"]},
                "reason": {"type": "string", "enum": ["no_candidates", "wrong_target", "unreachable", "needs_reobserve", "no_valid_mask", "empty_grip"]}}
            if role == "point":
                properties["same_object"] = {"type": "string"}
            if role == "grasp" and getattr(context, "first_grasp_only", False):
                properties["actor_visual_assessment"] = {"type": "string", "enum": ["held", "empty", "uncertain"]}
            required = []
    if name == "finish" and getattr(context, "active_perception", False):
        fields = (("status", "execution_ref", "actor_visual_assessment") if role == 'prime' else
                  ("point_ref", "waypoint_ref", "status", "reason") if role == 'point' else
                  ("candidate_ref", "status", "reason"))
        if role == 'point' and getattr(context, 'target_intent_mode', False) and not any(t in tools for t in ('waypoint','view_waypoint','propose_waypoint','propose_downward_waypoint','shift_waypoint')):
            fields = ('point_ref', 'status', 'reason')
        if getattr(context, 'review_driven', False):
            if role == 'grasp':fields=('decision','candidate_ref','reason')
            if role == 'refiner':fields=('candidate_ref','waypoint_ref','status','reason')
        properties = {k: {"type": "string"} for k in fields}
        required = []
        if getattr(context, 'intent_driven', False):
            if role == 'prime':
                properties = {'status':{'type':'string','enum':['completed','failed','unknown']},
                    'reason':{'type':'string','minLength':1,'maxLength':8192},
                    'evidence_image_refs':_EVIDENCE}
                required = ['status']
            else:
                properties.update(reason={'type':'string'}, recommendation={'type':'string'},
                                  evidence_image_refs=_EVIDENCE)
                if 'status' in properties:
                    properties['status'] = {'type':'string','enum':['success','failed']}
                if role == 'grasp' or (role == 'place' and
                        getattr(context, 'requested_refine_routes', False)):
                    properties['status'] = {'type':'string', 'enum':[
                        'success','failed','needs_observation','needs_refinement']}
                if (getattr(context, 'explicit_geometry_enabled', False) and role == 'refiner'
                        and not any(t in tools for t in ('nudge_grasp', 'nudge_place'))):
                    properties['status'] = {'type':'string', 'enum':['success','failed','needs_observation']}
                if role == 'refiner' and any(t in tools for t in ('nudge_grasp', 'nudge_place')):
                    # Paused-execution refinement: the only decision is whether the
                    # gripper closes/releases with the current pose.
                    properties = {'status': {'type': 'string', 'enum': ['continue', 'abort']},
                                  'reason': {'type': 'string'},
                                  'evidence_image_refs': _EVIDENCE}
                    required = ['status']
                else:
                    required = []
    description = "Typed " + name + "; use opaque current-session references only."
    if name == 'finish' and getattr(context, 'active_perception', False):
        description += {'point':' Return exactly {point_ref} or {waypoint_ref} on success, without status/reason. Failure: {status: failed, reason}.',
            'grasp':' Return exactly {candidate_ref, reason} or {status: needs_point, reason}.',
            'place':' Return exactly {candidate_ref, reason} for an inspected choice or {status: failed, reason}.',
            'refiner':' Return {candidate_ref, reason} or {waypoint_ref, reason} for the reviewed pose; failure {status: failed, reason}.',
            'prime':' Return {status: completed|failed|unknown}. In first-grasp-only mode after execution also include execution_ref and actor_visual_assessment.'}.get(role, '')
        if role == 'point' and getattr(context, 'target_intent_mode', False) and not any(t in tools for t in ('waypoint','view_waypoint','propose_waypoint','propose_downward_waypoint','shift_waypoint')):
            description = 'Return exactly {point_ref} on success or {status: failed, reason} on failure.'
        if getattr(context, 'review_driven', False) and role=='grasp':
            description='Return decision accepted or needs_refinement with candidate_ref and reason; needs_observation or failed with reason.'
        if getattr(context, 'intent_driven', False):
            description = {
                'point': 'Return point_ref or waypoint_ref for the inspected selection; otherwise status failed.',
                'grasp': 'Return status success, needs_refinement, needs_observation or failed with reason. Add candidate_ref when relevant. The decision field is also accepted.',
                'place': 'Return candidate_ref for the inspected choice; otherwise status failed.',
                'refiner': ('Return status continue to close/release with the current pose or abort to stop without closing; optional reason.'
                            if any(t in tools for t in ('nudge_grasp', 'nudge_place')) else
                            'Return candidate_ref or waypoint_ref for the reviewed pose; otherwise status failed.'),
                'prime': ('Return status completed, failed or unknown. Optionally include a concise reason '
                          'and evidence_image_refs from supplied images for the final assessment. '
                          'Status alone remains valid. Completion is an unverified agent assessment.')
            }.get(role, description)
            if role != 'prime':
                description += ' Reason, evidence images and recommendation are optional. Status success selects a result; failed/needs_observation reports never approve attached references.'
                if role in ('point', 'place', 'refiner'):
                    description += (' Success may include status success with exactly one selection reference; '
                                    'this approves a selection, not task execution. Failure may attach references as context. '
                                    'Never substitute an image_ref for a selection reference.')
    if name == 'finish' and role == 'progress_review':
        text_schema = {'type':'string', 'minLength':1, 'maxLength':4000}
        assessment_schema = {'type':'string', 'enum':['supported','contradicted','uncertain']}
        refs_schema = {'type':'array', 'items':{'type':'string'}, 'minItems':1, 'maxItems':2}
        condition_properties = {'condition':text_schema, 'assessment':assessment_schema,
                                'reason':text_schema, 'evidence_image_refs':refs_schema}
        properties = {
            'objective_assessment':assessment_schema,
            'conditions':{'type':'array', 'minItems':1, 'maxItems':12, 'items':{
                'type':'object', 'properties':condition_properties,
                'required':list(condition_properties), 'additionalProperties':False}},
            'reason':text_schema, 'evidence_image_refs':refs_schema,
            'unknowns':{'type':'array', 'items':text_schema, 'maxItems':12}}
        required = list(properties)
        description = ('Review the original objective and supplied conditions against current RGB. '
            'Return all review fields, with reasons and current image references. '
            'Historical conditions or hidden contact may remain uncertain. '
            'This is visual review, not native task verification or permission to execute.')
    return object_schema(properties, required), description


def finish_schema(**kwargs):
    return _finish_contract(**kwargs)[0]


def finish_description(**kwargs):
    return _finish_contract(**kwargs)[1]
