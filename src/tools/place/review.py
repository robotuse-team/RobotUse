"""Independent Place subagent policy, tools and session-local choice boundary.

The shared Prime session runner supplies a fresh provider conversation for role
``place``. This agent owns AnyPlace proposal generation and visual review;
Prime alone authorizes validation and physical execution of the returned choice.
"""
from src.backend.controller import BoundaryError, public_result, _text


class PlaceAgent:
    # This session exposes candidate generation and inspection only.
    tools = ('place_candidates', 'inspect_place_candidate', 'finish')
    prompt = (
        'You are Place, an independent visual placement judge. Your delegated task contains the destination_ref, '
        'placement instruction and current visual hold/destination assessments. Current paired front/wrist RGB '
        'is supplied automatically each turn. Use place_candidates with the delegated destination and assessments '
        'to request the AnyPlace tool. It predicts unranked OBJECT transforms from measured object/destination clouds. '
        'If the object is not visibly held or the destination may have moved, finish {status: failed, reason}. '
        'The backend filters the closed-hand release goal for held-object/gripper scene collision, static IK and '
        'robot self collision, and generates lift -> transit -> release -> retreat with the existing planner. '
        'At most four passing candidates are uniformly sampled. Compare ALL offered review cards before choosing. '
        'Cards use the same calibrated mesh rendering as Grasp: cyan/magenta fingers, orange predicted object '
        'surfaces, saved FRONT/WRIST destination RGB and a virtual scene view. Photos were captured BEFORE pickup; '
        'overlays show predicted release poses, including occluded geometry, not current or future sensor images. '
        'Compare the object orientation and support relationship with the requested placement. Inspect a promising '
        'candidate with inspect_place_candidate; finish exactly {candidate_ref: inspected choice, reason: visual basis}. '
        'If none is suitable, finish exactly {status: failed, reason: concrete problem}. Never automatically choose '
        'the first candidate. No opening, other-goal or joint-path collision proof is provided. The stored route '
        'will be executed only after Prime validates your returned choice. You do not move, execute, or delegate. '
        'Use supplied opaque references only; never invent metric poses, joints, code, or hidden object geometry.'
    )

    def dispatch(self, owner, tool, args, scope, task, sid):
        if tool == 'place_candidates':
            keys = ('destination_ref', 'hold_assessment', 'destination_assessment')
            if any(args[k] != task[k] for k in keys):
                raise BoundaryError('use the destination and assessments supplied in this Place delegation')
            # One pool per child conversation; repeated calls cannot resample a preferred pose.
            if 'place_bundle' not in scope:
                try:
                    raw = owner.backend.place_candidates(**args)
                except Exception as exc:
                    diagnostic = getattr(owner, '_backend_failure', None)
                    if callable(diagnostic):diagnostic(sid, 'place', tool, exc)
                    owner._event('backend_error', sid, role='place', tool=tool)
                    return {'error': 'placement_generation_failed', 'recovery': 'Finish failed; inference/planning evidence is saved privately.'}
                output = public_result('place_candidates', raw)
                scope['place_bundle'] = output
                scope['candidates'].update({c['candidate_ref']: owner._epoch for c in output['candidates']})
            return scope['place_bundle']
        ref = args['candidate_ref']
        if scope['candidates'].get(ref) != owner._epoch:
            raise BoundaryError('inspect a candidate generated in this Place session')
        if tool == 'refine_place_candidate':
            try:
                raw = owner.backend.refine_place_candidate(**args)
            except Exception:
                owner._event('backend_error', sid, role='place', tool=tool)
                return {'error': 'refinement_failed', 'recovery': 'Keep the original candidate or inspect another.'}
            if not raw.get('accepted', True):
                return {'error': 'refinement_rejected', 'reason_code': raw.get('reason_code', 'rejected'),
                        'recovery': 'The adjusted release goal failed its own checks. Try a smaller adjustment, '
                                    'keep the original candidate, or inspect another.'}
            output = public_result('refine_place_candidate', raw)
            # The refinement carries its own review card, so it is a chooseable candidate.
            scope['candidates'][output['candidate_ref']] = owner._epoch
            scope['inspected_candidates'][output['candidate_ref']] = owner._epoch
            return output
        output = public_result('inspect_place_candidate', owner.backend.inspect_place_candidate(**args))
        if output['candidate_ref'] != ref:
            raise BoundaryError('placement inspection candidate mismatch')
        scope['inspected_candidates'][ref] = owner._epoch
        return output

    def finish(self, args, scope, epoch):
        if set(args) == {'status', 'reason'} and args['status'] == 'failed':
            return {'status': 'failed', 'reason': _text(args['reason'])}
        if (set(args) != {'candidate_ref', 'reason'}
                or scope['candidates'].get(args.get('candidate_ref')) != epoch
                or scope['inspected_candidates'].get(args.get('candidate_ref')) != epoch):
            raise BoundaryError('Place must return an inspected current-session candidate with a visual reason')
        return {'candidate_ref': _text(args['candidate_ref']), 'reason': _text(args['reason'])}
