"""Fixed four-call grasp ensemble with 2/2/1/1 publication slots per source."""
from src.agent.intent_prompts import GRASP_APPROACH_PROMPT
from src.backend.controller import BoundaryError

ENSEMBLE_SOURCES = (
    ('graspgen', None, None, 2),
    ('moveit_vertical', 'vertical', 'face', 2),
    ('moveit_vertical_edge', 'vertical', 'edge', 1),
    ('moveit_horizontal', 'horizontal', 'face', 1),
)
ENSEMBLE_INSTRUCTIONS = (
    'Grasp ensemble is fixed at startup; you cannot enable, disable or reconfigure it. '
    'delegate_grasp accepts instruction and point_ref only and returns up to six candidates. '
    'Compare their previews and validation evidence. Diagnostic candidates are not executable; '
    'refine and revalidate them before execution. Fewer candidates may be available. '
    'Repeated delegation of the same current target reuses the entire pool. '
)


def generate_ensemble(backend, point_ref):
    candidates, diagnostics, reports, images, sources = [], [], [], [], []
    for source, direction, family, slots in ENSEMBLE_SOURCES:
        backend._record('grasp_ensemble_call', dict(point_ref=point_ref, ensemble_source=source,
            preferred_direction=direction, grasp_type=family, batch_size=slots))
        bundle = backend._grasp_candidates_single(point_ref, preferred_direction=direction,
            grasp_type=family, batch_size=slots)
        for key, destination in (('candidates', candidates), ('diagnostic_candidates', diagnostics),
                                 ('source_reports', reports)):
            destination.extend({**entry, 'ensemble_source': source} for entry in bundle.get(key, []))
        images.extend(bundle.get('image_refs', []))
        executable = len(bundle.get('candidates', []))
        diagnostic = len(bundle.get('diagnostic_candidates', []))
        sources.append(dict(ensemble_source=source, requested_slots=slots, executable_count=executable,
            diagnostic_count=diagnostic, missing_count=slots-executable-diagnostic,
            reason_codes=list(dict.fromkeys(r['reason_code'] for r in bundle.get('source_reports', [])
                                           if r.get('reason_code')))))
    result = dict(candidates=candidates, diagnostic_candidates=diagnostics,
        source_reports=reports, image_refs=list(dict.fromkeys(images)), generator='ensemble',
        candidate_limit=sum(item[3] for item in ENSEMBLE_SOURCES), ensemble_sources=sources, task_budget=backend.grasp_budget(),
        reason_code='candidate_available' if candidates else 'no_feasible_candidate')
    backend._record('grasp_ensemble_candidates', dict(point_ref=point_ref, **result))
    return result


class EnsembleGraspOrchestrator:
    def _optional_arguments(self, tool):
        return () if tool == 'delegate_grasp' else super()._optional_arguments(tool)

    @staticmethod
    def _grasp_bundle_key(args):
        return (args['point_ref'], 'ensemble')

    def _prompt(self, role):
        prompt = super()._prompt(role).replace(GRASP_APPROACH_PROMPT, '')
        prompt = prompt.replace('Each request shows at most 6 poses total, including any nonexecutable diagnostics. ', '')
        prompt = prompt.replace('Changing preferred_direction requests a different pool; repeating the same target and preference reuses its pool. ', '')
        prompt = prompt.replace('delegate_grasp accepts optional batch_size 1–6 for a fresh pool; omission keeps the existing default. ', '')
        return prompt + ' ' + ENSEMBLE_INSTRUCTIONS

    def _intent_dispatch(self, role, tool, args, scope, task, sid):
        if tool == 'delegate_grasp' and set(args) - {'instruction', 'point_ref'}:
            raise BoundaryError('ensemble delegate_grasp accepts only instruction and point_ref')
        return super()._intent_dispatch(role, tool, args, scope, task, sid)
