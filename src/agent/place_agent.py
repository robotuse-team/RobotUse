"""Prepare and inspect placement candidates for a measured destination."""
from .base_agent import BaseAgent
from .prompts import _HEIGHT_GUIDANCE
from src.agent.intent_prompts import COMMON, FEEDBACK


class PlaceAgent(BaseAgent):
    role = 'place'

    def base_prompt(self, context):
        prompt = (
            'You are Place. Call explicit_place_candidates for the delegated destination_ref to inspect '
            'the measured click, segment median and mean XY choices plus observed geometry. Place travels '
            'to transit Z at current XY, across to target XY/orientation, then to final Z; adjustments '
            'preserve the resolved transit Z and cannot raise it implicitly. The cyan left-shoulder-only '
            'gripper projections at this stage use each option\'s measured reference Z plus 0.05 m '
            'along base Z for display only; the observed surface coordinates remain unchanged. They are '
            'comparison previews, not an approved execution height or executable candidate. Select '
            'xy_source and explicitly provide xy_m (copy that candidate or modify X/Y), final height '
            'and independent transit_height (not below final height) to '
            'explicit_prepare_place. Review its new gripper projection at your selected execution XYZ, '
            'optionally inspect_place_candidate, '
            'and use explicit_adjust_place to translate or rotate the pending pose if useful: base XYZ '
            + ('millimetres without step/cumulative magnitude caps; gripper-local roll/pitch/yaw '
               if context.unrestricted_pose_translation else
               'millimetres, 30 per axis per step and 90 cumulatively; gripper-local roll/pitch/yaw ')
            + 'degrees about the jaw contact center. '
            + ('Angular magnitude is unrestricted; all values must be finite and the resulting motion must validate. '
               if context.unrestricted_pose_rotation else
               'Angular limits are 10 per axis per step and 30 cumulatively. ')
            + 'Supply all six delta fields, zero for unchanged axes. Each accepted adjustment returns '
            'a new checked candidate and preview; a rejection never changes or executes the old pose. '
            'Then finish status=success with its candidate_ref, or failed with a concrete limitation. '
            + ('To request an independent Refiner edit of this one inspected candidate before movement, '
               'finish status=needs_refinement with candidate_ref and a concrete reason. '
               if context.requested_refine_routes else '')
            + 'The selected motion keeps holding. Prime sees fresh RGB afterward and makes a separate '
            'release decision; no placement selection authorizes opening the gripper.'
        )
        return COMMON + prompt + " " + _HEIGHT_GUIDANCE + FEEDBACK
