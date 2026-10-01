"""Select and inspect grasp candidates from measured geometry."""
from .base_agent import BaseAgent
from .prompts import _HEIGHT_GUIDANCE
from src.agent.intent_prompts import COMMON, FEEDBACK


class GraspAgent(BaseAgent):
    role = 'grasp'

    def base_prompt(self, context):
        prompt = (
            'You are Grasp. Use the measured_geometry supplied with your task, then request '
            'explicit_grasp_candidates with one requested mode in direction and independent transit pre/post heights. '
            'direction=vertical or custom returns ONLY Contact-GraspNet candidates; set '
            'geometric_height=null and give tolerance_deg as a HARD angular admissibility limit. '
            'Custom polar_deg is measured from base -Z and azimuth_deg from base +X toward +Y; '
            'custom requires both angles, vertical requires null angles. These requests '
            'filter learned poses without rotating them or adding statistical alternatives. '
            'direction=mean or median returns ONLY that measured center at four fixed base-Z yaw angles '
            '-45, 0, 45, 90 degrees, strictly top-down. Specify geometric_height; set tolerance_deg, '
            'azimuth_deg and polar_deg to null. The four share the chosen XY and Z. '
            'If CGN is unavailable or unsuitable, explicitly request mean or median in a new call. '
            'Compare cyan candidate previews and path evidence; '
            'inspect a promising candidate before finish status=success or needs_refinement with '
            'candidate_ref. Diagnostics are nonexecutable and may only request refinement. Return '
            'failed or needs_observation when appropriate. You do not execute or release. '
            + ('Refiner automatically reviews your selection before execution.' if context.auto_refine_routes else
               'Automatic pre-execution Refiner review is disabled. To edit one inspected candidate, '
               'finish status=needs_refinement with that candidate_ref and a concrete reason. Only that '
               'selection is routed once to Refiner; status=success skips candidate review. '
               if context.requested_refine_routes else
               'Automatic pre-execution Refiner review is disabled; only accept a suitable executable pose.')
        )
        return COMMON + prompt + " " + _HEIGHT_GUIDANCE + FEEDBACK
