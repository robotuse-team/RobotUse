"""Inspect and refine pending grasp and placement poses."""
from .base_agent import BaseAgent
from .prompts import _HEIGHT_GUIDANCE
from src.agent.intent_prompts import COMMON, FEEDBACK


class RefinerAgent(BaseAgent):
    role = 'refiner'

    def base_prompt(self, context):
        prompt = (
            'You are Refiner. Inspect the selected pose and geometry, keep it if suitable or use '
            + ('adjust_grasp with finite unbounded base XYZ millimetres and local rotations in degrees. '
               if context.unrestricted_pose_translation else
               'adjust_grasp with bounded base XYZ millimetres and local rotations in degrees. ')
            + ('All sources, including initially top-down mean/median candidates, permit finite roll/pitch/yaw '
               'without per-step or cumulative angular limits. Edited poses must pass the same motion checks. '
               if context.unrestricted_pose_rotation else
               'Mean/median poses permit yaw ONLY: roll_deg and pitch_deg remain zero. CGN poses permit '
               'full rotations, limited to 10 degrees per axis/step and 30 cumulatively. ')
            + 'Return status=success with '
            'an inspected current candidate_ref or failed/needs_observation; a failure never approves '
            'the previous choice. During inflight_refinement, the gripper is paused OPEN before descent: '
            + ('nudge_grasp uses finite base XYZ millimetres without step/cumulative caps and local rotations '
               if context.unrestricted_pose_translation else
               'nudge_grasp uses base XYZ millimetres (30 per step, 90 cumulative) and local rotations ')
            + ('with finite values and no angular magnitude limits. '
               if context.unrestricted_pose_rotation else
               '(10 degrees per step, 30 cumulative). Geometric poses still forbid roll/pitch. ')
            + 'Inspect the new preview after any nudge. Explicit finish status=continue permits closing; status=abort '
            'keeps it open. A failed or incomplete session never approves closing.'
        )
        return COMMON + prompt + " " + _HEIGHT_GUIDANCE + FEEDBACK
