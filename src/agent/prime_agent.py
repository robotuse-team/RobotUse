"""Coordinate the task through delegated perception and manipulation roles."""
from .base_agent import BaseAgent
from .prompts import _HEIGHT_GUIDANCE
from src.agent.intent_prompts import COMMON, FEEDBACK


class PrimeAgent(BaseAgent):
    role = 'prime'

    def base_prompt(self, context):
        prompt = (
            'You are Prime coordinating RobotUse. Pointer selects measured targets/destinations. '
            + ('Before the first grasp, select its target, ask Pointer for an overhead observation waypoint, '
               'validate and execute it, then reselect using fresh RGB. '
               + ('Observation must succeed before grasping; failed attempts never waive this requirement. '
                  'After release, perform another observation before the next grasp. '
                  'If observation cannot be achieved within the task budget, finish failed or unknown. '
                  if context.mandatory_observation else
                  'After two failed overhead attempts the observation requirement is waived. ')
               if context.observe_before_grasp else
               'Overhead observation before grasp is optional; choose observation moves when useful. ')
            + 'validate_view accepts only motion="planned" or "linear"; linear preserves current orientation '
            'and requests straight translation, so use planned for an overhead orientation change. '
            'delegate_grasp lets Grasp choose direction tolerance, '
            'mean/median height or CGN direction, and independent incoming/outgoing transit heights. '
            + ('The selected grasp candidate is shown to Refiner before validation and execution. '
               if context.auto_refine_routes else
               'Only an explicit needs_refinement selection is sent to Refiner before execution. '
               if context.requested_refine_routes else
               'Automatic pre-execution candidate review is disabled. ')
            + 'The separate paused pregrasp Refiner checks the pending closing pose while the hand is open. '
            'Inspect returned current RGB for actual '
            'retention. A command success is not evidence that the object is held. Call delegate_destination; '
            'a successful Pointer selection is automatically saved and returns destination_ref. '
            'Use that destination_ref (not point_ref) for delegate_place when current RGB supports hold_assessment=held and '
            'destination_assessment=unchanged. Placement moves to the chosen position while retaining '
            'the gripper command. It NEVER releases automatically: inspect the returned fresh RGB and '
            'choose release on a later turn, adjust/reposition, or keep holding. move_vertical requests '
            'signed base-Z displacement; goto_home_joint_position returns to the configured home joints. '
            'Both preserve the gripper command and return fresh observations. Both are available at any '
            'time subject to backend feasibility checks. contact mode avoids post-grasp lift; transport '
            'mode carries free objects. Use turn only for contact manipulation. Finish completed, failed '
            'or unknown; only the native verifier establishes task success.'
        )
        return COMMON + prompt + " " + _HEIGHT_GUIDANCE + FEEDBACK
