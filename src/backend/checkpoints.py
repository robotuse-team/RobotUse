"""Opt-in measured approach checkpoints and bounded fresh-geometry recovery."""
from copy import deepcopy
from src.backend.refinement_routes import AutoRefineIntentBackend
from src.backend.grasp_observation import ReobserveIntentOrchestrator
from src.tools.motion.planning import MotionPlanningError


class GraspCheckpointError(MotionPlanningError):
    pass


class CheckpointIntentBackend(AutoRefineIntentBackend):
    def _checkpoint(self, stage, diagnostics):
        # Scene relaxation never relaxes the measured arrival requirement.
        from src.tools.motion.tolerances import cartesian_tolerances
        position, orientation = cartesian_tolerances(getattr(self, 'connector', None))
        accepted = diagnostics['position_error_m'] <= position and diagnostics['orientation_error_rad'] <= orientation
        observation = self.observe()
        evidence = dict(diagnostics, stage=stage, accepted=accepted, position_tolerance_m=position,
            orientation_tolerance_rad=orientation)
        evidence['thresholds_enforced'] = True
        self._record('grasp_execution_checkpoint', {**evidence, 'observation': observation})
        # Kept for an optional paused refinement at this measured pose.
        self.last_checkpoint_observation = observation
        self.last_checkpoint_evidence = evidence
        if not accepted:
            self.last_grasp_checkpoint = evidence
            exc = GraspCheckpointError('Grasp stopped before closing: measured pose missed checkpoint')
            exc.evidence = dict(execution_stage=stage, motion_completed=False,
                execution_diagnostics=[diagnostics], checkpoint=evidence)
            raise exc

    def execute_grasp(self, candidate_ref, validation_ref):
        self.last_grasp_checkpoint = None
        self.validations[validation_ref][3].execution_checkpoint = self._checkpoint
        return super().execute_grasp(candidate_ref, validation_ref)

    def _execution_failure(self, operation, execution_ref, exc):
        result = super()._execution_failure(operation, execution_ref, exc)
        if isinstance(exc, GraspCheckpointError):
            result['reason_code'] = 'grasp_checkpoint_not_reached'
        return result


class CheckpointIntentOrchestrator(ReobserveIntentOrchestrator):
    def _skill_call(self, tool, args, scope, task, sid, owner):
        result = super()._skill_call(tool, args, scope, task, sid, owner)
        if tool == 'execute_grasp' and getattr(self.backend, 'last_grasp_checkpoint', None):
            result['checkpoint'] = deepcopy(self.backend.last_grasp_checkpoint)
            result['reason_code'] = 'grasp_checkpoint_not_reached'
            result['gripper_close_commanded'] = False
        return result

    def _intent_dispatch(self, role, tool, args, scope, task, sid):
        result = super()._intent_dispatch(role, tool, args, scope, task, sid)
        if role != 'prime' or tool != 'delegate_grasp': return result
        # Retry only this measured pre-close failure, never arbitrary failures or
        # a grasp that may already hold an object. All calls use normal budgets.
        for attempt in range(2):
            feedback = result.get('result', {})
            if feedback.get('reason_code') != 'grasp_checkpoint_not_reached': break
            self._event('grasp_checkpoint_recovery', sid, stage='reobserve', attempt=attempt+1,
                        feedback=feedback.get('checkpoint'))
            instruction = args['instruction'] + ' The previous approach stopped before closing because '
            instruction += 'the measured hand pose missed its checkpoint. Use the fresh current RGB-D '
            instruction += 'to reselect the same target for a new grasp; account for its current position.'
            selected = self._skill_call('delegate_point', {'instruction': instruction}, scope, task, sid, 'grasp')
            point = selected.get('result', {}).get('point_ref')
            if not point:
                result['recovery'] = selected
                break
            retry = {**args, 'point_ref': point, 'instruction': args['instruction'] +
                ' Previous approach missed the measured checkpoint: '+str(feedback.get('checkpoint'))+
                '. Evaluate new geometry from the actual current robot state. Choose a different useful '
                'pose/approach or request another view if the same approach remains unsuitable.'}
            result = super()._intent_dispatch(role, tool, retry, scope, task, sid)
        return result
