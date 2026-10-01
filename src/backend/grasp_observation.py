"""Opt-in: request useful new geometry instead of forcing an unsuitable grasp."""
from src.backend.skill_execution import SkillOwnedIntentOrchestrator


class ReobserveIntentOrchestrator(SkillOwnedIntentOrchestrator):
    def _finish(self, role, args, scope):
        if role == 'refiner' and args.get('status') == 'needs_observation':
            result = super()._finish(role, {**args, 'status': 'failed'}, scope)
            return {**result, 'status': 'needs_observation'}
        return super()._finish(role, args, scope)

    def _prompt(self, role):
        prompt = super()._prompt(role)
        if role in ('grasp', 'refiner'):
            prompt += (
                ' Judge whether the pose suits the object and support surface before route feasibility. '
                'Use a suitable candidate directly or make a small useful correction. If a candidate needs '
                'a fundamentally different approach, or visible geometry is insufficient to choose one, '
                'you may finish status=needs_observation with a short reason and recommendation such as '
                'a closer overhead view or a view of the occluded side. No coordinates or new fields needed. '
                'This returns control without executing the candidate. Another view is an option, not a '
                'mandatory step or guarantee of a better grasp. Scene relaxation is optional and does not '
                'make an unsuitable pose appropriate. A clearance threshold is not measured penetration; '
                'a few colliding points do not establish a harmless collision. Describe the depicted '
                'approach direction accurately; a view from above does not make a side approach top-down.')
        if role == 'prime':
            prompt += (
                ' If Grasp or Refiner returns needs_observation, consider asking Pointer for an observe '
                'waypoint that exposes the requested object surfaces. Move using the existing waypoint '
                'tools, then select the target in returned current RGB and request a new grasp. '
                'Changing instruction alone reuses the current candidate pool. No fixed reobserve sequence '
                'or required number of retries is imposed.')
        return prompt

    def _skill_call(self, tool, args, scope, task, sid, owner):
        if tool == 'delegate_refiner':
            original = args['instruction'].split(' Repair the selected pose/approach if useful.')[0]
            args = {**args, 'instruction': original + ' Assess the selected pose. Make a small useful '
                'correction, keep it if suitable, or return needs_observation with a helpful viewing '
                'direction when new geometry is preferable to a large pose change. Scene relaxation '
                'is optional and only useful if the pose remains appropriate.'}
        return super()._skill_call(tool, args, scope, task, sid, owner)
