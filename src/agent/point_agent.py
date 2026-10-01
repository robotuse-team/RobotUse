"""Select measured targets, destinations and observation waypoints."""
from .base_agent import BaseAgent
from src.agent.intent_prompts import intent_prompt


class PointAgent(BaseAgent):
    role = 'point'

    def base_prompt(self, context):
        return (
            intent_prompt(self.role) + (
            ' RobotUse supports observe waypoints while retaining a measured transport grasp. '
            'Select current SAM regions and assess returned overlays; use fresh references after motion.')
        )
