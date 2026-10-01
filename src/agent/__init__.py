"""RobotUse agent roles and their factory."""
from .prime_agent import PrimeAgent
from .point_agent import PointAgent
from .grasp_agent import GraspAgent
from .place_agent import PlaceAgent
from .refiner_agent import RefinerAgent


_AGENT_TYPES = {agent.role: agent for agent in (
    PrimeAgent, PointAgent, GraspAgent, PlaceAgent, RefinerAgent,
)}


def create_agent(role):
    return _AGENT_TYPES[role]()


def run_role(orchestrator, role, task, parent_id=None):
    agent_type = _AGENT_TYPES.get(role)
    if agent_type is None:
        # Route custom role names through the shared agent loop.
        from .base_agent import run_agent_loop
        return run_agent_loop(orchestrator, role, task, parent_id)
    return agent_type().run(orchestrator, task, parent_id)
