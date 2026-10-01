"""Validate tool definitions before simulator, worker or LLM startup."""
from src.tools.discovery import discover_tools


def load_tool_registry():
    registry = discover_tools()
    _validate_role_inventory(registry)
    return registry


def _validate_role_inventory(registry):
    """Check every role/task branch before starting the episode.

    Reuse the actual inherited inventory rather than maintaining a second
    central list of tool names. No backend, provider or session is created.
    """
    from itertools import product
    from src.agent.playbook.loader import ROLES
    from src.backend.orchestrator import AgentOrchestrator

    probe = object.__new__(AgentOrchestrator)
    probe.tool_registry = registry
    flags = ('waypoint_task', 'waypoint_ref', 'placement_refinement',
             'inflight_refinement', 'place_refinement')
    for values in product((False, True), repeat=len(flags)):
        for stage in (None, 'pregrasp', 'release'):
            task = {**dict(zip(flags, values)), 'stage': stage}
            for role in ROLES:
                if role != 'common':
                    probe._tools(role, task)
