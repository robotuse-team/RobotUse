"""Access a compiled model through connector wrappers."""
from typing import Any


class LiveSelfCoreError(RuntimeError):
    """Raised when the live model cannot yield a valid robot-only snapshot."""


def find_model_and_data(connector: Any, depth: int = 6):
    """Walk the connector's wrappers to the compiled model and its data."""

    seen, stack = set(), [(connector, 0)]
    while stack:
        node, level = stack.pop()
        if level > depth or id(node) in seen:
            continue
        seen.add(id(node))
        sim = getattr(node, "sim", None)
        if sim is not None and hasattr(getattr(sim, "model", None), "ngeom"):
            return sim.model, sim.data
        for attr in ("unwrapped", "env", "handle", "_env"):
            child = getattr(node, attr, None)
            if child is not None:
                stack.append((child, level + 1))
    raise LiveSelfCoreError("no compiled MuJoCo model reachable from the connector")
