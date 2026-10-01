"""gap — graph as policy.

Compile language instructions into typed, verified robot skill graphs and
execute them on simulators or real robots.

    import gap
    conn = gap.connector.sim("libero", task="libero_object/0")
    result = gap.execute(graph, conn)        # open-robot-skills auto-discovered
    g = gap.agent.generate("pick up the soup can")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.1.0.dev0"

# Lazy submodule re-exports keep `import gap` light (no torch/mujoco/jax at
# import time). Populated as subsystems land: runtime (P1), connector (P3),
# agent (P5), benchmark/viz (P6).
_LAZY_ATTRS = {
    "execute": "gap.runtime.execute",
    "connector": "gap.connector",
    "agent": "gap.agent",
    "benchmark": "gap.benchmark",
    "viz": "gap.viz",
    "builder": "gap.builder",
    "types": "gap.types",
    "errors": "gap.errors",
    "NodeContext": "gap.runtime.context",
    "CancelToken": "gap.runtime.context",
}

if TYPE_CHECKING:  # pragma: no cover
    pass


def __getattr__(name: str):
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module 'gap' has no attribute {name!r}")
    import importlib

    module = importlib.import_module(target)
    if target == f"gap.{name}":
        # Plain submodule re-export (gap.types, gap.builder, ...).
        value = module
    else:
        # Named attribute living in the target module (gap.execute is the
        # `execute` function inside gap.runtime.execute, etc.).
        value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS))
