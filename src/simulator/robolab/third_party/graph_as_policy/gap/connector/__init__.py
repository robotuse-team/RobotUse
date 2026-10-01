"""gap.connector — backends that own an environment and its tools.

A connector binds an environment instance (sim or real) to the runtime:
it registers the ``robot.*`` / ``sim.*`` tools on a fresh ToolRegistry,
assembles :class:`gap.types.Observation` snapshots, and exposes
capabilities + ground-truth world snapshots where the backend supports
them.

    import gap
    conn = gap.connector.sim("libero", task="libero_object/0")
    result = gap.execute(graph, conn)   # open-robot-skills auto-discovered
"""

from __future__ import annotations

from gap.connector.collector import DataCollector
from gap.connector.core import Capabilities, Connector
from gap.connector.real import RealConnector, real
from gap.connector.sim import SimConnector, sim

__all__ = [
    "Capabilities",
    "Connector",
    "DataCollector",
    "RealConnector",
    "SimConnector",
    "real",
    "sim",
]
