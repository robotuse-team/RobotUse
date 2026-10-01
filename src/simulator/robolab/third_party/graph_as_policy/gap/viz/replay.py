"""Re-execute tool nodes using saved request data.

The proto era replayed nodes by reissuing a serialized ``request.bin``
through a gRPC channel; the de-proto'd trace records each tool request as
plain JSON (``request.json``) plus a ``request.meta.json`` carrying
``{"tool": <flat name>}``. Replay dispatches the saved inputs through a
:class:`gap.tools.ToolRegistry`.

Replay is best-effort: large arrays are summarized (not round-tripped) by
the trace serializer, so tools whose inputs were summarized will receive
placeholder strings.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def replay_node(
    trial_dir: str | Path,
    node_id: str,
    tool_registry: Any,
) -> dict:
    """Re-execute a tool node by reissuing its saved request.

    Args:
        trial_dir: Path to the trial output directory.
        node_id: The node ID to replay (``node_data/<node_id>`` must carry
            a recorded request).
        tool_registry: A :class:`gap.tools.ToolRegistry` to dispatch through.

    Returns:
        ``{"ok": bool, "tool": ..., "response" | "error": ...}``.
    """
    trial_dir = Path(trial_dir)
    node_dir = trial_dir / "node_data" / node_id

    # Load request metadata
    meta_path = node_dir / "request.meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No request metadata for node '{node_id}' — "
            "only tool nodes with saved requests can be replayed"
        )

    with open(meta_path) as f:
        meta = json.load(f)
    tool = meta["tool"]

    request_path = node_dir / "request.json"
    if not request_path.exists():
        raise FileNotFoundError(f"No request.json for node '{node_id}'")
    with open(request_path) as f:
        request = json.load(f) or {}
    if not isinstance(request, dict):
        raise ValueError(
            f"request.json for node '{node_id}' is not an object"
        )

    try:
        response = tool_registry.invoke(tool, **request)
        from gap.runtime.tracing import _json_default, _serialize_value
        response_json = json.loads(
            json.dumps(_serialize_value(response), default=_json_default)
        )
        return {
            "ok": True,
            "tool": tool,
            "response": response_json,
        }
    except Exception as e:
        return {
            "ok": False,
            "tool": tool,
            "error": str(e),
        }
