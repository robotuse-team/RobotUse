"""gap_core.rpc — out-of-process tool RPC.

Bundle tools whose SKILL.md declares ``gap.serving.protocol: stdio-msgpack``
run in their own venv (the bundle's per-pyproject .venv). gap-runtime spawns
one persistent server per bundle and calls into it via msgpack frames over
stdio (no shell, no port). The server-side process imports the bundle's
``tools.py`` (against the bundle's own deps) and dispatches by tool name.

Public surface:
    from gap_core.rpc.client import ToolClient
    from gap_core.rpc.server import main as tool_server_main
    from gap_core.rpc.codec import encode_frame, decode_frame
"""
