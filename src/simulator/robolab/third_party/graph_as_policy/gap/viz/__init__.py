"""gap.viz — web-based trace visualization + static graph rendering.

Three entry points:

- :func:`serve` — start the interactive web viewer over a trace/output
  directory (FastAPI + the bundled React frontend).
- :func:`render` — render a v3 ``workflow.json`` to a paper-ready
  matplotlib PDF/PNG (lazy re-export from :mod:`gap.viz.render`; importing
  ``gap.viz`` does not pull in matplotlib).
- :func:`to_text` — render a v3 ``workflow.json`` as Unicode box-drawing
  terminal text (pure stdlib; what ``print(graph)`` and ``gap generate``
  show).
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["serve", "render", "to_text", "create_app"]


def serve(
    root: str | Path = "outputs",
    *,
    port: int = 9432,
    services: str | Path | None = None,
    host: str = "127.0.0.1",
    open_browser: bool = False,
) -> None:
    """Serve the interactive workflow visualizer over *root*.

    Args:
        root: Output directory, scanned recursively for trials
            (directories containing ``dag_trace.json`` or ``workflow.json``).
        port: HTTP port to bind (default 9432).
        services: Optional open-robot-skills checkout path; its bundles register
            tools so node tooltips show port schemas.
        host: Bind address (default loopback).
        open_browser: Open the system browser once the server is up.
    """
    import uvicorn

    from .server import create_app

    root_dir = Path(root).resolve()
    app = create_app(
        root_dir=root_dir,
        skills=Path(services) if services else None,
    )
    if open_browser:
        import threading
        import webbrowser
        threading.Timer(
            1.0, lambda: webbrowser.open(f"http://{host}:{port}")
        ).start()
    uvicorn.run(app, host=host, port=port, log_level="info")


def create_app(*args, **kwargs):
    """Thin lazy re-export of :func:`gap.viz.server.create_app`."""
    from .server import create_app as _create_app
    return _create_app(*args, **kwargs)


def __getattr__(name: str):
    # Lazy re-export: `from gap.viz import render` without importing
    # matplotlib at package-import time.
    if name == "render":
        from .render import render as _render
        globals()["render"] = _render
        return _render
    if name == "to_text":
        from .text import to_text as _to_text
        globals()["to_text"] = _to_text
        return _to_text
    raise AttributeError(f"module 'gap.viz' has no attribute {name!r}")
