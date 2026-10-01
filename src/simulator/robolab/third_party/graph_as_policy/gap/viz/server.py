"""FastAPI application factory and static file serving."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import api


def frontend_dist_dir() -> Path:
    """Locate the built frontend bundle.

    The ``frontend/dist`` directory is checked into the repo and shipped
    as package data in the wheel (see the hatch force-include entry in
    pyproject.toml), so a path-relative lookup works both from a checkout
    and an installed package.
    """
    return Path(__file__).parent / "frontend" / "dist"


def create_app(
    root_dir: Path | None = None,
    skills: Path | None = None,
) -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(title="gap Workflow Visualizer")

    # CORS for dev mode (Vite dev server on different port).
    # Restrict to localhost origins so that — if a user ever runs this with
    # --host 0.0.0.0 — a malicious webpage cannot silently hit the API.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Configure API with paths
    api.configure(root_dir=root_dir, skills=skills)

    # Include API routes
    app.include_router(api.router)

    # Serve frontend static files
    frontend_dist = frontend_dist_dir()
    if frontend_dist.exists():
        # Serve index.html for the root and all non-API routes (SPA)
        @app.get("/")
        async def index():
            from fastapi.responses import FileResponse
            return FileResponse(frontend_dist / "index.html")

        app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

        # Catch-all for SPA client-side routing
        @app.get("/{full_path:path}")
        async def spa_fallback(full_path: str):
            from fastapi.responses import FileResponse
            # Don't intercept API routes
            if full_path.startswith("api/"):
                from fastapi import HTTPException
                raise HTTPException(404)
            file_path = frontend_dist / full_path
            if file_path.exists() and file_path.is_file():
                return FileResponse(file_path)
            return FileResponse(frontend_dist / "index.html")

    return app
