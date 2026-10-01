"""``gap viz`` subcommand — interactive web-based workflow visualizer."""

from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "viz",
        help="Interactive web-based workflow visualizer",
    )
    sp.add_argument(
        "--root",
        default="outputs",
        help="Root output directory (scanned recursively for trials, default: outputs)",
    )
    sp.add_argument(
        "--services",
        default=None,
        help="Optional open-robot-skills checkout; its bundles register tools "
             "used to enrich node tooltips with port schemas.",
    )
    sp.add_argument(
        "--port", type=int, default=9432,
        help="Port to serve on (default: 9432)",
    )
    sp.add_argument(
        "--host", default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    sp.add_argument(
        "--no-browser", action="store_true",
        help="Don't open browser automatically",
    )
    sp.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )
    sp.set_defaults(func=_handle)


def _handle(args: argparse.Namespace) -> int:
    import logging
    import sys
    from pathlib import Path

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    root_dir = Path(args.root).resolve()
    if not root_dir.exists():
        print(f"Error: Directory not found: {root_dir}", file=sys.stderr)
        return 1

    skills = Path(args.services) if args.services else None

    from gap.viz.trial_loader import discover_trials
    trials = discover_trials(root_dir)

    from gap.viz.server import create_app
    app = create_app(root_dir=root_dir, skills=skills)

    url = f"http://{args.host}:{args.port}"
    print("gap Visualizer")
    print(f"  URL: {url}")
    print(f"  Root: {root_dir}")
    print(f"  Trials found: {len(trials)}")
    for t in trials[:10]:
        print(f"    - {t}")
    if len(trials) > 10:
        print(f"    ... and {len(trials) - 10} more")
    if skills:
        print(f"  Skills: {skills} (tool schemas enabled)")

    if not args.no_browser:
        import threading
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0
