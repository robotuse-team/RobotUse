"""Unified ``gap`` CLI — single entry point for all gap commands.

Subcommands are registered lazily so that heavy imports (jax, mujoco, …)
only happen when the subcommand is actually invoked.
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    """The fully-registered ``gap`` argument parser.

    Exposed for tooling (the agent-skill CLI-reference generator and the
    sync tests under ``tests/agent_skill/``) — registration is lazy and
    light, so building the parser never imports heavy deps.
    """
    parser = argparse.ArgumentParser(
        prog="gap",
        description="gap — graph as policy: typed, verified robot skill graphs",
    )
    sub = parser.add_subparsers(dest="command")

    # Import each subcommand module and let it register its parser.
    # These modules must NOT import heavy deps at module level.
    from .benchmark import register as _reg_benchmark
    from .check import register as _reg_check
    from .generate import register as _reg_generate
    from .policy import register as _reg_policy
    from .registry import register as _reg_registry
    from .run import register as _reg_run
    from .skills import register as _reg_skills
    from .tools import register as _reg_tools
    from .trace_diff import register as _reg_trace_diff
    from .viz import register as _reg_viz

    _reg_run(sub)
    _reg_check(sub)
    _reg_skills(sub)
    _reg_tools(sub)
    _reg_registry(sub)
    _reg_generate(sub)
    _reg_viz(sub)
    _reg_trace_diff(sub)
    _reg_benchmark(sub)
    _reg_policy(sub)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if hasattr(args, "func"):
        result = args.func(args)
        if isinstance(result, int):
            sys.exit(result)
    else:
        parser.print_help()
        sys.exit(1)
