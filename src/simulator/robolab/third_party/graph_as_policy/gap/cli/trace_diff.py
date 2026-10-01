"""``gap trace-diff`` subcommand — align two DAG traces by node name.

Usage::

    gap trace-diff outputs/rehearsal/trace outputs/trial_01/trace

Both inputs may be either directories containing ``dag_trace.json`` or
the trace files themselves. Reports per-node verdict agreement and a
roll-up rate, with an optional ``--out`` JSON dump for downstream tools.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "trace-diff",
        help="Align two dag_trace.json files by node name and report "
             "per-node and aggregate agreement.",
    )
    sp.add_argument(
        "trace_a", type=Path,
        help="First trace directory (or its dag_trace.json) — e.g. the "
             "rehearsal / reference run.",
    )
    sp.add_argument(
        "trace_b", type=Path,
        help="Second trace directory (or its dag_trace.json) — e.g. the "
             "real run.",
    )
    sp.add_argument(
        "--out", type=Path, default=None,
        help="If set, write the structured diff as JSON to this path.",
    )
    sp.add_argument(
        "--quiet", action="store_true", default=False,
        help="Suppress the human-readable summary; only write --out if set.",
    )
    sp.set_defaults(func=_handle)


def _handle(args: argparse.Namespace) -> int:
    from gap.runtime.trace_diff import diff_trace_dirs, format_summary, to_json_dict

    diff = diff_trace_dirs(args.trace_a, args.trace_b)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(to_json_dict(diff), indent=2))
    if not args.quiet:
        print(format_summary(diff), file=sys.stdout)
    # Exit 0 when at least one matched node and verdict agreement == 1.0,
    # 1 otherwise. This makes the command useful in CI.
    if diff.matched_count == 0:
        return 2
    return 0 if diff.verdict_agreement_rate == 1.0 else 1
