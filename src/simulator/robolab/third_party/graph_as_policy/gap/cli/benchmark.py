"""``gap benchmark`` subcommand — run a benchmark grid / acceptance gate."""

from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "benchmark",
        help="Run a benchmark sweep (families x variations x modes) or an "
             "acceptance gate",
    )
    sp.add_argument(
        "config",
        help="Path to a benchmark YAML (a 'benchmark:' grid block, or a "
             "plain pipeline config with 'suites:' for suites mode)",
    )
    sp.add_argument(
        "--gate", action="store_true",
        help="Exit nonzero when the overall success rate is below the "
             "config's gate_threshold (default 0.90), when any cell "
             "errored, or when no trial ran",
    )
    sp.add_argument(
        "--resume", action="store_true",
        help="Reuse the latest run dir and skip cells whose results "
             "already exist; the merged summary is rebuilt",
    )
    sp.add_argument(
        "--families", nargs="*", default=None, metavar="FAMILY",
        help="Restrict the grid to these families (grid mode only)",
    )
    sp.add_argument(
        "--modes", nargs="*", default=None, metavar="MODE",
        help="Restrict the grid to these modes (grid mode only)",
    )
    sp.add_argument(
        "--output-dir", default=None, metavar="DIR",
        help="Override the config's output directory",
    )
    sp.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )
    sp.set_defaults(func=_handle)


def _handle(args: argparse.Namespace) -> int:
    import logging
    from pathlib import Path

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    import gap.benchmark as bench

    try:
        cfg = bench.BenchmarkConfig.from_yaml(args.config)
    except Exception as exc:
        print(f"FAIL: could not load {args.config}: {exc}")
        return 2

    if args.families:
        if cfg.suites_mode:
            print("error: --families has no effect on a suites-mode config")
            return 2
        cfg.families = list(args.families)
        cfg.__post_init__()  # re-validate
    if args.modes:
        if cfg.suites_mode:
            print("error: --modes has no effect on a suites-mode config")
            return 2
        cfg.modes = list(args.modes)
        cfg.__post_init__()
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir).resolve()

    try:
        summary = bench.run(cfg, gate=args.gate, resume=args.resume)
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1

    print(
        f"benchmark complete: {summary.n_success}/{summary.n_trials} trials "
        f"(success_rate={summary.success_rate:.4f}, "
        f"completion_rate={summary.completion_rate:.4f})"
    )
    if summary.avg_physical_execution_s > 0:
        print(
            f"avg physical execution: {summary.avg_physical_execution_s:.2f} s/trial"
        )
    if summary.run_dir is not None:
        print(f"summary: {summary.run_dir / 'summary.tsv'}")
    if args.gate:
        verdict = "PASS" if summary.ok else "FAIL"
        print(
            f"gate {verdict}: success_rate={summary.success_rate:.4f} "
            f"threshold={summary.gate_threshold:.4f}"
        )
        return 0 if summary.ok else 1
    return 0
