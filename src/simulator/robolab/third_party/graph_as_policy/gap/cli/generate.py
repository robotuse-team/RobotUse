"""``gap generate`` subcommand — LLM graph generation from an instruction."""

from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "generate",
        help="Generate a workflow graph from a natural-language instruction",
    )
    sp.add_argument(
        "instruction",
        help='The task, e.g. "pick up the alphabet soup and put it in the basket"',
    )
    sp.add_argument(
        "--skills", action="append", default=None, metavar="PATH",
        help="Skill registry root(s); repeatable, precedence-ordered. "
             "Default: the resolved registry set — $GAP_SKILLS_PATH, "
             "project [tool.gap], user config, or an open-robot-skills "
             "checkout next to the graph-as-policy checkout; "
             "--config skills: also works",
    )
    sp.add_argument(
        "--provider", default=None, choices=["openrouter", "vertex"],
        help="LLM provider override (default: openrouter)",
    )
    sp.add_argument(
        "--model", default=None,
        help="LLM model override (default: the provider default)",
    )
    sp.add_argument(
        "--out", default=None, metavar="DIR",
        help="Output directory (default: outputs/generated_<timestamp>)",
    )
    sp.add_argument(
        "--config", default=None, metavar="YAML",
        help="Optional pipeline config YAML (llm/composition/skills knobs)",
    )
    sp.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )
    sp.set_defaults(func=_handle)


def _handle(args: argparse.Namespace) -> int:
    import logging
    import os
    import sys

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from gap.agent import PipelineConfig, generate_sync
    from gap.skills import resolve_registries
    from gap.viz.text import to_text

    config = None
    skills = args.skills
    if args.config:
        config = PipelineConfig.from_yaml(args.config)
        if skills is None and config.skills is not None:
            skills = config.skills
    if skills is None:
        try:
            skills = resolve_registries(required=True).paths()
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}")
            return 2

    try:
        graph = generate_sync(
            args.instruction,
            skills=skills,
            model=args.model,
            provider=args.provider,
            out_dir=args.out,
            config=config,
        )
    except Exception as exc:
        print(f"FAIL: {exc}")
        low = str(exc).lower()
        if any(s in low for s in (
            "could not resolve authentication", "401", "unauthorized",
            "api key", "api_key", "credential",
        )):
            provider = args.provider or (config.llm.provider if config else None)
            if provider is None:
                import os as _os

                provider = _os.environ.get("GAP_LLM_PROVIDER", "openrouter")
            print(
                f"\nhint: no LLM credentials for provider {provider!r} "
                f"(openrouter is the default).\n"
                "  openrouter:  export OPENROUTER_API_KEY=...\n"
                "  vertex:      gcloud auth application-default login\n"
                "               export GOOGLE_CLOUD_PROJECT=<project> \\\n"
                "                      GAP_LLM_PROVIDER=vertex GAP_LLM_MODEL=<gemini-model>\n"
                "               (install the SDK per run: uv run --extra vertex gap generate ...)\n"
                "`gap check` shows which providers are configured."
            )
        return 1

    n_subgraphs = len(graph.workflow.get("subgraphs", {}))
    print(f"OK: wrote {graph.path} ({n_subgraphs} subgraph(s), {len(graph.code)} generated file(s))")
    print()
    print(to_text(graph.workflow,
                  color=sys.stdout.isatty() and not os.environ.get("NO_COLOR")))
    print()
    print(f"run it with: gap run {graph.path}")
    return 0
