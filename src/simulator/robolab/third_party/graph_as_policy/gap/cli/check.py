"""``gap check`` — capability report: what can run *here*?

The SkyPilot-``sky check`` of gap. Reports the environment (GPU, LLM
provider credentials), every active registry, per-bundle operational
status (deps importable, declared ``gap.requires`` met, weights cached),
and the per-skill runnability rollup, with a fix hint per failure.

Diagnostic by design: exit 0 whenever a report is produced (a laptop
without a GPU is not an error); ``--strict`` flips any not-ready bundle
into exit 1 for CI gating; usage/resolution errors exit 2.
"""

from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "check",
        help="Capability report: which tool bundles are operational here, "
             "and which skills are therefore runnable",
    )
    p.add_argument(
        "--skills", action="append", default=None, metavar="PATH",
        help="Registry root(s); repeatable. Overrides $GAP_SKILLS_PATH and "
             "configured registries (default: resolve --skills > "
             "$GAP_SKILLS_PATH > project [tool.gap] > user config > "
             "auto-discovered open-robot-skills sibling)",
    )
    p.add_argument(
        "--registry", default=None, metavar="NAME",
        help="Restrict the report to one active registry",
    )
    p.add_argument(
        "--format", default="pretty", choices=["pretty", "json"],
        help="Output format (json is a stable machine-readable schema)",
    )
    p.add_argument(
        "--strict", action="store_true",
        help="Exit 1 when any non-shadowed bundle is not ready (CI gating)",
    )
    p.add_argument(
        "--probe", action="store_true",
        help="Issue a 1-token API ping to each configured LLM and VLM "
             "provider (default is a static env-var/ADC presence check). "
             "Use this to catch stale creds and wrong model names without "
             "running a full job — a VLM bundle silently falling through "
             "to the openrouter default with no API key would surface here.",
    )
    p.set_defaults(func=_handle_check)


_STATUS_LABEL = {
    "ready": "READY",
    "not-ready": "NOT READY",
    "shadowed": "SHADOWED",
    "blocked": "BLOCKED",
    "ok": "OK",
    "missing": "MISSING",
    "unknown": "unknown",
    "error": "ERROR",
}


def _handle_check(args: argparse.Namespace) -> int:
    from gap.skills import resolve_registries
    from gap.skills.capability import build_check_report
    from gap.skills.registries import RegistrySet

    try:
        registry_set = resolve_registries(args.skills)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 2

    if args.registry is not None:
        try:
            registry_set = RegistrySet([registry_set.get(args.registry)])
        except KeyError as exc:
            print(f"error: {exc.args[0]}")
            return 2

    report = build_check_report(registry_set, probe=args.probe)

    if args.format == "json":
        import json

        print(json.dumps(report.to_json_dict(), indent=2))
    else:
        _print_pretty(report)

    if args.strict:
        not_ready = [
            b for b in report.bundles
            if not b.shadowed_by and b.status == "not-ready"
        ]
        if not_ready or not report.registries:
            return 1
    return 0


def _print_pretty(report) -> None:
    env = report.environment
    print("environment")
    print(f"  python {env.python_version} · gap {env.gap_version}")
    gpu = env.gpu
    line = f"  gpu: {_STATUS_LABEL[gpu.status]}"
    if gpu.detail:
        line += f" — {gpu.detail}"
    if gpu.fix_hint and not gpu.ok:
        line += f"  ({gpu.fix_hint})"
    print(line)
    providers = " · ".join(
        f"{name} {_STATUS_LABEL[probe.status]}"
        + (f" ({probe.fix_hint})" if probe.fix_hint and not probe.ok else "")
        + (
            # On --probe failures, surface the actual error inline so the
            # user doesn't need to re-run with --format json to see it.
            f" — {probe.detail}" if not probe.ok and probe.detail
            and probe.detail.startswith("ping ") else ""
        )
        for name, probe in env.llm_providers.items()
    )
    print(f"  llm: {providers}")
    # VLM dispatches to ONE resolved provider per run (see resolve_vlm_env);
    # show that single line with the resolution detail so a vlm bundle
    # silently falling through to the openrouter default in a vertex shell
    # is loud here, not at "perceive selected box 0" runtime.
    for name, probe in env.vlm_provider.items():
        line = f"  vlm: {name} {_STATUS_LABEL[probe.status]}"
        if probe.detail:
            line += f" — {probe.detail}"
        if probe.fix_hint and not probe.ok:
            line += f"  ({probe.fix_hint})"
        print(line)

    if not report.registries:
        print(
            "\nno skill registries active — clone open-robot-skills next to "
            "this checkout, `gap registry add <name> <path>`, or pass --skills"
        )
        return

    print(f"\nregistries ({len(report.registries)} active)")
    for i, spec in enumerate(report.registries, 1):
        print(f"  {i}. {spec.name}  {spec.path}  [{spec.source}]")

    for spec in report.registries:
        rows = [b for b in report.bundles if b.registry == spec.name]
        if not rows:
            continue
        print(f"\n[registry {spec.name}]")
        for b in sorted(rows, key=lambda r: (r.kind, r.name)):
            label = _STATUS_LABEL[b.status]
            if b.shadowed_by:
                print(f"  [{b.kind}] {b.name}: {label} by registry "
                      f"{b.shadowed_by!r}")
                continue
            print(f"  [{b.kind}] {b.name}: {label}")
            if not b.deps.ok:
                line = f"      deps: {b.deps.detail or b.deps.status}"
                if b.deps.fix_hint:
                    line += f" — {b.deps.fix_hint}"
                print(line)
            for req_label, probe in b.requirements:
                if probe.ok:
                    continue
                line = f"      {req_label}: {probe.detail or probe.status}"
                if probe.fix_hint:
                    line += f" — {probe.fix_hint}"
                print(line)
            if b.weights.status in ("missing", "unknown", "error"):
                line = f"      weights: {b.weights.detail or b.weights.status}"
                if b.weights.fix_hint:
                    line += f" — {b.weights.fix_hint}"
                print(line)

        skill_rows = [s for s in report.skills if s.registry == spec.name]
        for s in skill_rows:
            if s.status == "ready":
                continue
            reasons = []
            if s.blocked_by:
                reasons.append("blocked by: " + ", ".join(s.blocked_by))
            if not s.self_ready:
                reasons.append("own deps/requirements not ready")
            if s.unknown_tools:
                reasons.append("unknown tools: " + ", ".join(s.unknown_tools))
            print(f"  [skill] {s.name}: BLOCKED — {'; '.join(reasons)}")

    active = [b for b in report.bundles if not b.shadowed_by]
    n_ready = sum(1 for b in active if b.status == "ready")
    n_not = sum(1 for b in active if b.status == "not-ready")
    n_shadowed = len(report.bundles) - len(active)
    skills_ready = sum(1 for s in report.skills if s.status == "ready")
    summary = (
        f"\n{len(active)} bundle(s): {n_ready} ready, {n_not} not ready"
    )
    if n_shadowed:
        summary += f", {n_shadowed} shadowed"
    summary += (
        f" · {len(report.skills)} skill(s): {skills_ready} ready, "
        f"{len(report.skills) - skills_ready} blocked"
    )
    print(summary)
