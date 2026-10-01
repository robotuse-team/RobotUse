"""``gap tools`` subcommands — list / show the merged tool catalog.

The flat tool namespace a graph may dispatch on: connector-owned
``robot.*`` / ``sim.*`` tools (registered by the live connector at run
time), every registry bundle's ``@tool`` functions, and the codegen-scope
meta-tools. Importable bundles are live-introspected (full input/output
schemas); bundles whose optional deps are not installed degrade to
static rows from their SKILL.md ``gap.tools`` declarations, marked with
an install hint. ``--static`` skips bundle imports entirely.
"""

from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "tools",
        help="Inspect the merged tool catalog (connector + registry bundles)",
    )
    sub = sp.add_subparsers(dest="tools_command")

    lp = sub.add_parser("list", help="List every tool in the active registries")
    _add_common(lp)
    lp.add_argument(
        "--format", default="pretty", choices=["pretty", "markdown", "json"],
        help="Output format",
    )
    lp.add_argument("--tag", default=None, help="Only tools carrying this tag")
    lp.add_argument(
        "--scope", default=None, choices=["runtime", "codegen"],
        help="Only tools of this scope (runtime = dispatchable from graphs)",
    )
    lp.add_argument(
        "--static", action="store_true",
        help="Never import bundles — catalog from SKILL.md declarations only",
    )
    lp.set_defaults(func=_handle_list)

    shp = sub.add_parser("show", help="Full schema and runnability for one tool")
    shp.add_argument("name", help="Tool name, e.g. geometry.compute_obb")
    _add_common(shp)
    shp.add_argument(
        "--format", default="pretty", choices=["pretty", "json"],
        help="Output format",
    )
    shp.set_defaults(func=_handle_show)

    # NOTE: no sp.set_defaults(func=...) — `gap tools` with no subcommand
    # falls through to the top-level help in main() (mirrors `gap skills`).


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--skills", action="append", default=None, metavar="PATH",
        help="Registry root(s); repeatable (default: resolved registries)",
    )
    parser.add_argument(
        "--registry", default=None, metavar="NAME",
        help="Restrict to one active registry",
    )


# ---------------------------------------------------------------------------
# Catalog assembly
# ---------------------------------------------------------------------------


def _resolve_set(args: argparse.Namespace):
    from gap.skills import resolve_registries
    from gap.skills.registries import RegistrySet

    registry_set = resolve_registries(args.skills)
    if args.registry is not None:
        registry_set = RegistrySet([registry_set.get(args.registry)])
    return registry_set


def _gather(registry_set, *, static: bool) -> list[dict]:
    """One row per tool across connector + bundles (+ meta when live)."""
    from gap.agent._catalog import (
        build_codegen_tool_registry,
        connector_tool_descriptors,
    )
    from gap.skills.registries import load_registry_set, registry_dist_name
    from gap.skills.validate import load_checkout_extras, validate_checkout

    rows: list[dict] = []

    def first_sentence(text: str) -> str:
        import re

        text = " ".join(str(text).split())
        m = re.search(r"(?<=[.!?])\s", text)
        return text[: m.start()] if m else text

    def desc_row(desc, *, bundle: str, registry: str, status: str) -> dict:
        return {
            "name": desc.name,
            "summary": first_sentence(desc.summary),
            "scope": desc.scope,
            "tags": sorted(desc.tags),
            "bundle": bundle,
            "registry": registry,
            "status": status,
            "inputs": {
                f.name: f.type_str for f in desc.schema.inputs.values()
            },
            "outputs": {
                f.name: f.type_str for f in desc.schema.outputs.values()
            },
        }

    # Live path: import every importable bundle, then read the assembled
    # registry (connector descriptors + drained @tools + meta-tools).
    loaded = None
    if not static:
        loaded = load_registry_set(registry_set)
        tool_reg = build_codegen_tool_registry()
        connector_names = set(connector_tool_descriptors())
        for name in sorted(tool_reg._tools):
            desc = tool_reg._tools[name]
            module = str(desc.metadata.get("module", ""))
            if name in connector_names:
                rows.append(desc_row(
                    desc, bundle="connector", registry="-",
                    status="connector (registered at run time)",
                ))
            elif module.startswith("gap_skills."):
                bundle = module.split(".")[2]
                registry = (
                    loaded.get(bundle).registry if bundle in loaded else "?"
                )
                rows.append(desc_row(
                    desc, bundle=bundle, registry=registry or "-", status="ok",
                ))
            else:
                rows.append(desc_row(
                    desc, bundle="core", registry="-", status="codegen meta-tool",
                ))
    else:
        for _, desc in sorted(connector_tool_descriptors().items()):
            rows.append(desc_row(
                desc, bundle="connector", registry="-",
                status="connector (registered at run time)",
            ))

    # Static rows: every declared gap.tools entry whose bundle did not make
    # it into the live registry (deps missing / --static). Shadowed bundles
    # are inert duplicates — skipped.
    claimed: set[str] = set()
    for spec in registry_set:
        extras = load_checkout_extras(spec.path) or {}
        dist = spec.dist_name or registry_dist_name(spec.path)
        for report in validate_checkout(spec.path):
            if report.name in claimed:
                continue
            claimed.add(report.name)
            if report.meta is None or not report.meta.tools:
                continue
            if loaded is not None and report.name in loaded:
                continue  # live rows already cover it
            from gap.skills.capability import dep_fix_hint

            hint = dep_fix_hint(
                spec.path, dist_name=dist, bundle=report.name,
                has_extra=report.name in extras,
            )
            status = "static (--static)" if static else (
                f"static (deps not installed — {hint})"
            )
            for tool_name, summary in sorted(report.meta.tools.items()):
                rows.append({
                    "name": tool_name,
                    "summary": first_sentence(summary),
                    "scope": "runtime",
                    "tags": sorted(report.meta.tags),
                    "bundle": report.name,
                    "registry": spec.name,
                    "status": status,
                    "inputs": None,
                    "outputs": None,
                })

    rows.sort(key=lambda r: (r["bundle"], r["name"]))
    return rows


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def _handle_list(args: argparse.Namespace) -> int:
    try:
        registry_set = _resolve_set(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    except KeyError as exc:
        print(f"error: {exc.args[0]}")
        return 2

    rows = _gather(registry_set, static=args.static)
    if args.tag:
        rows = [r for r in rows if args.tag in r["tags"]]
    if args.scope:
        rows = [r for r in rows if r["scope"] == args.scope]
    if not rows:
        print("no tools matched")
        return 1

    if args.format == "json":
        import json

        print(json.dumps(rows, indent=2))
        return 0

    headers = ("NAME", "SUMMARY", "SCOPE", "TAGS", "BUNDLE", "REGISTRY", "STATUS")
    cells = []
    for r in rows:
        summary = r["summary"]
        if args.format == "pretty" and len(summary) > 58:
            summary = summary[:55] + "..."
        cells.append((
            r["name"], summary, r["scope"], ",".join(r["tags"]) or "-",
            r["bundle"], r["registry"], r["status"],
        ))

    if args.format == "markdown":
        print("| " + " | ".join(headers) + " |")
        print("|" + "|".join("---" for _ in headers) + "|")
        for row in cells:
            print("| " + " | ".join(
                f"`{row[0]}`" if i == 0 else str(c)
                for i, c in enumerate(row)
            ) + " |")
        return 0

    widths = [
        max(len(headers[c]), *(len(str(row[c])) for row in cells))
        for c in range(len(headers))
    ]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for row in cells:
        print(fmt.format(*row))
    return 0


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


def _handle_show(args: argparse.Namespace) -> int:
    try:
        registry_set = _resolve_set(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    except KeyError as exc:
        print(f"error: {exc.args[0]}")
        return 2

    from gap.agent._catalog import build_codegen_tool_registry
    from gap.skills.registries import load_registry_set

    loaded = load_registry_set(registry_set)
    tool_reg = build_codegen_tool_registry()

    desc = tool_reg._tools.get(args.name)
    rows = _gather(registry_set, static=False)
    row = next((r for r in rows if r["name"] == args.name), None)

    if row is None:
        import difflib

        close = difflib.get_close_matches(
            args.name, [r["name"] for r in rows], n=3,
        )
        print(f"error: unknown tool {args.name!r}")
        if close:
            print("did you mean: " + ", ".join(close))
        return 1

    runnability = _bundle_runnability(row["bundle"], registry_set, loaded)

    if args.format == "json":
        import json

        payload = dict(row)
        if desc is not None:
            payload["inputs"] = {
                f.name: {
                    "type": f.type_str,
                    "required": f.required,
                    "default": None if f.required else f.default,
                    "description": f.description,
                }
                for f in desc.schema.inputs.values()
            }
            payload["outputs"] = {
                f.name: {"type": f.type_str, "description": f.description}
                for f in desc.schema.outputs.values()
            }
        payload["runnability"] = runnability
        print(json.dumps(payload, indent=2, default=str))
        return 0

    print(f"{row['name']} — {row['summary']}")
    print(f"  scope: {row['scope']} · tags: {','.join(row['tags']) or '-'}")
    print(f"  bundle: {row['bundle']} · registry: {row['registry']}")
    print(f"  status: {row['status']}")
    print(f"  runnability: {runnability}")
    if desc is not None:
        if desc.schema.inputs:
            print("\n  inputs:")
            for f in desc.schema.inputs.values():
                req = "required" if f.required else f"default={f.default!r}"
                line = f"    {f.name}: {f.type_str}  ({req})"
                if f.description:
                    line += f" — {f.description}"
                print(line)
        if desc.schema.outputs:
            print("\n  outputs:")
            for f in desc.schema.outputs.values():
                line = f"    {f.name}: {f.type_str}"
                if f.description:
                    line += f" — {f.description}"
                print(line)
    else:
        print(
            "\n  (schemas unavailable — bundle deps not installed; the "
            "summary above comes from SKILL.md)"
        )
    return 0


def _bundle_runnability(bundle: str, registry_set, loaded) -> str:
    """One line: can the owning bundle run here?"""
    if bundle == "connector":
        return "satisfied by the live connector at run time"
    if bundle == "core":
        return "always available (gap core)"
    from gap.skills.capability import _probe_requirements, probe_gpu

    if bundle not in loaded:
        return "not-ready (deps not importable — see status)"
    meta = loaded.get(bundle).meta
    failures = [
        f"{label}: {probe.detail or probe.status}"
        for label, probe in _probe_requirements(meta, gpu=probe_gpu())
        if not probe.ok
    ]
    if failures:
        return "not-ready — " + "; ".join(failures)
    return "ready"
