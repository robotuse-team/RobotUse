"""``gap skills`` subcommands — list / check / table / new / test for skill bundles.

All subcommands are registry-aware: they operate over the resolved set of
skill registries (``--skills`` flags > ``$GAP_SKILLS_PATH`` > project
``[tool.gap]`` > user config > the auto-discovered open-robot-skills
sibling — see :mod:`gap.skills.registries`), and accept ``--registry NAME``
to restrict to one of them.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gap.skills.registries import RegistrySpec


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "skills",
        help="Inspect, verify, scaffold, and test skill bundles",
    )
    sub = sp.add_subparsers(dest="skills_command")

    lp = sub.add_parser("list", help="List discovered bundles across registries")
    _add_registry_args(lp)
    lp.set_defaults(func=_handle_list)

    cp = sub.add_parser(
        "check",
        help="Validate every bundle: SKILL.md format + import probe "
             "(PASS/WARN/FAIL per bundle; non-zero exit on FAIL)",
    )
    _add_registry_args(cp)
    cp.add_argument(
        "--download", action="store_true",
        help="After the checks, run each bundle's optional prefetch() "
             "to download model weights",
    )
    cp.set_defaults(func=_handle_check)

    tp = sub.add_parser(
        "table",
        help="Dump a catalog table of all discovered bundles",
    )
    _add_registry_args(tp)
    tp.add_argument(
        "--format", default="pretty", choices=["pretty", "markdown", "json"],
        help="Output format (default: pretty terminal table; markdown is "
             "paste-ready for READMEs; json is machine-readable)",
    )
    tp.add_argument(
        "--kind", default=None, choices=["tool", "skill"],
        help="Only bundles of this kind (markdown output then drops the "
             "Kind column — one paste-ready table per README section)",
    )
    tp.set_defaults(func=_handle_table)

    np_ = sub.add_parser("new", help="Scaffold a new bundle (with a unit test)")
    np_.add_argument("name", help="Bundle name (== directory name)")
    np_.add_argument(
        "--kind", required=True, choices=["tool", "skill"],
        help="Bundle kind: tools/<name> or skills/<name>",
    )
    _add_registry_args(np_)
    np_.set_defaults(func=_handle_new)

    pt = sub.add_parser(
        "test",
        help="Run bundle unit tests from the owning registry's tests/ dir "
             "(no bundle names = every registry's full suite). Put flags "
             "first; pass pytest args after `--`, e.g. "
             "`gap skills test sam3 -- -m gpu -x`",
    )
    pt.add_argument(
        "bundles", nargs="*",
        help="Bundle names to test (default: everything)",
    )
    _add_registry_args(pt)
    pt.set_defaults(func=_handle_test)

    ip = sub.add_parser(
        "install",
        help="Sync per-bundle venvs via `uv sync --project <bundle_dir>`. "
             "No-op for bundles without a pyproject.toml. To wipe a venv "
             "later, just `rm -rf <bundle>/.venv`.",
    )
    ip.add_argument(
        "bundles", nargs="*",
        help="Bundle names to install (default: nothing — pair with --all "
             "or --workflow)",
    )
    ip.add_argument(
        "--all", action="store_true",
        help="Install every bundle with a pyproject.toml across active "
             "registries (skips bundles that have none)",
    )
    ip.add_argument(
        "--workflow", default=None, metavar="DIR",
        help="Install just the bundles a workflow references (same "
             "discovery as the launcher's boot_policies)",
    )
    _add_registry_args(ip)
    ip.set_defaults(func=_handle_install)

    # NOTE: no sp.set_defaults(func=...) here — argparse applies parent
    # defaults to the namespace before the sub-subparser runs, which would
    # mask the per-subcommand handlers. `gap skills` with no subcommand
    # falls through to the top-level help in main().


def _add_registry_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--skills", action="append", default=None, metavar="PATH",
        help="Registry checkout root(s); repeatable. Overrides "
             "$GAP_SKILLS_PATH and configured registries (default: the "
             "resolved registry set — see `gap registry list`)",
    )
    parser.add_argument(
        "--registry", default=None, metavar="NAME",
        help="Restrict to one active registry by name",
    )


def _registry_set(args: argparse.Namespace, *, required: bool = True):
    """Resolve the active registries (flag > env > project > user > auto)."""
    from gap.skills import resolve_registries
    from gap.skills.registries import RegistrySet

    registry_set = resolve_registries(args.skills, required=required)
    if getattr(args, "registry", None):
        registry_set = RegistrySet([registry_set.get(args.registry)])
    return registry_set


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def _handle_list(args: argparse.Namespace) -> int:
    from gap.skills.validate import validate_checkout

    try:
        registry_set = _registry_set(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    except KeyError as exc:
        print(f"error: {exc.args[0]}")
        return 2

    rows = []
    claimed: set[str] = set()
    for spec in registry_set:
        for report in validate_checkout(spec.path):
            shadowed = report.name in claimed
            claimed.add(report.name)
            meta = report.meta
            desc = (
                meta.description.strip().splitlines()[0] if meta and meta.description
                else "(SKILL.md rejected)"
            )
            tools = ", ".join(sorted(meta.tools)) if meta and meta.tools else "-"
            name = report.name + (" (shadowed)" if shadowed else "")
            rows.append((report.kind, name, spec.name, desc, tools))

    if not rows:
        roots = ", ".join(str(p) for p in registry_set.paths())
        print(f"no bundles found under {roots}")
        return 1

    rows.sort(key=lambda r: (r[0], r[1]))
    headers = ("KIND", "NAME", "REGISTRY", "DESCRIPTION", "TOOLS")
    widths = [
        max(len(headers[c]), *(len(r[c]) for r in rows)) for c in range(4)
    ] + [len(headers[4])]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    for row in rows:
        print(fmt.format(*row))
    return 0


# ---------------------------------------------------------------------------
# check (+ --download)
# ---------------------------------------------------------------------------


def _handle_check(args: argparse.Namespace) -> int:
    """Per-bundle format validation + import probe, per registry.

    Two layers, merged into one PASS/WARN/FAIL line per bundle:

    1. **Format** (static, no imports): SKILL.md frontmatter shape per
       kind, referenced resources on disk, ``allowed_tools`` resolvable,
       declared type names, pip-extra convention — via
       :mod:`gap.skills.validate` (the same rules the open-robot-skills test
       suite enforces).
    2. **Import probe**: each bundle is registered individually so one
       broken bundle doesn't mask the rest; ImportError hints are mapped
       to the owning registry's install story (uv sync / pip extra).

    Exit status is non-zero iff any bundle FAILs.
    """
    from gap.skills.capability import dep_fix_hint, probe_bundle_import
    from gap.skills.registries import registry_dist_name
    from gap.skills.validate import BundleIssue, load_checkout_extras, validate_checkout

    try:
        registry_set = _registry_set(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    except KeyError as exc:
        print(f"error: {exc.args[0]}")
        return 2

    multi = len(registry_set) > 1
    failures = 0
    totals = {"PASS": 0, "WARN": 0, "FAIL": 0}
    all_reports = []  # (spec, report, info-or-None)
    found_any = False

    for spec in registry_set:
        extras = load_checkout_extras(spec.path) or {}
        dist_name = spec.dist_name or registry_dist_name(spec.path)
        reports = validate_checkout(spec.path)
        if not reports:
            if multi:
                print(f"[registry {spec.name}] no bundles found under {spec.path}")
            continue
        found_any = True
        if multi:
            print(f"== registry {spec.name} ({spec.path}) ==")

        for report in reports:
            hint = dep_fix_hint(
                spec.path, dist_name=dist_name, bundle=report.name,
                has_extra=report.name in extras,
            )
            probe, info = probe_bundle_import(
                report.name, report.bundle_dir, kind=report.kind, fix_hint=hint,
            )
            if not probe.ok:
                message = f"import probe failed: {probe.detail}"
                if probe.fix_hint:
                    message += f" — {probe.fix_hint}"
                report.issues.append(BundleIssue("error", message))
            all_reports.append((spec, report, info))

            status = report.status
            totals[status] += 1
            if status == "FAIL":
                failures += 1
            venv_note = _venv_note(report.bundle_dir, report.name)
            print(f"[{report.kind}] {report.name}: {status}{venv_note}")
            for issue in report.issues:
                print(f"    {issue}")
        if multi:
            print()

    if not found_any:
        roots = ", ".join(str(p) for p in registry_set.paths())
        print(f"no bundles found under {roots}")
        return 1

    n = sum(totals.values())
    print(f"\n{n} bundle(s): {totals['PASS']} PASS, {totals['WARN']} WARN, "
          f"{totals['FAIL']} FAIL")

    if args.download:
        print()
        for spec, report, info in all_reports:
            prefix = f"[{report.kind}] {report.name}"
            if multi:
                prefix = f"[{spec.name}] {prefix}"
            if info is None:
                print(f"{prefix}: skipped prefetch (import failed)")
                continue

            # Out-of-process (RPC) bundles never have `tools_module`
            # populated — the engine skips `tools.py` for them on purpose
            # (the bundle's torch / sam3 / etc. live in its own venv).
            # For those bundles, look for `prefetch` BY STATIC SOURCE-LEVEL
            # CHECK and shell into the bundle's own venv to run it. Only
            # then does the engine have a fighting chance to call
            # `huggingface_hub.snapshot_download` inside the right place.
            serving = getattr(info.meta, "serving", None)
            is_rpc = (
                serving is not None
                and getattr(serving, "protocol", "in-process") != "in-process"
            )

            if is_rpc:
                tools_py = Path(info.bundle_dir) / "tools.py"
                if not tools_py.is_file() or "def prefetch(" not in tools_py.read_text():
                    print(f"{prefix}: declares no weights (no prefetch())")
                    continue
                cmd = [
                    "uv", "run", "--project", str(info.bundle_dir),
                    "--", "python", "-c",
                    "import tools; tools.prefetch()",
                ]
                try:
                    subprocess.run(
                        cmd, check=True, cwd=info.bundle_dir,
                        capture_output=True, text=True,
                    )
                    print(f"{prefix}: prefetch OK")
                except subprocess.CalledProcessError as exc:
                    failures += 1
                    err = (exc.stderr or "").strip().splitlines()
                    msg = err[-1] if err else f"exit {exc.returncode}"
                    print(f"{prefix}: prefetch FAILED: {msg}")
                except FileNotFoundError:
                    failures += 1
                    print(f"{prefix}: prefetch FAILED: `uv` not on PATH")
                continue

            # In-process bundle: call `prefetch` directly through the
            # already-imported module.
            prefetch = None
            for module in (info.tools_module, info.module):
                fn = getattr(module, "prefetch", None) if module is not None else None
                if callable(fn):
                    prefetch = fn
                    break
            if prefetch is None:
                print(f"{prefix}: declares no weights (no prefetch())")
                continue
            try:
                prefetch()
                print(f"{prefix}: prefetch OK")
            except Exception as exc:
                failures += 1
                print(f"{prefix}: prefetch FAILED: {exc}")

    return 1 if failures else 0


# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------


def _first_sentence(description: str) -> str:
    """First sentence of a bundle description, whitespace-normalized."""
    import re

    text = " ".join(description.split())
    m = re.search(r"(?<=[.!?])\s", text)
    return text[: m.start()] if m else text


def _table_rows(spec) -> list[dict]:
    """Catalog rows from static SKILL.md parsing (no bundle imports)."""
    from gap.skills.registries import registry_dist_name
    from gap.skills.validate import load_checkout_extras, validate_checkout

    extras = load_checkout_extras(spec.path) or {}
    dist = spec.dist_name or registry_dist_name(spec.path) or ""
    rows = []
    for report in validate_checkout(spec.path):
        meta = report.meta
        rows.append({
            "name": report.name,
            "kind": report.kind,
            "registry": spec.name,
            "description": _first_sentence(meta.description) if meta else "(SKILL.md rejected)",
            "tools": sorted(meta.tools) if meta else [],
            "extra": f"{dist}[{report.name}]" if dist and report.name in extras else "",
        })
    rows.sort(key=lambda r: (r["kind"], r["name"]))
    return rows


def _print_markdown_table(rows: list[dict], kind: str | None) -> None:
    cells = [
        (
            f"[{r['name']}]({'tools' if r['kind'] == 'tool' else 'skills'}/{r['name']}/)",
            r["kind"],
            r["description"],
            ", ".join(f"`{t}`" for t in r["tools"]) if r["tools"] else "—",
            f"`{r['extra']}`" if r["extra"] else "—",
        )
        for r in rows
    ]
    headers: tuple[str, ...] = ("Bundle", "Kind", "Description", "Tools", "Extra")
    if kind:  # one table per kind: the Kind column is redundant
        headers = tuple(h for h in headers if h != "Kind")
        cells = [(n, d, t, e) for n, _, d, t, e in cells]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for row in cells:
        print("| " + " | ".join(row) + " |")


def _handle_table(args: argparse.Namespace) -> int:
    try:
        registry_set = _registry_set(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    except KeyError as exc:
        print(f"error: {exc.args[0]}")
        return 2

    per_registry = [(spec, _table_rows(spec)) for spec in registry_set]
    kind = getattr(args, "kind", None)
    if kind:
        per_registry = [
            (spec, [r for r in rows if r["kind"] == kind])
            for spec, rows in per_registry
        ]
    all_rows = [r for _, rows in per_registry for r in rows]
    if not all_rows:
        roots = ", ".join(str(p) for p in registry_set.paths())
        print(f"no bundles found under {roots}")
        return 1

    multi = len(registry_set) > 1

    if args.format == "json":
        import json

        print(json.dumps(all_rows, indent=2))
        return 0

    if args.format == "markdown":
        # Bundle names link to their directory — paste-ready for a README
        # at the registry root, so multi-registry output emits one table
        # per registry.
        for i, (spec, rows) in enumerate(per_registry):
            if not rows:
                continue
            if multi:
                if i:
                    print()
                print(f"### {spec.name}\n")
            _print_markdown_table(rows, kind)
        return 0

    # pretty terminal table (backtick markup dropped)
    headers: tuple[str, ...] = (
        "Bundle", "Kind", "Registry", "Description", "Tools", "Extra",
    )
    cells = [
        (
            r["name"], r["kind"], r["registry"], r["description"],
            ", ".join(r["tools"]) if r["tools"] else "—",
            r["extra"] or "—",
        )
        for r in all_rows
    ]
    widths = [
        max(len(headers[c]), *(len(row[c]) for row in cells))
        for c in range(len(headers))
    ]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for row in cells:
        print(fmt.format(*row))
    return 0


# ---------------------------------------------------------------------------
# new
# ---------------------------------------------------------------------------

_TOOL_SKILL_MD = """\
---
name: {name}
description: TODO one-line description of what this tool bundle computes.
  Use when TODO the situation that calls for it.
compatibility: requires gap>=0.1
metadata: {{category: TODO, tags: []}}
gap:
  # Operational requirements consumed by `gap check` — uncomment what applies:
  # requires: {{gpu: true, env: [MY_API_KEY], env_any: [], weights: true}}
  tools:
    - {name}.run: TODO summary of the tool function.
---

# {name}

TODO: describe the model this bundle wraps and its tool functions.

## When to use

- TODO
"""

_TOOL_TOOLS_PY = '''\
"""{name} tool bundle."""

from gap_core.tools import tool


@tool(name="{name}.run", summary="TODO summary of the tool function.")
def run(text: str) -> str:
    """TODO: implement."""
    raise NotImplementedError
'''

_TOOL_TEST_PY = '''\
"""Unit tests for the {name} tool bundle (CPU-only; gpu/llm tests get markers)."""


def test_{uname}_tool_registered(tool_registry):
    assert "{name}.run" in tool_registry
    desc = tool_registry.get("{name}.run")
    assert desc.summary
    assert desc.schema.inputs  # the typed signature introspected cleanly


# Enable once run() is implemented:
# def test_{uname}_invoke(tool_registry):
#     out = tool_registry.invoke("{name}.run", None, text="hello")
#     assert out == "hello"
'''

_SKILL_SKILL_MD = """\
---
name: {name}
description: TODO one-line description of this manipulation strategy.
  Use when TODO the situation that calls for it.
compatibility: requires gap>=0.1
metadata: {{category: TODO, tags: []}}
gap:
  # Operational requirements consumed by `gap check` — uncomment what applies:
  # requires: {{gpu: true, env: [MY_API_KEY], env_any: [], weights: true}}
  allowed_tools: []
  exit_conditions:
    done: TODO meaning of success.
    failed: TODO meaning of failure.
  produces_outputs: {{}}
  required_inputs: {{}}
  canonical_scripts:
    - example: scripts/example.py
---

# {name}

TODO: describe the strategy.

## When to use

- TODO
"""

_SKILL_SCRIPT_PY = '''\
"""Canonical script for the {name} skill bundle."""

from typing import TypedDict


class Output(TypedDict):
    result: str


def run(ctx, *, example_input: str = "") -> Output:
    """TODO: implement."""
    return {{"result": example_input}}
'''

_SKILL_TEST_PY = '''\
"""Unit tests for the {name} skill bundle (CPU-only via FakeContext)."""

from gap.testing import FakeContext


def test_{uname}_example_script_runs(skills_registry):
    info = skills_registry.get("{name}")
    script = info.canonical_scripts["example"].module

    ctx = FakeContext({{
        # Can the script's tool calls here, e.g.:
        # "robot.get_observation": {{"cameras": []}},
    }})
    out = script.run(ctx, example_input="hi")
    assert set(out) == {{"result"}}
    assert out["result"] == "hi"
'''


def _handle_new(args: argparse.Namespace) -> int:
    try:
        registry_set = _registry_set(args, required=False)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    except KeyError as exc:
        print(f"error: {exc.args[0]}")
        return 2

    spec = registry_set.primary()
    if spec is None:
        print(
            "no skill registry found to scaffold into.\n"
            "  create one:        gap registry init <path> --add\n"
            "  or pass a target:  gap skills new ... --skills <path>"
        )
        return 2
    root = spec.path

    folder = "tools" if args.kind == "tool" else "skills"
    bundle_dir = root / folder / args.name
    if bundle_dir.exists():
        print(f"refusing to overwrite existing bundle: {bundle_dir}")
        return 1
    bundle_dir.mkdir(parents=True)
    uname = args.name.replace("-", "_")
    scaffolded = ["SKILL.md"]
    if args.kind == "tool":
        (bundle_dir / "SKILL.md").write_text(
            _TOOL_SKILL_MD.format(name=args.name)
        )
        (bundle_dir / "tools.py").write_text(_TOOL_TOOLS_PY.format(name=args.name))
        scaffolded.append("tools.py")
        test_body = _TOOL_TEST_PY.format(name=args.name, uname=uname)
    else:
        (bundle_dir / "SKILL.md").write_text(_SKILL_SKILL_MD.format(name=args.name))
        scripts = bundle_dir / "scripts"
        scripts.mkdir()
        (scripts / "example.py").write_text(_SKILL_SCRIPT_PY.format(name=args.name))
        (bundle_dir / "prompts").mkdir()
        (bundle_dir / "references").mkdir()
        scaffolded += ["scripts/example.py", "prompts/", "references/"]
        test_body = _SKILL_TEST_PY.format(name=args.name, uname=uname)

    # Unit-test scaffold in the owning registry's tests/ dir.
    tests_dir = root / "tests"
    tests_dir.mkdir(exist_ok=True)
    conftest = tests_dir / "conftest.py"
    if not conftest.is_file():
        from .registry import _INIT_CONFTEST

        conftest.write_text(_INIT_CONFTEST)
    test_path = tests_dir / f"test_{uname}.py"
    if test_path.exists():
        print(f"note: {test_path} already exists — leaving it untouched")
    else:
        test_path.write_text(test_body)
        scaffolded.append(f"tests/test_{uname}.py")

    print(f"scaffolded {args.kind} bundle at {bundle_dir}")
    print(f"  {'  '.join(scaffolded)}")
    print(
        "next:\n"
        f"  1. declare a {args.name!r} extra in {root}/pyproject.toml "
        f"([] when it has no deps)\n"
        f"  2. gap skills check --skills {root}\n"
        f"  3. gap skills test {args.name}"
    )
    return 0


# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------


def _handle_test(args: argparse.Namespace) -> int:
    """Run bundle tests with each owning registry's own pytest config.

    Invocations run as ``sys.executable -m pytest`` with cwd at the
    registry root, so the registry's ``[tool.pytest.ini_options]``
    (gpu/llm marker deselects, testpaths) governs — exactly like running
    pytest there by hand. A registry with its own separate venv should be
    tested with ``uv run pytest`` in that registry instead.
    """
    import subprocess
    import sys

    from gap.skills.validate import validate_checkout

    try:
        registry_set = _registry_set(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    except KeyError as exc:
        print(f"error: {exc.args[0]}")
        return 2

    # `gap skills test sam3 -- -m gpu -x`: argparse folds the post--
    # tokens into the positional list; everything from the first
    # dash-prefixed token on is pytest passthrough.
    raw = list(args.bundles or [])
    bundles: list[str] = []
    passthrough: list[str] = []
    for i, token in enumerate(raw):
        if token.startswith("-"):
            passthrough = raw[i:]
            break
        bundles.append(token)

    owners: dict[str, RegistrySpec] = {}
    for spec in registry_set:
        for report in validate_checkout(spec.path):
            owners.setdefault(report.name, spec)

    unknown = [b for b in bundles if b not in owners]
    if unknown:
        known = ", ".join(sorted(owners)) or "<none>"
        print(f"error: unknown bundle(s) {', '.join(unknown)} (known: {known})")
        return 2

    # Group work per registry, preserving registry precedence order.
    groups: list[tuple[RegistrySpec, list[str] | None]] = []
    if bundles:
        for spec in registry_set:
            mine = [b for b in bundles if owners[b] is spec]
            if mine:
                groups.append((spec, mine))
    else:
        groups = [(spec, None) for spec in registry_set]

    failures = 0
    for spec, spec_bundles in groups:
        tests_dir = spec.path / "tests"
        if not tests_dir.is_dir():
            if spec_bundles:
                print(f"error: registry {spec.name} has no tests/ directory "
                      f"(requested: {', '.join(spec_bundles)})")
                failures += 1
            else:
                print(f"skipping {spec.name}: no tests/ directory")
            continue

        invocations: list[list[str]] = []
        if spec_bundles is None:
            invocations.append(["tests"])
        else:
            file_targets: list[str] = []
            k_names: list[str] = []
            for bundle in spec_bundles:
                underscored = bundle.replace("-", "_")
                test_file = tests_dir / f"test_{underscored}.py"
                if test_file.is_file():
                    file_targets.append(f"tests/test_{underscored}.py")
                else:
                    print(f"note: no tests/test_{underscored}.py in "
                          f"{spec.name} — falling back to -k {underscored!r}")
                    k_names.append(underscored)
            if file_targets:
                invocations.append(file_targets)
            if k_names:
                invocations.append(["tests", "-k", " or ".join(k_names)])

        for targets in invocations:
            cmd = [sys.executable, "-m", "pytest", *targets, *passthrough]
            print(f"running: pytest {' '.join(targets + passthrough)}  "
                  f"(in {spec.path})")
            proc = subprocess.run(cmd, cwd=spec.path)
            if proc.returncode == 5:  # pytest: no tests collected
                if spec_bundles:
                    print(f"error: no tests matched in {spec.name}")
                    failures += 1
                else:
                    print(f"note: no tests collected in {spec.name}")
            elif proc.returncode != 0:
                failures += 1

    return 1 if failures else 0


def _venv_note(bundle_dir, bundle_name: str) -> str:
    """One-token annotation appended to a `check` status line.

    Empty when the bundle has no pyproject.toml (in-process bundles use
    gap's own venv); `(venv-ready)` when the bundle's .venv/ exists;
    install hint otherwise.
    """
    from pathlib import Path

    bundle_dir = Path(bundle_dir)
    if not (bundle_dir / "pyproject.toml").is_file():
        return ""
    if (bundle_dir / ".venv").is_dir():
        return " (venv-ready)"
    return f" (venv missing — run `gap skills install {bundle_name}`)"


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------


def _handle_install(args: argparse.Namespace) -> int:
    """Sync each bundle's per-bundle venv via ``uv sync --project <bundle_dir>``.

    Selection precedence: explicit ``bundles`` positional args > ``--workflow``
    discovery > ``--all`` across active registries. Bundles without a
    ``pyproject.toml`` are skipped with a note (in-process tool / pure skill
    bundles inherit gap's venv — there's nothing to install per-bundle).
    """
    import subprocess

    from gap.skills import load_skills

    try:
        registry_set = _registry_set(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    except KeyError as exc:
        print(f"error: {exc.args[0]}")
        return 2

    # 1. Discover the bundle catalog once.
    catalog: dict[str, Path] = {}  # bundle_name -> bundle_dir
    for spec in registry_set:
        reg = load_skills(spec.path)
        for info in reg.list_skills():
            catalog.setdefault(info.name, info.bundle_dir)

    # 2. Build the install set per selection rules.
    requested: list[str] = []
    if args.bundles:
        requested.extend(args.bundles)
    if args.workflow:
        wf_required = _required_bundles_for_workflow(args.workflow, registry_set)
        for name in wf_required:
            if name not in requested:
                requested.append(name)
    if args.all:
        for name in sorted(catalog):
            if name not in requested:
                requested.append(name)

    if not requested:
        print("nothing to install — pass bundle names, --workflow DIR, or --all")
        return 2

    unknown = [n for n in requested if n not in catalog]
    if unknown:
        available = ", ".join(sorted(catalog))
        print(f"error: unknown bundle(s) {unknown!r} (available: {available})")
        return 2

    # 3. Sync each bundle that owns a pyproject.toml.
    failures = 0
    for name in requested:
        bundle_dir = catalog[name]
        if bundle_dir is None:
            print(f"[{name}] skipped: bundle_dir unknown")
            continue
        pyproject = bundle_dir / "pyproject.toml"
        if not pyproject.is_file():
            print(f"[{name}] skipped: no pyproject.toml (in-process bundle)")
            continue

        cmd = ["uv", "sync", "--project", str(bundle_dir)]
        print(f"[{name}] running: {' '.join(cmd)}")
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            failures += 1
            print(f"[{name}] FAIL (uv sync returncode={proc.returncode})")
        else:
            print(f"[{name}] OK ({bundle_dir / '.venv'})")

    return 1 if failures else 0


def _required_bundles_for_workflow(workflow_dir: str, registry_set) -> list[str]:
    """Return the set of bundle names referenced by tools in a workflow.

    Mirrors the launcher's discovery (gap.runtime.policy_boot.required_policies)
    but covers *all* bundle kinds so `gap skills install --workflow` installs
    every per-bundle venv the workflow touches, not just policies.

    Tool nodes name their bundle statically; script and router nodes call
    tools at runtime via ``ctx.tool("<bundle>.<name>", ...)``, so their
    sources are scanned for that literal pattern too — otherwise a
    script-heavy graph (every grocery example) resolves to almost nothing.
    """
    import re

    from gap.runtime.workflow import load_workflow
    from gap.skills import load_skills

    try:
        wf = load_workflow(Path(workflow_dir) / "workflow.json")
    except Exception as exc:
        print(f"error: failed to load workflow at {workflow_dir}: {exc}")
        return []

    candidates: set[str] = set()
    script_paths: set[Path] = set()
    for sg in wf.subgraphs.values():
        for node in sg.nodes.values():
            if node.type == "tool" and node.tool:
                candidates.add(node.tool.split(".", 1)[0])
            if node.script:
                script_paths.add(Path(workflow_dir) / node.script)
    tool_call_re = re.compile(r"""ctx\.tool\(\s*["']([\w-]+)\.""")
    for path in sorted(script_paths):
        try:
            candidates.update(tool_call_re.findall(path.read_text()))
        except OSError:
            continue

    # Keep only names that resolve to a bundle in the active registries
    # (drops connector tools like robot.* / sim.*); same dedup precedence
    # as the catalog.
    bundle_names: set[str] = set()
    for spec in registry_set:
        reg = load_skills(spec.path)
        for bundle in candidates:
            try:
                reg.get(bundle)
            except KeyError:
                continue
            bundle_names.add(bundle)
    return sorted(bundle_names)
