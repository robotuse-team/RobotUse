"""``gap registry`` subcommands — init / list / add / remove skill registries.

Registries are local bundle checkouts layered by precedence (see
:mod:`gap.skills.registries`). ``add``/``remove`` manage the user config
(``~/.config/gap/registries.toml``) or, with ``--project``, the nearest
pyproject's ``[tool.gap].registries`` list; ``init`` scaffolds a brand-new
registry (pyproject + tools/ + skills/ + tests/) ready for
``gap skills new``. Registries are local paths in this release — for a
remote registry, clone it first.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

#: Same shape the Agent Skills spec uses for bundle names.
_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "registry",
        help="Manage skill registries (local bundle checkouts, layered by "
             "precedence)",
    )
    sub = sp.add_subparsers(dest="registry_command")

    lp = sub.add_parser("list", help="Show the active registries, in precedence order")
    lp.add_argument(
        "--format", default="pretty", choices=["pretty", "json"],
        help="Output format",
    )
    lp.set_defaults(func=_handle_list)

    ap = sub.add_parser("add", help="Add a registry to the user or project config")
    ap.add_argument("name", help="Registry name (lowercase letters/digits/hyphens)")
    ap.add_argument("path", help="Registry checkout root (a local directory)")
    scope = ap.add_mutually_exclusive_group()
    scope.add_argument(
        "--user", action="store_true",
        help="Write to ~/.config/gap/registries.toml (default)",
    )
    scope.add_argument(
        "--project", action="store_true",
        help="Write to the nearest pyproject.toml [tool.gap].registries",
    )
    ap.set_defaults(func=_handle_add)

    rp = sub.add_parser("remove", help="Remove a registry from the config")
    rp.add_argument("name", help="Registry name to remove")
    scope = rp.add_mutually_exclusive_group()
    scope.add_argument("--user", action="store_true", help="User config (default)")
    scope.add_argument(
        "--project", action="store_true",
        help="The nearest pyproject.toml [tool.gap].registries",
    )
    rp.set_defaults(func=_handle_remove)

    ip = sub.add_parser(
        "init",
        help="Scaffold a new, empty registry (pyproject + tools/ + skills/ "
             "+ tests/)",
    )
    ip.add_argument("path", help="Directory to scaffold the registry into")
    ip.add_argument(
        "--name", default=None,
        help="Registry/distribution name (default: the directory name)",
    )
    ip.add_argument(
        "--add", action="store_true",
        help="Also add it to the user config (highest precedence)",
    )
    ip.set_defaults(func=_handle_init)

    # NOTE: no sp.set_defaults(func=...) — `gap registry` with no subcommand
    # falls through to the top-level help in main() (mirrors `gap skills`).


def _looks_like_remote(path: str) -> bool:
    return "://" in path or path.startswith("git@") or path.endswith(".git")


def _remote_rejection(name: str, path: str) -> str:
    target = f"~/skills/{name}"
    return (
        "error: remote URLs are not supported yet — registries are local "
        "directories in this release.\n"
        f"  clone it first:  git clone {path} {target}\n"
        "  install deps:    follow the registry's README (pip extras / uv sync)\n"
        f"  then:            gap registry add {name} {target}"
    )


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def _handle_list(args: argparse.Namespace) -> int:
    from gap.skills import resolve_registries
    from gap.skills.registries import (
        find_project_config,
        is_registry_dir,
        read_user_registries,
        user_config_path,
    )
    from gap.skills.validate import validate_checkout

    registry_set = resolve_registries()
    resolved_paths = {spec.path for spec in registry_set}

    rows = []
    for i, spec in enumerate(registry_set, 1):
        reports = validate_checkout(spec.path)
        rows.append({
            "priority": i,
            "name": spec.name,
            "path": str(spec.path),
            "source": spec.source,
            "dist_name": spec.dist_name,
            "origin": spec.origin,
            "tools": sum(1 for r in reports if r.kind == "tool"),
            "skills": sum(1 for r in reports if r.kind == "skill"),
            "status": "ok",
        })

    # Configured-but-broken entries are invisible to resolution; surface
    # them here so `gap registry list` is the place typos show up.
    env_or_flag_override = any(s.source in ("flag", "env") for s in registry_set)
    if not env_or_flag_override:
        for name, path in read_user_registries():
            if path.resolve() in resolved_paths:
                continue
            rows.append({
                "priority": None, "name": name, "path": str(path),
                "source": "user", "dist_name": None,
                "origin": str(user_config_path()), "tools": 0, "skills": 0,
                "status": (
                    "broken (not a registry checkout)"
                    if not is_registry_dir(path) else "shadowed path"
                ),
            })
        project = find_project_config()
        if project is not None:
            pyproject, project_paths = project
            for path in project_paths:
                if path in resolved_paths:
                    continue
                rows.append({
                    "priority": None, "name": path.name, "path": str(path),
                    "source": "project", "dist_name": None,
                    "origin": f"{pyproject} [tool.gap]", "tools": 0,
                    "skills": 0,
                    "status": "broken (not a registry checkout)",
                })

    if args.format == "json":
        import json

        print(json.dumps(rows, indent=2))
        return 0

    if not rows:
        print(
            "no registries configured or discovered.\n"
            "  clone the canonical one next to this checkout: "
            "git clone https://github.com/graph-robots/open-robot-skills.git\n"
            "  or add your own:  gap registry add <name> <path>\n"
            "  or create one:    gap registry init <path>"
        )
        return 0

    headers = ("PRIORITY", "NAME", "PATH", "SOURCE", "DIST", "TOOLS", "SKILLS", "STATUS")
    cells = [
        (
            str(r["priority"]) if r["priority"] else "-",
            r["name"], r["path"], r["source"], r["dist_name"] or "-",
            str(r["tools"]), str(r["skills"]), r["status"],
        )
        for r in rows
    ]
    widths = [
        max(len(headers[c]), *(len(row[c]) for row in cells))
        for c in range(len(headers))
    ]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    for row in cells:
        print(fmt.format(*row))
    if env_or_flag_override:
        print(
            "\nnote: $GAP_SKILLS_PATH/--skills override is in force — "
            "project/user-configured registries are suppressed"
        )
    return 0


# ---------------------------------------------------------------------------
# add / remove
# ---------------------------------------------------------------------------


def _handle_add(args: argparse.Namespace) -> int:
    from gap.skills.registries import (
        is_registry_dir,
        read_user_registries,
        user_config_path,
        write_user_registries,
    )

    if _looks_like_remote(args.path):
        print(_remote_rejection(args.name, args.path))
        return 2
    if not _NAME_RE.match(args.name):
        print(
            f"error: invalid registry name {args.name!r} (lowercase "
            f"letters/digits with single hyphens)"
        )
        return 2
    path = Path(args.path).expanduser().resolve()
    if not is_registry_dir(path):
        print(
            f"error: {path} is not a registry checkout — it needs a tools/ "
            f"or skills/ directory containing at least one bundle (a "
            f"subdirectory with a SKILL.md). Scaffold one with "
            f"`gap registry init {args.path}`."
        )
        return 2

    if args.project:
        return _project_add(path)

    entries = read_user_registries()
    if any(name == args.name for name, _ in entries):
        print(
            f"error: registry {args.name!r} already configured — "
            f"`gap registry remove {args.name}` first"
        )
        return 1
    write_user_registries([(args.name, path), *entries])
    print(
        f"added {args.name!r} -> {path} ({user_config_path()})\n"
        f"highest precedence: its bundles shadow same-named bundles in "
        f"later entries and in the auto-discovered sibling"
    )
    return 0


def _handle_remove(args: argparse.Namespace) -> int:
    from gap.skills.registries import (
        read_user_registries,
        user_config_path,
        write_user_registries,
    )

    if args.project:
        return _project_remove(args.name)

    entries = read_user_registries()
    kept = [(name, path) for name, path in entries if name != args.name]
    if len(kept) == len(entries):
        available = ", ".join(name for name, _ in entries) or "<none>"
        print(f"error: registry {args.name!r} not in {user_config_path()} "
              f"(configured: {available})")
        return 1
    write_user_registries(kept)
    print(f"removed {args.name!r} from {user_config_path()}")
    return 0


def _nearest_pyproject(start: Path) -> Path | None:
    for root in (start, *start.parents):
        candidate = root / "pyproject.toml"
        if candidate.is_file():
            return candidate
    return None


_REGISTRIES_LINE_RE = re.compile(
    r"^(?P<indent>\s*)registries\s*=\s*\[(?P<body>[^\]\n]*)\]\s*$",
    re.MULTILINE,
)


def _project_add(path: Path) -> int:
    import json
    import os

    pyproject = _nearest_pyproject(Path.cwd())
    if pyproject is None:
        print("error: no pyproject.toml found walking up from the current "
              "directory — run from inside your project, or use --user")
        return 2
    text = pyproject.read_text(encoding="utf-8")
    try:
        rel = os.path.relpath(path, pyproject.parent)
    except ValueError:  # different drive (windows)
        rel = str(path)
    entry = json.dumps(rel if not Path(rel).is_absolute() else str(path))

    if "[tool.gap]" not in text:
        block = f'\n[tool.gap]\nregistries = [{entry}]\n'
        pyproject.write_text(text + block, encoding="utf-8")
        print(f"added [tool.gap].registries = [{entry}] to {pyproject}")
        return 0

    match = _REGISTRIES_LINE_RE.search(text)
    if match is None:
        print(
            f"error: {pyproject} has a [tool.gap] table but no single-line "
            f"`registries = [...]` to edit — add {entry} to "
            f"[tool.gap].registries manually"
        )
        return 1
    body = match.group("body").strip()
    new_body = entry if not body else f"{entry}, {body}"
    new_line = f"{match.group('indent')}registries = [{new_body}]"
    pyproject.write_text(
        text[: match.start()] + new_line + text[match.end():], encoding="utf-8",
    )
    print(f"prepended {entry} to [tool.gap].registries in {pyproject}")
    return 0


def _project_remove(name: str) -> int:
    import json

    from gap.skills.registries import find_project_config, registry_dist_name

    project = find_project_config()
    if project is None:
        print("error: no pyproject.toml with [tool.gap].registries found "
              "walking up from the current directory")
        return 2
    pyproject, paths = project
    matches = [
        p for p in paths
        if (registry_dist_name(p) or p.name) == name
    ]
    if not matches:
        names = ", ".join(registry_dist_name(p) or p.name for p in paths) or "<none>"
        print(f"error: registry {name!r} not in {pyproject} (configured: {names})")
        return 1

    text = pyproject.read_text(encoding="utf-8")
    match = _REGISTRIES_LINE_RE.search(text)
    if match is None:
        print(
            f"error: {pyproject} [tool.gap].registries is not a single-line "
            f"array — remove the entry manually"
        )
        return 1
    kept = []
    for raw in match.group("body").split(","):
        raw = raw.strip()
        if not raw:
            continue
        value = raw.strip("\"'")
        resolved = (pyproject.parent / Path(value).expanduser()).resolve()
        if resolved in [m.resolve() for m in matches]:
            continue
        kept.append(json.dumps(value))
    new_line = f"{match.group('indent')}registries = [{', '.join(kept)}]"
    pyproject.write_text(
        text[: match.start()] + new_line + text[match.end():], encoding="utf-8",
    )
    print(f"removed {name!r} from [tool.gap].registries in {pyproject}")
    return 0


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

_INIT_PYPROJECT = """\
[project]
name = "{name}"
version = "0.1.0"
description = "A gap skill registry: robot skill and tool bundles."
requires-python = ">=3.10"
dependencies = ["graph-as-policy"]

[project.optional-dependencies]
# One pip extra per bundle (extra name == bundle name; use [] when the
# bundle has no deps of its own). `gap check` derives install hints from
# this table.

[dependency-groups]
dev = ["pytest>=8"]

# Developing against a side-by-side gap checkout? Uncomment:
# [tool.uv.sources]
# graph-as-policy = {{ path = "../graph-as-policy", editable = true }}

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
# This package is a dependency-metadata carrier; bundles load by path.
bypass-selection = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-m 'not gpu and not llm'"
markers = [
  "gpu: needs model weights + an NVIDIA GPU",
  "llm: needs a live LLM API key",
]
"""

_INIT_CONFTEST = '''\
"""Shared fixtures: load this registry once per session."""

from __future__ import annotations

from pathlib import Path

import pytest

#: This registry's checkout root (parent of tests/).
SKILLS_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def skills_registry():
    from gap.skills import load_skills

    return load_skills(SKILLS_ROOT)


@pytest.fixture(scope="session")
def tool_registry(skills_registry):
    """ToolRegistry with the bundles' pending @tool registrations drained."""
    from gap_core.tools import ToolRegistry

    reg = ToolRegistry()
    reg.discover_pending()
    return reg
'''

_INIT_README = """\
# {name}

A [gap](https://github.com/graph-robots/graph-as-policy) skill registry —
robot skill bundles under `skills/` and model-backed tool bundles under
`tools/`, in the Agent Skills format.

## Use it

```bash
gap registry add {name} {path}
gap check                      # what can run here?
```

## Add a bundle

```bash
gap skills new my-skill --kind skill --registry {name}
gap skills check --skills {path}
gap skills test my-skill
```

See the open-robot-skills repo for the canonical examples and the full
SKILL.md contract.
"""

_INIT_GITIGNORE = """\
__pycache__/
*.egg-info/
.venv/
.pytest_cache/
.ruff_cache/
uv.lock
"""


def _handle_init(args: argparse.Namespace) -> int:
    root = Path(args.path).expanduser().resolve()
    name = args.name or root.name
    if not _NAME_RE.match(name):
        print(
            f"error: {name!r} is not a valid registry name (lowercase "
            f"letters/digits with single hyphens) — pass --name"
        )
        return 2
    collisions = [
        p for p in ("pyproject.toml", "tools", "skills")
        if (root / p).exists()
    ]
    if collisions:
        print(f"refusing to scaffold over existing {', '.join(collisions)} in {root}")
        return 1

    (root / "tools").mkdir(parents=True)
    (root / "skills").mkdir()
    (root / "tests").mkdir(exist_ok=True)
    (root / "tools" / ".gitkeep").write_text("")
    (root / "skills" / ".gitkeep").write_text("")
    (root / "pyproject.toml").write_text(_INIT_PYPROJECT.format(name=name))
    (root / "tests" / "conftest.py").write_text(_INIT_CONFTEST)
    (root / "README.md").write_text(_INIT_README.format(name=name, path=root))
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(_INIT_GITIGNORE)

    print(f"scaffolded registry {name!r} at {root}")
    if args.add:
        ns = argparse.Namespace(name=name, path=str(root), user=True, project=False)
        return _handle_add(ns)
    print(
        "next steps:\n"
        f"  gap registry add {name} {root}\n"
        f"  gap skills new <bundle-name> --kind tool|skill --registry {name}"
    )
    return 0
