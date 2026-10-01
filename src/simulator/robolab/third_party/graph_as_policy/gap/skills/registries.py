"""First-class skill registries — configure, resolve, and merge bundle sources.

A *registry* is any local directory holding bundles under ``tools/``
and/or ``skills/`` roots (each immediate subdirectory with a ``SKILL.md``
is a bundle). The canonical public example is `open-robot-skills
<https://github.com/graph-robots/open-robot-skills>`_, but gap treats it
as exactly that — an example. Labs and projects bring their own
registries and layer them.

Multiple registries are active at once, in precedence order. The merged
catalog is the union; a bundle name claimed by a higher-precedence
registry *shadows* same-named bundles below it (first wins, loud
warning), so a lab can override a public skill without forking the whole
registry. Resolution order:

1. explicit ``--skills`` flags / ``skills=`` arguments — **full
   override**: exactly the listed registries run, nothing merged in;
2. ``$GAP_SKILLS_PATH`` — an ``os.pathsep``-separated list (a single
   path keeps its historical meaning) — also a full override; every
   entry must be a valid registry (an explicitly-set env var must be
   valid — silently falling through would mask typos);
3. the nearest ``pyproject.toml`` with a ``[tool.gap].registries`` list,
   walking up from the cwd (paths relative to that file);
4. user config ``~/.config/gap/registries.toml`` (managed by
   ``gap registry add/remove``);
5. an ``open-robot-skills`` checkout auto-discovered next to the gap
   checkout or the cwd — the documented side-by-side layout.

Sources 3–5 merge (project entries first, then user, then auto),
deduplicated by resolved path. Registries are local directories in this
release; git-managed registries are future work (``gap registry add``
tells you to clone first).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .discovery import (
    _SIBLING_DIR_NAME,
    GAP_SKILLS_PATH_ENV,
    looks_like_skills_checkout,
)

logger = logging.getLogger(__name__)

__all__ = [
    "RegistrySet",
    "RegistrySpec",
    "as_registry_paths",
    "find_project_config",
    "is_registry_dir",
    "load_registry_set",
    "read_user_registries",
    "registry_dist_name",
    "resolve_registries",
    "user_config_path",
    "write_user_registries",
]

RegistrySource = Literal["flag", "env", "project", "user", "auto"]


def is_registry_dir(path: str | Path) -> bool:
    """True iff *path* could hold bundles — a ``tools/`` or ``skills/`` dir.

    Lenient on purpose: a freshly ``gap registry init``-ed registry has
    empty bundle roots and must still be addable/configurable. Auto-
    discovery uses the stricter :func:`looks_like_skills_checkout`
    (populated roots) so arbitrary directories are never picked up
    implicitly.
    """
    p = Path(path)
    return p.is_dir() and ((p / "tools").is_dir() or (p / "skills").is_dir())

#: Human labels for provenance columns/messages, keyed by source.
SOURCE_LABELS: dict[str, str] = {
    "flag": "--skills",
    "env": f"${GAP_SKILLS_PATH_ENV}",
    "project": "project pyproject [tool.gap]",
    "user": "user config",
    "auto": "auto-discovered",
}


@dataclass(frozen=True)
class RegistrySpec:
    """One resolved skill registry (a local bundle checkout)."""

    name: str
    """Unique name within a :class:`RegistrySet`. Named sources (user
    config) carry their configured name; unnamed sources derive it from
    the registry's pyproject ``[project].name``, falling back to the
    directory name."""

    path: Path
    """Absolute, resolved checkout root."""

    source: RegistrySource
    """Which resolution layer produced this entry."""

    dist_name: str | None = None
    """``[project].name`` from the registry's own pyproject, when it has
    one — used for ``pip install '<dist>[<bundle>]'`` fix hints."""

    origin: str = ""
    """Provenance detail for messages, e.g. ``"$GAP_SKILLS_PATH[1]"`` or
    the config file that named this registry."""


@dataclass
class RegistrySet:
    """An ordered set of registries; index 0 = highest precedence."""

    registries: list[RegistrySpec] = field(default_factory=list)

    def __iter__(self):
        return iter(self.registries)

    def __len__(self) -> int:
        return len(self.registries)

    def __bool__(self) -> bool:
        return bool(self.registries)

    def paths(self) -> list[Path]:
        return [spec.path for spec in self.registries]

    def names(self) -> list[str]:
        return [spec.name for spec in self.registries]

    def get(self, name: str) -> RegistrySpec:
        for spec in self.registries:
            if spec.name == name:
                return spec
        available = ", ".join(self.names()) or "<none>"
        raise KeyError(f"registry {name!r} not found (active: {available})")

    def primary(self) -> RegistrySpec | None:
        """The highest-precedence registry, or ``None`` when empty."""
        return self.registries[0] if self.registries else None


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def as_registry_paths(
    value: str | Path | Sequence[str | Path] | None,
) -> list[Path] | None:
    """Normalize every accepted ``skills=`` shape to a list of paths.

    ``None`` stays ``None`` (= "resolve from env/config/auto"); a single
    str/Path becomes a one-element list; an empty sequence is treated as
    "not given" (``None``) rather than "no registries".
    """
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return [Path(value).expanduser().resolve()]
    paths = [Path(v).expanduser().resolve() for v in value]
    return paths or None


def _tomllib():
    try:
        import tomllib
    except ImportError:  # pragma: no cover - py3.10
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            return None
    return tomllib


def registry_dist_name(root: str | Path) -> str | None:
    """``[project].name`` from the registry's own pyproject.toml, if any."""
    pyproject = Path(root) / "pyproject.toml"
    if not pyproject.is_file():
        return None
    toml = _tomllib()
    if toml is None:  # pragma: no cover - py3.10 without tomli
        return None
    try:
        data = toml.loads(pyproject.read_text(encoding="utf-8"))
    except Exception:
        return None
    name = data.get("project", {}).get("name")
    return str(name) if name else None


# ---------------------------------------------------------------------------
# User config (~/.config/gap/registries.toml)
# ---------------------------------------------------------------------------


def user_config_path() -> Path:
    """``$XDG_CONFIG_HOME/gap/registries.toml`` (default ``~/.config``)."""
    base = os.environ.get("XDG_CONFIG_HOME", "").strip()
    config_home = Path(base).expanduser() if base else Path.home() / ".config"
    return config_home / "gap" / "registries.toml"


def read_user_registries() -> list[tuple[str, Path]]:
    """``[(name, path), ...]`` from the user config, in precedence order.

    Tolerant: a missing or unparseable file is an empty list (the file is
    optional); malformed entries are skipped with a warning so one typo
    doesn't take every gap command down.
    """
    config = user_config_path()
    if not config.is_file():
        return []
    toml = _tomllib()
    if toml is None:  # pragma: no cover - py3.10 without tomli
        return []
    try:
        data = toml.loads(config.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("ignoring unparseable %s: %s", config, exc)
        return []
    entries: list[tuple[str, Path]] = []
    for entry in data.get("registry", []) or []:
        name = str(entry.get("name", "") or "")
        path = str(entry.get("path", "") or "")
        if not name or not path:
            logger.warning(
                "ignoring malformed [[registry]] entry in %s (need both "
                "name and path): %r", config, entry,
            )
            continue
        entries.append((name, Path(path).expanduser()))
    return entries


def write_user_registries(entries: list[tuple[str, Path]]) -> None:
    """Rewrite the user config canonically (order = precedence)."""
    import json

    config = user_config_path()
    config.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Skill registries managed by `gap registry add/remove`.",
        "# Order = precedence: earlier entries shadow same-named bundles",
        "# in later ones.",
    ]
    for name, path in entries:
        lines += [
            "",
            "[[registry]]",
            f"name = {json.dumps(name)}",   # JSON string == TOML basic string
            f"path = {json.dumps(str(path))}",
        ]
    config.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Project config (pyproject.toml [tool.gap].registries)
# ---------------------------------------------------------------------------


def find_project_config(cwd: str | Path | None = None) -> tuple[Path, list[Path]] | None:
    """The nearest pyproject.toml declaring ``[tool.gap].registries``.

    Walks up from *cwd* (continuing past pyprojects without a
    ``[tool.gap]`` table, so a nested package inside a configured
    workspace still sees the workspace config). Returns
    ``(pyproject_path, registry_paths)`` with paths resolved relative to
    the pyproject's directory, or ``None``.
    """
    toml = _tomllib()
    if toml is None:  # pragma: no cover - py3.10 without tomli
        return None
    start = Path(cwd or Path.cwd()).resolve()
    for root in (start, *start.parents):
        pyproject = root / "pyproject.toml"
        if not pyproject.is_file():
            continue
        try:
            data = toml.loads(pyproject.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("ignoring unparseable %s: %s", pyproject, exc)
            continue
        gap_table = data.get("tool", {}).get("gap")
        if not isinstance(gap_table, dict) or "registries" not in gap_table:
            continue
        raw = gap_table["registries"]
        if not isinstance(raw, list):
            logger.warning(
                "%s [tool.gap].registries must be a list of paths; ignoring",
                pyproject,
            )
            return None
        paths = [
            (root / Path(str(p)).expanduser()).resolve() for p in raw
        ]
        return pyproject, paths
    return None


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _default_search_from() -> tuple[Path, ...]:
    """Auto-discovery walk roots: the gap package dir and the cwd."""
    return (Path(__file__).resolve().parent, Path.cwd())


def _discover_sibling(
    search_from: Iterable[str | Path], tried: list[str],
) -> Path | None:
    """Today's sibling walk: first open-robot-skills checkout found."""
    seen: set[Path] = set()
    for start in search_from:
        start = Path(start).resolve()
        for root in (start, *start.parents):
            if root in seen:
                continue
            seen.add(root)
            # Running *inside* a checkout counts (e.g. `gap skills list`
            # from a registry root).
            if looks_like_skills_checkout(root):
                return root
            sibling = root / _SIBLING_DIR_NAME
            if looks_like_skills_checkout(sibling):
                return sibling.resolve()
        tried.append(
            f"a '{_SIBLING_DIR_NAME}' directory in {start} or any of its parents"
        )
    return None


def _unique_name(base: str, taken: set[str]) -> str:
    if base not in taken:
        return base
    n = 2
    while f"{base}-{n}" in taken:
        n += 1
    logger.warning(
        "registry name %r already taken by a higher-precedence registry; "
        "using %r", base, f"{base}-{n}",
    )
    return f"{base}-{n}"


def _make_spec(
    path: Path,
    *,
    source: RegistrySource,
    origin: str,
    name: str | None,
    taken: set[str],
) -> RegistrySpec:
    dist = registry_dist_name(path)
    base = name or dist or path.name
    unique = _unique_name(base, taken)
    taken.add(unique)
    return RegistrySpec(
        name=unique, path=path, source=source, dist_name=dist, origin=origin,
    )


def resolve_registries(
    explicit: str | Path | Sequence[str | Path] | None = None,
    *,
    required: bool = False,
    cwd: str | Path | None = None,
    search_from: Iterable[str | Path] | None = None,
    include_config: bool = True,
) -> RegistrySet:
    """Resolve the active, ordered set of skill registries.

    Args:
        explicit: ``--skills`` flag values / ``skills=`` argument — full
            override when given (a str/Path, or a sequence of them).
        required: When True, raise :class:`FileNotFoundError` (listing
            everything that was tried) instead of returning an empty set.
        cwd: Starting point for the project-config walk-up (tests).
        search_from: Override the auto-discovery walk roots (tests).
        include_config: When False, skip the project/user config layers —
            the historical ``find_skills_path`` semantics (explicit >
            env > auto-discovery). Used by the back-compat shim so legacy
            callers' behavior is bit-for-bit unchanged.

    Raises:
        FileNotFoundError: nothing found and ``required=True``.
        ValueError: ``$GAP_SKILLS_PATH`` is set but one of its entries is
            not a registry checkout.
    """
    taken: set[str] = set()
    tried: list[str] = []

    # 1. Explicit argument — full override, no shape validation (problems
    # surface loudly at load time, and hermetic callers pass fixtures).
    paths = as_registry_paths(explicit)
    if paths is not None:
        specs = []
        seen: set[Path] = set()
        for i, p in enumerate(paths):
            if p in seen:
                continue
            seen.add(p)
            origin = "--skills / skills= argument"
            if len(paths) > 1:
                origin += f" [{i}]"
            specs.append(_make_spec(p, source="flag", origin=origin, name=None, taken=taken))
        return RegistrySet(specs)
    tried.append("explicit path argument (not given)")

    # 2. $GAP_SKILLS_PATH — full override; every entry must be valid.
    env = os.environ.get(GAP_SKILLS_PATH_ENV, "").strip()
    if env:
        specs = []
        seen = set()
        entries = [e for e in env.split(os.pathsep) if e.strip()]
        for i, raw in enumerate(entries):
            p = Path(raw.strip()).expanduser()
            if not looks_like_skills_checkout(p):
                where = f"[{i}] " if len(entries) > 1 else ""
                raise ValueError(
                    f"${GAP_SKILLS_PATH_ENV} entry {where}{raw!r} is not a skill "
                    f"registry (expected a directory with a tools/ or skills/ "
                    f"bundle root containing at least one SKILL.md bundle)"
                )
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            origin = f"${GAP_SKILLS_PATH_ENV}"
            if len(entries) > 1:
                origin += f"[{i}]"
            specs.append(_make_spec(rp, source="env", origin=origin, name=None, taken=taken))
        if specs:
            return RegistrySet(specs)
    tried.append(f"${GAP_SKILLS_PATH_ENV} (not set)")

    # 3-5. Project config ∪ user config ∪ auto-discovered sibling.
    specs = []
    seen = set()

    if include_config:
        project = find_project_config(cwd)
        if project is not None:
            pyproject, project_paths = project
            for p in project_paths:
                if p in seen:
                    continue
                if not is_registry_dir(p):
                    logger.warning(
                        "skipping registry %s from %s [tool.gap].registries — "
                        "not a registry checkout (no tools/ or skills/ root)",
                        p, pyproject,
                    )
                    continue
                seen.add(p)
                specs.append(_make_spec(
                    p, source="project", origin=f"{pyproject} [tool.gap]",
                    name=None, taken=taken,
                ))
        else:
            tried.append("a pyproject.toml with [tool.gap].registries (none found)")

        user_entries = read_user_registries()
        if user_entries:
            for name, p in user_entries:
                rp = p.resolve()
                if rp in seen:
                    continue
                if not is_registry_dir(rp):
                    logger.warning(
                        "skipping registry %r (%s) from %s — not a registry "
                        "checkout; `gap registry remove %s` to drop it",
                        name, p, user_config_path(), name,
                    )
                    continue
                seen.add(rp)
                specs.append(_make_spec(
                    rp, source="user", origin=str(user_config_path()),
                    name=name, taken=taken,
                ))
        else:
            tried.append(f"{user_config_path()} (no entries)")

    sibling = _discover_sibling(
        _default_search_from() if search_from is None else search_from, tried,
    )
    if sibling is not None and sibling.resolve() not in seen:
        specs.append(_make_spec(
            sibling.resolve(), source="auto",
            origin="auto-discovered side-by-side checkout", name=None,
            taken=taken,
        ))

    if specs:
        return RegistrySet(specs)

    message = (
        "no skill registry found. Tried, in order: "
        + "; ".join(f"({i}) {t}" for i, t in enumerate(tried, 1))
        + ". Clone open-robot-skills next to the graph-as-policy checkout, "
        f"set ${GAP_SKILLS_PATH_ENV}=/path/to/registry, run "
        "`gap registry add <name> <path>`, or pass an explicit skills path."
    )
    if required:
        raise FileNotFoundError(message)
    logger.debug("%s", message)
    return RegistrySet([])


# ---------------------------------------------------------------------------
# Merged loading
# ---------------------------------------------------------------------------


def load_registry_set(
    registry_set: RegistrySet | Sequence[str | Path] | str | Path,
    *,
    only: list[str] | None = None,
    disable: list[str] | None = None,
):
    """Load an ordered registry set into ONE merged ``SkillsRegistry``.

    Bundles register in precedence order; a name already claimed by a
    higher-precedence registry is skipped with a warning *before* any of
    its modules import (see ``SkillsRegistry.discover``), so shadowing is
    safe for the process-global ``gap_skills.*`` synthetic namespace.
    """
    from ._registry import _KIND_DIRS, SkillsRegistry

    if isinstance(registry_set, RegistrySet):
        specs = list(registry_set)
    else:
        paths = as_registry_paths(registry_set) or []
        taken: set[str] = set()
        specs = [
            _make_spec(p, source="flag", origin="skills= argument", name=None, taken=taken)
            for p in paths
        ]

    reg = SkillsRegistry()
    for spec in specs:
        found_any = False
        for kind, folder in _KIND_DIRS:
            bundles_dir = spec.path / folder
            if not bundles_dir.is_dir():
                continue
            found_any = True
            reg.discover(
                bundles_dir, kind=kind, only=only, disable=disable,
                registry=spec.name,
            )
        if not found_any:
            logger.warning(
                "registry %r (%s) has neither a tools/ nor a skills/ "
                "directory", spec.name, spec.path,
            )
    return reg
