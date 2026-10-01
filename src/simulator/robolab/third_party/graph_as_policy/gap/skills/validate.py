"""Engine-side format validation for open-robot-skills bundles.

The single source of truth for the bundle-format rules that both
``gap skills check`` (the CLI) and the open-robot-skills repo test suite
enforce. Living engine-side means the checker works for third-party
bundle checkouts too, not just the canonical open-robot-skills repo.

Per bundle, :func:`validate_bundle_meta` checks:

- the description carries the spec's third-person "Use when …" cue
  (heuristic — warning only);
- the ``gap:`` block has the right shape for the bundle's kind: tool
  bundles declare ``gap.tools`` (name → summary, every name namespaced
  ``<bundle>.<func>``); skill bundles declare ``gap.exit_conditions``
  (unless they are callable-unit skills exposing ``gap.tools``);
- every referenced resource path exists on disk (``canonical_scripts``,
  ``prompts``, ``references``, ``examples``);
- ``gap.allowed_tools`` resolve against the known tool names — the
  connector surface (``robot.*`` / ``sim.*``) plus every tool declared by
  any bundle in the checkout;
- ``produces_outputs`` / ``required_inputs`` type names resolve in
  :data:`gap.schema.TYPE_REGISTRY`;
- the checkout's ``pyproject.toml`` has a pip extra named after the
  bundle (warning — the dependency-declaration convention).

:func:`validate_checkout` runs the whole checkout (frontmatter parsing
included — a SKILL.md the loader rejects becomes a FAIL report) and
returns one :class:`BundleReport` per bundle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

from gap_core.skills.meta import SkillMeta

from ._meta_from_skill_md import parse_skill_md

__all__ = [
    "BundleIssue",
    "BundleReport",
    "connector_tool_names",
    "known_tool_names",
    "load_checkout_extras",
    "validate_bundle_meta",
    "validate_checkout",
]

Severity = Literal["error", "warning"]

#: The three bundle roots, mirroring gap.skills._registry._KIND_DIRS.
_KIND_DIRS: tuple[tuple[Literal["tool", "skill", "policy"], str], ...] = (
    ("tool", "tools"),
    ("skill", "skills"),
    ("policy", "policies"),
)

# The spec's canonical cue is "Use when …", but any third-person usage cue
# ("Use for pan handles", "Use after a successful grasp") serves the same
# routing purpose for the coordinator.
_USE_WHEN_RE = re.compile(
    r"\buse\s+(?:it\s+|this\s+)?(?:when|for|after|before|to|on|with|if)\b",
    re.IGNORECASE,
)


@dataclass
class BundleIssue:
    """One finding from format validation."""

    severity: Severity
    message: str

    def __str__(self) -> str:
        return f"{self.severity}: {self.message}"


@dataclass
class BundleReport:
    """Aggregated format-validation result for one bundle."""

    name: str
    kind: Literal["tool", "skill", "policy"]
    bundle_dir: Path
    meta: SkillMeta | None = None
    issues: list[BundleIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[BundleIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[BundleIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def status(self) -> str:
        """``PASS`` | ``WARN`` | ``FAIL``."""
        if self.errors:
            return "FAIL"
        if self.warnings:
            return "WARN"
        return "PASS"


@lru_cache(maxsize=1)
def connector_tool_names() -> frozenset[str]:
    """Names of the connector-owned ``robot.*`` / ``sim.*`` tools.

    Derived from the real connector code (a throwaway :class:`SimConnector`
    around a null env — its construction and tool registration never touch
    the env), so the list can't drift from the runtime surface.
    """
    from types import SimpleNamespace

    from gap.connector.sim import SimConnector

    conn = SimConnector(None, SimpleNamespace())
    return frozenset(conn.tool_registry._tools)


def known_tool_names(metas: list[SkillMeta]) -> set[str]:
    """The flat tool namespace ``allowed_tools`` may reference.

    Connector tools plus every ``gap.tools`` name declared by any bundle
    in *metas* (declared, not imported — so the check also works when a
    bundle's optional deps are missing).
    """
    names = set(connector_tool_names())
    for meta in metas:
        names.update(meta.tools)
    return names


def load_checkout_extras(root: str | Path) -> dict[str, list[str]] | None:
    """The checkout's pyproject ``[project.optional-dependencies]`` table.

    Returns ``None`` when there is no parseable pyproject.toml (third-party
    checkouts without one skip the pip-extra convention check).
    """
    pyproject = Path(root) / "pyproject.toml"
    if not pyproject.is_file():
        return None
    try:
        import tomllib
    except ImportError:  # pragma: no cover - py3.10
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            return None
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except Exception:
        return None
    extras = data.get("project", {}).get("optional-dependencies", {})
    return dict(extras) if isinstance(extras, dict) else None


def validate_bundle_meta(
    meta: SkillMeta,
    *,
    kind: Literal["tool", "skill", "policy"],
    bundle_dir: str | Path | None = None,
    known_tools: set[str] | frozenset[str] | None = None,
    extras: dict[str, list[str]] | None = None,
) -> list[BundleIssue]:
    """Format-validate one parsed bundle.

    Args:
        meta: The bundle's parsed SKILL.md (``parse_skill_md`` already
            enforced the hard spec rules: name == dirname, description
            present and ≤1024 chars, gap extensions under ``gap:``).
        kind: Bundle kind from the folder it lives in.
        bundle_dir: Bundle directory for resource-existence checks
            (defaults to ``meta.bundle_dir``).
        known_tools: Resolvable flat tool names (see
            :func:`known_tool_names`); ``None`` skips the
            ``allowed_tools`` check.
        extras: The checkout's pip-extras table (see
            :func:`load_checkout_extras`); ``None`` skips the extra check.
    """
    issues: list[BundleIssue] = []
    bundle_dir = Path(bundle_dir or meta.bundle_dir or ".")

    def error(msg: str) -> None:
        issues.append(BundleIssue("error", msg))

    def warning(msg: str) -> None:
        issues.append(BundleIssue("warning", msg))

    # --- description heuristic -------------------------------------------
    if not _USE_WHEN_RE.search(meta.description):
        warning(
            'description should be third-person and include a "Use when …" '
            "sentence — it is the coordinator's entire view of the bundle"
        )

    # --- gap: block shape per kind ----------------------------------------
    if kind == "tool":
        if not meta.tools:
            error(
                "tool bundle declares no `gap.tools` — list every @tool "
                "function as `- <bundle>.<func>: one-line summary`"
            )
        if meta.exit_conditions:
            warning(
                "tool bundle declares `gap.exit_conditions` — that is a "
                "skill-bundle field; tool bundles document functions via "
                "`gap.tools`"
            )
        if meta.canonical_scripts:
            warning(
                "tool bundle declares `gap.canonical_scripts` — that is a "
                "skill-bundle field"
            )
    else:
        if not meta.exit_conditions and not meta.tools:
            error(
                "skill bundle declares neither `gap.exit_conditions` (for "
                "subgraph-owning skills) nor `gap.tools` (for callable-unit "
                "skills)"
            )

    # --- declared tool names ----------------------------------------------
    for tool_name in meta.tools:
        if not tool_name.startswith(f"{meta.name}."):
            error(
                f"declared tool {tool_name!r} is not namespaced under the "
                f"bundle (expected '{meta.name}.<func>'; 'robot.*'/'sim.*' "
                f"are reserved for connectors)"
            )

    # --- referenced resources exist ----------------------------------------
    for script in meta.canonical_scripts:
        if not (bundle_dir / script.path).is_file():
            error(f"canonical_scripts entry {script.name!r} -> {script.path} is missing")
    for logical, rel in meta.prompts.items():
        if not (bundle_dir / rel).is_file():
            error(f"prompts entry {logical!r} -> {rel} is missing")
    for ref in meta.references:
        if not (bundle_dir / ref.path).is_file():
            error(f"references entry {ref.path!r} is missing")
    for ex in meta.examples:
        if not (bundle_dir / ex.path).is_file():
            error(f"examples entry {ex.path!r} is missing")

    # --- allowed_tools resolve ----------------------------------------------
    if known_tools is not None and meta.allowed_tools:
        unknown = sorted(set(meta.allowed_tools) - set(known_tools))
        if unknown:
            error(
                f"allowed_tools reference unknown tools {unknown} (known: "
                f"connector robot.*/sim.* tools plus every bundle's declared "
                f"gap.tools)"
            )

    # --- declared I/O type names resolve ------------------------------------
    from gap_core.schema import TYPE_REGISTRY

    for label, mapping in (
        ("produces_outputs", meta.produces_outputs),
        ("required_inputs", meta.required_inputs),
    ):
        for name, type_name in mapping.items():
            if type_name not in TYPE_REGISTRY:
                error(
                    f"{label}[{name!r}] uses unknown type {type_name!r} "
                    f"(must be a gap.schema type name)"
                )

    # --- pip extra convention -------------------------------------------------
    # Bundles that own a per-bundle `pyproject.toml` are exempted: gap manages
    # their venv via `gap skills install <name>` (uv sync --project <dir>),
    # so they don't need to appear in the root extras table.
    if (
        extras is not None
        and meta.name not in extras
        and not (bundle_dir / "pyproject.toml").is_file()
    ):
        warning(
            f"pyproject.toml has no pip extra named {meta.name!r} — declare "
            f"the bundle's dependencies as one extra (empty list when it has "
            f"none)"
        )

    # --- gap.requires consistency ---------------------------------------------
    req = meta.requires
    gpu_tagged = "gpu" in meta.tags
    if req is not None and req.gpu and not gpu_tagged:
        warning(
            "gap.requires.gpu is true but metadata.tags lacks 'gpu' — add "
            "the tag so catalogs reflect the hardware need"
        )
    if gpu_tagged and (req is None or not req.gpu):
        warning(
            "metadata.tags includes 'gpu' but gap.requires does not declare "
            "gpu: true — `gap check` will not probe for a GPU"
        )
    if req is not None and req.weights:
        has_module = (bundle_dir / "tools.py").is_file() or (
            bundle_dir / "skill.py"
        ).is_file()
        if not has_module:
            warning(
                "gap.requires.weights is true but the bundle has no "
                "tools.py/skill.py to host weights_cached()/prefetch()"
            )

    return issues


def validate_checkout(
    root: str | Path,
    *,
    only: list[str] | None = None,
) -> list[BundleReport]:
    """Format-validate every bundle in an open-robot-skills checkout.

    Pure-static validation: SKILL.md files are parsed but no bundle code
    is imported, so the check works even when a bundle's optional pip
    extra is not installed. A SKILL.md the loader rejects (spec
    violations raised by :func:`parse_skill_md`) becomes a FAIL report.
    """
    root = Path(root)
    extras = load_checkout_extras(root)

    reports: list[BundleReport] = []
    for kind, folder in _KIND_DIRS:
        bundles_dir = root / folder
        if not bundles_dir.is_dir():
            continue
        for entry in sorted(bundles_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith("_"):
                continue
            if not (entry / "SKILL.md").is_file():
                continue
            if only is not None and entry.name not in only:
                continue
            report = BundleReport(name=entry.name, kind=kind, bundle_dir=entry)
            try:
                report.meta = parse_skill_md(entry / "SKILL.md")
            except (ValueError, OSError) as exc:
                report.issues.append(BundleIssue("error", f"SKILL.md rejected: {exc}"))
            reports.append(report)

    known = known_tool_names([r.meta for r in reports if r.meta is not None])
    for report in reports:
        if report.meta is None:
            continue
        report.issues.extend(validate_bundle_meta(
            report.meta,
            kind=report.kind,
            bundle_dir=report.bundle_dir,
            known_tools=known,
            extras=extras,
        ))
    return reports
