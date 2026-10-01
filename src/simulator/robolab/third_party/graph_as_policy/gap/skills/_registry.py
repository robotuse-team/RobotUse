"""Skill-bundle registry — discovers, registers, and manages open-robot-skills bundles.

A bundle is a directory containing at minimum a ``SKILL.md``. An open-robot-skills
checkout has three bundle roots, and the folder conveys the bundle's kind:

- ``<root>/tools/<bundle>/``    — model-backed callables (``kind="tool"``);
  exposes typed functions via ``@tool`` in ``tools.py``.
- ``<root>/skills/<bundle>/``   — manipulation strategies (``kind="skill"``);
  ship canonical scripts under ``scripts/`` that the subgraph_agent emits
  as ``type: script`` states, and *may* also expose a callable (a
  function-style ``run()`` or a class-based :class:`Skill`) when invocable
  as a single unit (tracking-objects).
- ``<root>/policies/<bundle>/`` — learned-policy skills (``kind="policy"``);
  one bundle per checkpoint. Owns its own pyproject + venv and declares its
  server launch recipe in SKILL.md ``gap.serving:`` — the launcher boots
  one server per referenced preset via ``uv run --project <bundle_dir>``.

Build a registry from a checkout with :func:`load_skills`; discovery walks
each immediate subdirectory of both roots:

- Reads SKILL.md → :class:`SkillMeta` via :func:`parse_skill_md` (which
  enforces ``name == dirname`` and the Agent Skills spec limits) and checks
  the ``compatibility:`` constraint against the installed gap version.
- Discovers each canonical script under ``scripts/`` (via
  :func:`extract_schema`) so the assembler can show the subgraph_agent
  typed-script schemas.
- Imports ``tools.py`` (when present) through the synthetic package so its
  ``@tool`` decorators land in ``gap.tools._registry._PENDING_TOOLS`` for
  ``ToolRegistry.discover_pending()`` to drain.
- Imports ``skill.py`` (when present) and wires up the function-style or
  class-based callable, exactly as the former atomic path did.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import logging
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from gap_core.skills.meta import Skill, SkillMeta
from gap_core.tools.schema import UnitSchema, extract_schema

from ._meta_from_skill_md import parse_skill_md

logger = logging.getLogger(__name__)

#: Synthetic-package prefix all bundles register under.
_SYNTHETIC_ROOT = "gap_skills"

#: Folder name (== synthetic-package namespace) for each bundle kind.
_KIND_DIRS: tuple[tuple[Literal["tool", "skill", "policy"], str], ...] = (
    ("tool", "tools"),
    ("skill", "skills"),
    ("policy", "policies"),
)


# Per-module-name locks so concurrent ``discover()`` calls (e.g. several
# ``SkillsRegistry`` instances built in parallel in the same process) don't
# race on ``sys.modules``. Without this, thread A can insert an un-exec'd
# shell module before ``exec_module`` runs, and thread B's early
# ``if module_name in sys.modules`` returns the shell — leading to spurious
# ``Module … has no callable run()`` failures.
_IMPORT_LOCKS_GUARD = threading.Lock()
_IMPORT_LOCKS: dict[str, threading.Lock] = {}


def _get_import_lock(name: str) -> threading.Lock:
    with _IMPORT_LOCKS_GUARD:
        lock = _IMPORT_LOCKS.get(name)
        if lock is None:
            lock = threading.Lock()
            _IMPORT_LOCKS[name] = lock
        return lock


def _ensure_synthetic_package(dotted: str, path: Path | None) -> ModuleType:
    """Idempotently install a synthetic namespace package in ``sys.modules``.

    Walks the dotted name from root to leaf, creating each missing
    ancestor as a bare :class:`ModuleType`. The leaf (when *path* is
    given) gets ``__path__`` set so importlib treats it as a package
    rooted at that directory — which lets ``importlib.import_module()``
    resolve and lets ``__file__``-walking helpers (used by
    :func:`gap.skills.load_prompt`) find the bundle directory.

    This is the same trick :mod:`pkgutil.extend_path` uses for namespace
    packages — standard Python plumbing.
    """
    if dotted in sys.modules:
        return sys.modules[dotted]
    parent, _, leaf = dotted.rpartition(".")
    if parent:
        _ensure_synthetic_package(parent, None)
    mod = ModuleType(dotted)
    mod.__package__ = dotted
    if path is not None:
        mod.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[dotted] = mod
    if parent:
        setattr(sys.modules[parent], leaf, mod)
    return mod


def load_skills(
    root: str | Path,
    *,
    only: list[str] | None = None,
    disable: list[str] | None = None,
) -> SkillsRegistry:
    """Construct a :class:`SkillsRegistry` from an open-robot-skills checkout.

    Discovers all three bundle roots — ``<root>/tools/``, ``<root>/skills/``,
    and ``<root>/policies/`` — assigning each bundle its ``kind`` from the
    folder it lives in. Bundle name collisions (within or across roots)
    raise :class:`ValueError` at registration time (silent last-wins is a
    debugging trap).
    """
    root = Path(root)
    reg = SkillsRegistry()
    found_any = False
    for kind, folder in _KIND_DIRS:
        bundles_dir = root / folder
        if not bundles_dir.is_dir():
            continue
        found_any = True
        reg.discover(bundles_dir, kind=kind, only=only, disable=disable)
    if not found_any:
        logger.warning(
            "open-robot-skills root %s has none of tools/, skills/, or policies/",
            root,
        )
    return reg


@dataclass
class ScriptInfo:
    """Schema for a single canonical script bundled in a skill."""

    name: str               # logical name from SKILL.md frontmatter
    path: Path              # absolute path on disk
    bundle_relative: str    # relative path the subgraph_agent emits in type:script states
    module: Any             # imported Python module
    schema: UnitSchema      # introspected I/O schema


@dataclass
class SkillInfo:
    """Metadata + cached schema for a registered skill bundle."""

    name: str
    kind: Literal["tool", "skill", "policy"]
    """Which bundle root the bundle was discovered under: ``tools/``,
    ``skills/``, or ``policies/``. Replaces the legacy ``runtime.shape``
    field."""

    bundle_dir: Path
    meta: SkillMeta
    schema: UnitSchema                        # for callable bundles; empty UnitSchema otherwise
    module: Any | None = None                 # imported skill.py module (callable bundles only)
    tools_module: Any | None = None           # imported tools.py module (@tool functions)
    skill_class: type | None = None           # Skill subclass (class-based callable only)
    canonical_scripts: dict[str, ScriptInfo] = field(default_factory=dict)
    """``{logical_name: ScriptInfo}`` for each declared script under ``scripts/``."""
    namespace: str = "skills"
    """Synthetic-package segment this bundle was registered under
    (``gap_skills.<namespace>.<name>.*``). Mirrors the kind folder."""
    registry: str = ""
    """Name of the skill registry this bundle was loaded from (e.g.
    ``"open-robot-skills"`` — see :mod:`gap.skills.registries`). Empty when
    the bundle was loaded directly via :func:`load_skills` without registry
    attribution."""


class SkillsRegistry:
    """Discovers, registers, and manages open-robot-skills bundles.

    The registry exposes the same public API surface as the dev tree's
    registry it replaces: ``discover()``, ``register_bundle()``, ``get()``,
    ``list_skills()``, ``__contains__``, ``__len__``, ``call()``,
    ``generate_docs()``, plus the underscore-prefixed ``_skills`` dict and
    ``_render_skill_doc`` reader that the prompt assembler uses directly.
    """

    def __init__(self) -> None:
        self._skills: dict[str, SkillInfo] = {}

    # ------------------------------------------------------------------
    # Discovery & registration
    # ------------------------------------------------------------------

    def discover(
        self,
        bundles_dir: str | Path,
        *,
        kind: Literal["tool", "skill", "policy"] = "skill",
        only: list[str] | None = None,
        disable: list[str] | None = None,
        registry: str = "",
    ) -> None:
        """Auto-discover all bundles in *bundles_dir*.

        Walks every immediate subdirectory of *bundles_dir*. A directory
        is registered as a bundle iff it contains ``SKILL.md``.
        Underscore-prefixed directories are skipped.

        Args:
            bundles_dir: Bundle root containing per-bundle subdirectories
                (``<registry>/tools`` or ``<registry>/skills``).
            kind: Bundle kind conveyed by the folder; also the synthetic
                package segment (bundles register as
                ``gap_skills.<tools|skills>.<bundle_name>.*``).
            only: Optional allowlist of bundle names; all others skipped.
            disable: Optional blocklist of bundle names.
            registry: Registry-name attribution for the discovered bundles
                (see :mod:`gap.skills.registries`). Also drives collision
                semantics: a name collision *within* one registry is a hard
                error, while a collision with a bundle from a different
                (higher-precedence) registry shadows — first wins, with a
                warning — and the shadowed bundle is skipped *before* any
                of its modules import.

        Raises:
            ValueError: if a bundle name is already registered by the same
                registry (collision across its roots), or a bundle violates
                the SKILL.md spec (name/dirname mismatch, over-long
                description, legacy frontmatter). Per-bundle *import*
                failures are logged and skipped — one bad bundle does not
                block the rest.
        """
        bundles_dir = Path(bundles_dir)
        if not bundles_dir.is_dir():
            logger.warning("skill bundles directory does not exist: %s", bundles_dir)
            return

        only_set = set(only) if only else None
        disable_set = set(disable or [])

        for entry in sorted(bundles_dir.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name.startswith("_"):
                continue
            skill_md = entry / "SKILL.md"
            if not skill_md.is_file():
                continue
            if only_set is not None and entry.name not in only_set:
                continue
            if entry.name in disable_set:
                continue
            if entry.name in self._skills:
                existing = self._skills[entry.name]
                if existing.registry == registry:
                    raise ValueError(
                        f"skill bundle name collision: {entry.name!r} is "
                        f"already registered from {existing.bundle_dir} "
                        f"(kind={existing.kind!r}); cannot also register from "
                        f"{entry} (kind={kind!r}). Use the `disable:` or `only:` "
                        f"knobs to resolve, or rename one of the bundles."
                    )
                logger.warning(
                    "bundle %r from registry %r (%s) is shadowed by the "
                    "same-named bundle from registry %r (%s) — registry "
                    "precedence order wins",
                    entry.name, registry or "<unnamed>", entry,
                    existing.registry or "<unnamed>", existing.bundle_dir,
                )
                continue
            try:
                self.register_bundle(entry.name, entry, kind=kind, registry=registry)
            except ValueError:
                # Spec violations are hard errors — a malformed bundle must
                # not be silently dropped from the catalog.
                raise
            except Exception:
                logger.warning("Failed to register skill bundle %r", entry.name, exc_info=True)

    def register_bundle(
        self,
        name: str,
        bundle_dir: Path,
        *,
        kind: Literal["tool", "skill", "policy"] = "skill",
        registry: str = "",
    ) -> None:
        """Register a single bundle by name.

        Parses SKILL.md, imports the canonical scripts / ``skill.py`` /
        ``tools.py`` through the synthetic package, runs
        :func:`extract_schema` on each callable, and stores a
        :class:`SkillInfo`.

        Raises:
            ValueError: if the bundle is malformed (missing required files,
                schema introspection fails, frontmatter violates the spec).
        """
        skill_md = bundle_dir / "SKILL.md"
        meta = parse_skill_md(skill_md)
        meta.kind = kind
        _check_compatibility(meta.compatibility, name)

        namespace = dict(_KIND_DIRS)[kind]

        # The synthetic namespace is process-global and keyed by bundle
        # name only. If a same-named bundle from a *different* directory
        # was imported earlier in this process (another SkillsRegistry,
        # different registry precedence), its modules win silently — warn
        # so precedence changes are made in a fresh process.
        dotted = f"{_SYNTHETIC_ROOT}.{namespace}.{name}"
        prior = sys.modules.get(dotted)
        prior_path = getattr(prior, "__path__", None) if prior is not None else None
        if prior_path and Path(prior_path[0]).resolve() != Path(bundle_dir).resolve():
            logger.warning(
                "synthetic package %s was already imported from %s; "
                "re-registering it from %s reuses the previously loaded "
                "modules — restart the process to change registry precedence",
                dotted, prior_path[0], bundle_dir,
            )

        # Pre-install the synthetic packages so canonical scripts that call
        # ``load_prompt(__package__, ...)`` resolve their bundle directory
        # via importlib's normal walk-up.
        _ensure_synthetic_package(_SYNTHETIC_ROOT, None)
        _ensure_synthetic_package(f"{_SYNTHETIC_ROOT}.{namespace}", None)
        _ensure_synthetic_package(dotted, bundle_dir)

        info = SkillInfo(
            name=name,
            kind=kind,
            bundle_dir=bundle_dir,
            meta=meta,
            schema=UnitSchema(name=name, description=meta.description),
            namespace=namespace,
            registry=registry,
        )

        self._load_canonical_scripts(info)
        self._load_callable(info)
        self._load_tools_module(info)

        self._skills[name] = info
        logger.debug(
            "Registered %s bundle %r (%d canonical scripts%s%s)",
            kind, name, len(info.canonical_scripts),
            ", callable" if info.module is not None or info.skill_class is not None else "",
            ", tools.py" if info.tools_module is not None else "",
        )

    def _load_canonical_scripts(self, info: SkillInfo) -> None:
        """Import each declared canonical script and introspect its schema."""
        if not info.meta.canonical_scripts:
            return
        # Pre-install the scripts subpackage so each canonical script's
        # ``__package__`` is a real package whose directory points at
        # scripts/. ``load_prompt`` walks up from there to find SKILL.md.
        scripts_dir = info.bundle_dir / "scripts"
        if scripts_dir.is_dir():
            _ensure_synthetic_package(
                f"{_SYNTHETIC_ROOT}.{info.namespace}.{info.name}.scripts", scripts_dir,
            )

        for entry in info.meta.canonical_scripts:
            sp = info.bundle_dir / entry.path
            if not sp.is_file():
                raise ValueError(
                    f"bundle {info.name!r} declares canonical_script "
                    f"{entry.name!r} at {entry.path}, but the file is missing"
                )
            module_name = f"{_SYNTHETIC_ROOT}.{info.namespace}.{info.name}.scripts.{sp.stem}"
            module = self._import_module(sp, module_name)
            script_schema = extract_schema(module, meta=None)
            script_schema.name = f"{info.name}::{entry.name}"
            info.canonical_scripts[entry.name] = ScriptInfo(
                name=entry.name,
                path=sp,
                bundle_relative=entry.path,
                module=module,
                schema=script_schema,
            )

    def _load_callable(self, info: SkillInfo) -> None:
        """Import ``skill.py`` (when present) and wire up its callable.

        Both import paths the former atomic shape supported keep working:
        a module-level ``run(ctx, ...)`` function, or a single class-based
        :class:`Skill` subclass with a ``run`` method.
        """
        skill_path = info.bundle_dir / "skill.py"
        if not skill_path.is_file():
            return
        module = self._import_module(
            skill_path, f"{_SYNTHETIC_ROOT}.{info.namespace}.{info.name}.skill",
        )
        self._wire_callable(info, module, source="skill.py")

    def _wire_callable(self, info: SkillInfo, module: Any, *, source: str) -> None:
        skill_class = _find_skill_class(module)

        if skill_class is not None:
            # SKILL.md is canonical for the bundle's metadata. If the class
            # also declares `meta = SkillMeta(...)` we honor it; otherwise
            # we install the SKILL.md-derived meta on the class so the
            # ``Skill.meta`` ClassVar contract is satisfied.
            klass_meta = getattr(skill_class, "meta", None)
            if klass_meta is None:
                skill_class.meta = info.meta
                klass_meta = info.meta
            run_attr = getattr(skill_class, "run", None)
            if run_attr is None or not callable(run_attr):
                raise ValueError(
                    f"Skill class {skill_class.__name__} in bundle {info.name!r} "
                    f"has no callable run() method"
                )
            module.run = run_attr
            module._meta = klass_meta
            schema_meta = klass_meta
        else:
            run_fn = getattr(module, "run", None)
            if run_fn is None or not callable(run_fn):
                if source == "tools.py":
                    # A plain @tool module — nothing to wire; the tool
                    # registry owns its functions.
                    return
                raise ValueError(
                    f"bundle {info.name!r} {source} has no callable run()"
                )
            module._meta = info.meta
            schema_meta = info.meta

        schema = extract_schema(module, schema_meta)
        schema.name = info.name

        info.module = module
        info.skill_class = skill_class
        info.schema = schema
        logger.debug(
            "Wired callable for bundle %r from %s (%d inputs, %d outputs, %s)",
            info.name, source, len(schema.inputs), len(schema.outputs),
            "class-based" if skill_class else "function-based",
        )

    def _load_tools_module(self, info: SkillInfo) -> None:
        """Import ``tools.py`` (when present) through the synthetic package.

        Importing is the integration point with :mod:`gap_core.tools`: the
        module's ``@tool`` decorators append pending registrations to
        ``gap_core.tools._registry._PENDING_TOOLS``, which
        ``ToolRegistry.discover_pending()`` drains when it builds the flat
        tool catalog. Class-based stateful tools
        (a :class:`Skill` subclass living in ``tools.py`` —
        pi05-libero, tracking-objects) are wired up as the bundle's
        callable too.

        Skipped for out-of-process bundles (``serving.protocol`` other
        than ``in-process``): their ``tools.py`` runs in the bundle's
        own venv (whose deps gap-runtime does NOT have — torch + sam3,
        cuRobo, openpi, …); ``gap.runtime.tool_bundle_manager`` boots
        the server and registers each tool through
        :meth:`ToolRegistry.register_rpc` after the catalog handshake.
        """
        serving = getattr(info.meta, "serving", None)
        if serving is not None and getattr(serving, "protocol", "in-process") != "in-process":
            return
        tools_path = info.bundle_dir / "tools.py"
        if not tools_path.is_file():
            return
        module = self._import_module(
            tools_path, f"{_SYNTHETIC_ROOT}.{info.namespace}.{info.name}.tools",
        )
        info.tools_module = module
        if info.module is None and info.skill_class is None:
            self._wire_callable(info, module, source="tools.py")

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> SkillInfo:
        """Look up a bundle by name."""
        if name not in self._skills:
            available = ", ".join(sorted(self._skills.keys()))
            raise KeyError(f"skill bundle {name!r} not found (available: {available})")
        return self._skills[name]

    def list_skills(
        self,
        category: str | None = None,
        *,
        kind: Literal["tool", "skill", "policy"] | None = None,
    ) -> list[SkillInfo]:
        """List all registered bundles, optionally filtered by category/kind.

        The coordinator's Skills catalog is ``list_skills(kind="skill")``
        plus ``list_skills(kind="policy")`` (both are subgraph-owning);
        tool bundles never own subgraphs and appear only through the flat
        tool catalog.
        """
        skills = list(self._skills.values())
        if category is not None:
            skills = [s for s in skills if s.meta.category == category]
        if kind is not None:
            skills = [s for s in skills if s.kind == kind]
        return skills

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def __len__(self) -> int:
        return len(self._skills)

    # ------------------------------------------------------------------
    # Execution (callable bundles only)
    # ------------------------------------------------------------------

    def call(
        self,
        name: str,
        ctx: Any,
        *,
        skill_instances: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Execute a callable bundle's ``run()``.

        Bundles without a callable cannot be ``call()``-ed directly — their
        canonical scripts are runtime-invoked via ``type: script`` states
        (the executor's script-state path), and plain ``@tool`` functions
        dispatch through the gap.tools registry. Calling one raises
        ValueError.
        """
        info = self.get(name)
        if info.skill_class is None and info.module is None:
            raise ValueError(
                f"skill bundle {name!r} has no callable run(); its canonical "
                f"scripts are runtime-invoked via type:script states and its "
                f"@tool functions dispatch through the tool registry"
            )

        if info.skill_class is not None:
            instance = None
            if skill_instances is not None:
                instance = skill_instances.get(name)
                if instance is None:
                    instance = info.skill_class()
                    skill_instances[name] = instance
            else:
                instance = info.skill_class()
            run_fn = instance.run
            sig = inspect.signature(run_fn)
            accepted = set(sig.parameters.keys()) - {"ctx", "self"}
            filtered = {k: v for k, v in kwargs.items() if k in accepted}
            return run_fn(ctx, **filtered)

        run_fn = info.module.run
        sig = inspect.signature(run_fn)
        accepted = set(sig.parameters.keys()) - {"ctx"}
        filtered = {k: v for k, v in kwargs.items() if k in accepted}
        return run_fn(ctx, **filtered)

    # ------------------------------------------------------------------
    # Documentation generation (matches the legacy _render_skill_doc shape)
    # ------------------------------------------------------------------

    def generate_docs(self, skill_names: list[str] | None = None) -> str:
        """Generate markdown documentation for the LLM prompt.

        If *skill_names* is None, generates docs for all registered bundles.
        Otherwise, only for the listed names.
        """
        if skill_names is not None:
            infos = [self._skills[n] for n in skill_names if n in self._skills]
        else:
            infos = list(self._skills.values())

        if not infos:
            return ""

        parts: list[str] = []
        for info in infos:
            parts.append(self._render_skill_doc(info))
            parts.append("\n---\n")
        return "\n".join(parts)

    def _render_skill_doc(self, info: SkillInfo) -> str:
        """Render markdown documentation for a single bundle.

        For callable bundles, emits the same input/output table the legacy
        registry rendered. Canonical scripts are listed with their schemas;
        tool bundles list their declared ``gap.tools`` summaries (the
        authoritative schemas come from the gap.tools registry).
        """
        meta = info.meta
        lines: list[str] = []

        lines.append(f"### {info.name}\n")
        lines.append(meta.description)
        lines.append("")

        if info.module is not None or info.skill_class is not None:
            schema = info.schema
            if schema.inputs:
                lines.append("**Input:**")
                lines.append("| Field | Type | Required | Default | Description |")
                lines.append("|-------|------|----------|---------|-------------|")
                for fi in schema.inputs.values():
                    req = "yes" if fi.required else "no"
                    default = "—" if fi.required else repr(fi.default)
                    lines.append(
                        f"| {fi.name} | {fi.type_str} | {req} | {default} | {fi.description} |"
                    )
                lines.append("")

            if schema.outputs:
                lines.append("**Output:**")
                lines.append("| Field | Type | Description |")
                lines.append("|-------|------|-------------|")
                for fi in schema.outputs.values():
                    lines.append(f"| {fi.name} | {fi.type_str} | {fi.description} |")
                lines.append("")

        if info.canonical_scripts:
            lines.append("**Canonical scripts:**")
            lines.append("| Logical name | Bundle path | Inputs | Outputs |")
            lines.append("|--------------|-------------|--------|---------|")
            for sname, sinfo in info.canonical_scripts.items():
                in_summary = ", ".join(f"{n}: {f.type_str}" for n, f in sinfo.schema.inputs.items())
                out_summary = ", ".join(f"{n}: {f.type_str}" for n, f in sinfo.schema.outputs.items())
                lines.append(f"| {sname} | `{sinfo.bundle_relative}` | {in_summary} | {out_summary} |")
            lines.append("")

        if meta.tools:
            lines.append("**Tools:**")
            for tname, summary in meta.tools.items():
                lines.append(f"- `{tname}` — {summary}")
            lines.append("")

        if meta.example:
            lines.append("**Example:**")
            lines.append("```json")
            lines.append(json.dumps(meta.example, indent=2))
            lines.append("```")
            lines.append("")

        if meta.errors:
            lines.append("**Errors:**")
            for err in meta.errors:
                lines.append(f"- {err}")
            lines.append("")

        if meta.tips:
            lines.append(f"**Tips:** {meta.tips}")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _import_module(path: Path, module_name: str) -> Any:
        # Serialize concurrent loads of the *same* module name so a second
        # thread can't observe the half-initialized shell that ``exec_module``
        # is still populating. Different module names load in parallel.
        with _get_import_lock(module_name):
            if module_name in sys.modules:
                return sys.modules[module_name]
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ValueError(f"Cannot load skill module: {path}")
            module = importlib.util.module_from_spec(spec)
            # Ensure ``__package__`` points at the synthetic parent so callers
            # using ``load_prompt(__package__, ...)`` resolve correctly.
            if "." in module_name:
                module.__package__ = module_name.rsplit(".", 1)[0]
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                # Don't leave a half-loaded shell behind for the next caller.
                sys.modules.pop(module_name, None)
                raise
            return module


def _find_skill_class(module: Any) -> type | None:
    """Return the first :class:`Skill` subclass exported by *module*, if any."""
    candidates = []
    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        obj = getattr(module, attr_name, None)
        if not isinstance(obj, type):
            continue
        if obj is Skill:
            continue
        try:
            if not issubclass(obj, Skill):
                continue
        except TypeError:
            continue
        if getattr(obj, "__module__", None) != module.__name__:
            continue
        candidates.append(obj)
    if len(candidates) > 1:
        raise ValueError(
            f"Module {module.__name__!r} exports multiple Skill subclasses "
            f"({[c.__name__ for c in candidates]}); only one per module is supported."
        )
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Compatibility constraint ("requires gap>=X.Y")
# ---------------------------------------------------------------------------

_COMPAT_RE = re.compile(r"gap\s*>=\s*(\d+(?:\.\d+)*)")


def _version_tuple(version: str) -> tuple[int, ...]:
    """``"0.1.0.dev0"`` → ``(0, 1, 0)`` — leading numeric components only."""
    parts: list[int] = []
    for piece in version.split("."):
        if not piece.isdigit():
            break
        parts.append(int(piece))
    return tuple(parts)


def _check_compatibility(compatibility: str, bundle: str) -> None:
    """Warn (never raise) when a bundle's ``compatibility:`` is unsatisfied.

    Accepts the spec form ``"requires gap>=X.Y"``. Deliberately
    packaging-free: a simple regex parse plus tuple comparison — bundles
    pinning anything fancier should rely on pip metadata instead.
    """
    if not compatibility:
        return
    m = _COMPAT_RE.search(compatibility)
    if m is None:
        logger.warning(
            "skill bundle %r has unparseable compatibility %r "
            "(expected something like 'requires gap>=0.1')",
            bundle, compatibility,
        )
        return
    import gap

    required = _version_tuple(m.group(1))
    installed = _version_tuple(gap.__version__)
    width = max(len(required), len(installed))
    if installed + (0,) * (width - len(installed)) < required + (0,) * (width - len(required)):
        logger.warning(
            "skill bundle %r requires gap>=%s but gap %s is installed; "
            "the bundle may not work as documented",
            bundle, m.group(1), gap.__version__,
        )
