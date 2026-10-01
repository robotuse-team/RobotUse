"""Locate a skill registry checkout without hardcoded relative paths.

Historical single-registry surface. :func:`find_skills_path` resolves
*one* checkout — the highest-precedence registry — and is kept as a
back-compat shim over :func:`gap.skills.registries.resolve_registries`,
which is the full multi-registry resolver (``--skills`` flags >
``$GAP_SKILLS_PATH`` list > project ``[tool.gap]`` > user config >
sibling auto-discovery). New code should use ``resolve_registries`` /
``load_registry_set``; this module keeps the legacy semantics
(explicit > env > sibling walk) bit-for-bit for existing callers.

A directory counts as a registry checkout when it has a ``tools/``
and/or ``skills/`` bundle root with at least one ``SKILL.md`` bundle.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, overload

logger = logging.getLogger(__name__)

#: Environment variable naming skill registry checkout root(s) —
#: ``os.pathsep``-separated; a single path keeps its historical meaning.
GAP_SKILLS_PATH_ENV = "GAP_SKILLS_PATH"

#: Directory name of the sibling checkout the auto-discovery walk looks for.
_SIBLING_DIR_NAME = "open-robot-skills"


def looks_like_skills_checkout(path: str | Path) -> bool:
    """True iff *path* is a directory with at least one populated bundle root.

    A skill registry has a ``tools/`` and/or a ``skills/`` directory, at
    least one of which contains a bundle (a subdirectory with a
    ``SKILL.md``). The canonical open-robot-skills checkout has both
    roots; a third-party registry may ship only one.
    """
    p = Path(path)
    if not p.is_dir():
        return False
    for folder in ("tools", "skills"):
        root = p / folder
        if not root.is_dir():
            continue
        try:
            has_bundle = any(
                (child / "SKILL.md").is_file()
                for child in root.iterdir()
                if child.is_dir() and not child.name.startswith("_")
            )
        except OSError:
            continue
        if has_bundle:
            return True
    return False


@overload
def find_skills_path(
    explicit: str | Path | None = ...,
    *,
    required: Literal[True],
    search_from: Iterable[str | Path] | None = ...,
) -> Path: ...


@overload
def find_skills_path(
    explicit: str | Path | None = ...,
    *,
    required: bool = ...,
    search_from: Iterable[str | Path] | None = ...,
) -> Path | None: ...


def find_skills_path(
    explicit: str | Path | None = None,
    *,
    required: bool = False,
    search_from: Iterable[str | Path] | None = None,
) -> Path | None:
    """Resolve ONE skill registry checkout (the highest-precedence one).

    Back-compat shim over :func:`gap.skills.registries.resolve_registries`
    with the historical semantics: explicit argument > ``$GAP_SKILLS_PATH``
    > sibling auto-discovery (project/user registry config is *not*
    consulted — callers that should see configured registries use
    ``resolve_registries`` directly).

    Args:
        explicit: An explicitly-provided path (CLI flag / function
            argument). Returned as-is (resolved) when given — explicit
            always wins.
        required: When True, raise :class:`FileNotFoundError` (with a
            message listing everything that was tried) instead of
            returning ``None``.
        search_from: Override the sibling-walk starting points (defaults
            to the installed ``gap`` package directory and the current
            working directory). Exposed for tests.

    Returns:
        The checkout path, or ``None`` when nothing was found and
        ``required`` is False.

    Raises:
        FileNotFoundError: nothing found and ``required=True``.
        ValueError: ``GAP_SKILLS_PATH`` is set but one of its entries does
            not point at a registry checkout (an explicitly-set env var
            must be valid — silently falling back would mask typos).
    """
    from .registries import resolve_registries

    registry_set = resolve_registries(
        explicit,
        required=required,
        search_from=search_from,
        include_config=False,
    )
    primary = registry_set.primary()
    if primary is None:
        return None
    if len(registry_set) > 1:
        logger.debug(
            "find_skills_path: %d registries active; returning the "
            "highest-precedence one (%s)", len(registry_set), primary.path,
        )
    return primary.path
