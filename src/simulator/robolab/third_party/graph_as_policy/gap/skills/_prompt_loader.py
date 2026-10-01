"""Load Jinja-style prompt templates bundled inside skill bundles.

Authors put the template in ``<bundle_dir>/prompts/<name>.md`` (under any
registered bundle, typically ``<open-robot-skills>/skills/<bundle>/prompts/``) and
call::

    from gap.skills import load_prompt

    text = load_prompt(__package__, "vlm_select_box",
                       n=n, label_list=", ".join(labels),
                       object_name=object_name, object_description="")

This works for canonical bundle scripts because the registry registers
each script under a synthetic package
(``gap_skills.<namespace>.<bundle>.scripts.<stem>``) whose parent
``gap_skills.<namespace>.<bundle>`` is rooted at the bundle directory.
``load_prompt`` walks up from the calling module to find ``SKILL.md`` and
treats that directory as the bundle root.

It does NOT work for ad-hoc LLM-emitted scripts (``type: script`` states
that aren't backed by a registered bundle) — those have no SKILL.md and
no prompts directory to load from.

Templates may use ``{{ var }}`` substitutions and ``{% if var %}...{% endif %}``
conditionals. The implementation uses str.format with a small extension for
optional blocks; jinja is not pulled in to keep the dependency small.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Any

# Match {% if var %}...{% endif %} blocks (single-pass, non-nested).
_IF_BLOCK = re.compile(
    r"\{%\s*if\s+([A-Za-z_][A-Za-z0-9_]*)\s*%\}(.*?)\{%\s*endif\s*%\}",
    re.DOTALL,
)
# Match {{ var }} substitutions.
_SUBST = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def load_prompt(skill_package: str, name: str, **vars: Any) -> str:
    """Load and render the prompt template ``<bundle>/prompts/<name>.md``.

    Args:
        skill_package: ``__package__`` of the calling module. May be the
            bundle root (e.g. ``"gap_skills.skills.perceiving-objects"``) or
            a sub-package like
            ``"gap_skills.skills.perceiving-objects.scripts"`` — the loader
            walks up until it finds a directory containing ``SKILL.md`` and
            treats that directory as the bundle root.
        name: Logical prompt name (matches the SKILL.md frontmatter
            ``gap.prompts:`` mapping key, or the bare filename without
            ``.md``).
        **vars: Substitution values for ``{{ var }}`` and ``{% if var %}``.

    Returns:
        Rendered prompt text.
    """
    bundle_dir = _bundle_dir(skill_package)
    candidate = bundle_dir / "prompts" / f"{name}.md"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Prompt {name!r} not found at {candidate}. "
            f"Available prompts: {sorted(p.stem for p in (bundle_dir / 'prompts').glob('*.md'))}"
        )
    raw = candidate.read_text(encoding="utf-8")
    body = _strip_frontmatter(raw)
    return _render(body, vars)


def _bundle_dir(skill_package: str) -> Path:
    """Return the bundle root for *skill_package*.

    The bundle root is the nearest ancestor directory containing ``SKILL.md``.
    Resolves the calling module's on-disk location via ``__file__`` when
    available, falling back to ``__path__`` for synthetic namespace packages
    (e.g. ``gap_skills.<ns>.<bundle>.scripts`` registered by
    :func:`gap.skills._registry._ensure_synthetic_package`).
    """
    module = importlib.import_module(skill_package)
    file = getattr(module, "__file__", None)
    if file is not None:
        here = Path(file).resolve().parent
    else:
        path_attr = getattr(module, "__path__", None)
        if not path_attr:
            raise ValueError(
                f"Cannot locate bundle directory for {skill_package!r}: "
                f"module has neither __file__ nor __path__"
            )
        here = Path(next(iter(path_attr))).resolve()
    for ancestor in (here, *here.parents):
        if (ancestor / "SKILL.md").is_file():
            return ancestor
    raise FileNotFoundError(
        f"No SKILL.md found in {here} or any ancestor; "
        f"cannot locate bundle root for {skill_package!r}"
    )


def _strip_frontmatter(text: str) -> str:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[i + 1:]).lstrip("\n")
    return text


def _render(body: str, vars: dict[str, Any]) -> str:
    def _if_repl(m: re.Match) -> str:
        var = m.group(1)
        inner = m.group(2)
        return inner if vars.get(var) else ""

    body = _IF_BLOCK.sub(_if_repl, body)

    def _subst_repl(m: re.Match) -> str:
        var = m.group(1)
        if var not in vars:
            raise KeyError(f"prompt template references undefined variable {var!r}")
        return str(vars[var])

    return _SUBST.sub(_subst_repl, body)
