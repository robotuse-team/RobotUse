"""gap.skills — runtime-side skill registry (loader / discovery / validation).

After the gap-core split this module holds ONLY the runtime symbols. The
skill-authoring dataclasses (Skill, SkillMeta, Param, Serving, …) live in
``gap_core.skills`` and bundle authors import them from there directly::

    from gap_core.skills import Skill, SkillMeta, Param, Serving

Runtime callers (gap.cli, gap.agent, gap.viz, …) import the loader and
registry plumbing from here::

    from gap.skills import load_skills, SkillsRegistry, parse_skill_md
"""

from __future__ import annotations

from ._meta_from_skill_md import parse_skill_md
from ._prompt_loader import load_prompt
from ._registry import ScriptInfo, SkillInfo, SkillsRegistry, load_skills
from .discovery import find_skills_path, looks_like_skills_checkout
from .registries import (
    RegistrySet,
    RegistrySpec,
    as_registry_paths,
    load_registry_set,
    resolve_registries,
)

__all__ = [
    "RegistrySet",
    "RegistrySpec",
    "ScriptInfo",
    "SkillInfo",
    "SkillsRegistry",
    "as_registry_paths",
    "find_skills_path",
    "load_prompt",
    "load_registry_set",
    "load_skills",
    "looks_like_skills_checkout",
    "parse_skill_md",
    "resolve_registries",
]
