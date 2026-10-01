"""gap_core.skills — skill-authoring metadata surface.

The dataclasses and base class a bundle author needs to declare a skill,
without the runtime registry / loader (that lives in `gap.skills` under
the graph-as-policy runtime distribution).

Skill authors import from here::

    from gap_core.skills import Skill, SkillMeta, Param, Serving
"""

from __future__ import annotations

from .meta import (
    CanonicalScript,
    ExampleDoc,
    Param,
    ReferenceDoc,
    Serving,
    Skill,
    SkillMeta,
    SkillRequires,
)

__all__ = [
    "CanonicalScript",
    "ExampleDoc",
    "Param",
    "ReferenceDoc",
    "Serving",
    "Skill",
    "SkillMeta",
    "SkillRequires",
]
