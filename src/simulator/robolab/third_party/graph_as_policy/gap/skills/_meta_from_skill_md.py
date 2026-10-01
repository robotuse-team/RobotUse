"""Parse a SKILL.md file into a :class:`SkillMeta`.

Frontmatter is YAML between two ``---`` lines at the top of the file.
The body is everything after the closing ``---``. Body is captured into
``meta.body`` for the prompt assembler to emit verbatim.

The frontmatter follows the Agent Skills spec at the top level — ``name``
(must equal the bundle directory name; ≤64 chars, lowercase-hyphen),
``description`` (≤1024 chars), optional ``license``, ``compatibility``,
``metadata`` — with **all gap extensions nested under one ``gap:`` key**
so spec fields are never overloaded::

    ---
    name: perceiving-objects
    description: Detect and localize a named object ... Use when ...
    compatibility: requires gap>=0.1
    metadata: {category: perception, tags: [perception, dino]}
    gap:
      allowed_tools: [robot.get_observation, grounding-dino.detect]
      exit_conditions: {found: "...", not_found: "..."}
      produces_outputs: {"<name>_obb": OrientedBoundingBox}
      canonical_scripts:
        - perceive_dino_vlm: scripts/perceive_dino_vlm.py
      prompts: {vlm_select_box: prompts/vlm_select_box.md}
    ---

The legacy dev-tree keys (``runtime.shape``, ``composes``, top-level
``allowed-tools``/``exit_conditions``/...) are rejected with a migration
hint — the tools/ vs skills/ folder split replaces ``shape``, and
``composes`` is gone.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from gap_core.skills.meta import (
    CanonicalScript,
    ExampleDoc,
    ReferenceDoc,
    Serving,
    SkillMeta,
    SkillRequires,
)

#: Agent Skills spec: lowercase letters/digits with single hyphens, ≤64 chars.
_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_MAX_NAME_LEN = 64
_MAX_DESCRIPTION_LEN = 1024

#: gap extension keys that must live under ``gap:`` — finding one at the top
#: level (or a removed legacy key) is a migration error, not silent fallback.
_GAP_ONLY_KEYS = frozenset({
    "allowed_tools", "allowed-tools", "exit_conditions", "produces_outputs",
    "required_inputs", "canonical_scripts", "prompts", "references",
    "examples", "errors", "tips", "hard_rules", "streaming", "tools",
    "requires", "serving",
})
_REMOVED_KEYS = frozenset({"runtime", "shape", "composes", "category", "tags", "contract"})


def parse_skill_md(path: Path) -> SkillMeta:
    """Read SKILL.md and produce a :class:`SkillMeta`.

    The bundle directory is the SKILL.md file's parent. Field semantics:

    - Required: ``name`` (== bundle directory name), ``description``.
    - ``params`` and ``outputs`` are NOT in frontmatter — they come from
      Python introspection of the bundle's callables (or canonical scripts).

    Raises:
        FileNotFoundError: if ``path`` doesn't exist.
        ValueError: if the frontmatter is malformed (no ``---`` delimiters,
            missing/mismatched ``name``, over-long ``description``, legacy
            legacy keys, gap extensions outside the ``gap:`` block).
    """
    if not path.is_file():
        raise FileNotFoundError(f"SKILL.md not found at {path}")
    text = path.read_text(encoding="utf-8")
    frontmatter, body = _split(text)
    data = yaml.safe_load(frontmatter) or {}
    if not isinstance(data, dict):
        raise ValueError(f"SKILL.md frontmatter at {path} must be a YAML mapping")
    return _meta_from_dict(data, body=body, bundle_dir=path.parent)


def _split(text: str) -> tuple[str, str]:
    """Return ``(frontmatter_yaml, body_markdown)``."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("SKILL.md must begin with a `---` frontmatter delimiter")
    end = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end == -1:
        raise ValueError("SKILL.md frontmatter is missing its closing `---`")
    fm = "\n".join(lines[1:end])
    body = "\n".join(lines[end + 1:]).strip("\n")
    return fm, body


def _meta_from_dict(data: dict, *, body: str, bundle_dir: Path) -> SkillMeta:
    _reject_misplaced_keys(data, bundle_dir)

    name = str(data.get("name", "") or "")
    if not name:
        raise ValueError(f"SKILL.md at {bundle_dir} has no `name` field")
    if name != bundle_dir.name:
        raise ValueError(
            f"SKILL.md at {bundle_dir} declares name {name!r}, which must "
            f"equal its bundle directory name {bundle_dir.name!r}"
        )
    if len(name) > _MAX_NAME_LEN or not _NAME_RE.match(name):
        raise ValueError(
            f"SKILL.md `name: {name}` is not a valid bundle name "
            f"(≤{_MAX_NAME_LEN} chars, lowercase letters/digits/hyphens)"
        )

    description = data.get("description", "") or ""
    if not description:
        raise ValueError(f"SKILL.md at {bundle_dir} has no `description` field")
    if len(description) > _MAX_DESCRIPTION_LEN:
        raise ValueError(
            f"SKILL.md at {bundle_dir} `description` is {len(description)} chars; "
            f"the Agent Skills spec caps it at {_MAX_DESCRIPTION_LEN}"
        )

    metadata = data.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a mapping")

    gap_ext = data.get("gap") or {}
    if not isinstance(gap_ext, dict):
        raise ValueError("the `gap:` frontmatter key must be a mapping")

    references = _list_of_refs(gap_ext.get("references"))
    examples = _list_of_examples(gap_ext.get("examples"))
    canonical_scripts = _list_of_canonical_scripts(gap_ext.get("canonical_scripts"))
    hard_rules = gap_ext.get("hard_rules") or []
    if not isinstance(hard_rules, list):
        raise ValueError("gap.hard_rules must be a list")

    prompts = gap_ext.get("prompts") or {}
    if isinstance(prompts, dict):
        prompts = {str(k): str(v) for k, v in prompts.items()}
    else:
        raise ValueError("gap.prompts must be a mapping {logical_name: path}")

    return SkillMeta(
        description=description,
        name=name,
        license=str(data.get("license", "") or ""),
        compatibility=str(data.get("compatibility", "") or ""),
        metadata=dict(metadata),
        allowed_tools=_str_list(gap_ext.get("allowed_tools")),
        category=str(metadata.get("category", "") or ""),
        tags=_str_list(metadata.get("tags")),
        examples=examples,
        errors=_str_list(gap_ext.get("errors")),
        tips=gap_ext.get("tips", "") or "",
        exit_conditions=dict(gap_ext.get("exit_conditions") or {}),
        produces_outputs=dict(gap_ext.get("produces_outputs") or {}),
        required_inputs=dict(gap_ext.get("required_inputs") or {}),
        hard_rules=list(hard_rules),
        canonical_scripts=canonical_scripts,
        prompts=prompts,
        references=references,
        streaming=bool(gap_ext.get("streaming", False)),
        tools=_tools_map(gap_ext.get("tools")),
        requires=_requires(gap_ext["requires"]) if "requires" in gap_ext else None,
        serving=_serving(gap_ext["serving"]) if "serving" in gap_ext else None,
        bundle_dir=bundle_dir,
        body=body,
    )


def _reject_misplaced_keys(data: dict, bundle_dir: Path) -> None:
    removed = sorted(k for k in data if k in _REMOVED_KEYS)
    if removed:
        raise ValueError(
            f"SKILL.md at {bundle_dir} uses removed legacy frontmatter keys "
            f"{removed}: `runtime.shape`/`composes` are gone (the tools/ vs "
            f"skills/ folder conveys the kind) and `category`/`tags` move "
            f"under `metadata:`"
        )
    misplaced = sorted(k for k in data if k in _GAP_ONLY_KEYS)
    if misplaced:
        raise ValueError(
            f"SKILL.md at {bundle_dir} has gap extension keys {misplaced} at "
            f"the top level; nest them under the `gap:` key"
        )


_REQUIRES_KEYS = frozenset({"gpu", "env", "env_any", "weights"})


def _requires(v: Any) -> SkillRequires:
    """``gap.requires`` — operational requirements for ``gap check``.

    A bare ``requires:`` (YAML null) or ``requires: {}`` both mean "the
    bundle explicitly declares no special requirements". Unknown subkeys
    are rejected — a typo'd key would otherwise silently disable a probe.
    """
    if v is None:
        return SkillRequires()
    if not isinstance(v, dict):
        raise ValueError("gap.requires must be a mapping")
    unknown = sorted(set(v) - _REQUIRES_KEYS)
    if unknown:
        raise ValueError(
            f"gap.requires has unknown keys {unknown} "
            f"(allowed: {sorted(_REQUIRES_KEYS)})"
        )
    env = _str_list(v.get("env"))
    env_any = _str_list(v.get("env_any"))
    for label, values in (("env", env), ("env_any", env_any)):
        for entry in values:
            if not entry.strip():
                raise ValueError(
                    f"gap.requires.{label} entries must be non-empty "
                    f"environment variable names"
                )
    return SkillRequires(
        gpu=bool(v.get("gpu", False)),
        env=env,
        env_any=env_any,
        weights=bool(v.get("weights", False)),
    )


_SERVING_KEYS = frozenset({
    "command", "protocol", "env", "requires_gpu", "weights_uri",
})
_SERVING_PROTOCOLS = frozenset({"websocket", "stdio-msgpack", "in-process"})


def _serving(v: Any) -> Serving:
    """``gap.serving`` — out-of-process launch recipe.

    Required for ``kind='policy'`` bundles (the launcher boots one server
    per referenced preset) and optional for ``kind='tool'`` (opts into
    the out-of-process RPC path). ``command`` must be a list[str] — never
    a shell string — so spawn is shell-free and the bundle's own venv
    activates via ``uv run --project <bundle_dir>`` at launch time.
    """
    if v is None:
        raise ValueError(
            "gap.serving must be a mapping with at minimum `command:` "
            "(a non-empty list[str]); to declare in-process dispatch, "
            "omit the block entirely"
        )
    if not isinstance(v, dict):
        raise ValueError("gap.serving must be a mapping")
    unknown = sorted(set(v) - _SERVING_KEYS)
    if unknown:
        raise ValueError(
            f"gap.serving has unknown keys {unknown} "
            f"(allowed: {sorted(_SERVING_KEYS)})"
        )
    command = v.get("command")
    if not isinstance(command, list) or not command:
        raise ValueError(
            "gap.serving.command must be a non-empty list[str] (NOT a "
            "shell string — gap spawns argv directly, no shell expansion)"
        )
    cmd = [str(arg) for arg in command]
    protocol = str(v.get("protocol", "in-process"))
    if protocol not in _SERVING_PROTOCOLS:
        raise ValueError(
            f"gap.serving.protocol={protocol!r} is not one of "
            f"{sorted(_SERVING_PROTOCOLS)}"
        )
    env_raw = v.get("env") or {}
    if not isinstance(env_raw, dict):
        raise ValueError("gap.serving.env must be a mapping of str -> str")
    env = {str(k): str(val) for k, val in env_raw.items()}
    return Serving(
        command=cmd,
        protocol=protocol,
        env=env,
        requires_gpu=bool(v.get("requires_gpu", False)),
        weights_uri=str(v.get("weights_uri", "") or ""),
    )


def _str_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, str):
        return [v]
    raise ValueError(f"expected list[str], got {type(v).__name__}")


def _list_of_refs(v: Any) -> list[ReferenceDoc]:
    if v is None:
        return []
    out: list[ReferenceDoc] = []
    for entry in v:
        if isinstance(entry, dict):
            out.append(ReferenceDoc(title=str(entry.get("title", "")), path=str(entry.get("path", ""))))
        elif isinstance(entry, str):
            out.append(ReferenceDoc(title=entry, path=entry))
        else:
            raise ValueError(f"references entry must be dict or str, got {type(entry).__name__}")
    return out


def _list_of_examples(v: Any) -> list[ExampleDoc]:
    if v is None:
        return []
    out: list[ExampleDoc] = []
    for entry in v:
        if isinstance(entry, dict):
            out.append(ExampleDoc(title=str(entry.get("title", "")), path=str(entry.get("path", ""))))
        elif isinstance(entry, str):
            out.append(ExampleDoc(title=entry, path=entry))
        else:
            raise ValueError(f"examples entry must be dict or str, got {type(entry).__name__}")
    return out


def _list_of_canonical_scripts(v: Any) -> list[CanonicalScript]:
    if v is None:
        return []
    out: list[CanonicalScript] = []
    if isinstance(v, list):
        for entry in v:
            if isinstance(entry, dict) and len(entry) == 1:
                # YAML form: `- name: path`
                name, path = next(iter(entry.items()))
                out.append(CanonicalScript(name=str(name), path=str(path)))
            elif isinstance(entry, dict) and "name" in entry and "path" in entry:
                out.append(CanonicalScript(name=str(entry["name"]), path=str(entry["path"])))
            else:
                raise ValueError(f"canonical_scripts entry must map a single name to a path, got {entry!r}")
    elif isinstance(v, dict):
        for name, path in v.items():
            out.append(CanonicalScript(name=str(name), path=str(path)))
    else:
        raise ValueError("gap.canonical_scripts must be a list or dict")
    return out


def _tools_map(v: Any) -> dict[str, str]:
    """``gap.tools`` — list of ``{name: summary}`` entries (or a mapping)."""
    if v is None:
        return {}
    out: dict[str, str] = {}
    if isinstance(v, list):
        for entry in v:
            if isinstance(entry, dict) and len(entry) == 1:
                # YAML form: `- name: summary`
                name, summary = next(iter(entry.items()))
                out[str(name)] = str(summary)
            elif isinstance(entry, dict) and "name" in entry:
                out[str(entry["name"])] = str(entry.get("summary", ""))
            else:
                raise ValueError(f"gap.tools entry must map a single name to a summary, got {entry!r}")
    elif isinstance(v, dict):
        for name, summary in v.items():
            out[str(name)] = str(summary)
    else:
        raise ValueError("gap.tools must be a list or dict")
    return out
