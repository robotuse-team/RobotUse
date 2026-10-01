"""AgentSpec parser and AgentRegistry.

Each ``<name>.md`` in ``gap/agent/prompts/`` is one codegen subagent. The
parser splits out YAML frontmatter (between two ``---`` lines), captures
the body, and resolves ``includes:`` paths against that directory.

The frontmatter schema is intentionally small:

```yaml
---
name: coordinator
description: >
  Decompose a robotic task into a topology of subgraphs ...
tools: [read_skill_reference, read_skill_example, ...]
includes: [_workflow_spec.md]
model: gemini-3.1-flash-lite-preview   # optional
---
```

The body is everything after the closing ``---``. The prompt assembler
concatenates the body, the resolved includes, plus per-call context to
produce the final system prompt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_AGENTS_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass
class AgentSpec:
    """Parsed subagent definition: frontmatter + body + resolved includes."""

    name: str
    description: str
    body: str
    """Raw body markdown (everything after the closing ``---``)."""

    tools: tuple[str, ...] = ()
    """Codegen tool names this agent is allowed to call. Whitelist."""

    includes: tuple[str, ...] = ()
    """Prompts-dir-relative paths to shared docs (resolved on read)."""

    model: str | None = None
    """Optional override of the LLM model for this agent."""

    source_path: Path | None = None


@dataclass
class AgentRegistry:
    _agents: dict[str, AgentSpec] = field(default_factory=dict)

    def discover(self, agents_dir: str | Path | None = None) -> None:
        """Walk *agents_dir* and register every ``<name>.md`` it contains.

        Files starting with ``_`` are shared docs and are skipped.
        """
        agents_dir = Path(agents_dir) if agents_dir else _AGENTS_DIR
        if not agents_dir.is_dir():
            logger.warning("Agents directory does not exist: %s", agents_dir)
            return
        for md in sorted(agents_dir.glob("*.md")):
            if md.name.startswith("_"):
                continue
            try:
                spec = parse_agent_md(md)
                self._agents[spec.name] = spec
            except Exception:
                logger.warning("Failed to register agent %s", md, exc_info=True)

    def get(self, name: str) -> AgentSpec:
        if name not in self._agents:
            available = ", ".join(sorted(self._agents.keys()))
            raise KeyError(f"agent {name!r} not found (available: {available})")
        return self._agents[name]

    def list_agents(self) -> list[AgentSpec]:
        return list(self._agents.values())

    def __contains__(self, name: str) -> bool:
        return name in self._agents

    def __len__(self) -> int:
        return len(self._agents)


def parse_agent_md(path: Path) -> AgentSpec:
    """Parse a single ``<agent>.md`` file into an :class:`AgentSpec`."""
    text = path.read_text(encoding="utf-8")
    fm, body = _split(text)
    data = yaml.safe_load(fm) or {}
    name = data.get("name") or path.stem
    if name != path.stem:
        raise ValueError(
            f"agent file {path}: frontmatter name {name!r} does not match "
            f"file stem {path.stem!r}"
        )
    description = data.get("description") or ""
    if not description:
        raise ValueError(f"agent file {path}: missing required `description`")

    tools = tuple(_str_list(data.get("tools")))
    includes = tuple(_str_list(data.get("includes")))
    model = data.get("model")

    return AgentSpec(
        name=name,
        description=description.strip(),
        body=body,
        tools=tools,
        includes=includes,
        model=model,
        source_path=path,
    )


def _split(text: str) -> tuple[str, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("agent file must begin with `---` frontmatter delimiter")
    end = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end == -1:
        raise ValueError("agent file frontmatter is missing its closing `---`")
    return "\n".join(lines[1:end]), "\n".join(lines[end + 1:]).strip("\n")


def _str_list(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, str):
        return [v]
    raise ValueError(f"expected list[str], got {type(v).__name__}")


def default_agent_registry() -> AgentRegistry:
    reg = AgentRegistry()
    reg.discover()
    return reg


def read_include(name: str) -> str:
    """Read a shared doc by name (e.g. ``_workflow_spec.md``)."""
    p = _AGENTS_DIR / name
    if not p.is_file():
        raise FileNotFoundError(f"shared doc {name!r} not found at {p}")
    # Force UTF-8 — Path.read_text() defaults to the locale codec, which
    # is ASCII on POSIX shells, and these include files contain em dashes.
    return p.read_text(encoding="utf-8")
