"""Skill metadata dataclasses — populated from SKILL.md frontmatter.

``SkillMeta`` is not authored as a Python literal. The
:class:`SkillsRegistry` parses each bundle's ``SKILL.md`` frontmatter into
a ``SkillMeta`` instance at discovery time. The frontmatter is the Agent
Skills spec core (``name``, ``description``, ``license``, ``compatibility``,
``metadata``) plus all gap extensions nested under one ``gap:`` key
(``allowed_tools``, ``exit_conditions``, ``produces_outputs``,
``required_inputs``, ``canonical_scripts``, ``prompts``, ``references``,
``examples``, ``errors``, ``tips``, ``hard_rules``, ``streaming``,
``tools``).

The runtime contract is stable: the prompt assembler and the validator
read the same field names regardless of which bundle authored them. The
former dev-tree ``runtime.shape`` field is gone — :attr:`SkillMeta.kind`
(``"tool"`` vs ``"skill"``, conveyed by the bundle's folder) replaces it,
and ``composes`` (gRPC service FQNs) is dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

_MISSING = object()  # sentinel — actual default comes from run() signature


@dataclass
class Param:
    """Description for a single skill parameter.

    The *default* field is a sentinel by default — the registry reads the
    actual default from the ``run()`` signature. Only set it here to
    **override** the display value in generated docs.
    """

    description: str
    default: Any = _MISSING


@dataclass
class ReferenceDoc:
    """A long-form reference doc bundled in ``references/``."""

    title: str
    path: str  # relative to bundle root


@dataclass
class ExampleDoc:
    """A bundled example invocation under ``examples/``."""

    title: str
    path: str  # relative to bundle root


@dataclass
class CanonicalScript:
    """A canonical script the subgraph_agent may emit as a ``type: script`` state."""

    name: str       # logical name (e.g. "perceive_dino_vlm")
    path: str       # relative to bundle root (e.g. "scripts/perceive_dino_vlm.py")


@dataclass
class SkillRequires:
    """Operational requirements a bundle declares for ``gap check``.

    The atomic capability question is "can this bundle run *here*?" —
    deps importable (probed automatically), plus whatever this block
    declares. Keep it small: four keys, all optional. Authored in
    SKILL.md frontmatter under ``gap.requires``::

        gap:
          requires: {gpu: true, env: [MY_API_KEY], env_any: [], weights: true}
    """

    gpu: bool = False
    """Needs a local NVIDIA GPU (``gap check`` probes via nvidia-smi)."""

    env: list[str] = field(default_factory=list)
    """Environment variables that must ALL be set and non-empty."""

    env_any: list[str] = field(default_factory=list)
    """Environment variables of which AT LEAST ONE must be set."""

    weights: bool = False
    """Downloads model weights on first use. ``gap check`` reports the
    cache state via the bundle's optional ``weights_cached() -> bool | None``
    hook (filesystem checks only — never downloads); without the hook the
    state is reported as unknown."""


@dataclass
class Serving:
    """How gap launches a bundle's out-of-process server.

    Populated from SKILL.md frontmatter ``gap.serving``. Required for
    ``kind='policy'`` bundles (the launcher spawns one server per
    referenced policy preset); optional for ``kind='tool'`` bundles
    (declaring it opts the bundle into the out-of-process RPC path
    instead of in-process ``@tool`` dispatch).

    Example (policy)::

        gap:
          serving:
            command: ["python", "-m", "pi05_libero.server",
                      "--policy.config=pi05_libero",
                      "--port", "{port}"]
            protocol: websocket

    Example (tool)::

        gap:
          serving:
            command: ["python", "-m", "gap_tool_server", "--bundle", "sam3"]
            protocol: stdio-msgpack

    The launcher prepends ``uv run --project <bundle_dir> --`` so the
    bundle runs in its own venv (see ``gap skills install``). ``{port}``
    in any ``command`` element is substituted with an OS-allocated free
    port at spawn time (websocket protocol only)."""

    command: list[str]
    """Argv (NOT a shell string). One ``{port}`` placeholder is allowed
    for ``protocol: websocket``."""

    protocol: str = "in-process"
    """``websocket`` (policies), ``stdio-msgpack`` (out-of-process tools),
    or ``in-process`` (the default — equivalent to omitting the block;
    kept as an explicit escape hatch for tools that want to declare the
    block for documentation while staying in-process)."""

    env: dict[str, str] = field(default_factory=dict)
    """Extra environment variables passed to the spawned process.
    Merged over ``os.environ`` at spawn time."""

    requires_gpu: bool = False
    """The server needs a GPU at the spawn host. Surfaced by ``gap check``;
    independent of ``gap.requires.gpu`` (which gates the client-side import)."""

    weights_uri: str = ""
    """Where the server downloads weights from on first run (informational —
    the server, not gap, performs the fetch)."""


@dataclass
class SkillMeta:
    """Structured metadata for a skill bundle.

    Built from SKILL.md frontmatter at registration time. All fields are
    optional except *description*. Class-based skills may also pass
    ``meta = SkillMeta(...)`` directly via ``Skill.meta`` if they want to
    override frontmatter-derived metadata, though the SKILL.md path is
    canonical.
    """

    description: str

    # Agent Skills spec core fields
    name: str = ""
    """Frontmatter ``name``; must equal the bundle directory name."""

    license: str = ""
    compatibility: str = ""
    """Version constraint, e.g. ``"requires gap>=0.1"``. The loader logs a
    warning when the installed gap version does not satisfy it."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Free-form spec ``metadata`` mapping (category/tags/author/...)."""

    # Bundle kind — set by the registry from the bundle's folder, never
    # from frontmatter. Replaces the legacy ``runtime.shape`` field.
    kind: Literal["tool", "skill", "policy"] = "skill"
    """``tool`` bundles (under ``tools/``) expose model-backed callables via
    ``tools.py``; ``skill`` bundles (under ``skills/``) own subgraphs and ship
    canonical scripts (and *may* also expose a callable via ``tools.py``);
    ``policy`` bundles (under ``policies/``) drive one learned-policy
    checkpoint and own their server's launch recipe via :attr:`serving`."""

    # Schema-related (populated from Python introspection at the registry, not frontmatter)
    params: dict[str, Param] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)

    # gap: extension fields
    allowed_tools: list[str] = field(default_factory=list)

    category: str = ""
    tags: list[str] = field(default_factory=list)

    example: dict | None = None
    examples: list[ExampleDoc] = field(default_factory=list)

    errors: list[str] = field(default_factory=list)
    tips: str = ""

    exit_conditions: dict[str, str] = field(default_factory=dict)
    """``{end_state_name: meaning}``. Declared by the skill's SKILL.md
    frontmatter; the coordinator wires conditional edges against these."""

    produces_outputs: dict[str, str] = field(default_factory=dict)
    """``{output_name: type_name}``. Subgraph-level outputs the skill expects
    callers to bind. Names may contain ``<name>`` for substitution."""

    required_inputs: dict[str, str] = field(default_factory=dict)
    """``{input_name: type_name}``. Subgraph-level inputs the skill requires
    the coordinator to declare and bind. Surfaced in the coordinator's Skills
    catalog so it knows which inputs to wire from upstream subgraphs."""

    hard_rules: list[str] = field(default_factory=list)
    """Anchored references like ``perception_pipeline_invariants.md#emit-both-obb-and-mask``
    or inline rule strings. Anchored refs resolve to the bundle's own
    ``references/`` directory — every skill bundle is self-contained and
    embeds the docs it needs (no shared library)."""

    canonical_scripts: list[CanonicalScript] = field(default_factory=list)
    """Scripts the subgraph_agent may emit as ``type: script`` states."""

    prompts: dict[str, str] = field(default_factory=dict)
    """``{logical_name: bundle_relative_path}``. Loaded at skill-runtime via
    :func:`gap.skills.load_prompt`."""

    references: list[ReferenceDoc] = field(default_factory=list)
    """Long-form rationale; lazy-loaded by the codegen ``read_skill_reference`` tool."""

    streaming: bool = False
    """Whether the bundle's callable streams intermediate values via
    ``ctx.publish``. The validator checks ``streaming: true`` nodes against
    this contract."""

    tools: dict[str, str] = field(default_factory=dict)
    """Tool bundles: ``{tool_name: one_line_summary}`` for each function the
    bundle's ``tools.py`` exposes via ``@tool``. Documentation only — the
    authoritative schemas come from the gap.tools registry."""

    requires: SkillRequires | None = None
    """Operational requirements consumed by ``gap check`` (GPU, env vars,
    downloaded weights). ``None`` means the bundle declares nothing —
    semantically the same as an empty block, but ``gap check`` and the
    registry test suites can tell "undeclared" from an explicit
    ``requires: {}``."""

    serving: Serving | None = None
    """How gap launches the bundle's out-of-process server. Required for
    ``kind='policy'``; optional for ``kind='tool'`` (declaring it opts the
    bundle into the RPC path). ``None`` means in-process dispatch."""

    # Bundle metadata
    bundle_dir: Path | None = None
    """Filesystem path of the owning bundle. Set by the registry at discovery."""

    body: str = ""
    """The SKILL.md body (everything after the closing ``---``). Empty for
    legacy class-based skills that don't have a SKILL.md yet."""


class Skill:
    """Base class for stateful, class-based skills.

    Skills that need to retain state across multiple invocations within a
    single workflow execution should subclass ``Skill`` and define::

        class MySkill(Skill):
            meta = SkillMeta(description="...", params={...}, outputs={...})

            def run(self, ctx: NodeContext, ...) -> Output:
                ...

    The runtime instantiates one instance per skill per workflow execution.
    Instance attributes set in ``__init__`` (or anywhere in ``run``) persist
    across all visits to the corresponding state during that execution and
    are discarded when the executor's ``finally`` block runs.

    Function-style callable skills (module-level ``_meta`` + ``def run(ctx, ...)``)
    continue to work unchanged. Subclassing ``Skill`` is opt-in and is
    motivated by genuine state needs — long-running loops with replan
    caches (pi05-libero), trackers accumulating evidence
    (tracking-objects), etc.
    """

    meta: ClassVar[SkillMeta]

    def run(self, ctx, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must override run(self, ctx, ...)"
        )
