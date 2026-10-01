# Script contract — emitted Python files

Scripts (whether canonical scripts bundled in a skill or ad-hoc scripts
emitted by the `coder` subagent) follow a uniform contract.

## Shape

```python
from typing import TypedDict

from gap import NodeContext
from gap_core.types import OrientedBoundingBox, Se3Pose, Vec3


class Output(TypedDict):
    pose: Se3Pose


def run(ctx: NodeContext, obb: OrientedBoundingBox, z_offset: float) -> Output:
    ...
```

- `run(ctx: NodeContext, ...) -> Output` with type-annotated parameters.
- `Output` is a `TypedDict` declaring the output fields. Outputs from a
  state are referenced via `{"$ref": "<state>.<field>"}`.
- `NodeContext` exposes:
  - `ctx.tool(name, **kwargs)` — invoke any registered tool by its flat
    catalog name (connector `robot.*` / `sim.*` tools, bundle tools like
    `sam3.segment_text`, in-process `geometry.*` helpers). Returns the
    tool's result dict. This is the only dispatch surface.
  - Use `print()` for debug output.
- Scripts are Python: `True` / `False`, not JSON.

## The gap_core.types vocabulary (numpy-first TypedDicts)

`gap_core.types` defines plain `TypedDict`s carrying floats and numpy
arrays — access fields with **dict subscripts**, never attribute
access:

```python
z = pose["position"]["z"]          # ✅
z = pose.position.z                # ❌ AttributeError — these are dicts
```

The **complete field reference for every gap_core.types type** — exact key
names, nesting, and array shapes — is generated from `gap_core.schema` and
injected below under "Type field reference". Consult it instead of
guessing; it is authoritative and always current (e.g. the
`Se3Pose["rotation"]` vs `OrientedBoundingBox["orientation"]` asymmetry
that trips up generated code).

## Loading bundled prompt templates

Canonical scripts that belong to a registered bundle load VLM prompt
templates via:

```python
from gap.skills import load_prompt

vlm_prompt = load_prompt(
    __package__, "<prompt_name>",
    var1=value1, var2=value2,
)
```

The loader walks up to the bundle's SKILL.md and resolves
`prompts/<prompt_name>.md` against the bundle root. This works for
canonical scripts because the registry installs each script under a
synthetic package (`gap_skills.<kind>.<bundle>.scripts.<stem>`)
that points at the bundle directory.

It does NOT work for ad-hoc scripts emitted via `request_inline_script`
— those don't belong to a bundle, so they have no `prompts/` directory
and `__package__` is a synthetic `_gap_script_*` name. If an inline
script needs a prompt, embed the template literal in the script itself.
