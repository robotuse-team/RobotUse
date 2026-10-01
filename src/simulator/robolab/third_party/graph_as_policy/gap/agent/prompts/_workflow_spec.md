# Workflow v3 spec

The structural spec for v3 workflows: the top-level workflow shape, the
SubgraphDef shape, node types, edge semantics, `$ref` syntax, and
validation rules.

## Top-level workflow

```json
{
  "version": 3,
  "meta": { "name": "...", "description": "..." },
  "nodes": {
    "<sg_node_name>": { "type": "subgraph", "ref": "<sg_def_name>" },
    "done":  { "type": "end", "status": "success" },
    "abort": { "type": "end", "status": "failure",
               "recovery": [ { "tool": "robot.open_gripper", "inputs": {} },
                             { "tool": "robot.go_home", "inputs": {} } ] }
  },
  "edges": [["START", "<first_subgraph_node>"]],
  "conditional_edges": {
    "<sg_node_name>": {
      "router_field": "exit",
      "mapping": { "<exit_value>": "<next_node>", ... }
    }
  },
  "subgraphs": { "<sg_def_name>": { /* SubgraphDef */ } }
}
```

`START` and `END` are virtual node names. The top level orchestrates
subgraph nodes and end nodes; cross-subgraph routing is via the
`conditional_edges` block reading each subgraph's `exit` value.

`meta` is an open string→string map; recognized keys are `name`,
`description`, and `observation_stream_hz`. Postcondition-checkpoint
enforcement is controlled by the caller via `gap.execute(...,
checkpoints="off" | "warn" | "raise")` — it is not a workflow.json
field.

## SubgraphDef shape

```json
{
  "skill":  "<skill bundle name; echoed from your spec>",
  "inputs":  { "<name>": "<type name string>" },
  "outputs": { "<name>": { "$ref": "<node>.<field>" } },
  "nodes": {
    "<node_name>": { /* NodeDef */ },
    "<terminal_marker>": { "type": "noop" }
  },
  "edges": [
    ["START", "<first_node>"],
    ["<first_node>", "<second_node>"],
    ["<terminal_marker>", "END"]
  ],
  "conditional_edges": { },
  "exit": { "router_field": null, "success_values": ["<success_exit>"] },
  "on_error": "<failure_exit_value>"
}
```

Input/output **type names** are bare strings from the `gap_core.schema` type
registry — e.g. `"OrientedBoundingBox"`, `"Mask"`, `"PointCloud"`,
`"Se3Pose"`, `"Observation"`, `"Trajectory"`, plus the scalars `"str"` /
`"int"` / `"float"` / `"bool"`.

`skill` usually names a registered bundle, but for an **invented
(generated) skill** it names a brand-new skill the coordinator defined
that has no bundle. Such a subgraph is fully self-contained: its
behavior lives entirely in its own `type: script` / `type: tool` nodes,
and the runtime never resolves the `skill` name against any registry
(it is metadata only).

Scripts go in separate fenced blocks, namespaced under
`scripts/<subgraph_name>/`:

````python:scripts/<subgraph_name>/<script>.py
...
````

## Node types

The two production dispatch types are `tool` and `script`. `noop` and
`router` are control-flow markers. `subgraph` and `end` only appear at
the top level of the workflow (not inside subgraphs).

```json
{ "type": "tool", "tool": "robot.open_gripper",
  "inputs": { "settle_steps": 40 } }                       /* connector tool */

{ "type": "tool", "tool": "sam3.segment_box",
  "inputs": { "image": { "$ref": "obs.rgb" }, "box": { "$ref": "perceive.box" } } }

{ "type": "tool", "tool": "pi05-libero.run",
  "inputs": { "observation_stream": { "$ref": "in.observation_stream" },
              "prompt": "pick up the object and place it in the basket",
              "gripper_cycle_termination": true } }          /* learned-policy skill: owns its model + checkpoint, no policy_id */

{ "type": "tool", "tool": "tracking-objects", "streaming": true,
  "inputs": { "observation_stream": { "$ref": "in.observation_stream" } } }

{ "type": "script", "script": "scripts/<subgraph_name>/foo.py",
  "inputs": { ... } }                                      /* canonical bundle script
                                                              or rare inline helper */

{ "type": "noop" }                                          /* named terminal marker */
{ "type": "router", "script": "scripts/<sg>/route.py", "inputs": { ... } }
```

- **`tool`** — the canonical dispatch for **anything**: connector tools
  (registered as `robot.*` / `sim.*`, e.g. `robot.get_observation`,
  `robot.open_gripper`), tool-bundle functions (e.g.
  `sam3.segment_text`, `grounding-dino.detect`,
  `geometry.filter_and_compute_obb`, `curobo.plan_to_grasp_poses`),
  callable skill bundles (registered by bundle name, e.g.
  `tracking-objects`), and learned-policy skills (one bundle per model
  checkpoint, e.g. `pi05-libero`, `molmoact-libero` — each owns its server
  and takes no `policy_id`). `tool:` is always a single flat name.
- **`script`** — local Python file in the workflow folder. Prefer to
  point at a canonical bundle script (listed in the chosen skill's
  "Canonical scripts" table); only emit your own inline Python for
  ad-hoc helpers that no canonical and no tool covers.
- **`noop`** — empty body. Used as a named subgraph terminal so the node
  name becomes the subgraph's exit value when `exit.router_field=null`.
- **`router`** — Send dispatch. Function returns either a string target
  for static routing or a list of `{"to": "...", "inputs": {...}}` dicts
  for dynamic fan-out.

`streaming: true` is only valid on `tool` and `script` nodes. A
streaming node has **no outgoing edges** — it's a pure source. Consumers
read its latest published value via `{"$ref": "<node>"}`. The chosen
tool's bundle contract must declare `streaming: true`.

## Edge semantics

- **Static edges**: `["src", "dst"]` pairs.
- **Multiple outgoing edges from one node = parallel super-step.**
- **`conditional_edges`** dispatches on a router field. From a
  non-router source the field is on the source's output (set
  `router_field` to its name). From a router source set
  `router_field: null`.
- **`exit.router_field: null`** means the subgraph's exit value is the
  name of the terminal node (the one whose edge points to `END`). For
  this to work, declare your terminals as `noop` nodes named after the
  exit values (e.g. `"found": {"type": "noop"}` with edge
  `["found", "END"]`).
- **`exit.router_field: "<field>"`** reads the field on the terminal
  node's output as the exit value.
- **`on_error`** is an optional subgraph-level catch: when any node
  raises, the runtime emits this string as the subgraph's exit value
  (bypassing the terminal-node read). The `on_error` symbol is the
  subgraph's single failure exit. It is **never** a declared node and
  **never** a `conditional_edges` mapping target — failures surface
  only by raising.

## Reference syntax

| Form | Meaning |
|---|---|
| `{"$ref": "observe"}` | Full output of the `observe` node. |
| `{"$ref": "observe.cameras"}` | Nested field walk on the output. |
| `{"$ref": "tracker"}` | Snapshot of the latest published value of the streaming `tracker` node. |
| `{"$ref": "in.<name>"}` | Cross-subgraph input from your declared `inputs` schema. |

## Validation rules

Your subgraph fails validation if:

1. Any node referenced by an edge or conditional-edge mapping is not
   declared in `nodes` (and is not `START`/`END`).
2. Any non-`END` node is unreachable from `START`.
3. Any non-streaming, non-end node has no outgoing edge or
   conditional-edges entry.
4. A streaming node has any outgoing edge or conditional-edges entry.
5. A node with `streaming: true` invokes a skill whose contract has
   `streaming: false` (or vice versa) — when the skill registry is
   available.
6. `conditional_edges` from a non-router source omits `router_field`.
7. `conditional_edges` from a router source sets `router_field` to
   non-null (router scripts return the target directly).
8. `outputs` binding references an unknown node or an end node.
9. `exit.success_values` is empty (S7).
10. `on_error` collides with a declared node (S9) or appears as a
    `conditional_edges` mapping target (S10).
11. With `router_field: null`, any name in `exit.success_values` is
    not declared as a `noop` node; OR with `router_field` set, any
    name in `exit.success_values` collides with a node name (S11).
12. Any `{"$ref": "in.<name>"}` references an input name not declared
    in your `inputs` schema (the executor-injected
    `observation_stream` is exempt).
13. `inputs.<name>` declared on a reachable subgraph has no upstream
    producer subgraph that declares an output of the same name.

Errors are fed back; fix every error and re-emit the full subgraph.

## Data types

Declared input/output type names are bare strings from the `gap_core.schema`
type registry (`"OrientedBoundingBox"`, `"Se3Pose"`, `"PointCloud"`,
`"Mask"`, `"Observation"`, … plus the scalars `"str"` / `"int"` /
`"float"` / `"bool"`). The **complete, authoritative field reference for
every type** — exact key names, nesting, and array shapes — is generated
from `gap_core.schema` and injected into your prompt below under "Type field
reference"; consult it instead of guessing field names. In scripts,
import the types and use **dict subscripts** (never attribute access):

```python
from gap_core.types import OrientedBoundingBox, PointCloud, Se3Pose, Vec3
```
