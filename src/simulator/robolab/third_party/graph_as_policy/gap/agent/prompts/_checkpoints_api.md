# Postcondition checkpoints

Each subgraph declares a small number of **postcondition checkpoints** —
boolean predicates over the simulator's `World` snapshot taken right
after the subgraph exits. The verification harness
(`gap.runtime.verify`) evaluates these once per visit; results drive
execution-time enforcement and the feedback for the next attempt.

A checkpoint is a postcondition, not a node. It does NOT change the
state machine. It tells the verification harness "this subgraph
claimed to land the robot above the target — here is the privileged
ground-truth check for that claim".

> **MANDATORY:** every subgraph MUST declare **≥ 1 `validate=True`
> checkpoint** with a non-empty `rationale`. The parser rejects a
> workflow with a checkpoint-less subgraph and re-prompts you.
> Keep the total ≤ 6.

## Why checkpoints exist

Without checkpoints an entire trial collapses to one terminal bool.
With checkpoints, every subgraph contributes a few graded signals
against privileged state (positions, contacts, joint state, AABBs) — so
failure localizes to the offending subgraph instead of bubbling up as
"terminal=False".

## The API

```python
sg.add_checkpoint(
    name,
    predicate,
    *,
    diagnostics=None,
    rationale="",
    validate=True,
    weight=1.0,
)
```

- **`name`** — short identifier unique within the subgraph
  (e.g. `"ee_above_target"`, `"target_held"`, `"settled_in_basket"`).
- **`predicate`** — accepts either signature:
  - `Callable[[World], bool]` — the classic privileged-only check.
  - `Callable[[World, dict], bool]` — also receives the subgraph's
    bound outputs (the `set_outputs(...)` dict resolved at exit). Use
    this when the postcondition is "did the value this subgraph
    *produced* match the privileged ground truth?" — e.g. perception
    OBB vs. real body pose, computed grasp pose vs. body AABB,
    `compute_drop_pose` output vs. container cavity. The values in
    the dict are gap_core.types TypedDicts — index with string keys
    (`o["grasp_pose"]["position"]["z"]`), never attribute access.

  The world snapshot is taken AFTER the subgraph exits, with streaming
  nodes flushed. The harness introspects the predicate's arity per call.
- **`diagnostics`** — optional `Callable[[World], dict]` (or 2-arg
  `Callable[[World, dict], dict]`) returning scalar values surfaced
  into the feedback when the predicate fails. Never gates anything;
  raising it doesn't fail the checkpoint.
- **`rationale`** — one-line natural-language description of what the
  predicate means. Shown verbatim in the feedback; the next attempt's
  LLM uses it to decide which checkpoint is too tight or too loose.
  **Required** (non-empty) on `validate=True` checkpoints.
- **`validate=True`** (default) — this is a *hard* postcondition: it
  drives triage and is enforced at execution time when the connector
  exposes ground truth (`gap.execute(..., checkpoints="warn"|"raise")`;
  in `"raise"` mode a failure routes to the subgraph's `on_error`). If
  any `validate=True` checkpoint in subgraph `k` fails, all downstream
  subgraphs are considered un-evaluated for coverage accounting.
  `validate=False` checkpoints are **probes**: they surface in the
  feedback but do not gate downstream evaluation and are never enforced.
- **`weight`** — soft weight on `validate=True` checkpoints; reserved for
  future weighted scoring (today: 1.0, all checkpoints equal).

## The `World` API surface

Predicates receive a `World` snapshot with these primary entry points
(see `gap.runtime.verify` for the full surface):

```python
world.body(name)              # -> Body for named scene object (raises BodyNotFoundError if absent)
world.has_body(name)          # -> bool: is a body with this name in the scene?
world.robot()                 # -> Robot view
world.held_body()             # -> Body currently grasped (or None)
world.body_inside(region)     # -> Body whose COM is inside the region
world.bodies_displaced(...)   # -> bodies that moved during the rollout
world.history()               # -> list of prior World snapshots
world.eventually(fn)          # -> True if fn(w) was True at any time
world.always(fn)              # -> True if fn(w) was True at every time
```

`Body` exposes pose / velocity / AABB / contact predicates (these are
real objects — attribute access is correct here):

```python
body.position                 # (3,) world frame
body.top_z                    # AABB upper z
body.is_grasped()             # any robot link in this body's contacts
body.is_in(container)         # inside cavity + (optionally) in contact
body.is_above(other, ...)
body.is_on(other, tol_m=0.05)        # AABB overlap — LENIENT (a frypan with just its
                                     # handle hanging over the burner satisfies is_on).
                                     # Use is_on_strict for placement on elongated objects.
body.is_on_strict(other, tol_m=0.05, tol_xy_m=0.0)
                                     # Z-near AND self centroid inside other XY AABB —
                                     # the predicate you want for "the body is actually
                                     # resting on the support", not just hanging over.
body.is_settled(speed_thresh=0.08)
body.distance_to(other) / xy_distance_to(other)
body.is_axis_aligned(local_axis="z", world_axis="z", tol_rad=0.20)
```

`Robot`:

```python
robot.ee_position             # (3,) world frame, panda_hand
robot.ee_quaternion_wxyz      # (4,) world frame
robot.joint_pos               # (n,) per-joint angles
robot.gripper_open_fraction   # 0 = closed, 1 = open
robot.gripper_is_open() / gripper_is_closed()
```

## Granularity

Checkpoints are **subgraph-or-coarser**, never per-node. A subgraph
typically declares one **terminal** `validate=True` checkpoint (the
postcondition the subgraph promised to satisfy) plus 1-3 `validate=False`
probes for diagnostic data.

| Subgraph kind | `validate=True` | Probes (`validate=False`) |
|---|---|---|
| Setup | `gripper_open`, `arm_at_home` | `arm_velocity_zero` |
| Perception | `target_perceived` | `target_within_workspace` |
| Planning | `grasp_pose_emitted` | `ik_solved`, `clearance_ok` |
| Approach | `ee_above_target` | `ee_orientation_aligned` |
| Manipulation (grasp) | `target_held` | `object_lifted`, `gripper_open_fraction` |
| Transport / place | `target_in_container` | `drop_xy_error`, `transport_stable` |
| Recovery | `held_or_explicitly_aborted` | `retries_remaining` |

## Worked examples

### Transport subgraph (using a temporal predicate)

```python
sg.add_checkpoint(
    "transport_stable",
    predicate=lambda w: w.always(
        lambda s: s.body("alphabet soup").is_grasped()
    ),
    rationale="target stayed grasped across every snapshot of this subgraph",
    validate=True,
)
```

## Output-anchored checkpoints (2-arg predicates)

A 2-arg predicate `lambda w, o: ...` lets a subgraph verify the
**values it just produced** against privileged ground truth. The second
arg is the subgraph's bound-outputs dict — keyed by whatever names this
subgraph declared in `sg.set_outputs(...)`. This is how you write
"perception correctness" or "planning correctness" postconditions
without circularity: one side is the workflow output, the other side is
the privileged `World`.

The output values are **gap_core.types TypedDicts** (plain dicts) — index
them with string keys. The privileged `w` side keeps attribute access.

When the LLM has declared:
```python
sg.set_outputs(
    target_obb=Ref("filter_obb"),
    target_mask=Ref("perceive.mask"),
    target_cloud=Ref("perceive.cloud"),
)
```
the predicate can reach into `o["target_obb"]` (an `OrientedBoundingBox`
TypedDict) and compare against the privileged `w.body("alphabet soup")`.
Use 2-arg predicates **whenever** a bound output can be cross-checked —
perception OBBs against body poses, computed grasp poses against target
AABBs, computed drop poses against container cavities:

### Perception sanity (frame-free 1-arg/2-arg OBB shape check)

⚠ **Frames:** perception OBBs/clouds/derived poses are in the **robot's
base frame** (camera and IK are anchored there). `w.body(...).position`,
`cavity_*`, `aabb_*` are in **world frame**. On a Franka the base sits
at world `x ≈ -0.6`, so a direct
`abs(o["...obb"]["center"]["x"] - w.body(...).position[0])` reads
~0.6 m **even when perception is perfect** — a guaranteed false-fail.

Validate perception implicitly via downstream behavior
(`target_held` → frame-independent contacts; `target_in_container` →
world-frame `Body.is_in(Body)`), and use a 1-arg sanity check on the
OBB alone:

```python
sg.add_checkpoint(
    "target_obb_is_plausible",
    predicate=lambda w, o: (
        0.01 < o["target_obb"]["extent"]["x"] < 0.30
        and 0.01 < o["target_obb"]["extent"]["y"] < 0.30
        and 0.01 < o["target_obb"]["extent"]["z"] < 0.40
    ),
    diagnostics=lambda w, o: {
        "extent": [
            float(o["target_obb"]["extent"]["x"]),
            float(o["target_obb"]["extent"]["y"]),
            float(o["target_obb"]["extent"]["z"]),
        ],
    },
    rationale="perception emitted a non-degenerate, can-sized OBB (frame-free)",
    validate=True,
)
```

The output keys to read from `o` are exactly the names the agent itself
chose in `sg.set_outputs(...)`. If you didn't bind the value as an
output, the predicate can't see it — use a 1-arg `predicate=lambda w: ...`
instead.

## Hard rules

1. Predicate is either `Callable[[World], bool]` or
   `Callable[[World, dict], bool]`. Not a string, not a class, not a
   coroutine.
2. `validate=True` checkpoints are evaluated even when the subgraph took
   the `on_error` path — `passed=False` then drives feedback to the
   upstream agent. `validate=False` probes are also evaluated to enrich
   diagnostics.
3. A predicate that raises is reported as `passed=False,
   eval_error="..."`. Diagnostics raising is silently dropped. Use this
   to short-circuit "preconditions not even present" cases — `lambda w:
   w.body("alphabet soup").is_grasped()` raises `BodyNotFoundError` when
   perception didn't run, which is the right signal.
4. Names must be unique within a subgraph. Cross-subgraph collisions are
   fine (each checkpoint is namespaced by `subgraph.name`).
5. Checkpoints are NOT in `workflow.json`. They are written to
   `<workflow_dir>/checkpoints/<sg>.py` sidecars (loaded by
   `gap.runtime.verify.load_checkpoints`); execution-time enforcement is
   controlled by `gap.execute(..., checkpoints="off"|"warn"|"raise")`
   and requires a connector that exposes world snapshots.
