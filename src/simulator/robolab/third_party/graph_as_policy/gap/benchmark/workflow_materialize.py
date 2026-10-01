"""Per-(suite, task) workflow templating for template-driven modes.

The ``llm_plus_policy`` / ``policy_only`` workflow templates use
``{{target}}`` / ``{{target_full}}`` / ``{{container}}`` /
``{{policy_id}}`` placeholders; substitution is exactly the launcher's
``{{key}}`` convention — we reuse
:func:`gap.agent.launcher.materialize_workflow` (copy + substitute +
JSON-validate) rather than reimplement it.
"""

from __future__ import annotations

import re
from pathlib import Path

# "pick (up) (the) <target> and place it in (the) <container>."
# The optional ``(?:\s+up)?`` must be a suffix of ``pick`` (NOT a
# ``Pick|Pick up`` alternation, which matches ``Pick`` first and
# swallows "up" into the target, e.g. target="up the milk").
_PROMPT_RX = re.compile(
    r"pick(?:\s+up)?\s+(?:the\s+)?(?P<target>.+?)\s+and\s+place\s+it\s+"
    r"in\s+(?:the\s+)?(?P<container>.+?)\.?$",
    re.IGNORECASE,
)


def parse_target_container(prompt: str) -> tuple[str, str]:
    """Best-effort (target, container) from a LIBERO language prompt.

    Returns ``("", "")`` when the prompt doesn't match the pick-place
    grammar; the caller decides whether that's fatal.
    """
    m = _PROMPT_RX.match(prompt.strip())
    if not m:
        return "", ""
    return m.group("target").strip(), m.group("container").strip()


def resolve_task_prompt(suite_name: str, task_id: int) -> str:
    """Resolve the LIBERO language instruction from task metadata.

    Reuses the launcher's resolver (metadata only — no sim env), the
    same path ``task: "auto"`` uses.
    """
    from gap.agent.launcher import _resolve_libero_prompts

    resolved = _resolve_libero_prompts(suite_name, [task_id])
    return resolved.get(task_id, "")


def materialize_for_task(
    *,
    template_dir: str | Path,
    dest_parent: Path,
    suite_name: str,
    task_id: int,
    task_prompt: str | None = None,
    policy_id: str = "pi05-libero",
) -> str:
    """Materialize the template for one ``(suite, task)``.

    Args:
        template_dir: The workflow template (e.g.
            ``examples/steered_policy/graph``).
        dest_parent: Parent dir; the workflow is written to
            ``dest_parent/workflow`` (the launcher convention).
        suite_name / task_id: Identify the LIBERO task.
        task_prompt: Override the resolved prompt (else resolved from
            LIBERO metadata).
        policy_id: Name of the policy SKILL to steer (== its preset, e.g.
            ``pi05-libero`` / ``molmoact-libero``). Substituted into the
            template's ``{{policy_id}}`` placeholder, which forms the
            ``<skill>.run`` policy node; the skill owns its server, so no
            ``policies:`` entry is needed. Default ``pi05-libero``.

    Returns:
        Absolute path to the materialized workflow directory.

    Raises:
        ValueError: prompt unresolvable / unparseable, or templating
            produced invalid JSON (propagated from the launcher helper).
    """
    from gap.agent.launcher import materialize_workflow

    prompt = task_prompt or resolve_task_prompt(suite_name, task_id)
    if not prompt:
        raise ValueError(
            f"could not resolve a language prompt for {suite_name}/"
            f"task_{task_id} (LIBERO metadata unavailable?)"
        )
    target, container = parse_target_container(prompt)
    # Only a hard error when the template actually consumes ``{{target}}``.
    # A cyclic clean-all-items template (grocery_packing) parameterizes
    # just ``{{policy_id}}`` and steers a generic-"object" prompt, so an
    # unparseable LIBERO language string (e.g. "Pack every item from the
    # floor into the basket") is fine for it.
    try:
        template_text = (Path(template_dir) / "workflow.json").read_text()
    except OSError:
        template_text = ""
    if "{{target" in template_text and not target:
        raise ValueError(
            f"could not parse a target object from prompt {prompt!r} "
            f"({suite_name}/task_{task_id})"
        )

    objects = {
        "target": target,
        "target_full": target,  # prompt gives no richer descriptor
        "container": container or "basket",
        "policy_id": policy_id,
    }
    Path(dest_parent).mkdir(parents=True, exist_ok=True)
    return materialize_workflow(str(template_dir), objects, Path(dest_parent))
