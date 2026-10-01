"""Call-count guards for tool dispatch.

Prevents runaway loops when workflows or skills call rate-limited tools.
Tools are classified by their registry *tags* (``perception``, ``planning``,
``sim_step``). Limits resolve in priority order: per-workflow
:func:`set_limits` overrides, then the ``GAP_MAX_PERCEPTION_CALLS`` /
``GAP_MAX_PLANNING_CALLS`` / ``GAP_MAX_SIM_STEPS`` environment variables,
else unlimited.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterable
from enum import Enum

from gap_core.errors import GuardLimitExceeded


class CallCategory(Enum):
    PERCEPTION = "perception"
    PLANNING = "planning"
    SIM_STEP = "sim_step"


_ENV_VARS = {
    CallCategory.PERCEPTION: "GAP_MAX_PERCEPTION_CALLS",
    CallCategory.PLANNING: "GAP_MAX_PLANNING_CALLS",
    CallCategory.SIM_STEP: "GAP_MAX_SIM_STEPS",
}

# Tool tag -> category. Tags come from ToolDescriptor.tags (set by @tool
# declarations, bundle frontmatter, or the connector's register_callable).
_TAG_CATEGORIES = {
    "perception": CallCategory.PERCEPTION,
    "planning": CallCategory.PLANNING,
    "sim_step": CallCategory.SIM_STEP,
}

_lock = threading.Lock()
_counters: dict[CallCategory, int] = {cat: 0 for cat in CallCategory}
# Per-workflow programmatic limits; take priority over env vars when set.
_limits: dict[CallCategory, int | None] = {cat: None for cat in CallCategory}


def classify_tags(tags: Iterable[str]) -> CallCategory | None:
    """Classify a tool's tag tuple into a call category.

    Args:
        tags: The tool's ``ToolDescriptor.tags``.

    Returns:
        The category of the first rate-limited tag, or ``None`` if the
        tool is not rate-limited.
    """
    for tag in tags:
        category = _TAG_CATEGORIES.get(tag)
        if category is not None:
            return category
    return None


def set_limits(
    *,
    perception: int | None = None,
    planning: int | None = None,
    sim_step: int | None = None,
) -> None:
    """Set per-workflow call limits, overriding the environment variables.

    Replaces all three overrides at once: a category left as ``None`` falls
    back to its ``GAP_MAX_*`` environment variable (else unlimited). Call
    ``set_limits()`` with no arguments to clear all programmatic limits.
    """
    with _lock:
        _limits[CallCategory.PERCEPTION] = perception
        _limits[CallCategory.PLANNING] = planning
        _limits[CallCategory.SIM_STEP] = sim_step


def _resolve_limit(category: CallCategory) -> int | None:
    """Resolve the active limit: set_limits override, env var, else None."""
    with _lock:
        override = _limits[category]
    if override is not None:
        return override

    limit_str = os.environ.get(_ENV_VARS[category])
    if limit_str is None:
        return None
    try:
        return int(limit_str)
    except (ValueError, TypeError):
        return None


def check_and_increment(category: CallCategory) -> None:
    """Increment the counter for *category* and raise if the limit is exceeded.

    Raises:
        GuardLimitExceeded: If the configured limit is exceeded.
    """
    limit = _resolve_limit(category)
    if limit is None:
        return

    with _lock:
        _counters[category] += 1
        count = _counters[category]

    if count > limit:
        env_var = _ENV_VARS[category]
        raise GuardLimitExceeded(
            f"Safety guard: {category.value} call limit exceeded "
            f"({count}/{limit}). Set {env_var} or set_limits() to increase."
        )


def check_and_increment_if_applicable(tags: Iterable[str]) -> None:
    """Classify *tags* and enforce the guard if the tool is rate-limited."""
    category = classify_tags(tags)
    if category is not None:
        check_and_increment(category)


def reset_counters() -> None:
    """Reset all call counters to zero. Call at the start of each workflow."""
    with _lock:
        for cat in CallCategory:
            _counters[cat] = 0
