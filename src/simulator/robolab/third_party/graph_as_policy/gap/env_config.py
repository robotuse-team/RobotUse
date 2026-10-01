"""Centralized environment-variable configuration for GaP's sim / connector /
environment layer.

A single place to **read and document** the ``GAP_*`` environment variables that
govern (a) how a graph executes on a simulator and (b) which sim environment is
built. Each accessor reads ``os.environ`` *on call* (never cached at import), so
per-process benchmark workers and tests that set a variable before constructing
a connector/env observe the value.

## Fast mode

Fast mode is **on by default** — the execution-speed flags take their fast
values (continuous streaming, no camera rendering during motion, cuRobo
CUDA-graph capture). Set ``GAP_FAST=0`` to restore the conservative (legacy)
path. The OSC servo is deliberately NOT in this set — it's opt-in via
``GAP_LIBERO_SERVO=1`` because it's only safe for free-space legs. Any
individual ``GAP_*`` flag explicitly set still wins over the unified default —
so you can keep fast mode on and turn one thing back off, e.g.::

    GAP_CUROBO_CUDA_GRAPH=0 gap run ...   # fast, but cuda-graph off

All of these preserve task success on the grocery pick-and-place; see
``docs/source/reference/environment-variables.md``.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


# ---------------------------------------------------------------------------
# Fast mode + primitive readers
# ---------------------------------------------------------------------------

def fast_enabled() -> bool:
    """Whether unified fast mode is on. **On by default** — set ``GAP_FAST=0``
    to restore the conservative (legacy) execution path."""
    return os.environ.get("GAP_FAST", "1").strip().lower() in _TRUE


def fast_bool(name: str, *, fast: bool, base: bool) -> bool:
    """Resolve a boolean speed flag.

    Precedence: an explicitly-set ``$name`` wins; otherwise ``GAP_FAST`` selects
    ``fast``; otherwise ``base`` (the conservative default).
    """
    raw = os.environ.get(name)
    if raw is not None:
        return raw.strip().lower() not in _FALSE
    return fast if fast_enabled() else base


def _opt(name: str) -> str | None:
    """A stripped env override, or ``None`` when unset/empty."""
    v = os.environ.get(name)
    v = v.strip() if v is not None else None
    return v or None


# ---------------------------------------------------------------------------
# Execution-speed flags (LIBERO sim / connector). Fast-mode aware.
# ---------------------------------------------------------------------------

def libero_stream() -> bool:
    """Continuous trajectory streaming vs legacy per-waypoint tracking.
    ``GAP_LIBERO_STREAM`` (default on)."""
    return fast_bool("GAP_LIBERO_STREAM", fast=True, base=True)


def libero_servo() -> bool:
    """Policy-like OSC Cartesian servo on ``go_to_pose_cartesian`` legs. Opt-in
    ONLY (``GAP_LIBERO_SERVO=1``) — NOT part of ``GAP_FAST``: it's safe for
    free-space transport legs (grocery_fulfillment) but degrades cluttered
    grasp legs (grocery_packing), where the servo stalls near the object."""
    return fast_bool("GAP_LIBERO_SERVO", fast=False, base=False)


def libero_motion_render() -> bool:
    """Render the cameras during motion segments. Off = skip the per-step
    offscreen render (faster). ``GAP_LIBERO_MOTION_RENDER`` (default on; fast off)."""
    return fast_bool("GAP_LIBERO_MOTION_RENDER", fast=False, base=True)


def libero_stream_max_step_frac() -> float:
    """Per-tick joint-step clamp as a fraction of the controller ``output_max``,
    bounded to ``[0.05, 1.0]``. Lower it to be gentler on a carried payload at
    some speed cost. ``GAP_LIBERO_STREAM_MAX_STEP_FRAC`` (default ``1.0``)."""
    try:
        frac = float(os.environ.get("GAP_LIBERO_STREAM_MAX_STEP_FRAC", "1.0"))
    except ValueError:
        frac = 1.0
    return min(1.0, max(0.05, frac))


def curobo_cuda_graph() -> bool:
    """Capture cuRobo's v0.8 pose planner into a CUDA graph: ~8× faster warm
    plans after a one-time capture (best at batch scale). Only safe for fixed-
    world plans. ``GAP_CUROBO_CUDA_GRAPH`` (default off; fast on)."""
    return fast_bool("GAP_CUROBO_CUDA_GRAPH", fast=True, base=False)


# ---------------------------------------------------------------------------
# Sim environment selection
# ---------------------------------------------------------------------------

def libero_joint_motion_mode() -> str:
    """``teleport`` (qpos write + short settle) or ``closed_loop`` (physics
    tracking). ``GAP_LIBERO_JOINT_MOTION_MODE`` (default ``closed_loop``);
    unknown values warn and fall back to ``closed_loop``."""
    mode = os.environ.get(
        "GAP_LIBERO_JOINT_MOTION_MODE", "closed_loop"
    ).strip().lower()
    if mode not in ("teleport", "closed_loop"):
        logger.warning(
            "Unknown GAP_LIBERO_JOINT_MOTION_MODE=%r; falling back to "
            "'closed_loop'", mode,
        )
        mode = "closed_loop"
    return mode


def libero_perturbed() -> bool:
    """Opt into the perturbed (scripted moving-basket) LIBERO variant.
    ``GAP_LIBERO_PERTURBED`` (default off)."""
    return os.environ.get("GAP_LIBERO_PERTURBED", "").strip().lower() in _TRUE


# ---------------------------------------------------------------------------
# Path overrides (return the override or ``None``; caller applies its default)
# ---------------------------------------------------------------------------

def vab_root() -> str | None:
    """``GAP_VAB_ROOT`` — vab LIBERO fork (variance/packing suites)."""
    return _opt("GAP_VAB_ROOT")


def libero_pro_root() -> str | None:
    """``GAP_LIBERO_PRO_ROOT`` — LIBERO-PRO fork (classic suites)."""
    return _opt("GAP_LIBERO_PRO_ROOT")


def ur_urdf() -> str | None:
    """``GAP_UR_URDF`` — UR URDF used for FK in the UR+ZED connector."""
    return _opt("GAP_UR_URDF")


def ur_zed_calib() -> str | None:
    """``GAP_UR_ZED_CALIB`` — UR+ZED hand-eye calibration ``.npy``."""
    return _opt("GAP_UR_ZED_CALIB")
