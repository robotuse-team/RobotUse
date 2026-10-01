"""gap.envs — simulation environments behind the connector layer.

Contains the minimum necessary code to run LIBERO simulation, plus the
env registry the connector resolves env names through.

The registry (and this package) import without the sim stack installed —
env classes are heavy (gymnasium / viser / libero / mujoco) and load
lazily on attribute access or via :func:`gap.envs.registry.resolve`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .registry import EnvConfig, register_env, registered_envs, resolve

if TYPE_CHECKING:
    from .base_env import BaseEnv
    from .franka_real_env import FrankaRealEnv
    from .libero_env import FrankaLiberoEnv
    from .libero_perturbed_env import FrankaLiberoPerturbedEnv
    from .loader import load_libero_task
    from .ur_zed_env import URZedEnv

_LAZY_ATTRS = {
    "BaseEnv": ("gap.envs.base_env", "BaseEnv"),
    "FrankaLiberoEnv": ("gap.envs.libero_env", "FrankaLiberoEnv"),
    "FrankaLiberoPerturbedEnv": (
        "gap.envs.libero_perturbed_env",
        "FrankaLiberoPerturbedEnv",
    ),
    "FrankaRealEnv": ("gap.envs.franka_real_env", "FrankaRealEnv"),
    "URZedEnv": ("gap.envs.ur_zed_env", "URZedEnv"),
    "load_libero_task": ("gap.envs.loader", "load_libero_task"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(target[0])
    return getattr(module, target[1])


__all__ = [
    "BaseEnv",
    "EnvConfig",
    "FrankaLiberoEnv",
    "FrankaLiberoPerturbedEnv",
    "FrankaRealEnv",
    "URZedEnv",
    "load_libero_task",
    "register_env",
    "registered_envs",
    "resolve",
]
