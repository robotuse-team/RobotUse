"""Abstract base environment for low-level robot control.

Ported from HyRL's hyrl/envs/base_env.py — minimal interface needed
by the gap connector layer.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, SupportsFloat, TypeVar

try:
    from gymnasium import Env
except ImportError:  # pragma: no cover - exercised by the bare-install CI lane
    # gymnasium ships with the [libero] extra. A minimal stand-in keeps this
    # module importable on a bare install (the env registry and the real-robot
    # envs only need BaseEnv as an interface; sim envs that genuinely depend
    # on gymnasium are constructed only after the extra is installed).
    class Env:  # type: ignore[no-redef]
        """Stand-in for gymnasium.Env when the [libero] extra is absent."""

        def close(self) -> None:
            pass

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class BaseEnv(Env):
    """Base environment class for low-level control environments.

    Subclasses of gymnasium.Env with the additional contract that
    ``get_observation()``, ``compute_reward()``, and ``task_completed()``
    are implemented.
    """

    max_steps: int = 999999

    @abstractmethod
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_observation(self) -> ObsType:
        raise NotImplementedError

    @abstractmethod
    def compute_reward(self) -> SupportsFloat:
        raise NotImplementedError

    @abstractmethod
    def task_completed(self) -> bool:
        raise NotImplementedError
