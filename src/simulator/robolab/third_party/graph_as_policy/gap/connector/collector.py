"""DataCollector — synchronized per-step demonstration recording into HDF5.

Hooks the connector's step-callback seam (see
:meth:`gap.connector.core.Connector.add_step_callback`) and records one row
per control step while an episode is open:

HDF5 layout (flat, append-only)::

    /observations/<camera>_rgb   uint8   [N, H, W, 3]
    /observations/state          float32 [N, dof+1]   (arm joints + gripper)
    /actions                     float32 [N, A]       (zero-padded to max A)
    /rewards                     float32 [N]
    /dones                       bool    [N]
    /episode_ends                int64   [E]   (exclusive end index per episode)
    /episode_success             bool    [E]

Episode boundaries come from :meth:`start_episode` / :meth:`end_episode`.

LeRobot conversion: this layout maps 1:1 onto a LeRobot dataset —
``/observations/<camera>_rgb`` becomes ``observation.images.<camera>``,
``/observations/state`` becomes ``observation.state``, ``/actions`` becomes
``action``, and ``episode_index``/``frame_index`` columns are derived by
binning row indices with ``/episode_ends`` (``next_done`` is ``/dones``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np

logger = logging.getLogger(__name__)


class DataCollector:
    """Record synchronized obs/action/reward rows from a connector.

    Args:
        connector: A :class:`gap.connector.core.Connector`; the collector
            registers itself on the connector's step-callback list.
        out_path: Output ``.h5``/``.hdf5`` file (parent dirs are created).
    """

    def __init__(self, connector: Any, out_path: str | Path) -> None:
        self.connector = connector
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.out_path, "w")
        self._recording = False
        self._rows = 0
        self._episode_ends: list[int] = []
        self._episode_success: list[bool] = []
        self._action_dim = 0
        connector.add_step_callback(self._on_step)

    # ------------------------------------------------------------------
    # Episode boundaries
    # ------------------------------------------------------------------

    def start_episode(self) -> None:
        """Begin recording rows; idempotent."""
        self._recording = True

    def end_episode(self, success: bool) -> None:
        """Close the open episode and tag its success flag."""
        self._recording = False
        self._episode_ends.append(self._rows)
        self._episode_success.append(bool(success))

    # ------------------------------------------------------------------
    # Step hook
    # ------------------------------------------------------------------

    def _on_step(
        self, action: np.ndarray | None, obs: dict, reward: float, done: bool
    ) -> None:
        if not self._recording:
            return
        dof = getattr(self.connector, "_arm_dof", 7)
        state = np.asarray(
            obs.get("robot_joint_pos_0", np.zeros(dof + 1)), dtype=np.float32
        ).reshape(-1)

        if action is None:
            # Hold step: synthesize the zero-arm action + gripper command the
            # env applied (matches env._step_once's action layout).
            gv = 1.0 - float(getattr(self.connector, "_gripper_fraction", 1.0)) * 2.0
            action = np.concatenate([np.zeros(dof, dtype=np.float32), [gv]])
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        self._action_dim = max(self._action_dim, action.shape[0])

        self._append("/actions", self._pad(action, self._action_dim))
        self._append("/observations/state", state)
        self._append("/rewards", np.float32(reward))
        self._append("/dones", np.bool_(done))
        for cam in getattr(self.connector, "camera_names", []):
            cam_data = obs.get(cam)
            rgb = (
                cam_data.get("images", {}).get("rgb")
                if isinstance(cam_data, dict) else None
            )
            if rgb is not None:
                self._append(
                    f"/observations/{cam}_rgb", np.asarray(rgb, dtype=np.uint8)
                )
        self._rows += 1

    @staticmethod
    def _pad(arr: np.ndarray, width: int) -> np.ndarray:
        if arr.shape[0] >= width:
            return arr[:width]
        return np.pad(arr, (0, width - arr.shape[0]))

    def _append(self, name: str, row: np.ndarray) -> None:
        row = np.asarray(row)
        if name not in self._file:
            self._file.create_dataset(
                name,
                shape=(0, *row.shape),
                maxshape=(None, *row.shape),
                dtype=row.dtype,
                chunks=(1, *row.shape) if row.shape else (1024,),
            )
        ds = self._file[name]
        if row.shape and row.shape[-1] != ds.shape[-1] and row.ndim == 1:
            # Action width grew mid-run (mixed 7/8-dim controllers): rebuild
            # column count by padding the existing dataset.
            data = np.zeros((ds.shape[0], row.shape[0]), dtype=ds.dtype)
            data[:, : ds.shape[1]] = ds[...]
            del self._file[name]
            ds = self._file.create_dataset(
                name, data=data, maxshape=(None, row.shape[0]), dtype=row.dtype,
            )
        ds.resize(ds.shape[0] + 1, axis=0)
        ds[-1] = row

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush episode indices and close the file; detaches the hook."""
        if self._file is None:
            return
        if self._recording:
            self.end_episode(success=False)
        self._file.create_dataset(
            "/episode_ends", data=np.asarray(self._episode_ends, dtype=np.int64)
        )
        self._file.create_dataset(
            "/episode_success", data=np.asarray(self._episode_success, dtype=bool)
        )
        self._file.attrs["num_steps"] = self._rows
        self._file.attrs["cameras"] = list(getattr(self.connector, "camera_names", []))
        self._file.close()
        self._file = None
        try:
            self.connector.remove_step_callback(self._on_step)
        except Exception:
            pass

    def __enter__(self) -> DataCollector:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["DataCollector"]
