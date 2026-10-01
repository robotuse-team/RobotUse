"""TrialLogger — continuous robot/scene data logging for post-hoc 3D replay.

Records timestamped data streams during workflow execution:
- Robot joint trajectories (per sim step)
- Camera frames (RGB, depth, intrinsics, extrinsics)
- Point clouds, OBBs, named poses, events

Data is buffered in memory and flushed to disk on close().
Storage format is designed for fast sequential replay in viser.

Usage::

    logger = TrialLogger(trial_dir)
    logger.set_meta(robot_urdf="/path/to/robot.urdf", arm_dof=6)

    # Per sim step:
    logger.log_joints(joints_array, gripper=0.8)

    # Per observation:
    logger.log_camera("zed_left", rgb, depth=depth, intrinsics=K, pose=cam_pose)

    # Per perception result:
    logger.log_pointcloud("target", points, colors=colors)
    logger.log_obb("target", center, extent, orientation)

    logger.close()
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class TrialLogger:
    """Logs continuous robot/scene data during a trial."""

    def __init__(self, output_dir: str | Path):
        self._root = Path(output_dir) / "scene_log"
        self._root.mkdir(parents=True, exist_ok=True)

        self._meta: dict[str, Any] = {}
        self._start_time: float | None = None

        # Joint trajectory buffer: [(timestamp, joints, gripper)]
        self._joint_timestamps: list[float] = []
        self._joint_positions: list[np.ndarray] = []
        self._joint_grippers: list[float] = []

        # Camera state: per-camera tracking
        self._cameras: dict[str, _CameraLog] = {}

        # Point clouds: {name: [(timestamp, points, colors)]}
        self._pointclouds: dict[str, list[tuple[float, np.ndarray, np.ndarray | None]]] = {}

        # Simple JSON-serializable logs
        self._obbs: list[dict] = []
        self._poses: list[dict] = []
        self._events: list[dict] = []

        self._closed = False

    def set_meta(self, **kwargs: Any) -> None:
        """Set metadata (robot_urdf, arm_dof, calibration_path, etc.)."""
        self._meta.update(kwargs)

    # ------------------------------------------------------------------
    # Logging methods
    # ------------------------------------------------------------------

    def log_joints(
        self,
        joints: np.ndarray,
        gripper: float = 1.0,
        arm_id: int = 0,
        timestamp: float | None = None,
    ) -> None:
        """Log robot joint positions. Called per sim step or observation.

        If ``timestamp`` is provided, it is used as the recorded wall-clock
        time (useful when draining a buffer of pre-captured sim steps so the
        replay preserves real inter-step spacing). Otherwise a fresh
        ``time.time()`` is taken via ``_tick``.
        """
        t = self._tick() if timestamp is None else float(timestamp)
        # Keep _start_time anchored even when caller supplies timestamps.
        if self._start_time is None:
            self._start_time = t
        self._joint_timestamps.append(t)
        self._joint_positions.append(np.asarray(joints, dtype=np.float32).copy())
        self._joint_grippers.append(float(gripper))

    def log_camera(
        self,
        name: str,
        rgb: np.ndarray,
        depth: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        pose: np.ndarray | None = None,
    ) -> None:
        """Log a camera frame. Writes image to disk immediately (large data)."""
        import cv2  # lazy: opencv is heavy and only camera logging needs it

        t = self._tick()
        cam = self._get_camera(name)

        # Write RGB as JPEG (compressed)
        idx = cam.frame_count
        img_dir = cam.dir / "images"
        img_dir.mkdir(exist_ok=True)
        img_path = img_dir / f"{idx:06d}.jpg"
        cv2.imwrite(str(img_path), rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])

        cam.timestamps.append(t)
        cam.frame_count += 1

        # Write depth
        if depth is not None:
            depth_dir = cam.dir / "depths"
            depth_dir.mkdir(exist_ok=True)
            np.save(str(depth_dir / f"{idx:06d}.npy"), depth.astype(np.float32))

        # Store intrinsics once
        if intrinsics is not None and cam.intrinsics is None:
            cam.intrinsics = np.asarray(intrinsics, dtype=np.float32)

        # Store pose per frame
        if pose is not None:
            cam.poses.append(np.asarray(pose, dtype=np.float32).copy())

    def log_pointcloud(
        self,
        name: str,
        points: np.ndarray,
        colors: np.ndarray | None = None,
    ) -> None:
        """Log a 3D point cloud."""
        t = self._tick()
        pts = np.asarray(points, dtype=np.float32).copy()
        clr = np.asarray(colors, dtype=np.float32).copy() if colors is not None else None
        self._pointclouds.setdefault(name, []).append((t, pts, clr))

    def log_obb(
        self,
        name: str,
        center: np.ndarray,
        extent: np.ndarray,
        orientation: np.ndarray | None = None,
    ) -> None:
        """Log an oriented bounding box."""
        t = self._tick()
        self._obbs.append({
            "timestamp": t,
            "name": name,
            "center": np.asarray(center, dtype=np.float64).tolist(),
            "extent": np.asarray(extent, dtype=np.float64).tolist(),
            "orientation": np.asarray(orientation, dtype=np.float64).tolist() if orientation is not None else [1, 0, 0, 0],
        })

    def log_pose(
        self,
        name: str,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
    ) -> None:
        """Log a named SE3 pose (e.g., grasp target, EE goal)."""
        t = self._tick()
        self._poses.append({
            "timestamp": t,
            "name": name,
            "position": np.asarray(position, dtype=np.float64).tolist(),
            "quaternion_wxyz": np.asarray(quaternion_wxyz, dtype=np.float64).tolist(),
        })

    def log_event(self, name: str, data: dict | None = None) -> None:
        """Log a named event marker on the timeline."""
        t = self._tick()
        self._events.append({
            "timestamp": t,
            "name": name,
            "data": data or {},
        })

    # ------------------------------------------------------------------
    # Flush and close
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush all buffers to disk."""
        if self._closed:
            return
        self._closed = True

        try:
            self._flush()
            logger.info("TrialLogger: saved scene_log to %s", self._root)
        except Exception:
            logger.error("TrialLogger: failed to flush", exc_info=True)

    def _flush(self) -> None:
        # Meta
        self._meta["start_time"] = self._start_time
        self._meta["end_time"] = time.time()
        self._meta["num_joint_steps"] = len(self._joint_timestamps)
        self._meta["camera_names"] = list(self._cameras.keys())
        with open(self._root / "meta.json", "w") as f:
            json.dump(self._meta, f, indent=2)

        # Joints
        if self._joint_timestamps:
            np.savez_compressed(
                str(self._root / "joints.npz"),
                timestamps=np.array(self._joint_timestamps, dtype=np.float64),
                positions=np.array(self._joint_positions, dtype=np.float32),
                grippers=np.array(self._joint_grippers, dtype=np.float32),
            )

        # Camera metadata (images already written to disk per-frame)
        for cam in self._cameras.values():
            if cam.timestamps:
                np.save(str(cam.dir / "timestamps.npy"),
                        np.array(cam.timestamps, dtype=np.float64))
            if cam.intrinsics is not None:
                np.save(str(cam.dir / "intrinsics.npy"), cam.intrinsics)
            if cam.poses:
                np.save(str(cam.dir / "poses.npy"),
                        np.array(cam.poses, dtype=np.float32))

        # Point clouds
        if self._pointclouds:
            pc_dir = self._root / "pointclouds"
            pc_dir.mkdir(exist_ok=True)
            for name, frames in self._pointclouds.items():
                timestamps = np.array([f[0] for f in frames], dtype=np.float64)
                save_dict: dict[str, np.ndarray] = {"timestamps": timestamps}
                for i, (_, pts, clr) in enumerate(frames):
                    save_dict[f"points_{i}"] = pts
                    if clr is not None:
                        save_dict[f"colors_{i}"] = clr
                save_dict["num_frames"] = np.array(len(frames))
                np.savez_compressed(str(pc_dir / f"{name}.npz"), **save_dict)

        # JSON logs
        if self._obbs:
            with open(self._root / "obbs.json", "w") as f:
                json.dump(self._obbs, f, indent=2)
        if self._poses:
            with open(self._root / "poses.json", "w") as f:
                json.dump(self._poses, f, indent=2)
        if self._events:
            with open(self._root / "events.json", "w") as f:
                json.dump(self._events, f, indent=2)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _tick(self) -> float:
        t = time.time()
        if self._start_time is None:
            self._start_time = t
        return t

    def _get_camera(self, name: str) -> _CameraLog:
        if name not in self._cameras:
            cam_dir = self._root / "cameras" / name
            cam_dir.mkdir(parents=True, exist_ok=True)
            self._cameras[name] = _CameraLog(dir=cam_dir)
        return self._cameras[name]


class _CameraLog:
    """Per-camera state for TrialLogger."""

    def __init__(self, dir: Path):
        self.dir = dir
        self.timestamps: list[float] = []
        self.poses: list[np.ndarray] = []
        self.intrinsics: np.ndarray | None = None
        self.frame_count: int = 0
