"""Camera and depth utilities for the gap simulation environments.

Ported from HyRL's hyrl/utils/camera_utils.py and depth_utils.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def obs_get_rgb(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Recursively search through observation dict to find RGB images.

    Args:
        obs: Observation dictionary with nested camera data.

    Returns:
        Mapping of camera names to RGB image arrays.
    """
    rgb_dict: dict[str, np.ndarray] = {}

    for key, value in obs.items():
        if isinstance(value, dict):
            if "images" in value and isinstance(value["images"], dict):
                if "rgb" in value["images"]:
                    rgb_dict[key] = value["images"]["rgb"]
            else:
                nested_rgb = obs_get_rgb(value)
                rgb_dict.update(nested_rgb)

    return rgb_dict


def depth_color_to_pointcloud(
    depth: npt.NDArray[np.float64],
    img: npt.NDArray[np.uint8],
    intrinsics: npt.NDArray[np.float64],
    subsample_factor: int = 1,
    depth_clip_range: tuple[float, float] = (0.015, 20.0),
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Convert depth and RGB image to a 3D point cloud.

    Args:
        depth: Depth image (H, W) in meters.
        img: RGB image (H, W, 3) uint8.
        intrinsics: Camera intrinsics matrix (3, 3).
        subsample_factor: Spatial subsampling factor.
        depth_clip_range: (near, far) depth clipping.

    Returns:
        (points, colors) arrays after filtering invalid values.
    """
    if len(depth.shape) != 2:
        raise ValueError(f"Depth array must be 2D, got shape {depth.shape}")
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError(f"Image array must be (H, W, 3), got shape {img.shape}")
    if depth.shape[:2] != img.shape[:2]:
        raise ValueError(
            f"Depth and image dimensions must match: {depth.shape[:2]} vs {img.shape[:2]}"
        )
    if intrinsics.shape != (3, 3):
        raise ValueError(f"Intrinsics must be (3, 3), got shape {intrinsics.shape}")
    if subsample_factor <= 0:
        raise ValueError(f"Subsample factor must be positive, got {subsample_factor}")

    H, W = depth.shape
    H_sub = H // subsample_factor
    W_sub = W // subsample_factor
    depth = depth[::subsample_factor, ::subsample_factor]
    img = img[::subsample_factor, ::subsample_factor]

    intrinsics = intrinsics.copy()
    intrinsics[0, 0] /= subsample_factor
    intrinsics[1, 1] /= subsample_factor
    intrinsics[0, 2] /= subsample_factor
    intrinsics[1, 2] /= subsample_factor

    w, h = np.meshgrid(np.arange(W_sub), np.arange(H_sub), indexing="xy")
    pixels = np.stack([w.flatten(), h.flatten()], axis=-1).astype(np.float32)

    z = depth.reshape(-1)
    x = (pixels[:, 0] - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (pixels[:, 1] - intrinsics[1, 2]) * z / intrinsics[1, 1]

    points = np.stack([x, y, z], axis=-1)
    colors = img.reshape(-1, img.shape[-1])[:, :3] / 255.0

    near_clip, far_clip = depth_clip_range
    valid_mask = (
        ~np.isnan(points).any(axis=1)
        & ~np.isinf(points).any(axis=1)
        & (points[:, 2] <= far_clip)
        & (points[:, 2] >= near_clip)
    )

    return points[valid_mask], colors[valid_mask]
