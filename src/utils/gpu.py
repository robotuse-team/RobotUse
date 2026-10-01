"""Resolve GPU visibility without discarding an inherited device allocation."""

import os


def gpu_visibility(variable="ROBOTUSE_GPU", environ=None):
    env = os.environ if environ is None else environ
    if variable in env:
        value = env[variable]
        if not value.isdecimal():
            raise ValueError(f"{variable} must select one nonnegative GPU index")
        return value
    return env.get("CUDA_VISIBLE_DEVICES", "0")


def physical_gpu_index(visibility):
    """Return the first visible numeric index; UUIDs/disabled devices are unknown."""
    first = visibility.split(",", 1)[0].strip()
    return int(first) if first.isdecimal() else None
