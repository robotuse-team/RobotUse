"""Placement motion with a separate explicit release action."""

__all__ = ["PlacementMotionMixin"]


def __getattr__(name):
    if name == "PlacementMotionMixin":
        from .execution import PlacementMotionMixin

        return PlacementMotionMixin
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
