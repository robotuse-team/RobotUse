"""Expose the grasp route planner and its argument contract."""

__all__ = ["plan_explicit_grasp"]


def __getattr__(name):
    if name == "plan_explicit_grasp":
        from src.tools.grasp.execution import plan_explicit_grasp

        return plan_explicit_grasp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
