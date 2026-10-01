"""Expose the current Robotiq frame calibration and native mesh asset loader.

Values and objects are owned by the existing implementation: no transformed
copies, profile substitution, or new calibration defaults are introduced here.
"""

__all__ = [
    "FLANGE_FROM_GRASP",
    "GRASP_TO_EE",
    "RoboLabGripperAssets",
    "load_calibration",
]


def __getattr__(name):
    if name in ("FLANGE_FROM_GRASP", "GRASP_TO_EE", "RoboLabGripperAssets"):
        from src.simulator.robolab import gripper as robolab_gripper

        return getattr(robolab_gripper, name)
    if name == "load_calibration":
        from src.simulator.robolab.configuration import load_calibration

        return load_calibration
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
