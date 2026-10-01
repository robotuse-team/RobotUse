"""Panda robot and gripper geometry names used by the MuJoCo path checker."""


from __future__ import annotations


PANDA_ROBOT_CONTACT_GEOMS = tuple(
    f"robot0_link{index}_collision" for index in range(7)
)


PANDA_GRIPPER_CONTACT_GEOMS = (
    "gripper0_hand_collision",
    "gripper0_finger1_collision",
    "gripper0_finger1_pad_collision",
    "gripper0_finger2_collision",
    "gripper0_finger2_pad_collision",
)
