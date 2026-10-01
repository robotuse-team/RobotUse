"""Bounded restart loop: return the arm to its recorded start, then retry.

A failed attempt leaves the arm wherever it stopped, often still holding the
object.  The next attempt would then read a scene the agent never saw and plan
from a pose nothing recorded, so every restart first puts the arm back into the
exact joint configuration captured at episode start.

This is not a simulator reset.  The world keeps whatever the failed attempt did
to it -- a dropped object stays dropped -- and the agent is told only that a
restart happened, so it judges the scene from its own fresh views rather than
from a claim made here about what the scene now is.

Two deliberate limits:

* The joint configuration is restored, not a Cartesian pose.  Joint space is
  what ``reset`` actually determines, and restoring it needs no IK and leaves
  no 7-DoF redundancy to resolve, so the restored arm shape is the recorded one
  rather than merely some shape with the same end-effector pose.
* The gripper is opened before the arm moves, unconditionally.  Carrying the
  object back would make the restart itself a transport the planner never
  certified.  Opening releases from wherever the arm stopped, which can be
  above the table, so a restart can drop what was held -- that is the intended
  trade and the reason the agent is told to re-read the scene.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


#: What ``move_to_joints`` is asked to converge to. It checks the L2 norm over
#: all joints, so this stays tight: loosening it makes the controller stop
#: earlier and the restored pose worse.
DRIVE_TOLERANCE_RAD = 0.01

#: Per-joint arrival tolerance, distinct from the controller's L2 drive target.
#: Allows tracking residuals while still rejecting incomplete returns.
JOINT_TOLERANCE_RAD = 0.03

#: Control steps per drive pass; repeated passes allow longer joint-space returns.
JOINT_MAX_STEPS = 400

#: Drive passes before the return is declared failed. Each pass re-measures, so
#: a long traverse finishes in a later pass while a genuinely blocked arm still
#: fails closed rather than looping.
RETURN_PASSES = 4


class RestartError(RuntimeError):
    """Raised when the arm could not be returned to its recorded start."""


if RETURN_PASSES < 1:
    # return_to_initial's failure message reports the pose the last pass
    # reached, so a zero-pass configuration would raise NameError on a name the
    # loop never bound instead of reporting a failed return.
    raise RestartError("RETURN_PASSES must be at least 1")


def read_joints(connector: Any, *, arm_id: int = 0) -> tuple[float, ...]:
    """Read one arm's joint positions through the public connector surface."""

    arms = connector.get_observation()["arms"]
    if arm_id >= len(arms):
        raise RestartError(f"connector exposes no arm {arm_id}")
    return tuple(float(value) for value in arms[arm_id]["joint_state"]["positions"])


@dataclass(frozen=True, slots=True)
class InitialArmState:
    """The arm configuration a restart returns to, captured once per episode."""

    joints: tuple[float, ...]
    gripper_fraction: float

    def __post_init__(self) -> None:
        if not self.joints:
            raise RestartError("initial arm state needs at least one joint")

    @classmethod
    def capture(cls, connector: Any, *, arm_id: int = 0) -> "InitialArmState":
        """Record the post-reset configuration, before any attempt moves."""

        arms = connector.get_observation()["arms"]
        if arm_id >= len(arms):
            raise RestartError(f"connector exposes no arm {arm_id}")
        return cls(
            joints=read_joints(connector, arm_id=arm_id),
            gripper_fraction=float(arms[arm_id]["gripper_fraction"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "joints": list(self.joints),
            "gripper_fraction": self.gripper_fraction,
        }


@dataclass(frozen=True, slots=True)
class RestartRecord:
    """What one restart did, for the run's chronological account."""

    failed_attempt: int
    reason: str
    gripper_fraction_before_release: float
    joint_error_rad: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": "attempt_restarted",
            "failed_attempt": self.failed_attempt,
            "next_attempt": self.failed_attempt + 1,
            "reason": self.reason,
            "gripper_fraction_before_release": self.gripper_fraction_before_release,
            "joint_error_rad": self.joint_error_rad,
        }


def return_to_initial(
    connector: Any,
    initial: InitialArmState,
    *,
    arm_id: int = 0,
) -> tuple[float, float]:
    """Open the gripper, drive the arm to ``initial``, and verify it arrived.

    Returns ``(gripper_fraction_before_release, joint_error_rad)``.  A recorded
    state whose joint count does not match the arm is refused before anything
    moves.  Raises ``RestartError`` when the arm did not converge within
    ``RETURN_PASSES``: continuing from a pose that is neither the failure pose
    nor the recorded start is worse than stopping, because nothing downstream
    would know which pose it planned against.  The error names both the pose
    reached and the pose wanted, because "did not converge" alone does not say
    whether the arm was still travelling or blocked.

    The drive is joint interpolation with no collision avoidance, the same
    primitive ``go_home`` uses. It is adequate here because the gripper is open
    and empty before the arm moves, but it is not a certified path.
    """

    current = read_joints(connector, arm_id=arm_id)
    if len(current) != len(initial.joints):
        raise RestartError(
            f"arm reports {len(current)} joints, recorded {len(initial.joints)}"
        )
    before = float(connector.get_gripper_fraction(arm_id=arm_id))
    connector.open_gripper(settle_steps=40, arm_id=arm_id)
    error = None
    for _ in range(RETURN_PASSES):
        connector.move_to_joints(
            list(initial.joints),
            tolerance=DRIVE_TOLERANCE_RAD,
            max_steps=JOINT_MAX_STEPS,
            arm_id=arm_id,
        )
        restored = read_joints(connector, arm_id=arm_id)
        error = max(abs(a - b) for a, b in zip(restored, initial.joints))
        if error <= JOINT_TOLERANCE_RAD:
            return before, error
    raise RestartError(
        f"arm did not return to its recorded start after {RETURN_PASSES} passes: "
        f"{error:.4f} rad > {JOINT_TOLERANCE_RAD} rad; "
        f"joints={[round(v, 5) for v in restored]} "
        f"recorded={[round(v, 5) for v in initial.joints]}"
    )


@dataclass
class RestartLoop:
    """Bounded retries that each begin from the recorded initial arm state."""

    initial: InitialArmState
    max_restarts: int = 2
    records: list[RestartRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.max_restarts, bool) or not isinstance(self.max_restarts, int):
            raise RestartError("max_restarts must be an integer")
        if self.max_restarts < 0:
            raise RestartError("max_restarts must not be negative")

    @property
    def attempt(self) -> int:
        """1 for the first attempt, 2 for the first restart, and so on."""

        return len(self.records) + 1

    @property
    def restarted(self) -> bool:
        return bool(self.records)

    def may_restart(self) -> bool:
        return len(self.records) < self.max_restarts

    def restart(self, connector: Any, *, reason: str, arm_id: int = 0) -> RestartRecord:
        """Return the arm to its start and open the next attempt.

        The caller checks ``may_restart`` first; calling past the budget is a
        programming error, not a runtime condition to absorb.
        """

        if not self.may_restart():
            raise RestartError(
                f"restart budget exhausted after {len(self.records)} restart(s)"
            )
        gripper_before, error = return_to_initial(connector, self.initial, arm_id=arm_id)
        record = RestartRecord(
            failed_attempt=self.attempt,
            reason=reason,
            gripper_fraction_before_release=gripper_before,
            joint_error_rad=error,
        )
        self.records.append(record)
        return record

    def notice(self) -> str:
        """Agent-visible preamble for a restarted attempt; empty on the first.

        It says that a restart happened and that the gripper was opened, and
        nothing about why.  The failure reason is run evidence, and some of it
        is outcome-adjacent, so it stays in the log and out of the prompt.
        """

        if not self.records:
            return ""
        return (
            f"RESTART {len(self.records)} of {self.max_restarts}: the previous "
            "attempt failed and the arm was returned to its starting "
            "configuration. The gripper was opened on the way back, so "
            "anything it held was released and the scene may differ from your "
            "earlier views. Read the views supplied now on their own terms.\n"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_restarts": self.max_restarts,
            "restarts_used": len(self.records),
            "initial_arm_state": self.initial.to_dict(),
            "restarts": [record.to_dict() for record in self.records],
        }
