"""Trusted action-precondition feedback without changing robot state or guards."""

from types import MappingProxyType


_REASONS = MappingProxyType({
    'grasp_requires_release': (
        'A previous grasp/closed-pusher command is still active. A closed command is not proof '
        'that the target is held. Inspect current RGB; continue placing if held, or release '
        'before requesting another grasp. This request did not execute or change robot state.'
    ),
    'grasp_mode_requires_open': (
        'Grasp mode changes affect the next grasp and must be chosen before closing. '
        'The current grasp remains unchanged. Continue the current operation, or release '
        'before changing purpose and grasping again. This request did not execute.'
    ),
    'turn_requires_contact_grasp': (
        'This turn tool rotates a held contact grasp about the tool axis; it is not a general '
        'transport-pose adjustment. Use the placement/refinement workflow for a carried object. '
        'No rotation executed and robot state is unchanged.'
    ),
    'waypoint_requires_current_state': (
        'This waypoint belongs to an earlier robot state or observation. Request a new waypoint '
        'using current RGB. No motion executed; this rejection did not change robot state.'
    ),

    'destination_requires_place_selection': (
        'Saving a destination requires a current Pointer-approved place selection. '
        'Use delegate_destination with current RGB, then save_destination with its returned point_ref. '
        'A pick selection has a different role; this rejection does not establish missing geometry '
        'or require an additional camera view.'
    ),
    'contact_requires_current_target': (
        'Closed-pusher contact motion requires a target selected in the current RGB observation. '
        'Prime must request Pointer to select the contact target in current front/wrist images, '
        'then request a new contact waypoint. A target selection from before movement is stale; '
        'changing waypoint direction or height does not refresh its measured target geometry.'
    ),
    'positive_optical_height_required': (
        'An observation surface-anchor waypoint requires a strictly positive optical '
        'height_offset_m. Additional dx_m/dy_m/dz_m offsets are applied afterward '
        'and do not replace that height. For an intended relative displacement, '
        'request shift_waypoint with explicit base-frame offsets.'
    ),
    'current_optical_axis_not_downward': (
        'An observation surface-anchor waypoint requires the current wrist optical '
        'axis to point downward. The backend preserves the current orientation; '
        'changing the surface pixel or height does not change that axis. '
        'Relative translation is available through shift_waypoint, but does not '
        'rotate the wrist or establish an improved view.'
    ),
    'observe_requires_open_command': (
        'Observation motion requires an open gripper command. Prime may explicitly '
        'release if opening is appropriate, or choose transport/contact when their '
        'preconditions apply. A closed command does not establish that an object is held.'
    ),
    'held_observation_requires_transport_grasp': (
        'Held observation is available for a transport grasp, not a contact-held '
        'handle or an empty pusher. No motion executed; retain the current command '
        'and use actions appropriate to its mode.'
    ),
    'held_observation_requires_measured_attachment': (
        'Held observation requires an executed transport grasp and its measured attachment '
        'for the held-object path check. That attachment is unavailable. No motion '
        'executed; reporting held in text cannot create the missing attachment.'
    ),
    'place_rotation_disabled': (
        'Paused place rotation was not enabled for this run. Use the available XYZ '
        'place nudge or return to the placement candidate workflow. No motion executed.'
    ),
    'contact_requires_closed_command': (
        'Contact motion requires an executed grasp or a closed pusher command. '
        'A closed command does not establish that an object is held.'
    ),
    'pusher_requires_contact_purpose': (
        'A closed empty pusher requires contact purpose; transport is unavailable '
        'in that command mode.'
    ),
    'release_requires_placement_retry': (
        'The last placement stopped before release because the measured hand pose missed '
        'its release goal, so the object is still held away from the destination. Opening '
        'here would drop it at an unverified position. Request delegate_place again from '
        'the current observation (new candidates are generated from the current pose); '
        'a plain release becomes available after that retry. No motion executed.'
    ),
    'observe_from_above_first': (
        'This configuration observes the target from above before the first grasp. '
        'Ask Pointer for a downward observation waypoint above the target '
        '(propose_downward_waypoint, about 0.30 m height), validate_view and execute_view it, '
        'then select the target in the returned RGB and request the grasp. After two rejected '
        'or failed overhead waypoint attempts, grasping from the current view is permitted. '
        'No motion executed; robot state is unchanged.'
    ),
})


class ActionPreconditionError(ValueError):
    """One of the fixed, publicly reportable action-precondition failures."""

    def __init__(self, reason_code: str):
        if type(reason_code) is not str or reason_code not in _REASONS:
            raise ValueError('Unknown action precondition reason code')
        self._reason_code = reason_code
        super().__init__(_REASONS[reason_code])

    @property
    def reason_code(self) -> str:
        return self._reason_code

    def public_feedback(self) -> dict[str, object]:
        """Return a fresh public record containing only trusted fixed strings."""
        return {
            'error': 'action_precondition_failed',
            'executed': False,
            'state_changed': False,
            'reason_code': self._reason_code,
            'reason': _REASONS[self._reason_code],
        }
