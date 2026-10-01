"""Shared agent policy text and metric height guidance."""
from src.tools.names import public_tool_text

_HEIGHT_GUIDANCE = (
    'Heights are explicit objects {reference,value_m}. reference=absolute means base contact-center Z; '
    'grasp, current_tcp, clicked_point, segment_median, segment_mean, observed_min and observed_max '
    'add a signed value_m offset in metres to that reference. current_tcp is measured when planning. '
    'For pick transit, grasp means each candidate grasp Z; elsewhere it requires a recorded held grasp. '
    'Unavailable references are rejected; do not use grasp to define its own geometric grasp height. '
    'Resolved heights stay fixed during execution and pose refinement. All positions use connector_base metres. '
    'Read current_tcp_pose_base, gripper_max_opening_m, grasp_axis_preapproach_distance_m, '
    'surface bounds/quantiles and principal_xy_axis_yaw_deg before selecting Z/yaw. '
    'The XY principal axis is undirected modulo 180 degrees and may be null for isotropic surfaces; '
    'it is not a prescribed grasp angle. Visible surfaces do not establish hidden shape or centre of mass. '
    'For a transport pick, command a post-pick contact-center height at least 0.20 m above the grasp height. '
    'After lifting, inspect the fresh observation; if more height is needed, Prime should use move_vertical '
    'to request additional upward motion before finishing. This is agent guidance, not an automatic height clamp. '
    'The contact center between the jaws differs from the robot flange. '
)


def extend_pose_prompt(role, prompt):
    if role in ('grasp', 'place', 'refiner'):
        prompt += (
            ' RobotUse POSE EDITOR: candidate requests show cyan camera projections. Inspect a candidate '
            'to receive its detailed pose card; adjustments return an updated card. Inspect '
            'SIDE, TOP and CLOSING PLANE, geometry_cues and pose_editor '
            'after each edit. Translation commands and target_center_mm are BASE XYZ in mm; '
            'target_center_local_mm is a separate measurement in gripper-local axes. '
            'The orange cross is the observed cloud bounding-box midpoint, NOT a recommended grasp '
            'point: do not nudge merely to zero its offset or approve a grasp because it is zero. '
            'Whole-cloud span is NOT the thickness at the intended finger contacts. Use the actual '
            'local contact geometry and fresh RGB; a paused target cloud is not tracked after approach. '
            'Rotations are local about the jaw centre. Purple +30 deg '
            'examples illustrate direction, not executed motion. Rotations have no per-step or cumulative '
            'angular limit, including edits of initially top-down geometric poses. BASE translation also has '
            'no per-step or cumulative magnitude cap. Finite values, route validity and configured collision '
            'checks still apply. Command totals/history are not an object pose or contact verdict. '
            'For place_refinement, inspect_place_candidate and explicit_adjust_place review/edit the '
            'delegated placement; return the inspected candidate_ref or failed/needs_observation. '
            'Placement review never executes or releases. A later Prime call owns release.')
    if role in ('prime', 'point', 'grasp', 'refiner'):
        prompt += (
            ' RobotUse CLICKED CONTACT: Point may select a visible contact feature on the intended object. '
            'Grasp may explicitly request direction=clicked to seed one pose at that registered surface XY, '
            'with chosen geometric_height and transit; use reference=clicked_point for a measured Z offset. '
            'Set tolerance_deg, azimuth_deg, polar_deg to null. The seed is top-down at base yaw zero, '
            'not an inferred surface normal or guaranteed grasp. Inspect and request Refiner for a different '
            'orientation/contact when needed. It uses the same planning budget and configured checks as '
            'other candidates; it never silently replaces a failed CGN request. Invalid depth, stale '
            'selection or a click outside the measured selected surface rejects without a center fallback.')
    if role in ('prime', 'grasp', 'place', 'refiner'):
        prompt += (
            ' RobotUse DIAGNOSTICS: planning_feedback identifies the actual failed segment and zero-based '
            'segment_index when known; it is not a collision/hold diagnosis. Change the implicated '
            'route segment, not unrelated heights. Missing details mean unknown, not a guessed failure. '
            'grasp_feedback.original_selection is the initial child choice; selected_candidate is the '
            'post-review choice. execution_pose records the dispatched candidate and pending/final '
            'commanded target, including paused edits. A target pose is not a measured achieved pose '
            'or proof of contact; check execution_started, paused_edit_applied and current RGB. '
            'grasp_plan_marked_executed is the backend plan flag, not an independent hold verdict.')
    return public_tool_text(prompt)
