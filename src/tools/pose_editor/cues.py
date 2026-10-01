"""Measured geometry cues, never a reach, contact or task-success verdict."""
import numpy as np


def grasp_cues(points, pose, *, jaw_offset=.136, opening=.085, translation_frame='local'):
    if translation_frame not in ('local', 'base'):
        raise ValueError('grasp cue translation_frame must be local or base')
    points, pose = np.asarray(points), np.asarray(pose)
    pivot = pose[:3, 3] + jaw_offset * pose[:3, 2]
    local = (points - pivot) @ pose[:3, :3]
    middle = (local.min(0) + local.max(0)) / 2
    base_offset = pose[:3, :3] @ middle
    offset = base_offset if translation_frame == 'base' else middle
    width = np.ptp(local[:, 0])
    return dict(kind='grasp', translation_frame=translation_frame, translation_unit='mm',
        pivot=pivot.tolist(), target_center=(pivot + base_offset).tolist(),
        target_center_position_frame='base',
        target_center_definition='observed_cloud_gripper_aligned_bounding_box_midpoint',
        target_center_is_grasp_target=False,
        target_center_mm_reference='jaw_contact_center', target_center_mm_frame=translation_frame,
        target_center_mm=(offset * 1000).round(1).tolist(),
        target_center_local_mm=(middle * 1000).round(1).tolist(),
        target_center_base_mm=(base_offset * 1000).round(1).tolist(),
        opening_mm=round(opening*1000, 1), measured_width_mm=round(width*1000, 1),
        measured_width_scope='whole_observed_cloud', measured_width_axis='gripper_local_x',
        lines=[
            f'Cloud bbox centre offset ({translation_frame.upper()} mm): ' +
                '   '.join(f'{a} {v*1000:+.0f}' for a,v in zip('XYZ', offset)),
            f'Whole-cloud span / local closing X: {width*1000:.0f} mm; jaw opening: {opening*1000:.0f} mm',
            'The orange cross is a cloud-bounds summary, not a recommended grasp point. '
            'Do not nudge merely to zero this offset; inspect the intended local contacts and fresh RGB.',
        ], scope='Orange cross: cloud bounds centre, NOT a grasp target; span is NOT contact thickness')


def place_cues(placed, destination, pose, *, jaw_offset=.136):
    placed, destination, pose = np.asarray(placed), np.asarray(destination), np.asarray(pose)
    center = (placed.min(0) + placed.max(0))/2
    target = (destination.min(0) + destination.max(0))/2
    shift = target[:2] - center[:2]
    top = float(np.percentile(destination[:, 2], 98))
    bottom = float(placed[:, 2].min())
    return dict(kind='place', translation_frame='base',
        pivot=(pose[:3, 3] + jaw_offset * pose[:3, 2]).tolist(),
        object_center=center.tolist(), destination_center=target.tolist(),
        center_shift_m=shift.round(5).tolist(), bottom_above_top_mm=round((bottom-top)*1000, 1),
        lines=[
            f'BASE translation toward footprint centre: dx_m {shift[0]:+.3f}   dy_m {shift[1]:+.3f}',
            f'Object bottom {(bottom-top)*1000:+.0f} mm relative to measured destination top (98th percentile).',
            'Green outline = observed footprint, NOT bowl interior. Check rim and support in SIDE before lowering.',
        ], scope='observed footprint only; cavity, support, collision and path remain unchecked')
