"""CPU tests for candidate filtering, angle boundaries, and guarded frame use."""
import numpy as np
import pytest

from src.tools.grasp.cgn_client import CAMERA_OPTICAL, ROBOT_BASE, ContactCenterGrasps
from src.tools.grasp.direction import filter_cgn_directions, select_top_down_grasp


def pose(polar_deg, azimuth_deg=0):
    """Construct a rigid orientation with polar measured from base -Z."""
    polar, azimuth = np.deg2rad([polar_deg, azimuth_deg])
    z = np.array([np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), -np.cos(polar)])
    x = np.array([np.cos(polar) * np.cos(azimuth), np.cos(polar) * np.sin(azimuth), np.sin(polar)])
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    return matrix


def test_vertical_cone_filters_before_score_sorting_without_changing_pose_or_score():
    poses = np.stack([pose(0), pose(30), pose(31), pose(180)])
    scores = np.array([0.2, 0.8, 0.99, 0.9])
    before = poses.copy()
    indices, metadata = filter_cgn_directions(poses, scores, direction="vertical", tolerance_deg=30)
    assert indices.tolist() == [1, 0]
    assert metadata["accepted_mask"] == [True, True, False, False]
    assert metadata["desired_approach_base"] == [0, 0, -1]
    assert metadata["angular_errors_deg"] == pytest.approx([0, 30, 31, 180])
    np.testing.assert_array_equal(poses, before)
    np.testing.assert_array_equal(scores, [0.2, 0.8, 0.99, 0.9])


def test_horizontal_band_treats_equal_up_and_down_tilts_equally():
    poses = np.stack([pose(90), pose(70), pose(110), pose(69), pose(111)])
    result = filter_cgn_directions(poses, [0.5, 0.8, 0.7, 1, 1], direction="horizontal", tolerance_deg=20)
    assert result.indices.tolist() == [1, 2, 0]
    assert result.metadata["angular_errors_deg"] == pytest.approx([0, 20, 20, 21, 21])
    assert result.metadata["desired_approach_base"] is None


def test_custom_cone_uses_polar_from_down_and_azimuth_from_positive_x():
    poses = np.stack([pose(90, 90), pose(90, 60), pose(90, 120), pose(90, 0), pose(0)])
    result = filter_cgn_directions(poses, [0.1, 0.3, 0.3, 0.99, 1], direction="custom",
                                   tolerance_deg=30, azimuth_deg=90, polar_deg=90)
    assert result.indices.tolist() == [1, 2, 0]  # stable ties retain CGN order
    np.testing.assert_allclose(result.metadata["desired_approach_base"], [0, 1, 0], atol=1e-15)
    for polar in (0, 180):
        result = filter_cgn_directions(np.stack([pose(0), pose(180)]), [1, 1], direction="custom",
                                       tolerance_deg=0, azimuth_deg=230, polar_deg=polar)
        assert result.indices.tolist() == [polar // 180]


def test_zero_and_full_tolerances_and_empty_candidates():
    poses = np.stack([pose(0), pose(90), pose(180)])
    assert filter_cgn_directions(poses, [1, 1, 1], direction="top_down", tolerance_deg=0).indices.tolist() == [0]
    assert filter_cgn_directions(poses, [1, 1, 1], direction="horizontal", tolerance_deg=0).indices.tolist() == [1]
    for direction, tolerance in (("vertical", 180), ("horizontal", 90)):
        assert filter_cgn_directions(poses, [1, 1, 1], direction=direction,
                                     tolerance_deg=tolerance).indices.tolist() == [0, 1, 2]
    result = filter_cgn_directions([], [], direction="vertical", tolerance_deg=30)
    assert result.indices.shape == (0,) and result.metadata["accepted_count"] == 0


@pytest.mark.parametrize("kwargs", [
    {"direction": "vertical", "tolerance_deg": -1},
    {"direction": "vertical", "tolerance_deg": 181},
    {"direction": "horizontal", "tolerance_deg": 91},
    {"direction": "diagonal", "tolerance_deg": 30},
    {"direction": "vertical", "tolerance_deg": True},
    {"direction": "vertical", "tolerance_deg": float("nan")},
    {"direction": "custom", "tolerance_deg": 30},
    {"direction": "custom", "tolerance_deg": 30, "azimuth_deg": 0, "polar_deg": 181},
    {"direction": "vertical", "tolerance_deg": 30, "azimuth_deg": 0},
])
def test_invalid_direction_contracts_are_rejected(kwargs):
    with pytest.raises(ValueError):
        filter_cgn_directions(pose(0)[None], [0.5], **kwargs)


@pytest.mark.parametrize("poses,scores", [
    (np.eye(3)[None], [1]), (pose(0)[None], []), (pose(0)[None], [float("nan")]),
    (np.diag([1, 1, -1, 1])[None], [1]), (np.zeros((1, 4, 4)), [1]),
])
def test_invalid_candidate_shapes_scores_and_rotations_are_rejected(poses, scores):
    with pytest.raises(ValueError):
        filter_cgn_directions(poses, scores, direction="vertical", tolerance_deg=30)


def test_named_capx_helper_uses_camera_frame_once_and_strict_dot_threshold():
    # Optical +Z becomes base -Z after this extrinsic rotation.
    camera = ContactCenterGrasps(np.eye(4)[None], [0.8], [[0, 0, 1]], CAMERA_OPTICAL)
    extrinsics = np.diag([1, -1, -1, 1])
    extrinsics[0, 3] = 2
    selected, score = select_top_down_grasp(camera, extrinsics)
    np.testing.assert_array_equal(selected, extrinsics)
    assert score == 0.8
    assert select_top_down_grasp(camera, extrinsics, vertical_threshold=1) == (None, -float("inf"))
    already_base = ContactCenterGrasps(extrinsics[None], [0.8], [[2, 0, -1]], ROBOT_BASE)
    with pytest.raises(ValueError, match="double conversion"):
        select_top_down_grasp(already_base, extrinsics)
