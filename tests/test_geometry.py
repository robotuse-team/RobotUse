import json

import numpy as np
import pytest

from src.tools.grasp.geometry import (
    observed_cloud_statistics, refine_top_down_contact_pose, resolve_height,
    top_down_contact_pose, validate_top_down_contact_pose, xy_candidates,
)


POINTS = np.array([[.2, -.4, .05], [.3, -.1, .1], [.4, -.2, .15],
                   [.5, -.3, .2], [1.6, -.5, .9]])


def test_measured_statistics_and_candidates_keep_click_and_distinct_centers():
    points = POINTS.copy()
    stats = observed_cloud_statistics(points, source='sam_front_wrist_fused')
    assert stats['frame'] == 'connector_base' and stats['units'] == 'metres'
    assert stats['source'] == 'sam_front_wrist_fused' and stats['count'] == 5
    np.testing.assert_allclose(stats['mean_xyz_m'], [.6, -.3, .28])
    np.testing.assert_allclose(stats['median_xyz_m'], [.4, -.3, .15])
    np.testing.assert_allclose(stats['min_xyz_m'], [.2, -.5, .05])
    np.testing.assert_allclose(stats['max_xyz_m'], [1.6, -.1, .9])
    np.testing.assert_allclose(stats['extents_xyz_m'], [1.4, .4, .85])
    assert stats['z_quantiles_m'] == pytest.approx(
        dict(p05=.06, p25=.1, p50=.15, p75=.2, p95=.76))
    choices = xy_candidates(points, clicked_xyz_m=[.7, -.6, .37])
    assert [item['candidate_id'] for item in choices] == ['clicked', 'median', 'mean']
    np.testing.assert_allclose([item['xy_m'] for item in choices],
                               [[.7, -.6], [.4, -.3], [.6, -.3]])
    assert [item['reference_z_m'] for item in choices] == pytest.approx([.37, .15, .28])
    assert choices[0]['source'] == 'observed_depth_at_selected_pixel'
    assert all(item['frame'] == 'connector_base' and item['units'] == 'metres'
               for item in choices)
    # Metadata is directly usable at a JSON tool boundary; input remains untouched.
    json.dumps(dict(statistics=stats, candidates=choices), allow_nan=False)
    np.testing.assert_array_equal(points, POINTS)


def test_absent_click_and_coincident_centers_keep_provenance():
    choices = xy_candidates([[.3, -.2, .1]])
    assert [item['candidate_id'] for item in choices] == ['median', 'mean']
    assert choices[0]['xy_m'] == choices[1]['xy_m'] == [.3, -.2]


@pytest.mark.parametrize('cloud', [[], [1., 2., 3.], [[1., 2.]],
    [[1., 2., 3., 4.]], [[1., 2., np.nan]], [[1., np.inf, 3.]],
    [[True, False, True]], [['1', '2', '3']], [[1j, 2., 3.]],
    [[1., 2., 3.], [1., 2.]]])
def test_invalid_clouds_fail_without_silent_filtering(cloud):
    with pytest.raises(ValueError):
        observed_cloud_statistics(cloud)
    with pytest.raises(ValueError):
        xy_candidates(cloud)


@pytest.mark.parametrize('click', [[1., 2.], [1., 2., np.nan], [True]*3, ['1']*3])
def test_supplied_click_requires_valid_measured_xyz(click):
    with pytest.raises(ValueError):
        xy_candidates(POINTS, clicked_xyz_m=click)


def test_nonbase_or_unidentified_cloud_is_not_relabelled_as_base():
    with pytest.raises(ValueError, match='connector_base'):
        observed_cloud_statistics(POINTS, frame='camera_optical')
    with pytest.raises(ValueError, match='source'):
        observed_cloud_statistics(POINTS, source='')


def test_height_is_agent_specified_absolute_or_explicit_relative():
    assert resolve_height(.42, mode='absolute') == .42
    assert resolve_height(.08, mode='surface_relative', reference_z_m=.37) == pytest.approx(.45)
    assert resolve_height(-.025, mode='surface_relative', reference_z_m=.1) == pytest.approx(.075)
    assert resolve_height(-.3, mode='absolute') == -.3  # No hidden height clamp.
    with pytest.raises(ValueError, match='requires an observed'):
        resolve_height(.08, mode='surface_relative')
    with pytest.raises(ValueError, match='does not take'):
        resolve_height(.42, mode='absolute', reference_z_m=.37)
    with pytest.raises(ValueError, match='mode'):
        resolve_height(.08, mode='automatic')
    with pytest.raises(TypeError):
        resolve_height(mode='absolute')


@pytest.mark.parametrize('value', [True, '0.2', np.nan, np.inf, [0.2], None])
def test_height_rejects_invalid_agent_values_or_surface_reference(value):
    with pytest.raises(ValueError):
        resolve_height(value, mode='absolute')
    with pytest.raises(ValueError):
        resolve_height(.08, mode='surface_relative', reference_z_m=value)


def test_top_down_contact_pose_uses_base_yaw_and_exact_contact_center():
    pose = top_down_contact_pose([.41, -.23], .185, 90.)
    np.testing.assert_allclose(pose[:3, 3], [.41, -.23, .185])
    np.testing.assert_allclose(pose[:3, :3], [[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]], atol=1e-15)
    # A point along local approach moves down from the chosen contact center.
    np.testing.assert_allclose(pose @ [0., 0., .1, 1.], [.41, -.23, .085, 1.])
    np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-15)
    assert np.linalg.det(pose[:3, :3]) == pytest.approx(1.)
    checked = validate_top_down_contact_pose(pose)
    checked[0, 3] = 99.
    assert pose[0, 3] == .41


def test_refinement_moves_in_base_axes_and_never_tilts_the_contact_pose():
    original = top_down_contact_pose([.41, -.23], .185, 90.)
    changed = refine_top_down_contact_pose(original, dx_m=.02, dy_m=-.03, dz_m=.04, yaw_deg=90.)
    np.testing.assert_allclose(changed[:3, 3], [.43, -.26, .225])
    np.testing.assert_allclose(changed[:3, :3], np.diag([-1., 1., -1.]), atol=1e-15)
    for _ in range(30):
        changed = refine_top_down_contact_pose(changed, yaw_deg=7.)
        np.testing.assert_array_equal(changed[:3, 2], [0., 0., -1.])
    np.testing.assert_allclose(original[:3, 3], [.41, -.23, .185])
    for forbidden in ('roll_deg', 'pitch_deg'):
        with pytest.raises(ValueError, match='yaw only'):
            refine_top_down_contact_pose(original, **{forbidden: .001})


@pytest.mark.parametrize('bad_pose', [np.eye(4), np.zeros((4, 4)), np.eye(3),
    np.diag([1., 1., -1., 1.]), np.diag([2., -1., -1., 1.]),
    np.diag([1., -1., -1., 2.]), np.full((4, 4), np.nan)])
def test_refinement_rejects_tilted_or_invalid_input_pose(bad_pose):
    with pytest.raises(ValueError):
        refine_top_down_contact_pose(bad_pose, yaw_deg=10.)


@pytest.mark.parametrize('xy,z,yaw', [([.1], .2, 0.), ([.1, np.inf], .2, 0.),
    ([.1, .2], np.nan, 0.), ([.1, .2], .2, True), ([.1, .2], .2, np.inf)])
def test_top_down_pose_requires_finite_numeric_agent_decisions(xy, z, yaw):
    with pytest.raises(ValueError):
        top_down_contact_pose(xy, z, yaw)


@pytest.mark.parametrize('reference,base', [
    ('absolute', 0.), ('current_tcp', .6), ('grasp', .25), ('clicked_point', .37),
    ('segment_median', .15), ('segment_mean', .28), ('observed_min', .05), ('observed_max', .9)])
def test_remote_height_reference_values(reference, base):
    from src.tools.grasp.backend import height_from_geometry
    assert height_from_geometry(dict(reference=reference, value_m=-.02),
        observed_cloud_statistics(POINTS), [.7, -.6, .37], current_tcp_z_m=.6,
        grasp_z_m=.25) == pytest.approx(base-.02)


def test_xy_principal_axis_is_undirected_and_isotropic_surface_has_none():
    assert observed_cloud_statistics([[0, 0, 0], [1, 1, .1], [2, 2, .2]])['principal_xy_axis_yaw_deg'] == pytest.approx(45.)
    assert observed_cloud_statistics([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])['principal_xy_axis_yaw_deg'] is None
    assert observed_cloud_statistics([[0, 0, 0]])['principal_xy_axis_yaw_deg'] is None
