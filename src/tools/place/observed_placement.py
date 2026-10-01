"""Pre-pick destination memory and agent-selected AnyPlace placement paths."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import time
from pathlib import Path
from uuid import uuid4

import numpy as np

from src.backend.robot_base import LiveBackend
from src.utils.logging_utils import write_json, append_json
from src.tools.perception.rgbd_adapter import normalized_pixel


@dataclass(frozen=True)
class DestinationMemory:
    point_ref: str
    observation_id: str
    epoch: int
    points: np.ndarray
    scene: np.ndarray
    anchor: np.ndarray
    directory: Path


def measured_anchor(frame, u, v):
    x, y = normalized_pixel(u, v, frame.width, frame.height)
    depth = float(frame.depth_m[y, x])
    if not np.isfinite(depth) or not .02 < depth < 3.:
        raise ValueError("destination click has no measured depth")
    k = np.asarray(frame.intrinsics)
    return np.asarray(frame.camera_to_base.apply(((x-k[0, 2])*depth/k[0, 0],
                                                 (y-k[1, 2])*depth/k[1, 1], depth)), dtype=float)


class ObservedPlacementBackend(LiveBackend):
    placement_sampling = 'uniform without replacement'
    placement_choice_description = 'uniformly sampled choices'

    def _sample_place_choices(self, eligible, *, seed):
        from src.tools.place.collision import sample_agent_choices
        return sample_agent_choices(eligible, seed=seed)

    def __init__(self, *, anyplace, **kwargs):
        kwargs.update(active_perception=True, multiview=True, first_grasp_only=True)
        super().__init__(**kwargs)
        self.anyplace = anyplace
        self.destinations, self.destination_anchors = {}, {}
        self.place_predictions, self.place_validations = {}, {}
        self.place_adjustments = {}  # refined placement -> cumulative degrees from its origin
        self.grasp_attachment = None
        self.grasp_attempted = False
        self.placement_execution = None
        self.destination_viewed = False
        self.destination_view_anchor = None
        self.destination_view_epoch = None
        self.placement_pools = {}
        self.selected_destination_ref = None
        self.grasped_point_ref = None
        self.grasp_jaw_width_m = None
        self.scene_grippers = {}
        self.filtered_placement_scenes = {}

    def _placement_scene(self, plan, destination):
        from src.tools.place.collision import remove_captured_gripper
        # The pick mask removes the original object from obstacles. The saved
        # destination scene still contains that original object, so do not add
        # it back as a phantom obstacle after picking.
        key = (plan.scene_point_ref, str(destination.directory))
        if key not in self.filtered_placement_scenes:
            captured = self.scene_grippers[plan.scene_observation_id]
            filtered, audit = remove_captured_gripper(plan.obstacle_points,
                np.asarray(captured['hand_pose']), captured['modeled_jaw_width_m'], mesh_source=getattr(self, 'gripper_assets', None))
            # Destination selections are preserved even if near the known hand.
            scene = np.vstack((filtered, destination.points))
            self.filtered_placement_scenes[key] = scene
            self._record('placement_scene_gripper_removed', {'point_ref':plan.scene_point_ref,
                'observation_id':plan.scene_observation_id, **audit})
        return self.filtered_placement_scenes[key]

    def point_destination(self, observation_id, view_id, u, v):
        if self.grasp_attempted and not getattr(self, 'postgrasp_destination', False):
            raise ValueError("destination acquisition must precede the grasp")
        result = super().point(observation_id, view_id, u, v, role="place")
        ref = result["point_ref"]
        self.destination_anchors[ref] = measured_anchor(self._point(ref).frame, u, v)
        # A single visible front selection may seed a view movement when wrist
        # visibility is missing. It is never a saved destination or grasp cloud.
        self.latest_fused_ref = ref
        self._record("destination_seed", result)
        return result

    def select_destination(self, observation_id, front_u, front_v, wrist_u, wrist_v):
        from src.tools.perception.multiview import CAMERAS
        from src.tools.perception.intent import measured_target_consistency
        if self.grasp_attempted and not getattr(self, 'postgrasp_destination', False):
            raise ValueError("destination acquisition must precede the grasp")
        images, refs = [], []
        for view, (u, v) in zip(CAMERAS, ((front_u, front_v), (wrist_u, wrist_v))):
            result = self.point_destination(observation_id, view, u, v)
            refs.append(result["point_ref"])
            images.extend(result["image_refs"])
        selected = [self._point(ref) for ref in refs]
        consistency = measured_target_consistency(*(g.object_points for g in selected))
        if consistency["status"] == "disagreement":
            return self._selection_feedback(observation_id, "destination_disagreement", images)
        geometry = self.point_adapter.fuse_points(selected, same_object="agent paired destination region",
                                                  epoch=self.epoch, role="place")
        ref = "pt_" + uuid4().hex
        self.points[ref] = (self.epoch, geometry)
        self.destination_anchors[ref] = self.destination_anchors[refs[0]].copy()
        self.latest_fused_ref = ref
        result = self.inspect_point_cloud(ref)
        result["image_refs"] = [*images, *result["image_refs"][:2]]
        self._record("select_destination", {**result, "source_point_refs": refs,
                     "anchor_base": self.destination_anchors[ref], "consistency": consistency})
        return result

    def move_to_view(self, point_ref):
        destination = self._point(point_ref).role == "place"
        anchor = self.destination_anchors.get(point_ref)
        result = super().move_to_view(point_ref)
        if destination and result.get("view_status") == "achieved":
            self.destination_viewed = True
            self.destination_view_anchor = anchor.copy()
            self.destination_view_epoch = self.epoch
        return result

    def _move_selected_view(self, connector, geometry, point_ref):
        if geometry.role != "place":
            return super()._move_selected_view(connector, geometry, point_ref)
        from src.tools.place.observation import observe_destination_top
        return observe_destination_top(connector, points=geometry.object_points,
            config=self.motion_config, recorder=self.recorder, point_ref=point_ref)

    def save_destination(self, point_ref):
        from src.tools.perception.multiview import FusedGeometry
        geometry = self._point(point_ref)
        postgrasp = getattr(self, 'postgrasp_destination', False)
        if (self.grasp_attempted and not postgrasp) or geometry.role != "place" or (not postgrasp and not isinstance(geometry, FusedGeometry)):
            raise ValueError("save a pre-pick paired destination selection")
        if not postgrasp and (not self.destination_viewed or self.destination_view_epoch != self.epoch):
            raise ValueError("acquire the destination wrist view before saving")
        if not postgrasp and np.linalg.norm(self.destination_anchors[point_ref] - self.destination_view_anchor) > .15:
            raise ValueError("saved destination differs from the destination viewed by wrist")
        ref = "dest_" + uuid4().hex
        directory = self.output_dir / ref
        directory.mkdir()
        points, scene = geometry.object_points.copy(), geometry.scene_points.copy()
        anchor = self.destination_anchors[point_ref].copy()
        np.savez_compressed(directory / "geometry.npz", destination_points=points, scene_points=scene, anchor=anchor)
        for array in (points, scene, anchor):
            array.setflags(write=False)
        self.destinations[ref] = DestinationMemory(point_ref, geometry.observation_id, self.epoch,
                                                   points, scene, anchor, directory)
        self.selected_destination_ref = ref
        views = self.observation_views[geometry.observation_id]
        metadata = {"destination_ref": ref, "point_ref": point_ref, "epoch": self.epoch,
                    "observation_id": geometry.observation_id, "frame": "connector_base", "units": "metres",
                    "point_count": len(points), "anchor_base": anchor.tolist(),
                    "geometry_sha256": hashlib.sha256((directory / "geometry.npz").read_bytes()).hexdigest(),
                    "source_views": [{"view_id": v["view_id"], "rgb_path": str(self.images.paths[v["image_ref"]])} for v in views],
                    "reuse_policy": "fixed base and static destination across camera/grasp motion; episode-local",
                    "limitations": ["occluded surfaces remain unknown", "destination displacement not tracked"],
                    "destination_view_motion_completed": self.destination_viewed}
        metadata["calibrated_rgbd_snapshot"] = str(self.output_dir / "rgbd" / (geometry.observation_id + "_snapshot.json"))
        if postgrasp:
            metadata.update(acquisition='after grasp, current observation',
                reuse_policy='current post-grasp destination; invalidate/reselect if destination moves',
                destination_view_motion_required=False)
        write_json(directory / "memory.json", metadata)
        self._record("destination_saved", metadata)
        return {"destination_ref": ref, "observation_id": geometry.observation_id,
                "image_refs": ([self.images.add(g.overlay_path) for g in getattr(geometry, 'per_view', (geometry,))]
                               if postgrasp else self.inspect_point_cloud(point_ref)["image_refs"]),
                "memory_policy": "Post-grasp current measured destination; no mandatory observation motion." if postgrasp else "Pre-pick measured destination retained in fixed base; use only if destination remains stationary."}

    def grasp_candidates(self, point_ref):
        if (not self.destinations and not getattr(self, 'postgrasp_destination', False)) or self._point(point_ref).role != "pick":
            raise ValueError("save destination first, then select the pick object")
        geometry = self._point(point_ref)
        self.point_adapter._check_current(geometry.observation_id)
        # _point enforces the same unchanged capture epoch. Store this observed
        # robot geometry before moving, so it cannot become a phantom obstacle.
        from src.tools.motion.planning import _pose_transform
        from src.tools.gripper.state import measured_opening
        tolerance = getattr(self, 'capture_width_tolerance_m', .0005)
        width, raw_qpos, _ = measured_opening(self.connector,
            max_width_m=getattr(self, 'max_gripper_width_m', .08), tolerance_m=tolerance)
        captured = {'observation_id':geometry.observation_id,
            'hand_pose':(_pose_transform(self.connector.get_ee_pose()) @ np.linalg.inv(self.grasp_to_ee)).tolist(),
            'raw_jaw_qpos':raw_qpos, 'modeled_jaw_width_m':width,
            'width_clamp_tolerance_m':tolerance, 'frame':'connector_base',
            'source':'same capture epoch robot proprioception; no scene object state'}
        self.point_adapter._check_current(geometry.observation_id)
        self.scene_grippers[geometry.observation_id] = captured
        write_json(self.output_dir/('scene_gripper_'+geometry.observation_id+'.json'), captured)
        result = super().grasp_candidates(point_ref)
        return result

    def _placement_pool(self, point_ref, destination_ref, object_points):
        key = (point_ref, destination_ref)
        if key not in self.placement_pools:
            destination = self.destinations[destination_ref]
            predictions = self.anyplace.predict(object_points, destination.points, anchor=destination.anchor)
            self.placement_pools[key] = (predictions, self.anyplace.last_output)
        return self.placement_pools[key]

    def validate_grasp(self, candidate_ref):
        result = super().validate_grasp(candidate_ref)
        if not result['accepted']:
            return result
        validation_ref = result['validation_ref']
        plan = self.validations[validation_ref][3]
        point_ref = self.candidate_point_refs[candidate_ref]
        plan.scene_observation_id = self._point(point_ref).observation_id
        plan.scene_point_ref = point_ref
        return result

    def execute_grasp(self, candidate_ref, validation_ref):
        if not self.destinations and not getattr(self, 'postgrasp_destination', False):
            raise ValueError("save destination before grasp execution")
        self.grasp_attachment = None
        self.grasp_jaw_width_m = None
        self.placement_pools.clear()
        self.first_grasp_evidence = {}
        self.grasp_attempted = True
        self.grasped_point_ref = self.candidate_point_refs[candidate_ref]
        result = super().execute_grasp(candidate_ref, validation_ref)
        if self.held_plan is not None:
            closed = self.first_grasp_evidence.get("observations", {}).get("post_close", {})
            if "robot_ee_pose" not in closed:
                raise ValueError("actual close pose missing; attachment unavailable")
            self.grasp_attachment = np.asarray(closed["robot_ee_pose"], dtype=float)
            from src.tools.gripper.state import measured_opening
            self.grasp_jaw_width_m, raw_qpos, _ = measured_opening(self.connector,
                max_width_m=getattr(self, 'max_gripper_width_m', .08))
            qpos = np.asarray(raw_qpos, dtype=float)
            plan = self.held_plan
            directory = self.output_dir / ("attachment_" + uuid4().hex)
            directory.mkdir()
            np.savez_compressed(directory / "attachment.npz", object_points=plan.target_points,
                                grasp_transform=plan.grasp_transform, grasp_to_ee=plan.grasp_to_ee,
                                actual_closed_ee=self.grasp_attachment, measured_jaw_qpos=qpos)
            metadata = {"candidate_ref": candidate_ref, "execution_ref": result["execution_ref"],
                        "point_ref": self.candidate_point_refs[candidate_ref], "frame": "connector_base",
                        "object_cloud_policy": getattr(self, "object_cloud_policy", "source_view"),
                        "object_point_count": len(plan.target_points),
                        "actual_closed_ee": self.grasp_attachment.tolist(),
                        "grasp_transform": plan.grasp_transform.tolist(), "grasp_to_ee": plan.grasp_to_ee.tolist(),
                        "measured_jaw_qpos":qpos.tolist(), "measured_jaw_width_m":self.grasp_jaw_width_m,
                        "release_equation": getattr(self, 'attachment_release_equation', "AnyPlace_relative_transform @ actual_closed_ee"),
                        "assumption": "object cloud stationary until close; rigid hold afterward; slip unmeasured"}
            write_json(directory / "attachment.json", metadata)
            self._record("grasp_attachment_saved", metadata)
        return result

    def place_candidates(self, destination_ref, hold_assessment, destination_assessment):
        if hold_assessment != "held" or destination_assessment != "unchanged":
            raise ValueError("inspect current RGB: held object and unchanged destination required")
        if self.held_plan is None or self.grasp_attachment is None:
            raise ValueError("no executed grasp attachment")
        destination = self.destinations[destination_ref]
        pool_key = (self.grasped_point_ref, destination_ref)
        generation = {
            'status': 'reused' if pool_key in getattr(self, 'placement_pools', {}) else 'requested',
            'destination_ref': destination_ref,
            'instruction_scope': 'place_evaluation',
        }
        predictions, inference_dir = self._placement_pool(self.grasped_point_ref, destination_ref, self.held_plan.target_points)
        generated_count = len(predictions)
        dedup = getattr(self, 'pose_dedup', None)
        if dedup is not None:
            # Compare actual release EE poses, not object delta translations.
            indices, audit = dedup.select(
                [prediction.transform @ self.grasp_attachment for prediction in predictions],
                ids=[prediction.source_index for prediction in predictions])
            self._record('placement_pose_dedup', {**audit, 'destination_ref': destination_ref,
                'point_ref': self.grasped_point_ref, 'comparison_frame': 'release_ee_base'})
            predictions = tuple(predictions[i] for i in indices)
        from src.tools.place.planning import plan_anyplace
        from src.tools.motion.planning import MotionPlanningError
        from src.tools.place.visualization import render_placement, candidate_overviews
        from src.tools.place.release_validation import check_release_geometry, release_goal_ik
        entries, failures, eligible, checked, release_passed = [], [], [], [], []
        state = self._robot_state()
        scene = self._placement_scene(self.held_plan, destination)
        # Collision validity is checked only at release; the original planner
        # must also generate the complete four-goal route before sampling.
        for prediction in predictions:
            geometry = check_release_geometry(self.held_plan.target_points, scene, prediction.transform,
                self.grasp_attachment, self.held_plan.grasp_to_ee, jaw_width_m=self.grasp_jaw_width_m,
                release_clearance_m=self.motion_config.release_clearance_m,
                mesh_source=getattr(self, 'gripper_assets', None))
            checked.append((prediction, geometry))
        geometric_survivors = [(prediction, geometry) for prediction, geometry in checked
                               if geometry['compatible']]
        ik_values = (release_goal_ik(self.connector.ik,
            [geometry['release_ee_pose'] for _, geometry in geometric_survivors],
            jaw_width_m=self.grasp_jaw_width_m) if geometric_survivors else [])
        if len(ik_values) != len(geometric_survivors):
            raise ValueError('release IK result count differs from requested poses')
        ik_by_source = {prediction.source_index: ik for (prediction, _), ik in
                        zip(geometric_survivors, ik_values)}
        for prediction, geometry in checked:
            ik = ik_by_source.get(prediction.source_index,
                {'accepted': False, 'status': 'not_checked', 'reason_code': 'geometry_rejected'})
            compatibility = {**geometry, 'ik': ik, 'compatible': bool(geometry['compatible'] and ik['accepted'])}
            append_json(self.output_dir/'placement_checks.jsonl', {'time_unix_s': time.time(),
                'phase': 'release_goal_only', 'source_index': prediction.source_index, **compatibility})
            if not compatibility['compatible']:
                failures.append({'source_index': prediction.source_index, 'compatibility': compatibility,
                                 'error': 'release goal geometry, static IK or robot self collision rejected'})
                continue
            release_passed.append(prediction.source_index)
            try:
                plan = plan_anyplace(self.connector, grasp_plan=self.held_plan,
                    relative_transform=prediction.transform, closed_ee_pose=self.grasp_attachment,
                    config=self.motion_config, jaw_width_m=self.grasp_jaw_width_m)
            except MotionPlanningError as exc:
                failure = {'source_index':prediction.source_index, 'phase':'four_goal_route',
                           'accepted':False, 'error':str(exc), 'evidence':getattr(exc, 'evidence', None)}
                failures.append(failure)
                append_json(self.output_dir/'placement_checks.jsonl', {'time_unix_s':time.time(), **failure})
                continue
            plan.jaw_width_m = self.grasp_jaw_width_m
            append_json(self.output_dir/'placement_routes.jsonl', {'source_index':prediction.source_index,
                'targets':plan.targets, 'target_labels':plan.target_labels, 'segments':plan.segments,
                'release_self_collision':plan.release_self_collision})
            append_json(self.output_dir/'placement_checks.jsonl', {'time_unix_s':time.time(),
                'phase':'four_goal_route', 'source_index':prediction.source_index, 'accepted':True})
            ref = 'place_' + uuid4().hex
            eligible.append((ref, prediction, plan, compatibility))
        sample_seed = int(self.anyplace.seed) + self.epoch
        chosen = self._sample_place_choices(eligible, seed=sample_seed)
        self._record('placement_candidate_sampling', {'generated_count':generated_count, 'deduplicated_count':len(predictions),
            'release_goal_accepted_source_indices':release_passed,
            'eligible_source_indices':[p.source_index for _,p,_,_ in eligible],
            'offered_source_indices':[p.source_index for _,p,_,_ in chosen],
            'sampling_seed':sample_seed,'maximum_choices':4,'sampling':self.placement_sampling+' after release-goal geometry, static IK, robot self collision and four-goal route generation',
            'validation_policy':'release_goal_and_four_goal_route'})
        for ref,prediction,plan,compatibility in chosen:
            self.place_predictions[ref] = (self.epoch, destination_ref, prediction, state, plan)
            self._record('placement_gripper_compatibility', {'candidate_ref': ref,
                         'source_index': prediction.source_index, **compatibility})
            paths = render_placement(destination, self.held_plan.target_points, prediction.transform,
                                     plan, self.output_dir / ref, grasp_to_ee=self.held_plan.grasp_to_ee, mesh_source=getattr(self, 'gripper_assets', None))
            entries.append({"candidate_ref": ref, "image_refs": [self.images.add(p) for p in paths],
                            "path_description": "Original planner generated lift -> transit -> release -> retreat. Collision validity is checked at the closed-hand release goal only."})
        self._record("anyplace_candidates", {"destination_ref": destination_ref,
                     "candidate_generation": generation,
                     "inference_dir": str(inference_dir), "generated_count": generated_count, "deduplicated_count": len(predictions),
                     "planned_candidates": entries, "planning_failures": failures,
                     "release_clearance_m": self.motion_config.release_clearance_m})
        append_json(self.output_dir / "anyplace_planning.jsonl", {"destination_ref": destination_ref,
                    "generated_count": generated_count, "deduplicated_count": len(predictions), "feasible_count": len(eligible),
                    "offered_count":len(entries), "failures": failures})
        overview = [self.images.add(p) for p in candidate_overviews(entries, self.output_dir)]
        from collections import Counter
        reasons = Counter()
        for failure in failures:
            compatibility = failure.get('compatibility', {})
            if failure.get('phase') == 'four_goal_route':
                reason = 'route_generation_failed'
            elif compatibility.get('reason_code'):
                reason = compatibility['reason_code']
            elif compatibility.get('ik', {}).get('status') == 'not_checked':
                reason = 'release_geometry_rejected'
            elif not compatibility.get('ik', {}).get('accepted', False):
                reason = 'release_ik_or_self_collision_rejected'
            else:
                reason = 'release_geometry_rejected'
            reasons[reason] += 1
        diagnostic_images = []
        if not entries and failures:
            # A rejected proposal is explanatory evidence, never an executable
            # candidate. Do not register it in place_predictions/validations.
            from src.tools.place.planning import preview_anyplace
            rejected_index = failures[0]['source_index']
            rejected = next(p for p in predictions if p.source_index == rejected_index)
            try:
                preview = preview_anyplace(self.connector, relative_transform=rejected.transform,
                    closed_ee_pose=self.grasp_attachment, config=self.motion_config)
                preview.jaw_width_m = self.grasp_jaw_width_m
                preview.validation_label = 'REJECTED / diagnostic only / cannot execute'
                directory = self.output_dir / ('rejected_place_' + uuid4().hex)
                paths = render_placement(destination, self.held_plan.target_points, rejected.transform,
                    preview, directory, grasp_to_ee=self.held_plan.grasp_to_ee, mesh_source=getattr(self, 'gripper_assets', None))
                diagnostic_images = [self.images.add(path) for path in paths[:1]]
            except Exception as exc:
                # Failure to illustrate a rejection must not erase the actual
                # validation result or trigger another model inference.
                self._record('rejected_placement_preview_failed', {'error': repr(exc)})
        overview += diagnostic_images
        feedback = {'generated_count': generated_count, 'deduplicated_count': len(predictions),
                    'geometry_passed_count': len(geometric_survivors),
                    'ik_checked_count': len(ik_values), 'route_passed_count': len(eligible),
                    'rejection_counts': dict(reasons), 'diagnostic_image_refs': diagnostic_images,
                    'diagnostic_images_executable': False,
                    'scope': 'observed geometry, static IK/self collision, and route generation; '
                             'these checks do not identify whether another observation would help'}
        self._record('placement_validation_feedback', {'destination_ref': destination_ref, **feedback})
        return {"candidates": entries, "image_refs": overview,
                "candidate_generation": generation,
                "validation_feedback": feedback,
                "candidate_policy": "Up to four "+self.placement_choice_description+" with a valid third release goal (closed-gripper/held-object scene geometry, static IK and cuRobo robot self collision) AND a complete four-goal route from the original planner. Only passing choices receive candidate references. If none pass, a labeled rejected proposal may be shown as diagnostic evidence and cannot be selected. Every review card pairs saved front/wrist RGB with a calibrated closed-gripper mesh and predicted object cloud, plus a virtual scene view. Compare both views, inspect a choice, then select it. The same stored route is executed after selection; opening and other-goal/path collision checks are not performed."}

    def inspect_place_candidate(self, candidate_ref):
        self._place_candidate(candidate_ref)
        paths = sorted((self.output_dir / candidate_ref).glob("preview-*.png"))
        return {"candidate_ref": candidate_ref, "image_refs": [self.images.add(p) for p in paths]}

    def _place_candidate(self, candidate_ref):
        value = self.place_predictions[candidate_ref]
        if value[0] != self.epoch or self.held_plan is None or value[3] != self._robot_state():
            raise ValueError("placement candidate stale or held object unavailable")
        return value

    def refine_place_candidate(self, candidate_ref, roll_deg, pitch_deg, yaw_deg):
        """Rotate a placement about the jaw centre holding it, then re-check it fully.

        A refinement has no prechecked route, so it earns one the same way an
        original candidate does: release-goal geometry, static IK and a complete
        four-goal route. A rejection leaves the original candidate untouched.
        """
        from dataclasses import replace
        from src.tools.place.planning import plan_anyplace
        from src.tools.place.release_validation import check_release_geometry, release_goal_ik
        from src.tools.place.visualization import render_placement
        from src.tools.motion.planning import MotionPlanningError
        from src.tools.pose_editor.refinement import (
            RefinementError, checked_adjustment, refined_object_transform, tcp_offset_from)

        epoch, destination_ref, prediction, state, _ = self._place_candidate(candidate_ref)
        degrees = lambda values: dict(zip(('roll_deg', 'pitch_deg', 'yaw_deg'), values))
        try:
            step, total = checked_adjustment((roll_deg, pitch_deg, yaw_deg),
                                             self.place_adjustments.get(candidate_ref, (0., 0., 0.)))
        except RefinementError as exc:
            self._record('placement_refinement_rejected',
                         {'candidate_ref': candidate_ref, 'error': str(exc)})
            return {'accepted': False, 'reason_code': 'adjustment_budget_exhausted'}
        grasp_to_ee, clearance = self.held_plan.grasp_to_ee, self.motion_config.release_clearance_m
        transform = refined_object_transform(
            prediction.transform, step, closed_ee=self.grasp_attachment, grasp_to_ee=grasp_to_ee,
            release_clearance_m=clearance, tcp_offset_z_m=getattr(getattr(self, 'gripper_assets', None), 'jaw_center_offset_m', tcp_offset_from(grasp_to_ee)))
        destination = self.destinations[destination_ref]
        geometry = check_release_geometry(
            self.held_plan.target_points, self._placement_scene(self.held_plan, destination), transform,
            self.grasp_attachment, grasp_to_ee, jaw_width_m=self.grasp_jaw_width_m,
            release_clearance_m=clearance, mesh_source=getattr(self, 'gripper_assets', None))
        ik = release_goal_ik(self.connector.ik, [geometry['release_ee_pose']],
                             jaw_width_m=self.grasp_jaw_width_m)[0]
        rejection = None
        if not (geometry['compatible'] and ik['accepted']):
            rejection = geometry.get('reason_code', 'release_goal_ik_rejected')
        else:
            try:
                plan = plan_anyplace(self.connector, grasp_plan=self.held_plan, relative_transform=transform,
                                     closed_ee_pose=self.grasp_attachment, config=self.motion_config,
                                     jaw_width_m=self.grasp_jaw_width_m)
            except MotionPlanningError as exc:
                rejection, plan = 'four_goal_route_failed', None
                append_json(self.output_dir / 'placement_checks.jsonl', {
                    'time_unix_s': time.time(), 'phase': 'refined_four_goal_route',
                    'origin_candidate_ref': candidate_ref, 'accepted': False, 'error': str(exc)})
        if rejection is not None:
            self._record('placement_refinement_rejected', {'origin_candidate_ref': candidate_ref,
                'adjustment_deg': degrees(step), 'reason_code': rejection,
                'compatibility': {**geometry, 'ik': ik}})
            return {'accepted': False, 'reason_code': rejection}
        plan.jaw_width_m = self.grasp_jaw_width_m
        ref = 'place_' + uuid4().hex
        self.place_predictions[ref] = (self.epoch, destination_ref, replace(prediction, transform=transform),
                                       state, plan)
        self.place_adjustments[ref] = total
        append_json(self.output_dir / 'placement_routes.jsonl', {'candidate_ref': ref,
            'origin_candidate_ref': candidate_ref, 'targets': plan.targets,
            'target_labels': plan.target_labels, 'segments': plan.segments,
            'release_self_collision': plan.release_self_collision})
        paths = render_placement(destination, self.held_plan.target_points, transform, plan,
                                 self.output_dir / ref, grasp_to_ee=grasp_to_ee, mesh_source=getattr(self, 'gripper_assets', None))
        self._record('placement_refinement', {'candidate_ref': ref, 'origin_candidate_ref': candidate_ref,
            'adjustment_deg': degrees(step), 'cumulative_deg': degrees(total),
            'pivot': 'jaw centre; rotation applied in the gripper frame', 'poses_modified': True,
            'source_index': prediction.source_index, 'image_paths': paths,
            'validation_policy': 'release_goal_and_four_goal_route'})
        return {'accepted': True, 'candidate_ref': ref, 'adjustment_deg': degrees(step),
                'cumulative_deg': degrees(total), 'image_refs': [self.images.add(p) for p in paths]}

    def validate_place(self, candidate_ref):
        self._place_candidate(candidate_ref)
        ref = "pv_" + uuid4().hex
        self.place_validations[ref] = candidate_ref
        return {"candidate_ref": candidate_ref, "validation_ref": ref, "accepted": True}

    def execute_place_candidate(self, candidate_ref, validation_ref):
        value = self._place_candidate(candidate_ref)
        if self.place_validations.pop(validation_ref) != candidate_ref:
            raise ValueError("placement validation does not match candidate")
        ref = "exec_" + uuid4().hex
        if self.plan_only:
            return {"execution_ref": ref, "status": "unknown"}
        from src.tools.place.planning import execute_anyplace
        # The state/epoch check above binds the exact prechecked route. Do not
        # replan to another, unchecked joint solution after the agent chooses.
        plan = value[4]
        if self.recorder:
            self.recorder.register_plan(plan, kind='anyplace_selected_route', candidate_ref=candidate_ref)
        self.epoch += 1
        self.point_adapter.latest = None
        context = self.recorder.active("execute_place", candidate_ref=candidate_ref, execution_ref=ref) if self.recorder else nullcontext()
        try:
            with context:
                if getattr(self, 'require_release_tracking', False):
                    result = execute_anyplace(self.connector, plan, require_release_tracking=True)
                else:
                    result = execute_anyplace(self.connector, plan)
            append_json(self.output_dir / "private_execution_results.jsonl", {"operation": "anyplace", **result})
            public = {"execution_ref": ref, "status": "succeeded" if result["status"] == "released" else "unknown"}
            public['execution_feedback'] = {
                'release_commanded': bool(result['release_commanded']),
                'retreat_executed': result.get('retreat_executed'),
                'retreat_reason_code': result.get('retreat_reason_code'),
                'task_success': None,
                'release_tracking_check': result.get('release_tracking_check'),
                'tracking_diagnostics': [
                    {'goal': label, 'position_error_m': diagnostic.get('position_error_m'),
                     'orientation_error_rad': diagnostic.get('orientation_error_rad')}
                    for label, diagnostic in zip(getattr(plan, 'target_labels', ()),
                                                 result.get('execution_diagnostics', ()))],
                'interpretation': 'release command execution is not object placement verification; '
                                  'inspect fresh RGB and assess progress toward the task'}
            if result["release_commanded"]:
                self.held_plan = None
                self.grasp_attachment = None
        except Exception as exc:
            public = self._execution_failure("anyplace_execute_place", ref, exc)
            # A failed transit never commands release. Preserve the provisional
            # attachment; clearing it would invite a new grasp of a closed hand.
            released = getattr(plan, 'release_commanded', None)
            public['execution_feedback'] = {'release_commanded': released}
            if released is not False:
                self.held_plan = None
                self.grasp_attachment = None
        self.placement_execution = public
        self._record("anyplace_execution", {**public, "candidate_ref": candidate_ref, "destination_ref": value[1]})
        return public
