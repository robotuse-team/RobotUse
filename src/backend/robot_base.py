"""Sensor, grasp and motion tools; no model conversation or CLI policy."""
from __future__ import annotations
from contextlib import nullcontext
import json
from pathlib import Path
from uuid import uuid4
import numpy as np
from PIL import Image, ImageDraw
from src.utils.logging_utils import append_json


class _ObservationExecutionConnector:
    """Observe execution calls without changing calibration or planning policy."""
    def __init__(self, connector, on_attempt):
        self._connector, self._on_attempt = connector, on_attempt

    def __getattr__(self, name):
        value = getattr(self._connector, name)
        if name == "execute_trajectory" and callable(value):
            def execute(*args, **kwargs):
                self._on_attempt()
                return value(*args, **kwargs)
            return execute
        if name == "tool_registry" and value is not None:
            from types import SimpleNamespace
            def invoke(tool, *args, **kwargs):
                if tool == "robot.execute_trajectory":
                    self._on_attempt()
                return value.invoke(tool, *args, **kwargs)
            return SimpleNamespace(invoke=invoke)
        return value


def render_candidate(geometry, prediction, ref, path, *, expected_open_width_m=None,
                     max_width_m=.08, finger_tip_z_m=.105, finger_base_z_m=.06,
                     jaw_center_offset_m=.105):
    frame = geometry.frame
    image = Image.fromarray(np.asarray(frame.rgb).copy())
    draw = ImageDraw.Draw(image)
    rotation = np.asarray(frame.camera_to_base.rotation)
    translation = np.asarray(frame.camera_to_base.translation)
    intrinsics = np.asarray(frame.intrinsics)
    def project(local):
        world = prediction.pose[:3, :3] @ np.asarray(local) + prediction.pose[:3, 3]
        camera = rotation.T @ (world - translation)
        if camera[2] <= 0:
            return None
        pixel = intrinsics @ camera
        return tuple(float(v) for v in pixel[:2] / pixel[2])
    # Numeric gripper geometry is supplied by the caller, in the grasp frame.
    if isinstance(max_width_m, bool) or not np.isfinite(max_width_m) or max_width_m <= 0:
        raise ValueError('maximum gripper opening must be positive and finite')
    width = max_width_m if expected_open_width_m is None else expected_open_width_m
    if isinstance(width, bool) or not np.isfinite(width) or not 0 <= width <= max_width_m:
        raise ValueError('expected gripper opening outside calibrated range')
    half = width / 2
    anchors = [(-half, 0, finger_tip_z_m), (-half, 0, finger_base_z_m),
               (half, 0, finger_base_z_m), (half, 0, finger_tip_z_m)]
    tip = (0, 0, jaw_center_offset_m)
    if getattr(prediction, "gripper_adapter", None):
        from src.tools.grasp.generator_adapter import GraspGenLiberoAdapter
        anchors = GraspGenLiberoAdapter.finger_points(anchors)
        tip = GraspGenLiberoAdapter.finger_points(tip)
    pixels = [project(p) for p in anchors]
    if all(p is not None for p in pixels):
        draw.line(pixels, fill="cyan", width=3)
    centre, approach = project(tip), project((0, 0, -0.05))
    if centre and approach:
        draw.line((approach, centre), fill="yellow", width=2)
    draw.text((8, 8), ref + " | Grasp proposal", fill="yellow", stroke_width=1, stroke_fill="black")
    opening = (f'NOMINAL OPEN {max_width_m*1000:g}mm; expected opening UNKNOWN' if expected_open_width_m is None else
               f'EXPECTED total joint opening {width*1000:.1f}mm (not measured)')
    draw.text((8, image.height-18), opening, fill='yellow', stroke_width=1, stroke_fill='black')
    image.save(path)


def rank_predictions(predictions, *, downward_weight=1.0, topk=6):
    """Soft-rank actual predictions; never rotate, synthesize, or overwrite scores.

    GraspGen local +Z is the approach axis in connector_base; world down is -Z.
    Preference score = raw discriminator score + weight * downward alignment.
    """
    if not np.isfinite(downward_weight) or downward_weight < 0 or topk < 1:
        raise ValueError("invalid downward ranking configuration")
    ranked = []
    for index, prediction in enumerate(predictions):
        direction = np.asarray(prediction.pose, dtype=float)[:3, 2].copy()
        direction /= np.linalg.norm(direction)  # metadata only; pose is untouched
        alignment = float(np.clip(-direction[2], -1, 1))
        meta = {"source_score": float(prediction.score), "original_rank": index + 1,
            "direction_base": direction.tolist(), "downward_alignment": alignment,
            "downward_angle_deg": float(np.degrees(np.arccos(alignment))),
            "downward_weight": downward_weight,
            "preference_score": float(prediction.score) + downward_weight * alignment}
        ranked.append((prediction, meta))
    ranked.sort(key=lambda entry: -entry[1]["preference_score"])
    for index, (_, meta) in enumerate(ranked):
        meta["published_rank"] = index + 1
    return ranked[:topk]


def robot_pose_matrix(pose):
    """Canonical finite rigid 4x4 EE evidence; connector adapters may return mappings."""
    from collections.abc import Mapping
    from src.tools.motion.planning import _pose_transform, _transform
    return _pose_transform(pose) if isinstance(pose, Mapping) else _transform(pose)


def measured_lift_m(closed_pose, lifted_pose):
    """Measure actual close-to-lift Z displacement, never pre-macro transit height."""
    return float(robot_pose_matrix(lifted_pose)[2, 3] - robot_pose_matrix(closed_pose)[2, 3])


class LiveBackend:
    def __init__(self, *, connector, point_adapter, graspgen, images, output_dir,
                 grasp_to_ee, public_to_planner=None, allow_partial_safety=False, plan_only=False, recorder=None,
                 max_gripper_width_m=.08, capture_width_tolerance_m=.0005,
                 gripper_assets=None, gripper_render_options=None, grasp_opening_policy=None,
                 motion_config=None, prefer_downward_grasps=False, downward_weight=1.0, topk=6, multiview=False, active_perception=False, first_grasp_only=False, contact_manipulation=False):
        self.active_perception, self.first_grasp_only = active_perception, first_grasp_only
        self.contact_manipulation = contact_manipulation
        self.grasp_mode = 'contact' if contact_manipulation else 'transport'
        self.closed_push = False  # A gripper command, never an object-holding claim.
        self.first_grasp_evidence = {}
        self.grasp_execution_evidence = {}
        if public_to_planner is not None:
            from src.tools.motion.planning import PlannerFrameConnector
            connector = PlannerFrameConnector(connector, public_to_planner=public_to_planner)
        # Keep one wrapper identity for validation, proprioception, execution and place.
        # point_adapter retains the RAW capture connector; no pose correction touches RGBD.
        self.connector, self.point_adapter, self.graspgen = connector, point_adapter, graspgen
        self.images, self.output_dir = images, output_dir
        self.grasp_to_ee = grasp_to_ee
        self.gripper_assets = gripper_assets
        self.grasp_opening_policy = grasp_opening_policy
        self.max_gripper_width_m = max_gripper_width_m
        self.capture_width_tolerance_m = capture_width_tolerance_m
        self.gripper_render_options = dict(gripper_render_options or {})
        from src.tools.motion.planning import MotionConfig
        self.motion_config = motion_config or MotionConfig(collision_checks_enabled=False)
        self.prefer_downward_grasps = prefer_downward_grasps and not active_perception
        self.downward_weight, self.topk = downward_weight, topk
        self.allow_partial_safety, self.plan_only = True, plan_only
        self.points, self.candidates, self.validations = {}, {}, {}
        self.waypoints = {}
        self.contact_stroke_refs = set()
        self.push_waypoint_stages = {}
        self.push_contact_height_m = None
        self.recorder = recorder
        self.candidate_point_refs = {}
        self.candidate_adjustments = {}  # refined candidate -> cumulative degrees from its origin
        self.candidate_routes = {}
        self.candidate_open_widths = {}  # total joint opening; None/absent means unknown
        self.candidate_path_scenes = {}
        self._candidate_inspection_cache = {}
        self.epoch = 0
        self.multiview = multiview
        self.latest_observation_id = None
        self.observation_image_refs = {}  # saved sensor evidence, never a recapture for inspection
        self.observation_views = {}
        self._target_selection_cache = {}
        self.high_observation_epoch = None
        self.latest_fused_ref = None
        self.latest_view_ref = None
        self.inspected_cloud_refs = set()
        self.held_plan = None
        self.validated_count = 0

    @property
    def target_intent_mode(self):
        return self.active_perception and self.first_grasp_only

    def _record(self, kind, payload):
        if self.recorder is not None:
            self.recorder.event(kind, payload)

    def observe(self):
        result = self.point_adapter.observe(epoch=self.epoch) if self.multiview else self.point_adapter.observe()
        self.latest_observation_id = result["observation_id"]
        self.latest_fused_ref = None
        self.latest_view_ref = None
        self._target_selection_cache.clear()
        public = {"observation_id": result["observation_id"], "views": [
            {"view_id": entry["view_id"], "image_ref": self.images.add(entry["image_path"])}
            for entry in result["images"]]}
        self.observation_image_refs[result["observation_id"]] = [v["image_ref"] for v in public["views"]]
        self.observation_views[result["observation_id"]] = public["views"]
        self._record("observation", {**public, "source_images": result["images"]})
        return public

    def point(self, observation_id, view_id, u, v, *, role=None):
        geometry = self.point_adapter.point(observation_id, view_id, u, v,
                                            role=role or ("place" if self.held_plan is not None else "pick"))
        ref = "pt_" + uuid4().hex
        self.points[ref] = (self.epoch, geometry)
        if self.recorder is not None:
            from src.tools.perception.rgbd_adapter import normalized_pixel
            frame = geometry.frame
            x, y = normalized_pixel(u, v, frame.width, frame.height)
            depth = float(frame.depth_m[y, x])
            k = frame.intrinsics
            clicked_xyz = None
            if np.isfinite(depth) and 0.02 < depth < 3.0:
                clicked_xyz = frame.camera_to_base.apply(((x-k[0][2])*depth/k[0][0], (y-k[1][2])*depth/k[1][1], depth))
            center_xy = np.median(geometry.object_points[:, :2], axis=0)
            support = geometry.object_points[np.linalg.norm(geometry.object_points[:, :2]-center_xy, axis=1) <= self.motion_config.support_radius_m]
            self._record("point_geometry", {"point_ref": ref, "observation_id": observation_id,
                "view_id": view_id, "u": u, "v": v, "pixel_xy": [x, y],
                "clicked_point_base": clicked_xyz, "clicked_depth_m": depth if np.isfinite(depth) else None,
                "geometry_frame": "connector_base", "role": geometry.role,
                "destination_cloud_median_xy": center_xy, "local_support_height_m": float(support[:, 2].max()) if len(support) else None,
                "raw_mask_path": geometry.mask_path.parent / "sam_mask.png",
                "mask_path": geometry.mask_path, "overlay_path": geometry.overlay_path,
                "point_cloud_path": geometry.mask_path.parent / "observed_points.npz",
                "rgb_path": frame.rgb_path, "object_point_count": len(geometry.object_points)})
        if self.multiview:
            self.latest_fused_ref = None
            self.latest_view_ref = None
        return {"point_ref": ref, "observation_id": observation_id,
                "image_refs": [self.images.add(geometry.overlay_path)]}

    def _saved_observation(self, observation_id=None):
        """Reissue saved image IDs without advancing physics or sensor revisions."""
        observation_id = observation_id or self.latest_observation_id
        return {"observation_id": observation_id, "views": [
            {"view_id": view["view_id"], "image_ref": self.images.add(self.images.paths[view["image_ref"]])}
            for view in self.observation_views.get(observation_id, ())[:2]]}

    def _selection_feedback(self, observation_id, reason_code, image_refs=()):
        saved = self._saved_observation(observation_id)
        return {"observation_id": observation_id, "reason_code": reason_code,
                "image_refs": [*[view["image_ref"] for view in saved["views"]], *image_refs][:4]}

    def select_target(self, observation_id, front_u, front_v, wrist_u, wrist_v):
        """Two clicks express one target intent; existing per-view RGBD does the work."""
        from copy import deepcopy
        from src.tools.perception.multiview import CAMERAS
        from src.tools.perception.rgbd_adapter import normalized_pixel
        from src.tools.perception.intent import INTENT_PROVENANCE, measured_target_consistency
        if not self.target_intent_mode or not self.multiview or self.held_plan is not None:
            return self._selection_feedback(observation_id, "selection_unavailable")
        coordinates = (front_u, front_v, wrist_u, wrist_v)
        try:
            for u, v in ((front_u, front_v), (wrist_u, wrist_v)):
                normalized_pixel(u, v, 2, 2)  # validate before any inference
        except (TypeError, ValueError):
            return self._selection_feedback(observation_id, "invalid_selection")
        try:
            if observation_id != self.latest_observation_id:
                raise ValueError("selection must use the latest observation")
            self.point_adapter._check_current(observation_id)
        except (KeyError, ValueError):
            return self._selection_feedback(observation_id, "stale_observation")
        key = (self.epoch, observation_id, *coordinates)
        self.latest_view_ref = None
        if key in self._target_selection_cache:
            public = deepcopy(self._target_selection_cache[key])
            public["image_refs"] = [self.images.add(self.images.paths[ref]) for ref in public["image_refs"]]
            self.latest_fused_ref = public.get("point_ref")
            self._record("select_target_cache_hit", {"observation_id": observation_id,
                         "point_ref": public.get("point_ref"), "reason_code": public.get("reason_code")})
            return public
        self.latest_fused_ref = None
        image_refs, point_refs = [], []
        try:
            for view_id, (u, v) in zip(CAMERAS, ((front_u, front_v), (wrist_u, wrist_v))):
                selected = self.point(observation_id, view_id, u, v)
                point_refs.append(selected["point_ref"])
                image_refs.extend(selected["image_refs"])
            geometries = [self._point(ref) for ref in point_refs]
            audit = measured_target_consistency(*(geometry.object_points for geometry in geometries))
            self._record("target_intent_consistency", {"observation_id": observation_id,
                         "source_point_refs": point_refs, **audit})
            if audit["status"] == "disagreement":
                public = self._selection_feedback(observation_id, "target_disagreement", image_refs)
            else:
                # The paired action itself is the assertion; no language or ray
                # correspondence is needed to use the established fusion adapter.
                public = self.fuse_points(point_refs, "paired clicks indicate the same intended object")
                public["image_refs"] = [*image_refs, *public["image_refs"][:2]]
                public["provenance"] = INTENT_PROVENANCE
        except Exception as exc:
            self.latest_fused_ref = None
            append_json(self.output_dir / "private_selection_errors.jsonl", {
                "observation_id": observation_id, "source_point_refs": point_refs,
                "exception": repr(exc), "epoch": self.epoch})
            public = self._selection_feedback(observation_id, "selection_failed", image_refs)
        self._target_selection_cache[key] = deepcopy(public)
        self._record("select_target", {**public, "source_point_refs": point_refs,
                     "paired_clicks": list(coordinates), "epoch": self.epoch})
        return public

    def select_view_target(self, observation_id, view_id, u, v):
        """Select measured single-view geometry only to recover another view."""
        from src.tools.perception.multiview import CAMERAS
        from src.tools.perception.rgbd_adapter import normalized_pixel
        if (not (self.target_intent_mode or self.contact_manipulation) or not self.multiview
                or self.held_plan is not None or self.closed_push):
            return self._selection_feedback(observation_id, "selection_unavailable")
        try:
            if view_id not in CAMERAS:
                raise ValueError("unknown camera")
            normalized_pixel(u, v, 2, 2)
        except (TypeError, ValueError):
            return self._selection_feedback(observation_id, "invalid_selection")
        try:
            if observation_id != self.latest_observation_id:
                raise ValueError("latest observation required")
            self.point_adapter._check_current(observation_id)
        except (KeyError, ValueError):
            return self._selection_feedback(observation_id, "stale_observation")
        self.latest_view_ref = self.latest_fused_ref = None
        try:
            public = self.point(observation_id, view_id, u, v)
            self.latest_view_ref = public["point_ref"]
            public.update(source_views=[view_id], provenance="single-view measured RGBD; viewing motion only")
        except Exception as exc:
            append_json(self.output_dir / "private_selection_errors.jsonl", {
                "observation_id": observation_id, "view_id": view_id,
                "exception": repr(exc), "epoch": self.epoch, "purpose": "view_only"})
            public = self._selection_feedback(observation_id, "selection_failed")
        self._record("select_view_target", {**public, "purpose": "view_only"})
        return public

    def move_to_view(self, point_ref):
        """Prime requests an above-target view from the selected measured cloud."""
        if (not (self.target_intent_mode or self.contact_manipulation) or not self.multiview
                or self.held_plan is not None or self.closed_push or self.plan_only):
            return {**self._saved_observation(), "view_status": "requested_view_failed",
                    "reason_code": "view_motion_unavailable"}
        try:
            geometry = self._point(point_ref)
            if point_ref not in (self.latest_fused_ref, self.latest_view_ref):
                raise ValueError("current selected target required")
            self.point_adapter._check_current(geometry.observation_id)
        except (KeyError, ValueError):
            return {**self._saved_observation(), "view_status": "requested_view_failed",
                    "reason_code": "stale_target"}
        attempted = False
        def invalidate_on_attempt():
            nonlocal attempted
            if attempted:
                return
            attempted = True
            self.epoch += 1
            self.validations.clear()
            self.point_adapter.latest = None
            self.latest_observation_id = self.latest_fused_ref = None
            self.latest_view_ref = None
            self.high_observation_epoch = None
            self._target_selection_cache.clear()
        connector = _ObservationExecutionConnector(self.connector, invalidate_on_attempt)
        status, reason_code = "achieved", None
        try:
            context = self.recorder.active("move_to_view", point_ref=point_ref) if self.recorder else nullcontext()
            with context:
                result = self._move_selected_view(connector, geometry, point_ref)
            self._record("target_view_motion", {**result, "point_ref": point_ref, "epoch": self.epoch})
            self.high_observation_epoch = self.epoch
        except Exception as exc:
            status = "requested_view_failed"
            reason_code = "view_motion_failed" if attempted else "view_planning_failed"
            append_json(self.output_dir / "private_observation_errors.jsonl", {
                "point_ref": point_ref, "exception": repr(exc), "epoch": self.epoch,
                "motion_attempted": attempted})
        if attempted:
            try:
                observation = self.observe()
            except Exception as exc:
                # Never return the pre-motion snapshot as current after a
                # partial move whose actual sensor state could not be captured.
                self.latest_observation_id = self.latest_fused_ref = None
                self.point_adapter.latest = None
                append_json(self.output_dir / "private_observation_errors.jsonl", {
                    "point_ref": point_ref, "exception": repr(exc), "epoch": self.epoch,
                    "stage": "refresh_after_view_motion"})
                observation = {"observation_id": None, "views": []}
                status, reason_code = "requested_view_failed", "observation_refresh_failed"
        else:
            observation = self._saved_observation()
        public = {**observation, "view_status": status}
        if reason_code is not None:
            public["reason_code"] = reason_code
        self._record("move_to_view", {**public, "point_ref": point_ref,
                     "motion_attempted": attempted, "epoch": self.epoch})
        return public

    def _move_selected_view(self, connector, geometry, point_ref):
        from src.tools.observation.motion import observe_above_cloud
        return observe_above_cloud(connector, target_points=geometry.object_points,
            config=self.motion_config, holding=False, recorder=self.recorder, point_ref=point_ref)

    def inspect_point_cloud(self, point_ref):
        geometry = self._point(point_ref)
        result = self.point_adapter.inspect_point_cloud(geometry)
        public = {"point_ref": point_ref, "observation_id": geometry.observation_id,
                  "image_refs": [self.images.add(path) for path in result["image_paths"][:2]],
                  "source_views": list(result["source_views"]),
                  "provenance": "Calibrated measured RGBD; model-selected SAM2 masks; source_views identify contributing cameras; no completion or object-pose oracle."}
        self.inspected_cloud_refs.add(point_ref)
        self._record("inspect_point_cloud", {**public, "image_paths": result["image_paths"], "adapter_provenance": result["provenance"]})
        return public

    def fuse_points(self, point_refs, same_object):
        if not self.multiview or self.held_plan is not None:
            raise ValueError("fusion requires multiview pick mode, not placement")
        if len(point_refs) != 2 or len(set(point_refs)) != 2:
            raise ValueError("explicit distinct front and wrist same-object points required")
        if not self.active_perception and self.high_observation_epoch != self.epoch:
            raise ValueError("high observe_target required before final cloud")
        geometries = [self._point(ref) for ref in point_refs]
        if not self.active_perception and any(ref not in self.inspected_cloud_refs for ref in point_refs):
            raise ValueError("inspect both selected clouds before fusion")
        geometry = self.point_adapter.fuse_points(geometries, same_object=same_object, epoch=self.epoch)
        ref = "pt_" + uuid4().hex
        self.points[ref] = (self.epoch, geometry)
        self.latest_fused_ref = ref
        public = self.inspect_point_cloud(ref)
        # Returned image is model-visible now; require an explicit inspection turn
        # before accepting the final reference for grasp generation.
        if not self.active_perception:
            self.inspected_cloud_refs.discard(ref)
        self._record("fuse_points", {**public, "source_point_refs": point_refs, "same_object": same_object,
            "image_paths": [self.images.paths[ref] for ref in public["image_refs"]]})
        return public

    def observe_target(self, point_ref):
        if not self.multiview or self.held_plan is not None:
            raise ValueError("view motion requires multiview input and no active held-object plan")
        geometry = self._point(point_ref)
        from src.tools.perception.multiview import CAMERAS
        if geometry.view_id != CAMERAS[0]:
            raise ValueError("select a fresh front point for high observation")
        if self.plan_only:
            raise ValueError("high observation requires motion; unavailable in plan-only mode")
        from src.tools.observation.motion import observe_above_cloud
        self.epoch += 1  # invalidate BEFORE any attempted motion, including failures
        self.point_adapter.latest = None
        self.latest_observation_id = None
        self.latest_fused_ref = None
        self.high_observation_epoch = None
        context = self.recorder.active("observe_target", point_ref=point_ref) if self.recorder else nullcontext()
        with context:
            result = observe_above_cloud(self.connector, target_points=geometry.object_points,
                                         config=self.motion_config, holding=self.held_plan is not None,
                                         recorder=self.recorder, point_ref=point_ref)
        self._record("observation_motion", {**result, "point_ref": point_ref, "epoch": self.epoch})
        self.high_observation_epoch = self.epoch
        return self.observe()

    def saved_views(self, point_ref):
        geometry = self._point(point_ref)
        return {'image_refs': [self.images.add(self.images.paths[r])
                for r in self.observation_image_refs.get(geometry.observation_id, [])[:2]]}

    def waypoint(self, observation_id, front_u, front_v, wrist_u, wrist_v, reason):
        from src.tools.observation.waypoint import triangulate, render_waypoint
        if self.held_plan is not None and not self.contact_manipulation:
            raise ValueError('observation waypoint unavailable while holding')
        self.point_adapter._check_current(observation_id)
        frames = self.point_adapter.frames[observation_id]
        xyz, audit = triangulate(frames, [(front_u, front_v), (wrist_u, wrist_v)])
        ref = 'wp_' + uuid4().hex
        audit.update(point_base=xyz.tolist(), observation_id=observation_id, epoch=self.epoch, reason=reason)
        paths = render_waypoint(frames, audit, self.output_dir / ref)
        self.waypoints[ref] = (self.epoch, observation_id, xyz)
        self._record('waypoint', {'waypoint_ref': ref, **audit, 'image_paths': paths})
        return {'waypoint_ref': ref, 'observation_id': observation_id,
                'image_refs': [self.images.add(p) for p in paths]}

    def move(self, waypoint_ref):
        from types import SimpleNamespace
        from src.tools.motion.planning import _pose_transform, _plan, _execute_checked, MotionPlanningError
        epoch, observation_id, xyz = self.waypoints[waypoint_ref]
        holding = self.held_plan is not None or self.closed_push
        if epoch != self.epoch or (holding and not self.contact_manipulation) or self.plan_only:
            raise ValueError('waypoint is stale or motion unavailable')
        self.point_adapter._check_current(observation_id)
        target = _pose_transform(self.connector.get_ee_pose())
        target[:3, 3] = xyz  # preserve current orientation; no synthetic grasp pose
        if self.closed_push and self.held_plan is None and self.push_waypoint_stages.get(waypoint_ref) == 'approach':
            # Empty-pusher staging uses a downward diagonal wrist posture. The
            # initial straight wrist can reach contact but lose horizontal IK
            # mobility near the foreground. A held grasp never takes this path.
            c = np.sqrt(.5)
            target[:3, :3] = np.array([[c, c, 0], [c, -c, 0], [0, 0, -1]])
        try:
            if (self.contact_manipulation and self.grasp_mode == 'contact' and holding
                    and waypoint_ref in self.contact_stroke_refs):
                from src.tools.motion.contact import plan_contact_stroke
                segments, targets, *_ = plan_contact_stroke(self.connector, target, self.motion_config,
                    record=lambda attempt: self._record('contact_orientation_attempt',
                        {'waypoint_ref': waypoint_ref, **attempt}))
            else:
                segments, targets, *_ = _plan(self.connector, (target,), np.empty((0,3)), self.motion_config, None)
        except MotionPlanningError:
            # A closed empty pusher may rotate above the support for reachability.
            # Never alter the orientation of a held grasp or a contact stroke.
            current = _pose_transform(self.connector.get_ee_pose())
            if (self.held_plan is not None or not self.closed_push
                    or self.push_waypoint_stages.get(waypoint_ref) != 'contact'
                    or current[2, 3]-target[2, 3] < .10):
                raise
            for angle in (45, -45, 90):
                rad = np.deg2rad(angle)
                yaw = np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]])
                above, contact = current.copy(), target.copy()
                above[:3, :3] = contact[:3, :3] = yaw @ current[:3, :3]
                try:
                    segments, targets, *_ = _plan(self.connector, (above, contact), np.empty((0,3)), self.motion_config, None)
                    self._record('push_approach_orientation', {'waypoint_ref': waypoint_ref, 'yaw_degrees': angle,
                        'source': 'bounded robot IK fallback above observed support; empty closed hand only'})
                    break
                except MotionPlanningError:
                    continue
            else:
                raise MotionPlanningError('closed-pusher approach orientations are unreachable')
        if self.recorder:
            self.recorder.register_plan(SimpleNamespace(segments=segments, targets=targets,
                target_labels=('point_waypoint',)*len(targets), segment_labels=('point_waypoint',)*len(targets), transit_policy='point',
                high_transit_z_m=None), kind='contact' if holding else 'observation', point_ref=waypoint_ref)
        self.epoch += 1
        self.validations.clear(); self.point_adapter.latest = None
        self.latest_fused_ref = self.latest_observation_id = None
        status = 'achieved'
        if self.closed_push and self.push_waypoint_stages.get(waypoint_ref) == 'approach':
            self.push_contact_height_m = None
        try:
            context = self.recorder.active('move', point_ref=waypoint_ref) if self.recorder else nullcontext()
            with context:
                if holding:
                    # Existing trajectory execution retains this gripper command.
                    self.connector.set_gripper(0.0)
                diagnostics = _execute_checked(self.connector, segments, targets,
                    collision_checks_enabled=self.motion_config.collision_checks_enabled)
            error = float(np.linalg.norm(_pose_transform(self.connector.get_ee_pose())[:3,3]-xyz))
            if error > .02:
                status = 'requested_view_failed'
            elif self.closed_push and self.push_waypoint_stages.get(waypoint_ref) == 'contact':
                self.push_contact_height_m = float(xyz[2])
            self._record('waypoint_motion', {'waypoint_ref':waypoint_ref, 'endpoint_error_m':error,
                         'view_status':status, 'holding_command':holding, 'diagnostics':diagnostics})
        except Exception as exc:
            status = 'requested_view_failed'
            append_json(self.output_dir/'private_observation_errors.jsonl', {'waypoint_ref':waypoint_ref,'error':repr(exc)})
        return {**self.observe(), 'view_status':status}

    def turn(self, angle_deg):
        from types import SimpleNamespace
        from src.tools.motion.contact import turn_angle, turn_target
        from src.tools.motion.planning import _pose_transform, _plan, _execute_checked
        angle_deg = turn_angle(angle_deg)
        if (not self.contact_manipulation or self.grasp_mode != 'contact'
                or self.held_plan is None or self.closed_push or self.plan_only):
            from src.core.action_feedback import ActionPreconditionError
            raise ActionPreconditionError('turn_requires_contact_grasp')
        target = turn_target(_pose_transform(self.connector.get_ee_pose()), angle_deg)
        # Same planning policy as contact translation; no IK orientation fallback.
        segments, targets, *_ = _plan(self.connector, (target,), np.empty((0, 3)), self.motion_config, None)
        ref = 'exec_' + uuid4().hex
        if self.recorder:
            self.recorder.register_plan(SimpleNamespace(segments=segments, targets=targets,
                target_labels=('contact_turn',), segment_labels=('contact_turn',),
                transit_policy='contact_turn', high_transit_z_m=None), kind='contact')
        self.epoch += 1
        self.validations.clear(); self.point_adapter.latest = None
        self.latest_fused_ref = self.latest_observation_id = None
        status = 'failed'
        diagnostics = []
        try:
            context = self.recorder.active('turn', execution_ref=ref) if self.recorder else nullcontext()
            with context:
                self.connector.set_gripper(0.0)
                diagnostics = _execute_checked(self.connector, segments, targets,
                    collision_checks_enabled=self.motion_config.collision_checks_enabled)
            # A planned turn with no actual rotation must not report succeeded,
            # even when the existing simulation route disables tracking gates.
            if (diagnostics[-1]['position_error_m'] <= .02
                    and diagnostics[-1]['orientation_error_rad'] <= np.deg2rad(min(3, abs(angle_deg) / 2))):
                status = 'succeeded'
        except Exception as exc:
            append_json(self.output_dir/'private_execution_errors.jsonl',
                        {'operation': 'turn', 'execution_ref': ref, 'error': repr(exc)})
        self._record('contact_turn', {'execution_ref': ref, 'angle_deg': angle_deg,
            'axis': 'current_tcp_local_z', 'target': target.tolist(),
            'status': status, 'diagnostics': diagnostics})
        return {**self.observe(), 'execution_ref': ref, 'status': status}

    def contact_waypoint(self, observation_id, view_id, u, v, reason):
        from src.tools.observation.waypoint import horizontal_contact_point, render_waypoint
        from src.tools.motion.planning import _pose_transform
        if not self.contact_manipulation or (self.held_plan is None and not self.closed_push):
            raise ValueError('a held grasp is required to establish the contact plane')
        self.point_adapter._check_current(observation_id)
        frame = next(f for f in self.point_adapter.frames[observation_id] if f.view_id == view_id)
        current = _pose_transform(self.connector.get_ee_pose())[:3, 3]
        height = self.push_contact_height_m if self.closed_push and self.push_contact_height_m is not None else current[2]
        xyz, audit = horizontal_contact_point(frame, (u, v), height)
        if self.closed_push and self.push_contact_height_m is not None:
            audit['height_source'] = 'established measured support + LIBERO fingertip clearance; no accumulated tracking drift'
        if np.linalg.norm(xyz-current) > .3:
            raise ValueError('choose a contact waypoint within 30 cm of the current grasp')
        ref = 'wp_' + uuid4().hex
        audit.update(point_base=xyz.tolist(), current_ee_base=current.tolist(), observation_id=observation_id, epoch=self.epoch, reason=reason)
        paths = render_waypoint((frame,), audit, self.output_dir/ref)
        self.waypoints[ref] = (self.epoch, observation_id, xyz)
        self._record('waypoint', {'waypoint_ref': ref, **audit, 'image_paths': paths})
        return {'waypoint_ref': ref, 'observation_id': observation_id,
                'image_refs': [self.images.add(p) for p in paths]}

    def contact_destination(self, observation_id, view_id, source_u, source_v, u, v, reason):
        from src.tools.observation.waypoint import object_destination_point, render_waypoint
        from src.tools.motion.planning import _pose_transform
        if not self.contact_manipulation or (self.held_plan is None and not self.closed_push):
            raise ValueError('establish a grasp or push contact before selecting its destination')
        if self.closed_push and self.push_contact_height_m is None:
            raise ValueError('reach push_waypoint stage contact before selecting the object destination')
        self.point_adapter._check_current(observation_id)
        frame = next(f for f in self.point_adapter.frames[observation_id] if f.view_id == view_id)
        current = _pose_transform(self.connector.get_ee_pose())[:3, 3]
        xyz, audit = object_destination_point(frame, (source_u, source_v), (u, v), current,
            contact_height_m=self.push_contact_height_m if self.closed_push else None)
        if np.linalg.norm(xyz-current) > .3:
            raise ValueError('choose an intermediate object destination within 30 cm')
        ref = 'wp_' + uuid4().hex
        audit.update(point_base=xyz.tolist(), observation_id=observation_id, epoch=self.epoch, reason=reason)
        paths = render_waypoint((frame,), audit, self.output_dir/ref)
        self.waypoints[ref] = (self.epoch, observation_id, xyz)
        self.contact_stroke_refs.add(ref)
        self._record('waypoint', {'waypoint_ref': ref, **audit, 'image_paths': paths})
        return {'waypoint_ref': ref, 'observation_id': observation_id,
                'image_refs': [self.images.add(p) for p in paths]}

    def push_waypoint(self, observation_id, view_id, u, v, stage, reason):
        from src.tools.observation.waypoint import measured_push_point, render_waypoint
        from src.tools.motion.planning import _pose_transform
        if not self.contact_manipulation or not self.closed_push or self.held_plan is not None:
            raise ValueError('surface pushing requires closed empty fingers')
        self.point_adapter._check_current(observation_id)
        # This surface offset is calibrated for a downward hand only.
        current = _pose_transform(self.connector.get_ee_pose())
        if current[2, 2] > -.95:
            raise ValueError('surface pushing requires the downward hand orientation')
        frame = next(f for f in self.point_adapter.frames[observation_id] if f.view_id == view_id)
        xyz, audit = measured_push_point(frame, (u, v), stage)
        ref = 'wp_' + uuid4().hex
        self.push_waypoint_stages[ref] = stage
        audit.update(point_base=xyz.tolist(), current_ee_base=current[:3, 3].tolist(), observation_id=observation_id, epoch=self.epoch, reason=reason)
        paths = render_waypoint((frame,), audit, self.output_dir/ref)
        self.waypoints[ref] = (self.epoch, observation_id, xyz)
        self._record('waypoint', {'waypoint_ref': ref, **audit, 'image_paths': paths})
        return {'waypoint_ref': ref, 'observation_id': observation_id,
                'image_refs': [self.images.add(p) for p in paths]}

    def compare_cloud_update(self, previous_ref, current_ref, same_object):
        previous = self.points[previous_ref][1]  # old evidence used only for comparison, never planning
        current = self._point(current_ref)
        report = self.point_adapter.compare_cloud_update(previous, current, same_object=same_object)
        self._record("cloud_update", {"previous_point_ref": previous_ref, "point_ref": current_ref, **report})
        # Paths/calibration private; only measured differences and limitations cross to Grasp.
        fields = ("policy", "previous_observation_id", "observation_id", "previous_epoch", "epoch",
            "previous_point_count", "point_count", "distance_threshold_m", "new_measured_point_count",
            "new_measured_fraction", "previous_points_not_reobserved_fraction", "new_to_previous_distance_m",
            "measured_centroid_shift_m", "missing_surface_coverage_verified", "limitations", "same_object_assertion")
        return json.dumps({key: report[key] for key in fields}, allow_nan=False)

    def _hold_snapshot(self, stage):
        from src.tools.motion.robot_state import _current_robot_state
        from src.tools.perception.multiview import simulation_epoch
        from src.tools.gripper.evidence import read_gripper_fraction
        measurement = read_gripper_fraction(self.connector, stage=stage)
        snapshot = {"gripper_measurement": measurement, "gripper_fraction": measurement["value"]}
        # Attach immediately: a failed RGB/pose/clock read cannot discard another
        # sensor's already observed evidence or erase a preceding stage.
        self.first_grasp_evidence.setdefault("observations", {})[stage] = snapshot
        if stage in ("post_lift", "post_hold"):
            self.first_grasp_evidence.setdefault("gripper_fractions", {})[stage] = measurement["value"]
        errors = {}
        try:
            observation = self.observe()
            snapshot.update(observation=observation,
                image_paths=[str(self.images.paths[v["image_ref"]]) for v in observation["views"]])
        except Exception as exc:
            errors["observation"] = type(exc).__name__
        try:
            pose, joints = _current_robot_state(self.connector)
            snapshot["robot_ee_pose"] = robot_pose_matrix(pose).tolist()
        except Exception as exc:
            errors["robot_pose"] = type(exc).__name__
        try:
            step, seconds = simulation_epoch(self.point_adapter.connector)
            snapshot.update(simulation_time_s=seconds, simulator_step=step)
        except Exception as exc:
            errors["simulation_clock"] = type(exc).__name__
        if errors:
            snapshot["capture_errors"] = errors
            raise RuntimeError("pickup sensor snapshot incomplete")
        return snapshot

    def _point(self, ref):
        epoch, geometry = self.points[ref]
        if epoch != self.epoch:
            raise ValueError("stale point reference; observe and point again after motion")
        if self.multiview and geometry.observation_id != self.latest_observation_id:
            raise ValueError("point is not from the latest synchronized observation")
        return geometry

    def _candidate_path_scene(self, point_ref):
        from src.tools.motion.path_collision import make_candidate_path_collision as CandidatePathCollision
        geometry = self._point(point_ref)
        if point_ref not in self.candidate_path_scenes:
            # A point reference must still describe this stationary capture.
            def check_capture():
                if self.multiview:
                    self.point_adapter._check_current(geometry.observation_id)
                elif self.point_adapter.latest != geometry.observation_id:
                    raise ValueError('stale observation; observe again before candidates')
            check_capture()
            checker = CandidatePathCollision(self.connector,
                clearance_m=getattr(self.graspgen, "official_clearance_m", .002))
            check_capture()
            points = geometry.scene_points
            destinations = getattr(self, 'destinations', {})
            if destinations:
                points = np.vstack((points, *[d.points for d in destinations.values()]))
            scene, _ = checker.remove_captured_robot(points)
            self.candidate_path_scenes[point_ref] = (checker, scene)
        checker, scene = self.candidate_path_scenes[point_ref]
        # The same point can be retried: cached geometry must use the new request's margin.
        checker.clearance_m = getattr(self.graspgen, "official_clearance_m", .002)
        return checker, scene

    def _grasp_options(self, pose, geometry, prediction, candidate_ref):
        from src.tools.grasp.contact import contact_opening
        policy = getattr(self, 'grasp_opening_policy', None)
        if policy is not None:
            opening = policy(pose, geometry.object_points)
        elif getattr(self, 'gripper_assets', None) is not None:
            opening = dict(open_width_m=self.max_gripper_width_m, required_width_m=None,
                method='native gripper full opening; adaptive Panda finger volume is inapplicable')
        else:
            opening = contact_opening(pose, geometry.object_points,
                libero_adapter=bool(getattr(prediction, 'gripper_adapter', None)))
        self.candidate_open_widths[candidate_ref] = opening['open_width_m']
        event = 'contact_grasp_opening' if self.grasp_mode == 'contact' else 'grasp_opening'
        self._record(event, {'candidate_ref': candidate_ref, **opening})
        return {'lift_after_grasp': self.grasp_mode != 'contact',
                'open_width_m': opening['open_width_m']}

    def grasp_candidates(self, point_ref):
        if self.closed_push:
            raise ValueError('release the closed pusher before requesting a grasp')
        if self.active_perception and point_ref != self.latest_fused_ref:
            raise ValueError('current measured front+wrist fusion required')
        geometry = self._point(point_ref)
        if self.multiview:
            if self.held_plan is not None:
                raise ValueError("cannot grasp while holding")
            if (not self.active_perception and (point_ref != self.latest_fused_ref or self.high_observation_epoch != self.epoch
                    or point_ref not in self.inspected_cloud_refs)):
                raise ValueError("inspect latest high-observation fused front+wrist cloud first")
        from src.tools.motion.planning import plan_grasp
        # Use the same observed-cloud padding in the worker and execution plan.
        self.graspgen.open_width_m = None
        self.graspgen.adaptive_contact_opening = True
        result = []

        def accept_candidate(prediction):
            approach = (rank_predictions([prediction], downward_weight=self.downward_weight, topk=1)[0][1]
                        if self.prefer_downward_grasps else None)
            checker, scene = self._candidate_path_scene(point_ref)
            state = self._robot_state()
            ref = "g_" + uuid4().hex
            try:
                plan = plan_grasp(self.connector, grasp_transform=prediction.pose,
                    max_width_m=getattr(self, 'max_gripper_width_m', .08),
                    target_points=geometry.object_points, obstacle_points=geometry.scene_points,
                    grasp_to_ee=self.grasp_to_ee, frame='connector_base', config=self.motion_config,
                    **self._grasp_options(prediction.pose, geometry, prediction, ref))
                # The executor opens the fingers before the approach.
                evidence = checker.check(plan, scene, stop_label='grasp', jaw_width_m=plan.open_width_m)
            except Exception as exc:
                evidence = {'accepted': False, 'error': f'{type(exc).__name__}: {exc}'}
            if not evidence['accepted']:
                return False
            self.candidate_routes[ref] = (state, plan)
            self.candidates[ref] = (self.epoch, geometry, prediction)
            self.candidate_point_refs[ref] = point_ref
            path = self.output_dir / (ref + ".png")
            if self.active_perception:
                from src.tools.pose_editor.inspection import render_candidate_inspection
                inspection = render_candidate_inspection(geometry, prediction, self.output_dir / ref,
                    candidate_ref=ref, graspgen_root=getattr(self.graspgen, "checkout", None), views=1, static_pose=True,
                    expected_open_width_m=plan.open_width_m)
                preview_paths = inspection["image_paths"]
                self._record("candidate_geometry_preview", {"candidate_ref": ref, **inspection})
            else:
                render_candidate(geometry, prediction, ref, path, expected_open_width_m=plan.open_width_m,
                    max_width_m=getattr(self, 'max_gripper_width_m', .08),
                    **getattr(self, 'gripper_render_options', {}))
                preview_paths = [path]
            entry = {"candidate_ref": ref, "image_refs": [self.images.add(p) for p in preview_paths]}
            if approach is not None:
                entry["approach"] = approach
            result.append(entry)
            return True

        collect = getattr(self.graspgen, "predict_with_path_filter", None)
        if callable(collect) and getattr(self.graspgen, "target_candidates", 0):
            def rank_batch(predictions):
                if self.prefer_downward_grasps and predictions:
                    return [p for p, _ in rank_predictions(predictions,
                        downward_weight=self.downward_weight, topk=len(predictions))]
                return predictions
            predictions = collect(geometry.object_points, geometry.scene_points,
                                  accept_candidate, rank_batch=rank_batch)
        else:
            predictions = self.graspgen.predict(geometry.object_points, geometry.scene_points)
            ranked = (rank_predictions(predictions, downward_weight=self.downward_weight,
                        topk=min(self.topk, 6) if self.multiview else self.topk)
                      if self.prefer_downward_grasps else [(p, None) for p in
                        (predictions[:6] if self.multiview else predictions)])
            for prediction, _ in ranked:
                accept_candidate(prediction)
        self._record("grasp_candidates", {"point_ref": point_ref, "candidates": [
            {**entry, "score": self.candidates[entry["candidate_ref"]][2].score,
             "grasp_transform": self.candidates[entry["candidate_ref"]][2].pose,
             "image_paths": [self.images.paths[image_ref] for image_ref in entry["image_refs"]]}
            for entry in result], "geometry_frame": "connector_base",
            "ranking": "source_score + downward_weight * dot(local_+Z, base_-Z)" if self.prefer_downward_grasps else "source_score",
            "prediction_pool_count": len(predictions),
            "poses_modified": any(getattr(p, "gripper_adapter", None) for p in predictions)})
        public = {"candidates": result}
        if self.multiview:
            # Six candidates + current two RGBs fit the provider image budget.
            public["image_refs"] = self.saved_views(point_ref)["image_refs"]
        return public

    def inspect_candidate(self, candidate_ref):
        self._latest_candidate(candidate_ref)
        epoch, geometry, prediction = self.candidates[candidate_ref]
        if epoch != self.epoch:
            raise ValueError("stale candidate")
        if candidate_ref in self._candidate_inspection_cache:
            # Candidate geometry is immutable. Reuse its original evidence;
            # never redraw over files already referenced by an earlier event.
            paths = self._candidate_inspection_cache[candidate_ref]
            self._record('candidate_inspection_cache_hit', {'candidate_ref': candidate_ref})
            return {'candidate_ref': candidate_ref, 'image_refs': [self.images.add(p) for p in paths]}
        from src.tools.pose_editor.inspection import render_candidate_inspection
        inspection = render_candidate_inspection(geometry, prediction, self.output_dir / (candidate_ref + "_inspection"),
            candidate_ref=candidate_ref, graspgen_root=getattr(self.graspgen, "checkout", None), views=2, static_pose=True,
            expected_open_width_m=self.candidate_routes[candidate_ref][1].open_width_m)
        self._record("candidate_geometry_inspection", {"candidate_ref": candidate_ref, **inspection})
        # Restore saved whole-scene/arm context without observe(), new epochs, or physics.
        rgb_refs = [self.images.add(self.images.paths[ref])
                    for ref in self.observation_image_refs.get(geometry.observation_id, [])[:2]]
        # Fresh image IDs make reused saved RGB current under the provider image budget.
        cloud_refs = self.inspect_point_cloud(self.candidate_point_refs[candidate_ref])["image_refs"][:2]
        mesh_refs = [self.images.add(p) for p in inspection["image_paths"][:2]]
        refs = [*rgb_refs, *mesh_refs, *cloud_refs]
        self._candidate_inspection_cache[candidate_ref] = [self.images.paths[r] for r in refs]
        return {"candidate_ref": candidate_ref, "image_refs": refs}

    def preview_candidate(self, candidate_ref, azimuth_deg, elevation_deg, zoom):
        """Orbit/zoom the saved scene without changing a pose, epoch or robot state."""
        from src.tools.pose_editor.preview_controls import render_preview
        self._latest_candidate(candidate_ref)
        epoch, geometry, prediction = self.candidates[candidate_ref]
        if epoch != self.epoch:
            raise ValueError('stale candidate')
        route = self.candidate_routes.get(candidate_ref)
        expected_width = (route[1].open_width_m if route is not None else
                          self.candidate_open_widths.get(candidate_ref))
        rendered = render_preview(geometry, prediction,
            self.output_dir / ('preview_' + uuid4().hex), candidate_ref=candidate_ref,
            azimuth_deg=azimuth_deg, elevation_deg=elevation_deg, zoom=zoom,
            graspgen_root=getattr(self.graspgen, 'checkout', None),
            expected_open_width_m=expected_width)
        self._record('candidate_orbit_preview', {**rendered['metadata'], 'image_paths': rendered['image_paths']})
        return {'candidate_ref': candidate_ref,
                'image_refs': [self.images.add(p) for p in rendered['image_paths']]}

    def _robot_state(self):
        # Robot proprioception only; no object state or simulator geometry.
        from src.tools.motion.robot_state import _current_robot_state
        pose, joints = _current_robot_state(self.connector)
        return json.dumps({"pose": pose, "joints": joints}, sort_keys=True, default=lambda x: np.asarray(x).tolist())

    def _latest_candidate(self, candidate_ref):
        if self.multiview:
            ref = self.candidate_point_refs[candidate_ref]
            self._point(ref)
            if self.target_intent_mode and ref != self.latest_fused_ref:
                raise ValueError("candidate is not bound to current selected target")
            if not self.active_perception and (ref != self.latest_fused_ref or ref not in self.inspected_cloud_refs):
                raise ValueError("candidate is not bound to latest inspected fused cloud")

    def refine_candidate(self, candidate_ref, roll_deg, pitch_deg, yaw_deg, dx_mm=0., dy_mm=0., dz_mm=0.):
        """Derive a small rotation of an existing candidate about its jaw centre.

        The original candidate is left intact and the refinement becomes an
        ordinary candidate: its new route passes the same generation-time
        checks, and validate_grasp verifies that route is still current.
        """
        from dataclasses import replace
        from src.tools.motion.planning import plan_grasp
        from src.tools.pose_editor.refinement import (
            RefinementError, checked_adjustment, preview_refined_pose, refined_grasp_pose,
            tcp_offset_from)
        self._latest_candidate(candidate_ref)
        epoch, geometry, prediction = self.candidates[candidate_ref]
        if epoch != self.epoch:
            raise ValueError("stale candidate")
        try:
            step, total = checked_adjustment((roll_deg, pitch_deg, yaw_deg),
                                             self.candidate_adjustments.get(candidate_ref, (0., 0., 0.)))
        except RefinementError as exc:
            self._record("candidate_refinement_rejected", {"candidate_ref": candidate_ref, "error": str(exc)})
            return {"accepted": False, "reason_code": "adjustment_budget_exhausted"}
        offset = getattr(getattr(self, 'gripper_assets', None), 'jaw_center_offset_m', tcp_offset_from(self.grasp_to_ee))
        pose = refined_grasp_pose(prediction.pose, step, tcp_offset_z_m=offset)
        from src.tools.pose_editor.geometry import checked_translation
        previous_translation = getattr(self, 'candidate_translations', {}).get(candidate_ref, (0., 0., 0.))
        translation, total_translation = checked_translation((dx_mm, dy_mm, dz_mm), previous_translation)
        pose[:3, 3] += prediction.pose[:3, :3] @ (np.asarray(translation) / 1000.)
        ref = "g_" + uuid4().hex
        # Render first: a refinement the reviewer cannot be shown is not a candidate,
        # and must not be left behind as one.
        grasp_options = self._grasp_options(pose, geometry, prediction, ref)
        self.candidate_open_widths[ref] = grasp_options['open_width_m']
        paths = preview_refined_pose(self.point_adapter.frames[geometry.observation_id], pose,
                                     self.output_dir / ref, tcp_offset_z_m=offset,
                                     expected_open_width_m=grasp_options['open_width_m'],
                                     mesh_source=getattr(self, 'gripper_assets', None))
        point_ref = self.candidate_point_refs[candidate_ref]
        # Rotating a pose invalidates its original route. Check a new route
        # against the captured scene, including saved destination geometry,
        # before publishing the refinement just as grasp_candidates does.
        try:
            checker, scene = self._candidate_path_scene(point_ref)
            state = self._robot_state()
            plan = plan_grasp(self.connector, grasp_transform=pose,
                max_width_m=getattr(self, 'max_gripper_width_m', .08),
                target_points=geometry.object_points, obstacle_points=geometry.scene_points,
                grasp_to_ee=self.grasp_to_ee, frame='connector_base', config=self.motion_config,
                **grasp_options)
            evidence = checker.check(plan, scene, stop_label='grasp', jaw_width_m=plan.open_width_m)
        except Exception as exc:
            evidence = {'accepted': False, 'error': f'{type(exc).__name__}: {exc}'}
        if not evidence['accepted']:
            self._record('candidate_refinement_rejected', {'candidate_ref': candidate_ref,
                'reason_code': 'candidate_path_rejected', 'path_check': evidence})
            return {'accepted': False, 'reason_code': 'candidate_path_rejected'}
        self.candidates[ref] = (self.epoch, geometry, replace(prediction, pose=pose))
        self.candidate_point_refs[ref] = point_ref
        self.candidate_routes[ref] = (state, plan)
        self.candidate_adjustments[ref] = total
        if not hasattr(self, 'candidate_translations'):
            self.candidate_translations = {}
        self.candidate_translations[ref] = total_translation
        degrees = lambda values: dict(zip(("roll_deg", "pitch_deg", "yaw_deg"), values))
        self._record("candidate_refinement", {"candidate_ref": ref, "origin_candidate_ref": candidate_ref,
            "adjustment_deg": degrees(step), "cumulative_deg": degrees(total),
            "translation_local_mm": translation, "cumulative_translation_local_mm": total_translation,
            "pivot": "jaw centre; rotation applied in the gripper frame", "poses_modified": True,
            "grasp_transform": pose.tolist(), "image_paths": paths})
        return {"accepted": True, "candidate_ref": ref, "adjustment_deg": degrees(step),
                "cumulative_deg": degrees(total), "image_refs": [self.images.add(p) for p in paths]}

    def validate_grasp(self, candidate_ref):
        self._latest_candidate(candidate_ref)
        epoch, geometry, prediction = self.candidates[candidate_ref]
        if epoch != self.epoch:
            raise ValueError("stale candidate")
        validation_ref = "v_" + uuid4().hex
        try:
            state = self._robot_state()
            checked_state, plan = self.candidate_routes[candidate_ref]
            if checked_state != state:
                raise ValueError('candidate path is stale; generate candidates again')
        except Exception as exc:
            append_json(self.output_dir / "private_planner_errors.jsonl", {
                "candidate_ref": candidate_ref, "error": f"{type(exc).__name__}: {exc}"})
            public = {"candidate_ref": candidate_ref, "validation_ref": validation_ref, "accepted": False}
            if self.target_intent_mode:
                public.update(reason_code="planning_failed",
                    image_refs=self.saved_views(self.candidate_point_refs[candidate_ref])["image_refs"])
            return public
        if self.recorder is not None:
            self.recorder.register_plan(plan, kind="grasp", candidate_ref=candidate_ref,
                point_ref=self.candidate_point_refs.get(candidate_ref))
        self.validations[validation_ref] = (self.epoch, candidate_ref, state, plan)
        self.validated_count += 1
        return {"candidate_ref": candidate_ref, "validation_ref": validation_ref, "accepted": True}

    def execute_grasp(self, candidate_ref, validation_ref):
        """Archive each started attempt once, before later grasps replace live evidence."""
        from copy import deepcopy
        previous = self.first_grasp_evidence
        result = None
        try:
            result = self._execute_grasp(candidate_ref, validation_ref)
            return result
        finally:
            evidence = self.first_grasp_evidence
            if evidence is not previous and evidence.get('execution_ref'):
                ref = evidence['execution_ref']
                snapshot = deepcopy(evidence)
                snapshot.update(candidate_ref=candidate_ref,
                    point_ref=self.candidate_point_refs.get(candidate_ref),
                    command_status=(result or {}).get('status', 'unknown'),
                    scope='sensor and command evidence; no native outcome or physical hold inference')
                if isinstance(result, dict) and type(result.get('terminal')) is bool:
                    snapshot['terminal'] = result['terminal']
                directory = self.output_dir / 'grasp_executions'
                directory.mkdir(exist_ok=True)
                # Exclusive creation makes accidental reuse fail visibly.
                with (directory / (ref + '.json')).open('x') as stream:
                    json.dump(snapshot, stream, indent=2, allow_nan=False)
                    stream.write('\n')
                if not hasattr(self, 'grasp_execution_evidence'):
                    self.grasp_execution_evidence = {}
                self.grasp_execution_evidence[ref] = snapshot
                append_json(self.output_dir / 'grasp_executions.jsonl', dict(
                    execution_ref=ref, candidate_ref=candidate_ref,
                    command_status=snapshot['command_status'],
                    evidence_path=str(directory / (ref + '.json'))))

    def first_grasp_diagnostic_evidence(self):
        """Return a detached snapshot of the first attempt, never the latest grasp."""
        from copy import deepcopy
        records = getattr(self, 'grasp_execution_evidence', {})
        return deepcopy(next(iter(records.values()), {}))

    def _execute_grasp(self, candidate_ref, validation_ref):
        from src.tools.motion.planning import execute_grasp
        self._latest_candidate(candidate_ref)
        epoch, bound_ref, state, plan = self.validations.pop(validation_ref)
        if epoch != self.epoch or candidate_ref != bound_ref or state != self._robot_state():
            raise ValueError("validation is stale or robot moved")
        ref = "exec_" + uuid4().hex
        self.first_grasp_evidence = dict(execution_ref=ref, execution_status='unknown')
        if self.plan_only:
            return {"execution_ref": ref, "status": "unknown"}
        try:
            if self.first_grasp_only:
                self._hold_snapshot("pre_grasp")
            self.epoch += 1
            self.point_adapter.latest = None
            self.held_plan = None  # the explicit new grasp attempt supersedes a provisional hold
            context = self.recorder.active("execute_grasp", candidate_ref=candidate_ref, validation_ref=validation_ref, execution_ref=ref) if self.recorder else nullcontext()
            with context:
                kwargs = {"on_closed": lambda: self._hold_snapshot("post_close")} if self.first_grasp_only else {}
                result = execute_grasp(self.connector, plan, allow_partial_safety=self.allow_partial_safety, **kwargs)
            append_json(self.output_dir / "private_execution_results.jsonl", result)
            status = {"executed": "succeeded", "released": "succeeded"}.get(result.get("status"), result.get("status", "unknown"))
            if status == "succeeded":
                self.held_plan = plan
            if self.first_grasp_only:
                self.first_grasp_evidence.update(execution_ref=ref, execution_status=status,
                    macro_proprioception={key: result.get(key) for key in
                        ("grasp_evidence", "gripper_fraction", "gripper_measurement")})
                lifted = self._hold_snapshot("post_lift")
                closed = self.first_grasp_evidence["observations"]["post_close"]
                self.first_grasp_evidence["measured_lift_m"] = measured_lift_m(closed["robot_ee_pose"], lifted["robot_ee_pose"])
                # Existing close primitive holds current robot joints during settling.
                hold_context = self.recorder.active("verification_hold", execution_ref=ref) if self.recorder else nullcontext()
                with hold_context:
                    self.connector.close_gripper(settle_steps=20)
                self.first_grasp_evidence["hold_settle_steps"] = 20
                held = self._hold_snapshot("post_hold")
                self.first_grasp_evidence["measured_hold_s"] = held['simulation_time_s'] - lifted['simulation_time_s']
                self.first_grasp_evidence["hold_duration_source"] = 'actual simulator clocks'
        except Exception as exc:
            return self._execution_failure("execute_grasp", ref, exc)
        public = {"execution_ref": ref, "status": status if status in ("succeeded", "failed", "unknown") else "unknown"}
        feedback = getattr(self, '_public_grasp_feedback', None)
        return feedback(public, result) if feedback is not None else public

    def simulation_budget(self):
        from src.runtime.budget import simulation_budget
        return simulation_budget(getattr(self, 'connector', None))

    def _execution_failure(self, operation, execution_ref, exc):
        # Private details never cross the public orchestration/model boundary.
        # Exception only: KeyboardInterrupt must unwind to main without a retry.
        macro = getattr(exc, "evidence", None)
        # A later grasp replaces first_grasp_evidence. Bind the original failed
        # measurement to this execution before recovery can replace that state.
        failure_evidence = {key: macro[key] for key in (
            "gripper_measurement", "gripper_fraction", "motion_completed",
            "execution_stage") if key in macro} if isinstance(macro, dict) else {}
        append_json(self.output_dir / "private_execution_errors.jsonl", {
            "operation": operation, "execution_ref": execution_ref,
            "exception": repr(exc), "epoch": self.epoch,
            **({"failure_evidence": failure_evidence} if failure_evidence else {})})
        public = {"execution_ref": execution_ref, "status": "failed"}
        from src.runtime.budget import MotionBudgetExceeded
        budget = self.simulation_budget()
        if budget is not None:
            if budget['terminal']:
                public.update(terminal=True, reason_code=budget['reason_code'])
            elif isinstance(exc, MotionBudgetExceeded):
                public.update(reason_code='insufficient_simulation_time',
                    required_simulation_s=exc.required_steps * budget['step_duration_s'])
        if self.first_grasp_only and operation == "execute_grasp":
            self.first_grasp_evidence.update(execution_ref=execution_ref, execution_status="failed")
            if isinstance(macro, dict):
                # Keep the original failed read even if the next sample recovers.
                self.first_grasp_evidence["macro_proprioception"] = macro
            try:
                snapshot = self._hold_snapshot("execution_failure")
                if isinstance(macro, dict) and macro.get("motion_completed") is True:
                    snapshot["reached_stage"] = macro.get("execution_stage")
            except Exception as capture_exc:
                append_json(self.output_dir / "private_execution_errors.jsonl", {
                    "operation": "capture_execution_failure", "execution_ref": execution_ref,
                    "exception": repr(capture_exc), "epoch": self.epoch})
            if self.target_intent_mode:
                observations = self.first_grasp_evidence.get("observations", {})
                latest = observations.get("execution_failure", {}).get("observation", {})
                public.setdefault('reason_code', 'execution_failed')
                public['image_refs'] = [view["image_ref"] for view in latest.get("views", ())[:2]]
        return public

    def close_for_push(self):
        from src.runtime.budget import gripper_settle_steps
        if not self.contact_manipulation or self.held_plan is not None or self.plan_only:
            raise ValueError('closed-finger pushing requires no held grasp')
        ref = 'exec_' + uuid4().hex
        self.epoch += 1
        self.validations.clear()
        self.point_adapter.latest = None
        self.latest_fused_ref = self.latest_observation_id = None
        try:
            context = self.recorder.active('close_for_push', execution_ref=ref) if self.recorder else nullcontext()
            with context:
                self.connector.close_gripper(settle_steps=gripper_settle_steps(self.connector, 'close', 40))
            self.closed_push = True
            self.push_contact_height_m = None
        except Exception as exc:
            return self._execution_failure('close_for_push', ref, exc)
        return {'execution_ref': ref, 'status': 'succeeded'}

    def release(self):
        from src.runtime.budget import gripper_settle_steps
        if not self.contact_manipulation or self.plan_only:
            raise ValueError('release requires a current contact grasp')
        ref = 'exec_' + uuid4().hex
        self.epoch += 1
        self.validations.clear()
        self.point_adapter.latest = None
        self.latest_fused_ref = self.latest_observation_id = None
        self.held_plan = None
        self.closed_push = False
        self.push_contact_height_m = None
        try:
            context = self.recorder.active('release', execution_ref=ref) if self.recorder else nullcontext()
            with context:
                self.connector.open_gripper(settle_steps=gripper_settle_steps(self.connector, 'open', 40))
        except Exception as exc:
            return self._execution_failure('release', ref, exc)
        return {'execution_ref': ref, 'status': 'succeeded'}

    def place(self, point_ref):
        if self.grasp_mode == 'contact':
            raise ValueError("contact manipulation uses held waypoints, not placement")
        if self.first_grasp_only:
            raise ValueError("placement disabled in first-grasp mode")
        from src.tools.motion.planning import plan_place, execute_place
        if self.held_plan is None:
            raise ValueError("no successfully held grasp")
        geometry = self._point(point_ref)
        ref = "exec_" + uuid4().hex
        try:
            plan = plan_place(self.connector, grasp_plan=self.held_plan,
                destination_points=geometry.object_points, obstacle_points=geometry.scene_points, frame="connector_base", config=self.motion_config)
        except Exception as exc:
            return self._execution_failure("plan_place", ref, exc)
        if self.recorder is not None:
            self.recorder.register_plan(plan, kind="place", point_ref=point_ref)
        if self.plan_only:
            return {"execution_ref": ref, "status": "unknown"}
        self.epoch += 1
        self.point_adapter.latest = None
        try:
            context = self.recorder.active("execute_place", point_ref=point_ref, execution_ref=ref) if self.recorder else nullcontext()
            with context:
                result = execute_place(self.connector, plan, allow_partial_safety=self.allow_partial_safety)
        except Exception as exc:
            if getattr(plan, 'release_commanded', None) is not False:
                self.held_plan = None  # opening or retreat may have partially completed
            return self._execution_failure("execute_place", ref, exc)
        append_json(self.output_dir / "private_execution_results.jsonl", result)
        status = {"executed": "succeeded", "released": "succeeded"}.get(result.get("status"), result.get("status", "unknown"))
        if status == "succeeded":
            self.held_plan = None
        return {"execution_ref": ref, "status": status if status in ("succeeded", "failed", "unknown") else "unknown"}
