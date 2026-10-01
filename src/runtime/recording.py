"""Instance-local, streaming dual-camera simulator evidence (never model input).

Captures frames through the simulator's post-control-step recording callback.
Direct render and calibration reads never advance simulation.
No simulator object poses/segmentation are queried or logged. All geometry uses
connector_base, reproducing the calibrated robot-only planner EE transform.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import time
from typing import Any
from uuid import uuid4
import numpy as np

from src.tools.perception.runtime import quaternion_camera_to_base


class RecorderError(RuntimeError):
    pass


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


class AgentInterfaceRecorder:
    def __init__(self, *, connector, ee_connector, output_dir, objective, model,
                 every_n_steps=2, writer_factory=None, preview_dir=None):
        if type(every_n_steps) is not int or every_n_steps < 1:
            raise ValueError("recording stride must be a positive control-step count")
        self.connector, self.ee_connector, self.env = connector, ee_connector, connector.env
        self.public_to_planner = np.eye(4) if ee_connector is connector else np.asarray(ee_connector._public_to_planner, dtype=float)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.every_n_steps = every_n_steps
        self.control_freq = float(self.env._control_freq)
        if not np.isfinite(self.control_freq) or self.control_freq <= 0:
            raise RecorderError("missing finite simulator control frequency")
        self.fps = self.control_freq / every_n_steps
        self.objective, self.model = objective, model
        self.writer_factory = writer_factory
        self.preview = None
        if preview_dir is not None:
            from src.runtime.live_preview import LivePreview
            self.preview = LivePreview(preview_dir)
        self.writers, self.video_counts = {}, {"front": 0, "wrist": 0}
        self.frame_count, self.event_count = 0, 0
        self.last_step = None
        self.operation, self.segment_id, self.plan_ref = "initial", None, None
        self._segments, self._restorations = {}, []
        self.error, self.closed, self.installed = None, False, False
        self.start_wall_time = time.time()
        self._write_manifest("created")

    def _append(self, filename, payload):
        with (self.output_dir / filename).open("a") as stream:
            stream.write(json.dumps(payload, default=_json, allow_nan=False) + "\n")

    def _clock(self):
        step = int(self.env._sim_step_count)
        from src.runtime.clock import simulation_time_s
        simulation_time = simulation_time_s(self.env)
        return {"simulator_step": step, "simulation_time_s": simulation_time,
                "wall_time_unix_s": time.time(), "monotonic_ns": time.monotonic_ns()}

    def _write_manifest(self, status):
        (self.output_dir / "manifest.json").write_text(json.dumps({
            "schema": "agent-interface.manifest.v1", "status": status, "error": self.error,
            "objective": self.objective, "model": self.model, "geometry_frame": "connector_base",
            "control_dt_s": 1 / self.control_freq, "control_frequency_hz": self.control_freq,
            "video_fps": self.fps, "every_n_control_steps": self.every_n_steps,
            "sampling_policy": "initial frame then every N actual simulator control steps; no added physics steps",
            "time_policy": "simulation_time_s from environment clock; no inference-wait frames/audio",
            "calibration_policy": "per-frame calibrated robot/camera matrices in connector_base",
            "ee_policy": "fresh environment-profile end-effector matrix in connector_base; optional measured public_to_planner",
            "paired_frames": self.frame_count, "video_frame_counts": self.video_counts,
            "streams": {"front": "front.mp4", "wrist": "wrist.mp4"},
            "telemetry": "frames.jsonl", "events": "events.jsonl", "plans": "plans.jsonl",
            "images_are_recorder_only": True, "start_wall_time_unix_s": self.start_wall_time,
        }, indent=2, allow_nan=False) + "\n")

    def event(self, kind, payload):
        if self.closed:
            raise RecorderError("recorder already closed")
        self._append("events.jsonl", {"schema": "agent-interface.event.v1",
            "event_index": self.event_count, "kind": kind,
            "frame_index": self.frame_count - 1, "next_frame_index": self.frame_count,
            **self._clock(), "operation": self.operation,
            "segment_id": self.segment_id, "plan_ref": self.plan_ref, "payload": payload})
        self.event_count += 1

    def _patch(self, owner, name, replacement):
        existed = name in vars(owner)
        previous = vars(owner).get(name)
        self._restorations.append((owner, name, existed, previous))
        setattr(owner, name, replacement)

    def install(self):
        try:
            self._install_impl()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self._write_manifest("failed")
            raise

    def _install_impl(self):
        if self.installed:
            raise RecorderError("recorder already installed")
        if not callable(getattr(self.env, "_record_frame", None)):
            raise RecorderError("unsupported simulator: no post-control-step record callback")
        self._patch(self.env, "_record_frame", lambda: self.capture())
        self._patch(self.env, "_record_frames", True)
        # Invoke hook every step; capture itself applies explicit sampling stride.
        self._patch(self.env, "_subsample_rate", 1)
        refresh = getattr(self.env, "refresh_camera_obs", None)
        if callable(refresh):
            def refresh_existing(*args, **kwargs):
                result = refresh(*args, **kwargs)
                self.capture()
                return result
            self._patch(self.env, "refresh_camera_obs", refresh_existing)
        execute = getattr(self.connector, "execute_trajectory", None)
        def around_segment(segment, call):
            binding = self._segments.get(id(segment))
            previous = self.segment_id, self.plan_ref
            self.segment_id, self.plan_ref = binding if binding else ("unregistered_segment", None)
            self.event("segment_start", {"segment_id": self.segment_id, "plan_ref": self.plan_ref})
            try:
                return call()
            finally:
                self.event("segment_end", {"segment_id": self.segment_id, "plan_ref": self.plan_ref})
                self.segment_id, self.plan_ref = previous
        if callable(execute):
            def execute_recorded(segment, *args, **kwargs):
                return around_segment(segment, lambda: execute(segment, *args, **kwargs))
            self._patch(self.connector, "execute_trajectory", execute_recorded)
        else:
            # Real SimConnector currently dispatches this through its registry.
            # Do NOT add execute_trajectory: that would change motion's branch.
            registry = self.connector.tool_registry
            invoke = registry.invoke
            def invoke_recorded(name, *args, **kwargs):
                if name != "robot.execute_trajectory":
                    return invoke(name, *args, **kwargs)
                return around_segment(kwargs.get("trajectory"), lambda: invoke(name, *args, **kwargs))
            self._patch(registry, "invoke", invoke_recorded)
        self.installed = True
        self.capture(force=True)
        self.event("instruction", {"instruction": self.objective})
        self._write_manifest("recording")

    def _robot_camera_snapshot(self):
        """Fresh robot/camera-only matrices: independent of disabled RGB caches.

        Read native camera frames without refresh and the calibrated robot EE pose.
        The MuJoCo-compatible path derives the same matrices from robot links.
        Neither path advances physics or reads scene-object transforms.
        """
        from src.tools.motion.planning import transform_to_pose
        native = getattr(self.env, 'capture_rgbd', None)
        if callable(native):
            cameras = {}
            for frame in native(refresh=False):
                rigid = quaternion_camera_to_base(frame['pose'])
                cameras[frame['name']] = dict(intrinsics=frame['intrinsics'],
                    camera_to_base=dict(rotation=np.asarray(rigid.rotation).tolist(), translation=list(rigid.translation)))
            from src.backend.robot_base import robot_pose_matrix
            actual = robot_pose_matrix(self.ee_connector.get_ee_pose())
            return cameras, transform_to_pose(actual)
        sim = self.env.handle.env.sim
        base = np.asarray(self.env.base_link_wxyz_xyz, dtype=float)
        if base.shape != (7,) or not np.isfinite(base).all():
            raise RecorderError("invalid robot base calibration")
        base_rigid = quaternion_camera_to_base({"position": dict(zip("xyz", base[4:])),
            "rotation": dict(zip("wxyz", base[:4]))})
        base_matrix = np.eye(4)
        base_matrix[:3, :3] = base_rigid.rotation
        base_matrix[:3, 3] = base_rigid.translation
        base_inverse = np.linalg.inv(base_matrix)
        width, height = int(self.env._render_width), int(self.env._render_height)
        cameras = {}
        opengl_to_optical = np.diag([1., -1., -1., 1.])  # OpenGL camera axes to optical axes: Ry(pi) @ Rz(pi)
        for name in ("agentview", "robot0_eye_in_hand"):
            try:
                camera_id = sim.model.camera_name2id(name)
                world_camera = np.eye(4)
                world_camera[:3, :3] = np.asarray(sim.data.get_camera_xmat(name)).reshape(3, 3)
                world_camera[:3, 3] = sim.data.get_camera_xpos(name)
            except Exception as exc:
                raise RecorderError(f"missing synchronized camera: {name}") from exc
            camera_to_base = base_inverse @ world_camera @ opengl_to_optical
            fovy = float(sim.model.cam_fovy[camera_id])
            focal = .5 * height / np.tan(fovy * np.pi / 360.)
            intrinsic = np.array([[focal, 0, .5*width], [0, focal, .5*height], [0, 0, 1.]])
            cameras[name] = {"intrinsics": intrinsic,
                "camera_to_base": {"rotation": camera_to_base[:3, :3].tolist(),
                                   "translation": camera_to_base[:3, 3].tolist()}}
        # Read this ROBOT link freshly, not cached gripper_link_wxyz_xyz.
        gripper = np.eye(4)
        gripper[:3, :3] = np.asarray(sim.data.xmat[self.env.gripper_link_idx]).reshape(3, 3)
        gripper[:3, 3] = sim.data.xpos[self.env.gripper_link_idx]
        gripper_to_public = np.array([[0., -1., 0., 0.], [1., 0., 0., 0.],
                                      [0., 0., 1., -.107], [0., 0., 0., 1.]])
        public = base_inverse @ gripper @ gripper_to_public
        # Match the Connector public getter's configured translation, if
        # present; the calibrated correction maps that public frame to planner.
        arm_bases = getattr(self.connector, "_arm_bases", None)
        if arm_bases is not None:
            public[:3, 3] += np.asarray(arm_bases[0], dtype=float)
        actual = public @ self.public_to_planner
        return cameras, transform_to_pose(actual)

    def capture(self, *, force=False):
        if self.error:
            raise RecorderError(self.error)
        if self.closed:
            raise RecorderError("capture after recorder close")
        before = self._clock()
        step = before["simulator_step"]
        if step == self.last_step or (not force and step % self.every_n_steps):
            return
        try:
            cameras, actual_pose = self._robot_camera_snapshot()
            mapping = {"front": "agentview", "wrist": "robot0_eye_in_hand"}
            native_render = getattr(self.env, "render_rgb", None)
            sim = None if callable(native_render) else self.env.handle.env.sim
            width, height = int(self.env._render_width), int(self.env._render_height)
            views, pixels = {}, {}
            for view_id, camera_name in mapping.items():
                if camera_name not in cameras:
                    raise RecorderError(f"missing synchronized camera: {camera_name}")
                camera = cameras[camera_name]
                intrinsic = np.asarray(camera["intrinsics"], dtype=float)
                if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
                    raise RecorderError("invalid camera calibration")
                transform = camera["camera_to_base"]
                # Fresh rendering bypasses potentially stale cached RGB during motion.
                frame = (np.asarray(native_render(camera_name)) if callable(native_render) else
                         np.asarray(sim.render(camera_name=camera_name, width=width,
                                               height=height, depth=False))[::-1])
                if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
                    raise RecorderError("invalid recorder RGB frame")
                pixels[view_id] = np.ascontiguousarray(frame)
                views[view_id] = {"camera_name": camera_name, "video_frame_index": self.frame_count,
                    "width": width, "height": height, "intrinsics": intrinsic.tolist(),
                    "camera_to_base": transform}
            xyz = [float(actual_pose["position"][key]) for key in "xyz"]
            if not np.isfinite(xyz).all():
                raise RecorderError("nonfinite calibrated EE trace")
            after = self._clock()
            if (after["simulator_step"] != step or
                    after["simulation_time_s"] != before["simulation_time_s"]):
                raise RecorderError("sensor recording unexpectedly advanced simulation")
            for view_id, frame in pixels.items():
                if view_id not in self.writers:
                    factory = self.writer_factory
                    if factory is None:
                        import imageio.v2 as imageio
                        factory = imageio.get_writer
                    self.writers[view_id] = factory(str(self.output_dir / (view_id + ".mp4")),
                        fps=self.fps, codec="libx264", macro_block_size=1, quality=8)
                self.writers[view_id].append_data(frame)
                self.video_counts[view_id] += 1
            self._append("frames.jsonl", {"schema": "agent-interface.frame.v1",
                "frame_index": self.frame_count, **before, "geometry_frame": "connector_base",
                "operation": self.operation, "segment_id": self.segment_id, "plan_ref": self.plan_ref,
                "actual_ee_xyz": xyz, "actual_ee_pose": actual_pose, "views": views})
            self.frame_count += 1
            self.last_step = step
            if self.preview is not None:
                self.preview.publish(pixels)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self._append("errors.jsonl", {**before, "error": self.error, "frame_index": self.frame_count})
            self._write_manifest("failed")
            raise RecorderError(self.error) from exc

    @contextmanager
    def active(self, operation, **payload):
        previous = self.operation
        self.operation = operation
        self.event("operation_start", payload)
        try:
            yield
        finally:
            self.event("operation_end", payload)
            self.operation = previous

    def register_plan(self, plan, *, kind, point_ref=None, candidate_ref=None):
        ref = "plan_" + uuid4().hex
        segments = []
        if len(plan.segments) != len(plan.targets):
            raise RecorderError("plan target/segment count mismatch")
        target_labels = list(getattr(plan, "target_labels", ()))
        segment_labels = list(getattr(plan, "segment_labels", ()))
        for labels in (target_labels, segment_labels):
            if labels and (len(labels) != len(plan.targets) or not all(isinstance(x, str) and x for x in labels)):
                raise RecorderError("plan semantic labels do not align with targets")
        for index, (segment, target) in enumerate(zip(plan.segments, plan.targets)):
            segment_id = ref + ":" + str(index)
            self._segments[id(segment)] = (segment_id, ref)
            segments.append({"segment_id": segment_id, "index": index, "target": target,
                "joint_waypoint_count": len(segment.get("waypoints", [])),
                "label": segment_labels[index] if segment_labels else None})
        payload = {"schema": "agent-interface.plan.v1", "plan_ref": ref, "kind": kind,
            "point_ref": point_ref, "candidate_ref": candidate_ref, "geometry_frame": "connector_base",
            "targets": list(plan.targets), "target_labels": target_labels, "segment_labels": segment_labels,
            "transit_policy": getattr(plan, "transit_policy", "legacy"),
            "high_transit_z_m": getattr(plan, "high_transit_z_m", None),
            "segments": segments, "frame_index": self.frame_count - 1,
            **self._clock()}
        self._append("plans.jsonl", payload)
        self.event("plan_created", payload)
        return ref

    def close(self):
        if self.closed:
            return
        close_errors = []
        for writer in self.writers.values():
            try:
                writer.close()
            except Exception as exc:
                close_errors.append(repr(exc))
        for owner, name, existed, previous in reversed(self._restorations):
            if existed:
                setattr(owner, name, previous)
            else:
                delattr(owner, name)
        self._restorations.clear()
        self.closed = True
        if close_errors:
            self.error = "video finalization failed: " + "; ".join(close_errors)
        self._write_manifest("failed" if self.error else "completed")
        if self.error:
            raise RecorderError(self.error)
