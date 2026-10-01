"""3D scene replay via viser — loads scene_log data and builds an interactive 3D viewer.

Supports both:
- Discrete snapshots (real robot, few observations)
- Continuous trajectories (simulation, per-step joint logging)

Heavy dependencies (viser, trimesh, yourdfpy, mujoco) are imported lazily
inside the functions that need them so that importing :mod:`gap.viz` stays
cheap.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_UR_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

_PANDA_JOINT_NAMES = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7",
]

# robot_descriptions module names guessed from arm DOF when the trial meta
# carries no usable robot_urdf.
_DOF_TO_DESCRIPTION = {6: "ur5e_description", 7: "panda_description"}


def _mat_to_pos_wxyz(T: np.ndarray):
    from scipy.spatial.transform import Rotation

    pos = T[:3, 3].astype(np.float32)
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    return pos, np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)


def _description_urdf(module_name: str) -> Path | None:
    try:
        import importlib
        mod = importlib.import_module(f"robot_descriptions.{module_name}")
        return Path(mod.URDF_PATH)
    except Exception:
        return None


def _resolve_urdf(trial_dir: Path, meta: dict) -> Path | None:
    urdf_path = meta.get("robot_urdf")

    # Absolute path that exists on disk
    if urdf_path and Path(urdf_path).exists():
        return Path(urdf_path)

    # robot_descriptions package name (e.g., "panda_description")
    if urdf_path and not str(urdf_path).endswith(".urdf"):
        found = _description_urdf(str(urdf_path))
        if found is not None:
            return found

    # Fallback: guess from arm DOF
    dof = meta.get("arm_dof", 6)
    desc = _DOF_TO_DESCRIPTION.get(int(dof))
    if desc is not None:
        return _description_urdf(desc)

    return None


class SceneReplay:
    """Loads scene_log data and builds a viser scene with timeline controls."""

    def __init__(self, trial_dir: Path):
        self.trial_dir = trial_dir
        self.scene_log_dir = trial_dir / "scene_log"

        self.meta: dict = {}
        self.joint_timestamps: np.ndarray | None = None
        self.joint_positions: np.ndarray | None = None
        self.joint_grippers: np.ndarray | None = None
        self.camera_data: dict[str, dict] = {}
        self._pointclouds: dict[str, list[tuple[float, np.ndarray, np.ndarray | None]]] = {}
        self.obbs: list[dict] = []
        self.poses: list[dict] = []
        self.events: list[dict] = []

        self._load()

    def _load(self) -> None:
        if not self.scene_log_dir.exists():
            logger.warning("No scene_log directory in %s", self.trial_dir)
            return

        meta_path = self.scene_log_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.meta = json.load(f)

        joints_path = self.scene_log_dir / "joints.npz"
        if joints_path.exists():
            data = np.load(str(joints_path))
            self.joint_timestamps = data["timestamps"]
            self.joint_positions = data["positions"]
            self.joint_grippers = data["grippers"]

        cameras_dir = self.scene_log_dir / "cameras"
        if cameras_dir.exists():
            for cam_dir in cameras_dir.iterdir():
                if not cam_dir.is_dir():
                    continue
                cam_name = cam_dir.name
                cam: dict = {"dir": cam_dir}

                ts_path = cam_dir / "timestamps.npy"
                if ts_path.exists():
                    cam["timestamps"] = np.load(str(ts_path))

                intr_path = cam_dir / "intrinsics.npy"
                if intr_path.exists():
                    cam["intrinsics"] = np.load(str(intr_path))

                poses_path = cam_dir / "poses.npy"
                if poses_path.exists():
                    cam["poses"] = np.load(str(poses_path))

                images_dir = cam_dir / "images"
                if images_dir.exists():
                    cam["images"] = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))

                self.camera_data[cam_name] = cam

        obbs_path = self.scene_log_dir / "obbs.json"
        if obbs_path.exists():
            with open(obbs_path) as f:
                self.obbs = json.load(f)

        poses_path = self.scene_log_dir / "poses.json"
        if poses_path.exists():
            with open(poses_path) as f:
                self.poses = json.load(f)

        events_path = self.scene_log_dir / "events.json"
        if events_path.exists():
            with open(events_path) as f:
                self.events = json.load(f)

        # Also load OBBs and point clouds from per-node trace data
        self._load_from_node_trace()

    def _load_from_node_trace(self) -> None:
        """Load OBBs and point clouds from per-node trace data."""
        node_data_dir = self.trial_dir / "node_data"
        if not node_data_dir.exists():
            return

        for node_dir in sorted(node_data_dir.iterdir()):
            if not node_dir.is_dir():
                continue
            node_name = node_dir.name

            # Load OBBs from output.json
            output_path = node_dir / "output.json"
            if output_path.exists():
                try:
                    with open(output_path) as f:
                        output = json.load(f)
                    obb = output.get("obb") if isinstance(output, dict) else None
                    if obb and "center" in obb:
                        self.obbs.append({
                            "timestamp": 0,
                            "name": node_name.split(".")[-1],
                            "center": _vec3(obb.get("center")),
                            "extent": _vec3(obb.get("extent")),
                            "orientation": _quat_wxyz(obb.get("orientation")),
                        })
                except Exception:
                    pass

            # Load only the final output point cloud (not duplicates from inputs/sub-calls)
            assets_dir = node_dir / "assets"
            if assets_dir.exists():
                output_clouds = sorted(assets_dir.glob("output_*cloud*.npz"))
                for npz_file in output_clouds[:1]:
                    try:
                        data = np.load(str(npz_file))
                        positions = data.get("positions")
                        if positions is not None and len(positions) > 0:
                            colors = data.get("colors")
                            short = node_name.split(".")[-1]
                            self._pointclouds[short] = [
                                (0, positions.astype(np.float32),
                                 colors.astype(np.float32) if colors is not None else None)
                            ]
                    except Exception:
                        pass

    @property
    def num_steps(self) -> int:
        if self.joint_timestamps is not None and len(self.joint_timestamps) > 0:
            return len(self.joint_timestamps)
        for cam in self.camera_data.values():
            if "timestamps" in cam:
                return len(cam["timestamps"])
        return 1

    @property
    def has_data(self) -> bool:
        return bool(self.meta) or self.num_steps > 0


def _vec3(v: Any) -> list[float]:
    """Normalize a gap.types Vec3 — list/tuple/ndarray or {x,y,z} dict."""
    if isinstance(v, dict):
        return [float(v.get("x", 0)), float(v.get("y", 0)), float(v.get("z", 0))]
    if v is None:
        return [0.0, 0.0, 0.0]
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    return [float(x) for x in arr[:3]]


def _quat_wxyz(q: Any) -> list[float]:
    """Normalize a quaternion — list (wxyz) or {w,x,y,z} dict."""
    if isinstance(q, dict):
        return [
            float(q.get("w", 1)), float(q.get("x", 0)),
            float(q.get("y", 0)), float(q.get("z", 0)),
        ]
    if q is None:
        return [1.0, 0.0, 0.0, 0.0]
    arr = np.asarray(q, dtype=np.float64).reshape(-1)
    if len(arr) >= 4:
        return [float(x) for x in arr[:4]]
    return [1.0, 0.0, 0.0, 0.0]


def _pose_to_mat(pos: np.ndarray, wxyz: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from (pos_xyz, quat_wxyz)."""
    from scipy.spatial.transform import Rotation

    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    w, x, y, z = (float(q) for q in wxyz)
    R = Rotation.from_quat([x, y, z, w]).as_matrix()  # scipy: xyzw
    T[:3, :3] = R
    return T


def _mat3_to_wxyz(mat: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → (w, x, y, z) viser quaternion."""
    from scipy.spatial.transform import Rotation

    q = Rotation.from_matrix(mat).as_quat()
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)


def _add_mjcf_static_scene(
    server: Any,
    mjcf_path: Path,
    skip_body_prefixes: tuple[str, ...] = (),
    prefix: str = "/scene",
) -> int:
    """Render the non-articulated geoms of an MJCF into the viser scene.

    The robots are animated by ``_add_urdf_meshes`` already, so we exclude
    any geoms whose body name starts with one of ``skip_body_prefixes``.
    Everything else — walls, floor, fixtures — is added as a static mesh /
    primitive at its keyframe pose so the 3D Scene tab looks like the
    actual scene instead of arms in a void.

    Returns the number of geoms added (useful for logging).
    """
    import trimesh

    try:
        import mujoco  # type: ignore[import-not-found]
    except ImportError:
        logger.debug("mujoco not available; skipping static scene load")
        return 0
    if not mjcf_path.exists():
        logger.warning("MJCF not found: %s", mjcf_path)
        return 0

    try:
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    except Exception as exc:  # pragma: no cover - depends on local MJCF
        logger.warning("Failed to load MJCF %s: %s", mjcf_path, exc)
        return 0
    data = mujoco.MjData(model)
    # Use the "home" keyframe (if present) so object bodies sit at their
    # declared rest pose, not collapsed at origin.
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_kinematics(model, data)

    added = 0
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        body_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        )
        if any(body_name.startswith(p) for p in skip_body_prefixes):
            continue
        # Convention: groups 0/1/2 are visual, 3+ collision-only / disabled.
        group = int(model.geom_group[gid])
        if group >= 3:
            continue

        gtype = int(model.geom_type[gid])
        pos = data.geom_xpos[gid].astype(np.float32)
        mat = data.geom_xmat[gid].reshape(3, 3)
        wxyz = _mat3_to_wxyz(mat)
        size = model.geom_size[gid].astype(np.float64)
        rgba = model.geom_rgba[gid]
        color = (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
        opacity = float(rgba[3])

        geom_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            or f"geom_{gid}"
        )
        scene_name = f"{prefix}/{body_name or 'world'}/{geom_name}"

        try:
            if gtype == mujoco.mjtGeom.mjGEOM_BOX:
                dims = (2 * size[0], 2 * size[1], 2 * size[2])
                server.scene.add_box(
                    scene_name, dimensions=dims, color=color,
                    position=pos, wxyz=wxyz, opacity=opacity,
                )
                added += 1
            elif gtype == mujoco.mjtGeom.mjGEOM_PLANE:
                # MuJoCo planes are infinite if size[0]==0; clamp to a
                # large finite extent so viser has something to draw.
                hx = float(size[0]) if size[0] > 0 else 5.0
                hy = float(size[1]) if size[1] > 0 else 5.0
                dims = (max(2 * hx, 0.001), max(2 * hy, 0.001), 0.002)
                server.scene.add_box(
                    scene_name, dimensions=dims, color=color,
                    position=pos, wxyz=wxyz, opacity=opacity,
                )
                added += 1
            elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                server.scene.add_icosphere(
                    scene_name, radius=float(size[0]), color=color,
                    position=pos, wxyz=wxyz, opacity=opacity,
                )
                added += 1
            elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
                meshid = int(model.geom_dataid[gid])
                if meshid < 0:
                    continue
                v0 = int(model.mesh_vertadr[meshid])
                vn = int(model.mesh_vertnum[meshid])
                f0 = int(model.mesh_faceadr[meshid])
                fn = int(model.mesh_facenum[meshid])
                verts = model.mesh_vert[v0:v0 + vn].astype(np.float32).copy()
                faces = model.mesh_face[f0:f0 + fn].astype(np.int32).copy()
                if verts.size == 0 or faces.size == 0:
                    continue
                mesh = trimesh.Trimesh(
                    vertices=verts, faces=faces, process=False,
                )
                face_color = [color[0], color[1], color[2],
                              int(opacity * 255)]
                mesh.visual = trimesh.visual.ColorVisuals(
                    mesh, face_colors=face_color,
                )
                server.scene.add_mesh_trimesh(
                    scene_name, mesh=mesh, position=pos, wxyz=wxyz,
                )
                added += 1
            # Other geom types (capsule, cylinder, ellipsoid, hfield)
            # are rare in these scenes; skipping rather than approximating
            # to avoid visual surprises.
        except Exception as exc:
            logger.debug("Failed to add geom %s (type=%d): %s",
                         scene_name, gtype, exc)
            continue

    return added


def _add_urdf_meshes(server: Any, urdf, joint_names: list[str],
                     joints: np.ndarray, prefix: str = "/robot",
                     base_T: np.ndarray | None = None) -> None:
    """Render URDF meshes at given joint config.

    If ``base_T`` is provided, all link frames are pre-multiplied by it so
    the robot is placed at an arbitrary world-frame base pose (needed for
    multi-arm scenes where each URDF instance has its own base).
    """
    import trimesh

    cfg = {}
    for i, name in enumerate(joint_names):
        if i < len(joints):
            cfg[name] = float(joints[i])
    urdf.update_cfg(cfg)

    for link_name in urdf.link_map:
        link = urdf.link_map[link_name]
        T = urdf.get_transform(link_name)
        if T is None:
            continue
        if base_T is not None:
            T = base_T @ T

        pos, wxyz = _mat_to_pos_wxyz(T)
        server.scene.add_frame(
            f"{prefix}/{link_name}", position=pos, wxyz=wxyz,
            axes_length=0.0, axes_radius=0.0,
        )

        for vi, visual in enumerate(link.visuals):
            if visual.geometry is None or visual.geometry.mesh is None:
                continue
            mesh_path = visual.geometry.mesh.filename
            if hasattr(urdf, '_filename_handler'):
                mesh_path = str(urdf._filename_handler(mesh_path))
            elif not Path(mesh_path).is_absolute():
                continue
            try:
                mesh = trimesh.load(mesh_path, force="mesh")
            except Exception:
                continue

            if visual.geometry.mesh.scale is not None:
                mesh.apply_scale(visual.geometry.mesh.scale)

            if visual.origin is not None:
                vp, vwxyz = _mat_to_pos_wxyz(visual.origin)
            else:
                vp = np.zeros(3, dtype=np.float32)
                vwxyz = np.array([1, 0, 0, 0], dtype=np.float32)

            server.scene.add_mesh_trimesh(
                f"{prefix}/{link_name}/visual_{vi}",
                mesh=mesh, position=vp, wxyz=vwxyz,
            )


_active_server: Any | None = None


_REPLAY_PORT = 8890


def stop_replay_server() -> None:
    """Stop the active viser replay server if one is running."""
    global _active_server
    if _active_server is not None:
        try:
            _active_server.stop()
        except Exception:
            pass
        _active_server = None
        import time
        time.sleep(1.0)


def start_replay_server(trial_dir: str | Path, port: int = _REPLAY_PORT) -> str:
    """Start a viser server with 3D scene replay. Returns the URL.

    Stops any previously running replay server first (always reuses the same port).
    """
    global _active_server
    import viser
    import yourdfpy

    stop_replay_server()

    # Force-free the port if a stale process holds it
    import subprocess
    subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)
    import time as _time
    _time.sleep(0.5)

    trial_dir = Path(trial_dir)
    replay = SceneReplay(trial_dir)

    if not replay.has_data:
        raise ValueError(f"No scene_log data in {trial_dir}")

    urdf_path = _resolve_urdf(trial_dir, replay.meta)
    if urdf_path is None:
        raise ValueError("Cannot find robot URDF. Set robot_urdf in TrialLogger.set_meta().")

    urdf = yourdfpy.URDF.load(str(urdf_path))

    dof = replay.meta.get("arm_dof", 6)
    joint_names = _UR_JOINT_NAMES if dof == 6 else _PANDA_JOINT_NAMES

    # Multi-arm handling: when scene_log records >1 arms, joints columns
    # are packed as [arm0_j1..jN, arm1_j1..jN, ...]. Each arm is rendered
    # at its world-frame base read from meta["arm_bases"]; missing bases
    # default to the origin (legacy single-arm behavior).
    num_arms = int(replay.meta.get("num_arms", 1) or 1)
    arm_bases_meta = replay.meta.get("arm_bases") or []
    arm_base_Ts: list[np.ndarray | None] = []
    for i in range(num_arms):
        if i < len(arm_bases_meta):
            pos, wxyz = arm_bases_meta[i]
            arm_base_Ts.append(_pose_to_mat(np.asarray(pos), np.asarray(wxyz)))
        else:
            arm_base_Ts.append(None)

    server = viser.ViserServer(host="0.0.0.0", port=port)

    max_step = max(replay.num_steps - 1, 0)

    # --- GUI ---
    step_slider = server.gui.add_slider(
        "Step", min=0, max=max(max_step, 1), step=1, initial_value=0,
    )

    # --- World frame + grid ---
    server.scene.add_frame("/world", position=(0, 0, 0), axes_length=0.15, axes_radius=0.003)
    server.scene.add_grid("/grid", width=1.5, height=1.5, cell_size=0.1, position=(0, 0, -0.22))

    # --- Static scene geometry from MJCF (if env exposed one) ---
    # When the trial logger meta carries a ``scene_mjcf`` path, dump the
    # non-articulated parts of the MJCF (walls, floor, fixtures, …) into
    # the viser scene once at startup. Without this, bimanual scenes look
    # hollow — just two arms at world-frame bases with nothing else to
    # anchor them.
    scene_mjcf = replay.meta.get("scene_mjcf")
    if scene_mjcf:
        try:
            skip_prefixes = tuple(
                replay.meta.get("scene_skip_body_prefixes") or ()
            )
            n_added = _add_mjcf_static_scene(
                server, Path(scene_mjcf),
                skip_body_prefixes=skip_prefixes,
            )
            logger.info("Loaded %d static scene geoms from %s",
                        n_added, scene_mjcf)
        except Exception:
            logger.debug("Failed to load static scene from %s",
                         scene_mjcf, exc_info=True)

    def update_scene(step_idx: int) -> None:
        from scipy.spatial.transform import Rotation

        # --- Robot(s) ---
        if replay.joint_positions is not None and step_idx < len(replay.joint_positions):
            row = replay.joint_positions[step_idx]
            # Trim if a previous run logged extra columns (e.g. concat'd
            # gripper); we only consume dof * num_arms.
            row = np.asarray(row).reshape(-1)
            for arm_idx in range(num_arms):
                start = arm_idx * dof
                end = start + dof
                if end > len(row):
                    break
                _add_urdf_meshes(
                    server, urdf, joint_names, row[start:end],
                    prefix=f"/robot_{arm_idx}",
                    base_T=arm_base_Ts[arm_idx],
                )

        # --- Camera ---
        for cam_name, cam in replay.camera_data.items():
            cam_timestamps = cam.get("timestamps")
            cam_poses = cam.get("poses")
            cam_images = cam.get("images", [])
            K = cam.get("intrinsics")

            if cam_timestamps is None or len(cam_timestamps) == 0:
                continue

            # Find nearest camera frame to current step timestamp
            if replay.joint_timestamps is not None and step_idx < len(replay.joint_timestamps):
                t = replay.joint_timestamps[step_idx]
                cam_idx = int(np.argmin(np.abs(cam_timestamps - t)))
            else:
                cam_idx = min(step_idx, len(cam_timestamps) - 1)

            if cam_poses is not None and cam_idx < len(cam_poses):
                pose = cam_poses[cam_idx]
                cam_pos = pose[:3].astype(np.float32)
                cam_wxyz = pose[3:7].astype(np.float32)

                server.scene.add_frame(
                    f"/camera/{cam_name}", position=cam_pos, wxyz=cam_wxyz,
                    axes_length=0.05, axes_radius=0.002,
                )

                frustum_kwargs: dict = {"scale": 0.12, "color": (100, 150, 255)}
                if K is not None:
                    fy = float(K[1, 1])
                    cx, cy = float(K[0, 2]), float(K[1, 2])
                    img_w, img_h = int(cx * 2), int(cy * 2)
                    frustum_kwargs["fov"] = float(2 * np.arctan(img_h / (2 * fy)))
                    frustum_kwargs["aspect"] = float(img_w / img_h)

                if cam_images and cam_idx < len(cam_images):
                    import imageio.v3 as iio
                    frustum_kwargs["image"] = iio.imread(str(cam_images[cam_idx]))

                server.scene.add_camera_frustum(
                    f"/camera/{cam_name}/frustum", **frustum_kwargs,
                )

        # --- OBBs ---
        if replay.obbs:
            for i, obb in enumerate(replay.obbs):
                center = np.array(obb["center"], dtype=np.float32)
                extent = np.array(obb["extent"], dtype=np.float64)
                orientation = np.array(obb.get("orientation", [1, 0, 0, 0]), dtype=np.float64)

                corners_local = np.array([
                    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
                ], dtype=np.float64) * extent

                R = Rotation.from_quat([orientation[1], orientation[2], orientation[3], orientation[0]]).as_matrix()
                corners = (R @ corners_local.T).T + center

                edges = [
                    (0, 1), (1, 2), (2, 3), (3, 0),
                    (4, 5), (5, 6), (6, 7), (7, 4),
                    (0, 4), (1, 5), (2, 6), (3, 7),
                ]
                seg_pts = np.array(
                    [[corners[a], corners[b]] for a, b in edges],
                    dtype=np.float32,
                )
                server.scene.add_line_segments(
                    f"/obb/{i}", points=seg_pts, colors=(255, 50, 50), line_width=3.0,
                )
                server.scene.add_label(
                    f"/obb/{i}/label",
                    text=obb.get("name", f"obb_{i}"),
                    position=center + np.array([0, 0, 0.03], dtype=np.float32),
                )

        # --- Point clouds ---
        for i, (_pc_name, frames) in enumerate(replay._pointclouds.items()):
            if frames:
                _, pts, clr = frames[0]
                if clr is not None:
                    colors = np.clip(clr * 255, 0, 255).astype(np.uint8) if clr.max() <= 1.0 else clr.astype(np.uint8)
                else:
                    colors = np.full((len(pts), 3), [0, 200, 0], dtype=np.uint8)
                server.scene.add_point_cloud(
                    f"/pointcloud/{i}",
                    points=pts,
                    colors=colors,
                    point_size=0.003,
                    point_shape="circle",
                )

    # Initial render
    update_scene(0)

    @step_slider.on_update
    def _on_step(event) -> None:
        update_scene(int(event.target.value))

    _active_server = server
    url = f"http://localhost:{port}"
    logger.info("Viser 3D replay started at %s (%d steps)", url, replay.num_steps)
    return url
