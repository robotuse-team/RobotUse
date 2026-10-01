"""LiberoWorldAdapter — verify.World snapshots from sim ground truth.

Builds :class:`gap.runtime.verify.World` snapshots for checkpoint
enforcement from the LIBERO/robosuite MuJoCo sim underneath the env:

- **object poses** from the sim's body state (``data.body_xpos`` /
  ``data.body_xquat``, wxyz) — the same ground truth the SimBridge GetState
  path surfaced; when no MuJoCo sim is reachable, the env obs dict's
  ``cube_poses`` entry is used as a fallback;
- **AABBs** precomputed once per reset from the MuJoCo *model* geoms
  (per-body local bounds from geom type/size/pos/quat, preferring the
  model's own ``geom_aabb`` when present), cached by body and re-oriented
  into world frame at snapshot time;
- **contacts** from ``sim.data.contact`` pairs, resolved to object names via
  body-subtree membership and filtered to named bodies + robot/gripper
  links, then canonicalized with :func:`verify.contacts_from_pairs`;
- **robot view** (joints, EE pose, gripper open fraction) into
  :class:`verify.Robot`.

Wired up as :meth:`gap.connector.sim.SimConnector.world_snapshot`.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from gap.runtime.verify import Robot, World, contacts_from_pairs
from gap.runtime.verify.world import Body

logger = logging.getLogger(__name__)

# Panda link prefixes from verify.World defaults, extended with robosuite's
# "gripper0_*" namespace so finger contacts register as grasps.
_ROBOT_LINK_PREFIXES: tuple[str, ...] = (
    "robot", "panda_", "Robotiq", "finger_", "gripper",
)

# MuJoCo geom types (mjtGeom)
_GEOM_PLANE = 0
_GEOM_HFIELD = 1
_GEOM_SPHERE = 2
_GEOM_CAPSULE = 3
_GEOM_ELLIPSOID = 4
_GEOM_CYLINDER = 5
_GEOM_BOX = 6
_GEOM_MESH = 7


def find_mujoco_sim(env: Any) -> Any | None:
    """Walk the env wrapper chain looking for a MuJoCo sim handle.

    Accepts the FrankaLiberoEnv shape (``env.handle.env.sim``) and any
    nesting of ``.env`` wrappers; returns the first object exposing both
    ``.model`` and ``.data``.
    """
    seen: set[int] = set()
    frontier = [env]
    for _ in range(8):
        nxt = []
        for obj in frontier:
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            sim = getattr(obj, "sim", None)
            if sim is not None and hasattr(sim, "model") and hasattr(sim, "data"):
                return sim
            for attr in ("handle", "env"):
                child = getattr(obj, attr, None)
                if child is not None:
                    nxt.append(child)
        if not nxt:
            break
        frontier = nxt
    return None


def _find_object_map(env: Any) -> dict[str, int] | None:
    """Find the env's ``{object_name: root_body_id}`` registry, if any."""
    seen: set[int] = set()
    frontier = [env]
    for _ in range(8):
        nxt = []
        for obj in frontier:
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            for attr in ("obj_body_id", "_obj_body_id"):
                mapping = getattr(obj, attr, None)
                if isinstance(mapping, dict) and mapping:
                    return {str(k): int(v) for k, v in mapping.items()}
            for attr in ("handle", "env"):
                child = getattr(obj, attr, None)
                if child is not None:
                    nxt.append(child)
        if not nxt:
            break
        frontier = nxt
    return None


def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in q[:4])
    n = (w * w + x * x + y * y + z * z) ** 0.5
    if n == 0.0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def _geom_local_box(model: Any, gid: int) -> tuple[np.ndarray, np.ndarray] | None:
    """(center, half-extents) of geom *gid* in the geom's own frame."""
    gtype = int(np.asarray(model.geom_type)[gid])
    if gtype in (_GEOM_PLANE, _GEOM_HFIELD):
        return None
    size = np.asarray(model.geom_size)[gid].astype(np.float64)
    # Prefer the model's own per-geom AABB when available (exact for meshes).
    aabb = getattr(model, "geom_aabb", None)
    if aabb is not None:
        row = np.asarray(aabb)[gid].astype(np.float64)
        if np.any(row[3:] > 0):
            return row[:3], row[3:]
    if gtype == _GEOM_SPHERE:
        half = np.array([size[0]] * 3)
    elif gtype == _GEOM_CAPSULE:
        half = np.array([size[0], size[0], size[1] + size[0]])
    elif gtype == _GEOM_CYLINDER:
        half = np.array([size[0], size[0], size[1]])
    elif gtype in (_GEOM_BOX, _GEOM_ELLIPSOID):
        half = size[:3].copy()
    else:  # mesh without geom_aabb: bounding-sphere radius
        rbound = float(np.asarray(model.geom_rbound)[gid])
        half = np.array([rbound] * 3)
    return np.zeros(3), half


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z)."""
    t = float(np.trace(R))
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        return np.array([
            0.25 * s,
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
        ])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(max(R[i, i] - R[j, j] - R[k, k] + 1.0, 0.0)) * 2.0
    q = np.empty(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def _box_corners(center: np.ndarray, half: np.ndarray) -> np.ndarray:
    signs = np.array([
        [sx, sy, sz]
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ])
    return center[None, :] + signs * half[None, :]


class LiberoWorldAdapter:
    """Build verify.World snapshots from a LIBERO-style env's ground truth."""

    def __init__(
        self,
        env: Any,
        *,
        robot_link_prefixes: tuple[str, ...] = _ROBOT_LINK_PREFIXES,
        arm_dof: int = 7,
    ) -> None:
        self.env = env
        self.robot_link_prefixes = tuple(robot_link_prefixes)
        self.arm_dof = int(arm_dof)
        self._sim: Any | None = None
        self._objects: dict[str, int] = {}            # object name -> root body id
        self._body_owner: dict[int, str] = {}          # any body id -> contact name
        self._local_aabbs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._tabletop_name = "table"
        self._prepared = False
        self.refresh()

    # ------------------------------------------------------------------
    # Reset-time precomputation (cached by body)
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """(Re)build body registries + local AABBs. Call after env reset —
        a robosuite hard reset rebuilds the MjModel."""
        self._sim = find_mujoco_sim(self.env)
        self._objects = {}
        self._body_owner = {}
        self._local_aabbs = {}
        self._base_bid: int | None = None
        self._prepared = False
        if self._sim is None:
            return
        model = self._sim.model

        # Robot base body: snapshots are expressed in this body's frame so
        # privileged truth lands in the SAME frame as perception clouds and
        # motion targets (both robot-base-framed). Without this, generated
        # checkpoints comparing subgraph outputs against w.body(...) fail
        # by the base offset (~0.6 m x in LIBERO) on every run.
        for base_name in ("robot0_base", "robot0_link0"):
            bid = self._body_id(model, base_name)
            if bid is not None:
                self._base_bid = bid
                break

        objects = _find_object_map(self.env) or self._free_joint_bodies(model)
        # Tabletop body, when present.
        table_id = self._body_id(model, "table")
        if table_id is None:
            table_id = self._body_id(model, "table_top")
        nbody = int(getattr(model, "nbody", 0))
        parentid = np.asarray(model.body_parentid)

        roots = dict(objects)
        if table_id is not None:
            roots[self._tabletop_name] = table_id

        # Subtree membership: each model body maps to the object/table whose
        # root it descends from; robot links keep their raw names.
        root_of = {bid: name for name, bid in roots.items()}
        for b in range(nbody):
            cur = b
            for _ in range(nbody):
                if cur in root_of:
                    self._body_owner[b] = root_of[cur]
                    break
                parent = int(parentid[cur])
                if parent == cur or parent < 0:
                    break
                cur = parent
            if b not in self._body_owner:
                name = self._body_name(model, b)
                if name and name.startswith(self.robot_link_prefixes):
                    self._body_owner[b] = name

        self._objects = objects

        # Per-body local AABBs from model geoms, accumulated over the subtree.
        for name, root in roots.items():
            lo = np.full(3, np.inf)
            hi = np.full(3, -np.inf)
            members = [b for b in range(nbody) if self._body_owner.get(b) == name]
            for b in members:
                offset_pos, offset_rot = self._subtree_transform(model, root, b)
                gids = np.where(np.asarray(model.geom_bodyid) == b)[0]
                for gid in gids:
                    box = _geom_local_box(model, int(gid))
                    if box is None:
                        continue
                    center, half = box
                    gpos = np.asarray(model.geom_pos)[gid].astype(np.float64)
                    grot = _quat_wxyz_to_rotmat(np.asarray(model.geom_quat)[gid])
                    corners = _box_corners(center, half)        # geom frame
                    corners = (grot @ corners.T).T + gpos        # body frame
                    corners = (offset_rot @ corners.T).T + offset_pos  # root frame
                    lo = np.minimum(lo, corners.min(axis=0))
                    hi = np.maximum(hi, corners.max(axis=0))
            if np.all(np.isfinite(lo)):
                self._local_aabbs[name] = (lo, hi)
            else:
                self._local_aabbs[name] = (
                    -0.05 * np.ones(3), 0.05 * np.ones(3),
                )
        self._prepared = True

    @staticmethod
    def _body_id(model: Any, name: str) -> int | None:
        try:
            return int(model.body_name2id(name))
        except Exception:
            return None

    @staticmethod
    def _body_name(model: Any, bid: int) -> str | None:
        try:
            return model.body_id2name(bid)
        except Exception:
            return None

    def _free_joint_bodies(self, model: Any) -> dict[str, int]:
        """Fallback object discovery: bodies hung on a free joint."""
        objects: dict[str, int] = {}
        try:
            jnt_type = np.asarray(model.jnt_type)
            jnt_bodyid = np.asarray(model.jnt_bodyid)
        except Exception:
            return objects
        for j in range(len(jnt_type)):
            if int(jnt_type[j]) != 0:  # mjJNT_FREE
                continue
            bid = int(jnt_bodyid[j])
            name = self._body_name(model, bid)
            if not name or name.startswith(self.robot_link_prefixes):
                continue
            key = name[:-5] if name.endswith("_main") else name
            objects[key] = bid
        return objects

    def _subtree_transform(
        self, model: Any, root: int, body: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Static transform of *body* in *root*'s frame (model rest pose).

        Interior joints inside an object subtree are rare in LIBERO scenes;
        accumulated ``body_pos``/``body_quat`` chains are sufficient.
        """
        pos = np.zeros(3)
        rot = np.eye(3)
        cur = body
        parentid = np.asarray(model.body_parentid)
        body_pos = np.asarray(model.body_pos)
        body_quat = np.asarray(model.body_quat)
        for _ in range(int(getattr(model, "nbody", 0))):
            if cur == root:
                return pos, rot
            p = body_pos[cur].astype(np.float64)
            r = _quat_wxyz_to_rotmat(body_quat[cur])
            pos = r @ pos + p
            rot = r @ rot
            parent = int(parentid[cur])
            if parent == cur or parent < 0:
                break
            cur = parent
        return pos, rot

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self, env_id: int = 0) -> World:
        if self._sim is None or not self._prepared:
            self.refresh()
        if self._sim is None:
            return self._snapshot_from_obs(env_id)

        sim = self._sim
        model, data = sim.model, sim.data

        contacts = contacts_from_pairs(self._contact_pairs(model, data))

        # World -> robot-base transform (p_base = R_b^T @ (p_world - t_b)):
        # keeps privileged truth in the frame all workflow outputs use.
        base_t = np.zeros(3)
        base_Rt = np.eye(3)
        if self._base_bid is not None:
            base_t = np.asarray(data.body_xpos)[self._base_bid].astype(np.float64)
            base_Rt = _quat_wxyz_to_rotmat(
                np.asarray(data.body_xquat)[self._base_bid].astype(np.float64)
            ).T

        bodies: dict[str, Body] = {}
        names = dict(self._objects)
        if self._tabletop_name in self._local_aabbs:
            table_id = self._body_id(model, "table") or self._body_id(model, "table_top")
            if table_id is not None:
                names[self._tabletop_name] = table_id
        for name, bid in names.items():
            pos = np.asarray(data.body_xpos)[bid].astype(np.float64).copy()
            quat = np.asarray(data.body_xquat)[bid].astype(np.float64).copy()  # wxyz
            lo, hi = self._local_aabbs.get(
                name, (-0.05 * np.ones(3), 0.05 * np.ones(3))
            )
            R = _quat_wxyz_to_rotmat(quat)
            pos = base_Rt @ (pos - base_t)
            R = base_Rt @ R
            quat = _rotmat_to_quat_wxyz(R)
            corners = (R @ _box_corners((lo + hi) / 2.0, (hi - lo) / 2.0).T).T + pos
            lin, ang = self._body_velocity(data, bid)
            bodies[name] = Body(
                name=name,
                position=pos,
                quaternion_wxyz=quat,
                aabb_lower=corners.min(axis=0),
                aabb_upper=corners.max(axis=0),
                linear_velocity=base_Rt @ lin,
                angular_velocity=base_Rt @ ang,
                contacts=contacts.get(name, frozenset()),
            )

        robot_view = self._robot_view(model, data, base_t=base_t, base_Rt=base_Rt)
        time_s = 0.0
        if hasattr(self.env, "get_current_time_s"):
            try:
                time_s = float(self.env.get_current_time_s())
            except Exception:
                time_s = 0.0

        return World(
            env_id=int(env_id),
            bodies=bodies,
            robot_view=robot_view,
            time_s=time_s,
            robot_link_prefixes=self.robot_link_prefixes,
            tabletop_body_name=self._tabletop_name,
        )

    # Alias so SimConnector.world_snapshot can pass straight through.
    __call__ = snapshot

    def _contact_pairs(self, model: Any, data: Any) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        try:
            ncon = int(data.ncon)
        except Exception:
            return pairs
        geom_bodyid = np.asarray(model.geom_bodyid)
        for i in range(ncon):
            con = data.contact[i]
            b1 = int(geom_bodyid[int(con.geom1)])
            b2 = int(geom_bodyid[int(con.geom2)])
            n1 = self._body_owner.get(b1)
            n2 = self._body_owner.get(b2)
            # Filtered to named bodies + robot/gripper links only.
            if n1 is None or n2 is None or n1 == n2:
                continue
            pairs.append((n1, n2))
        return pairs

    def _body_velocity(self, data: Any, bid: int) -> tuple[np.ndarray, np.ndarray]:
        try:
            cvel = np.asarray(data.cvel)[bid].astype(np.float64)
            return cvel[3:6].copy(), cvel[0:3].copy()
        except Exception:
            return np.zeros(3), np.zeros(3)

    def _robot_view(
        self,
        model: Any,
        data: Any,
        *,
        base_t: np.ndarray | None = None,
        base_Rt: np.ndarray | None = None,
    ) -> Robot | None:
        joint_names: list[str] = []
        joint_pos: list[float] = []
        qpos = np.asarray(data.qpos)
        for i in range(1, self.arm_dof + 1):
            jn = f"robot0_joint{i}"
            try:
                addr = model.get_joint_qpos_addr(jn)
            except Exception:
                addr = None
            if addr is None:
                continue
            if isinstance(addr, tuple):
                addr = addr[0]
            joint_names.append(jn)
            joint_pos.append(float(qpos[int(addr)]))
        if not joint_pos:
            return None

        ee_pos = np.array([0.5, 0.0, 0.5])
        ee_quat = np.array([1.0, 0.0, 0.0, 0.0])
        eef_id = self._body_id(model, "gripper0_eef")
        if eef_id is not None:
            ee_pos = np.asarray(data.body_xpos)[eef_id].astype(np.float64).copy()
            ee_quat = np.asarray(data.body_xquat)[eef_id].astype(np.float64).copy()
            if base_t is not None and base_Rt is not None:
                ee_pos = base_Rt @ (ee_pos - base_t)
                ee_quat = _rotmat_to_quat_wxyz(
                    base_Rt @ _quat_wxyz_to_rotmat(ee_quat)
                )

        fraction = float(getattr(self.env, "_gripper_fraction", 1.0))
        try:
            addr = model.get_joint_qpos_addr("gripper0_finger_joint1")
            if isinstance(addr, tuple):
                addr = addr[0]
            fraction = float(np.clip(qpos[int(addr)] / 0.04, 0.0, 1.0))
        except Exception:
            pass

        return Robot(
            body_name="robot",
            joint_pos=np.asarray(joint_pos, dtype=np.float64),
            joint_names=tuple(joint_names),
            ee_position=ee_pos,
            ee_quaternion_wxyz=ee_quat,
            gripper_open_fraction=fraction,
        )

    def _snapshot_from_obs(self, env_id: int) -> World:
        """No-MuJoCo fallback: object poses from the obs dict's cube_poses
        (the SimBridge GetState source) with nominal 10 cm AABBs."""
        obs = self.env.get_observation()
        bodies: dict[str, Body] = {}
        for name, pose_data in obs.get("cube_poses", {}).items():
            arr = np.asarray(pose_data, dtype=np.float64)
            pos, quat = arr[:3], arr[3:7]
            bodies[str(name)] = Body(
                name=str(name),
                position=pos.copy(),
                quaternion_wxyz=quat.copy(),
                aabb_lower=pos - 0.05,
                aabb_upper=pos + 0.05,
                linear_velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
                contacts=frozenset(),
            )
        return World(
            env_id=int(env_id),
            bodies=bodies,
            robot_view=None,
            robot_link_prefixes=self.robot_link_prefixes,
        )


__all__ = ["LiberoWorldAdapter", "find_mujoco_sim"]
