"""Export Panda visual meshes in the visual-pose annotation frame, without simulation.

Output uses indexed triangles with normals, in metres, ready for WebGL. Geometry
comes exclusively from XML group=1 visual meshes. This export does not establish
contact, collision freedom, reachability, or an executable robot pose.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

from src.runtime.paths import REPOSITORY_ROOT as ROOT
DEFAULT_ASSETS = ROOT / "src/tools/pose_editor/third_party/robosuite/robosuite/models/assets/grippers"
IDENTITY = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
# Native grip_site +X is across the jaws. Map its -X to display +Y.
DISPLAY_FROM_GRIP = ((0., 1., 0.), (-1., 0., 0.), (0., 0., 1.))


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def mv(r, v):
    return tuple(sum(r[i][j] * v[j] for j in range(3)) for i in range(3))


def mm(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def transpose(r):
    return tuple(zip(*r))


def unit(v):
    length = math.sqrt(sum(x*x for x in v))
    return tuple(x / length for x in v) if length else (0., 0., 1.)


def cross(a, b):
    return (a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0])


def quaternion(values):
    w, x, y, z = map(float, values.split())
    length = math.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = (q / length for q in (w, x, y, z))
    return ((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
            (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
            (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))


def local_transform(element):
    return quaternion(element.get("quat", "1 0 0 0")), tuple(map(float, element.get("pos", "0 0 0").split()))


def compose(a, b):
    return mm(a[0], b[0]), add(a[1], mv(a[0], b[1]))


def inverse(transform):
    rotation = transpose(transform[0])
    return rotation, mv(rotation, tuple(-x for x in transform[1]))


def binary_stl(path):
    blob = path.read_bytes()
    count = struct.unpack_from("<I", blob, 80)[0]
    if len(blob) != 84 + 50 * count:
        raise ValueError(f"Not an exact binary STL: {path}")
    return [tuple(tuple(f[3+j*3:6+j*3]) for j in range(3))
            for i in range(count)
            for f in [struct.unpack_from("<12fH", blob, 84 + 50*i)]]


def indexed_mesh(triangles, transform):
    """Preserve creases over 45 degrees, smooth tessellation within each surface."""
    face_normals, incident = [], {}
    for i, triangle in enumerate(triangles):
        a, b, c = triangle
        normal = unit(cross(tuple(b[j]-a[j] for j in range(3)), tuple(c[j]-a[j] for j in range(3))))
        face_normals.append(normal)
        for point in triangle:
            incident.setdefault(tuple(round(v, 8) for v in point), []).append(i)
    positions, normals, indices, unique = [], [], [], {}
    limit = math.cos(math.radians(45))
    for i, triangle in enumerate(triangles):
        for point in triangle:
            neighbors = (face_normals[j] for j in incident[tuple(round(v, 8) for v in point)])
            smooth = [n for n in neighbors if sum(a*b for a, b in zip(n, face_normals[i])) >= limit]
            normal = unit(tuple(sum(n[j] for n in smooth) for j in range(3)))
            output_point = tuple(round(v, 7) for v in add(transform[1], mv(transform[0], point)))
            output_normal = tuple(round(v, 6) for v in mv(transform[0], normal))
            key = output_point + output_normal
            if key not in unique:
                unique[key] = len(positions) // 3
                positions.extend(output_point)
                normals.extend(output_normal)
            indices.append(unique[key])
    return {"positions": positions, "normals": normals, "indices": indices,
            "vertex_count": len(positions)//3, "triangle_count": len(indices)//3}


def export(assets: Path, output: Path, jaw_separation: float):
    if not math.isfinite(jaw_separation) or not 0 <= jaw_separation <= .08:
        raise ValueError("Jaw separation must be between 0 and 0.08 metres")
    xml_path = assets / "panda_gripper.xml"
    xml = ET.parse(xml_path).getroot()
    mesh_paths = {mesh.attrib["name"]: assets / mesh.attrib["file"] for mesh in xml.findall("asset/mesh")}
    joint_values = {"finger_joint1": jaw_separation/2, "finger_joint2": -jaw_separation/2}
    world_geoms, sites = [], {}

    def visit(body, parent):
        frame = compose(parent, local_transform(body))
        for joint in body.findall("joint"):
            value = joint_values.get(joint.get("name"), 0.)
            axis = tuple(map(float, joint.get("axis", "0 0 1").split()))
            frame = compose(frame, (IDENTITY, tuple(value*x for x in axis)))
        for site in body.findall("site"):
            sites[site.attrib["name"]] = compose(frame, local_transform(site))
        for geom in body.findall("geom"):
            if geom.get("group") == "1" and geom.get("mesh"):
                world_geoms.append((geom, compose(frame, local_transform(geom))))
        for child in body.findall("body"):
            visit(child, frame)

    for body in xml.findall("worldbody/body"):
        visit(body, (IDENTITY, (0., 0., 0.)))
    display_from_world = compose((DISPLAY_FROM_GRIP, (0., 0., 0.)), inverse(sites["grip_site"]))
    parts, sources = [], {xml_path}
    for geom, world_transform in world_geoms:
        path = mesh_paths[geom.attrib["mesh"]]
        sources.add(path)
        transform = compose(display_from_world, world_transform)
        data = indexed_mesh(binary_stl(path), transform)
        parts.append({"name": geom.attrib["name"], "mesh_name": geom.attrib["mesh"],
                      "color": list(map(float, geom.get("rgba", "1 1 1 1").split()))[:3],
                      "source_to_display": {"rotation": transform[0], "translation": transform[1]}, **data})
    points = [part["positions"] for part in parts]
    bounds = {"min": [min(value for flat in points for value in flat[axis::3]) for axis in range(3)],
              "max": [max(value for flat in points for value in flat[axis::3]) for axis in range(3)]}
    payload = {
        "schema_version": 1, "units": "m", "parts": parts, "bounds": bounds,
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "geometry": "Actual robosuite Panda visual STL assets; no collision geometry included",
            "frame": "Origin is native grip_site; +Z approach into contact, +Y jaw opening, +X completes right-handed axes",
            "display_from_grip_site_rotation": DISPLAY_FROM_GRIP,
            "frame_derivation": "R_demo_from_grip=Rz(-90deg); T_demo_from_mesh=R_demo_from_grip * inverse(T_world_from_grip_site) * T_world_from_mesh. XML body, slide-joint and geom transforms are retained.",
            "jaw_separation_m": jaw_separation, "joint_positions_m": joint_values,
            "jaw_separation_definition": "Distance between finger joint axes; mesh surface gap differs slightly from this nominal separation",
            "normal_generation": "Area-independent neighboring facet averaging within 45 degrees; larger creases are retained",
            "rounding": {"position_m": 1e-7, "normal": 1e-6},
            "limitations": ["Visualization only; no collision, IK or physics validation"],
            "sources": [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "bytes": path.stat().st_size} for path in sorted(sources)],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size,
                      "parts": [{k: p[k] for k in ("name", "vertex_count", "triangle_count", "source_to_display")} for p in parts],
                      "bounds": bounds}))
