"""MoveIt model exported from the same native USD as RoboLab's path checker.

USD joints have two local frames. A URDF intermediate link preserves both,
including the nonzero pivots of the Robotiq mechanism. Scene objects never enter
this model; the exporter accepts only RoboLabRobotModel's robot-only geometry.
"""
from itertools import combinations
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from src.simulator.robolab.gripper import FLANGE_FROM_GRASP
from src.simulator.robolab.robot_model import _text


def excluded_pair(a, b, parents):
    """Same adjacent-link/internal-mechanism policy as RoboLabPathCollision."""
    if not a.startswith('panda_link') and not b.startswith('panda_link'):
        return True
    ancestors = {a: 0}
    for distance in range(1, 3):
        a = parents.get(a)
        if a is None:
            break
        ancestors[a] = distance
    for distance in range(3):
        if b in ancestors and distance + ancestors[b] <= 2:
            return True
        b = parents.get(b)
        if b is None:
            break
    return False


def add_collision_policy(semantic, links, parents):
    """Ignore the fixed mount's world contact, retaining native self checks.

    Like LIBERO's panda.srdf, the fixed base may contact its support surface.
    Re-enable every non-excluded base/robot pair: unlike LIBERO, RoboLab has
    no separate *_sc links, so disabling only the default would hide self hits.
    """
    base = 'panda_link0'
    ET.SubElement(semantic, 'disable_default_collisions', link=base)
    for a, b in combinations(links, 2):
        if excluded_pair(a, b, parents):
            ET.SubElement(semantic, 'disable_collisions', link1=a, link2=b, reason='native_pair_policy')
        elif base in (a, b):
            ET.SubElement(semantic, 'enable_collisions', link1=a, link2=b)


def export_moveit_model(native, output, *, asset_path):
    import trimesh
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    robot = ET.Element('robot', name='robolab_droid')
    short = lambda path: path.rsplit('/', 1)[-1]
    links = {short(path): ET.SubElement(robot, 'link', name=short(path))
             for path in native.body_paths}
    parents = {}

    def origin(element, matrix):
        # At a +/-pi/2 pitch Euler angles are nonunique, but still encode the
        # exact rotation. SciPy's canonical choice is verified by FK checks.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Gimbal lock detected', category=UserWarning)
            rpy = Rotation.from_matrix(matrix[:3, :3]).as_euler('xyz')
        ET.SubElement(element, 'origin', xyz=_text(matrix[:3, 3]), rpy=_text(rpy))

    def joint(name, parent, child, matrix, kind='fixed'):
        element = ET.SubElement(robot, 'joint', name=name, type=kind)
        ET.SubElement(element, 'parent', link=parent)
        ET.SubElement(element, 'child', link=child)
        origin(element, matrix)
        return element

    for row in native.joints:
        parent, child = short(row['parent']), short(row['child'])
        parents[child] = parent
        if not row['revolute']:
            joint(child + '_fixed', parent, child, row['local0'] @ np.linalg.inv(row['local1']))
            continue
        pivot = row['name'] + '_pivot'
        ET.SubElement(robot, 'link', name=pivot)
        element = joint(row['name'], parent, pivot, row['local0'], 'revolute')
        ET.SubElement(element, 'axis', xyz=_text(np.eye(3)[row['axis']]))
        ET.SubElement(element, 'limit', lower=str(row['limits'][0]), upper=str(row['limits'][1]),
                      effort='87', velocity='2.175')
        if 'mimic' in row:
            source, ratio = row['mimic']
            ET.SubElement(element, 'mimic', joint=source, multiplier=str(ratio), offset='0')
        joint(row['name'] + '_body', pivot, child, np.linalg.inv(row['local1']))

    hashes = {}
    for name, triangles in native.visual_triangles.items():
        # Match the conservative per-shape convex geometry of native path checks.
        mesh = trimesh.convex.convex_hull(triangles.reshape(-1, 3))
        path = output / (name + '.stl')
        mesh.export(path)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        collision = ET.SubElement(links[native.geom_body[name]], 'collision', name=name)
        ET.SubElement(ET.SubElement(collision, 'geometry'), 'mesh', filename=path.as_uri())

    ET.SubElement(robot, 'link', name='robolab_grasp')
    joint('robolab_grasp_frame', 'base_link', 'robolab_grasp', FLANGE_FROM_GRASP)
    semantic = ET.Element('robot', name='robolab_droid')
    group = ET.SubElement(semantic, 'group', name='panda_arm')
    ET.SubElement(group, 'chain', base_link='panda_link0', tip_link='robolab_grasp')
    add_collision_policy(semantic, links, parents)
    for name, tree in (('robot.urdf', robot), ('robot.srdf', semantic)):
        ET.indent(tree)
        ET.ElementTree(tree).write(output/name, encoding='utf-8', xml_declaration=True)
        hashes[name] = hashlib.sha256((output/name).read_bytes()).hexdigest()
    metadata = dict(source_asset=str(asset_path), source_sha256=hashlib.sha256(Path(asset_path).read_bytes()).hexdigest(),
        gripper='robotiq_2f_85', max_opening_m=.085, tip_link='robolab_grasp',
        geometry='per-shape convex hulls from native robot USD; no scene objects',
        pair_policy='exclude graph distance <=2 and internal gripper mechanism pairs',
        fixed_base_world_policy='panda_link0 world collision disabled for fixed support contact; native robot self-collision pairs retained',
        joint_count=len(native.joints), mesh_count=len(hashes)-2, files_sha256=hashes)
    (output/'model.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return output
