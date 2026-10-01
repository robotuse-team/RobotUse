"""Private robot-only collision/FK model extracted from RoboLab's native USD.

MuJoCo is used only for offline geometry queries, never to run RoboLab physics.
Only bodies reachable from panda_link0 through robot joints are admitted. USD
scene siblings (including objects embedded in the asset) are excluded.
"""
from __future__ import annotations
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation


def _text(values):
    return ' '.join(format(float(v), '.10g') for v in np.asarray(values).reshape(-1))


def _matrix(pos, quat):
    value = np.eye(4)
    value[:3, 3] = pos
    value[:3, :3] = Rotation.from_quat([*quat.GetImaginary(), quat.GetReal()]).as_matrix()
    return value


class RoboLabRobotModel:
    def __init__(self, asset_path):
        import mujoco
        import trimesh
        from pxr import Usd, UsdGeom, UsdPhysics
        stage = Usd.Stage.Open(str(asset_path))
        root = '/panda/panda_link0'
        joints = []
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            parents, children = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if not parents or not children:
                continue
            parent, child = str(parents[0]), str(children[0])
            if not parent.startswith('/panda/') or not child.startswith('/panda/'):
                continue
            revolute = prim.IsA(UsdPhysics.RevoluteJoint)
            if not revolute and not prim.IsA(UsdPhysics.FixedJoint):
                raise ValueError(f'unsupported robot joint {prim.GetPath()}')
            local0 = _matrix(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
            local1 = _matrix(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())
            row = dict(parent=parent, child=child, name=prim.GetName(), local0=local0, local1=local1,
                       revolute=revolute)
            if revolute:
                rj = UsdPhysics.RevoluteJoint(prim)
                row.update(axis='XYZ'.index(rj.GetAxisAttr().Get()),
                           limits=np.radians([rj.GetLowerLimitAttr().Get(), rj.GetUpperLimitAttr().Get()]))
                for rel in prim.GetRelationships():
                    if rel.GetName().endswith(':referenceJoint') and rel.GetTargets():
                        prefix = rel.GetName().rsplit(':', 1)[0]
                        row['mimic'] = (rel.GetTargets()[0].name,
                            -float(prim.GetAttribute(prefix+':gearing').Get() or 0))
            joints.append(row)
        tree = {root: None}
        ordered = []
        remaining = joints.copy()
        while remaining:
            ready = [j for j in remaining if j['parent'] in tree]
            if not ready:
                raise ValueError('robot joint graph is disconnected')
            for j in ready:
                if j['child'] in tree:
                    raise ValueError('robot joint graph contains multiple parents')
                tree[j['child']] = j
                ordered.append(j)
                remaining.remove(j)
        self.joints = ordered
        self.body_paths = tree
        xml = ET.Element('mujoco', model='robolab_droid_collision')
        ET.SubElement(xml, 'compiler', angle='radian', autolimits='true', balanceinertia='true')
        ET.SubElement(xml, 'option', gravity='0 0 0')
        asset = ET.SubElement(xml, 'asset')
        world = ET.SubElement(xml, 'worldbody')
        bodies = {root: ET.SubElement(world, 'body', name='panda_link0')}
        for j in ordered:
            local = j['local0'] @ np.linalg.inv(j['local1'])
            body = ET.SubElement(bodies[j['parent']], 'body', name=j['child'].rsplit('/',1)[-1],
                pos=_text(local[:3,3]), quat=_text(Rotation.from_matrix(local[:3,:3]).as_quat()[[3,0,1,2]]))
            bodies[j['child']] = body
            if j['revolute']:
                axis = j['local1'][:3,:3][:,j['axis']]
                ET.SubElement(body, 'joint', name=j['name'], type='hinge',
                    pos=_text(j['local1'][:3,3]), axis=_text(axis), range=_text(j['limits']))
            ET.SubElement(body, 'inertial', pos='0 0 0', mass='1', diaginertia='.01 .01 .01')
        self.visual_triangles = {}
        self.geom_body = {}
        cache = UsdGeom.XformCache()
        index = 0
        for body_path, body in bodies.items():
            body_prim = stage.GetPrimAtPath(body_path)
            body_world = np.asarray(cache.GetLocalToWorldTransform(body_prim)).T
            for prim in Usd.PrimRange(body_prim, Usd.TraverseInstanceProxies()):
                # Descendant rigid bodies own their own geometry.
                ancestor = prim
                while ancestor and str(ancestor.GetPath()) not in tree:
                    ancestor = ancestor.GetParent()
                if str(ancestor.GetPath()) != body_path:
                    continue
                if not prim.IsA(UsdGeom.Mesh) and not prim.IsA(UsdGeom.Cylinder):
                    continue
                local = np.linalg.inv(body_world) @ np.asarray(cache.GetLocalToWorldTransform(prim)).T
                if prim.IsA(UsdGeom.Mesh):
                    mesh = UsdGeom.Mesh(prim)
                    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
                    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
                    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
                    faces, offset = [], 0
                    for count in counts:
                        face = indices[offset:offset+count]; offset += count
                        faces.extend([[face[0],face[k],face[k+1]] for k in range(1,count-1)])
                    vertices = points @ local[:3,:3].T + local[:3,3]
                    triangles = vertices[np.asarray(faces)]
                else:
                    shape = UsdGeom.Cylinder(prim)
                    mesh = trimesh.creation.cylinder(radius=float(shape.GetRadiusAttr().Get()),
                        height=float(shape.GetHeightAttr().Get()), sections=24)
                    axis = shape.GetAxisAttr().Get()
                    if axis == 'X': mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[0,1,0]))
                    if axis == 'Y': mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0]))
                    mesh.apply_transform(local)
                    triangles = mesh.triangles
                if not len(triangles):
                    continue
                name = f'robot_mesh_{index}'; index += 1
                self.visual_triangles[name] = triangles.copy()
                self.geom_body[name] = body_path.rsplit('/',1)[-1]
                # Convex hulls conservatively enclose source collision/visual geometry.
                hull = trimesh.convex.convex_hull(triangles.reshape(-1,3))
                ET.SubElement(asset,'mesh',name=name,vertex=_text(hull.vertices),face=_text(hull.faces))
                ET.SubElement(body,'geom',name=name,type='mesh',mesh=name,contype='0',conaffinity='0',group='0')
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
        self.data = mujoco.MjData(self.model)
        self.arm_addresses = [self.model.jnt_qposadr[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,f'panda_joint{i}')] for i in range(1,8)]
        self.mimic = {j['name']:j['mimic'] for j in ordered if 'mimic' in j}
        self.flange_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
        if self.flange_id < 0 or not self.visual_triangles:
            raise ValueError('native Robotiq robot geometry missing')

    def set_joints(self, joints, finger_angle):
        import mujoco
        values = {f'panda_joint{i+1}': float(q) for i,q in enumerate(joints)}
        values['finger_joint'] = float(finger_angle)
        for name, (source, ratio) in self.mimic.items():
            values[name] = values[source] * ratio
        for name,value in values.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,name)
            self.data.qpos[self.model.jnt_qposadr[jid]] = value
        mujoco.mj_kinematics(self.model, self.data)

    def body_matrix(self, name):
        import mujoco
        bid = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,name)
        if bid < 0: raise ValueError(f'unknown robot body {name}')
        result = np.eye(4)
        result[:3,:3] = self.data.xmat[bid].reshape(3,3)
        result[:3,3] = self.data.xpos[bid]
        return result
