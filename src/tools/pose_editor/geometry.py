"""Measured front-point viewing rules and bounded local pose corrections."""
import numpy as np
from src.tools.observation.views import look_at_optical


VIEW_RULES = dict(surface_depth='registered depth at clicked front pixel; invalid depth rejects',
                  clearance_m=.22, max_translation_m=.12, max_rotation_deg=20.,
                  lateral_offset_m=.18, max_requests=4, minimum_progress_m=.01,
                  repeated_no_progress_limit=2, unknown_surfaces='never completed')


def checked_translation(step, applied=(0., 0., 0.), *, step_limit_mm=10, cumulative_limit_mm=30,
                        unrestricted=False):
    if len(step) != 3 or len(applied) != 3 or any(
            type(v) not in (int, float) or not np.isfinite(v) for v in (*step, *applied)):
        raise ValueError('translation and previous totals must contain three finite numbers')
    if not unrestricted and any(abs(v) > step_limit_mm for v in step):
        raise ValueError(f'local translation must be finite and within {step_limit_mm} mm per axis')
    total = tuple(a+b for a,b in zip(step, applied))
    if not all(np.isfinite(v) for v in total):
        raise ValueError('cumulative translation must remain finite')
    if not unrestricted and any(abs(v)>cumulative_limit_mm for v in total):
        raise ValueError(f'cumulative local translation exceeds {cumulative_limit_mm} mm')
    return tuple(step), total


def project_measured(frame, points, tolerance_m=.03):
    """Project existing measured surfaces; return only depth-consistent pixels."""
    rotation=np.asarray(frame.camera_to_base.rotation);origin=np.asarray(frame.camera_to_base.translation)
    local=(np.asarray(points)-origin) @ rotation
    valid=local[:,2]>.02;local=local[valid]
    if not len(local):return None
    k=np.asarray(frame.intrinsics)
    xy=local[:,:2]/local[:,2,None]*[k[0,0],k[1,1]]+[k[0,2],k[1,2]]
    inside=(xy[:,0]>=0)&(xy[:,0]<frame.width)&(xy[:,1]>=0)&(xy[:,1]<frame.height)
    xy,local=xy[inside],local[inside]
    if not len(xy):return None
    pix=np.rint(xy).astype(int);pix[:,0]=np.clip(pix[:,0],0,frame.width-1);pix[:,1]=np.clip(pix[:,1],0,frame.height-1)
    depth=frame.depth_m[pix[:,1],pix[:,0]]
    visible=np.isfinite(depth)&(np.abs(depth-local[:,2])<tolerance_m)
    if visible.sum()<8:return None
    xy=xy[visible];center=np.median(xy,axis=0)
    x,y=xy[np.argmin(np.linalg.norm(xy-center,axis=1))]
    return float(x/(frame.width-1)*1000),float(y/(frame.height-1)*1000)


def short_view_pose(current, ee_from_optical, anchor, direction):
    """Aim calibrated wrist at the measured reference, with short SE(3) steps."""
    from scipy.spatial.transform import Rotation, Slerp
    offsets={'up':(0,0), 'left':(0,.18), 'right':(0,-.18), 'front':(-.18,0), 'rear':(.18,0)}
    if direction not in offsets:raise ValueError('unknown observation direction')
    anchor=np.asarray(anchor,float);current=np.asarray(current,float)
    position=anchor.copy();position[:2]+=offsets[direction]
    position[2]=max(anchor[2]+VIEW_RULES['clearance_m'],current[2,3]+.04)
    desired=look_at_optical(position,anchor) @ np.linalg.inv(ee_from_optical)
    delta=desired[:3,3]-current[:3,3];distance=float(np.linalg.norm(delta))
    pose=current.copy();pose[:3,3]+=delta*min(1.,VIEW_RULES['max_translation_m']/max(distance,1e-9))
    # Rise before significant reorientation/lateral travel near the surface.
    if current[2,3]<anchor[2]+.14:
        pose=current.copy();pose[2,3]+=min(.12,anchor[2]+.18-current[2,3])
    else:
        rotations=Rotation.from_matrix(np.stack([current[:3,:3],desired[:3,:3]]))
        angle=float((rotations[0].inv()*rotations[1]).magnitude())
        fraction=min(1.,np.radians(20)/max(angle,1e-9))
        pose[:3,:3]=Slerp([0,1],rotations)([fraction]).as_matrix()[0]
    return pose
