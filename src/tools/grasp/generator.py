"""Process-isolated GraspGenX for RoboLab's native Robotiq 2F-85.

Uses the shared candidate accumulation and path-validation contract. Checkpoints
and gripper descriptions must already be installed; inference loads no assets
from simulator object state. Direction requests are routed to MoveIt by the shared candidate tool.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from src.tools.grasp.prediction import GraspGenBackend, GraspPrediction, checked_points, checked_predictions


class GraspGenXBackend(GraspGenBackend):
    generator_name = 'graspgenx'

    def __init__(self, *, python, checkout, checkpoint_dir, gripper_dir, output_dir,
                 mesh_assets=None, num_grasps=32, topk=32, threshold=.5,
                 target_candidates=3, max_generated=96, official_clearance_m=.002, timeout_s=1800):
        self.python=str(python);self.source_checkout=Path(checkout).resolve()
        self.checkpoint_dir=Path(checkpoint_dir).resolve();self.gripper_dir=Path(gripper_dir).resolve()
        for path in (self.source_checkout/'graspgenx/grasp_server.py',
                     self.checkpoint_dir/'release/gen/config.yaml',
                     self.checkpoint_dir/'release/dis/config.yaml',
                     self.gripper_dir/'gripper_descriptions/assets/x_grippers/robotiq_2f_85/config.json'):
            if not path.is_file():raise FileNotFoundError(f'Install GraspGenX/2F-85 assets first: {path}')
        if min(num_grasps,topk,max_generated)<1 or not 0<=threshold<=1:
            raise ValueError('invalid GraspGenX generation budget')
        self.output_dir=Path(output_dir);self.mesh_assets=mesh_assets
        self.checkout=mesh_assets if mesh_assets is not None else self.source_checkout
        self.num_grasps,self.topk,self.threshold=num_grasps,topk,threshold
        self.target_candidates,self.max_generated=target_candidates,max_generated
        self.initial_official_clearance_m=self.official_clearance_m=official_clearance_m
        self.timeout_s=timeout_s;self.calls=0;self.clearance_batch_floor_m=None
        self.open_width_m=None;self.adaptive_contact_opening=False;self.libero_adapter=False
        self.preferred_direction=None

    def predict(self,object_points,scene_points):
        from src.runtime.worker import call_worker
        obj,scene=checked_points(object_points,minimum=100),checked_points(scene_points)
        self.calls+=1
        directory=self.output_dir/f'graspgen_{self.calls:03d}'
        directory.mkdir(parents=True,exist_ok=False)
        source,result=directory/'observed_points.npz',directory/'predictions.npz'
        np.savez_compressed(source,object_points=obj,scene_points=scene)
        call_worker(self.python,Path(__file__).resolve(),dict(checkout=str(self.source_checkout),
            checkpoint_dir=str(self.checkpoint_dir),gripper_dir=str(self.gripper_dir),input=str(source),
            output=str(result),num_grasps=self.num_grasps,threshold=self.threshold),
            self.output_dir/'graspgenx-worker.log',self.timeout_s)
        with np.load(result,allow_pickle=False) as data:
            predictions=checked_predictions(data['poses'],data['scores'])
        return tuple(GraspPrediction(p.pose,p.score,gripper_name='robotiq_2f_85') for p in predictions[:self.topk])

    def prewarm(self):
        from copy import copy
        worker=copy(self);worker.output_dir=self.output_dir/'prewarm/graspgenx';worker.calls=0
        worker.num_grasps=worker.topk=1
        rng=np.random.default_rng(0)
        worker.predict(rng.normal(size=(256,3),scale=.015)+[.5,0,.1],np.array([[0,0,0]],dtype=float))


_SAMPLERS={}


def _worker(args):
    # Valid overrides prevent upstream's import hook from cloning anything.
    os.environ['GRASPGENX_CHECKPOINT_DIR']=args.checkpoint_dir
    os.environ['GRASPGENX_GRIPPER_CFG_DIR']=args.gripper_dir
    sys.path.insert(0,args.checkout)
    import torch
    from graspgenx.grasp_server import GraspGenXSampler
    from graspgenx.utils.checkpoint_io import load_model_cfg
    key=(args.checkpoint_dir,args.gripper_dir)
    if key not in _SAMPLERS:
        root=Path(args.checkpoint_dir)/'release'
        cfg=load_model_cfg(str(root/'gen'),str(root/'dis'))
        _SAMPLERS[key]=GraspGenXSampler(cfg,'robotiq_2f_85',
            assets_dir=str(Path(args.gripper_dir)/'gripper_descriptions/assets'))
    with np.load(args.input,allow_pickle=False) as data:
        points=torch.as_tensor(data['object_points'],dtype=torch.float32,device='cuda')
    poses,scores,_=_SAMPLERS[key].sample(points,threshold=args.threshold,
                                       num_grasps=args.num_grasps,remove_outliers=False)
    if len(poses):
        poses,scores=poses.detach().cpu().numpy(),scores.detach().cpu().numpy()
    else:
        poses,scores=np.empty((0,4,4)),np.empty(0)
    checked_predictions(poses,scores)
    np.savez_compressed(args.output,poses=poses,scores=scores)
    Path(args.output).with_suffix('.json').write_text(json.dumps(dict(generated_count=args.num_grasps,
        accepted_count=len(scores),source='NVlabs/GraspGenX',gripper='robotiq_2f_85',
        frame='connector_base_T_gripper',poses_modified=False,scene_filter=False),indent=2)+'\n')
