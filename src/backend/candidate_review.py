"""Backend geometry for the review-driven Prime/Pointer/Selector/Refiner/Place policy."""
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4
import json
import numpy as np

from src.tools.place.observed_placement import ObservedPlacementBackend, measured_anchor
from src.backend.robot_base import LiveBackend
from src.tools.pose_editor.geometry import VIEW_RULES, project_measured, short_view_pose, checked_translation


class ReviewDrivenBackend(ObservedPlacementBackend):
    postgrasp_destination = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.region_tracks = {}
        self.view_proposals = {}
        self.view_validations = {}

    def select_region(self, observation_id, view_id, u, v, *, purpose='pick', target_ref=None):
        """One human-intent click; backend projects measured surfaces into the other RGBD."""
        from src.tools.perception.multiview import FusedGeometry, voxel_merge
        from src.tools.perception.intent import measured_target_consistency
        if purpose not in ('pick', 'place'):raise ValueError('invalid selection purpose')
        if purpose == 'place' and (not self.grasp_attempted or self.held_plan is None):
            raise ValueError('destination requires an executed grasp and fresh observation')
        self.point_adapter._check_current(observation_id)
        initial=LiveBackend.point(self, observation_id, view_id, u, v, role=purpose)
        geometry=self._point(initial['point_ref']); selected=[geometry]
        overlays=list(initial['image_refs']); notes=[]
        for frame in self.point_adapter.frames[observation_id]:
            if frame.view_id == view_id:continue
            pixel=project_measured(frame,geometry.object_points)
            if pixel is None:
                notes.append(dict(view_id=frame.view_id,status='target_not_depth_visible'));continue
            try:
                other=LiveBackend.point(self,observation_id,frame.view_id,*pixel,role=purpose)
                alternate=self._point(other['point_ref'])
                audit=measured_target_consistency(geometry.object_points,alternate.object_points)
                if audit['status']=='disagreement':
                    notes.append(dict(view_id=frame.view_id,status='automatic_mask_disagrees'));continue
                selected.append(alternate);overlays+=other['image_refs']
            except Exception as exc:
                self._record('automatic_region_projection_failed',dict(observation_id=observation_id,error=repr(exc)))
                notes.append(dict(view_id=frame.view_id,status='automatic_selection_failed'))
        # FusedGeometry supports one or two measured views. Unknown surfaces stay absent.
        fused=FusedGeometry(observation_id,str(self.epoch),'backend depth-consistent target projection',tuple(selected),
            voxel_merge({g.view_id:g.object_points for g in selected}),
            voxel_merge({g.view_id:g.scene_points for g in selected}),.002)
        ref='pt_'+uuid4().hex;self.points[ref]=(self.epoch,fused);self.latest_fused_ref=ref
        anchor=measured_anchor(geometry.frame,u,v)
        if purpose=='place':self.destination_anchors[ref]=anchor
        track=target_ref or 'target_'+uuid4().hex
        self.region_tracks[track]=dict(point_ref=ref,anchor=anchor,points=fused.object_points.copy(),purpose=purpose)
        result=dict(point_ref=ref,target_ref=track,observation_id=observation_id,image_refs=overlays,
                    source_views=list(fused.source_views),geometry_status='paired_measured' if len(selected)==2 else 'single_measured',
                    missing_views=notes)
        self._record('region_overlay', {**result,'purpose':purpose,'clicked_view':view_id,'u':u,'v':v,
                                      'backend_policy':'depth-consistent reprojection; Pointer reviews masks only'})
        return result

    def refresh_target(self,target_ref):
        """Reuse a stationary target identity only when current measured surfaces agree."""
        from src.tools.perception.intent import measured_target_consistency
        track=self.region_tracks[target_ref];old=track['points'].copy()
        frame=self.point_adapter.frames[self.latest_observation_id][0]
        pixel=project_measured(frame,old)
        if pixel is None:return {'target_ref':target_ref,'tracking_status':'needs_pointer'}
        result=self.select_region(self.latest_observation_id,frame.view_id,*pixel,purpose=track['purpose'],target_ref=target_ref)
        new=self.region_tracks[target_ref]
        consistent=measured_target_consistency(old,new['points'])['status']!='disagreement'
        consistent=consistent and np.linalg.norm(np.median(old,axis=0)-np.median(new['points'],axis=0))<.06
        if not consistent:
            self.region_tracks[target_ref]=track
            return {'target_ref':target_ref,'tracking_status':'needs_pointer'}
        result['tracking_status']='retained';self._record('target_refreshed',result)
        return result

    def view_waypoint(self,observation_id,u,v,direction,*,target_ref=None):
        from src.tools.motion.planning import _pose_transform
        from src.tools.observation.views import fresh_wrist_state
        self.point_adapter._check_current(observation_id)
        if self.held_plan is not None:raise ValueError('view motion while holding is unavailable')
        frame=self.point_adapter.frames[observation_id][0]
        anchor=measured_anchor(frame,u,v)
        state=fresh_wrist_state(self.connector)
        pose=short_view_pose(state['ee'],state['ee_from_optical'],anchor,direction)
        ref='view_'+uuid4().hex
        geometry=self.points[self.region_tracks[target_ref]['point_ref']][1] if target_ref else None
        if geometry is None:raise ValueError('view waypoint requires retained target evidence')
        self.view_proposals[ref]=dict(epoch=self.epoch,observation_id=observation_id,pose=pose,
            anchor=anchor,target_ref=target_ref,geometry=geometry,direction=direction,
            translation=(0.,0.,0.),rotation=(0.,0.,0.),start=self._robot_state())
        output=self.preview_view(ref,-45.,30.,1.)
        self._record('view_waypoint',dict(waypoint_ref=ref,target_ref=target_ref,observation_id=observation_id,
            front_click=[u,v],direction=direction,anchor_base=anchor.tolist(),ee_pose=pose.tolist(),rules=VIEW_RULES))
        return {**output,'observation_id':observation_id,'rules':VIEW_RULES}

    def continue_view(self,target_ref,direction):
        track=self.region_tracks[target_ref]
        frame=self.point_adapter.frames[self.latest_observation_id][0]
        pixel=project_measured(frame,track['points'])
        if pixel is None:raise ValueError('target no longer matches current front depth; request Pointer')
        return self.view_waypoint(self.latest_observation_id,*pixel,direction,target_ref=target_ref)

    def _view(self,ref):
        proposal=self.view_proposals[ref]
        if proposal['epoch']!=self.epoch:raise ValueError('stale view pose')
        self.point_adapter._check_current(proposal['observation_id'])
        return proposal

    def preview_view(self,waypoint_ref,azimuth_deg,elevation_deg,zoom):
        from src.tools.pose_editor.preview_controls import render_preview
        p=self._view(waypoint_ref)
        prediction=SimpleNamespace(pose=p['pose']@np.linalg.inv(self.grasp_to_ee),gripper_adapter=getattr(self.graspgen, 'libero_adapter', True))
        result=render_preview(p['geometry'],prediction,self.output_dir/('view_preview_'+uuid4().hex),
            candidate_ref=waypoint_ref,azimuth_deg=azimuth_deg,elevation_deg=elevation_deg,zoom=zoom,
            graspgen_root=self.graspgen.checkout)
        output=dict(waypoint_ref=waypoint_ref,image_refs=[self.images.add(p) for p in result['image_paths']])
        self._record('view_pose_preview',{**output,**result,'robot_motion':False})
        return output

    def refine_view(self,waypoint_ref,dx_mm,dy_mm,dz_mm,roll_deg,pitch_deg,yaw_deg):
        from src.tools.pose_editor.refinement import checked_adjustment,_local_rotation
        p=self._view(waypoint_ref)
        step,total=checked_translation((dx_mm,dy_mm,dz_mm),p['translation'])
        angles,cumulative=checked_adjustment((roll_deg,pitch_deg,yaw_deg),p['rotation'])
        pose=p['pose'].copy();pose[:3,3]+=pose[:3,:3]@(np.asarray(step)/1000)
        pose[:3,:3]=pose[:3,:3]@_local_rotation(angles)
        ref='view_'+uuid4().hex
        self.view_proposals[ref]={**p,'pose':pose,'translation':total,'rotation':cumulative}
        result=self.preview_view(ref,-45.,30.,1.)
        self._record('view_refined',dict(origin_ref=waypoint_ref,waypoint_ref=ref,ee_pose=pose.tolist(),
            translation_local_mm=step,rotation_local_deg=angles))
        return result

    def validate_view(self,waypoint_ref):
        from src.tools.motion.planning import _plan
        from src.tools.motion.path_collision import make_candidate_path_collision as CandidatePathCollision
        p=self._view(waypoint_ref);ref='vv_'+uuid4().hex
        try:
            checker=CandidatePathCollision(self.connector,clearance_m=.0005)
            scene,_=checker.remove_captured_robot(np.vstack([p['geometry'].scene_points,p['geometry'].object_points]))
            segments,targets,*_=_plan(self.connector,(p['pose'],),scene,self.motion_config,None)
            plan=SimpleNamespace(segments=segments,targets=targets,target_labels=('view',),segment_labels=('view',),
                transit_policy='short_observation',high_transit_z_m=None,start_joints=json.loads(self._robot_state())['joints'])
            evidence=checker.check(plan,scene,stop_label='view',jaw_width_m=getattr(self, 'max_gripper_width_m', .08))
            accepted=bool(evidence['accepted'])
            if accepted:self.view_validations[ref]=(waypoint_ref,self._robot_state(),plan)
        except Exception as exc:
            accepted=False;evidence={'error':repr(exc)}
        self._record('view_validation',dict(waypoint_ref=waypoint_ref,validation_ref=ref,accepted=accepted,evidence=evidence))
        return dict(waypoint_ref=waypoint_ref,validation_ref=ref,accepted=accepted,
                    reason_code='accepted' if accepted else 'view_path_rejected')

    def execute_view(self,waypoint_ref,validation_ref):
        from src.tools.motion.planning import _execute_checked,_pose_transform
        p=self._view(waypoint_ref)
        ref,state,plan=self.view_validations.pop(validation_ref)
        if ref!=waypoint_ref or state!=self._robot_state():raise ValueError('view validation stale or mismatched')
        before=_pose_transform(self.connector.get_ee_pose())
        self.epoch+=1;self.validations.clear();self.latest_fused_ref=None;self.point_adapter.latest=None
        self.latest_observation_id=None;status='achieved'
        try:
            if self.recorder:self.recorder.register_plan(plan,kind='observation',point_ref=waypoint_ref)
            with self.recorder.active('execute_view',point_ref=waypoint_ref) if self.recorder else nullcontext():
                _execute_checked(self.connector,plan.segments,plan.targets,
                                 collision_checks_enabled=self.motion_config.collision_checks_enabled)
            actual=_pose_transform(self.connector.get_ee_pose())
            if np.linalg.norm(actual[:3,3]-p['pose'][:3,3])>.02:status='requested_view_failed'
        except Exception as exc:
            status='requested_view_failed';self._record('view_execution_error',dict(error=repr(exc),waypoint_ref=waypoint_ref))
        actual=_pose_transform(self.connector.get_ee_pose())
        progress=float(np.linalg.norm(actual[:3,3]-before[:3,3]))
        observation=self.observe()
        try:tracked=self.refresh_target(p['target_ref'])
        except Exception:tracked=dict(target_ref=p['target_ref'],tracking_status='needs_pointer')
        output={**observation,'view_status':status,'translation_m':progress,'target_update':tracked}
        self._record('view_executed',{**output,'waypoint_ref':waypoint_ref,'validation_ref':validation_ref})
        return output

    def adjust_grasp(self,**kwargs):
        return self.refine_candidate(**kwargs)

    def _placement_scene(self,plan,destination):
        # Destination was just captured AFTER grasp. Remove current robot and held cloud.
        from src.tools.motion.path_collision import make_candidate_path_collision as CandidatePathCollision
        from scipy.spatial import cKDTree
        from src.tools.motion.planning import _pose_transform
        checker=CandidatePathCollision(self.connector,clearance_m=.0005)
        scene,_=checker.remove_captured_robot(destination.scene)
        relative=_pose_transform(self.connector.get_ee_pose())@np.linalg.inv(self.grasp_attachment)
        held=plan.target_points@relative[:3,:3].T+relative[:3,3]
        distances,_=cKDTree(held).query(scene)
        scene=scene[distances>.012]
        self._record('postgrasp_placement_scene',dict(destination_ref=self.selected_destination_ref,
            observation_id=destination.observation_id,removed_held_samples=int((distances<=.012).sum())))
        return np.vstack([scene,destination.points])
