"""Opt-in waypoint straight translation; retain the current TCP orientation."""
from types import SimpleNamespace
import numpy as np
from src.backend.collision_policy import RelaxedIntentBackend, RelaxedIntentOrchestrator
from src.backend.controller import ARGUMENTS

# Waypoint validation accepts planned or straight-translation motion.
ARGUMENTS['validate_view'] = ('waypoint_ref', 'motion')


def straight_targets(start, endpoint, spacing=.01):
    distance=float(np.linalg.norm(endpoint[:3,3]-start[:3,3]))
    count=max(1,int(np.ceil(distance/spacing)))
    result=[]
    for fraction in np.linspace(0,1,count+1)[1:]:
        pose=start.copy();pose[:3,3]=start[:3,3]+fraction*(endpoint[:3,3]-start[:3,3])
        result.append(pose)
    return tuple(result)


def line_error(pose, start, endpoint):
    delta=endpoint[:3,3]-start[:3,3];length2=float(delta@delta)
    fraction=float((pose[:3,3]-start[:3,3])@delta/length2) if length2>1e-12 else 0.
    nearest=start[:3,3]+np.clip(fraction,0,1)*delta
    distance=float(np.linalg.norm(pose[:3,3]-nearest))
    angle=float(np.arccos(np.clip((np.trace(start[:3,:3].T@pose[:3,:3])-1)/2,-1,1)))
    return distance,angle


class LinearIntentBackend(RelaxedIntentBackend):
    def validate_view(self, waypoint_ref, motion='planned'):
        if motion not in ('planned','linear'):raise ValueError('motion must be planned or linear')
        p=self._view(waypoint_ref)
        if 'original_waypoint_pose' not in p:
            p['original_waypoint_pose']=p['pose'].copy()
            p['original_orientation_policy']=p['orientation_policy']
        p['pose']=p['original_waypoint_pose'].copy()
        p['orientation_policy']=p['original_orientation_policy']
        # A change of mode must not leave an executable token for the older route.
        self.view_validations={k:v for k,v in self.view_validations.items() if v[0]!=waypoint_ref}
        p['motion']=motion
        result=super().validate_view(waypoint_ref)
        if result['accepted']:
            plan=self.view_validations[result['validation_ref']][2]
            plan.linear_motion=motion=='linear'
        return {**result,'motion':motion,'orientation_policy':'preserve_current' if motion=='linear' else p['orientation_policy']}

    def execute_view(self, waypoint_ref, validation_ref):
        mode=self._view(waypoint_ref).get('motion','planned')
        result=super().execute_view(waypoint_ref,validation_ref)
        return {**result,'motion':mode,'reason_code':'achieved' if result.get('view_status')=='achieved' else 'waypoint_execution_incomplete'}

    def _plan_waypoint(self, p, scene, checker):
        if p.get('motion')!='linear':return super()._plan_waypoint(p,scene,checker)
        import mujoco
        from src.tools.motion.planning import _pose_transform, _plan, MotionPlanningError
        from src.tools.motion.path_collision import path_samples
        start=_pose_transform(self.connector.get_ee_pose())
        endpoint=p['pose'].copy();endpoint[:3,:3]=start[:3,:3]
        # Also update the endpoint feedback/recorder to the orientation actually requested.
        p['pose']=endpoint;p['orientation_policy']='preserve_current'
        result=_plan(self.connector,straight_targets(start,endpoint),scene,self.motion_config,None)
        segments,targets,joints,*_=result
        plan=SimpleNamespace(segments=segments,targets=targets,start_joints=joints,
            target_labels=tuple(f'waypoint_step_{i}' for i in range(len(targets)-1))+('waypoint',),segment_labels=tuple(f'waypoint_step_{i}' for i in range(len(targets)-1))+('waypoint',))
        if hasattr(checker, 'native'):
            max_distance = max_angle = 0.
            for q, _ in path_samples(plan, checker.radii, 'waypoint'):
                pose = self.connector.ik.model.fk(q)
                distance, angle = line_error(pose, start, endpoint)
                max_distance, max_angle = max(max_distance, distance), max(max_angle, angle)
                if distance > .003 or angle > .05:
                    raise MotionPlanningError('planned native path leaves the straight EE corridor')
            self._record('linear_waypoint_plan', dict(waypoint_motion='linear',
                max_path_error_m=max_distance, max_orientation_error_rad=max_angle))
            return result
        hand=next(g for g,n in checker.names.items() if n=='gripper0_hand_collision')
        position,rotation=checker.capture_poses[hand]
        capture=np.eye(4);capture[:3,:3]=rotation;capture[:3,3]=position
        hand_to_tcp=np.linalg.inv(capture)@start
        max_distance=max_angle=0.
        for q,_ in path_samples(plan,checker.radii+float(np.linalg.norm(hand_to_tcp[:3,3])),'waypoint'):
            checker.data.qpos[checker.addresses]=q;mujoco.mj_kinematics(checker.model,checker.data)
            position,rotation=checker._pose(hand)
            pose=np.eye(4);pose[:3,:3]=rotation;pose[:3,3]=position
            distance,angle=line_error(pose@hand_to_tcp,start,endpoint)
            max_distance=max(max_distance,distance);max_angle=max(max_angle,angle)
            if distance>.003 or angle>.05:
                raise MotionPlanningError('planned joint path leaves the straight TCP corridor; no detour allowed')
        self._record('linear_waypoint_plan',dict(waypoint_motion='linear',max_path_error_m=max_distance,
            max_orientation_error_rad=max_angle,targets=len(targets),position_tolerance_m=.003,
            orientation_tolerance_rad=.05,execution_check_spacing_m=.01))
        return result

    def _execute_waypoint(self, plan):
        if not getattr(plan,'linear_motion',False):return super()._execute_waypoint(plan)
        from src.tools.motion.planning import _execute_checked, MotionPlanningError
        from src.tools.motion.tolerances import cartesian_tolerances
        position_limit, angle_limit = cartesian_tolerances(self.connector, position=.005, orientation=.05)
        results=[]
        for segment,target in zip(plan.segments,plan.targets):
            feedback=_execute_checked(self.connector,(segment,),(target,),collision_checks_enabled=self.motion_config.collision_checks_enabled)[0]
            results.append(feedback)
            self._record('linear_waypoint_tracking',feedback)
            if feedback['position_error_m']>position_limit or feedback['orientation_error_rad']>angle_limit:
                raise MotionPlanningError('linear waypoint tracking failed; stopped before next subsegment')
        return results


class LinearIntentOrchestrator(RelaxedIntentOrchestrator):
    def _prompt(self, role):
        prompt=super()._prompt(role)
        if role in ('prime','point'):
            prompt+=(' Existing waypoints also support straight translation. Prime calls validate_view '
                'with motion="linear" to preserve CURRENT gripper orientation and travel straight to the '
                'waypoint position, then uses the existing execute_view and validation_ref. motion="planned" '
                'keeps ordinary waypoint planning. No new preview step is required. Use short contact '
                'waypoints for pushing or pulling; inspect returned RGB before continuing. Keep existing '
                'contact grasp/closed-pusher preconditions and turn for local-Z rotation. A rejected '
                'straight route never executes a detour. No fixed manipulation sequence is required.')
        return prompt
