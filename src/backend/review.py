"""Review-driven policy with explicit observation and refinement decisions."""
from copy import deepcopy
from src.backend.controller import PrimeOrchestrator, ARGUMENTS, BoundaryError, BudgetExceeded, public_result, _text
from src.tools.place.review import PlaceAgent

DELTAS=('dx_mm','dy_mm','dz_mm','roll_deg','pitch_deg','yaw_deg')
ARGUMENTS.update({
    'select_region':('observation_id','view_id','u','v'),
    'view_waypoint':('observation_id','u','v','direction'),
    'continue_view':('target_ref','direction'),
    'preview_view':('waypoint_ref','azimuth_deg','elevation_deg','zoom'),
    'refine_view':('waypoint_ref',*DELTAS),
    'validate_view':('waypoint_ref',),'execute_view':('waypoint_ref','validation_ref'),
    'adjust_grasp':('candidate_ref',*DELTAS),
    'delegate_refiner':('instruction','candidate_ref'),
    'delegate_view_refiner':('instruction','waypoint_ref'),
})

COMMON=('Only supplied RGB, overlays and measured-surface previews are available. '
        'Current observations are reused while stationary and refreshed after execution. '
        'A virtual preview never reveals unmeasured hidden surfaces, performs motion, or establishes contact. '
        'Use opaque references and image pixels; backend owns geometry, IK and path calculations. ')


class ReviewDrivenOrchestrator(PrimeOrchestrator):
    def __init__(self,*args,**kwargs):
        kwargs.update(active_perception=True,first_grasp_only=False,debug_reset_on_failed_grasp=False,waypoint_views=False)
        super().__init__(*args,**kwargs)
        self.target_intent_mode=True
        self.place_agent=PlaceAgent()
        self.destinations=set();self.target_ref=None;self.current_point=None
        self.observation_requested=False;self.refinement_choices=set()
        self.view_requests=0;self.no_progress=0

    def _tools(self,role,task):
        if role=='prime':return ('delegate_point','delegate_grasp','delegate_refiner','delegate_waypoint',
            'delegate_view_refiner','continue_view','validate_view','execute_view','validate_grasp','execute_grasp',
            'delegate_destination','save_destination','delegate_place','validate_place','execute_place_candidate','finish')
        if role=='point':return ('view_waypoint','finish') if task.get('waypoint_task') else ('select_region','finish')
        if role=='grasp':return ('inspect_candidate','preview_candidate','finish')
        if role=='refiner':return ('preview_view','refine_view','finish') if task.get('waypoint_ref') else ('inspect_candidate','preview_candidate','adjust_grasp','finish')
        if role=='place':return self.place_agent.tools
        raise BoundaryError('unknown role')

    def _agent_observation(self,role,task,observation):
        if role=='point' and task.get('waypoint_task'):
            return {**observation,'views':observation['views'][:1]}
        return observation

    def _loop(self,role,task,parent_id=None):
        if task.get('observation'):
            task={**task,'observation':self._agent_observation(role,task,task['observation'])}
        if role=='point' and task.get('waypoint_task'):
            task={**task,'role_instruction':'Front-only observation reference. No wrist point is required.'}
        return super()._loop(role,task,parent_id)

    def _prompt(self,role):
        prompts={
        'prime':('You are Prime coordinating Pointer, Selector, Refiner and Place. '
            'FIRST delegate_point to identify the pick target using current front/wrist. Then delegate_grasp to Selector. '
            'Do not schedule an observation move before candidate evaluation requests it. '
            'Selector returns decision accepted: validate_grasp then execute_grasp; needs_refinement: delegate_refiner '
            'on that candidate, then validate the returned new choice; needs_observation: delegate_waypoint for a FRONT '
            'reference, optionally delegate_view_refiner, validate_view and execute_view. failed: try another candidate or stop. '
            'A front reference is a visible measured surface, not a free-space destination. Backend computes short viewing '
            'position and direction and returns a preview before validation. Review it before approving validation. '
            'After movement inspect updated wrist and target_update. If tracking_status retained, use its point_ref '
            'without another Pointer call and delegate_grasp again. If evidence remains insufficient, Selector can request '
            'another short move via continue_view(target_ref,direction); use another Pointer only if tracking failed or '
            'a different reference is needed. Up to four moves total, stop after two with no progress or a blocked route. '
            'Do not move when candidates are adequate. No forced number of views or blanket top-down grasp rule. '
            'After grasp inspect fresh RGB for holding; if empty/uncertain stop or explicitly reselect for another attempt. '
            'When held, delegate_destination using the NEW observation, then save_destination on its approved point_ref. '
            'Never select destination before grasp. delegate_place uses this destination, hold_assessment held and '
            'destination_assessment="unchanged" exactly (relative to that fresh selection). Use those enum literals, not explanatory sentences. Place calls actual AnyPlace; propagate '
            'inference failures, never fall back to legacy placement. Validate_place the selected candidate, then '
            'execute_place_candidate. Inspect new RGB. Finish {status:completed|failed|unknown}; completion is unverified.'),
        'point':('You are Pointer. Your only geometry input is RGB and returned point/mask overlays, not point clouds. '
            'For pick or destination selection, inspect BOTH current front/wrist images. Click the target in either visible '
            'view with select_region(observation_id,view_id,u,v), normalized [0,1000]. Backend handles segmentation, '
            'depth, projection into the other camera, consistency and fusion; manual stereo correspondences are not required. '
            'Review returned overlays, correct a wrong region with another click, then finish exactly {point_ref}. '
            'For waypoint_task use only FRONT: view_waypoint(observation_id,u,v,direction), direction up|left|right|front|rear. '
            'Click a visible reference surface near the target. Backend computes depth, clearance, short translation and '
            'wrist orientation. Inspect the returned proposed pose; finish {waypoint_ref} to approve or correct the click. '
            'You cannot move. Failure is {status:failed,reason:concrete problem}.'),
        'grasp':('You are Selector. Backend GraspGen candidates and current target evidence are supplied. '
            'Compare enclosures, centering, approach, fingers/palm and observed surface coverage. Use inspect_candidate '
            'or preview_candidate to orbit/zoom a promising candidate. Always finish with an EXPLICIT decision: '
            '{decision:accepted,candidate_ref,reason} for an inspected suitable pose; '
            '{decision:needs_refinement,candidate_ref,reason} for an inspected pose requiring a small position/angle '
            'correction, describing local direction/axis; {decision:needs_observation,reason} when visibility/shape '
            'is inadequate, including a missing wrist target or empty pool due to insufficient observations; '
            '{decision:failed,reason} when more observation/refinement would not help. '
            'Do not repeatedly preview a nearly-right pose instead of requesting its needed adjustment. '
            'Do not request motion just because only one view contributed if measured shape supports a good candidate.'),
        'refiner':('You are Refiner. Only the delegated grasp or viewing pose may be corrected. '
            'Inspect its current measured-scene preview. For grasp use adjust_grasp(candidate_ref,dx_mm,dy_mm,dz_mm, '
            'roll_deg,pitch_deg,yaw_deg). For a viewing pose use refine_view with waypoint_ref and the same deltas. '
            'Translations follow local pose X/Y/Z (10mm per step, 30mm total per axis); rotations follow local axes '
            '(10deg per step, 30deg total). Preview orbit/zoom changes only the camera. A correction returns a NEW '
            'immutable ref and preview. Inspect the result; adjust again within limits or return the original if best. '
            'Finish {candidate_ref,reason} for grasp or {waypoint_ref,reason} for view, or {status:failed,reason}. '
            'Backend rechecks a grasp route; Prime must validate any returned choice before execution.'),
        'place':self.place_agent.prompt.replace('Photos were captured BEFORE pickup','Photos were captured AFTER pickup')}
        return COMMON+prompts[role]

    def _finish(self,role,args,scope):
        if role=='grasp':
            decision=args.get('decision')
            if decision in ('needs_observation','failed') and set(args)=={'decision','reason'}:
                return {**args,'reason':_text(args['reason'])}
            if decision not in ('accepted','needs_refinement') or set(args)!={'decision','candidate_ref','reason'}:
                raise BoundaryError('Selector must return an explicit decision and reason')
            if scope['inspected_candidates'].get(args['candidate_ref'])!=self._epoch:
                raise BoundaryError('inspect this current candidate before deciding')
            return {**args,'reason':_text(args['reason'])}
        if role=='refiner':
            if set(args)=={'status','reason'} and args['status']=='failed':return {**args,'reason':_text(args['reason'])}
            key='waypoint_ref' if 'waypoint_ref' in args else 'candidate_ref'
            records=scope['waypoints'] if key=='waypoint_ref' else scope['inspected_candidates']
            if set(args)!={key,'reason'} or records.get(args[key])!=self._epoch:
                raise BoundaryError('return the inspected delegated/refined reference and reason')
            return {**args,'reason':_text(args['reason'])}
        if role=='point':
            if set(args)=={'status','reason'} and args['status']=='failed':return {**args,'reason':_text(args['reason'])}
            if set(args)=={'waypoint_ref'} and scope['waypoints'].get(args['waypoint_ref'])==self._epoch:
                self._waypoint_refs[args['waypoint_ref']]=self._epoch;return args
            if set(args)!={'point_ref'} or scope['points'].get(args['point_ref'])!=self._epoch:
                raise BoundaryError('return an approved current point overlay reference')
            self._point_refs[args['point_ref']]=self._epoch;return args
        if role=='place':return self.place_agent.finish(args,scope,self._epoch)
        if args.get('status')=='completed':
            execution=getattr(self.backend,'placement_execution',None)
            obs=scope.get('latest_observation')
            if not execution or execution.get('status')!='succeeded' or not obs or scope['observations'][obs][0]!=self._epoch:
                raise BoundaryError('completed requires actual placement execution followed by new observation')
        return super()._finish(role,args,scope)

    def _delegate(self,role,task,scope,sid):
        if self._delegations>=self.budgets.max_delegations:raise BudgetExceeded()
        self._delegations+=1
        task={**task,'observation':deepcopy(self._current_observation)}
        status,result=self._loop(role,task,sid)
        if result.get('candidate_ref'):scope['candidates'][result['candidate_ref']]=self._epoch
        if result.get('waypoint_ref'):self._waypoint_refs[result['waypoint_ref']]=self._epoch
        return {'status':status,'result':result}

    def _dispatch(self,role,tool,args,scope,task,sid):
        try:
            return self._dispatch_checked(role,tool,args,scope,task,sid)
        except ValueError as exc:
            if isinstance(exc, BoundaryError):raise
            self._event('backend_error',sid,role=role,tool=tool,reason='invalid_or_unavailable_geometry')
            return {'error':'geometry_operation_rejected','recovery':'Correct the visible selection or bounded adjustment, choose another candidate, or finish failed.'}

    def _backend_failure(self,sid,role,tool,exc):
        from src.utils.logging_utils import append_json
        from pathlib import Path
        import traceback,time
        directory=getattr(self.backend,'output_dir',None)
        if directory is not None:
            append_json(Path(directory)/'private_backend_errors.jsonl',dict(time_unix_s=time.time(),session_id=sid,
                role=role,tool=tool,exception_type=type(exc).__name__,exception=str(exc),traceback=traceback.format_exc()))

    def _dispatch_checked(self,role,tool,args,scope,task,sid):
        if role=='place':return self.place_agent.dispatch(self,tool,args,scope,task,sid)
        if tool=='delegate_point':
            output=self._delegate('point',args,scope,sid)
            if output['result'].get('point_ref'):self.current_point=output['result']['point_ref']
            return output
        if tool=='select_region':
            obs=scope['observations'].get(args['observation_id'])
            if role!='point' or not obs or obs[0]!=self._epoch or args['view_id'] not in obs[1]:
                raise BoundaryError('use this Pointer session current observation')
            try:output=self.backend.select_region(**args,purpose='place' if task.get('destination_task') else 'pick')
            except Exception as exc:
                self._backend_failure(sid,role,tool,exc)
                self._event('backend_error',sid,role=role,tool=tool)
                return {'error':'region_worker_timeout' if isinstance(exc,TimeoutError) else 'region_selection_failed',
                    'recovery':'Model startup/inference timed out; this is not evidence of a wrong point.' if isinstance(exc,TimeoutError) else 'correct visible point or finish failed'}
            scope['points'][output['point_ref']]=self._epoch
            if not task.get('destination_task'):self.target_ref=output['target_ref']
            self._target_evidence[output['point_ref']]=deepcopy(output)
            return output
        if tool=='delegate_grasp':
            ref=args['point_ref']
            if self._point_refs.get(ref)!=self._epoch:raise BoundaryError('use current approved/tracked target')
            if ref not in self._candidate_bundles:
                try:self._candidate_bundles[ref]=public_result('grasp_candidates',self.backend.grasp_candidates(point_ref=ref))
                except Exception:
                    self._event('backend_error',sid,role=role,tool='grasp_candidates')
                    return {'error':'candidate_generation_failed'}
            bundle=deepcopy(self._candidate_bundles[ref])
            bundle['candidates']=[c for c in bundle['candidates'] if c['candidate_ref'] not in self._rejected_candidates]
            output=self._delegate('grasp',{**args,'candidate_bundle':bundle,
                'target_evidence':self._target_evidence.get(ref,{}),'remaining_moves':4-self.view_requests},scope,sid)
            decision=output['result'].get('decision')
            self.observation_requested=decision=='needs_observation'
            if decision=='needs_refinement':self.refinement_choices.add(output['result']['candidate_ref'])
            scope.setdefault('grasp_decisions',{})[output['result'].get('candidate_ref')]=decision
            return output
        if tool in ('delegate_refiner','delegate_view_refiner'):
            key='candidate_ref' if tool=='delegate_refiner' else 'waypoint_ref';ref=args[key]
            owned=scope['candidates'] if key=='candidate_ref' else self._waypoint_refs
            if owned.get(ref)!=self._epoch:raise BoundaryError('refine a current reviewed pose')
            output=self._delegate('refiner',{**args,'refinement_task':True},scope,sid)
            if output['result'].get('candidate_ref'):
                scope.setdefault('grasp_decisions',{})[output['result']['candidate_ref']]='accepted'
            return output
        if tool in ('delegate_waypoint','continue_view'):
            if not self.observation_requested:raise BoundaryError('Selector must first request additional observation')
            if self.view_requests>=4 or self.no_progress>=2:raise BoundaryError('view budget/no-progress stop reached')
            if tool=='delegate_waypoint':
                return self._delegate('point',{**args,'waypoint_task':True,'target_ref':self.target_ref},scope,sid)
            if args['target_ref']!=self.target_ref:raise BoundaryError('retain the current target identity')
            output=self.backend.continue_view(**args);self._waypoint_refs[output['waypoint_ref']]=self._epoch
            return output
        if tool=='view_waypoint':
            if not task.get('waypoint_task') or args['observation_id']!=scope['latest_observation']:
                raise BoundaryError('front waypoint requires current delegated observation')
            try:output=self.backend.view_waypoint(**args,target_ref=task['target_ref'])
            except Exception:return {'error':'view_reference_unavailable','recovery':'correct the front reference or finish failed'}
            scope['waypoints'][output['waypoint_ref']]=self._epoch;return output
        if tool in ('preview_view','refine_view'):
            if role!='refiner' or scope['waypoints'].get(args['waypoint_ref'])!=self._epoch:raise BoundaryError('view pose outside Refiner session')
            output=getattr(self.backend,tool)(**args);scope['waypoints'][output['waypoint_ref']]=self._epoch;return output
        if tool in ('validate_view','execute_view'):
            if self._waypoint_refs.get(args['waypoint_ref'])!=self._epoch:raise BoundaryError('view pose not approved by Pointer/Refiner')
            if tool=='validate_view':
                output=self.backend.validate_view(**args)
                if output['accepted']:scope.setdefault('view_validations',{})[output['validation_ref']]=args['waypoint_ref']
                else:self.no_progress+=1
                return output
            if self.view_requests>=4 or self.no_progress>=2:raise BoundaryError('view limit reached')
            if scope.get('view_validations',{}).pop(args['validation_ref'],None)!=args['waypoint_ref']:
                raise BoundaryError('matching accepted view validation required')
            self.view_requests+=1
            try:output=self.backend.execute_view(**args)
            finally:self._epoch=self.backend.epoch;self._current_observation=None
            self._remember_observation(output,scope,invalidate=True)
            update=output.get('target_update',{})
            if update.get('tracking_status')=='retained':
                self.current_point=update['point_ref'];self._point_refs[self.current_point]=self._epoch
                self._target_evidence[self.current_point]=update
            self.no_progress=self.no_progress+1 if output['translation_m']<.01 or output['view_status']!='achieved' else 0
            self.observation_requested=False
            return output
        if tool=='adjust_grasp':
            if scope['candidates'].get(args['candidate_ref'])!=self._epoch:raise BoundaryError('candidate outside Refiner session')
            output=self.backend.adjust_grasp(**args)
            if not output.get('accepted',True):return output
            scope['candidates'][output['candidate_ref']]=self._epoch
            scope['inspected_candidates'][output['candidate_ref']]=self._epoch;return output
        if tool=='validate_grasp' and scope.get('grasp_decisions',{}).get(args['candidate_ref'])!='accepted':
            raise BoundaryError('Selector requested refinement; review Refiner result before validation')
        if tool=='delegate_destination':
            if not self.backend.grasp_attempted or self.backend.held_plan is None:raise BoundaryError('pick first, then select destination in fresh RGB')
            return self._delegate('point',{**args,'destination_task':True},scope,sid)
        if tool=='save_destination':
            if self._point_refs.get(args['point_ref'])!=self._epoch:raise BoundaryError('use Pointer approved current destination')
            output=self.backend.save_destination(**args);self.destinations.add(output['destination_ref']);return output
        if tool=='delegate_place':
            if args['destination_ref'] not in self.destinations:raise BoundaryError('use saved post-grasp destination')
            if args['hold_assessment']!='held' or args['destination_assessment']!='unchanged':
                raise BoundaryError('Use exactly hold_assessment="held" and destination_assessment="unchanged" when current RGB supports them; otherwise stop. Do not send explanatory sentences as enum values.')
            return self._delegate('place',args,scope,sid)
        if tool in ('validate_place','execute_place_candidate'):
            ref=args['candidate_ref']
            if scope['candidates'].get(ref)!=self._epoch:raise BoundaryError('use Place returned current candidate')
            if tool=='execute_place_candidate' and scope['validations'].pop(args['validation_ref'],None)!=ref:
                raise BoundaryError('accepted matching placement validation required')
            try:output=public_result(tool,getattr(self.backend,tool)(**args))
            except Exception:
                self._event('backend_error',sid,role=role,tool=tool);return {'error':'anyplace_operation_failed','fallback':False}
            finally:
                if tool=='execute_place_candidate':self._epoch=self.backend.epoch;self._current_observation=None
            if tool=='validate_place' and output['accepted']:scope['validations'][output['validation_ref']]=ref
            if tool=='execute_place_candidate':output['observation']=self._environment_observation(scope,sid,refresh=True)
            return output
        output=super()._dispatch(role,tool,args,scope,task,sid)
        if tool=='execute_grasp':
            self._epoch=self.backend.epoch
            output['observation']=self._environment_observation(scope,sid,refresh=True)
        return output
