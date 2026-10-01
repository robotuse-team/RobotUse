import pytest

from src.backend.controller import BoundaryError, Budgets
from src.llm.manager import ImageRegistry
from src.backend.delegation import DelegationOrchestrator
from src.tools.grasp.arguments import property_schema, validate_argument
from test_prime_delegation import Factory, FakeBackend, a


def height(value=.2, mode='absolute', reference='none'):
    refs = dict(none='absolute', clicked='clicked_point', median='segment_median',
                mean='segment_mean', min='observed_min', max='observed_max')
    return dict(value_m=value, reference=refs.get(reference, reference) if mode == 'absolute' or reference != 'none' else 'none')


def request():
    return dict(point_ref='p1', direction='custom', tolerance_deg=25., azimuth_deg=45., polar_deg=70.,
                geometric_height=None,
                transit=dict(pre=height(.4), post=height(.55)))


def pointer():
    return [a('select_region', observation_id='o0', view_id='front', u=500, v=500),
            a('finish', point_ref='p1')]


class Backend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.epoch = 0; self.held_plan = None; self.grasp_attempted = False
        self.latest_fused_ref = 'p1'; self.multiview = True; self.captures = 0
        self.place_predictions = {}; self.place_diagnostics = {}
        self._group = lambda ref: ('grasp', self.epoch, 'p1')

    def observe(self):
        self.captures += 1
        return dict(observation_id=f'o{self.epoch}', views=[
            dict(view_id='front', image_ref='front_rgb'), dict(view_id='wrist', image_ref='wrist_rgb')])

    def select_region(self, observation_id, **args):
        return dict(point_ref='p1', target_ref='target', observation_id=observation_id, image_refs=['mask'])

    def inspect_candidate(self, candidate_ref):
        return dict(candidate_ref=candidate_ref, image_refs=['mesh', 'front_rgb', 'wrist_rgb'])

    def execute_grasp(self, **args):
        self.calls.append(('execute_grasp', args)); self.epoch += 1
        self.held_plan = object(); self.grasp_attempted = True
        return dict(execution_ref='grasped', status='succeeded')

    def explicit_geometry(self, **args):
        self.calls.append(('explicit_geometry', args))
        return dict(point_ref=args['point_ref'], frame='connector_base', mean_xyz_m=[.4, -.2, .15],
                    median_xyz_m=[.38, -.22, .13])

    def explicit_grasp_candidates(self, **args):
        self.calls.append(('explicit_grasp_candidates', args))
        return dict(candidates=[dict(candidate_ref='c1', image_refs=['preview'], source='median')],
                    diagnostic_candidates=[], geometry=self.explicit_geometry(point_ref=args['point_ref']))

    def explicit_place_candidates(self, **args):
        self.calls.append(('explicit_place_candidates', args))
        return dict(xy_candidates=[dict(candidate_id='median', xy_m=[.4, -.2], reference_z_m=.15)])

    def explicit_prepare_place(self, **args):
        self.calls.append(('explicit_prepare_place', args))
        return dict(candidate_ref='pc', image_refs=['place_preview'], accepted=True)

    def explicit_inspect_place(self, **args):
        self.calls.append(('explicit_inspect_place', args))
        return dict(candidate_ref=args['candidate_ref'], image_refs=['place_preview'])

    def explicit_adjust_place(self, **args):
        self.calls.append(('explicit_adjust_place', args))
        return dict(candidate_ref='pc_adjusted', accepted=True, image_refs=['adjusted_place_preview'])

    def explicit_execute_place(self, **args):
        self.calls.append(('explicit_execute_place', args)); self.epoch += 1
        return dict(execution_ref='placed_holding', status='succeeded', release_commanded=False)

    def move_vertical(self, **args):
        self.calls.append(('move_vertical', args)); self.epoch += 1
        return dict(execution_ref='vertical', status='succeeded')

    def goto_home_joint_position(self):
        self.calls.append(('goto_home_joint_position', {})); self.epoch += 1
        return dict(execution_ref='home', status='succeeded')

    def release(self):
        self.calls.append(('release', {}))
        self.epoch += 1; self.held_plan = None
        return dict(execution_ref='released', status='succeeded')


def grasp_actions():
    return [a('explicit_grasp_candidates', **request()),
            a('inspect_candidate', candidate_ref='c1'), a('finish', status='success', candidate_ref='c1')]


def runner_for_grasp(refiner_actions, *, auto_refine_routes=True):
    backend = Backend()
    factory = Factory(prime=[[a('delegate_point', instruction='object'),
        a('delegate_grasp', instruction='pick object', point_ref='p1'), a('finish', status='unknown')]],
        point=[pointer()], grasp=[grasp_actions()], refiner=[refiner_actions])
    runner = DelegationOrchestrator(backend, factory, auto_refine_routes=auto_refine_routes)
    runner._overhead_done = True
    return runner, backend, factory


def test_agent_parameters_reach_backend_and_selected_grasp_always_reaches_refiner():
    runner, backend, factory = runner_for_grasp([
        a('inspect_candidate', candidate_ref='c1'), a('finish', status='success', candidate_ref='c1')])
    result = runner.run('pick the object')
    assert not [event for event in result.events if event['kind'] in ('action_rejected', 'backend_error')]
    assert next(args for tool, args in backend.calls if tool == 'explicit_grasp_candidates') == request()
    assert [role for role, _, _ in factory.sessions] == ['prime', 'point', 'grasp', 'refiner']
    assert any(tool == 'execute_grasp' for tool, _ in backend.calls)
    grasp = next(session for role, _, session in factory.sessions if role == 'grasp')
    assert grasp.inputs[0][0][1]['content']['measured_geometry']['median_xyz_m'] == [.38, -.22, .13]
    assert 'explicit_geometry' not in runner._tools('grasp', {})
    assert not any(event.get('tool') == 'explicit_geometry' for event in result.events)
    review = next(session for role, _, session in factory.sessions if role == 'refiner')
    assert review.inputs[0][0][1]['content']['prior_candidate_inspection']['image_refs']
    context = review.inputs[0][0][1]['content']['selection_context']
    assert context['measured_geometry']['median_xyz_m'] == [.38, -.22, .13]
    assert context['candidate_geometry']['source'] == 'median'
    assert runner._current_observation['observation_id'] == 'o1'
    serialized_prompts = ' '.join(session.inputs[0][0][0]['content'] for _, _, session in factory.sessions)
    for removed in ('AnyPlace', 'MoveIt', 'GraspGen'):
        assert removed not in serialized_prompts


@pytest.mark.parametrize('decision', ['failed', 'needs_observation', 'model_error'])
def test_missing_or_declining_refiner_never_executes_original_grasp(decision):
    actions = [] if decision == 'model_error' else [a('finish', status=decision, reason='insufficient evidence')]
    runner, backend, _ = runner_for_grasp(actions)
    result = runner.run('pick the object')
    assert not any(tool in ('validate_grasp', 'execute_grasp') for tool, _ in backend.calls)
    output = next(event['result'] for event in result.events
                  if event['kind'] == 'tool_result' and event.get('tool') == 'delegate_grasp')
    assert output['executed'] is False


def test_observe_first_gate_precedes_geometry_or_model_generation():
    runner, backend, factory = runner_for_grasp([])
    runner.observe_before_grasp = True
    runner._overhead_done = False
    result = runner.run('pick the object')
    output = next(event['result'] for event in result.events
                  if event['kind'] == 'tool_result' and event.get('tool') == 'delegate_grasp')
    assert output['reason_code'] == 'observe_from_above_first'
    assert not any(role == 'grasp' for role, _, _ in factory.sessions)
    assert not any(tool == 'explicit_grasp_candidates' for tool, _ in backend.calls)


def test_grasp_feedback_reaches_prime_and_next_child_across_new_pointer_session():
    backend = Backend()
    generations = []
    def candidates(**args):
        source, yaw = ('median', 90.) if not generations else ('mean', -45.)
        generations.append(source)
        return dict(candidates=[dict(candidate_ref='c1', source=source, yaw_deg=yaw,
            contact_center_xyz_m=[.4, -.2, .1], transit={'post_pick_z_m': .35},
            image_refs=['preview'], hidden_object_pose='SECRET')])
    backend.explicit_grasp_candidates = candidates
    def execute(**args):
        backend.epoch += 1
        return dict(execution_ref='attempt', status='failed' if backend.epoch == 1 else 'succeeded',
                    reason_code='motion_failed' if backend.epoch == 1 else 'completed', private='SECRET')
    backend.execute_grasp = execute
    factory = Factory(prime=[[a('delegate_point', instruction='cube'),
        a('delegate_grasp', instruction='pick cube', point_ref='p1'),
        a('delegate_point', instruction='cube again'),
        a('delegate_grasp', instruction='Previous motion failed; reconsider the cube grasp', point_ref='p1'),
        a('finish', status='unknown')]], point=[pointer(), [
            a('select_region', observation_id='o1', view_id='front', u=500, v=500),
            a('finish', point_ref='p1')]], grasp=[grasp_actions(), grasp_actions()])
    result = DelegationOrchestrator(backend, factory).run('pick cube')
    assert not [e for e in result.events if e['kind'] in ('action_rejected', 'backend_error')]
    prime = next(s for role, _, s in factory.sessions if role == 'prime')
    feedbacks = [m['content']['grasp_feedback'] for m in prime.inputs[-1][0]
                 if m.get('role') == 'tool' and m.get('tool') == 'delegate_grasp']
    first, second = feedbacks
    assert first['selection']['candidate']['source'] == 'median'
    assert first['selection']['candidate']['yaw_deg'] == 90.
    assert first['tool_feedback']['status'] == 'failed'
    assert second['selection']['candidate']['source'] == 'mean'
    assert second['tool_feedback']['status'] == 'succeeded'
    assert 'SECRET' not in repr(feedbacks)
    assert 'image_refs' not in repr(feedbacks)
    child = [s for role, _, s in factory.sessions if role == 'grasp'][1]
    previous = child.inputs[0][0][1]['content']['previous_grasp_feedback']
    assert previous == first
    assert previous['reference_only'] is True


def test_validation_failure_returns_selected_candidate_feedback_without_execution():
    runner, backend, _ = runner_for_grasp([], auto_refine_routes=False)
    backend.accepted = False
    result = runner.run('pick cube')
    output = next(e['result'] for e in result.events
                  if e['kind'] == 'tool_result' and e.get('tool') == 'delegate_grasp')
    feedback = output['grasp_feedback']
    assert feedback['selection']['candidate']['source'] == 'median'
    assert feedback['stage'] == 'validation'
    assert feedback['tool_feedback']['accepted'] is False
    assert feedback['executed'] is False
    assert not any(tool == 'execute_grasp' for tool, _ in backend.calls)


def test_all_six_candidates_remain_inspectable_without_legacy_refinement_tool():
    backend = Backend()
    def candidates(**args):
        return dict(candidates=[dict(candidate_ref=f'c{i}', image_refs=[f'preview{i}']) for i in range(6)])
    backend.explicit_grasp_candidates = candidates
    factory = Factory(prime=[[a('delegate_point', instruction='object'),
        a('delegate_grasp', instruction='pick', point_ref='p1'), a('finish', status='unknown')]],
        point=[pointer()], grasp=[[a('explicit_grasp_candidates', **request()),
            *[a('inspect_candidate', candidate_ref=f'c{i}') for i in range(6)],
            a('finish', status='success', candidate_ref='c5')]],
        refiner=[[a('inspect_candidate', candidate_ref='c5'), a('finish', candidate_ref='c5')]])
    runner = DelegationOrchestrator(backend, factory, budgets=Budgets(child_steps=12))
    runner._overhead_done = True
    result = runner.run('pick')
    assert not [event for event in result.events if event['kind'] in ('action_rejected', 'backend_error')]
    assert next(args['candidate_ref'] for tool, args in backend.calls if tool == 'execute_grasp') == 'c5'
    grasp = next(session for role, _, session in factory.sessions if role == 'grasp')
    assert all('refine_candidate' not in tools for _, tools in grasp.inputs)


def test_diagnostic_can_be_reviewed_but_never_selected_for_execution_without_repair():
    backend = Backend()
    diagnostic = dict(candidate_ref='bad', image_refs=['diagnostic'], source_view='front',
                      executable=False, reason_code='path_rejected', validation_feedback={})
    backend.explicit_grasp_candidates = lambda **args: dict(candidates=[], diagnostic_candidates=[diagnostic])
    factory = Factory(prime=[[a('delegate_point', instruction='object'),
        a('delegate_grasp', instruction='pick', point_ref='p1'), a('finish', status='unknown')]],
        point=[pointer()], grasp=[[a('explicit_grasp_candidates', **request()),
            a('inspect_candidate', candidate_ref='bad'),
            a('finish', status='success', candidate_ref='bad'),
            a('finish', status='needs_refinement', candidate_ref='bad')]],
        refiner=[[a('inspect_candidate', candidate_ref='bad'),
                  a('finish', status='failed', reason='Cannot repair path')]])
    runner = DelegationOrchestrator(backend, factory); runner._overhead_done = True
    result = runner.run('pick')
    rejections = [event for event in result.events if event['kind'] == 'action_rejected']
    assert len(rejections) == 1
    assert not any(tool in ('validate_grasp', 'execute_grasp') for tool, _ in backend.calls)


@pytest.mark.parametrize('release', [False, True])
def test_place_stops_holding_and_only_later_prime_turn_can_release(release):
    backend = Backend(); backend.held_plan = object(); backend.grasp_attempted = True
    held = backend.held_plan
    prime = [a('delegate_place', instruction='put it in the dish', destination_ref='dest',
               hold_assessment='held', destination_assessment='unchanged')]
    if release:
        prime.append(a('release'))
    prime.append(a('finish', status='unknown'))
    factory = Factory(prime=[prime], place=[[
        a('explicit_place_candidates', destination_ref='dest'),
        a('explicit_prepare_place', destination_ref='dest', xy_source='median', xy_m=[.43, -.24], height=height(.22), transit_height=height(.6)),
        a('inspect_place_candidate', candidate_ref='pc'), a('finish', status='success', candidate_ref='pc')]])
    runner = DelegationOrchestrator(backend, factory); runner.destinations.add('dest')
    result = runner.run('place object')
    assert not [event for event in result.events if event['kind'] in ('action_rejected', 'backend_error')]
    assert sum(tool == 'explicit_execute_place' for tool, _ in backend.calls) == 1
    assert sum(tool == 'release' for tool, _ in backend.calls) == int(release)
    assert (backend.held_plan is None) if release else (backend.held_plan is held)
    session = next(session for role, _, session in factory.sessions if role == 'prime')
    placement_result = next(message['content'] for message in session.inputs[1][0]
                            if message.get('role') == 'tool' and message.get('tool') == 'delegate_place')
    assert placement_result['release_requires_agent_decision'] is True
    assert placement_result['observation']['observation_id'] == 'o1'
    assert placement_result['result']['release_commanded'] is False


def test_initial_cyan_options_are_visible_but_execution_needs_explicit_height_and_new_preview(tmp_path):
    backend = Backend(); backend.held_plan = object()
    initial_refs = [f'{source}_{view}' for source in ('clicked', 'median', 'mean')
                    for view in ('front', 'wrist')]
    options = dict(destination_ref='dest', image_refs=initial_refs,
        xy_candidates=[dict(candidate_id=source, xy_m=[.4, -.2], reference_z_m=z)
                       for source, z in (('clicked', .11), ('median', .12), ('mean', .13))],
        candidate_previews=[dict(xy_source=source, reference_contact_center_xyz_m=[.4, -.2, z],
            preview_contact_center_xyz_m=[.4, -.2, z + .05], preview_z_offset_m=.05,
            height_status='reference_surface_plus_offset_not_execution_height', executable=False,
            image_refs=initial_refs[2*i:2*i+2])
            for i, (source, z) in enumerate((('clicked', .11), ('median', .12), ('mean', .13)))],
        height_decision_required=True)
    backend.explicit_place_candidates = lambda **args: options
    chosen_height = height(.42)
    def prepare(**args):
        backend.calls.append(('explicit_prepare_place', args))
        return dict(candidate_ref='pc', accepted=True, image_refs=['prepared_front', 'prepared_wrist'],
                    height=args['height'], contact_center_xyz_m=[*args['xy_m'], args['height']['value_m']])
    backend.explicit_prepare_place = prepare
    factory = Factory(prime=[[a('delegate_place', instruction='put in dish', destination_ref='dest',
        hold_assessment='held', destination_assessment='unchanged'), a('finish', status='unknown')]],
        place=[[a('explicit_place_candidates', destination_ref='dest'),
            a('finish', status='success', candidate_ref='median'),  # Reference preview cannot execute.
            a('explicit_prepare_place', destination_ref='dest', xy_source='median', xy_m=[.43, -.24]),
            a('explicit_prepare_place', destination_ref='dest', xy_source='median', xy_m=[.43, -.24], height=chosen_height, transit_height=height(.6)),
            a('finish', status='success', candidate_ref='pc')]])
    registry = ImageRegistry()
    for ref in ['front_rgb', 'wrist_rgb', *initial_refs, 'prepared_front', 'prepared_wrist']:
        registry.paths[ref] = tmp_path / (ref + '.png')
    factory.images = registry
    runner = DelegationOrchestrator(backend, factory); runner.destinations.add('dest')
    result = runner.run('place')
    assert len([event for event in result.events if event['kind'] == 'action_rejected']) == 2
    assert not [event for event in result.events if event['kind'] == 'backend_error']
    place = next(session for role, _, session in factory.sessions if role == 'place')
    initial_messages = place.inputs[1][0]
    presented = next(message['content'] for message in initial_messages
                     if message.get('role') == 'tool' and message.get('tool') == 'explicit_place_candidates')
    assert presented == options and presented['height_decision_required'] is True
    assert registry.current_turn_refs(initial_messages, role='place') == ['front_rgb', 'wrist_rgb', *initial_refs]
    # The next model decision sees the pose at its chosen Z, before selecting it.
    final_messages = place.inputs[-1][0]
    assert registry.current_turn_refs(final_messages, role='place') == [
        'front_rgb', 'wrist_rgb', 'prepared_front', 'prepared_wrist']
    prepared = next(message['content'] for message in final_messages
                    if message.get('role') == 'tool' and message.get('tool') == 'explicit_prepare_place'
                    and message['content'].get('candidate_ref'))
    assert prepared['contact_center_xyz_m'] == [.43, -.24, .42]
    assert [args['height'] for tool, args in backend.calls if tool == 'explicit_prepare_place'] == [chosen_height]
    assert [args for tool, args in backend.calls if tool == 'explicit_execute_place'] == [{'candidate_ref': 'pc'}]
    assert not any(tool == 'release' for tool, _ in backend.calls)


def test_vertical_and_home_preserve_hold_and_refresh_scope():
    backend = Backend(); backend.held_plan = object(); held = backend.held_plan
    factory = Factory(prime=[[a('move_vertical', dz_m=.12), a('goto_home_joint_position'),
                             a('finish', status='unknown')]])
    runner = DelegationOrchestrator(backend, factory)
    runner._point_refs['old'] = 0
    runner.run('raise object then return home')
    assert backend.held_plan is held and not any(tool == 'release' for tool, _ in backend.calls)
    assert runner._epoch == 2 and runner._current_observation['observation_id'] == 'o2'
    assert runner._point_refs == {}
    assert next(args for tool, args in backend.calls if tool == 'move_vertical') == {'dz_m': .12}


def test_place_adjustment_and_inspection_execute_only_selected_pose_revision():
    backend = Backend(); backend.held_plan = object()
    adjustment = dict(candidate_ref='pc', dx_mm=25., dy_mm=-10., dz_mm=15.,
                      roll_deg=5., pitch_deg=-4., yaw_deg=10.)
    factory = Factory(prime=[[a('delegate_place', instruction='orient over dish', destination_ref='dest',
        hold_assessment='held', destination_assessment='unchanged'), a('finish', status='unknown')]],
        place=[[a('explicit_prepare_place', destination_ref='dest', xy_source='median', xy_m=[.4, -.2], height=height(.2), transit_height=height(.6)),
            a('explicit_adjust_place', **adjustment), a('inspect_place_candidate', candidate_ref='pc_adjusted'),
            a('finish', candidate_ref='pc_adjusted')]])
    runner = DelegationOrchestrator(backend, factory); runner.destinations.add('dest')
    result = runner.run('place')
    assert not [event for event in result.events if event['kind'] in ('action_rejected', 'backend_error')]
    assert next(args for tool, args in backend.calls if tool == 'explicit_adjust_place') == adjustment
    assert next(args for tool, args in backend.calls if tool == 'explicit_inspect_place') == {'candidate_ref': 'pc_adjusted'}
    assert next(args for tool, args in backend.calls if tool == 'explicit_execute_place') == {'candidate_ref': 'pc_adjusted'}
    assert not any(tool == 'release' for tool, _ in backend.calls)


def test_rejected_place_adjustment_does_not_grant_new_candidate_or_change_old_selection():
    backend = Backend(); runner = DelegationOrchestrator(backend, Factory())
    backend.explicit_adjust_place = lambda **args: dict(accepted=False, candidate_ref='bad', image_refs=['rejected'])
    scope = {'candidates': {'pc': 0}, 'inspected_candidates': {'pc': 0}}
    output = runner._intent_dispatch('place', 'explicit_adjust_place', {'candidate_ref': 'pc'}, scope, {}, 'sid')
    assert output['accepted'] is False
    assert scope == {'candidates': {'pc': 0}, 'inspected_candidates': {'pc': 0}}


@pytest.mark.parametrize('key,limit', [('dx_mm', 30), ('dy_mm', 30), ('dz_mm', 30),
                                    ('roll_deg', 10), ('pitch_deg', 10), ('yaw_deg', 10)])
def test_place_refinement_runtime_and_provider_share_step_limits(key, limit):
    runner = DelegationOrchestrator(Backend(), Factory())
    schema = property_schema(key, tool='explicit_adjust_place')
    assert schema['minimum'] == -limit and schema['maximum'] == limit
    assert runner._tool_argument('explicit_adjust_place', key, limit) == limit
    with pytest.raises(BoundaryError):
        runner._tool_argument('explicit_adjust_place', key, limit + .001)


@pytest.mark.parametrize('status,decision,expected', [
    ('completed', 'continue', 'continue'), ('completed', 'abort', 'abort'),
    ('completed', None, 'abort'), ('error', 'continue', 'abort'),
    ('budget_exhausted', 'continue', 'abort')])
def test_paused_close_requires_explicit_completed_refiner_approval(monkeypatch, status, decision, expected):
    runner = DelegationOrchestrator(Backend(), Factory())
    runner._inflight_call = ({}, {'instruction': 'pick'}, 'sid', 'grasp')
    monkeypatch.setattr(runner, '_delegate', lambda *a: dict(status=status, result={'status': decision}))
    assert runner._inflight_refinement('pregrasp', {})['status'] == expected
    assert runner._inflight_refinement('release', {})['status'] == 'abort'


def test_paused_budget_boundary_and_missing_session_keep_gripper_open(monkeypatch):
    runner = DelegationOrchestrator(Backend(), Factory())
    assert runner._inflight_refinement('pregrasp', {})['status'] == 'abort'
    runner._inflight_call = ({}, {}, 'sid', 'grasp')
    def reject(*args):
        raise BoundaryError('invalid refiner answer')
    monkeypatch.setattr(runner, '_delegate', reject)
    assert runner._inflight_refinement('pregrasp', {})['status'] == 'abort'
    runner._delegations = runner.budgets.max_delegations
    assert runner._inflight_refinement('pregrasp', {})['status'] == 'abort'


def test_role_and_target_boundaries_block_foreign_geometry_and_motion():
    runner = DelegationOrchestrator(Backend(), Factory()); runner._point_refs['p1'] = 0
    for role, tool, args, task in [
        ('prime', 'explicit_geometry', {'point_ref': 'p1'}, {'point_ref': 'p1'}),
        ('grasp', 'explicit_geometry', {'point_ref': 'p1'}, {'point_ref': 'p1'}),
        ('grasp', 'explicit_geometry', {'point_ref': 'other'}, {'point_ref': 'p1'}),
        ('place', 'explicit_prepare_place', {'destination_ref': 'other'}, {'destination_ref': 'dest'}),
        ('grasp', 'move_vertical', {'dz_m': .1}, {}),
    ]:
        with pytest.raises(BoundaryError):
            runner._intent_dispatch(role, tool, args, {}, task, 'sid')
    runner._epoch = 1
    with pytest.raises(BoundaryError, match='stale'):
        runner._intent_dispatch('grasp', 'explicit_grasp_candidates', request(), {}, {'point_ref': 'p1'}, 'sid')


@pytest.mark.parametrize('key,value', [
    ('geometric_height', height(.2, reference='unknown')),
    ('geometric_height', height(.2, 'surface_relative', 'none')),
    ('geometric_height', {'mode': 'absolute', 'value_m': .2}),
    ('geometric_height', height(float('nan'))),
    ('transit', {'pre': height(.3)}),
    ('transit', {'pre': height(.3), 'post': height(True)}),
    ('direction', 'diagonal'), ('direction', 'horizontal'), ('tolerance_deg', -1), ('polar_deg', 181),
    ('xy_m', [1]), ('xy_m', [True, .2]),
])
def test_nested_agent_decisions_have_strict_runtime_validation(key, value):
    with pytest.raises(BoundaryError):
        validate_argument('explicit_grasp_candidates', key, value)


def test_provider_schema_nested_decisions_and_legacy_preview_remain_distinct():
    schema = property_schema('transit', tool='explicit_grasp_candidates')
    assert schema['required'] == ['pre', 'post']
    assert schema['properties']['pre']['required'] == ['reference', 'value_m']
    assert schema['additionalProperties'] is False
    assert property_schema('direction', tool='explicit_grasp_candidates')['enum'] == ['vertical', 'custom', 'mean', 'median']
    assert property_schema('direction', tool='view_waypoint') is None
    assert property_schema('azimuth_deg', tool='preview_candidate') is None
    runner = DelegationOrchestrator(Backend(), Factory())
    assert runner._tool_argument('explicit_grasp_candidates', 'azimuth_deg', 270.) == 270.
    with pytest.raises(BoundaryError):
        runner._tool_argument('preview_candidate', 'azimuth_deg', 270.)
    assert runner._optional_arguments('delegate_grasp') == ()
    assert validate_argument('explicit_grasp_candidates', 'transit', request()['transit']) == request()['transit']


@pytest.mark.parametrize('direction,azimuth,polar', [
    ('vertical', 0., None), ('horizontal', None, 90.), ('custom', None, 40.), ('custom', 20., None)])
def test_direction_requires_angles_exactly_for_custom_requests(direction, azimuth, polar):
    runner = DelegationOrchestrator(Backend(), Factory()); runner._point_refs['p1'] = 0
    args = dict(request(), direction=direction, azimuth_deg=azimuth, polar_deg=polar)
    with pytest.raises(BoundaryError):
        runner._intent_dispatch('grasp', 'explicit_grasp_candidates', args,
                                {'candidates': {}}, {'point_ref': 'p1'}, 'sid')


def test_horizontal_request_rejected_before_backend_call():
    backend = Backend()
    runner = DelegationOrchestrator(backend, Factory())
    runner._point_refs['p1'] = 0
    with pytest.raises(BoundaryError, match='direction must be one of'):
        runner._intent_dispatch('grasp', 'explicit_grasp_candidates',
            dict(request(), direction='horizontal'),
            {'candidates': {}}, {'point_ref': 'p1'}, 'sid')
    assert not backend.calls


@pytest.mark.parametrize('direction', ['vertical'])
def test_noncustom_requests_forward_null_angles(direction):
    backend = Backend(); runner = DelegationOrchestrator(backend, Factory()); runner._point_refs['p1'] = 0
    args = dict(request(), direction=direction, azimuth_deg=None, polar_deg=None)
    for key in ('azimuth_deg', 'polar_deg'):
        assert runner._tool_argument('explicit_grasp_candidates', key, None) is None
    output = runner._intent_dispatch('grasp', 'explicit_grasp_candidates', args,
                                     {'candidates': {}}, {'point_ref': 'p1'}, 'sid')
    assert output['candidates']
    assert next(values for tool, values in backend.calls if tool == 'explicit_grasp_candidates') == args


def test_default_grasp_does_not_require_overhead_observation():
    runner, backend, factory = runner_for_grasp([
        a('inspect_candidate', candidate_ref='c1'), a('finish', candidate_ref='c1')])
    runner._overhead_done = False
    assert not runner.observe_before_grasp
    assert 'Overhead observation before grasp is optional' in runner._prompt('prime')
    result = runner.run('pick')
    assert any(role == 'grasp' for role, _, _ in factory.sessions)
    assert not any(event.get('result', {}).get('reason_code') == 'observe_from_above_first'
                   for event in result.events if isinstance(event.get('result'), dict))


def test_auto_refine_disabled_executes_selected_grasp_without_refiner():
    runner, backend, factory = runner_for_grasp([], auto_refine_routes=False)
    runner.run('pick the object')
    assert [role for role, _, _ in factory.sessions] == ['prime', 'point', 'grasp']
    assert any(tool == 'execute_grasp' for tool, _ in backend.calls)
    assert 'Automatic pre-execution Refiner review is disabled' in runner._prompt('grasp')


@pytest.mark.parametrize('reason', ['needs_refinement', 'diagnostic'])
def test_auto_refine_disabled_does_not_execute_unapproved_selection(reason):
    runner, backend, factory = runner_for_grasp([], auto_refine_routes=False)
    original = runner._delegate
    def delegate(role, *args, **kwargs):
        result = original(role, *args, **kwargs)
        if role == 'grasp':
            if reason == 'diagnostic':
                runner._diagnostic_candidates.add(result['result']['candidate_ref'])
            else:
                result['result']['decision'] = 'needs_refinement'
        return result
    runner._delegate = delegate
    runner.run('pick the object')
    assert not any(tool in ('validate_grasp', 'execute_grasp') for tool, _ in backend.calls)
    assert 'refiner' not in [role for role, _, _ in factory.sessions]


def test_destination_pointer_success_is_saved_without_prime_save_turn():
    backend = Backend(); backend.held_plan = object(); backend.grasp_attempted = True
    def save(point_ref):
        backend.calls.append(('save_destination', {'point_ref': point_ref}))
        return dict(destination_ref='dest', point_ref=point_ref)
    backend.save_destination = save
    factory = Factory(prime=[[a('delegate_destination', instruction='bowl'),
                              a('finish', status='unknown')]], point=[pointer()])
    runner = DelegationOrchestrator(backend, factory)
    result = runner.run('place object')
    assert 'save_destination' not in runner._tools('prime', {})
    assert ('save_destination', {'point_ref': 'p1'}) in backend.calls
    assert 'dest' in runner.destinations
    output = next(e['result'] for e in result.events
                  if e['kind'] == 'tool_result' and e.get('tool') == 'delegate_destination')
    assert output['destination_ref'] == output['result']['destination_ref'] == 'dest'
    assert output['destination_saved']
    assert [role for role, _, _ in factory.sessions] == ['prime', 'point']


def test_failed_destination_pointer_does_not_save():
    backend = Backend(); backend.held_plan = object(); backend.grasp_attempted = True
    backend.save_destination = lambda **kw: pytest.fail('failed selection must not save')
    factory = Factory(prime=[[a('delegate_destination', instruction='bowl'), a('finish', status='unknown')]],
                      point=[[a('finish', status='failed', reason='not visible')]])
    runner = DelegationOrchestrator(backend, factory)
    runner.run('place object')
    assert not runner.destinations


@pytest.mark.parametrize('mode', ['mean', 'median'])
def test_center_mode_passes_explicit_height_without_direction_angles(mode):
    backend = Backend(); runner = DelegationOrchestrator(backend, Factory()); runner._point_refs['p1'] = 0
    args = dict(request(), direction=mode, azimuth_deg=None, polar_deg=None,
                tolerance_deg=None, geometric_height=height(.03))
    runner._intent_dispatch('grasp', 'explicit_grasp_candidates', args,
                            {'candidates': {}}, {'point_ref': 'p1'}, 'sid')
    assert ('explicit_grasp_candidates', args) in backend.calls


def test_pick_minimum_lift_is_visible_to_prime_and_grasp():
    runner = DelegationOrchestrator(Backend(), Factory())
    for role in ('prime', 'grasp'):
        assert 'at least 0.20 m above the grasp height' in runner._prompt(role)
        assert 'move_vertical' in runner._prompt(role)
