"""Opt-in native RoboLab initial-pose randomization and private provenance."""
import json
import math
from pathlib import Path


def validate_xy_range(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('initial-pose XY range must be positive and finite')
    return value


def configure_initial_pose(cfg, xy_range_m):
    """Use native task asset order and the same reset policy as CapX."""
    from isaaclab.assets import RigidObjectCfg
    from robolab.core.events.reset_pose import RandomizeInitPoseUniform

    radius = validate_xy_range(xy_range_m)
    source = cfg.scene.scene.spawn.usd_path
    # The sampler consumes RNG draws in this order. Sorting the names changes
    # which object receives each draw, even for the same task and seed.
    objects = [name for name in cfg.contact_object_list
               if isinstance(getattr(cfg.scene, name, None), RigidObjectCfg)
               and name not in ('robot', 'table')]
    if not objects:
        raise ValueError('initial-pose randomization found no dynamic task objects')
    ranges = {'x': (-radius, radius), 'y': (-radius, radius), 'z': (0., 0.)}
    if getattr(cfg.events, 'randomize_init_pose', None) is not None:
        raise ValueError('task already defines randomize_init_pose; do not silently replace it')
    existing = [name for name in dir(cfg.events) if not name.startswith('_')
                and hasattr(getattr(cfg.events, name), 'mode')]
    if any(getattr(getattr(cfg.events, name).func, '__name__', '') != 'reset_scene_to_default'
           for name in existing):
        raise ValueError(f'Refuse to replace custom task events: {existing}')
    native = RandomizeInitPoseUniform.from_params(
        objects=objects, pose_range=ranges, velocity_range={}, collision_margin=.01,
        max_retries=100)
    # The native randomizer also resets unselected assets to their defaults.
    # Use it as the single reset event, matching the comparison baseline.
    cfg.events = native
    return dict(enabled=True, implementation='RandomizeInitPoseUniform.from_params',
                event_function='robolab.core.events.reset_pose:reset_pose_uniform',
                objects=objects, object_order='native contact_object_list',
                pose_range=ranges, velocity_range={},
                collision_margin_m=.01, max_retries=100,
                scope='independent object XY offsets; Z and orientation unchanged',
                scene_source=str(source))


def record_initial_pose(env, seed):
    """Keep simulator state in native artifacts, never in agent observations."""
    config = getattr(env, 'initial_pose_randomization', None)
    if not config or not env.output_dir:
        return
    objects = {}
    for name in config['objects']:
        asset = env._env.scene[name]
        objects[name] = dict(
            default_pose_wxyz=env._numpy(asset.data.default_root_state[0, :7]).tolist(),
            actual_pose_wxyz=env._numpy(asset.data.root_state_w[0, :7]).tolist())
    record = dict(seed=seed, configuration=config, objects=objects)
    directory = Path(env.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'initial_pose_randomization.jsonl').open('a') as stream:
        stream.write(json.dumps(record)+'\n')
