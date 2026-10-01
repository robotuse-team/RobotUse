"""Initialize the pinned native RoboLab and the unchanged GAP connector."""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
from types import ModuleType

from src.runtime.paths import REPOSITORY_ROOT

ROBOLAB_SOURCE = Path(__file__).parent / 'third_party' / 'robolab'


def initialize():
    """Pin native imports and bridge two names hard-coded by immutable vendor.

    The aliases point to the current RobotUse implementations without modifying
    the external source files.
    """
    if not (ROBOLAB_SOURCE / 'robolab/__init__.py').is_file():
        raise FileNotFoundError('Initialize the RoboLab submodule before running RobotUse')
    paths = (REPOSITORY_ROOT, ROBOLAB_SOURCE,
             REPOSITORY_ROOT / 'src/simulator/robolab/third_party/graph_as_policy',
             REPOSITORY_ROOT / 'src/simulator/robolab/third_party/graph_as_policy/gap-core/src')
    for name, source in (('robolab', ROBOLAB_SOURCE), ('gap', paths[2]), ('gap_core', paths[3])):
        existing = sys.modules.get(name)
        if existing is None:
            continue
        location = getattr(existing, '__file__', None)
        if location is None or not Path(location).resolve().is_relative_to(source.resolve()):
            raise RuntimeError(f'{name} was imported from another checkout; start a fresh RobotUse process')
    for path in reversed(paths):
        # A runtime wrapper may already append this checkout after another
        # installed copy. Presence alone does not establish import priority.
        value = str(path)
        sys.path[:] = [entry for entry in sys.path if entry != value]
        sys.path.insert(0, value)
    os.environ['ROBOLAB_ROOT'] = str(ROBOLAB_SOURCE.resolve())
    # Vendor imports the collision type and reads the CLI ownership flag by
    # these names. Installing aliases changes no vendor source or algorithms.
    package = sys.modules.setdefault('robot_skill_selector', ModuleType('robot_skill_selector'))
    if not hasattr(package, '__path__'):
        package.__path__ = []
    for name in ('collision', 'cli'):
        module = importlib.import_module(f'src.simulator.robolab.{name}')
        alias = 'robolab_' + name
        sys.modules['robot_skill_selector.' + alias] = module
        setattr(package, alias, module)


def get_factory():
    initialize()
    from gap.envs.robolab_env import make_env
    return make_env


def create_connector(**options):
    initialize()
    from gap.connector import sim
    return sim('robolab', **options)


def is_successful_episode_end(connector, error):
    """Recognize the native terminal guard only with confirmed task success."""
    if not isinstance(error, RuntimeError) or str(error) != (
            'RoboLab episode ended; explicit reset required before motion'):
        return False
    try:
        budget = connector.env.simulation_budget()
        return (budget['terminal'] is True
                and budget['reason_code'] == 'episode_terminated'
                and connector.check_success()[0] is True)
    except Exception:
        return False


def run_native_cli(main):
    initialize()
    from .cli import run_native_cli as run
    return run(main)


def __getattr__(name):
    if name == 'make_env':
        return get_factory()
    if name == 'RoboLabEnv':
        initialize()
        from gap.envs.robolab_env import RoboLabEnv
        return RoboLabEnv
    raise AttributeError(name)
