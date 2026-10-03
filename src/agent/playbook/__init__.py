"""Role-scoped policy files and the versions shared by CLI and web UI."""
from pathlib import Path

from .loader import DecisionPlaybook

POLICY_PATHS = {str(version): Path(__file__).with_name('policies') / f'v{version}.md'
                for version in range(4)}
VERSION_CHOICES = tuple(f'v{version}' for version in POLICY_PATHS)
DEFAULT_VERSION = '3'
DEFAULT_PATH = POLICY_PATHS[DEFAULT_VERSION]

__all__ = ['DecisionPlaybook', 'POLICY_PATHS', 'VERSION_CHOICES',
           'DEFAULT_PATH', 'DEFAULT_VERSION']
