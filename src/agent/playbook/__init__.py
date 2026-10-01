"""Version declarations for role-scoped policy documents."""
from .loader import DecisionPlaybook
from .v3 import PATH as DEFAULT_PATH, VERSION as DEFAULT_VERSION

__all__ = ['DecisionPlaybook', 'DEFAULT_PATH', 'DEFAULT_VERSION']
