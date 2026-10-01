"""Grasp implementation and CGN wire client used by RobotUse."""

__all__ = ["GraspBackend", "ContactGraspNetClient", "get_backend_class"]


def get_backend_class():
    """Return the original backend class; the runtime owns mixin composition."""
    from .backend import GraspBackend

    return GraspBackend


def __getattr__(name):
    if name == "GraspBackend":
        return get_backend_class()
    if name == "ContactGraspNetClient":
        from .cgn_client import ContactGraspNetClient

        return ContactGraspNetClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
