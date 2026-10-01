"""Existing point-prompt RGB-D adapters and their measured geometry type."""

__all__ = ["PointGeometry", "PointRGBDAdapter", "MultiviewPointRGBDAdapter"]


def __getattr__(name):
    if name in ("PointGeometry", "PointRGBDAdapter"):
        from src.tools.perception import rgbd_adapter as point_rgbd_adapter

        return getattr(point_rgbd_adapter, name)
    if name == "MultiviewPointRGBDAdapter":
        from src.tools.perception.multiview_adapter import MultiviewPointRGBDAdapter

        return MultiviewPointRGBDAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
