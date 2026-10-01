"""Definitions owned by gripper tools; release remains a separate decision."""

from ..base_tool import ToolSpec, define_tool, dispatch_existing

TOOLS = (
    define_tool('close_for_push', {
    }),
    define_tool('release', {
    }),
)
