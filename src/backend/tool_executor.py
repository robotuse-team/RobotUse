"""Execute registered tools through their checked boundary."""
from src.tools.base_tool import ToolExecutionContext


def execute_tool(registry, name, arguments, *, dispatch):
    return registry.dispatch(name, arguments, context=ToolExecutionContext(dispatch=dispatch))
