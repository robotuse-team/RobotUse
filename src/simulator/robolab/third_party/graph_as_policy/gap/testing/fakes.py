"""Scripted stand-ins for the runtime objects skills interact with."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from gap_core.errors import ToolError

from gap.runtime.context import CancelToken


@dataclass
class ToolCallRecord:
    tool: str
    kwargs: dict[str, Any]


class FakeContext:
    """A ``NodeContext`` stand-in with scripted per-tool responses.

    ``tool_responses`` maps tool name → one of:
      - a plain value: returned on every call
      - a callable: invoked with the call kwargs, its return value returned
        (raise inside it to exercise error paths)
      - a list of values: popped front-to-back, one per call (StopIteration →
        ToolError when exhausted)

    Every dispatch is appended to ``calls`` for assertions. Tools without a
    scripted response raise ToolError, so tests fail loudly on unexpected
    calls.
    """

    def __init__(
        self,
        tool_responses: dict[str, Any] | None = None,
        *,
        node_id: str = "test_node",
    ):
        self._responses = dict(tool_responses or {})
        self._sequences: dict[str, list[Any]] = {
            name: list(vals)
            for name, vals in self._responses.items()
            if isinstance(vals, list)
        }
        self.calls: list[ToolCallRecord] = []
        self.published: list[Any] = []
        self.cancel_token = CancelToken()
        self.policy_executor: Any = None
        self._node_id = node_id
        self._lock = threading.Lock()

    # -- NodeContext surface -------------------------------------------------

    def tool(self, name: str, **kwargs: Any) -> Any:
        with self._lock:
            self.calls.append(ToolCallRecord(tool=name, kwargs=kwargs))
            if name in self._sequences:
                seq = self._sequences[name]
                if not seq:
                    raise ToolError(name, "FakeContext: scripted responses exhausted")
                return seq.pop(0)
            if name not in self._responses:
                raise ToolError(
                    name,
                    "FakeContext: no scripted response "
                    f"(scripted: {sorted(self._responses) or 'none'})",
                )
        value = self._responses[name]
        if callable(value):
            return value(**kwargs)
        return value

    def publish(self, value: Any) -> None:
        self.published.append(value)

    # -- assertion helpers ----------------------------------------------------

    def calls_to(self, tool: str) -> list[ToolCallRecord]:
        return [c for c in self.calls if c.tool == tool]

    def call_count(self, tool: str) -> int:
        return len(self.calls_to(tool))
