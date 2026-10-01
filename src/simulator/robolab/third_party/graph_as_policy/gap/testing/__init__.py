"""Public test fixtures for gap and open-robot-skills bundle authors.

These are the same fakes gap's own suite uses, exported so a skill bundle can
be unit-tested without a robot, a GPU, or an LLM:

    from gap.testing import FakeContext, make_test_observation

    ctx = FakeContext(tool_responses={
        "grounding-dino.detect": {"detections": [...]},
        "vlm.query": "A",
    })
    out = my_script.run(ctx, object_name="soup can")
    assert ctx.calls[0].tool == "grounding-dino.detect"
"""

from gap.testing.fakes import FakeContext, ToolCallRecord
from gap.testing.graphs import assert_graph_valid
from gap.testing.observations import make_test_observation

__all__ = [
    "FakeContext",
    "ToolCallRecord",
    "make_test_observation",
    "assert_graph_valid",
]
