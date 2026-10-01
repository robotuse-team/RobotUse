"""Tool-owned inputs must reach both the LLM wire and execution boundary."""
import json
from types import SimpleNamespace

import pytest

from src.backend.orchestrator import AgentOrchestrator
from src.core.contracts import BoundaryError
from src.llm.image_context import ImageRegistry
from src.llm.manager import OpenRouterSession
from src.tools import ToolExecutionContext, ToolRegistrationError, ToolRegistry, discover_tools
from src.tools.base_tool import define_tool
from test_modular_tool_registry import _package


@pytest.fixture
def registered_numeric_tool(tmp_path, monkeypatch):
    package, _ = _package(tmp_path, monkeypatch, {"numeric": '''
from src.tools.base_tool import define_tool

def call(context, name, arguments):
    return dict(arguments)

TOOLS = (define_tool('sample_motion', {
    'travel_speed': {'type': 'number', 'minimum': 0.01, 'maximum': 2.0},
    'sample_count': {'type': 'integer', 'minimum': 1, 'maximum': 10},
    'capture': {'type': 'boolean'},
}, required=('travel_speed',), handler=call),)
'''})
    return ToolRegistry((*discover_tools().tools.values(), *discover_tools(package).tools.values()))


@pytest.mark.parametrize("native", [True, False])
def test_new_numeric_and_optional_fields_need_no_central_key_table(
        tmp_path, monkeypatch, registered_numeric_tool, native):
    registry = registered_numeric_tool
    factory = SimpleNamespace(images=ImageRegistry(), model="offline-test", key="offline-test",
        output_dir=tmp_path, json_action_fallback=not native, active_perception=True,
        intent_driven=True, review_driven=True, target_intent_mode=True, explicit_geometry_enabled=True,
        image_history_policy="current_turn", tool_registry=registry)
    arguments = {"travel_speed": 0.25, "sample_count": 3, "capture": True}
    action = {"tool": "sample_motion", "arguments": arguments}
    reply = ({"tool_calls": [{"id": "offline", "type": "function", "function": {
        "name": action["tool"], "arguments": json.dumps(arguments)}}]} if native else
        {"content": json.dumps(action)})
    payload = {"choices": [{"message": reply, "finish_reason": "tool_calls" if native else "stop"}]}
    captured = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self):
            return json.dumps(payload).encode()

    def offline_request(request, **kwargs):
        captured.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr("src.llm.manager.urllib.request.urlopen", offline_request)
    selected = OpenRouterSession(factory, "prime", "offline-contract").next_action(
        [{"role": "user", "content": "Use the supplied numeric arguments."}], ("sample_motion",))
    assert selected.arguments == arguments
    schema = registry["sample_motion"].schema()
    if native:
        assert captured[0]["tools"][0]["function"]["parameters"] == schema
    else:
        instructions = captured[0]["messages"][0]["content"]
        assert '"travel_speed": {"type": "number"' in instructions
        assert '"sample_count": {"type": "integer"' in instructions
        assert '"capture": {"type": "boolean"' in instructions

    controller = object.__new__(AgentOrchestrator)
    controller.tool_registry = registry
    assert controller._required_arguments("sample_motion") == ("travel_speed",)
    assert controller._optional_arguments("sample_motion") == ("sample_count", "capture")
    cleaned = {key: controller._tool_argument("sample_motion", key, value)
               for key, value in selected.arguments.items()}
    assert registry.dispatch("sample_motion", cleaned,
        context=ToolExecutionContext(lambda *_: pytest.fail("unexpected inherited dispatch"))) == arguments
    for key, invalid in (("travel_speed", "0.25"), ("travel_speed", float("nan")),
                         ("travel_speed", 3.0), ("sample_count", 1.2),
                         ("sample_count", True), ("capture", 1), ("undeclared", 0)):
        with pytest.raises(BoundaryError):
            controller._tool_argument("sample_motion", key, invalid)


def test_nested_grasp_and_place_inputs_use_the_declared_geometry_contract():
    registry = discover_tools()
    height = {"reference": "clicked_point", "value_m": 0.04}
    assert registry.validate_argument("prepare_place", "height", height) == height
    assert registry.validate_argument("grasp_candidates", "geometric_height", None) is None
    assert registry.validate_argument("prepare_place", "xy_m", (0.3, -0.1)) == [0.3, -0.1]
    for invalid in ({**height, "hidden_field": 1}, {"reference": "made_up", "value_m": 0},
                    {"reference": "absolute", "value_m": float("inf")}, {"reference": "absolute"}):
        with pytest.raises(BoundaryError):
            registry.validate_argument("prepare_place", "height", invalid)


def test_conditional_optional_place_angles_match_provider_and_controller():
    registry = discover_tools()
    for enabled in (False, True):
        factory = SimpleNamespace(place_rotation=enabled)
        controller = SimpleNamespace(interaction_features=SimpleNamespace(place_rotation=enabled))
        assert registry["nudge_place"].schema(context=factory) == registry["nudge_place"].schema(context=controller)
        assert registry.optional_arguments("nudge_place", context=controller) == (
            ("roll_deg", "pitch_deg", "yaw_deg") if enabled else ())


def test_zero_turn_and_pixel_types_preserve_existing_boundary_behavior():
    registry = discover_tools()
    with pytest.raises(BoundaryError, match="nonzero"):
        registry.validate_argument("turn", "angle_deg", 0)
    assert type(registry.validate_argument("select_region", "u", 100)) is int
    assert type(registry.validate_argument("select_region", "u", 100.0)) is float


def test_returned_schema_edits_do_not_change_registered_inputs():
    registry = discover_tools()
    schema = registry["grasp_candidates"].schema()
    schema["properties"]["direction"]["enum"].append("invented")
    assert "invented" not in registry["grasp_candidates"].schema()["properties"]["direction"]["enum"]


@pytest.mark.parametrize("declaration", [
    {"value": {"type": "number", "minimum": "zero"}},
    {"value": {"type": "number", "minimum": 2, "maximum": 1}},
    {"value": {"type": "string", "pattern": "unsupported"}},
    {"value": {"type": ["number", "string"]}},
    {"value": {"type": "array"}},
    {"not a key": {"type": "string"}},
])
def test_malformed_or_unsupported_schema_is_rejected_at_registration(declaration):
    with pytest.raises(ToolRegistrationError):
        ToolRegistry((define_tool("broken", declaration),))
