"""Tool discovery must fail closed without changing the existing RobotUse boundary."""

from dataclasses import FrozenInstanceError, replace
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import pytest

from src.tools import (
    ToolExecutionContext, ToolRegistrationError, ToolRegistry, ToolSpec, discover_tools,
)
from src.tools.base_tool import dispatch_existing
from src.tools.schema import object_schema
from src.tools.list import list_tools


def _package(tmp_path, monkeypatch, definitions):
    name = "fixture_tools_" + uuid4().hex
    root = tmp_path / name
    root.mkdir()
    (root / "__init__.py").write_text("")
    for group, source in definitions.items():
        directory = root / group
        directory.mkdir()
        (directory / "__init__.py").write_text("")
        if source is not None:
            (directory / "tool.py").write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    return name, root


def _definition(name, arguments=()):
    return (
        "from src.tools.base_tool import ToolSpec, dispatch_existing\n"
        "from src.tools.schema import object_schema\n"
        f"TOOLS = (ToolSpec({name!r}, {arguments!r}, dispatch_existing, "
        f"input_schema=object_schema({dict.fromkeys(arguments, {'type':'string'})!r})),)\n"
    )


def test_actual_role_variants_and_internal_calls_are_registered():
    from src.backend.orchestrator import AgentOrchestrator
    from src.tools.names import TOOL_SCHEMA_ALIASES
    from test_prime_delegation import Factory
    from test_orchestrator import Backend

    registry = discover_tools()
    runner = AgentOrchestrator(Backend(), Factory())
    flags = (
        "waypoint_task", "waypoint_ref", "placement_refinement",
        "inflight_refinement", "place_refinement",
    )
    exposed = set()
    for values in product((False, True), repeat=len(flags)):
        for stage in (None, "pregrasp", "release"):
            task = {**dict(zip(flags, values)), "stage": stage}
            for role in ("prime", "point", "grasp", "place", "refiner"):
                names = runner._tools(role, task)
                assert registry.require(names) == names
                exposed.update(names)
    internal = {
        "delegate_refiner", "validate_grasp", "execute_grasp",
        "save_destination", "execute_place",
    }
    # refine_candidate is retained by the inherited loop's dynamic branch.
    boundary = {spec.name for spec in list_tools(registry) if spec.kind != "backend"}
    assert boundary == exposed | internal | {"refine_candidate"}
    assert {spec.name for spec in list_tools(registry)
            if spec.visibility == "internal" and spec.kind != "backend"} == internal
    assert dict(registry.aliases) == TOOL_SCHEMA_ALIASES
    assert registry["finish"].kind == "control"
    for spec in list_tools(registry):
        if spec.kind == "tool":
            assert spec.required_arguments == tuple(spec.schema()["required"])
    assert registry["curobo_plan"].kind == "backend"
    assert "curobo_plan" not in exposed


def test_discovery_is_sorted_and_never_imports_nested_dependencies(tmp_path, monkeypatch):
    name, root = _package(tmp_path, monkeypatch, {
        "zeta": _definition("zeta_action"), "alpha": _definition("alpha_action"),
    })
    for relative in ("third_party", "__pycache__", "alpha/third_party", "alpha/worker"):
        directory = root / relative
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").write_text("raise RuntimeError('must not import dependency')\n")
        (directory / "tool.py").write_text("raise RuntimeError('must not import nested tool')\n")
    registry = discover_tools(name)
    assert tuple(registry) == ("alpha_action", "zeta_action")
    assert registry.require(("zeta_action", "alpha_action")) == ("zeta_action", "alpha_action")


def test_new_folder_without_registration_fails_even_after_prior_discovery(tmp_path, monkeypatch):
    name, root = _package(tmp_path, monkeypatch, {"working": _definition("working")})
    discover_tools(name)
    missing = root / "forgotten"
    missing.mkdir()
    (missing / "__init__.py").write_text("")
    with pytest.raises(ToolRegistrationError, match="missing tool.py"):
        discover_tools(name)


def test_new_tool_handler_registers_at_startup_without_a_legacy_map_entry(tmp_path, monkeypatch):
    from src.runtime import bootstrap

    name, _ = _package(tmp_path, monkeypatch, {"new_tool": (
        "from src.tools.base_tool import ToolSpec\n"
        "def call(context, name, arguments):\n"
        "    return {'received': arguments['value']}\n"
        "from src.tools.schema import object_schema\n"
        "TOOLS = (ToolSpec('new_tool', ('value',), call, "
        "input_schema=object_schema({'value': {'type': 'number'}})),)\n"
    )})
    registry = ToolRegistry((*discover_tools().tools.values(), *discover_tools(name).tools.values()))
    monkeypatch.setattr(bootstrap, "discover_tools", lambda: registry)
    loaded = bootstrap.load_tool_registry()

    def inherited_dispatch(name, arguments):
        raise AssertionError("new handler must not require the legacy dispatcher")

    assert loaded.required_arguments("new_tool") == ("value",)
    assert loaded.dispatch("new_tool", {"value": 7},
        context=ToolExecutionContext(inherited_dispatch)) == {"received": 7}


def test_unmarked_tool_folder_is_rejected(tmp_path, monkeypatch):
    name, root = _package(tmp_path, monkeypatch, {"working": _definition("working")})
    (root / "forgotten").mkdir()
    with pytest.raises(ToolRegistrationError, match="missing __init__.py"):
        discover_tools(name)


@pytest.mark.parametrize("source,match", [
    (None, "missing tool.py"),
    ("", "nonempty tuple"),
    ("TOOLS = ()\n", "nonempty tuple"),
    ("TOOLS = []\n", "nonempty tuple"),
    ("TOOLS = (object(),)\n", "ToolSpec"),
    ("raise ImportError('dependency missing')\n", "failed to import"),
])
def test_incomplete_packages_fail_at_discovery(tmp_path, monkeypatch, source, match):
    name, _ = _package(tmp_path, monkeypatch, {"broken": source})
    with pytest.raises(ToolRegistrationError, match=match):
        discover_tools(name)


def test_duplicate_tool_names_across_packages_fail_at_discovery(tmp_path, monkeypatch):
    name, _ = _package(tmp_path, monkeypatch, {
        "first": _definition("collision"), "second": _definition("collision"),
    })
    with pytest.raises(ToolRegistrationError, match="duplicate tool name"):
        discover_tools(name)


@pytest.mark.parametrize("binding", ["FORGOTTEN", "_forgotten"])
def test_named_tool_declaration_omitted_from_nonempty_tools_fails(tmp_path, monkeypatch, binding):
    source = (
        "from src.tools.base_tool import ToolSpec, dispatch_existing\n"
        "REGISTERED = ToolSpec('registered', (), dispatch_existing)\n"
        f"{binding} = ToolSpec('forgotten', (), dispatch_existing)\n"
        "TOOLS = (REGISTERED,)\n"
    )
    name, _ = _package(tmp_path, monkeypatch, {"partial": source})
    with pytest.raises(ToolRegistrationError, match=f"omitted from TOOLS: {binding}"):
        discover_tools(name)


def test_named_definitions_and_multiple_bindings_to_registered_spec_are_allowed(tmp_path, monkeypatch):
    source = (
        "from src.tools.base_tool import ToolSpec, dispatch_existing\n"
        "DECLARED = ToolSpec('declared', (), dispatch_existing)\n"
        "ALIAS = DECLARED\n"
        "TOOLS = (DECLARED,)\n"
    )
    name, _ = _package(tmp_path, monkeypatch, {"complete": source})
    assert tuple(discover_tools(name)) == ("declared",)


def test_equal_recreated_spec_does_not_hide_an_unregistered_declaration(tmp_path, monkeypatch):
    source = (
        "from src.tools.base_tool import ToolSpec, dispatch_existing\n"
        "DECLARED = ToolSpec('declared', (), dispatch_existing)\n"
        "TOOLS = (ToolSpec('declared', (), dispatch_existing),)\n"
    )
    name, _ = _package(tmp_path, monkeypatch, {"partial": source})
    with pytest.raises(ToolRegistrationError, match="omitted from TOOLS: DECLARED"):
        discover_tools(name)


@pytest.mark.parametrize("changes,match", [
    ({"name": ""}, "invalid tool name"),
    ({"name": "not-a-tool"}, "invalid tool name"),
    ({"internal_name": "bad alias"}, "invalid tool internal_name"),
    ({"required_arguments": ["candidate_ref"]}, "must be a tuple"),
    ({"required_arguments": ("candidate_ref", "candidate_ref")}, "duplicate required argument"),
    ({"required_arguments": ("not valid",)}, "invalid required argument"),
    ({"required_arguments": (None,)}, "invalid required argument"),
    ({"handler": None}, "handler must be callable"),
    ({"handler": lambda: None}, "handler must accept"),
    ({"handler": lambda context, name, arguments, fourth: None}, "handler must accept"),
    ({"kind": "unknown"}, "invalid tool kind"),
    ({"kind": "backend"}, "backend tools must remain internal"),
    ({"visibility": "unknown"}, "invalid visibility"),
])
def test_invalid_tool_definitions_are_rejected(changes, match):
    spec = replace(ToolSpec("working", ("candidate_ref",), dispatch_existing,
        input_schema=object_schema({"candidate_ref": {"type": "string"}})), **changes)
    with pytest.raises(ToolRegistrationError, match=match):
        ToolRegistry((spec,))


@pytest.mark.parametrize("second", [
    ToolSpec("public", (), dispatch_existing),
    ToolSpec("legacy", (), dispatch_existing),
    ToolSpec("other", (), dispatch_existing, internal_name="legacy"),
    ToolSpec("other", (), dispatch_existing, internal_name="public"),
])
def test_alias_collisions_are_rejected(second):
    first = ToolSpec("public", (), dispatch_existing, internal_name="legacy")
    with pytest.raises(ToolRegistrationError, match="duplicate tool name or alias"):
        ToolRegistry((first, second))


def test_registry_and_definitions_are_immutable():
    registry = discover_tools()
    with pytest.raises(TypeError):
        registry.tools["new"] = registry["finish"]
    with pytest.raises(TypeError):
        registry.aliases["new"] = "old"
    with pytest.raises(FrozenInstanceError):
        registry["finish"].name = "changed"
    with pytest.raises(ToolRegistrationError, match="cannot be empty"):
        ToolRegistry(())
    with pytest.raises(ToolRegistrationError, match="unregistered tool"):
        registry.require(("unregistered_action",))


def test_missing_or_inconsistent_owned_schema_fails_before_execution():
    with pytest.raises(ToolRegistrationError, match="missing input_schema"):
        ToolRegistry((ToolSpec("missing", ("numeric_value",), dispatch_existing),))
    with pytest.raises(ToolRegistrationError, match="required fields must match"):
        ToolRegistry((ToolSpec("mismatch", ("required_value",), dispatch_existing,
            input_schema=object_schema({"different_value": {"type": "number"}})),))
    with pytest.raises(ToolRegistrationError, match="explicit supported type"):
        ToolRegistry((ToolSpec("missing_type", ("value",), dispatch_existing,
            input_schema=object_schema({"value": {}})),))


def test_dispatch_preserves_existing_internal_name_arguments_result_and_failure():
    registry = discover_tools()
    arguments = {"candidate_ref": "current", "dx_mm": float("nan")}
    result = object()
    calls = []

    def existing_boundary(name, args):
        calls.append((name, args))
        return result

    context = ToolExecutionContext(dispatch=existing_boundary)
    assert registry.dispatch("adjust_place", arguments, context=context) is result
    assert registry.dispatch("explicit_adjust_place", arguments, context=context) is result
    assert calls == [("explicit_adjust_place", arguments), ("explicit_adjust_place", arguments)]
    assert all(args is arguments for _, args in calls)

    failure = ValueError("existing stale-reference rejection")

    def reject(name, args):
        raise failure

    with pytest.raises(ValueError) as caught:
        registry.dispatch("execute_grasp", arguments, context=ToolExecutionContext(dispatch=reject))
    assert caught.value is failure


def test_discovery_does_not_import_model_simulator_or_legacy_runtime():
    root = Path(__file__).resolve().parents[1]
    code = """
import json, sys
from src.tools import discover_tools
registry = discover_tools()
forbidden = ('torch', 'numpy', 'PIL', 'transformers', 'isaaclab', 'robot_skill_selector')
print(json.dumps([name for name in sys.modules
                  if any(name == prefix or name.startswith(prefix + '.') for prefix in forbidden)]))
"""
    completed = subprocess.run([sys.executable, "-c", code], cwd=root,
                               capture_output=True, text=True, check=True)
    assert json.loads(completed.stdout) == []
