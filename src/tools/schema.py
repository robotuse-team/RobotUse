"""Small JSON-schema helpers shared by tool-owned input contracts.

There is deliberately no tool-name or argument-name table here. A new tool
declares all of its fields in its own package; the same schema drives provider
serialization and validation at the agent boundary.
"""
from copy import deepcopy
import math
from collections.abc import Mapping

from src.core.contracts import BoundaryError


def object_schema(properties, required=None, **metadata):
    return dict(type="object", properties=deepcopy(properties),
                required=list(properties if required is None else required),
                additionalProperties=False, **metadata)


def text_schema(**metadata):
    return dict(type="string", **metadata)


def number_schema(**metadata):
    return dict(type="number", **metadata)


def bounded_text(value, schema):
    """Retain tools whose stateful handler owns enum rejection and feedback."""
    return validate_value({"type": "string"}, value, "text")


def normalized_pixel(value, schema):
    """Preserve the original int/float value for normalized image coordinates."""
    validate_value(schema, value, "coordinate")
    return value


def validate_value(schema, value, name):
    """Validate the declared JSON types, retaining RobotUse's finite-number boundary."""
    kinds = schema.get("type")
    kinds = kinds if isinstance(kinds, list) else [kinds]
    if value is None and "null" in kinds:
        return None
    if "number" in kinds or "integer" in kinds:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise BoundaryError(f"{name} must be a finite number")
        if "number" not in kinds and type(value) is not int:
            raise BoundaryError(f"{name} must be an integer")
        if not schema.get("minimum", -math.inf) <= value <= schema.get("maximum", math.inf):
            raise BoundaryError(f"{name} is outside its declared range")
        result = float(value) if "number" in kinds else value
    elif "string" in kinds:
        if not isinstance(value, str) or not value.strip() or len(value) > schema.get("maxLength", 8192):
            raise BoundaryError("expected bounded nonempty string")
        if len(value) < schema.get("minLength", 0):
            raise BoundaryError(f"{name} is shorter than its declared length")
        result = value
    elif "boolean" in kinds:
        if type(value) is not bool:
            raise BoundaryError(f"{name} must be a boolean")
        result = value
    elif "array" in kinds:
        if not isinstance(value, (list, tuple)):
            raise BoundaryError(f"{name} must be an array")
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", math.inf):
            raise BoundaryError(f"{name} has an invalid number of items")
        result = [validate_value(schema["items"], item, f"{name}[{index}]")
                  for index, item in enumerate(value)]
    elif "object" in kinds:
        if not isinstance(value, Mapping):
            raise BoundaryError(f"{name} must be an object")
        properties = schema.get("properties", {})
        if not set(schema.get("required", ())) <= set(value):
            raise BoundaryError(f"{name} is missing required fields")
        if schema.get("additionalProperties") is False and not set(value) <= set(properties):
            raise BoundaryError(f"{name} contains undeclared fields")
        result = {key: validate_value(properties[key], item, f"{name}.{key}")
                  if key in properties else item for key, item in value.items()}
    else:
        raise BoundaryError(f"{name} has no supported declared type")
    if "enum" in schema and value not in schema["enum"]:
        raise BoundaryError(f"{name} must be one of {schema['enum']}")
    return result


def validate_schema(schema, name):
    """Reject malformed declarations at discovery, before runtime imports."""
    if not isinstance(schema, Mapping):
        raise ValueError(f"{name}: input schema must be a mapping")
    declared = schema.get("type")
    kinds = declared if isinstance(declared, list) else [declared]
    allowed = {"object", "array", "string", "number", "integer", "boolean", "null"}
    if not kinds or any(not isinstance(kind, str) or kind not in allowed for kind in kinds):
        raise ValueError(f"{name}: input schema requires an explicit supported type")
    if len(kinds) > 1 and (len(kinds) != 2 or "null" not in kinds):
        raise ValueError(f"{name}: only nullable type unions are supported")
    supported = {"type", "description", "properties", "required", "additionalProperties", "items",
                 "minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems", "enum"}
    if unknown := set(schema) - supported:
        raise ValueError(f"{name}: unsupported schema keywords: {sorted(unknown)}")
    for key in ("minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems"):
        if key in schema and (type(schema[key]) not in (int, float) or not math.isfinite(schema[key])):
            raise ValueError(f"{name}: {key} must be a finite numeric bound")
    for lower, upper in (("minimum", "maximum"), ("minLength", "maxLength"), ("minItems", "maxItems")):
        if schema.get(lower, -math.inf if lower == "minimum" else 0) > schema.get(upper, math.inf):
            raise ValueError(f"{name}: {lower} exceeds {upper}")
    if "enum" in schema and (not isinstance(schema["enum"], (list, tuple)) or not schema["enum"]):
        raise ValueError(f"{name}: enum must contain allowed values")
    if "object" in kinds:
        properties = schema.get("properties")
        required = schema.get("required")
        if not isinstance(properties, Mapping) or not isinstance(required, (list, tuple)):
            raise ValueError(f"{name}: object schema requires properties and required")
        if len(set(required)) != len(required) or not set(required) <= set(properties):
            raise ValueError(f"{name}: required fields must be declared exactly once")
        if schema.get("additionalProperties") is not False:
            raise ValueError(f"{name}: object schema must reject undeclared fields")
        for key, value in properties.items():
            validate_schema(value, f"{name}.{key}")
    if "array" in kinds:
        validate_schema(schema.get("items"), f"{name}[]")
