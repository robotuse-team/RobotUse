"""Shared, unclipped interpretation of public gripper proprioception.

These measurements can disqualify pickup evidence; a nonempty reading alone
never establishes that the intended object is held. Records are JSON-safe so
invalid telemetry survives error handling without becoming a valid fraction.
"""
from __future__ import annotations

import math
from numbers import Integral, Real


EMPTY_GRIPPER_MAX_FRACTION = 0.02


def classify_gripper_fraction(raw_value, *, source, stage):
    numeric = isinstance(raw_value, Real) and not isinstance(raw_value, bool)
    try:
        value = float(raw_value) if numeric else None
    except (OverflowError, TypeError, ValueError):
        value = None
    valid = value is not None and math.isfinite(value) and 0 <= value <= 1
    if numeric and isinstance(raw_value, Integral):
        raw = int(raw_value)
    elif numeric and value is not None:
        raw = value if math.isfinite(value) else str(value)
    elif raw_value is None or isinstance(raw_value, (str, bool)):
        raw = raw_value
    else:
        # Do not stringify arbitrary objects: repr may contain unrelated data.
        raw = None
    return {
        "raw_value": raw,
        "raw_type": type(raw_value).__name__,
        "source": source,
        "stage": stage,
        "validity": "missing" if raw_value is None else ("valid" if valid else "invalid"),
        "value": value if valid else None,
        "empty": value <= EMPTY_GRIPPER_MAX_FRACTION if valid else None,
        "nonempty": value > EMPTY_GRIPPER_MAX_FRACTION if valid else None,
    }


def read_gripper_fraction(connector, *, stage):
    source = "connector.get_gripper_fraction"
    try:
        getter = getattr(connector, "get_gripper_fraction", None)
        raw = getter() if callable(getter) else None
    except Exception as exc:
        measurement = classify_gripper_fraction(None, source=source, stage=stage)
        measurement.update(validity="invalid", error=type(exc).__name__)
        return measurement
    return classify_gripper_fraction(raw, source=source, stage=stage)
