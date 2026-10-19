"""Convert Pydantic models into JSON Schemas that structured outputs and tools accept.

Structured outputs support a subset of JSON Schema: every object needs
``additionalProperties: false`` and numeric/string/array constraints are not allowed.
Pydantic still validates the full model client-side, so dropping those constraints
from the wire schema loses nothing.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

_DROPPED_KEYWORDS = frozenset(
    {
        "title",
        "default",
        "examples",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
        "discriminator",
    }
)
_SUPPORTED_FORMATS = frozenset(
    {"date-time", "time", "date", "duration", "email", "hostname", "uri", "ipv4", "ipv6", "uuid"}
)
_SCHEMA_LIST_KEYWORDS = ("anyOf", "allOf", "prefixItems")


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return a self-contained (``$ref``-free) strict JSON Schema for ``model``."""
    raw = model.model_json_schema()
    defs: dict[str, Any] = raw.get("$defs", {})
    result = _transform(raw, defs, ())
    if not isinstance(result, dict):  # pragma: no cover - model schemas are always objects
        raise TypeError("expected an object schema")
    return result


def _transform(node: Any, defs: dict[str, Any], stack: tuple[str, ...]) -> Any:
    if not isinstance(node, dict):
        return node

    if "$ref" in node:
        ref: str = node["$ref"]
        name = ref.rsplit("/", 1)[-1]
        if not ref.startswith("#/$defs/") or name not in defs:
            raise ValueError(f"Unsupported $ref in schema: {ref}")
        if name in stack:
            raise ValueError(f"Recursive schemas are not supported (model {name!r})")
        siblings = {k: v for k, v in node.items() if k != "$ref"}
        return _transform({**defs[name], **siblings}, defs, (*stack, name))

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _DROPPED_KEYWORDS or key == "$defs":
            continue
        if key == "format" and value not in _SUPPORTED_FORMATS:
            continue
        if key == "properties":
            out[key] = {name: _transform(sub, defs, stack) for name, sub in value.items()}
        elif key == "oneOf":
            out["anyOf"] = [_transform(sub, defs, stack) for sub in value]
        elif key in _SCHEMA_LIST_KEYWORDS:
            out[key] = [_transform(sub, defs, stack) for sub in value]
        elif key in ("items", "not"):
            out[key] = _transform(value, defs, stack)
        elif key == "additionalProperties":
            if value not in (False, None):
                raise ValueError(
                    "Free-form mappings (dict fields) are not supported in structured "
                    "outputs; use a list of objects instead."
                )
        else:
            out[key] = value

    if out.get("type") == "object" or "properties" in out:
        out.setdefault("properties", {})
        out["additionalProperties"] = False
    return out
