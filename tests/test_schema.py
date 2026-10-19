from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel, Field

from mywork_agents.core.schema import strict_json_schema


class Owner(BaseModel):
    title: str = Field(description="A field literally named 'title'.", max_length=10)
    email: str | None = None


class Report(BaseModel):
    owners: list[Owner] = Field(min_length=1)
    status: Literal["ok", "bad"]
    score: int = Field(ge=0, le=10)


def test_inlines_refs_and_closes_objects() -> None:
    schema = strict_json_schema(Report)

    assert "$defs" not in schema
    assert schema["additionalProperties"] is False
    owner = schema["properties"]["owners"]["items"]
    assert owner["type"] == "object"
    assert owner["additionalProperties"] is False
    assert owner["properties"]["title"] == {
        "description": "A field literally named 'title'.",
        "type": "string",
    }


def test_drops_unsupported_constraints_but_keeps_enums() -> None:
    schema = strict_json_schema(Report)

    assert schema["properties"]["score"] == {"type": "integer"}
    assert "minItems" not in schema["properties"]["owners"]
    assert schema["properties"]["status"]["enum"] == ["ok", "bad"]
    assert "title" not in schema  # the model title keyword is dropped
    assert schema["required"] == ["owners", "status", "score"]


def test_optional_fields_become_any_of_null() -> None:
    owner = strict_json_schema(Owner)
    assert owner["properties"]["email"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}
    assert owner["required"] == ["title"]


def test_rejects_free_form_dicts() -> None:
    class Bad(BaseModel):
        labels: dict[str, str]

    with pytest.raises(ValueError, match="dict fields"):
        strict_json_schema(Bad)


def test_rejects_recursive_models() -> None:
    class Node(BaseModel):
        children: list[Node]

    Node.model_rebuild()
    with pytest.raises(ValueError, match="Recursive"):
        strict_json_schema(Node)
