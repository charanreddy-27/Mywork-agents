"""The Claude API boundary, including end-to-end requests through the real SDK."""

from __future__ import annotations

import json
from typing import Any

import anthropic
import httpx2
import pytest

from mywork_agents.core.llm import (
    FALLBACK_BETA,
    AnthropicLLM,
    LLMRequest,
    MissingCredentialsError,
    ToolInputParseError,
    convert_message,
)
from mywork_agents.models import estimate_cost, model_info

TOOL = {
    "name": "read_file",
    "description": "Read a file.",
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    },
    "eager_input_streaming": True,
}
SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}


def request(**overrides: Any) -> LLMRequest:
    values: dict[str, Any] = {
        "system": "You are a test.",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1000,
        "effort": "medium",
        "tools": [TOOL],
        "output_schema": SCHEMA,
    }
    values.update(overrides)
    return LLMRequest(**values)


# ------------------------------------------------------------- build_params


def test_params_for_current_opus() -> None:
    params = AnthropicLLM("claude-opus-5-5").build_params(request())

    assert params["thinking"] == {"type": "adaptive"}
    assert params["output_config"] == {
        "effort": "medium",
        "format": {"type": "json_schema", "schema": SCHEMA},
    }
    assert params["cache_control"] == {"type": "ephemeral"}
    assert params["tools"] == [TOOL]
    assert params["betas"] == [FALLBACK_BETA]
    assert params["fallbacks"] == "default"
    assert "temperature" not in params
    assert "tool_choice" not in params  # forced tool use is rejected on this model


def test_params_without_fallbacks() -> None:
    assert "fallbacks" not in AnthropicLLM("claude-haiku-5-5").build_params(request())
    disabled = AnthropicLLM("claude-opus-5-5", fallbacks=False).build_params(request())
    assert "fallbacks" not in disabled
    assert "betas" not in disabled


def test_params_for_legacy_model_skip_thinking_and_effort() -> None:
    params = AnthropicLLM("claude-haiku-4-5").build_params(request(output_schema=None, tools=[]))
    assert "thinking" not in params
    assert "output_config" not in params
    assert "tools" not in params


# ---------------------------------------------------------- convert_message


def test_convert_message_collects_text_tools_sources_and_usage() -> None:
    message = anthropic.types.Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "content": [
                {"type": "thinking", "thinking": "", "signature": "sig"},
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_1",
                    "name": "web_search",
                    "input": {"query": "webhook retries"},
                },
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_1",
                    "content": [
                        {
                            "type": "web_search_result",
                            "url": "https://example.com/a",
                            "title": "A",
                            "encrypted_content": "x",
                        }
                    ],
                },
                {
                    "type": "text",
                    "text": "Use exponential backoff.",
                    "citations": [
                        {
                            "type": "web_search_result_location",
                            "url": "https://example.com/a",
                            "title": "A",
                            "cited_text": "backoff",
                            "encrypted_index": "x",
                        }
                    ],
                },
                {"type": "text", "text": " With jitter."},
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 2,
                "server_tool_use": {"web_search_requests": 1, "web_fetch_requests": 0},
            },
        }
    )
    response = convert_message(message)

    assert response.text == "Use exponential backoff. With jitter."
    assert [s.url for s in response.cited_sources] == ["https://example.com/a"]
    assert [s.title for s in response.search_results] == ["A"]
    assert response.server_tool_calls[0].input == {"query": "webhook retries"}
    assert response.usage.web_search_requests == 1
    assert response.usage.total_input_tokens == 15
    assert response.content is not None and len(response.content) == 5


def test_convert_message_tool_use_and_refusal_dicts() -> None:
    response = convert_message(
        {
            "content": [
                {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "x"}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    assert response.tool_calls[0].name == "read_file"

    refused = convert_message(
        {"content": [], "stop_reason": "refusal", "stop_details": {"category": "cyber"}}
    )
    assert refused.refusal_category == "cyber"


def test_mid_output_fallback_drops_declined_partial_blocks() -> None:
    response = convert_message(
        {
            "content": [
                {"type": "thinking", "thinking": "", "signature": "s"},
                {"type": "text", "text": "Partial "},
                {"type": "tool_use", "id": "t0", "name": "read_file", "input": {"path": "a"}},
                {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {}},
                {"type": "web_search_tool_result", "tool_use_id": "s1", "content": []},
                {"type": "server_tool_use", "id": "s2", "name": "web_search", "input": {}},
                {"type": "fallback", "from": {"model": "a"}, "to": {"model": "b"}},
                {"type": "text", "text": "answer."},
                {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "b"}},
            ],
            "stop_reason": "tool_use",
        }
    )
    kinds = [block["type"] for block in response.content]
    assert kinds == [
        "text",
        "server_tool_use",
        "web_search_tool_result",
        "fallback",
        "text",
        "tool_use",
    ]
    assert [call.id for call in response.tool_calls] == ["t1"]  # t0 is never executed
    assert response.text == "Partial answer."


def test_search_error_object_is_ignored() -> None:
    response = convert_message(
        {
            "content": [
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "s1",
                    "content": {
                        "type": "web_search_tool_result_error",
                        "error_code": "max_uses_exceeded",
                    },
                }
            ],
            "stop_reason": "end_turn",
        }
    )
    assert response.search_results == []


def test_cost_estimate() -> None:
    from mywork_agents.core.llm import Usage

    usage = Usage(input_tokens=1_000_000, output_tokens=100_000)
    assert estimate_cost("claude-opus-5-5", usage) == pytest.approx(4.0 + 2.0)
    assert estimate_cost("some-future-model", usage) is None
    assert model_info("claude-opus-4-1").adaptive_thinking is False
    assert model_info("some-future-model").effort is True


# ------------------------------------------------------- through the real SDK


def _sse(events: list[dict[str, Any]]) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def _stream_events(tool_json: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5-5",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 12, "output_tokens": 1},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Checking."},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "read_file",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": tool_json},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"type": "message_delta", "stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 20},
        },
        {"type": "message_stop"},
    ]


class Recorder:
    def __init__(self, tool_json: str = '{"path": "a.py"}') -> None:
        self.tool_json = tool_json
        self.requests: list[httpx2.Request] = []

    def __call__(self, req: httpx2.Request) -> httpx2.Response:
        self.requests.append(req)
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse(_stream_events(self.tool_json)),
        )

    def client(self, api_key: str | None = "test-key") -> anthropic.Anthropic:
        return anthropic.Anthropic(
            api_key=api_key,
            base_url="https://api.anthropic.com",
            http_client=httpx2.Client(transport=httpx2.MockTransport(self)),
            max_retries=0,
        )


def test_streams_through_beta_api_with_fallbacks() -> None:
    recorder = Recorder()
    response = AnthropicLLM("claude-opus-5-5", client=recorder.client()).complete(request())

    sent = recorder.requests[0]
    body = json.loads(sent.content)
    assert sent.url.path == "/v1/messages"
    assert FALLBACK_BETA in sent.headers["anthropic-beta"]
    assert body["stream"] is True
    assert body["fallbacks"] == "default"
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"]["format"]["schema"] == SCHEMA
    assert body["tools"][0]["eager_input_streaming"] is True

    assert response.stop_reason == "tool_use"
    assert response.text == "Checking."
    assert response.tool_calls[0].input == {"path": "a.py"}
    assert response.usage.output_tokens == 20


def test_streams_through_standard_api_without_fallbacks() -> None:
    recorder = Recorder()
    AnthropicLLM("claude-haiku-5-5", client=recorder.client()).complete(request())
    sent = recorder.requests[0]
    assert "anthropic-beta" not in sent.headers
    assert "fallbacks" not in json.loads(sent.content)


def test_unparseable_tool_json_raises_parse_error() -> None:
    recorder = Recorder(tool_json="definitely not json")
    with pytest.raises(ToolInputParseError):
        AnthropicLLM("claude-haiku-5-5", client=recorder.client()).complete(request())


def test_missing_credentials_are_reported_clearly() -> None:
    recorder = Recorder()
    llm = AnthropicLLM("claude-haiku-5-5", client=recorder.client(api_key=None))
    with pytest.raises(MissingCredentialsError, match="ANTHROPIC_API_KEY"):
        llm.complete(request())
    assert recorder.requests == []
