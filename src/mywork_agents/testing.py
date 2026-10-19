"""Test helpers: a scripted stand-in for the Claude API.

Use :class:`FakeLLM` to test your own agents without network access or API cost::

    llm = FakeLLM([
        FakeLLM.tool_use("read_file", {"path": "README.md"}),
        FakeLLM.json({"summary": "Looks good", "findings": []}),
    ])
    ctx = AgentContext(settings=Settings(), make_llm=lambda model: llm)
    result = MyAgent().run(ctx, "task")
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any

from mywork_agents.core.llm import LLMRequest, LLMResponse, Source, ToolCall, Usage

Scripted = LLMResponse | Callable[[LLMRequest], LLMResponse]


class FakeLLM:
    def __init__(self, responses: Iterable[Scripted] = (), *, model: str = "fake-model") -> None:
        self.model = model
        self._responses = list(responses)
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        # Snapshot the message list: the agent keeps appending to the same list.
        self.requests.append(
            LLMRequest(
                system=request.system,
                messages=list(request.messages),
                max_tokens=request.max_tokens,
                effort=request.effort,
                tools=list(request.tools),
                output_schema=request.output_schema,
            )
        )
        if not self._responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        scripted = self._responses.pop(0)
        response = scripted(request) if callable(scripted) else scripted
        if not response.model:
            response.model = self.model
        return response

    def add(self, *responses: Scripted) -> None:
        """Queue more scripted responses."""
        self._responses.extend(responses)

    @property
    def remaining(self) -> int:
        return len(self._responses)

    # ------------------------------------------------------------- builders

    @staticmethod
    def text(
        text: str,
        *,
        stop_reason: str = "end_turn",
        cited: Iterable[Source] = (),
        search_results: Iterable[Source] = (),
        usage: Usage | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            content=[{"type": "text", "text": text}],
            stop_reason=stop_reason,
            text=text,
            cited_sources=list(cited),
            search_results=list(search_results),
            usage=usage or Usage(input_tokens=100, output_tokens=50, requests=1),
        )

    @staticmethod
    def json(data: Any, **kwargs: Any) -> LLMResponse:
        return FakeLLM.text(json.dumps(data), **kwargs)

    @staticmethod
    def tool_use(name: str, args: dict[str, Any], *more: tuple[str, dict[str, Any]]) -> LLMResponse:
        calls = [(name, args), *more]
        content: list[Any] = []
        tool_calls = []
        for index, (call_name, call_args) in enumerate(calls):
            call_id = f"toolu_fake_{index}"
            content.append(
                {"type": "tool_use", "id": call_id, "name": call_name, "input": call_args}
            )
            tool_calls.append(ToolCall(call_id, call_name, call_args))
        return LLMResponse(
            content=content,
            stop_reason="tool_use",
            tool_calls=tool_calls,
            usage=Usage(input_tokens=100, output_tokens=20, requests=1),
        )
