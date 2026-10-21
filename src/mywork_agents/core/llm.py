"""The boundary between agents and the Claude API.

Agents talk to an :class:`LLM` (a one-method protocol), which keeps them testable with
:class:`mywork_agents.testing.FakeLLM`. :class:`AnthropicLLM` is the real implementation:
it streams every request (no HTTP timeouts on long turns), asks for structured output
through ``output_config.format`` and opts into server-side refusal fallbacks on models
that support them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from mywork_agents.errors import ConfigError
from mywork_agents.models import model_info

if TYPE_CHECKING:
    import anthropic

    from mywork_agents.config import Effort

FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: Any


@dataclass(frozen=True)
class Source:
    url: str
    title: str | None = None


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    web_search_requests: int = 0
    requests: int = 0
    # Estimated per request (price tiers depend on each request's prompt size).
    cost_usd: float = 0.0
    unpriced_requests: int = 0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        self.web_search_requests += other.web_search_requests
        self.requests += other.requests
        self.cost_usd += other.cost_usd
        self.unpriced_requests += other.unpriced_requests

    @property
    def estimated_cost(self) -> float | None:
        """Estimated USD cost, or None if a request used a model without known prices."""
        return None if self.unpriced_requests else self.cost_usd

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens

    def to_dict(self) -> dict[str, int | float | None]:
        cost = self.estimated_cost
        return {
            "estimated_cost_usd": round(cost, 6) if cost is not None else None,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "web_search_requests": self.web_search_requests,
            "requests": self.requests,
        }


@dataclass
class LLMRequest:
    system: str
    messages: list[dict[str, Any]]
    max_tokens: int
    effort: Effort | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    output_schema: dict[str, Any] | None = None


@dataclass
class LLMResponse:
    # Raw content blocks, appended back to the conversation unchanged (this preserves
    # thinking blocks and server-tool results between turns).
    content: list[Any]
    stop_reason: str | None
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    server_tool_calls: list[ToolCall] = field(default_factory=list)
    cited_sources: list[Source] = field(default_factory=list)
    search_results: list[Source] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    refusal_category: str | None = None


NO_CREDENTIALS_HELP = (
    "No Claude API credentials found. Set the ANTHROPIC_API_KEY environment variable "
    "or run `ant auth login`."
)


class MissingCredentialsError(ConfigError):
    """Neither an API key, an auth token nor an `ant` profile is configured."""


class ToolInputParseError(Exception):
    """The stream carried tool-input JSON that could not be parsed; re-issue the turn."""


class LLM(Protocol):
    model: str

    def complete(self, request: LLMRequest) -> LLMResponse: ...


class AnthropicLLM:
    """:class:`LLM` backed by the official ``anthropic`` SDK."""

    def __init__(
        self,
        model: str,
        *,
        client: anthropic.Anthropic | None = None,
        fallbacks: bool = True,
        timeout: float = 600.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.fallbacks = fallbacks
        self._client = client
        self._timeout = timeout
        self._max_retries = max_retries

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            import anthropic

            # Credentials resolve from ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN or an
            # `ant auth login` profile.
            self._client = anthropic.Anthropic(timeout=self._timeout, max_retries=self._max_retries)
        return self._client

    @property
    def uses_fallbacks(self) -> bool:
        return self.fallbacks and model_info(self.model).refusal_fallbacks

    def build_params(self, request: LLMRequest) -> dict[str, Any]:
        """Keyword arguments for ``messages.stream``; pure, so it is easy to test."""
        info = model_info(self.model)
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "system": request.system,
            "messages": request.messages,
            # Automatic prompt caching: the tool loop resends a growing prefix each turn.
            "cache_control": {"type": "ephemeral"},
        }
        if info.adaptive_thinking:
            params["thinking"] = {"type": "adaptive"}
        output_config: dict[str, Any] = {}
        if request.effort and info.effort:
            output_config["effort"] = request.effort
        if request.output_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": request.output_schema}
        if output_config:
            params["output_config"] = output_config
        if request.tools:
            params["tools"] = request.tools
        if self.uses_fallbacks:
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"
        return params

    def complete(self, request: LLMRequest) -> LLMResponse:
        params = self.build_params(request)
        api = self.client.beta.messages if self.uses_fallbacks else self.client.messages
        try:
            with api.stream(**params) as stream:
                for _event in stream:
                    pass
                message = stream.get_final_message()
        except UnicodeError:
            raise  # the request itself could not be encoded; retrying will not help
        except ValueError as exc:
            # The SDK raises ValueError (and only that) for tool-input JSON it cannot
            # parse at all. API errors are not ValueErrors and propagate untouched.
            raise ToolInputParseError(str(exc)) from exc
        except TypeError as exc:
            # With no credentials anywhere, the SDK fails the request with a TypeError.
            if "authentication method" in str(exc):
                raise MissingCredentialsError(NO_CREDENTIALS_HELP) from exc
            raise
        return convert_message(message)


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def echoable_content(content: list[Any]) -> list[Any]:
    """Drop what must not be sent back after a mid-output refusal fallback.

    Before the last ``fallback`` block, only text and paired server-tool blocks may be
    echoed: thinking, ``tool_use`` (never executed), unpaired ``server_tool_use`` and
    unknown block types are omitted. Everything after the boundary is kept.
    """
    boundary = max(
        (i for i, block in enumerate(content) if _attr(block, "type") == "fallback"), default=-1
    )
    if boundary < 0:
        return content
    before = content[:boundary]
    answered = {
        _attr(block, "tool_use_id")
        for block in before
        if str(_attr(block, "type", "")).endswith("_tool_result")
    }
    kept = []
    for block in before:
        kind = str(_attr(block, "type", ""))
        if (
            kind == "text"
            or (kind == "server_tool_use" and _attr(block, "id") in answered)
            or (kind.endswith("_tool_result") and _attr(block, "tool_use_id") is not None)
        ):
            kept.append(block)
    return kept + content[boundary:]


def convert_message(message: Any) -> LLMResponse:
    """Turn an SDK ``Message`` (or ``BetaMessage``) into an :class:`LLMResponse`."""
    content = echoable_content(list(_attr(message, "content", None) or []))
    texts: list[str] = []
    tool_calls: list[ToolCall] = []
    server_calls: list[ToolCall] = []
    cited: list[Source] = []
    results: list[Source] = []

    for block in content:
        kind = _attr(block, "type")
        if kind == "text":
            texts.append(_attr(block, "text", ""))
            for citation in _attr(block, "citations", None) or []:
                url = _attr(citation, "url")
                if url:
                    cited.append(Source(url, _attr(citation, "title")))
        elif kind == "tool_use":
            tool_calls.append(
                ToolCall(_attr(block, "id"), _attr(block, "name"), _attr(block, "input"))
            )
        elif kind == "server_tool_use":
            server_calls.append(
                ToolCall(_attr(block, "id"), _attr(block, "name"), _attr(block, "input"))
            )
        elif kind == "web_fetch_tool_result":
            fetched = _attr(block, "content")
            if _attr(fetched, "type") == "web_fetch_result" and _attr(fetched, "url"):
                document = _attr(fetched, "content")
                results.append(Source(_attr(fetched, "url"), _attr(document, "title")))
        elif kind == "web_search_tool_result":
            found = _attr(block, "content")
            # A successful search returns a list; an error returns a single object.
            if isinstance(found, list):
                for item in found:
                    url = _attr(item, "url")
                    if url:
                        results.append(Source(url, _attr(item, "title")))

    usage = Usage(requests=1)
    raw_usage = _attr(message, "usage")
    if raw_usage is not None:
        usage.input_tokens = _attr(raw_usage, "input_tokens", 0) or 0
        usage.output_tokens = _attr(raw_usage, "output_tokens", 0) or 0
        usage.cache_creation_input_tokens = _attr(raw_usage, "cache_creation_input_tokens", 0) or 0
        usage.cache_read_input_tokens = _attr(raw_usage, "cache_read_input_tokens", 0) or 0
        server_usage = _attr(raw_usage, "server_tool_use")
        if server_usage is not None:
            usage.web_search_requests = _attr(server_usage, "web_search_requests", 0) or 0

    stop_reason = _attr(message, "stop_reason")
    refusal_category = None
    if stop_reason == "refusal":
        details = _attr(message, "stop_details")
        refusal_category = _attr(details, "category") if details is not None else None

    return LLMResponse(
        content=content,
        stop_reason=stop_reason,
        text="".join(texts),
        tool_calls=tool_calls,
        server_tool_calls=server_calls,
        cited_sources=cited,
        search_results=results,
        usage=usage,
        model=_attr(message, "model", "") or "",
        refusal_category=refusal_category,
    )
