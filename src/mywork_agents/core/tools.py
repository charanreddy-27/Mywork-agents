"""Client-side tools: plain Python functions with a Pydantic argument model.

Tool input is validated against the model before the function runs. That matters
because requests are streamed with ``eager_input_streaming`` on, which hands
validation of tool input to the client.
"""

from __future__ import annotations

import inspect
import json
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError

from mywork_agents.core.schema import strict_json_schema
from mywork_agents.errors import ToolError

if TYPE_CHECKING:
    from mywork_agents.config import Settings
    from mywork_agents.history import HistoryStore

MAX_TOOL_OUTPUT_CHARS = 30_000

ArgsT = TypeVar("ArgsT", bound=BaseModel)


@dataclass
class ToolContext:
    """What a tool may touch: the workspace root, settings and run history."""

    workspace: Path
    settings: Settings
    history: HistoryStore | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolOutcome:
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    func: Callable[[ToolContext, Any], str]

    def definition(self, *, eager_input_streaming: bool = False) -> dict[str, Any]:
        spec: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "input_schema": strict_json_schema(self.args_model),
        }
        if eager_input_streaming:
            spec["eager_input_streaming"] = True
        return spec

    def invoke(self, ctx: ToolContext, raw_input: Any) -> ToolOutcome:
        outcome = self._invoke(ctx, raw_input)
        return ToolOutcome(valid_utf8(outcome.content), outcome.is_error)

    def _invoke(self, ctx: ToolContext, raw_input: Any) -> ToolOutcome:
        try:
            args = self.args_model.model_validate(raw_input)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or 'input'}: {err['msg']}"
                for err in exc.errors()
            )
            payload = {"INVALID_JSON": json.dumps(raw_input, default=str), "errors": problems}
            return ToolOutcome(json.dumps(payload), is_error=True)
        try:
            result = self.func(ctx, args)
        except ToolError as exc:
            return ToolOutcome(str(exc), is_error=True)
        except Exception as exc:  # a broken tool should not kill the whole run
            return ToolOutcome(f"{type(exc).__name__}: {exc}", is_error=True)
        return ToolOutcome(truncate_output(result))


def valid_utf8(text: str) -> str:
    """Replace lone surrogates (e.g. from undecodable file names) so the request can be sent."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        # U+DC80..U+DCFF are escaped bytes (os.fsdecode): turn them back into bytes so the
        # decoder can recover what it can; any other lone surrogate becomes U+FFFD.
        cleaned = "".join(
            "\ufffd" if "\ud800" <= char <= "\udc7f" or "\udd00" <= char <= "\udfff" else char
            for char in text
        )
        return cleaned.encode("utf-8", "surrogateescape").decode("utf-8", "replace")
    return text


def truncate_output(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return (
        text[:limit]
        + f"\n\n[Output truncated: showing the first {limit:,} of {len(text):,} characters. "
        "Narrow the request (for example with offset/limit or a more specific path) to see more.]"
    )


def tool(
    name: str | None = None, *, description: str | None = None
) -> Callable[[Callable[[ToolContext, ArgsT], str]], Tool]:
    """Turn ``def fn(ctx: ToolContext, args: SomeArgs) -> str`` into a :class:`Tool`.

    The argument model comes from the type hint of the second parameter and the
    description from the docstring (write it for Claude: say *when* to use the tool).
    """

    def decorator(func: Callable[[ToolContext, ArgsT], str]) -> Tool:
        params = list(inspect.signature(func).parameters)
        if len(params) != 2:
            raise TypeError(f"{func.__name__} must take (ctx, args)")
        try:
            hints = typing.get_type_hints(func)
        except NameError as exc:
            raise TypeError(
                f"{func.__name__}: could not resolve its type hints ({exc}); "
                "define the argument model at module level"
            ) from exc
        args_model = hints.get(params[1])
        if not (isinstance(args_model, type) and issubclass(args_model, BaseModel)):
            raise TypeError(f"{func.__name__}: annotate `{params[1]}` with a pydantic model")
        doc = description or inspect.getdoc(func)
        if not doc:
            raise TypeError(f"{func.__name__} needs a docstring or description")
        return Tool(name or func.__name__, doc, args_model, func)

    return decorator
