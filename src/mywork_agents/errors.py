"""Exception hierarchy. Every message is written to be shown to the user as-is."""

from __future__ import annotations


class MyworkError(Exception):
    """Base class for all errors raised by mywork-agents."""


class ConfigError(MyworkError):
    """The configuration file or environment is invalid."""


class InputError(MyworkError):
    """The user supplied input the agent cannot work with (missing file, empty diff, ...)."""


class ToolError(MyworkError):
    """A tool failed. Raised inside tools; the agent loop reports it back to Claude."""


class AgentError(MyworkError):
    """An agent run could not complete."""


class RefusalError(AgentError):
    """Claude declined the request."""

    def __init__(self, category: str | None) -> None:
        self.category = category
        detail = f" (category: {category})" if category else ""
        super().__init__(f"Claude declined this request{detail}.")


class TruncatedOutputError(AgentError):
    """The response hit max_tokens before it was complete."""


class MaxTurnsExceededError(AgentError):
    """The agent used every turn it was allowed without finishing."""


class OutputValidationError(AgentError):
    """Claude's final answer did not match the agent's output schema."""
