"""Progress events an agent emits while it runs (rendered live by the CLI)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TurnStarted:
    turn: int


@dataclass(frozen=True)
class ToolStarted:
    name: str
    input: Any


@dataclass(frozen=True)
class ToolFinished:
    name: str
    ok: bool
    summary: str


@dataclass(frozen=True)
class ServerToolUsed:
    name: str
    input: Any


@dataclass(frozen=True)
class Retrying:
    reason: str


Event = TurnStarted | ToolStarted | ToolFinished | ServerToolUsed | Retrying


def describe_call(name: str, args: Any, limit: int = 80) -> str:
    """Compact one-line rendering of a tool call, e.g. ``read_file(path='src/app.py')``."""
    if isinstance(args, dict):
        parts = [f"{key}={value!r}" for key, value in args.items()]
        text = f"{name}({', '.join(parts)})"
    else:
        text = f"{name}()"
    return text if len(text) <= limit else text[: limit - 1] + "…"
