"""Give agents read access to earlier runs, so today's work builds on yesterday's."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from mywork_agents.core.tools import ToolContext, tool


class RecentRunsArgs(BaseModel):
    agents: list[str] = Field(
        default_factory=lambda: ["plan", "standup"],
        description="Agents whose past outputs to return, e.g. ['plan', 'standup', 'meeting'].",
    )
    days: int = Field(default=3, ge=1, le=30, description="How many days back to look.")
    limit: int = Field(default=3, ge=1, le=10, description="Maximum number of runs to return.")


@tool()
def recent_runs(ctx: ToolContext, args: RecentRunsArgs) -> str:
    """Read the user's recent mywork runs (earlier plans, standups, meeting notes), newest first.

    Use this to carry work over from previous days: unfinished priorities, follow-ups and
    commitments the user made.
    """
    if ctx.history is None:
        return "Run history is disabled, so there are no earlier runs to read."
    found = ctx.history.recent_markdown(
        args.agents, now=datetime.now().astimezone(), days=args.days, limit=args.limit
    )
    return found or f"No runs of {', '.join(args.agents)} in the last {args.days} days."
