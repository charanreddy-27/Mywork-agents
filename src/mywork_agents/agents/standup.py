"""Standup update written from your git activity, notes and yesterday's plan."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

from pydantic import BaseModel, Field

from mywork_agents.core.agent import Agent
from mywork_agents.render import bullets, join_sections


class Standup(BaseModel):
    yesterday: list[str] = Field(description="What got done since the last standup, as outcomes.")
    today: list[str] = Field(description="What the user will focus on today.")
    blockers: list[str] = Field(description="Things blocking progress; empty if none.")
    summary: str = Field(description="A one-line TL;DR for a status channel.")


INSTRUCTIONS = """\
You write the user's daily standup update from evidence: their commits, notes and recent plans.

Describe outcomes, not activity ("Shipped retry logic for webhook delivery", not "worked on \
webhooks"), grouping related commits into one bullet and translating commit jargon into plain \
language a teammate outside the code would follow. Keep each bullet to one line and the whole \
update readable in under 30 seconds. For "today", continue unfinished work and the priorities \
from the most recent plan. Only list blockers that are supported by the notes or plans; never \
invent progress, tickets or blockers. If there is little evidence, keep the update short and \
honest."""


def previous_workday_start(today: date) -> datetime:
    """Start of the previous working day (Friday when today is Saturday, Sunday or Monday)."""
    back = {0: 3, 6: 2, 5: 1}.get(today.weekday(), 1)
    return datetime.combine(today - timedelta(days=back), time.min)


class StandupAgent(Agent[Standup]):
    name = "standup"
    title = "Standup update"
    description = "Write yesterday / today / blockers from git commits, notes and past plans."
    group = "Planning"
    instructions = INSTRUCTIONS
    output_model = Standup
    default_effort = "low"

    def build_task(
        self,
        *,
        since_label: str,
        commits: dict[str, str],
        notes: str = "",
        previous: str = "",
    ) -> str:
        parts = [f"Write my standup update. It covers activity since {since_label}."]
        if commits:
            for repo, log in commits.items():
                body = log.strip() or "No commits in this period."
                parts.append(f'<commits repo="{repo}">\n{body}\n</commits>')
        else:
            parts.append("No git repositories were scanned.")
        if notes.strip():
            parts.append(f"<notes>\n{notes.strip()}\n</notes>")
        if previous.strip():
            parts.append(f"<previous_runs>\n{previous.strip()}\n</previous_runs>")
        return "\n\n".join(parts)

    def task_from_text(self, text: str) -> str:
        return f"Write my standup update.\n\n<notes>\n{text.strip()}\n</notes>"

    def render(self, output: Standup) -> str:
        return join_sections(
            f"_{output.summary}_",
            "**Yesterday**\n\n" + bullets(output.yesterday),
            "**Today**\n\n" + bullets(output.today),
            "**Blockers**\n\n" + bullets(output.blockers, empty="- None"),
        )
