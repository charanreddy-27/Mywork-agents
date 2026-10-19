"""Daily planner: a realistic, time-blocked plan from tasks, notes and the calendar."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from mywork_agents.core.agent import Agent
from mywork_agents.render import bullets, join_sections, section, table


class Priority(BaseModel):
    title: str
    why: str = Field(description="Why it matters today, in one sentence.")


class TimeBlock(BaseModel):
    start: str = Field(description="Start time, 24-hour HH:MM.")
    end: str = Field(description="End time, 24-hour HH:MM.")
    activity: str
    kind: Literal["focus", "meeting", "collaboration", "admin", "break", "buffer"]


class DeferredTask(BaseModel):
    task: str
    reason: str


class DailyPlan(BaseModel):
    date: str = Field(description="The plan's date, ISO format.")
    focus: str = Field(description="The theme of the day in one sentence.")
    top_priorities: list[Priority] = Field(
        description="At most three outcomes that would make the day a success."
    )
    schedule: list[TimeBlock] = Field(description="Chronological time blocks for the day.")
    quick_wins: list[str] = Field(description="Small tasks to batch into gaps (under 15 minutes).")
    deferred: list[DeferredTask] = Field(description="Tasks consciously not done today.")
    risks: list[str] = Field(description="Conflicts, over-commitment or dependencies to watch.")


INSTRUCTIONS = """\
You are a pragmatic planning coach. Build a realistic plan for the user's day from their task \
list, notes, calendar and recent history.

What a good plan looks like:
- At most three top priorities, chosen for impact and urgency, not for ease.
- Calendar events are fixed: schedule around them and never move them.
- The most demanding priority gets a protected focus block of 60-120 minutes, early in the day \
unless the calendar prevents it. Shallow work (email, admin) is batched, not scattered.
- Leave slack: fill at most about 75% of free time, include short breaks and a buffer before \
the end of the working day. Stay inside the user's working hours.
- Carry over unfinished priorities and promises from earlier plans and standups when they \
still matter, and say so in "why".
- If there is more work than time, put the rest in "deferred" with an honest reason rather \
than cramming it in.
- Flag risks such as back-to-back meetings, deadlines at risk, or tasks blocked on others."""


class PlanAgent(Agent[DailyPlan]):
    name = "plan"
    title = "Daily plan"
    description = "Turn tasks, notes and your calendar into a realistic time-blocked day."
    group = "Planning"
    instructions = INSTRUCTIONS
    output_model = DailyPlan

    def build_task(
        self,
        *,
        day: date,
        tasks: str = "",
        notes: str = "",
        calendar: str = "",
        previous: str = "",
    ) -> str:
        parts = [f"Plan my day for {day:%A, %d %B %Y} ({day.isoformat()})."]
        parts.append(
            f"<calendar>\n{calendar.strip() or 'No calendar events provided.'}\n</calendar>"
        )
        if tasks.strip():
            parts.append(f"<tasks>\n{tasks.strip()}\n</tasks>")
        if notes.strip():
            parts.append(f"<notes>\n{notes.strip()}\n</notes>")
        if previous.strip():
            parts.append(f"<previous_runs>\n{previous.strip()}\n</previous_runs>")
        if not (tasks.strip() or notes.strip() or calendar.strip()):
            parts.append(
                "No tasks, notes or calendar were provided: build the best plan you can from "
                "previous runs and say clearly what information would improve it."
            )
        return "\n\n".join(parts)

    def task_from_text(self, text: str) -> str:
        return f"Plan my day.\n\n<tasks>\n{text.strip()}\n</tasks>"

    def render(self, output: DailyPlan) -> str:
        priorities = "\n".join(
            f"{i}. **{p.title}** — {p.why}" for i, p in enumerate(output.top_priorities, 1)
        )
        schedule = table(
            ["Time", "Block", "Type"],
            [[f"{b.start}-{b.end}", b.activity, b.kind] for b in output.schedule],
        )
        deferred = bullets(f"{d.task} — _{d.reason}_" for d in output.deferred)
        return join_sections(
            f"# Plan for {output.date}",
            f"**Focus:** {output.focus}",
            section("Top priorities", priorities or "_None_"),
            section("Schedule", schedule),
            section("Quick wins", bullets(output.quick_wins)),
            section("Deferred", deferred),
            section("Watch out for", bullets(output.risks)),
        )
