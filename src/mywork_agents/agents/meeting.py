"""Meeting notes: summary, decisions, owned action items and a follow-up email."""

from __future__ import annotations

from pydantic import BaseModel, Field

from mywork_agents.core.agent import Agent
from mywork_agents.loaders.transcript import clean_transcript
from mywork_agents.render import bullets, join_sections, quote, section, table


class ActionItem(BaseModel):
    task: str = Field(description="What needs to happen, starting with a verb.")
    owner: str | None = Field(description="Who committed to it, if anyone did.")
    due: str | None = Field(description="Due date if one was agreed (ISO when possible).")


class MeetingNotes(BaseModel):
    title: str = Field(description="A short descriptive title for the meeting.")
    date: str | None = Field(description="Meeting date if known (ISO).")
    attendees: list[str]
    summary: str = Field(description="Three to five sentences a non-attendee could act on.")
    key_points: list[str]
    decisions: list[str] = Field(description="Only decisions that were actually made.")
    action_items: list[ActionItem]
    open_questions: list[str] = Field(description="Unresolved questions and parked topics.")
    follow_up_email: str = Field(
        description="A concise follow-up email to attendees: summary, decisions, action items."
    )


INSTRUCTIONS = """\
You turn raw meeting transcripts and notes into crisp, trustworthy meeting notes.

Be faithful to what was said. A decision is only a decision if the group agreed to it; an idea \
that was floated belongs in key points or open questions. Assign an action item owner only when \
someone took it on or was clearly asked and accepted; otherwise leave the owner empty rather \
than guessing. Resolve relative dates ("next Tuesday") against the meeting date if it is known, \
otherwise against the current date.

Keep the notes short enough to read in a minute: merge repetitive discussion, drop small talk, \
and keep names and numbers exact. The follow-up email should be ready to send, written from the \
user's perspective, with a subject line on its first line ("Subject: ...")."""


class MeetingAgent(Agent[MeetingNotes]):
    name = "meeting"
    title = "Meeting notes"
    description = "Turn a transcript into decisions, owned action items and a follow-up email."
    group = "Communication"
    instructions = INSTRUCTIONS
    output_model = MeetingNotes

    def build_task(self, transcript: str, *, title: str | None = None) -> str:
        hint = f"Meeting title: {title}\n\n" if title else ""
        return (
            f"{hint}Write meeting notes for this transcript.\n\n"
            f"<transcript>\n{clean_transcript(transcript)}\n</transcript>"
        )

    def task_from_text(self, text: str) -> str:
        return self.build_task(text)

    def render(self, output: MeetingNotes) -> str:
        meta = " · ".join(
            part
            for part in (
                output.date or "",
                f"Attendees: {', '.join(output.attendees)}" if output.attendees else "",
            )
            if part
        )
        actions = (
            table(
                ["Owner", "Action", "Due"],
                [[a.owner or "Unassigned", a.task, a.due or ""] for a in output.action_items],
            )
            if output.action_items
            else "_None_"
        )
        return join_sections(
            f"# {output.title}",
            f"_{meta}_" if meta else "",
            output.summary,
            section("Decisions", bullets(output.decisions)),
            section("Action items", actions),
            section("Key points", bullets(output.key_points)),
            section("Open questions", bullets(output.open_questions)),
            section("Follow-up email", quote(output.follow_up_email)),
        )
