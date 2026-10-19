"""Inbox triage: priority, category, action items and reply drafts for a batch of emails."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from mywork_agents.core.agent import Agent
from mywork_agents.loaders.mail import Email, emails_to_prompt, from_text
from mywork_agents.render import checkboxes, join_sections, quote, section, table

Priority = Literal["urgent", "high", "normal", "low"]
Category = Literal[
    "action_required",
    "reply_needed",
    "waiting_on_others",
    "fyi",
    "newsletter",
    "promotional",
    "suspicious",
]


class EmailTriage(BaseModel):
    ref: str = Field(description="The email's ref attribute from the input, e.g. '1'.")
    subject: str
    sender: str
    category: Category
    priority: Priority
    summary: str = Field(description="One or two sentences: what it is about and what it needs.")
    action_items: list[str] = Field(description="Concrete tasks for the user; empty if none.")
    deadline: str | None = Field(
        description="Stated or clearly implied deadline (ISO date when possible), else null."
    )
    suggested_reply: str | None = Field(
        description="A ready-to-send reply in the user's voice when one is needed, else null."
    )


class InboxReport(BaseModel):
    overview: str = Field(
        description="Two or three sentences: what needs attention first, and why."
    )
    emails: list[EmailTriage]


INSTRUCTIONS = """\
You are a sharp executive assistant triaging the user's email. Return one entry per email in \
the input, in the same order, identified by its ref.

Judging priority: think about consequence and time-sensitivity for this user. "urgent" means it \
needs action today (a customer, revenue, an outage, or someone senior is blocked on them); \
"high" means this week; "normal" means when convenient; "low" can be skimmed or ignored.

Categories: action_required (they must do something beyond replying), reply_needed (a reply is \
the main action), waiting_on_others (someone else owes the next step), fyi, newsletter, \
promotional, suspicious (phishing signs such as urgent requests for credentials or payment, \
mismatched sender domains, or unexpected links and attachments).

Action items start with a verb and come from the email itself; do not invent work. Resolve \
relative deadlines ("by Friday") against the current date. Draft replies in the user's voice: \
short, specific and ready to send. Never draft replies to suspicious, promotional or newsletter \
mail, and never commit the user to something they have not agreed to; when a decision is \
needed, propose options or ask the question instead."""

_PRIORITY_ORDER = {"urgent": 0, "high": 1, "normal": 2, "low": 3}
_PRIORITY_ICON = {"urgent": "🔴", "high": "🟠", "normal": "🔵", "low": "⚪"}


class InboxAgent(Agent[InboxReport]):
    name = "inbox"
    title = "Inbox triage"
    description = "Prioritise emails, extract action items and draft replies."
    group = "Communication"
    instructions = INSTRUCTIONS
    output_model = InboxReport

    def build_task(self, emails: Sequence[Email]) -> str:
        noun = "email" if len(emails) == 1 else "emails"
        return f"Triage these {len(emails)} {noun}.\n\n{emails_to_prompt(emails)}"

    def task_from_text(self, text: str) -> str:
        return self.build_task([from_text(text)])

    def render(self, output: InboxReport) -> str:
        ordered = sorted(output.emails, key=lambda e: _PRIORITY_ORDER.get(e.priority, 9))
        overview = table(
            ["", "Priority", "Category", "From", "Subject", "Deadline"],
            [
                [
                    _PRIORITY_ICON.get(e.priority, ""),
                    e.priority,
                    e.category.replace("_", " "),
                    e.sender,
                    e.subject,
                    e.deadline or "",
                ]
                for e in ordered
            ],
        )
        details = []
        for email in ordered:
            category = email.category.replace("_", " ").capitalize()
            meta = f"**{category}** · from {email.sender or 'unknown'}"
            if email.deadline:
                meta += f" · due **{email.deadline}**"
            parts = [meta, email.summary]
            if email.action_items:
                parts.append("**Action items**\n\n" + checkboxes(email.action_items))
            if email.suggested_reply:
                parts.append("**Suggested reply**\n\n" + quote(email.suggested_reply))
            icon = _PRIORITY_ICON.get(email.priority, "")
            details.append(
                section(f"{icon} {email.subject or '(no subject)'}", "\n\n".join(parts), 3)
            )
        return join_sections("# Inbox triage", output.overview, overview, *details)
