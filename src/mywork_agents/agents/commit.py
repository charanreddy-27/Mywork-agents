"""Commit message from the staged diff, matching the repository's conventions."""

from __future__ import annotations

import textwrap
from typing import Literal

from pydantic import BaseModel, Field

from mywork_agents.core.agent import Agent

CommitStyle = Literal["conventional", "plain"]


class CommitMessage(BaseModel):
    type: Literal[
        "feat", "fix", "docs", "style", "refactor", "perf", "test", "build", "ci", "chore", "revert"
    ]
    scope: str | None = Field(description="Optional short scope such as 'auth' or 'cli'.")
    subject: str = Field(description="Imperative mood, no trailing period, at most 60 characters.")
    body: str | None = Field(
        description="Why the change was made and anything notable; null for trivial changes."
    )
    breaking_change: str | None = Field(
        description="What breaks for users and how to migrate; null if nothing breaks."
    )

    def format(self, style: CommitStyle = "conventional") -> str:
        subject = self.subject.strip().rstrip(".")
        if style == "conventional":
            scope = f"({self.scope})" if self.scope else ""
            bang = "!" if self.breaking_change else ""
            header = f"{self.type}{scope}{bang}: {subject}"
        else:
            header = subject[:1].upper() + subject[1:]
        parts = [header]
        if self.body and self.body.strip():
            parts.append(_wrap(self.body))
        if self.breaking_change:
            parts.append(_wrap(f"BREAKING CHANGE: {self.breaking_change}"))
        return "\n\n".join(parts) + "\n"


def _wrap(text: str, width: int = 72) -> str:
    paragraphs = []
    for paragraph in text.strip().split("\n\n"):
        lines = paragraph.splitlines()
        if any(line.lstrip().startswith(("-", "*")) for line in lines):
            paragraphs.append(
                "\n".join(
                    textwrap.fill(line, width, subsequent_indent="  ") for line in lines if line
                )
            )
        else:
            paragraphs.append(textwrap.fill(" ".join(lines), width))
    return "\n\n".join(paragraphs)


INSTRUCTIONS = """\
You write excellent git commit messages from a staged diff.

The subject says what the change does in the imperative mood ("add", "fix", "remove") and is \
specific enough to be useful in `git log --oneline`. The body explains why the change was made \
and anything a future reader would not get from the diff: the problem, the approach and its \
trade-offs, follow-up work. Leave the body empty for trivial changes. Do not narrate the diff \
file by file, and do not mention tests or formatting unless that is the point of the change.

Pick the type that best describes the user-visible effect. Match the conventions visible in the \
repository's recent commit subjects (scope naming, tone). If the diff mixes unrelated changes, \
describe the main one and mention in the body that it should be split."""


class CommitAgent(Agent[CommitMessage]):
    name = "commit"
    title = "Commit message"
    description = "Write a commit message for your staged changes (and optionally commit)."
    group = "Development"
    instructions = INSTRUCTIONS
    output_model = CommitMessage
    default_effort = "low"
    default_max_tokens = 16_000

    def __init__(self, *, style: CommitStyle = "conventional") -> None:
        self.style = style

    def build_task(
        self, diff: str, *, recent_subjects: list[str] | None = None, hint: str | None = None
    ) -> str:
        parts = ["Write a commit message for this staged diff."]
        if hint:
            parts.append(f"Context from the author: {hint}")
        if recent_subjects:
            listed = "\n".join(f"- {s}" for s in recent_subjects)
            parts.append(f"<recent_commits>\n{listed}\n</recent_commits>")
        parts.append(f"<diff>\n{diff.strip()}\n</diff>")
        return "\n\n".join(parts)

    def task_from_text(self, text: str) -> str:
        return self.build_task(text)

    def render(self, output: CommitMessage) -> str:
        return output.format(self.style)
