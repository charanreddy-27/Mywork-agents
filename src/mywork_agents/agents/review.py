"""Code review of a diff, with tools to read the surrounding code before judging it."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from mywork_agents.core.agent import Agent
from mywork_agents.render import bullets, join_sections, section, table
from mywork_agents.tools.files import FILE_TOOLS
from mywork_agents.tools.git import GIT_TOOLS

Severity = Literal["critical", "high", "medium", "low", "nit"]
SEVERITY_RANK: dict[str, int] = {"critical": 4, "high": 3, "medium": 2, "low": 1, "nit": 0}
_ICON = {"critical": "🛑", "high": "🔴", "medium": "🟠", "low": "🔵", "nit": "⚪"}


class Finding(BaseModel):
    severity: Severity
    category: Literal[
        "bug", "security", "performance", "reliability", "maintainability", "tests", "docs", "style"
    ]
    file: str
    line: int | None = Field(description="Line number in the new version of the file, if any.")
    title: str = Field(description="One-line statement of the problem.")
    explanation: str = Field(description="What is wrong and a concrete scenario where it breaks.")
    suggestion: str | None = Field(description="How to fix it; a short code snippet is welcome.")


class CodeReview(BaseModel):
    summary: str = Field(description="What the change does and your overall assessment.")
    verdict: Literal["approve", "comment", "request_changes"]
    findings: list[Finding]
    strengths: list[str] = Field(description="What the change does well (brief).")
    tests_to_add: list[str] = Field(description="Specific test cases worth adding.")


INSTRUCTIONS = """\
You are a senior engineer reviewing a code change. Your job is to catch the problems that \
matter before they ship: correctness bugs, security issues, data loss, broken error handling, \
concurrency and resource leaks, API or schema compatibility breaks, performance cliffs, and \
missing tests for risky logic.

Work like a careful reviewer: use the tools to read the code around each hunk, callers of \
changed functions, and related tests before you report something, so every finding is verified \
rather than guessed. Each finding needs a concrete scenario (inputs or state that lead to the \
failure). Prefer a few high-confidence findings over a long list; skip style preferences unless \
they hide a real problem, and use "nit" sparingly. Comment on unchanged code only when the \
change makes it worse.

Severity: critical = exploitable vulnerability, data loss or outage; high = a bug users will \
hit; medium = a bug in an edge case or a significant maintainability risk; low = minor issue; \
nit = polish. Verdict: request_changes if anything is critical or high, approve if only low \
and nit findings remain, otherwise comment. Report line numbers from the new file version."""


class ReviewAgent(Agent[CodeReview]):
    name = "review"
    title = "Code review"
    description = "Review a diff for bugs, security issues and missing tests."
    group = "Development"
    instructions = INSTRUCTIONS
    output_model = CodeReview
    tools = FILE_TOOLS + GIT_TOOLS
    default_effort = "high"
    default_max_tokens = 64_000
    default_max_turns = 30

    def build_task(self, diff: str, *, context: str | None = None, focus: str | None = None) -> str:
        parts = ["Review this change."]
        if context:
            parts.append(f"<change_description>\n{context.strip()}\n</change_description>")
        if focus:
            parts.append(f"Pay particular attention to: {focus}")
        parts.append(f"<diff>\n{diff.strip()}\n</diff>")
        return "\n\n".join(parts)

    def task_from_text(self, text: str) -> str:
        return self.build_task(text)

    def render(self, output: CodeReview) -> str:
        verdict = {
            "approve": "✅ Approve",
            "comment": "💬 Comment",
            "request_changes": "❌ Request changes",
        }[output.verdict]
        findings = sorted(output.findings, key=lambda f: -SEVERITY_RANK[f.severity])
        overview = (
            table(
                ["", "Severity", "Location", "Issue"],
                [[_ICON[f.severity], f.severity, _location(f), f.title] for f in findings],
            )
            if findings
            else "No issues found."
        )
        details = []
        for finding in findings:
            body = [f"`{_location(finding)}` · {finding.category}", finding.explanation]
            if finding.suggestion:
                body.append(f"**Suggestion:** {finding.suggestion}")
            details.append(
                section(f"{_ICON[finding.severity]} {finding.title}", "\n\n".join(body), 3)
            )
        return join_sections(
            f"# Code review — {verdict}",
            output.summary,
            section("Findings", overview),
            *details,
            section("Strengths", bullets(output.strengths)) if output.strengths else "",
            section("Tests to add", bullets(output.tests_to_add)) if output.tests_to_add else "",
        )


def _location(finding: Finding) -> str:
    return f"{finding.file}:{finding.line}" if finding.line else finding.file


def worst_severity(review: CodeReview) -> int:
    """Rank of the most severe finding (-1 when there are none)."""
    return max((SEVERITY_RANK[f.severity] for f in review.findings), default=-1)
