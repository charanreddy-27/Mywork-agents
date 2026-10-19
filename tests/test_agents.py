"""Each built-in agent end to end with FakeLLM: prompt building, tools and rendering."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from mywork_agents.agents import (
    BUILTIN_AGENTS,
    CommitAgent,
    InboxAgent,
    MeetingAgent,
    PlanAgent,
    ReadmeAgent,
    ResearchAgent,
    ReviewAgent,
    StandupAgent,
    discover_agents,
    get_agent_class,
)
from mywork_agents.agents.commit import CommitMessage
from mywork_agents.agents.review import CodeReview, worst_severity
from mywork_agents.agents.standup import previous_workday_start
from mywork_agents.core.agent import Agent, AgentContext
from mywork_agents.core.llm import Source
from mywork_agents.errors import InputError
from mywork_agents.loaders.mail import load_emails
from mywork_agents.testing import FakeLLM

from .conftest import EXAMPLES

MakeCtx = Callable[..., AgentContext]


def run(agent: Agent[Any], make_ctx: MakeCtx, task: str, *responses: Any, **ctx: Any) -> Any:
    llm = FakeLLM(list(responses))
    result = agent.run(make_ctx(llm, **ctx), task)
    return result, llm


def test_every_builtin_agent_has_a_valid_schema_and_metadata(make_ctx: MakeCtx) -> None:
    for name, cls in BUILTIN_AGENTS.items():
        agent = cls()
        assert agent.name == name
        assert agent.description and agent.title and agent.instructions
        preview = agent.preview(make_ctx(FakeLLM()), "x")
        if agent.output_model is not None:
            assert preview["output_schema"]["additionalProperties"] is False


def test_inbox(make_ctx: MakeCtx) -> None:
    agent = InboxAgent()
    task = agent.build_task(load_emails([EXAMPLES / "emails"]))
    assert "Triage these 4 emails." in task

    output = {
        "overview": "Handle the payment incident first.",
        "emails": [
            {
                "ref": "2",
                "subject": "Python Weekly",
                "sender": "newsletter",
                "category": "newsletter",
                "priority": "low",
                "summary": "Weekly links.",
                "action_items": [],
                "deadline": None,
                "suggested_reply": None,
            },
            {
                "ref": "1",
                "subject": "Payment retries failing",
                "sender": "Priya",
                "category": "action_required",
                "priority": "urgent",
                "summary": "EU retries fail.",
                "action_items": ["Decide on rollback by noon"],
                "deadline": "2026-10-19",
                "suggested_reply": "On it, update by 11:30.",
            },
        ],
    }
    result, _ = run(agent, make_ctx, task, FakeLLM.json(output))
    markdown = result.markdown
    assert markdown.index("Payment retries failing") < markdown.index("Python Weekly")
    assert "- [ ] Decide on rollback by noon" in markdown
    assert "> On it, update by 11:30." in markdown
    assert "due **2026-10-19**" in markdown


def test_meeting(make_ctx: MakeCtx) -> None:
    agent = MeetingAgent()
    task = agent.build_task((EXAMPLES / "meeting.vtt").read_text(), title="Release sync")
    assert task.startswith("Meeting title: Release sync")
    assert "-->" not in task

    output = {
        "title": "Release sync",
        "date": None,
        "attendees": ["Priya", "Charan"],
        "summary": "Release depends on the flaky test.",
        "key_points": ["Retry design nearly done"],
        "decisions": ["Do not quarantine payment tests"],
        "action_items": [{"task": "Send design doc", "owner": "Charan", "due": "2026-10-21"}],
        "open_questions": ["Should customers configure retries?"],
        "follow_up_email": "Subject: Release sync notes\n\nHi all, ...",
    }
    result, _ = run(agent, make_ctx, task, FakeLLM.json(output))
    assert "| Charan | Send design doc | 2026-10-21 |" in result.markdown
    assert "> Subject: Release sync notes" in result.markdown


def test_plan(make_ctx: MakeCtx) -> None:
    agent = PlanAgent()
    task = agent.build_task(
        day=date(2026, 10, 22), tasks="- fix test", calendar="- 10:00-10:15 Standup"
    )
    assert "Thursday, 22 October 2026" in task
    assert "<tasks>\n- fix test\n</tasks>" in task
    assert "No tasks, notes or calendar" in agent.build_task(day=date(2026, 10, 22))

    output = {
        "date": "2026-10-22",
        "focus": "Unblock the release.",
        "top_priorities": [{"title": "Fix flaky test", "why": "Blocks release"}],
        "schedule": [
            {"start": "09:00", "end": "10:00", "activity": "Fix flaky test", "kind": "focus"},
            {"start": "10:00", "end": "10:15", "activity": "Standup", "kind": "meeting"},
        ],
        "quick_wins": ["Reply to Lena"],
        "deferred": [{"task": "Blog post", "reason": "Not urgent"}],
        "risks": [],
    }
    result, _ = run(agent, make_ctx, task, FakeLLM.json(output))
    assert "1. **Fix flaky test** — Blocks release" in result.markdown
    assert "| 09:00-10:00 | Fix flaky test | focus |" in result.markdown
    assert "Blog post — _Not urgent_" in result.markdown


def test_standup(make_ctx: MakeCtx) -> None:
    agent = StandupAgent()
    task = agent.build_task(
        since_label="2026-10-16 00:00", commits={"api": "abc  2026-10-16  Dev  fix: retry"}
    )
    assert '<commits repo="api">' in task
    output = {
        "yesterday": ["Fixed retries"],
        "today": ["Ship"],
        "blockers": [],
        "summary": "On track",
    }
    result, llm = run(agent, make_ctx, task, FakeLLM.json(output))
    assert llm.requests[0].effort == "low"
    assert "**Blockers**\n\n- None" in result.markdown


@pytest.mark.parametrize(
    ("today", "expected"),
    [
        (date(2026, 10, 19), date(2026, 10, 16)),  # Monday -> Friday
        (date(2026, 10, 20), date(2026, 10, 19)),  # Tuesday -> Monday
        (date(2026, 10, 25), date(2026, 10, 23)),  # Sunday -> Friday
    ],
)
def test_previous_workday(today: date, expected: date) -> None:
    assert previous_workday_start(today).date() == expected


def test_research_tools_and_sources(make_ctx: MakeCtx) -> None:
    agent = ResearchAgent(depth="deep")
    task = agent.build_task("How should webhooks retry?", urls=["https://example.com/doc"])
    llm = FakeLLM(
        [
            FakeLLM.text(
                "## TL;DR\n- Use backoff",
                cited=[Source("https://a.example", "A")],
                search_results=[Source("https://b.example", "B")],
            )
        ]
    )
    result = agent.run(make_ctx(llm), task)

    request = llm.requests[0]
    assert request.effort == "high"
    assert request.output_schema is None
    assert [t.get("type") for t in request.tools] == ["web_search_20260209", "web_fetch_20260209"]
    assert request.tools[0]["max_uses"] == 15
    assert "1. [A](https://a.example)" in result.markdown
    assert result.output.sources[0].url == "https://a.example"

    offline = ResearchAgent(search=False)
    tools = offline.preview(make_ctx(FakeLLM()), offline.build_task("q"))["tools"]
    assert [t["name"] for t in tools] == ["fetch_url"]

    legacy = ResearchAgent()
    preview = legacy.preview(
        make_ctx(FakeLLM(), settings=_settings(model="claude-haiku-4-5")), legacy.build_task("q")
    )
    assert [t["name"] for t in preview["tools"]] == ["fetch_url", "web_search"]


def _settings(**values: Any) -> Any:
    from mywork_agents.config import Settings

    return Settings.model_validate(values)


REVIEW = {
    "summary": "Adds division.",
    "verdict": "request_changes",
    "findings": [
        {
            "severity": "low",
            "category": "docs",
            "file": "calc.py",
            "line": None,
            "title": "No docstring",
            "explanation": "Missing docs.",
            "suggestion": None,
        },
        {
            "severity": "high",
            "category": "bug",
            "file": "calc.py",
            "line": 6,
            "title": "Division by zero",
            "explanation": "divide(1, 0) raises.",
            "suggestion": "Guard b == 0.",
        },
    ],
    "strengths": ["Small change"],
    "tests_to_add": ["divide by zero"],
}


def test_review_reads_files_and_renders(make_ctx: MakeCtx, tmp_path: Path) -> None:
    (tmp_path / "calc.py").write_text("def divide(a, b):\n    return a / b\n")
    agent = ReviewAgent()
    task = agent.build_task("+def divide(a, b):", context="Add division", focus="errors")
    result, llm = run(
        agent,
        make_ctx,
        task,
        FakeLLM.tool_use("read_file", {"path": "calc.py"}),
        FakeLLM.json(REVIEW),
        workspace=tmp_path,
    )
    tool_result = llm.requests[1].messages[2]["content"][0]
    assert "return a / b" in tool_result["content"]
    assert llm.requests[0].effort == "high"
    markdown = result.markdown
    assert markdown.startswith("# Code review — ❌ Request changes")
    assert markdown.index("Division by zero") < markdown.index("No docstring")
    assert "`calc.py:6` · bug" in markdown
    assert worst_severity(CodeReview.model_validate(REVIEW)) == 3


def test_commit_message_formats() -> None:
    message = CommitMessage(
        type="feat",
        scope="api",
        subject="Add retry headers.",
        body="Clients could not tell when to retry. " * 4,
        breaking_change="Retry-After is now seconds, not a date.",
    )
    text = message.format("conventional")
    header, body, breaking = text.strip().split("\n\n")
    assert header == "feat(api)!: Add retry headers"
    assert all(len(line) <= 72 for line in body.splitlines())
    assert breaking.startswith("BREAKING CHANGE: Retry-After")

    plain = CommitMessage(
        type="fix", scope=None, subject="handle empty input", body=None, breaking_change=None
    )
    assert plain.format("plain") == "Handle empty input\n"


def test_commit_agent(make_ctx: MakeCtx) -> None:
    agent = CommitAgent(style="plain")
    task = agent.build_task("+x", recent_subjects=["fix: a"], hint="faster")
    assert "<recent_commits>\n- fix: a\n</recent_commits>" in task
    assert "Context from the author: faster" in task
    output = {
        "type": "perf",
        "scope": None,
        "subject": "speed up x",
        "body": None,
        "breaking_change": None,
    }
    result, llm = run(agent, make_ctx, task, FakeLLM.json(output))
    assert result.markdown == "Speed up x\n"
    assert llm.requests[0].effort == "low"


def test_readme_strips_wrapping_fence(make_ctx: MakeCtx) -> None:
    agent = ReadmeAgent()
    task = agent.build_task(existing="# Old", notes="mention the CLI")
    assert "<existing_readme>\n# Old\n</existing_readme>" in task
    result, _ = run(agent, make_ctx, task, FakeLLM.text("```markdown\n# Project\n\nHello\n```"))
    assert result.output == "# Project\n\nHello\n"


def test_registry_and_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    class ChangelogAgent(Agent[str]):
        name = "changelog"
        title = "Changelog"
        description = "Write a changelog."
        instructions = "Write changelogs."

    class FakeEntryPoint:
        def __init__(self, name: str, target: Any) -> None:
            self.name = name
            self._target = target

        def load(self) -> Any:
            if isinstance(self._target, Exception):
                raise self._target
            return self._target

    points = [
        FakeEntryPoint("changelog", ChangelogAgent),
        FakeEntryPoint("broken", ImportError("nope")),
        FakeEntryPoint("review", ChangelogAgent),  # cannot shadow a built-in
    ]
    monkeypatch.setattr("mywork_agents.agents.entry_points", lambda group: points)

    with pytest.warns(UserWarning, match="broken"):
        agents = discover_agents()
    assert agents["changelog"] is ChangelogAgent
    assert agents["review"] is ReviewAgent

    with pytest.warns(UserWarning):
        assert get_agent_class("changelog") is ChangelogAgent
    with pytest.raises(InputError, match="Available agents"), pytest.warns(UserWarning):
        get_agent_class("nope")
