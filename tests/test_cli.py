"""The `mywork` CLI, with the Claude API replaced by FakeLLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest
from typer.testing import CliRunner, Result

from mywork_agents import cli
from mywork_agents.config import Settings
from mywork_agents.core.llm import LLMRequest, LLMResponse
from mywork_agents.history import HistoryStore
from mywork_agents.testing import FakeLLM

from .conftest import EXAMPLES, git
from .test_agents import REVIEW

runner = CliRunner()

PLAN = {
    "date": "2026-10-22",
    "focus": "Unblock the release.",
    "top_priorities": [{"title": "Fix flaky test", "why": "Blocks release"}],
    "schedule": [{"start": "09:00", "end": "10:00", "activity": "Fix test", "kind": "focus"}],
    "quick_wins": [],
    "deferred": [],
    "risks": [],
}
STANDUP = {"yesterday": ["Added b"], "today": ["Ship"], "blockers": [], "summary": "On track"}


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    llm = FakeLLM()
    models: list[str] = []

    def make(settings: Settings, model: str) -> FakeLLM:
        models.append(model)
        return llm

    monkeypatch.setattr(cli, "make_llm", make)
    llm.models = models  # type: ignore[attr-defined]
    return llm


def invoke(*args: str, input: str | None = None) -> Result:
    return runner.invoke(cli.app, list(args), input=input)


def script(llm: FakeLLM, *responses: Any) -> None:
    llm.add(*responses)


def test_version_and_agents() -> None:
    assert "mywork-agents 0.1.0" in invoke("--version").stdout
    listing = invoke("agents").stdout
    for name in ("plan", "standup", "inbox", "meeting", "research", "review", "commit", "readme"):
        assert name in listing


def test_plan_end_to_end_saves_history(fake: FakeLLM) -> None:
    script(fake, FakeLLM.json(PLAN))
    result = invoke(
        "plan",
        "--tasks", str(EXAMPLES / "tasks.md"),
        "--notes", str(EXAMPLES / "notes.md"),
        "--calendar", str(EXAMPLES / "calendar.ics"),
        "--date", "2026-10-22",
        "--model", "claude-sonnet-5-5",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("# Plan for 2026-10-22")
    assert "✓ plan · fake-model · 1 turn" in result.stderr
    assert fake.models == ["claude-sonnet-5-5"]  # type: ignore[attr-defined]

    task = fake.requests[0].messages[0]["content"]
    assert "Team standup" in task and "Product sync" in task
    assert "Fix flaky `test_payment_retry`" in task

    # The next plan sees this one through history.
    script(fake, FakeLLM.json(PLAN))
    assert invoke("plan", "--json").exit_code == 0
    assert "<previous_runs>" in fake.requests[1].messages[0]["content"]

    listing = invoke("history", "list")
    assert listing.stdout.count("plan") >= 2
    run_id = listing.stdout.split()[4]
    shown = invoke("history", "show", run_id, "--json")
    assert json.loads(shown.stdout)["agent"] == "plan"


def test_plan_reads_config_defaults_and_json(fake: FakeLLM, isolated_env: Path) -> None:
    (isolated_env / "TODO.md").write_text("- [ ] auto-discovered task\n")
    (isolated_env / "mywork.toml").write_text('history = false\n[profile]\nname = "Asha"\n')
    script(fake, FakeLLM.json(PLAN))
    result = invoke("plan", "--json", "-o", "plan.json")
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["output"]["focus"] == "Unblock the release."
    assert json.loads((isolated_env / "plan.json").read_text()) == data
    assert "auto-discovered task" in fake.requests[0].messages[0]["content"]
    assert "Name: Asha" in fake.requests[0].system
    assert "run " not in result.stderr  # history disabled


def test_dry_run_never_calls_claude(fake: FakeLLM) -> None:
    result = invoke("meeting", str(EXAMPLES / "meeting.vtt"), "--dry-run", "--json")
    assert result.exit_code == 0, result.output
    preview = json.loads(result.stdout)
    assert preview["agent"] == "meeting"
    assert fake.requests == []

    text = invoke(
        "plan", "--calendar", str(EXAMPLES / "calendar.ics"), "--date", "thursday", "--dry-run"
    )
    assert "Dry run · plan" in text.stdout
    assert "[recurring]" in text.stdout


def test_meeting_from_stdin_and_inbox(fake: FakeLLM) -> None:
    notes = {
        "title": "Sync",
        "date": None,
        "attendees": [],
        "summary": "Short.",
        "key_points": [],
        "decisions": [],
        "action_items": [],
        "open_questions": [],
        "follow_up_email": "Subject: Sync",
    }
    script(fake, FakeLLM.json(notes))
    result = invoke("meeting", "-", input="Ana: we agreed to ship.\n")
    assert result.exit_code == 0, result.output
    assert "Ana: we agreed to ship." in fake.requests[0].messages[0]["content"]

    script(fake, FakeLLM.json({"overview": "Quiet.", "emails": []}))
    result = invoke("inbox", str(EXAMPLES / "emails"), "-", input="Subject: pasted\n\nHello")
    assert result.exit_code == 0, result.output
    assert "Triage these 5 emails." in fake.requests[1].messages[0]["content"]


def test_research(fake: FakeLLM) -> None:
    script(fake, FakeLLM.text("## TL;DR\n- yes"))
    result = invoke("research", "is", "this", "useful?", "--depth", "quick")
    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("# is this useful?")
    assert fake.requests[0].effort == "low"

    script(fake, FakeLLM.text("ok"))
    invoke("research", "q", "--effort", "max")
    assert fake.requests[1].effort == "max"  # an explicit flag beats --depth


def test_review_with_fail_on(fake: FakeLLM, repo: Path) -> None:
    (repo / "calc.py").write_text("def divide(a, b):\n    return a / b\n")
    script(fake, FakeLLM.json(REVIEW))
    result = invoke("review", "--repo", str(repo), "--fail-on", "high")
    assert result.exit_code == 1
    assert "Division by zero" in result.stdout
    assert "at or above 'high'" in result.stderr
    assert "return a / b" in fake.requests[0].messages[0]["content"]

    script(fake, FakeLLM.json(REVIEW))
    assert invoke("review", "--repo", str(repo), "--fail-on", "critical").exit_code == 0


def test_review_errors(fake: FakeLLM, repo: Path, tmp_path: Path) -> None:
    empty = invoke("review", "--repo", str(repo))
    assert empty.exit_code == 1
    assert "Nothing to review" in empty.stderr

    not_repo = invoke("review", "--repo", str(tmp_path / "home"))
    assert not_repo.exit_code == 1
    assert "not a git repository" in not_repo.stderr

    script(fake, FakeLLM.json(REVIEW))
    patch = invoke("review", "--diff", "-", input="--- a/x\n+++ b/x\n+print(1)\n")
    assert patch.exit_code == 0, patch.output


def test_commit_apply(fake: FakeLLM, repo: Path) -> None:
    nothing = invoke("commit", "--repo", str(repo))
    assert nothing.exit_code == 1
    assert "Nothing is staged" in nothing.stderr

    (repo / "calc.py").write_text("def add(a, b):\n    return b + a\n")
    git(repo, "add", "calc.py")
    message = {
        "type": "refactor",
        "scope": "calc",
        "subject": "swap operands",
        "body": None,
        "breaking_change": None,
    }
    script(fake, FakeLLM.json(message))
    result = invoke("commit", "--repo", str(repo), "--apply", "--yes")
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == "refactor(calc): swap operands"
    assert git(repo, "log", "-1", "--pretty=%s").strip() == "refactor(calc): swap operands"
    assert "- feat: add calculator" in fake.requests[0].messages[0]["content"]


def test_standup_uses_git_and_history(fake: FakeLLM, repo: Path, isolated_env: Path) -> None:
    store = HistoryStore(Settings().resolved_history_dir)
    script(fake, FakeLLM.json(PLAN))
    invoke("plan")
    assert len(store.entries()) == 1

    script(fake, FakeLLM.json(STANDUP))
    result = invoke("standup", "--repo", str(repo), "--since", "1 year ago")
    assert result.exit_code == 0, result.output
    task = fake.requests[1].messages[0]["content"]
    assert "feat: add calculator" in task
    assert "<previous_runs>" in task and "Plan for 2026-10-22" in task
    assert "**Yesterday**" in result.stdout

    script(fake, FakeLLM.json(STANDUP))
    outside = invoke("standup")
    assert outside.exit_code == 0
    assert "not a git repository" in outside.stderr


def test_readme_write(fake: FakeLLM, tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "README.md").write_text("# Old\n")
    script(fake, FakeLLM.text("# New README\n"))
    result = invoke("readme", str(project), "--write", "--yes")
    assert result.exit_code == 0, result.output
    assert (project / "README.md").read_text() == "# New README\n"
    assert "# Old" in fake.requests[0].messages[0]["content"]


def test_run_generic_agent(fake: FakeLLM) -> None:
    script(fake, FakeLLM.json(STANDUP))
    result = invoke("run", "standup", "fixed the build")
    assert result.exit_code == 0, result.output
    assert "<notes>\nfixed the build\n</notes>" in fake.requests[0].messages[0]["content"]

    unknown = invoke("run", "nope", "x")
    assert unknown.exit_code == 1
    assert "Unknown agent" in unknown.stderr


def _api_error(cls: type[anthropic.APIStatusError], status: int) -> anthropic.APIStatusError:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("boom", response=httpx2.Response(status, request=request), body=None)


@pytest.mark.parametrize(
    ("error", "code", "message"),
    [
        (_api_error(anthropic.AuthenticationError, 401), 2, "credentials were rejected"),
        (_api_error(anthropic.RateLimitError, 429), 2, "Rate limited"),
        (_api_error(anthropic.NotFoundError, 404), 2, "Check the model ID"),
        (
            anthropic.APIConnectionError(
                request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
            ),
            2,
            "Could not reach",
        ),
    ],
)
def test_api_errors_are_friendly(fake: FakeLLM, error: Exception, code: int, message: str) -> None:
    def fail(_: LLMRequest) -> LLMResponse:
        raise error

    script(fake, fail)
    result = invoke("research", "q")
    assert result.exit_code == code
    assert message in result.stderr


def test_refusal_exit_code(fake: FakeLLM) -> None:
    script(fake, LLMResponse(content=[], stop_reason="refusal", refusal_category="bio"))
    result = invoke("research", "q")
    assert result.exit_code == 3
    assert "declined" in result.stderr


def test_input_errors() -> None:
    missing = invoke("meeting", "nope.vtt")
    assert missing.exit_code == 1
    assert "No such file" in missing.stderr
    bad_date = invoke("plan", "--date", "someday", "--dry-run")
    assert bad_date.exit_code == 1
    assert "Unrecognised date" in bad_date.stderr


def test_config_commands(isolated_env: Path) -> None:
    created = invoke("config", "init", "--project")
    assert created.exit_code == 0
    assert (isolated_env / "mywork.toml").is_file()
    assert invoke("config", "init", "--project").exit_code == 1  # no silent overwrite

    shown = invoke("config", "show")
    assert json.loads(shown.stdout)["model"] == "claude-opus-5-5"
    assert "mywork.toml" in shown.stderr

    (isolated_env / "mywork.toml").write_text('effort = "loud"\n')
    broken = invoke("config", "show")
    assert broken.exit_code == 1
    assert "Invalid configuration" in broken.stderr


def test_doctor(monkeypatch: pytest.MonkeyPatch, fake: FakeLLM) -> None:
    missing = invoke("doctor")
    assert missing.exit_code == 1
    assert "set ANTHROPIC_API_KEY" in missing.stdout

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    script(fake, FakeLLM.text("OK"))
    healthy = invoke("doctor", "--ping")
    assert healthy.exit_code == 0, healthy.output
    assert "ANTHROPIC_API_KEY is set" in healthy.stdout
    assert "fake-model replied" in healthy.stdout


def test_history_clear(fake: FakeLLM) -> None:
    script(fake, FakeLLM.json(PLAN))
    invoke("plan")
    assert invoke("history", "clear", "--yes").exit_code == 0
    assert "No runs yet." in invoke("history", "list").stderr
