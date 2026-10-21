"""Regression tests for defects found by the pre-release review (each was reproduced first)."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from datetime import date, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from typer.testing import CliRunner, Result

from mywork_agents import cli
from mywork_agents.agents.commit import CommitAgent, CommitMessage
from mywork_agents.agents.readme import ReadmeAgent
from mywork_agents.agents.research import ResearchAgent
from mywork_agents.config import Settings, load_settings, project_template, user_config_path
from mywork_agents.core.agent import Agent, AgentContext
from mywork_agents.core.llm import AnthropicLLM, LLMRequest, LLMResponse, Usage
from mywork_agents.core.tools import ToolContext, tool
from mywork_agents.errors import TruncatedOutputError
from mywork_agents.loaders.ics import events_on, local_timezone, parse_ics
from mywork_agents.loaders.mail import load_emails
from mywork_agents.loaders.transcript import clean_transcript
from mywork_agents.models import estimate_cost, model_info
from mywork_agents.testing import FakeLLM

from .conftest import git
from .test_agents import REVIEW

UTC = timezone.utc
IST = ZoneInfo("Asia/Kolkata")
LA = ZoneInfo("America/Los_Angeles")
runner = CliRunner()
MakeCtx = Callable[..., AgentContext]


def vcal(*events: str) -> str:
    return "BEGIN:VCALENDAR\r\n" + "".join(events) + "END:VCALENDAR\r\n"


def vevent(*lines: str) -> str:
    return "BEGIN:VEVENT\r\n" + "".join(f"{line}\r\n" for line in lines) + "END:VEVENT\r\n"


def summaries(text: str, day: date, tz: Any = UTC) -> list[str]:
    return [
        f"{o.start:%H:%M} {o.summary}" if o.start else o.summary
        for o in events_on(parse_ics(text, tz), day, tz)
    ]


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    llm = FakeLLM()
    monkeypatch.setattr(cli, "make_llm", lambda settings, model: llm)
    return llm


def invoke(*args: str, input: str | None = None) -> Result:
    return runner.invoke(cli.app, list(args), input=input)


# ===================================================================== core


def test_encoding_errors_are_not_retried_as_malformed_tool_input() -> None:
    class Stream:
        def __enter__(self) -> Any:
            raise UnicodeEncodeError("utf-8", "\udce9", 0, 1, "surrogates not allowed")

        def __exit__(self, *args: object) -> None:
            pass

    class Client:
        class messages:  # mimics the SDK's client.messages attribute
            @staticmethod
            def stream(**kwargs: Any) -> Stream:
                return Stream()

    llm = AnthropicLLM("claude-haiku-5-5", client=Client())  # type: ignore[arg-type]
    request = LLMRequest(system="s", messages=[{"role": "user", "content": "x"}], max_tokens=10)
    with pytest.raises(UnicodeEncodeError):
        llm.complete(request)


class NameArgs(BaseModel := __import__("pydantic").BaseModel):  # type: ignore[misc]
    pass


@tool()
def odd_name(ctx: ToolContext, args: NameArgs) -> str:
    """Return a file name that is not valid UTF-8."""
    return "caf\udce9.txt and \ud800"


def test_tool_output_is_always_valid_utf8(tmp_path: Path) -> None:
    outcome = odd_name.invoke(ToolContext(workspace=tmp_path, settings=Settings()), {})
    outcome.content.encode("utf-8")  # must not raise
    assert outcome.content == "caf�.txt and �"


def test_dated_model_ids_inherit_capabilities_and_prices() -> None:
    dated = model_info("claude-haiku-4-5-20251001")
    assert dated.adaptive_thinking is False and dated.effort is False
    assert dated.input_per_mtok == 1.0
    params = AnthropicLLM("claude-haiku-4-5-20251001").build_params(
        LLMRequest(system="s", messages=[], max_tokens=10, effort="low")
    )
    assert "thinking" not in params and "output_config" not in params


def test_context_window_overflow_is_not_a_finished_answer(make_ctx: MakeCtx) -> None:
    class TextAgent(Agent[str]):
        name = "t"
        title = "T"
        description = "d"
        instructions = "i"

    partial = FakeLLM.text("## TL;DR\n- The answer is", stop_reason="model_context_window_exceeded")
    with pytest.raises(TruncatedOutputError, match="context window"):
        TextAgent().run(make_ctx(FakeLLM([partial])), "q")


def test_cost_is_estimated_per_request_with_long_prompt_tiers(make_ctx: MakeCtx) -> None:
    long = Usage(input_tokens=150_000, output_tokens=10_000)
    assert estimate_cost("claude-haiku-5-5", long) == pytest.approx(0.075 + 0.025)
    short = Usage(input_tokens=50_000, output_tokens=10_000)
    assert estimate_cost("claude-haiku-5-5", short) == pytest.approx(0.005 + 0.005)

    class TextAgent(Agent[str]):
        name = "t"
        title = "T"
        description = "d"
        instructions = "i"

    llm = FakeLLM([FakeLLM.text("a", usage=Usage(input_tokens=150_000, output_tokens=10_000))])
    llm.model = "claude-haiku-5-5"
    result = TextAgent().run(make_ctx(llm), "q")
    assert result.usage.estimated_cost == pytest.approx(0.1)

    unknown = FakeLLM([FakeLLM.text("a")], model="some-future-model")
    assert TextAgent().run(make_ctx(unknown), "q").usage.estimated_cost is None


def test_env_vars_outrank_per_agent_file_settings(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (isolated_env / "mywork.toml").write_text(
        '[agents.standup]\nmodel = "claude-haiku-5-5"\neffort = "xhigh"\n'
    )
    monkeypatch.setenv("MYWORK_MODEL", "claude-sonnet-5-5")
    monkeypatch.setenv("MYWORK_EFFORT", "low")
    resolved = load_settings(isolated_env).resolve(
        "standup", default_effort="medium", default_max_tokens=1, default_max_turns=1
    )
    assert (resolved.model, resolved.effort) == ("claude-sonnet-5-5", "low")

    flags = {"model": "claude-opus-5-5", "agents": {"standup": {"model": "claude-opus-5-5"}}}
    resolved = load_settings(isolated_env, overrides=flags).resolve(
        "standup", default_effort="medium", default_max_tokens=1, default_max_turns=1
    )
    assert resolved.model == "claude-opus-5-5"  # CLI flags still win


def test_config_paths_expand_home(isolated_env: Path) -> None:
    (Path.home() / "x.toml").write_text('model = "claude-sonnet-5-5"\n')
    assert load_settings(isolated_env, config_file=Path("~/x.toml")).model == "claude-sonnet-5-5"


def test_project_template_does_not_override_user_config(isolated_env: Path) -> None:
    user = user_config_path()
    user.parent.mkdir(parents=True, exist_ok=True)
    user.write_text(
        'model = "claude-sonnet-5-5"\nhistory = false\n[profile]\nworking_hours = "10:00-19:00"\n'
    )
    assert invoke("config", "init", "--project").exit_code == 0
    settings = load_settings(isolated_env)
    assert (settings.model, settings.history) == ("claude-sonnet-5-5", False)
    assert settings.profile.working_hours == "10:00-19:00"
    assert "# model =" in project_template()


# ================================================================== loaders


def test_recurrence_id_overrides_move_and_cancel_single_occurrences() -> None:
    master = vevent(
        "UID:standup-1",
        "DTSTART;TZID=Asia/Kolkata:20261005T100000",
        "DTEND;TZID=Asia/Kolkata:20261005T101500",
        "RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR",
        "SUMMARY:Team standup",
    )
    moved = vevent(
        "UID:standup-1",
        "RECURRENCE-ID;TZID=Asia/Kolkata:20261020T100000",
        "DTSTART;TZID=Asia/Kolkata:20261020T113000",
        "DTEND;TZID=Asia/Kolkata:20261020T114500",
        "SUMMARY:Team standup",
    )
    cancelled = vevent(
        "UID:standup-1",
        "RECURRENCE-ID;TZID=Asia/Kolkata:20261021T100000",
        "DTSTART;TZID=Asia/Kolkata:20261021T100000",
        "STATUS:CANCELLED",
        "SUMMARY:Team standup",
    )
    text = vcal(master, moved, cancelled)
    assert summaries(text, date(2026, 10, 20), IST) == ["11:30 Team standup"]
    assert summaries(text, date(2026, 10, 21), IST) == []
    assert summaries(text, date(2026, 10, 22), IST) == ["10:00 Team standup"]


def test_until_and_exdate_are_instants_in_the_series_zone() -> None:
    old = vevent(
        "DTSTART;TZID=America/Los_Angeles:20261005T170000",
        "RRULE:FREQ=WEEKLY;WKST=SU;UNTIL=20261019T065959Z;BYDAY=MO",
        "SUMMARY:Old weekly",
    )
    new = vevent(
        "DTSTART;TZID=America/Los_Angeles:20261019T170000",
        "RRULE:FREQ=WEEKLY;BYDAY=MO",
        "SUMMARY:New weekly",
    )
    assert summaries(vcal(old, new), date(2026, 10, 19), LA) == ["17:00 New weekly"]

    tokyo = vevent(
        "DTSTART;TZID=Asia/Tokyo:20261015T080000",
        "RRULE:FREQ=DAILY;UNTIL=20261017T230000Z",
        "SUMMARY:Tokyo daily",
    )
    assert summaries(vcal(tokyo), date(2026, 10, 18), ZoneInfo("Asia/Tokyo")) == [
        "08:00 Tokyo daily"
    ]

    exdate = vevent(
        "DTSTART;TZID=America/Los_Angeles:20261010T170000",
        "RRULE:FREQ=DAILY",
        "EXDATE:20261013T000000Z",  # = Oct 12, 17:00 in Los Angeles
        "SUMMARY:LA daily",
    )
    assert summaries(vcal(exdate), date(2026, 10, 12), LA) == []
    assert summaries(vcal(exdate), date(2026, 10, 13), LA) == ["17:00 LA daily"]


def test_local_timezone_is_a_real_zone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    tz = local_timezone()
    assert isinstance(tz, ZoneInfo) and tz.key == "America/Los_Angeles"


def test_outlook_and_windows_time_zone_names() -> None:
    quoted = vevent(
        'DTSTART;TZID="(UTC-05:00) Eastern Time (US & Canada)":20261019T090000',
        'DTEND;TZID="(UTC-05:00) Eastern Time (US & Canada)":20261019T100000',
        "SUMMARY:Board call",
    )
    windows = vevent(
        "DTSTART;TZID=Pacific Standard Time:20261019T090000",
        "DTEND;TZID=Pacific Standard Time:20261019T100000",
        "SUMMARY:Call with US team",
    )
    assert summaries(vcal(quoted), date(2026, 10, 19), UTC) == ["13:00 Board call"]
    assert summaries(vcal(windows), date(2026, 10, 19), IST) == ["21:30 Call with US team"]


def test_rules_that_cannot_be_expanded_keep_only_the_first_instance() -> None:
    second_tuesday = vevent(
        "DTSTART:20261013T090000Z", "RRULE:FREQ=MONTHLY;BYDAY=TU;BYSETPOS=2", "SUMMARY:Review"
    )
    assert summaries(vcal(second_tuesday), date(2026, 10, 13)) == ["09:00 Review"]
    assert summaries(vcal(second_tuesday), date(2026, 11, 13)) == []  # never a wrong date

    last_day = vevent(
        "DTSTART:20261031T090000Z", "RRULE:FREQ=MONTHLY;BYMONTHDAY=-1", "SUMMARY:Close books"
    )
    assert summaries(vcal(last_day), date(2026, 11, 30)) == ["09:00 Close books"]
    assert summaries(vcal(last_day), date(2026, 11, 29)) == []


def test_unfolding_keeps_the_space_before_a_fold() -> None:
    text = vcal(vevent("DTSTART:20261019T090000Z", "SUMMARY:Quarterly planning for ", " Q4 budget"))
    assert summaries(text, date(2026, 10, 19)) == ["09:00 Quarterly planning for Q4 budget"]


def test_multi_day_recurring_events_cover_their_middle_days() -> None:
    text = vcal(
        vevent(
            "DTSTART:20261005T090000Z",
            "DTEND:20261012T090000Z",
            "RRULE:FREQ=WEEKLY;COUNT=2",
            "SUMMARY:On call",
        )
    )
    assert summaries(text, date(2026, 10, 14)) == ["09:00 On call"]  # inside the 2nd week
    assert summaries(text, date(2026, 10, 21)) == []  # COUNT=2: the series has ended


def test_youtube_style_vtt_is_cleaned() -> None:
    vtt = (
        "﻿WEBVTT\nKind: captions\nLanguage: en\n\n"
        "00:00:00.320 --> 00:00:02.000 align:start position:0%\n"
        "welcome<00:00:00.640><c> everyone</c>\n\n"
        "00:00:02.000 --> 00:00:04.000\n"
        "<v Ana>Ready?</v> said Ana. <v Ben>Yes.</v>\n"
    )
    assert clean_transcript(vtt).splitlines() == [
        "welcome everyone",
        "Ana: Ready? said Ana.",
        "Ben: Yes.",
    ]


def test_apple_mail_bundles_and_forwarded_messages(tmp_path: Path) -> None:
    import mailbox

    exports = tmp_path / "exports"
    bundle = exports / "Inbox.mbox"
    bundle.mkdir(parents=True)
    box = mailbox.mbox(bundle / "mbox")
    message = EmailMessage()
    message["Subject"] = "from bundle"
    message.set_content("bundle body")
    box.add(message)
    box.close()
    assert [e.subject for e in load_emails([exports])] == ["from bundle"]
    assert [e.subject for e in load_emails([bundle])] == ["from bundle"]

    inner = EmailMessage()
    inner["From"] = "boss@example.com"
    inner["Subject"] = "inner"
    inner.set_content("inner body text")
    outer = EmailMessage()
    outer["Subject"] = "Fwd: inner"
    outer.add_attachment(inner)
    fwd = tmp_path / "fwd.eml"
    fwd.write_bytes(bytes(outer))
    body = load_emails([fwd])[0].body
    assert "Forwarded message" in body and "inner body text" in body

    blank_plain = EmailMessage()
    blank_plain["Subject"] = "invoice"
    blank_plain.set_content("")
    blank_plain.add_alternative("<p>Your invoice is overdue</p>", subtype="html")
    alt = tmp_path / "alt.eml"
    alt.write_bytes(bytes(blank_plain))
    assert load_emails([alt])[0].body == "Your invoice is overdue"


# ====================================================================== CLI


def test_rich_markup_in_model_text_does_not_crash(fake: FakeLLM, tmp_path: Path) -> None:
    fake.add(
        FakeLLM.tool_use("search_files", {"pattern": r"split\([/\\]"}),
        FakeLLM.text("## TL;DR\n- [/INST] is a Llama 2 marker"),
    )
    result = invoke("research", "What does [/INST] mean?", "--no-search")
    assert result.exit_code == 0, result.output
    assert "search_files(pattern=" in result.stderr  # shown literally, no MarkupError
    listing = invoke("history", "list")
    assert listing.exit_code == 0
    assert "[/INST]" in listing.stdout


def test_review_paths_are_relative_to_the_current_directory(
    fake: FakeLLM, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (repo / "src" / "api").mkdir(parents=True)
    (repo / "src" / "api" / "handler.py").write_text("x = 1\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "add handler")
    (repo / "src" / "api" / "handler.py").write_text("x = 2\n")
    (repo / "calc.py").write_text("def add(a, b):\n    return b + a\n")
    monkeypatch.chdir(repo / "src")
    fake.add(FakeLLM.json(REVIEW))
    result = invoke("review", "api")
    assert result.exit_code == 0, result.output
    task = fake.requests[0].messages[0]["content"]
    assert "src/api/handler.py" in task and "calc.py" not in task


def test_review_diff_reads_files_from_the_repo_root(
    fake: FakeLLM, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (repo / "sub").mkdir()
    monkeypatch.chdir(repo / "sub")
    fake.add(FakeLLM.tool_use("read_file", {"path": "calc.py"}), FakeLLM.json(REVIEW))
    result = invoke("review", "--diff", "-", input="--- a/calc.py\n+++ b/calc.py\n+x\n")
    assert result.exit_code == 0, result.output
    assert "def add" in fake.requests[1].messages[2]["content"][0]["content"]


def test_review_warns_about_untracked_files(fake: FakeLLM, repo: Path) -> None:
    (repo / "newmodule.py").write_text("y = 1\n")
    (repo / ".env").write_text("SECRET=1\n")
    result = invoke("review", "--repo", str(repo))
    assert result.exit_code == 1
    assert "1 untracked file(s) are not included (newmodule.py)" in result.stderr


def test_history_write_failure_keeps_the_result(
    fake: FakeLLM, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    monkeypatch.setenv("MYWORK_HISTORY_DIR", str(blocker / "history"))
    fake.add(FakeLLM.text("## TL;DR\n- ok"))
    result = invoke("research", "q")
    assert result.exit_code == 0, result.output
    assert "## TL;DR" in result.stdout
    assert "Could not save this run to history" in result.stderr


def test_standup_handles_new_repos_same_names_and_zones(
    fake: FakeLLM, tmp_path: Path, isolated_env: Path
) -> None:
    repos = []
    for parent in ("a", "b"):
        path = tmp_path / parent / "api"
        path.mkdir(parents=True)
        git(path, "init", "-q")
        repos += ["-r", str(path)]
    (isolated_env / "mywork.toml").write_text('[profile]\ntimezone = "Asia/Kolkata"\n')
    result = invoke("standup", *repos, "--dry-run", "--json")
    assert result.exit_code == 0, result.output
    task = json.loads(result.stdout)["messages"][0]["content"]
    assert '<commits repo="api">' in task and '<commits repo="b/api">' in task
    assert "+0530." in task  # the default --since carries the profile's offset


def test_research_depth_beats_config_effort_and_lists_given_urls(
    fake: FakeLLM, isolated_env: Path
) -> None:
    (isolated_env / "mywork.toml").write_text('effort = "low"\n')
    fake.add(FakeLLM.text("summary"), FakeLLM.text("summary"))
    result = invoke(
        "research", "Summarise this RFC", "--depth", "deep", "--url", "https://example.com/rfc"
    )
    assert result.exit_code == 0, result.output
    assert fake.requests[0].effort == "high"
    assert "[https://example.com/rfc](https://example.com/rfc)" in result.stdout
    assert fake.requests[0].messages[0]["content"].count("Research question: Summarise") == 1

    invoke("research", "q", "--depth", "quick", "--effort", "max")
    assert fake.requests[1].effort == "max"


def test_commit_apply_shows_git_summary_literally(fake: FakeLLM, repo: Path) -> None:
    (repo / "calc.py").write_text("def add(a, b):\n    return b + a\n")
    git(repo, "add", "calc.py")
    message = {
        "type": "refactor",
        "scope": None,
        "subject": "handle __init__ files",
        "body": None,
        "breaking_change": None,
    }
    fake.add(FakeLLM.json(message))
    result = invoke("commit", "--repo", str(repo), "--apply", "--yes")
    assert result.exit_code == 0, result.output
    assert "[main " in result.stderr
    assert result.stdout.strip() == "refactor: handle __init__ files"
    assert CommitAgent.markdown_output is False


@pytest.mark.skipif(sys.platform != "linux", reason="needs a filesystem with byte file names")
def test_undecodable_file_names_do_not_break_requests(tmp_path: Path) -> None:
    from mywork_agents.tools.files import list_dir

    try:
        (tmp_path / os.fsdecode(b"caf\xe9.txt")).write_text("x")
    except (OSError, UnicodeError):
        pytest.skip("filesystem refuses non-UTF-8 names")
    outcome = list_dir.invoke(ToolContext(workspace=tmp_path, settings=Settings()), {})
    outcome.content.encode("utf-8")


# =================================================================== agents


def test_commit_wrapping_keeps_urls_lists_code_and_trailers() -> None:
    url = "https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api"
    body = (
        "Clients were dropping requests when the API rate limited them, which caused data loss.\n\n"
        f"See {url} for the headers.\n\n"
        "1. Read Retry-After and wait for the number of seconds it gives before retrying again\n\n"
        "    retry(after=parse(headers))\n\n"
        "Fixes #42\nCo-authored-by: Sam Lee <sam@example.com>"
    )
    text = CommitMessage(
        type="fix", scope="http", subject="retry on 429", body=body, breaking_change=None
    ).format()
    lines = text.splitlines()
    assert url in lines
    assert "    retry(after=parse(headers))" in lines
    assert lines[-2:] == ["Fixes #42", "Co-authored-by: Sam Lee <sam@example.com>"]
    assert any(line.startswith("  ") and "retrying" in line for line in lines)  # list indent


def test_readme_fences() -> None:
    def finalize(text: str) -> str:
        return ReadmeAgent().finalize(
            LLMResponse(content=[], stop_reason="end_turn", text=text), []
        )

    assert finalize("````markdown\n# P\n\n```bash\nx\n```\n````") == "# P\n\n```bash\nx\n```\n"
    starts_and_ends_with_code = "```\n  ascii art\n```\n\n# P\n\n```\npip install p\n```"
    assert finalize(starts_and_ends_with_code) == starts_and_ends_with_code + "\n"
    assert finalize("```\n# P\n\nplain\n```") == "# P\n\nplain\n"


def test_research_question_is_not_wrapped_as_data() -> None:
    task = ResearchAgent().build_task("Compare Celery and RQ in a table")
    assert task.startswith("Research question: Compare Celery and RQ")
    assert "<question>" not in task
