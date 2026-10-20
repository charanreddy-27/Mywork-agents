"""The ``mywork`` command-line interface.

Results go to stdout (Markdown, or JSON with ``--json``) so they can be piped;
progress, status and errors go to stderr.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, tzinfo
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Annotated, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.text import Text

from mywork_agents._version import __version__
from mywork_agents.agents import BUILTIN_AGENTS, discover_agents, get_agent_class
from mywork_agents.agents.commit import CommitAgent, CommitMessage
from mywork_agents.agents.inbox import InboxAgent
from mywork_agents.agents.meeting import MeetingAgent
from mywork_agents.agents.plan import PlanAgent
from mywork_agents.agents.readme import ReadmeAgent
from mywork_agents.agents.research import ResearchAgent
from mywork_agents.agents.review import SEVERITY_RANK, CodeReview, ReviewAgent, worst_severity
from mywork_agents.agents.standup import StandupAgent, previous_workday_start
from mywork_agents.config import (
    CONFIG_TEMPLATE,
    PROJECT_CONFIG_NAME,
    Settings,
    config_files,
    load_settings,
    user_config_path,
)
from mywork_agents.core.agent import Agent, AgentContext, RunResult
from mywork_agents.core.events import (
    Event,
    Retrying,
    ServerToolUsed,
    ToolFinished,
    ToolStarted,
    TurnStarted,
    describe_call,
)
from mywork_agents.core.llm import LLM, AnthropicLLM, LLMRequest
from mywork_agents.errors import InputError, MyworkError, RefusalError
from mywork_agents.history import HistoryEntry, HistoryStore
from mywork_agents.loaders import ics
from mywork_agents.loaders.mail import Email, from_text, load_emails
from mywork_agents.models import estimate_cost, model_info
from mywork_agents.tools import git

MAX_DIFF_CHARS = 800_000
MAX_CALENDAR_BYTES = 50_000_000  # full calendar exports with years of history get large

app = typer.Typer(
    name="mywork",
    help="🤖 [bold]mywork[/bold] — Claude-powered agents for your everyday work.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
history_app = typer.Typer(help="Browse and manage past runs.", no_args_is_help=True)
config_app = typer.Typer(help="Show or create configuration.", no_args_is_help=True)
app.add_typer(history_app, name="history", rich_help_panel="Housekeeping")
app.add_typer(config_app, name="config", rich_help_panel="Housekeeping")

out = Console()
err = Console(stderr=True)


class EffortChoice(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    xhigh = "xhigh"
    max = "max"


class DepthChoice(str, Enum):
    quick = "quick"
    standard = "standard"
    deep = "deep"


class FailOn(str, Enum):
    never = "never"
    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"


class StyleChoice(str, Enum):
    conventional = "conventional"
    plain = "plain"


@dataclass
class AppState:
    config_file: Path | None = None


@dataclass
class RunOptions:
    model: str | None = None
    effort: EffortChoice | None = None
    as_json: bool = False
    output: Path | None = None
    dry_run: bool = False


_RUN = "Run options"
ModelOpt = Annotated[
    str | None,
    typer.Option(
        "--model", "-m", help="Claude model ID (default: claude-opus-5-5).", rich_help_panel=_RUN
    ),
]
EffortOpt = Annotated[
    EffortChoice | None,
    typer.Option(
        "--effort", "-e", help="How hard Claude thinks (default: per agent).", rich_help_panel=_RUN
    ),
]
JsonOpt = Annotated[
    bool, typer.Option("--json", help="Print the result as JSON.", rich_help_panel=_RUN)
]
OutputOpt = Annotated[
    Path | None,
    typer.Option(
        "--output",
        "-o",
        dir_okay=False,
        help="Also save the result to this file.",
        rich_help_panel=_RUN,
    ),
]
DryRunOpt = Annotated[
    bool,
    typer.Option(
        "--dry-run",
        help="Show the request that would be sent, without calling Claude.",
        rich_help_panel=_RUN,
    ),
]


# ------------------------------------------------------------------- plumbing


def make_llm(settings: Settings, model: str) -> LLM:
    """Create the LLM client for a run (tests replace this with a fake)."""
    return AnthropicLLM(
        model,
        fallbacks=settings.fallbacks,
        timeout=settings.timeout,
        max_retries=settings.max_retries,
    )


def _state(ctx: typer.Context) -> AppState:
    return ctx.obj if isinstance(ctx.obj, AppState) else AppState()


def _load(ctx: typer.Context, agent: str | None = None, opts: RunOptions | None = None) -> Settings:
    overrides: dict[str, Any] = {}
    if opts is not None:
        effort = opts.effort.value if opts.effort else None
        overrides = {"model": opts.model, "effort": effort}
        # CLI flags beat per-agent config, so apply them to the agent's section too.
        agent_overrides = {k: v for k, v in overrides.items() if v is not None}
        if agent and agent_overrides:
            overrides["agents"] = {agent: agent_overrides}
    return load_settings(Path.cwd(), config_file=_state(ctx).config_file, overrides=overrides)


def _history(settings: Settings) -> HistoryStore | None:
    return HistoryStore(settings.resolved_history_dir) if settings.history else None


_warned_timezones: set[str] = set()


def _user_tz(settings: Settings) -> tzinfo:
    name = settings.profile.timezone
    if name:
        try:
            return ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError):
            if name not in _warned_timezones:
                _warned_timezones.add(name)
                err.print(f"[yellow]Unknown time zone {name!r}; using local time.[/]")
    return ics.local_timezone()


def _now(settings: Settings) -> datetime:
    return datetime.now(_user_tz(settings))


def _read_text(source: str | Path, *, what: str = "input") -> str:
    if str(source) == "-":
        if sys.stdin.isatty():
            raise InputError(f"Expected {what} on stdin (pipe something in, or pass a file).")
        return sys.stdin.read()
    path = Path(source).expanduser()
    if not path.is_file():
        raise InputError(f"No such file: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def _read_calendar(source: str) -> str:
    if source.startswith(("http://", "https://", "webcal://")):
        from mywork_agents.tools.web import download

        url = (
            "https://" + source.removeprefix("webcal://")
            if source.startswith("webcal://")
            else source
        )
        try:
            # The user typed this URL, so self-hosted calendars on the LAN are fine.
            return download(url, max_bytes=MAX_CALENDAR_BYTES, allow_private=True)[2]
        except MyworkError as exc:
            raise InputError(f"Could not download the calendar: {exc}") from exc
    return _read_text(source, what="calendar")


def _parse_day(value: str | None, today: date) -> date:
    if value is None or value.lower() == "today":
        return today
    lowered = value.lower()
    if lowered == "tomorrow":
        return today + timedelta(days=1)
    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    if lowered in weekdays:
        return today + timedelta(days=(weekdays.index(lowered) - today.weekday()) % 7)
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise InputError(
            f"Unrecognised date {value!r}; use YYYY-MM-DD, today, tomorrow or a weekday."
        ) from exc


def _human_tokens(count: int) -> str:
    return f"{count / 1000:.1f}k" if count >= 1000 else str(count)


class _Progress:
    """Spinner plus a line per tool call, on stderr."""

    def __init__(self, title: str) -> None:
        self.title = title
        self._status: Status | None = None

    def __enter__(self) -> _Progress:
        self._status = err.status(f"[bold]{self.title}[/bold] · thinking…", spinner="dots")
        self._status.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._status is not None:
            self._status.__exit__(exc_type, exc, tb)

    def handle(self, event: Event) -> None:
        if isinstance(event, TurnStarted):
            if self._status is not None:
                label = "thinking…" if event.turn == 1 else f"thinking (turn {event.turn})…"
                self._status.update(f"[bold]{self.title}[/bold] · {label}")
        elif isinstance(event, ToolStarted):
            err.print(f"  [dim]→ {describe_call(event.name, event.input)}[/dim]", highlight=False)
        elif isinstance(event, ToolFinished) and not event.ok:
            err.print(f"  [yellow]! {event.name}: {event.summary}[/yellow]", highlight=False)
        elif isinstance(event, ServerToolUsed):
            args = event.input if isinstance(event.input, dict) else {}
            if event.name == "web_search" and "query" in args:
                err.print(f"  [dim]🔎 {args['query']}[/dim]", highlight=False)
            elif event.name == "web_fetch" and "url" in args:
                err.print(f"  [dim]🌐 {args['url']}[/dim]", highlight=False)
            else:
                err.print(
                    f"  [dim]→ {describe_call(event.name, event.input)}[/dim]", highlight=False
                )
        elif isinstance(event, Retrying):
            err.print(f"  [yellow]↻ retrying: {event.reason}[/yellow]", highlight=False)


@contextmanager
def _errors() -> Iterator[None]:
    """Turn exceptions into one friendly line on stderr and a meaningful exit code."""
    import anthropic

    try:
        yield
    except typer.Exit:
        raise
    except RefusalError as exc:
        err.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(3) from exc
    except MyworkError as exc:
        err.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(1) from exc
    except anthropic.AuthenticationError as exc:
        err.print(
            "[red]✗[/red] Your Claude API credentials were rejected. Check ANTHROPIC_API_KEY."
        )
        raise typer.Exit(2) from exc
    except anthropic.PermissionDeniedError as exc:
        err.print(f"[red]✗[/red] Permission denied by the Claude API: {exc.message}")
        raise typer.Exit(2) from exc
    except anthropic.NotFoundError as exc:
        err.print(f"[red]✗[/red] Not found: {exc.message}. Check the model ID (--model).")
        raise typer.Exit(2) from exc
    except anthropic.RateLimitError as exc:
        err.print("[red]✗[/red] Rate limited by the Claude API. Wait a moment and try again.")
        raise typer.Exit(2) from exc
    except anthropic.BadRequestError as exc:
        err.print(f"[red]✗[/red] The Claude API rejected the request: {exc.message}")
        raise typer.Exit(2) from exc
    except anthropic.APIStatusError as exc:
        err.print(f"[red]✗[/red] Claude API error {exc.status_code}: {exc.message}")
        raise typer.Exit(2) from exc
    except anthropic.APIConnectionError as exc:
        err.print("[red]✗[/red] Could not reach the Claude API. Check your network connection.")
        raise typer.Exit(2) from exc
    except KeyboardInterrupt as exc:
        err.print("[yellow]Interrupted.[/yellow]")
        raise typer.Exit(130) from exc


def _execute(
    agent: Agent[Any],
    task: str,
    opts: RunOptions,
    settings: Settings,
    *,
    workspace: Path,
    input_preview: str,
) -> RunResult[Any] | None:
    history = _history(settings)
    progress = _Progress(agent.title)
    agent_ctx = AgentContext(
        settings=settings,
        make_llm=lambda model: make_llm(settings, model),
        workspace=workspace.resolve(),
        history=history,
        on_event=progress.handle,
        now=_now(settings),
    )
    if opts.dry_run:
        _print_preview(agent.preview(agent_ctx, task), as_json=opts.as_json)
        return None

    with progress:
        result = agent.run(agent_ctx, task)
    entry = (
        history.save(result, input_preview=input_preview, now=agent_ctx.now) if history else None
    )
    _emit(result, opts)
    _footer(result, entry)
    return result


def _emit(result: RunResult[Any], opts: RunOptions) -> None:
    text = (
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n"
        if opts.as_json
        else result.markdown
    )
    if opts.as_json or not out.is_terminal:
        sys.stdout.write(text if text.endswith("\n") else text + "\n")
        sys.stdout.flush()
    else:
        out.print(Markdown(result.markdown))
    if opts.output:
        opts.output.parent.mkdir(parents=True, exist_ok=True)
        opts.output.write_text(text, encoding="utf-8")
        err.print(f"[green]✓[/green] Saved to {opts.output}")


def _footer(result: RunResult[Any], entry: HistoryEntry | None) -> None:
    usage = result.usage
    parts = [
        f"[bold]{result.agent}[/bold]",
        result.model,
        f"{result.turns} turn{'s' if result.turns != 1 else ''}",
        f"{_human_tokens(usage.total_input_tokens)} in / {_human_tokens(usage.output_tokens)} out",
    ]
    if usage.web_search_requests:
        parts.append(f"{usage.web_search_requests} searches")
    cost = estimate_cost(result.model, usage)
    if cost is not None:
        parts.append(f"≈ ${cost:.3f}")
    parts.append(f"{result.duration_s:.1f}s")
    if entry is not None:
        parts.append(f"run {entry.id}")
    err.print("[green]✓[/green] " + " · ".join(parts), highlight=False)


def _print_preview(preview: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        sys.stdout.write(json.dumps(preview, indent=2, ensure_ascii=False, default=str) + "\n")
        return
    tools = [t.get("name", t.get("type", "?")) for t in preview["tools"]]
    settings = (
        f"model={preview['model']}  effort={preview['effort']}  "
        f"max_tokens={preview['max_tokens']}  max_turns={preview['max_turns']}\n"
        f"tools: {', '.join(tools) or 'none'}\n"
        f"structured output: {'yes' if preview['output_schema'] else 'no (free text)'}"
    )
    # Text(...) keeps prompt content literal: "[recurring]" is not Rich markup here.
    out.print(Panel(Text(settings), title=f"Dry run · {preview['agent']}", border_style="cyan"))
    out.print(Panel(Text(preview["system"]), title="System prompt", border_style="dim"))
    for message in preview["messages"]:
        body = Text(str(message["content"]))
        out.print(Panel(body, title=f"{message['role']} message", border_style="dim"))


# ------------------------------------------------------------------ commands


def _version_callback(value: bool) -> None:
    if value:
        out.print(f"mywork-agents {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    ctx: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config", envvar="MYWORK_CONFIG", help="Use this config file only.", dir_okay=False
        ),
    ] = None,
    version: Annotated[
        bool,
        typer.Option(
            "--version", "-V", callback=_version_callback, is_eager=True, help="Show the version."
        ),
    ] = False,
) -> None:
    """🤖 [bold]mywork[/bold] — Claude-powered agents for your everyday work."""
    ctx.obj = AppState(config_file=config)


@app.command(rich_help_panel="Planning")
def plan(
    ctx: typer.Context,
    tasks: Annotated[
        list[Path] | None,
        typer.Option("--tasks", "-t", help="Task list file(s), e.g. TODO.md. Repeatable."),
    ] = None,
    notes: Annotated[Path | None, typer.Option("--notes", "-n", help="Notes file.")] = None,
    calendar: Annotated[
        str | None,
        typer.Option("--calendar", "--cal", help="An .ics file or iCal URL for today's events."),
    ] = None,
    day: Annotated[
        str | None,
        typer.Option("--date", "-d", help="YYYY-MM-DD, today, tomorrow or a weekday."),
    ] = None,
    model: ModelOpt = None,
    effort: EffortOpt = None,
    as_json: JsonOpt = False,
    output: OutputOpt = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Plan your day: top priorities and time blocks around your calendar."""
    opts = RunOptions(model, effort, as_json, output, dry_run)
    with _errors():
        settings = _load(ctx, "plan", opts)
        now = _now(settings)
        target = _parse_day(day, now.date())

        task_files = [str(p) for p in tasks] if tasks else list(settings.files.tasks)
        if not task_files:
            task_files = [
                n for n in ("TODO.md", "todo.md", "tasks.md", "TASKS.md") if Path(n).is_file()
            ][:1]
        tasks_text = "\n\n".join(_read_text(p, what="tasks") for p in task_files)
        notes_source = str(notes) if notes else settings.files.notes
        notes_text = _read_text(notes_source, what="notes") if notes_source else ""

        calendar_source = calendar or settings.files.calendar
        calendar_text = ""
        if calendar_source:
            tz = _user_tz(settings)
            events = ics.parse_ics(_read_calendar(calendar_source), tz)
            calendar_text = (
                ics.format_occurrences(ics.events_on(events, target, tz)) or "No events."
            )

        history = _history(settings)
        previous = (
            history.recent_markdown(["plan", "standup"], now=now, days=4, limit=3)
            if history
            else ""
        )
        agent = PlanAgent()
        task = agent.build_task(
            day=target,
            tasks=tasks_text,
            notes=notes_text,
            calendar=calendar_text,
            previous=previous,
        )
        _execute(
            agent, task, opts, settings, workspace=Path.cwd(), input_preview=f"plan for {target}"
        )


@app.command(rich_help_panel="Planning")
def standup(
    ctx: typer.Context,
    repos: Annotated[
        list[Path] | None,
        typer.Option(
            "--repo", "-r", help="Git repositories to scan (default: current). Repeatable."
        ),
    ] = None,
    since: Annotated[
        str | None,
        typer.Option("--since", "-s", help="Start of the period (default: previous working day)."),
    ] = None,
    author: Annotated[
        str | None,
        typer.Option(
            "--author",
            "-a",
            help="Commit author to include (default: your git email; 'all' for everyone).",
        ),
    ] = None,
    notes: Annotated[Path | None, typer.Option("--notes", "-n", help="Notes file.")] = None,
    model: ModelOpt = None,
    effort: EffortOpt = None,
    as_json: JsonOpt = False,
    output: OutputOpt = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Write your standup (yesterday / today / blockers) from commits, notes and past plans."""
    opts = RunOptions(model, effort, as_json, output, dry_run)
    with _errors():
        settings = _load(ctx, "standup", opts)
        now = _now(settings)
        since_value = since or f"{previous_workday_start(now.date()):%Y-%m-%d %H:%M}"

        commits: dict[str, str] = {}
        for repo in repos or [Path.cwd()]:
            if not git.is_repo(repo):
                if repos:
                    raise InputError(f"{repo} is not a git repository.")
                err.print(
                    "[yellow]The current directory is not a git repository; skipping commits.[/]"
                )
                continue
            root = git.repo_root(repo)
            who = None if author == "all" else author or git.user_email(root)
            commits[root.name] = git.log(root, since=since_value, author=who, max_count=100)

        notes_source = str(notes) if notes else settings.files.notes
        notes_text = _read_text(notes_source, what="notes") if notes_source else ""
        history = _history(settings)
        previous = (
            history.recent_markdown(["plan", "standup"], now=now, days=4, limit=3)
            if history
            else ""
        )
        agent = StandupAgent()
        task = agent.build_task(
            since_label=since_value, commits=commits, notes=notes_text, previous=previous
        )
        _execute(
            agent,
            task,
            opts,
            settings,
            workspace=Path.cwd(),
            input_preview=f"standup since {since_value}",
        )


@app.command(rich_help_panel="Communication")
def inbox(
    ctx: typer.Context,
    sources: Annotated[
        list[str],
        typer.Argument(help="Email files (.eml, .mbox, .txt), folders of them, or - for stdin."),
    ],
    model: ModelOpt = None,
    effort: EffortOpt = None,
    as_json: JsonOpt = False,
    output: OutputOpt = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Triage emails: priorities, action items and ready-to-send reply drafts."""
    opts = RunOptions(model, effort, as_json, output, dry_run)
    with _errors():
        settings = _load(ctx, "inbox", opts)
        emails: list[Email] = []
        paths = [Path(s).expanduser() for s in sources if s != "-"]
        if "-" in sources:
            emails.append(from_text(_read_text("-", what="an email"), "stdin"))
        if paths:
            emails.extend(load_emails(paths))
        agent = InboxAgent()
        task = agent.build_task(emails)
        _execute(
            agent, task, opts, settings, workspace=Path.cwd(), input_preview=f"{len(emails)} emails"
        )


@app.command(rich_help_panel="Communication")
def meeting(
    ctx: typer.Context,
    transcript: Annotated[
        str,
        typer.Argument(help="Transcript or notes file (.txt, .md, .vtt, .srt), or - for stdin."),
    ],
    title: Annotated[str | None, typer.Option("--title", help="Meeting title, if known.")] = None,
    model: ModelOpt = None,
    effort: EffortOpt = None,
    as_json: JsonOpt = False,
    output: OutputOpt = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Turn a meeting transcript into decisions, owned action items and a follow-up email."""
    opts = RunOptions(model, effort, as_json, output, dry_run)
    with _errors():
        settings = _load(ctx, "meeting", opts)
        text = _read_text(transcript, what="a transcript")
        if not text.strip():
            raise InputError("The transcript is empty.")
        agent = MeetingAgent()
        task = agent.build_task(text, title=title)
        preview = title or (transcript if transcript != "-" else "transcript from stdin")
        _execute(agent, task, opts, settings, workspace=Path.cwd(), input_preview=preview)


@app.command(rich_help_panel="Research")
def research(
    ctx: typer.Context,
    question: Annotated[list[str], typer.Argument(help="What you want to know (quotes optional).")],
    urls: Annotated[
        list[str] | None, typer.Option("--url", "-u", help="Page to read first. Repeatable.")
    ] = None,
    depth: Annotated[
        DepthChoice, typer.Option("--depth", help="quick, standard or deep.")
    ] = DepthChoice.standard,
    no_search: Annotated[
        bool, typer.Option("--no-search", help="Do not search the web (still reads --url pages).")
    ] = False,
    model: ModelOpt = None,
    effort: EffortOpt = None,
    as_json: JsonOpt = False,
    output: OutputOpt = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Research a question on the web and get a cited brief."""
    opts = RunOptions(model, effort, as_json, output, dry_run)
    with _errors():
        settings = _load(ctx, "research", opts)
        text = " ".join(question).strip()
        if not text:
            raise InputError(
                'Ask a question, e.g. mywork research "best Python task queues in 2026"'
            )
        agent = ResearchAgent(depth=depth.value, search=not no_search)
        task = agent.build_task(text, urls=urls)
        _execute(agent, task, opts, settings, workspace=Path.cwd(), input_preview=text)


@app.command(rich_help_panel="Development")
def review(
    ctx: typer.Context,
    paths: Annotated[
        list[str] | None, typer.Argument(help="Limit the review to these paths.")
    ] = None,
    staged: Annotated[bool, typer.Option("--staged", help="Review only staged changes.")] = False,
    base: Annotated[
        str | None,
        typer.Option(
            "--base", "-b", help="Review what this branch adds on top of BASE (e.g. main)."
        ),
    ] = None,
    diff_file: Annotated[
        str | None, typer.Option("--diff", help="Review a patch file instead (- for stdin).")
    ] = None,
    repo: Annotated[Path, typer.Option("--repo", "-r", help="Repository to review.")] = Path(),
    focus: Annotated[
        str | None, typer.Option("--focus", "-f", help="What to pay extra attention to.")
    ] = None,
    context: Annotated[
        str | None,
        typer.Option("--context", help="What the change is meant to do (PR description)."),
    ] = None,
    fail_on: Annotated[
        FailOn,
        typer.Option(
            "--fail-on", help="Exit with code 1 if a finding is at least this severe (for CI)."
        ),
    ] = FailOn.never,
    model: ModelOpt = None,
    effort: EffortOpt = None,
    as_json: JsonOpt = False,
    output: OutputOpt = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Review code changes for bugs, security issues and missing tests."""
    opts = RunOptions(model, effort, as_json, output, dry_run)
    with _errors():
        settings = _load(ctx, "review", opts)
        workspace = repo
        if diff_file:
            diff_text = _read_text(diff_file, what="a diff")
            what = "patch file" if diff_file != "-" else "patch from stdin"
        else:
            if not git.is_repo(repo):
                raise InputError(
                    f"{repo.resolve()} is not a git repository (use --diff for a patch file)."
                )
            workspace = git.repo_root(repo)
            diff_text = git.diff(workspace, staged=staged, base=base, paths=paths or None)
            what = (
                f"changes vs {base}"
                if base
                else ("staged changes" if staged else "uncommitted changes")
            )
        if not diff_text.strip():
            raise InputError(
                "Nothing to review: the diff is empty. "
                "Make some changes, use --base main, or pass --diff."
            )
        if len(diff_text) > MAX_DIFF_CHARS:
            raise InputError(
                f"The diff is {len(diff_text):,} characters; review fewer files at a time "
                "(pass paths, e.g. `mywork review src/api`)."
            )
        agent = ReviewAgent()
        task = agent.build_task(diff_text, context=context, focus=focus)
        result = _execute(agent, task, opts, settings, workspace=workspace, input_preview=what)
        if result is not None and fail_on is not FailOn.never:
            review_output: CodeReview = result.output
            if worst_severity(review_output) >= SEVERITY_RANK[fail_on.value]:
                err.print(f"[red]✗[/red] Found issues at or above '{fail_on.value}' severity.")
                raise typer.Exit(1)


@app.command(rich_help_panel="Development")
def commit(
    ctx: typer.Context,
    repo: Annotated[Path, typer.Option("--repo", "-r", help="Repository to use.")] = Path(),
    hint: Annotated[
        str | None, typer.Option("--hint", help="Why you made the change, in a few words.")
    ] = None,
    style: Annotated[
        StyleChoice, typer.Option("--style", help="conventional (feat: ...) or plain.")
    ] = StyleChoice.conventional,
    apply: Annotated[
        bool, typer.Option("--apply", help="Create the commit with this message.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation.")] = False,
    model: ModelOpt = None,
    effort: EffortOpt = None,
    as_json: JsonOpt = False,
    output: OutputOpt = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Write a commit message for your staged changes (and optionally commit)."""
    opts = RunOptions(model, effort, as_json, output, dry_run)
    with _errors():
        settings = _load(ctx, "commit", opts)
        if not git.is_repo(repo):
            raise InputError(f"{repo.resolve()} is not a git repository.")
        root = git.repo_root(repo)
        diff_text = git.diff(root, staged=True)
        if not diff_text.strip():
            raise InputError("Nothing is staged. Stage your changes with `git add` first.")
        if len(diff_text) > MAX_DIFF_CHARS:
            raise InputError("The staged diff is too large to summarise; commit in smaller pieces.")
        agent = CommitAgent(style=style.value)
        task = agent.build_task(diff_text, recent_subjects=git.recent_subjects(root, 15), hint=hint)
        result = _execute(agent, task, opts, settings, workspace=root, input_preview="staged diff")
        if result is None or not apply:
            return
        message_model: CommitMessage = result.output
        message = message_model.format(style.value)
        if not yes and not typer.confirm("Create the commit with this message?", err=True):
            err.print("Not committed.")
            raise typer.Exit(1)
        summary = git.commit(root, message)
        err.print(f"[green]✓[/green] {summary.splitlines()[0] if summary else 'Committed.'}")


@app.command(rich_help_panel="Development")
def readme(
    ctx: typer.Context,
    path: Annotated[Path, typer.Argument(help="Project directory.", file_okay=False)] = Path(),
    notes: Annotated[
        str | None, typer.Option("--notes", help="Anything the README should emphasise.")
    ] = None,
    write: Annotated[
        bool, typer.Option("--write", "-w", help="Write README.md in the project directory.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Overwrite without asking.")] = False,
    model: ModelOpt = None,
    effort: EffortOpt = None,
    as_json: JsonOpt = False,
    output: OutputOpt = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Explore a project and write (or refresh) its README."""
    opts = RunOptions(model, effort, as_json, output, dry_run)
    with _errors():
        settings = _load(ctx, "readme", opts)
        if not path.is_dir():
            raise InputError(f"No such directory: {path}")
        target = path / "README.md"
        existing = (
            target.read_text(encoding="utf-8", errors="replace") if target.is_file() else None
        )
        agent = ReadmeAgent()
        task = agent.build_task(existing=existing, notes=notes)
        result = _execute(
            agent, task, opts, settings, workspace=path, input_preview=str(path.resolve())
        )
        if result is None or not write:
            return
        if target.exists() and not yes and not typer.confirm(f"Overwrite {target}?", err=True):
            err.print("README not written.")
            raise typer.Exit(1)
        target.write_text(result.output, encoding="utf-8")
        err.print(f"[green]✓[/green] Wrote {target}")


@app.command("run", rich_help_panel="Any agent")
def run_agent(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Agent name (see `mywork agents`).")],
    text: Annotated[
        str | None, typer.Argument(help="Input text. Omit to use --file or stdin.")
    ] = None,
    file: Annotated[
        Path | None, typer.Option("--file", "-f", help="Read the input from a file.")
    ] = None,
    workspace: Annotated[
        Path, typer.Option("--workspace", "-w", help="Directory the agent's file tools may read.")
    ] = Path(),
    model: ModelOpt = None,
    effort: EffortOpt = None,
    as_json: JsonOpt = False,
    output: OutputOpt = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Run any agent, including plugins, on free-form input."""
    opts = RunOptions(model, effort, as_json, output, dry_run)
    with _errors():
        cls = get_agent_class(name)
        settings = _load(ctx, name, opts)
        if text is not None:
            content = text
        elif file is not None:
            content = _read_text(file)
        elif not sys.stdin.isatty():
            content = sys.stdin.read()
        else:
            raise InputError("Give the agent some input: an argument, --file, or stdin.")
        try:
            agent = cls()
        except TypeError as exc:
            raise InputError(f"Agent {name!r} cannot be created without arguments: {exc}") from exc
        task = agent.task_from_text(content)
        _execute(agent, task, opts, settings, workspace=workspace, input_preview=content[:200])


@app.command("agents", rich_help_panel="Any agent")
def list_agents() -> None:
    """List the available agents."""
    table = Table(box=None, pad_edge=False, header_style="bold")
    table.add_column("Agent")
    table.add_column("Group", style="dim")
    table.add_column("What it does")
    agents = discover_agents()
    plugins = sorted(name for name in agents if name not in BUILTIN_AGENTS)
    for name in [*BUILTIN_AGENTS, *plugins]:
        cls = agents[name]
        plugin = name not in BUILTIN_AGENTS
        label = f"[cyan]{name}[/cyan]" + (
            f" [dim](plugin: mywork run {name})[/dim]" if plugin else ""
        )
        table.add_row(label, cls.group, cls.description)
    out.print(table)


@app.command(rich_help_panel="Housekeeping")
def doctor(
    ctx: typer.Context,
    ping: Annotated[
        bool, typer.Option("--ping", help="Also send a tiny test request to Claude.")
    ] = False,
) -> None:
    """Check your setup: config, credentials, git and (optionally) the API."""
    problems = 0

    def report(ok: bool | None, label: str, detail: str) -> None:
        nonlocal problems
        mark = {True: "[green]✓[/green]", False: "[red]✗[/red]", None: "[yellow]![/yellow]"}[ok]
        if ok is False:
            problems += 1
        out.print(f"{mark} [bold]{label}[/bold]  {detail}", highlight=False)

    import anthropic

    report(
        True,
        "mywork-agents",
        f"{__version__} (Python {platform.python_version()}, anthropic {anthropic.__version__})",
    )
    settings: Settings | None = None
    try:
        files = config_files(Path.cwd(), _state(ctx).config_file)
        settings = _load(ctx)
        report(True, "config", ", ".join(str(f) for f in files) or "defaults (no config file)")
    except MyworkError as exc:
        report(False, "config", str(exc))

    if settings is not None:
        info = model_info(settings.model)
        note = "" if info.known else " (unknown to mywork: no cost estimates)"
        report(True if info.known else None, "model", settings.model + note)

    if os.environ.get("ANTHROPIC_API_KEY"):
        report(True, "credentials", "ANTHROPIC_API_KEY is set")
    elif os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        report(True, "credentials", "ANTHROPIC_AUTH_TOKEN is set")
    elif (Path.home() / ".config" / "anthropic").is_dir():
        report(None, "credentials", "no env var; an `ant` profile may be used (try --ping)")
    else:
        report(False, "credentials", "set ANTHROPIC_API_KEY or run `ant auth login`")

    report(
        True if git.git_available() else None,
        "git",
        "available" if git.git_available() else "not found (standup, review and commit need it)",
    )

    if settings is not None:
        if settings.history:
            count = len(HistoryStore(settings.resolved_history_dir).entries())
            report(True, "history", f"{settings.resolved_history_dir} ({count} runs)")
        else:
            report(None, "history", "disabled")

    if ping and settings is not None:
        started = time.monotonic()
        try:
            with _errors():
                response = make_llm(settings, settings.model).complete(
                    LLMRequest(
                        system="You are a health check. Reply with the single word OK.",
                        messages=[{"role": "user", "content": "ping"}],
                        max_tokens=2048,
                        effort="low",
                    )
                )
            report(True, "api", f"{response.model} replied in {time.monotonic() - started:.1f}s")
        except typer.Exit:
            report(False, "api", "the test request failed (see above)")

    if problems:
        raise typer.Exit(1)


# ------------------------------------------------------------------- history


@history_app.command("list")
def history_list(
    ctx: typer.Context,
    agent: Annotated[str | None, typer.Option("--agent", "-a", help="Only this agent.")] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", help="How many runs to show.")] = 20,
) -> None:
    """List recent runs, newest first."""
    with _errors():
        settings = _load(ctx)
        entries = HistoryStore(settings.resolved_history_dir).entries(agent=agent)[:limit]
        if not entries:
            err.print("No runs yet.")
            return
        table = Table(box=None, pad_edge=False, header_style="bold")
        for column in ("ID", "When", "Agent", "Input"):
            table.add_column(column)
        for entry in entries:
            table.add_row(
                entry.id,
                f"{entry.created_at:%Y-%m-%d %H:%M}",
                entry.agent,
                entry.input_preview.replace("\n", " ")[:60],
            )
        out.print(table)


@history_app.command("show")
def history_show(
    ctx: typer.Context,
    run_id: Annotated[str, typer.Argument(help="Run ID (or a unique prefix).")],
    as_json: JsonOpt = False,
) -> None:
    """Show a past run's output."""
    with _errors():
        settings = _load(ctx)
        data = HistoryStore(settings.resolved_history_dir).get(run_id).load()
        if as_json or not out.is_terminal:
            text = (
                json.dumps(data, indent=2, ensure_ascii=False)
                if as_json
                else data.get("markdown", "")
            )
            sys.stdout.write(str(text) + "\n")
        else:
            out.print(Markdown(str(data.get("markdown", ""))))


@history_app.command("clear")
def history_clear(
    ctx: typer.Context,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation.")] = False,
) -> None:
    """Delete all saved runs."""
    with _errors():
        settings = _load(ctx)
        store = HistoryStore(settings.resolved_history_dir)
        if not yes and not typer.confirm(f"Delete every run in {store.root}?", err=True):
            raise typer.Exit(1)
        err.print(f"Deleted {store.clear()} runs.")


# -------------------------------------------------------------------- config


@config_app.command("show")
def config_show(ctx: typer.Context) -> None:
    """Show the effective configuration and where it came from."""
    with _errors():
        files = config_files(Path.cwd(), _state(ctx).config_file)
        settings = _load(ctx)
        err.print("Loaded from: " + (", ".join(map(str, files)) or "defaults only"))
        sys.stdout.write(settings.model_dump_json(indent=2) + "\n")


@config_app.command("init")
def config_init(
    project: Annotated[
        bool,
        typer.Option(
            "--project", help=f"Create ./{PROJECT_CONFIG_NAME} instead of the user config."
        ),
    ] = False,
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing file.")] = False,
) -> None:
    """Create a commented config file to edit."""
    target = Path.cwd() / PROJECT_CONFIG_NAME if project else user_config_path()
    if target.exists() and not force:
        err.print(f"[yellow]{target} already exists[/yellow] (use --force to overwrite).")
        raise typer.Exit(1)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(CONFIG_TEMPLATE, encoding="utf-8")
    err.print(f"[green]✓[/green] Created {target}")


@config_app.command("path")
def config_path() -> None:
    """Print where configuration files are looked up."""
    for label, path in (
        ("user", user_config_path()),
        ("project", Path.cwd() / PROJECT_CONFIG_NAME),
    ):
        state = "exists" if path.is_file() else "not found"
        out.print(f"{label:8} {path}  [dim]({state})[/dim]", highlight=False)


def main() -> None:
    # Never crash on a character the console encoding cannot show (e.g. Windows pipes).
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")
    app()
