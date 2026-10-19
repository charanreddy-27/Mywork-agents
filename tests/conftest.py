from __future__ import annotations

import subprocess
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mywork_agents.config import Settings
from mywork_agents.core.agent import AgentContext
from mywork_agents.core.events import Event
from mywork_agents.history import HistoryStore
from mywork_agents.testing import FakeLLM

FIXED_NOW = datetime(2026, 10, 19, 9, 30, tzinfo=timezone.utc)
EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep every test away from the real home directory, config and credentials."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(home / ".local" / "share"))
    for var in (
        "MYWORK_CONFIG",
        "MYWORK_MODEL",
        "MYWORK_EFFORT",
        "MYWORK_MAX_TOKENS",
        "MYWORK_MAX_TURNS",
        "MYWORK_HISTORY",
        "MYWORK_HISTORY_DIR",
        "MYWORK_FALLBACKS",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_PROFILE",
    ):
        monkeypatch.delenv(var, raising=False)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return work


@pytest.fixture
def make_ctx(tmp_path: Path) -> Callable[..., AgentContext]:
    def factory(
        llm: FakeLLM,
        *,
        settings: Settings | None = None,
        workspace: Path | None = None,
        history: HistoryStore | None = None,
        events: list[Event] | None = None,
    ) -> AgentContext:
        return AgentContext(
            settings=settings or Settings(),
            make_llm=lambda model: llm,
            workspace=workspace or tmp_path,
            history=history,
            on_event=events.append if events is not None else None,
            now=FIXED_NOW,
        )

    return factory


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / "repo"
    path.mkdir()
    git(path, "init", "-q", "-b", "main")
    git(path, "config", "user.email", "dev@example.com")
    git(path, "config", "user.name", "Dev")
    git(path, "config", "commit.gpgsign", "false")
    (path / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    git(path, "add", "calc.py")
    git(path, "commit", "-qm", "feat: add calculator")
    return path
