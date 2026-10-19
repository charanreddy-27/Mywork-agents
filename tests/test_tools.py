from __future__ import annotations

import os
from pathlib import Path

import httpx2
import pytest
from pydantic import BaseModel

from mywork_agents.config import Settings
from mywork_agents.core.tools import ToolContext, tool, truncate_output
from mywork_agents.errors import ToolError
from mywork_agents.tools import git as gitmod
from mywork_agents.tools.files import list_dir, read_file, search_files
from mywork_agents.tools.git import git_log, git_show
from mywork_agents.tools.web import (
    _check_url,
    download,
    fetch_url,
    html_to_text,
    web_server_tools,
)

from .conftest import git


@pytest.fixture
def workspace(tmp_path: Path) -> ToolContext:
    root = tmp_path / "ws"
    (root / "src").mkdir(parents=True)
    (root / "node_modules" / "dep").mkdir(parents=True)
    (root / "src" / "app.py").write_text("\n".join(f"line {i}" for i in range(1, 11)) + "\n")
    (root / "src" / "util.py").write_text("def helper():\n    return 'TODO: tidy'\n")
    (root / "node_modules" / "dep" / "index.js").write_text("// TODO: hidden")
    (root / ".env").write_text("API_KEY=secret TODO")
    (root / ".env.example").write_text("API_KEY=")
    (root / "blob.bin").write_bytes(b"\x00\x01\x02")
    return ToolContext(workspace=root, settings=Settings())


def call(t: object, ctx: ToolContext, **args: object) -> tuple[str, bool]:
    outcome = t.invoke(ctx, args)  # type: ignore[attr-defined]
    return outcome.content, outcome.is_error


def test_read_file_with_numbers_and_paging(workspace: ToolContext) -> None:
    text, error = call(read_file, workspace, path="src/app.py", offset=3, limit=2)
    assert not error
    assert "\n3 | line 3\n4 | line 4\n" in text
    assert "line 5" not in text
    assert "[lines 3-4 of 10]" in text


def test_read_file_guards(workspace: ToolContext, tmp_path: Path) -> None:
    (tmp_path / "outside.txt").write_text("secret")

    for path, message in [
        ("../outside.txt", "outside the workspace"),
        (".env", "secrets"),
        ("blob.bin", "binary"),
        ("missing.py", "No such file"),
    ]:
        text, error = call(read_file, workspace, path=path)
        assert error, path
        assert message in text, (path, text)

    text, error = call(read_file, workspace, path=".env.example")
    assert not error


def test_read_file_rejects_invalid_arguments(workspace: ToolContext) -> None:
    text, error = call(read_file, workspace, path="src/app.py", limit=999_999)
    assert error
    assert "INVALID_JSON" in text


def test_list_dir_skips_dependency_folders(workspace: ToolContext) -> None:
    text, error = call(list_dir, workspace)
    assert not error
    assert "src/" in text
    assert "app.py" in text
    assert "node_modules" not in text


def test_search_files(workspace: ToolContext) -> None:
    text, _ = call(search_files, workspace, pattern="TODO")
    assert text == "src/util.py:2: return 'TODO: tidy'"

    text, _ = call(search_files, workspace, pattern="todo", ignore_case=True, glob="*.md")
    assert text == "No matches."

    text, error = call(search_files, workspace, pattern="(")
    assert error
    assert "Invalid regular expression" in text


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_read_file_rejects_symlinks_out_of_workspace(
    workspace: ToolContext, tmp_path: Path
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    (workspace.workspace / "link.txt").symlink_to(outside)
    text, error = call(read_file, workspace, path="link.txt")
    assert error
    assert "outside the workspace" in text


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_search_skips_symlinks_out_of_workspace(workspace: ToolContext, tmp_path: Path) -> None:
    outside = tmp_path / "secret.txt"
    outside.write_text("TODO outside")
    (workspace.workspace / "src" / "leak.txt").symlink_to(outside)
    text, _ = call(search_files, workspace, pattern="outside")
    assert text == "No matches."


class DecoratorArgs(BaseModel):
    x: int


def test_tool_decorator_validation() -> None:
    with pytest.raises(TypeError, match="docstring"):

        @tool()
        def undocumented(ctx: ToolContext, args: DecoratorArgs) -> str:
            return ""

    with pytest.raises(TypeError, match="pydantic model"):

        @tool()
        def untyped(ctx: ToolContext, args: int) -> str:
            """Doc."""
            return ""

    class LocalArgs(BaseModel):
        y: int

    with pytest.raises(TypeError, match="module level"):

        @tool()
        def local(ctx: ToolContext, args: LocalArgs) -> str:
            """Doc."""
            return ""


def test_truncate_output_is_explicit() -> None:
    assert truncate_output("short") == "short"
    long = truncate_output("x" * 50, limit=10)
    assert long.startswith("x" * 10)
    assert "Output truncated" in long
    assert "50" in long


# --------------------------------------------------------------------- git


def test_diff_modes(repo: Path) -> None:
    (repo / "calc.py").write_text("def add(a, b):\n    return a + b + 0\n")
    (repo / "new.py").write_text("x = 1\n")
    git(repo, "add", "new.py")

    uncommitted = gitmod.diff(repo)
    assert "calc.py" in uncommitted and "new.py" in uncommitted
    staged = gitmod.diff(repo, staged=True)
    assert "new.py" in staged and "calc.py" not in staged
    assert gitmod.diff(repo, paths=["calc.py"]).count("diff --git") == 1


def test_diff_against_base_branch(repo: Path) -> None:
    git(repo, "checkout", "-q", "-b", "feature")
    (repo / "feature.py").write_text("y = 2\n")
    git(repo, "add", "feature.py")
    git(repo, "commit", "-qm", "feat: feature")
    assert "feature.py" in gitmod.diff(repo, base="main")


def test_diff_in_repo_without_commits(tmp_path: Path) -> None:
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    git(fresh, "init", "-q")
    (fresh / "a.py").write_text("a = 1\n")
    git(fresh, "add", "a.py")
    assert "a.py" in gitmod.diff(fresh)


def test_log_commit_and_subjects(repo: Path) -> None:
    (repo / "b.py").write_text("b = 1\n")
    git(repo, "add", "b.py")
    gitmod.commit(repo, "feat: add b\n\nBecause b matters.\n")
    assert gitmod.recent_subjects(repo) == ["feat: add b", "feat: add calculator"]
    assert "feat: add b" in gitmod.log(repo, author="dev@example.com")
    assert gitmod.user_email(repo) == "dev@example.com"
    assert gitmod.is_repo(repo)
    assert not gitmod.is_repo(repo.parent)


def test_git_tools_reject_option_injection(repo: Path) -> None:
    ctx = ToolContext(workspace=repo, settings=Settings())
    text, error = call(git_show, ctx, rev="--output=/tmp/pwned")
    assert error
    assert "Not a valid git revision" in text

    text, error = call(git_show, ctx, rev="HEAD", stat_only=True)
    assert not error
    assert "calc.py" in text

    text, error = call(git_log, ctx, path="calc.py")
    assert "feat: add calculator" in text


# --------------------------------------------------------------------- web


def test_html_to_text() -> None:
    title, text = html_to_text(
        "<html><head><title>Docs</title><style>p{}</style></head><body>"
        "<h1>Retries</h1><p>Use <b>backoff</b>.</p><script>alert(1)</script>"
        "<ul><li>one</li><li>two</li></ul></body></html>"
    )
    assert title == "Docs"
    assert "# Retries" in text
    assert "Use backoff." in text
    assert "- one" in text and "- two" in text
    assert "alert" not in text
    assert "p{}" not in text


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/x",
        "http://localhost:8000/",
        "http://127.0.0.1/",
        "http://10.0.0.5/admin",
        "http://169.254.169.254/latest/meta-data",
        "http://printer.local/",
        "file:///etc/passwd",
    ],
)
def test_check_url_blocks_local_targets(url: str) -> None:
    with pytest.raises(ToolError):
        _check_url(url)


def _client(handler: httpx2.MockTransport) -> httpx2.Client:
    return httpx2.Client(transport=handler, follow_redirects=True)


def test_download_html_and_json() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/page":
            return httpx2.Response(
                200,
                headers={"content-type": "text/html; charset=utf-8"},
                content=b"<title>T</title><p>Hello</p>",
            )
        if request.url.path == "/data":
            return httpx2.Response(200, json={"a": 1})
        if request.url.path == "/img":
            return httpx2.Response(200, headers={"content-type": "image/png"}, content=b"\x89PNG")
        if request.url.path == "/redirect":
            return httpx2.Response(302, headers={"location": "http://127.0.0.1/private"})
        return httpx2.Response(404)

    client = _client(httpx2.MockTransport(handler))
    assert download("https://example.com/page", client=client)[1:] == ("T", "Hello")
    assert '"a"' in download("https://example.com/data", client=client)[2]
    with pytest.raises(ToolError, match="404"):
        download("https://example.com/missing", client=client)
    with pytest.raises(ToolError, match="content type"):
        download("https://example.com/img", client=client)
    with pytest.raises(ToolError, match="private or local"):
        download("https://example.com/redirect", client=client)


def test_fetch_url_tool_uses_context_client(workspace: ToolContext) -> None:
    transport = httpx2.MockTransport(
        lambda request: httpx2.Response(
            200, headers={"content-type": "text/html"}, content=b"<h2>Hi</h2>"
        )
    )
    workspace.extra["http_client"] = _client(transport)
    text, error = call(fetch_url, workspace, url="https://example.com/")
    assert not error
    assert "URL: https://example.com/" in text
    assert "## Hi" in text


def test_web_server_tool_versions() -> None:
    modern = web_server_tools("claude-opus-5-5", max_uses=4)
    assert [t["type"] for t in modern] == ["web_search_20260209", "web_fetch_20260209"]
    assert modern[0]["max_uses"] == 4
    legacy = web_server_tools("claude-haiku-4-5", max_uses=2)
    assert [t["type"] for t in legacy] == ["web_search_20250305"]


# ----------------------------------------------------------------- history


def test_recent_runs_tool(workspace: ToolContext, tmp_path: Path) -> None:
    from datetime import datetime

    from mywork_agents.core.agent import RunResult
    from mywork_agents.core.llm import Usage
    from mywork_agents.history import HistoryStore
    from mywork_agents.tools.history import recent_runs

    text, _ = call(recent_runs, workspace)
    assert "disabled" in text

    workspace.history = HistoryStore(tmp_path / "history")
    text, _ = call(recent_runs, workspace, agents=["meeting"])
    assert text == "No runs of meeting in the last 3 days."

    run = RunResult("plan", "m", "x", "# Yesterday's plan", Usage(), 1, 1.0)
    workspace.history.save(run, input_preview="", now=datetime.now().astimezone())
    text, error = call(recent_runs, workspace)
    assert not error
    assert "# Yesterday's plan" in text


def test_describe_call() -> None:
    from mywork_agents.core.events import describe_call

    assert describe_call("read_file", {"path": "a.py"}) == "read_file(path='a.py')"
    assert describe_call("x", None) == "x()"
    assert describe_call("y", {"q": "z" * 200}, limit=20).endswith("…")
