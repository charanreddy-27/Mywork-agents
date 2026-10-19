"""Read-only git helpers (used directly by commands and wrapped as tools for Claude)."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from mywork_agents.core.tools import ToolContext, tool
from mywork_agents.errors import InputError, ToolError

GIT_TIMEOUT_S = 30
# Revisions and paths coming from the model must never be parsed as options.
_SAFE_REV = re.compile(r"^[A-Za-z0-9._/~^@{}:+-]+$")


class GitError(InputError):
    """A git command failed."""


def git_available() -> bool:
    return shutil.which("git") is not None


def run_git(repo: Path, *args: str, check: bool = True) -> str:
    if not git_available():
        raise GitError("git is not installed or not on PATH.")
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=GIT_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise GitError(f"git {' '.join(args)} timed out after {GIT_TIMEOUT_S}s") from exc
    if check and proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip() or f"exit code {proc.returncode}"
        raise GitError(f"git {' '.join(args)} failed: {message}")
    return proc.stdout


def is_repo(path: Path) -> bool:
    if not git_available() or not path.is_dir():
        return False
    try:
        return run_git(path, "rev-parse", "--is-inside-work-tree").strip() == "true"
    except GitError:
        return False


def repo_root(path: Path) -> Path:
    return Path(run_git(path, "rev-parse", "--show-toplevel").strip())


def current_branch(repo: Path) -> str:
    return run_git(repo, "rev-parse", "--abbrev-ref", "HEAD", check=False).strip() or "HEAD"


def user_email(repo: Path) -> str | None:
    return run_git(repo, "config", "user.email", check=False).strip() or None


def has_commits(repo: Path) -> bool:
    return bool(run_git(repo, "rev-parse", "--verify", "--quiet", "HEAD", check=False).strip())


def diff(
    repo: Path,
    *,
    staged: bool = False,
    base: str | None = None,
    paths: list[str] | None = None,
) -> str:
    """A diff of the repository.

    ``base`` set: what the current branch adds on top of ``base`` (``base...HEAD``).
    ``staged``: only staged changes. Otherwise: every uncommitted change (staged and
    unstaged) relative to HEAD.
    """
    args = ["diff", "--no-color", "--no-ext-diff", "-M"]
    if base:
        args.append(f"{check_rev(base)}...HEAD")
    elif staged or not has_commits(repo):
        args.append("--cached")
    else:
        args.append("HEAD")
    if paths:
        args += ["--", *paths]
    return run_git(repo, *args)


def commit(repo: Path, message: str) -> str:
    """Create a commit from the staged changes; git hooks run as usual."""
    if not git_available():
        raise GitError("git is not installed or not on PATH.")
    proc = subprocess.run(
        ["git", "-C", str(repo), "commit", "-F", "-"],
        input=message,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        raise GitError(f"git commit failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return proc.stdout.strip()


def log(
    repo: Path,
    *,
    since: str | None = None,
    author: str | None = None,
    max_count: int = 50,
    path: str | None = None,
) -> str:
    args = [
        "log",
        "--no-merges",
        "--no-color",
        f"--max-count={max_count}",
        "--date=format:%Y-%m-%d %H:%M",
        "--pretty=format:%h  %ad  %an  %s",
    ]
    if since:
        args.append(f"--since={since}")
    if author:
        args.append(f"--author={author}")
    if path:
        args += ["--", path]
    return run_git(repo, *args)


def recent_subjects(repo: Path, count: int = 10) -> list[str]:
    out = run_git(repo, "log", f"--max-count={count}", "--pretty=format:%s", check=False)
    return [line for line in out.splitlines() if line.strip()]


def check_rev(rev: str) -> str:
    if not rev or rev.startswith("-") or not _SAFE_REV.match(rev):
        raise ToolError(f"Not a valid git revision: {rev!r}")
    return rev


# ----------------------------------------------------------------- tools


class GitLogArgs(BaseModel):
    since: str | None = Field(default=None, description="e.g. '2 weeks ago' or '2026-10-01'.")
    path: str | None = Field(default=None, description="Only commits touching this path.")
    max_count: int = Field(default=30, ge=1, le=200)


@tool()
def git_log(ctx: ToolContext, args: GitLogArgs) -> str:
    """Show recent commits (hash, date, author, subject) in the workspace repository.

    Use this to understand a file's history or recent activity before judging a change.
    """
    # `path` goes after `--` and `since` is joined to its flag, so neither can inject options.
    output = log(ctx.workspace, since=args.since, max_count=args.max_count, path=args.path)
    return output or "No commits found."


class GitShowArgs(BaseModel):
    rev: str = Field(description="Commit hash, tag or other revision to show.")
    stat_only: bool = Field(default=False, description="Only list changed files, not the patch.")


@tool()
def git_show(ctx: ToolContext, args: GitShowArgs) -> str:
    """Show one commit: its message and patch (or just the changed files with stat_only).

    Use this when a commit subject alone does not explain what changed.
    """
    rev = check_rev(args.rev)
    flags = ["--stat"] if args.stat_only else ["--stat", "--patch"]
    return run_git(ctx.workspace, "show", "--no-color", *flags, rev, "--")


GIT_TOOLS = (git_log, git_show)
