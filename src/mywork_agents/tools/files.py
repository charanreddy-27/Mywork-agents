"""Read-only file tools, sandboxed to the workspace and blind to likely secrets."""

from __future__ import annotations

import fnmatch
import os
import re
import stat
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from mywork_agents.core.tools import ToolContext, tool
from mywork_agents.errors import ToolError

IGNORED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        "dist",
        "build",
        ".next",
        ".nuxt",
        "target",
        ".gradle",
        ".idea",
        "coverage",
        "htmlcov",
    }
)

# Version-control internals: .git/config can hold tokens (e.g. actions/checkout's header).
VCS_DIRS = frozenset({".git", ".hg", ".svn"})

# Never send these to the API: they usually hold credentials.
SENSITIVE_PATTERNS = (
    ".env*",
    "*.env",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa*",
    "id_ed25519*",
    "id_ecdsa*",
    ".npmrc",
    ".pypirc",
    ".netrc",
    ".git-credentials",
    "credentials.json",
    "service-account*.json",
    "*.keystore",
    "secrets.*",
)
SENSITIVE_ALLOWED = ("*.example", "*.sample", "*.template")
# Directories whose contents are secrets too, e.g. cookiecutter-django's .envs/.production/.
SENSITIVE_DIR_PATTERNS = (".env*",)

MAX_FILE_BYTES = 2_000_000


def is_sensitive(path: Path, root: Path | None = None) -> bool:
    """True for files that commonly hold credentials (by name, or by a parent directory)."""
    name = path.name.lower()
    allowed = any(fnmatch.fnmatch(name, pattern) for pattern in SENSITIVE_ALLOWED)
    if not allowed and any(fnmatch.fnmatch(name, pattern) for pattern in SENSITIVE_PATTERNS):
        return True
    if root is None:
        return False
    try:
        parents = path.relative_to(root).parts[:-1]
    except ValueError:
        return False
    return any(
        fnmatch.fnmatch(part.lower(), pattern)
        for part in parents
        for pattern in SENSITIVE_DIR_PATTERNS
    )


def resolve_in_workspace(ctx: ToolContext, raw: str) -> Path:
    """Resolve ``raw`` against the workspace, rejecting anything that escapes it."""
    root = ctx.workspace.resolve()
    target = (root / raw).resolve()
    if not target.is_relative_to(root):
        raise ToolError(
            f"{raw!r} is outside the workspace ({root}); only files inside it are readable."
        )
    if VCS_DIRS.intersection(target.relative_to(root).parts):
        raise ToolError(f"{raw!r} is inside version-control metadata, which is not readable.")
    return target


def split_lines(text: str) -> list[str]:
    """Split on newlines only, so numbers match git and grep (unlike str.splitlines)."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _display(ctx: ToolContext, path: Path) -> str:
    root = ctx.workspace.resolve()
    return path.relative_to(root).as_posix() if path != root else "."


def is_probably_binary(path: Path) -> bool:
    with path.open("rb") as fh:
        return b"\x00" in fh.read(8192)


class ReadFileArgs(BaseModel):
    path: str = Field(description="File path relative to the workspace root.")
    offset: int = Field(default=1, ge=1, description="1-based line number to start reading from.")
    limit: int = Field(default=400, ge=1, le=2000, description="Maximum number of lines to return.")


@tool()
def read_file(ctx: ToolContext, args: ReadFileArgs) -> str:
    """Read a text file from the workspace, with line numbers.

    Use this whenever you need the actual contents of a file (to check how a function is
    implemented, confirm a claim, or see code around a diff hunk). Read large files in
    chunks with offset/limit instead of guessing.
    """
    path = resolve_in_workspace(ctx, args.path)
    if not path.is_file():
        raise ToolError(f"No such file: {args.path}")
    root = ctx.workspace.resolve()
    # Check both the name asked for and the file it resolves to (symlinks).
    if is_sensitive(root / args.path, root) or is_sensitive(path, root):
        raise ToolError(f"{args.path} looks like it holds secrets, so it is not readable.")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ToolError(f"{args.path} is larger than 2 MB; search it with search_files instead.")
    if is_probably_binary(path):
        raise ToolError(f"{args.path} is a binary file.")
    lines = split_lines(path.read_text(encoding="utf-8", errors="replace"))
    total = len(lines)
    start = args.offset - 1
    chunk = lines[start : start + args.limit]
    if not chunk:
        return f"{args.path} has {total} lines; offset {args.offset} is past the end."
    width = len(str(start + len(chunk)))
    body = "\n".join(f"{start + i + 1:>{width}} | {line}" for i, line in enumerate(chunk))
    end = start + len(chunk)
    footer = f"\n[lines {args.offset}-{end} of {total}]" if end < total or start > 0 else ""
    return f"{_display(ctx, path)}\n{body}{footer}"


class ListDirArgs(BaseModel):
    path: str = Field(default=".", description="Directory relative to the workspace root.")
    depth: int = Field(default=2, ge=1, le=6, description="How many levels deep to list.")


@tool()
def list_dir(ctx: ToolContext, args: ListDirArgs) -> str:
    """List files and folders as an indented tree (dependency and build folders are skipped).

    Use this first to understand a project's layout before reading individual files.
    """
    root = resolve_in_workspace(ctx, args.path)
    if not root.is_dir():
        raise ToolError(f"No such directory: {args.path}")
    lines: list[str] = [f"{_display(ctx, root)}/"]
    limit = 500
    workspace_root = ctx.workspace.resolve()

    def walk(directory: Path, level: int) -> None:
        try:
            children = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return
        for child in children:
            if len(lines) >= limit:
                return
            if child.is_dir():
                if child.name in IGNORED_DIRS:
                    continue
                if child.is_symlink() and not child.resolve().is_relative_to(workspace_root):
                    lines.append(f"{'  ' * level}{child.name}/ (link outside the workspace)")
                    continue
                lines.append(f"{'  ' * level}{child.name}/")
                if level < args.depth:
                    walk(child, level + 1)
            else:
                lines.append(f"{'  ' * level}{child.name}")

    walk(root, 1)
    if len(lines) >= limit:
        lines.append(f"[stopped after {limit} entries; list a subdirectory to see more]")
    return "\n".join(lines)


class SearchFilesArgs(BaseModel):
    pattern: str = Field(description="Regular expression (Python syntax) to search for.")
    path: str = Field(default=".", description="Directory or file to search, relative to the root.")
    glob: str | None = Field(
        default=None,
        description="Only search matching files, e.g. '*.py', 'src/**/*.ts' or '*.{js,ts}'.",
    )
    ignore_case: bool = Field(default=False, description="Case-insensitive match.")
    max_results: int = Field(default=80, ge=1, le=500, description="Stop after this many matches.")


@tool()
def search_files(ctx: ToolContext, args: SearchFilesArgs) -> str:
    """Search file contents with a regular expression; returns `path:line: text` matches.

    Use this to find where a symbol is defined or used, or to check whether something
    exists anywhere in the project, instead of reading files one by one.
    """
    try:
        regex = re.compile(args.pattern, re.IGNORECASE if args.ignore_case else 0)
    except re.error as exc:
        raise ToolError(f"Invalid regular expression: {exc}") from exc
    base = resolve_in_workspace(ctx, args.path)
    if not base.exists():
        raise ToolError(f"No such path: {args.path}")

    root = ctx.workspace.resolve()
    wanted = _glob_matcher(args.glob) if args.glob else None
    files: list[Path] = []
    if base.is_file():
        files = [base]
    else:
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
            for filename in sorted(filenames):
                candidate = Path(dirpath) / filename
                if wanted and not wanted(filename, candidate.relative_to(root).as_posix()):
                    continue
                files.append(candidate)

    matches: list[str] = []
    for file in files:
        resolved = file.resolve()
        if (
            not resolved.is_relative_to(root)
            or is_sensitive(file, root)
            or is_sensitive(resolved, root)
        ):
            continue  # secrets (by link name or target) and links out of the workspace
        try:
            info = file.stat()
            # Regular files only: reading a named pipe would block forever.
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE_BYTES:
                continue
            if is_probably_binary(file):
                continue
            text = file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for number, line in enumerate(split_lines(text), 1):
            if regex.search(line):
                matches.append(f"{_display(ctx, file)}:{number}: {line.strip()[:200]}")
                if len(matches) >= args.max_results:
                    matches.append(f"[stopped after {args.max_results} matches]")
                    return "\n".join(matches)
    return "\n".join(matches) if matches else "No matches."


def _expand_braces(pattern: str) -> list[str]:
    match = re.search(r"\{([^{}]*)\}", pattern)
    if match is None:
        return [pattern]
    head, tail = pattern[: match.start()], pattern[match.end() :]
    return [
        expanded
        for option in match.group(1).split(",")
        for expanded in _expand_braces(head + option + tail)
    ]


def _glob_matcher(glob: str) -> Callable[[str, str], bool]:
    """Match a file by name ('*.py') or, for globs with '/', by workspace-relative path."""
    name_patterns: list[str] = []
    path_patterns: list[str] = []
    for pattern in _expand_braces(glob.strip()):
        if "/" not in pattern:
            name_patterns.append(pattern)
            continue
        path_patterns.append(pattern)
        shorter = pattern
        while shorter.startswith("**/"):  # '**/' may also match zero directories
            shorter = shorter[3:]
            path_patterns.append(shorter)

    def matches(name: str, relative_path: str) -> bool:
        return any(fnmatch.fnmatch(name, p) for p in name_patterns) or any(
            fnmatch.fnmatch(relative_path, p) for p in path_patterns
        )

    return matches


FILE_TOOLS = (list_dir, read_file, search_files)
