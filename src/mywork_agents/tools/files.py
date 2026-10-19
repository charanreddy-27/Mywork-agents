"""Read-only file tools, sandboxed to the workspace and blind to likely secrets."""

from __future__ import annotations

import fnmatch
import os
import re
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

# Never send these to the API: they usually hold credentials.
SENSITIVE_PATTERNS = (
    ".env",
    ".env.*",
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

MAX_FILE_BYTES = 2_000_000


def is_sensitive(path: Path) -> bool:
    name = path.name.lower()
    if any(fnmatch.fnmatch(name, pattern) for pattern in SENSITIVE_ALLOWED):
        return False
    return any(fnmatch.fnmatch(name, pattern) for pattern in SENSITIVE_PATTERNS)


def resolve_in_workspace(ctx: ToolContext, raw: str) -> Path:
    """Resolve ``raw`` against the workspace, rejecting anything that escapes it."""
    root = ctx.workspace.resolve()
    target = (root / raw).resolve()
    if not target.is_relative_to(root):
        raise ToolError(
            f"{raw!r} is outside the workspace ({root}); only files inside it are readable."
        )
    return target


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
    if is_sensitive(path):
        raise ToolError(f"{args.path} looks like it holds secrets, so it is not readable.")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ToolError(f"{args.path} is larger than 2 MB; search it with search_files instead.")
    if is_probably_binary(path):
        raise ToolError(f"{args.path} is a binary file.")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
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
        default=None, description="Only search files matching this glob, e.g. '*.py'."
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

    files: list[Path] = []
    if base.is_file():
        files = [base]
    else:
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
            for filename in sorted(filenames):
                if args.glob and not fnmatch.fnmatch(filename, args.glob):
                    continue
                files.append(Path(dirpath) / filename)

    root = ctx.workspace.resolve()
    matches: list[str] = []
    for file in files:
        if is_sensitive(file) or not file.resolve().is_relative_to(root):
            continue  # skip secrets and symlinks that point outside the workspace
        try:
            if file.stat().st_size > MAX_FILE_BYTES or is_probably_binary(file):
                continue
            text = file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                matches.append(f"{_display(ctx, file)}:{number}: {line.strip()[:200]}")
                if len(matches) >= args.max_results:
                    matches.append(f"[stopped after {args.max_results} matches]")
                    return "\n".join(matches)
    return "\n".join(matches) if matches else "No matches."


FILE_TOOLS = (list_dir, read_file, search_files)
