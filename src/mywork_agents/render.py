"""Small Markdown helpers shared by the agents' ``render`` methods."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def bullets(items: Iterable[str], *, empty: str = "_None_") -> str:
    lines = [f"- {item}" for item in items if item.strip()]
    return "\n".join(lines) if lines else empty


def checkboxes(items: Iterable[str], *, empty: str = "_None_") -> str:
    lines = [f"- [ ] {item}" for item in items if item.strip()]
    return "\n".join(lines) if lines else empty


def table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    def cell(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ").strip() or "—"

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def quote(text: str) -> str:
    return "\n".join(f"> {line}" if line.strip() else ">" for line in text.strip().splitlines())


def section(title: str, body: str, level: int = 2) -> str:
    return f"{'#' * level} {title}\n\n{body.strip()}"


def join_sections(*parts: str) -> str:
    return "\n\n".join(p.strip() for p in parts if p and p.strip()) + "\n"
