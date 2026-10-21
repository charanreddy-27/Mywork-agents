"""A local, append-only log of agent runs (one JSON file per run).

History is what lets agents build on each other: ``standup`` reads yesterday's
``plan``, and ``plan`` sees what you planned and reported before.
"""

from __future__ import annotations

import json
import re
import secrets
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mywork_agents.errors import InputError

if TYPE_CHECKING:
    from mywork_agents.core.agent import RunResult

# Only files with these shapes are runs; anything else under the root is left alone.
_DAY_DIR = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_RUN_FILE = re.compile(r"^\d{8}-\d{6}-.+-[0-9a-f]{4}$")


@dataclass(frozen=True)
class HistoryEntry:
    id: str
    agent: str
    created_at: datetime
    model: str
    input_preview: str
    path: Path

    def load(self) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
        return data


class HistoryStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def save(self, result: RunResult[Any], *, input_preview: str, now: datetime) -> HistoryEntry:
        run_id = f"{now:%Y%m%d-%H%M%S}-{result.agent}-{secrets.token_hex(2)}"
        day_dir = self.root / f"{now:%Y-%m-%d}"
        day_dir.mkdir(parents=True, exist_ok=True)
        path = day_dir / f"{run_id}.json"
        record = {
            "id": run_id,
            "created_at": now.isoformat(),
            "input_preview": input_preview[:500],
            **result.to_dict(),
        }
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        return HistoryEntry(run_id, result.agent, now, result.model, record["input_preview"], path)

    def entries(
        self, agent: str | None = None, since: datetime | None = None
    ) -> list[HistoryEntry]:
        """Entries, newest first."""
        found: list[HistoryEntry] = []
        for path in self._run_files():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict) or not isinstance(data.get("created_at"), str):
                    continue
                created = datetime.fromisoformat(data["created_at"])
            except (OSError, ValueError, TypeError):
                continue  # skip unreadable or foreign files rather than failing the listing
            if created.tzinfo is None:
                created = created.astimezone()  # treat naive timestamps as local time
            if agent and data.get("agent") != agent:
                continue
            if since and created < since:
                continue
            found.append(
                HistoryEntry(
                    id=data.get("id", path.stem),
                    agent=data.get("agent", "?"),
                    created_at=created,
                    model=data.get("model", ""),
                    input_preview=data.get("input_preview", ""),
                    path=path,
                )
            )
        found.sort(key=lambda e: e.created_at, reverse=True)
        return found

    def get(self, run_id: str) -> HistoryEntry:
        matches = [e for e in self.entries() if e.id == run_id or e.id.startswith(run_id)]
        if not matches:
            raise InputError(f"No run found with id {run_id!r}. See `mywork history list`.")
        if len(matches) > 1 and matches[0].id != run_id:
            raise InputError(f"{run_id!r} matches {len(matches)} runs; use more of the id.")
        return matches[0]

    def recent_markdown(
        self, agents: list[str], *, now: datetime, days: int, limit: int, max_chars: int = 4000
    ) -> str:
        """Recent outputs of ``agents`` as Markdown, for feeding back into a prompt."""
        since = now - timedelta(days=days)
        sections = []
        for entry in self.entries(since=since):
            if entry.agent not in agents:
                continue
            markdown = str(entry.load().get("markdown", "")).strip()
            if len(markdown) > max_chars:
                markdown = markdown[:max_chars] + "\n[... truncated]"
            sections.append(
                f"### {entry.agent} run on {entry.created_at:%A %d %B %Y, %H:%M}\n\n{markdown}"
            )
            if len(sections) >= limit:
                break
        return "\n\n".join(sections)

    def clear(self) -> int:
        """Delete run files (and day folders left empty); other files are never touched."""
        removed = 0
        for path in list(self._run_files()):
            path.unlink()
            removed += 1
        for day_dir in self._day_dirs():
            if not any(day_dir.iterdir()):
                day_dir.rmdir()
        return removed

    def _day_dirs(self) -> Iterator[Path]:
        if not self.root.is_dir():
            return
        for day_dir in sorted(self.root.iterdir()):
            if day_dir.is_dir() and _DAY_DIR.match(day_dir.name):
                yield day_dir

    def _run_files(self) -> Iterator[Path]:
        for day_dir in self._day_dirs():
            for path in sorted(day_dir.glob("*.json")):
                if path.is_file() and _RUN_FILE.match(path.stem):
                    yield path
