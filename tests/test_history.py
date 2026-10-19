from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mywork_agents.core.agent import RunResult
from mywork_agents.core.llm import Usage
from mywork_agents.errors import InputError
from mywork_agents.history import HistoryStore

NOW = datetime(2026, 10, 19, 9, 0, tzinfo=timezone.utc)


def result(agent: str, markdown: str) -> RunResult[str]:
    return RunResult(
        agent=agent,
        model="claude-opus-5-5",
        output=markdown,
        markdown=markdown,
        usage=Usage(input_tokens=1, output_tokens=2, requests=1),
        turns=1,
        duration_s=1.5,
    )


def test_save_list_get(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history")
    first = store.save(
        result("plan", "# Plan A"), input_preview="plan", now=NOW - timedelta(days=1)
    )
    second = store.save(result("standup", "Did things"), input_preview="standup", now=NOW)

    assert [e.id for e in store.entries()] == [second.id, first.id]
    assert [e.id for e in store.entries(agent="plan")] == [first.id]
    assert store.entries(since=NOW - timedelta(hours=1))[0].id == second.id
    assert store.get(first.id[:20]).id == first.id
    data = store.get(second.id).load()
    assert data["markdown"] == "Did things"
    assert data["usage"]["output_tokens"] == 2

    with pytest.raises(InputError, match="No run"):
        store.get("does-not-exist")
    with pytest.raises(InputError, match="matches 2 runs"):
        store.get("2026")


def test_recent_markdown_filters_and_truncates(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path)
    store.save(result("plan", "old plan"), input_preview="", now=NOW - timedelta(days=10))
    store.save(result("plan", "x" * 50), input_preview="", now=NOW - timedelta(days=1))
    store.save(result("meeting", "notes"), input_preview="", now=NOW)

    text = store.recent_markdown(["plan"], now=NOW, days=3, limit=5, max_chars=10)
    assert "old plan" not in text
    assert "notes" not in text
    assert "### plan run on Sunday 18 October 2026" in text
    assert "x" * 10 + "\n[... truncated]" in text


def test_corrupt_files_are_skipped_and_clear(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path)
    store.save(result("plan", "ok"), input_preview="", now=NOW)
    (tmp_path / "2026-10-19" / "junk.json").write_text("{not json")
    assert len(store.entries()) == 1
    assert store.clear() == 2
    assert store.entries() == []
    assert not any(tmp_path.iterdir())
