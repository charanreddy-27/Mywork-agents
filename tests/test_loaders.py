from __future__ import annotations

import mailbox
from datetime import date, timezone
from email.message import EmailMessage
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from mywork_agents.errors import InputError
from mywork_agents.loaders.ics import events_on, format_occurrences, parse_ics
from mywork_agents.loaders.mail import emails_to_prompt, from_text, load_emails
from mywork_agents.loaders.transcript import clean_transcript

from .conftest import EXAMPLES

UTC = timezone.utc
IST = ZoneInfo("Asia/Kolkata")

# ------------------------------------------------------------------ transcripts


def test_vtt_transcript_is_cleaned_and_merged() -> None:
    text = clean_transcript(
        "WEBVTT\n\nNOTE generated\nby a tool\n\n1\n00:00:01.000 --> 00:00:02.000\n"
        "<v Ana>Hello there.</v>\n\n2\n00:00:02.500 --> 00:00:04.000\n<v Ana>Second line.\n\n"
        "3\n00:00:04.000 --> 00:00:05.000\n<v Ben>Hi <i>Ana</i>.\n"
    )
    assert text == "Ana: Hello there. Second line.\nBen: Hi Ana."


def test_srt_transcript() -> None:
    srt = (
        "1\n00:00:01,000 --> 00:00:02,000\nJo: First\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\nmore words\n"
    )
    assert clean_transcript(srt) == "Jo: First more words"


def test_plain_text_is_untouched() -> None:
    assert clean_transcript("  Notes: we agreed on X.\n") == "Notes: we agreed on X."


def test_example_transcript() -> None:
    cleaned = clean_transcript((EXAMPLES / "meeting.vtt").read_text())
    assert cleaned.splitlines()[0].startswith("Priya: Okay, let's start.")
    assert "-->" not in cleaned


# ----------------------------------------------------------------------- email


def test_load_example_folder() -> None:
    emails = load_emails([EXAMPLES / "emails"])
    assert len(emails) == 4
    assert emails[0].subject == "Payment retries failing for EU customers"
    assert emails[0].sender == "Priya Sharma <priya@acme.example>"
    assert "14 tickets" in emails[0].body

    prompt = emails_to_prompt(emails)
    assert prompt.count("<email ref=") == 4
    assert '<email ref="3">' in prompt


def test_html_email_and_mbox(tmp_path: Path) -> None:
    message = EmailMessage()
    message["From"] = "a@example.com"
    message["Subject"] = "HTML only"
    message.set_content("<p>Hello <b>world</b></p><script>x()</script>", subtype="html")
    eml = tmp_path / "html.eml"
    eml.write_bytes(bytes(message))
    assert load_emails([eml])[0].body == "Hello world"

    box_path = tmp_path / "box.mbox"
    box = mailbox.mbox(box_path)
    for subject in ("one", "two"):
        msg = EmailMessage()
        msg["Subject"] = subject
        msg.set_content(f"body {subject}")
        box.add(msg)
    box.close()
    emails = load_emails([box_path])
    assert [e.subject for e in emails] == ["one", "two"]
    assert emails[1].source == "box.mbox#2"


def test_email_errors(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="No such file"):
        load_emails([tmp_path / "missing.eml"])
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(InputError, match=r"No \.eml"):
        load_emails([empty])
    with pytest.raises(InputError, match="empty"):
        from_text("   ")


# -------------------------------------------------------------------- calendar


def ics(*events: str) -> str:
    return "BEGIN:VCALENDAR\r\n" + "".join(events) + "END:VCALENDAR\r\n"


def vevent(*lines: str) -> str:
    return "BEGIN:VEVENT\r\n" + "".join(f"{line}\r\n" for line in lines) + "END:VEVENT\r\n"


def test_simple_events_and_timezones() -> None:
    text = ics(
        vevent("DTSTART:20261019T083000Z", "DTEND:20261019T090000Z", "SUMMARY:Sync\\, weekly"),
        vevent("DTSTART;VALUE=DATE:20261019", "DTEND;VALUE=DATE:20261020", "SUMMARY:Offsite"),
        vevent("DTSTART:20261020T083000Z", "DURATION:PT1H", "SUMMARY:Tomorrow"),
        vevent("DTSTART:20261019T100000Z", "STATUS:CANCELLED", "SUMMARY:Cancelled"),
    )
    events = parse_ics(text, UTC)
    found = events_on(events, date(2026, 10, 19), IST)
    assert [o.summary for o in found] == ["Offsite", "Sync, weekly"]
    assert f"{found[1].start:%H:%M}" == "14:00"  # 08:30 UTC in IST
    assert format_occurrences(found).splitlines() == [
        "- All day: Offsite",
        "- 14:00-14:30 Sync, weekly",
    ]


def test_folded_lines_and_midnight_end() -> None:
    text = ics(
        vevent(
            "DTSTART:20261019T230000Z",
            "DTEND:20261020T000000Z",
            "SUMMARY:Late call with a very long",
            "  title",
        )
    )
    events = parse_ics(text, UTC)
    assert events[0].summary == "Late call with a very long title"
    assert len(events_on(events, date(2026, 10, 19), UTC)) == 1
    assert events_on(events, date(2026, 10, 20), UTC) == []


@pytest.mark.parametrize(
    ("rule", "extra", "day", "expected"),
    [
        ("FREQ=WEEKLY;BYDAY=MO,WE", (), date(2026, 10, 21), True),
        ("FREQ=WEEKLY;BYDAY=MO,WE", (), date(2026, 10, 20), False),
        ("FREQ=WEEKLY;INTERVAL=2", (), date(2026, 10, 12), False),
        ("FREQ=WEEKLY;INTERVAL=2", (), date(2026, 10, 19), True),
        ("FREQ=DAILY;COUNT=3", (), date(2026, 10, 7), True),
        ("FREQ=DAILY;COUNT=3", (), date(2026, 10, 8), False),
        ("FREQ=DAILY;UNTIL=20261010T000000Z", (), date(2026, 10, 11), False),
        ("FREQ=DAILY", ("EXDATE:20261019T090000Z",), date(2026, 10, 19), False),
        ("FREQ=MONTHLY", (), date(2026, 11, 5), True),
        ("FREQ=MONTHLY;BYDAY=2TU", (), date(2026, 11, 10), False),
        ("FREQ=YEARLY", (), date(2027, 10, 5), True),
    ],
)
def test_recurrence_rules(rule: str, extra: tuple[str, ...], day: date, expected: bool) -> None:
    # Starts Monday 5 October 2026.
    text = ics(
        vevent(
            "DTSTART:20261005T090000Z",
            "DTEND:20261005T093000Z",
            f"RRULE:{rule}",
            *extra,
            "SUMMARY:R",
        )
    )
    found = events_on(parse_ics(text, UTC), day, UTC)
    assert bool(found) is expected
    if found:
        assert found[0].recurring
        assert f"{found[0].start:%H:%M}" == "09:00"


def test_example_calendar_on_a_thursday() -> None:
    events = parse_ics((EXAMPLES / "calendar.ics").read_text(), IST)
    found = events_on(events, date(2026, 10, 22), IST)
    assert [o.summary for o in found] == ["Team standup", "Lunch", "Product sync (demo slot)"]
    assert f"{found[0].start:%H:%M}" == "10:00"
