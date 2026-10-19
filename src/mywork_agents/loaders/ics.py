"""A small iCalendar (.ics) reader: the events on a given day, including common recurrences.

Supported: timed and all-day events, TZID/UTC/floating times, and RRULEs with
FREQ=DAILY/WEEKLY/MONTHLY/YEARLY plus INTERVAL, BYDAY, UNTIL, COUNT and EXDATE.
Monthly rules by weekday position (e.g. "second Tuesday") are not expanded.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_WEEKDAYS = ("MO", "TU", "WE", "TH", "FR", "SA", "SU")
_MAX_EXPANSION_DAYS = 366 * 20


@dataclass(frozen=True)
class CalendarEvent:
    summary: str
    start: datetime | date
    end: datetime | date | None = None
    location: str | None = None
    rrule: dict[str, str] | None = None
    exdates: frozenset[date] = frozenset()

    @property
    def all_day(self) -> bool:
        return not isinstance(self.start, datetime)


@dataclass(frozen=True)
class Occurrence:
    summary: str
    start: datetime | None  # None for all-day events
    end: datetime | None
    location: str | None
    recurring: bool


def parse_ics(text: str, default_tz: tzinfo) -> list[CalendarEvent]:
    events: list[CalendarEvent] = []
    current: dict[str, list[tuple[list[str], str]]] | None = None
    for line in _unfold(text):
        if line == "BEGIN:VEVENT":
            current = {}
        elif line == "END:VEVENT":
            if current is not None:
                event = _build_event(current, default_tz)
                if event is not None:
                    events.append(event)
            current = None
        elif current is not None and ":" in line:
            name_and_params, value = line.split(":", 1)
            name, *params = name_and_params.split(";")
            current.setdefault(name.upper(), []).append((params, value))
    return events


def events_on(events: list[CalendarEvent], day: date, tz: tzinfo) -> list[Occurrence]:
    """Occurrences of ``events`` on ``day`` (in ``tz``), sorted by start time."""
    found: list[Occurrence] = []
    for event in events:
        occurrence = _occurrence_on(event, day, tz)
        if occurrence is not None:
            found.append(occurrence)
    found.sort(key=lambda o: (o.start is not None, o.start or datetime.min.replace(tzinfo=tz)))
    return found


def format_occurrences(occurrences: list[Occurrence]) -> str:
    lines = []
    for item in occurrences:
        where = f" ({item.location})" if item.location else ""
        repeat = " [recurring]" if item.recurring else ""
        if item.start is None:
            lines.append(f"- All day: {item.summary}{where}{repeat}")
        else:
            end = f"-{item.end:%H:%M}" if item.end else ""
            lines.append(f"- {item.start:%H:%M}{end} {item.summary}{where}{repeat}")
    return "\n".join(lines)


# ------------------------------------------------------------------ parsing


def _unfold(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if raw.startswith((" ", "\t")) and lines:
            lines[-1] += raw[1:]
        elif raw.strip():
            lines.append(raw.strip())
    return lines


def _unescape(value: str) -> str:
    return (
        value.replace("\\n", "\n")
        .replace("\\N", "\n")
        .replace("\\,", ",")
        .replace("\\;", ";")
        .replace("\\\\", "\\")
    )


def _param(params: list[str], key: str) -> str | None:
    for item in params:
        if "=" in item:
            name, value = item.split("=", 1)
            if name.upper() == key:
                return value.strip('"')
    return None


def _zone(name: str | None, default_tz: tzinfo) -> tzinfo:
    if not name:
        return default_tz
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return default_tz  # e.g. Windows zone names exported by Outlook


def _parse_value(params: list[str], value: str, default_tz: tzinfo) -> datetime | date:
    value = value.strip()
    if _param(params, "VALUE") == "DATE" or len(value) == 8:
        return datetime.strptime(value[:8], "%Y%m%d").date()
    if value.endswith("Z"):
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    parsed = datetime.strptime(value[:15], "%Y%m%dT%H%M%S")
    return parsed.replace(tzinfo=_zone(_param(params, "TZID"), default_tz))


def _build_event(
    props: dict[str, list[tuple[list[str], str]]], default_tz: tzinfo
) -> CalendarEvent | None:
    if "DTSTART" not in props:
        return None
    if props.get("STATUS", [([], "")])[0][1].upper() == "CANCELLED":
        return None
    try:
        start_params, start_value = props["DTSTART"][0]
        start = _parse_value(start_params, start_value, default_tz)
        end: datetime | date | None = None
        if "DTEND" in props:
            end = _parse_value(*props["DTEND"][0], default_tz)
        elif "DURATION" in props:
            end = start + _parse_duration(props["DURATION"][0][1])
    except ValueError:
        return None  # skip events with dates we cannot read

    rrule = None
    if "RRULE" in props:
        rrule = dict(part.split("=", 1) for part in props["RRULE"][0][1].split(";") if "=" in part)
    exdates: set[date] = set()
    for params, value in props.get("EXDATE", []):
        for item in value.split(","):
            try:
                parsed = _parse_value(params, item, default_tz)
            except ValueError:
                continue
            exdates.add(parsed.date() if isinstance(parsed, datetime) else parsed)

    summary = _unescape(props.get("SUMMARY", [([], "(no title)")])[0][1]) or "(no title)"
    location = _unescape(props["LOCATION"][0][1]) if "LOCATION" in props else None
    return CalendarEvent(summary, start, end, location or None, rrule, frozenset(exdates))


def _parse_duration(value: str) -> timedelta:
    value = value.strip().lstrip("+")
    sign = -1 if value.startswith("-") else 1
    value = value.lstrip("-").lstrip("P")
    days = hours = minutes = seconds = weeks = 0
    number = ""
    in_time = False
    for char in value:
        if char == "T":
            in_time = True
        elif char.isdigit():
            number += char
        elif number:
            amount = int(number)
            number = ""
            if char == "W":
                weeks = amount
            elif char == "D":
                days = amount
            elif char == "H" and in_time:
                hours = amount
            elif char == "M" and in_time:
                minutes = amount
            elif char == "S" and in_time:
                seconds = amount
    return sign * timedelta(weeks=weeks, days=days, hours=hours, minutes=minutes, seconds=seconds)


# --------------------------------------------------------------- recurrence


def _occurrence_on(event: CalendarEvent, day: date, tz: tzinfo) -> Occurrence | None:
    recurring = event.rrule is not None
    if event.all_day:
        start_day = event.start if not isinstance(event.start, datetime) else event.start.date()
        end_day = event.end if not isinstance(event.end, datetime) else None
        length = max((end_day - start_day).days, 1) if end_day else 1
        if recurring:
            candidate = _last_occurrence_start(event, start_day, day, length)
            if candidate is None:
                return None
        elif not (start_day <= day < start_day + timedelta(days=length)):
            return None
        return Occurrence(event.summary, None, None, event.location, recurring)

    assert isinstance(event.start, datetime)
    start = event.start
    duration = (
        event.end - start
        if isinstance(event.end, datetime) and event.end >= start
        else timedelta(0)
    )
    if recurring:
        # Expand in the event's own zone so a 09:00 meeting stays at 09:00 across DST.
        for offset in (-1, 0, 1):
            candidate_day = day + timedelta(days=offset)
            if _occurs_on(event, start.date(), candidate_day):
                occ_start = datetime.combine(candidate_day, start.timetz()).astimezone(tz)
                occ_end = (occ_start + duration).astimezone(tz)
                if _spans(occ_start, occ_end, day):
                    return Occurrence(event.summary, occ_start, occ_end, event.location, True)
        return None

    occ_start = start.astimezone(tz)
    occ_end = (start + duration).astimezone(tz)
    if _spans(occ_start, occ_end, day):
        return Occurrence(event.summary, occ_start, occ_end, event.location, False)
    return None


def _spans(start: datetime, end: datetime, day: date) -> bool:
    # An event ending exactly at midnight does not spill into the next day.
    last = (end - timedelta(microseconds=1)).date() if end > start else start.date()
    return start.date() <= day <= last


def _last_occurrence_start(
    event: CalendarEvent, start_day: date, day: date, length: int
) -> date | None:
    for back in range(length):
        candidate = day - timedelta(days=back)
        if _occurs_on(event, start_day, candidate):
            return candidate
    return None


def _until(rule: dict[str, str]) -> date | None:
    raw = rule.get("UNTIL")
    if not raw:
        return None
    try:
        return datetime.strptime(raw[:8], "%Y%m%d").date()
    except ValueError:
        return None


def _matches(rule: dict[str, str], start: date, day: date) -> bool:
    freq = rule.get("FREQ", "").upper()
    interval = max(int(rule.get("INTERVAL", "1") or 1), 1)
    byday = {code[-2:].upper() for code in rule.get("BYDAY", "").split(",") if code}
    weekday = _WEEKDAYS[day.weekday()]
    if freq == "DAILY":
        return (day - start).days % interval == 0 and (not byday or weekday in byday)
    if freq == "WEEKLY":
        start_monday = start - timedelta(days=start.weekday())
        day_monday = day - timedelta(days=day.weekday())
        weeks = (day_monday - start_monday).days // 7
        wanted = byday or {_WEEKDAYS[start.weekday()]}
        return weeks % interval == 0 and weekday in wanted
    if freq == "MONTHLY":
        if byday and any(code[:-2] for code in rule.get("BYDAY", "").split(",")):
            return False  # positional weekday rules ("2TU") are not expanded
        months = (day.year - start.year) * 12 + day.month - start.month
        return months % interval == 0 and day.day == start.day
    if freq == "YEARLY":
        return (day.year - start.year) % interval == 0 and (day.month, day.day) == (
            start.month,
            start.day,
        )
    return False


def _occurs_on(event: CalendarEvent, start: date, day: date) -> bool:
    rule = event.rrule or {}
    if day < start or day in event.exdates:
        return False
    until = _until(rule)
    if until is not None and day > until:
        return False
    if not _matches(rule, start, day):
        return False
    count = rule.get("COUNT")
    if count is None:
        return True
    limit = int(count)
    seen = 0
    current = start
    while current <= day and (current - start).days <= _MAX_EXPANSION_DAYS:
        if _matches(rule, start, current):
            seen += 1
            if seen > limit:
                return False
        current += timedelta(days=1)
    return seen <= limit


def local_timezone() -> tzinfo:
    tz = datetime.now().astimezone().tzinfo
    return tz if tz is not None else timezone.utc
