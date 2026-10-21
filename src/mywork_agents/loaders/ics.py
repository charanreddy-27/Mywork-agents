"""A small iCalendar (.ics) reader: the events on a given day, including recurrences.

Supported: timed and all-day events; TZID (IANA, Windows and Outlook display names),
UTC and floating times; RRULEs with FREQ=DAILY/WEEKLY/MONTHLY/YEARLY plus INTERVAL,
BYDAY (daily/weekly), BYMONTH, BYMONTHDAY, WKST, UNTIL and COUNT; EXDATE; and
RECURRENCE-ID overrides (moved or cancelled single occurrences).

Rules this reader cannot expand exactly (BYSETPOS, positional BYDAY such as "2TU",
BYWEEKNO, ...) only yield their first occurrence rather than guessed dates.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_WEEKDAYS = ("MO", "TU", "WE", "TH", "FR", "SA", "SU")
_MAX_EXPANSION_DAYS = 366 * 20
_UNSUPPORTED_PARTS = ("BYSETPOS", "BYYEARDAY", "BYWEEKNO", "BYHOUR", "BYMINUTE", "BYSECOND")

# Windows time zone IDs (Outlook / Exchange exports) for the most common zones.
_WINDOWS_ZONES = {
    "dateline standard time": "Etc/GMT+12",
    "hawaiian standard time": "Pacific/Honolulu",
    "alaskan standard time": "America/Anchorage",
    "pacific standard time": "America/Los_Angeles",
    "mountain standard time": "America/Denver",
    "us mountain standard time": "America/Phoenix",
    "central standard time": "America/Chicago",
    "central america standard time": "America/Guatemala",
    "eastern standard time": "America/New_York",
    "atlantic standard time": "America/Halifax",
    "sa pacific standard time": "America/Bogota",
    "e. south america standard time": "America/Sao_Paulo",
    "argentina standard time": "America/Buenos_Aires",
    "utc": "UTC",
    "gmt standard time": "Europe/London",
    "greenwich standard time": "Atlantic/Reykjavik",
    "w. europe standard time": "Europe/Berlin",
    "romance standard time": "Europe/Paris",
    "central europe standard time": "Europe/Budapest",
    "central european standard time": "Europe/Warsaw",
    "e. europe standard time": "Europe/Chisinau",
    "gtb standard time": "Europe/Bucharest",
    "fle standard time": "Europe/Kyiv",
    "turkey standard time": "Europe/Istanbul",
    "israel standard time": "Asia/Jerusalem",
    "south africa standard time": "Africa/Johannesburg",
    "russian standard time": "Europe/Moscow",
    "arabian standard time": "Asia/Dubai",
    "pakistan standard time": "Asia/Karachi",
    "india standard time": "Asia/Kolkata",
    "sri lanka standard time": "Asia/Colombo",
    "nepal standard time": "Asia/Kathmandu",
    "bangladesh standard time": "Asia/Dhaka",
    "se asia standard time": "Asia/Bangkok",
    "china standard time": "Asia/Shanghai",
    "singapore standard time": "Asia/Singapore",
    "taipei standard time": "Asia/Taipei",
    "w. australia standard time": "Australia/Perth",
    "tokyo standard time": "Asia/Tokyo",
    "korea standard time": "Asia/Seoul",
    "aus central standard time": "Australia/Darwin",
    "aus eastern standard time": "Australia/Sydney",
    "e. australia standard time": "Australia/Brisbane",
    "new zealand standard time": "Pacific/Auckland",
}
# Outlook display names such as "(UTC-05:00) Eastern Time (US & Canada)".
_DISPLAY_NAME = re.compile(r"^\(UTC(?:([+-])(\d{1,2}):(\d{2}))?\)\s*(.*)$", re.IGNORECASE)
_DISPLAY_KEYWORDS = (
    ("eastern time", "America/New_York"),
    ("central time", "America/Chicago"),
    ("mountain time", "America/Denver"),
    ("pacific time", "America/Los_Angeles"),
    ("london", "Europe/London"),
    ("dublin", "Europe/Dublin"),
    ("amsterdam", "Europe/Berlin"),
    ("berlin", "Europe/Berlin"),
    ("paris", "Europe/Paris"),
    ("brussels", "Europe/Brussels"),
    ("madrid", "Europe/Madrid"),
    ("chennai", "Asia/Kolkata"),
    ("kolkata", "Asia/Kolkata"),
    ("mumbai", "Asia/Kolkata"),
    ("new delhi", "Asia/Kolkata"),
    ("beijing", "Asia/Shanghai"),
    ("singapore", "Asia/Singapore"),
    ("tokyo", "Asia/Tokyo"),
    ("seoul", "Asia/Seoul"),
    ("sydney", "Australia/Sydney"),
    ("melbourne", "Australia/Melbourne"),
    ("auckland", "Pacific/Auckland"),
    ("coordinated universal time", "UTC"),
)


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


@dataclass
class _Parsed:
    event: CalendarEvent
    uid: str | None
    recurrence_id: datetime | date | None
    cancelled: bool
    overrides: list[date] = field(default_factory=list)


def parse_ics(text: str, default_tz: tzinfo) -> list[CalendarEvent]:
    parsed: list[_Parsed] = []
    current: dict[str, list[tuple[list[str], str]]] | None = None
    for line in _unfold(text):
        upper = line.upper()
        if upper == "BEGIN:VEVENT":
            current = {}
        elif upper == "END:VEVENT":
            if current is not None:
                item = _build_event(current, default_tz)
                if item is not None:
                    parsed.append(item)
            current = None
        elif current is not None:
            prop = _split_property(line)
            if prop is not None:
                name, params, value = prop
                current.setdefault(name, []).append((params, value))
    return _apply_overrides(parsed)


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


def local_timezone() -> tzinfo:
    """The system's IANA zone (so DST is right on other dates), else its current offset."""
    name = os.environ.get("TZ", "").lstrip(":")
    if name:
        try:
            return ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError):
            pass
    localtime = Path("/etc/localtime")
    try:
        target = str(localtime.resolve())
        if "zoneinfo/" in target:
            return ZoneInfo(target.split("zoneinfo/", 1)[1])
        with localtime.open("rb") as fh:
            return ZoneInfo.from_file(fh, key="localtime")
    except (OSError, ValueError, ZoneInfoNotFoundError):
        pass
    tz = datetime.now().astimezone().tzinfo
    return tz if tz is not None else timezone.utc


# ------------------------------------------------------------------ parsing


def _unfold(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if raw.startswith((" ", "\t")) and lines:
            lines[-1] += raw[1:]  # keep the rest verbatim: the fold may follow a space
        elif raw.strip():
            lines.append(raw)
    return [line.strip() for line in lines]


def _split_property(line: str) -> tuple[str, list[str], str] | None:
    """Split ``NAME;PARAM="a:b";X=y:value`` honouring quotes (TZIDs may contain ':')."""
    in_quotes = False
    for index, char in enumerate(line):
        if char == '"':
            in_quotes = not in_quotes
        elif char == ":" and not in_quotes:
            head, value = line[:index], line[index + 1 :]
            break
    else:
        return None
    parts: list[str] = []
    buffer = ""
    in_quotes = False
    for char in head:
        if char == '"':
            in_quotes = not in_quotes
            buffer += char
        elif char == ";" and not in_quotes:
            parts.append(buffer)
            buffer = ""
        else:
            buffer += char
    parts.append(buffer)
    return parts[0].upper(), parts[1:], value


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
        pass
    windows = _WINDOWS_ZONES.get(name.strip().lower())
    if windows:
        return ZoneInfo(windows)
    display = _DISPLAY_NAME.match(name.strip())
    if display:
        sign, hours, minutes, label = display.groups()
        lowered = label.lower()
        for keyword, zone in _DISPLAY_KEYWORDS:
            if keyword in lowered:
                return ZoneInfo(zone)
        offset = timedelta(hours=int(hours or 0), minutes=int(minutes or 0))
        return timezone(-offset if sign == "-" else offset)
    return default_tz


def _parse_value(params: list[str], value: str, default_tz: tzinfo) -> datetime | date:
    value = value.strip()
    if (_param(params, "VALUE") or "").upper() == "DATE" or len(value) == 8:
        return datetime.strptime(value[:8], "%Y%m%d").date()
    if value.upper().endswith("Z"):
        return datetime.strptime(value[:15], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    parsed = datetime.strptime(value[:15], "%Y%m%dT%H%M%S")
    return parsed.replace(tzinfo=_zone(_param(params, "TZID"), default_tz))


def _local_date(value: datetime | date, start: datetime | date) -> date:
    """The calendar date of ``value`` in the zone of the series that starts at ``start``."""
    if isinstance(value, datetime):
        if isinstance(start, datetime) and start.tzinfo is not None:
            return value.astimezone(start.tzinfo).date()
        return value.date()
    return value


def _build_event(
    props: dict[str, list[tuple[list[str], str]]], default_tz: tzinfo
) -> _Parsed | None:
    if "DTSTART" not in props:
        return None
    try:
        start_params, start_value = props["DTSTART"][0]
        start = _parse_value(start_params, start_value, default_tz)
        end: datetime | date | None = None
        if "DTEND" in props:
            end = _parse_value(*props["DTEND"][0], default_tz)
        elif "DURATION" in props:
            end = start + _parse_duration(props["DURATION"][0][1])
        recurrence_id = (
            _parse_value(*props["RECURRENCE-ID"][0], default_tz)
            if "RECURRENCE-ID" in props
            else None
        )
    except ValueError:
        return None  # skip events with dates we cannot read

    rrule = None
    if "RRULE" in props:
        rrule = {
            key.upper(): value
            for key, value in (
                part.split("=", 1) for part in props["RRULE"][0][1].split(";") if "=" in part
            )
        }
    exdates: set[date] = set()
    for params, value in props.get("EXDATE", []):
        for item in value.split(","):
            try:
                exdates.add(_local_date(_parse_value(params, item, default_tz), start))
            except ValueError:
                continue

    summary = _unescape(props.get("SUMMARY", [([], "(no title)")])[0][1]) or "(no title)"
    location = _unescape(props["LOCATION"][0][1]) if "LOCATION" in props else None
    status = props.get("STATUS", [([], "")])[0][1].strip().upper()
    uid = props["UID"][0][1].strip() if "UID" in props else None
    event = CalendarEvent(summary, start, end, location or None, rrule, frozenset(exdates))
    return _Parsed(event, uid, recurrence_id, status == "CANCELLED")


def _apply_overrides(parsed: list[_Parsed]) -> list[CalendarEvent]:
    """Fold RECURRENCE-ID instances into their series: the original slot is excluded and
    the override (if not cancelled) stands on its own."""
    masters = {
        item.uid: item
        for item in parsed
        if item.uid and item.recurrence_id is None and item.event.rrule is not None
    }
    events: list[CalendarEvent] = []
    for item in parsed:
        if item.recurrence_id is None:
            continue
        master = masters.get(item.uid) if item.uid else None
        if master is not None:
            master.overrides.append(_local_date(item.recurrence_id, master.event.start))
        if not item.cancelled:
            events.append(item.event)
    for item in parsed:
        if item.recurrence_id is not None or item.cancelled:
            continue
        event = item.event
        if item.overrides:
            event = replace(event, exdates=event.exdates | frozenset(item.overrides))
        events.append(event)
    return events


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
            candidate = _last_occurrence_start(event, day, length)
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
        _, until_instant = _until(event)
        # Expand in the event's own zone so a 09:00 meeting stays at 09:00 across DST;
        # look back far enough to catch occurrences that started days earlier.
        for offset in range(-(duration.days + 1), 2):
            candidate_day = day + timedelta(days=offset)
            if not _occurs_on(event, candidate_day):
                continue
            series_start = datetime.combine(candidate_day, start.timetz())
            if until_instant is not None and series_start > until_instant:
                continue
            occ_start = series_start.astimezone(tz)
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


def _last_occurrence_start(event: CalendarEvent, day: date, length: int) -> date | None:
    for back in range(length):
        candidate = day - timedelta(days=back)
        if _occurs_on(event, candidate):
            return candidate
    return None


def _until(event: CalendarEvent) -> tuple[date | None, datetime | None]:
    """UNTIL as (last date in the series' zone, exact instant if it has a time)."""
    raw = (event.rrule or {}).get("UNTIL")
    if not raw:
        return None, None
    try:
        if "T" not in raw.upper():
            return datetime.strptime(raw[:8], "%Y%m%d").date(), None
        instant = datetime.strptime(raw[:15], "%Y%m%dT%H%M%S")
    except ValueError:
        return None, None
    if raw.upper().endswith("Z"):
        instant = instant.replace(tzinfo=timezone.utc)
    elif isinstance(event.start, datetime) and event.start.tzinfo is not None:
        instant = instant.replace(tzinfo=event.start.tzinfo)  # floating: the series' zone
    else:
        return instant.date(), None
    return _local_date(instant, event.start), instant


def _int_set(rule: dict[str, str], key: str) -> set[int]:
    values: set[int] = set()
    for item in rule.get(key, "").split(","):
        try:
            values.add(int(item))
        except ValueError:
            continue
    return values


def _days_in_month(day: date) -> int:
    following = (day.replace(day=28) + timedelta(days=4)).replace(day=1)
    return (following - timedelta(days=1)).day


def _month_day_matches(rule: dict[str, str], start: date, day: date) -> bool:
    wanted = _int_set(rule, "BYMONTHDAY")
    if not wanted:
        return day.day == start.day
    last = _days_in_month(day)
    return any(day.day == (value if value > 0 else last + value + 1) for value in wanted)


def _supported(rule: dict[str, str]) -> bool:
    freq = rule.get("FREQ", "").upper()
    if freq not in ("DAILY", "WEEKLY", "MONTHLY", "YEARLY"):
        return False
    if any(part in rule for part in _UNSUPPORTED_PARTS):
        return False
    codes = [code.strip().upper() for code in rule.get("BYDAY", "").split(",") if code.strip()]
    if any(code[:-2] for code in codes):  # positional ("2TU", "-1FR")
        return False
    return not (codes and freq in ("MONTHLY", "YEARLY"))


def _matches(rule: dict[str, str], start: date, day: date) -> bool:
    freq = rule.get("FREQ", "").upper()
    interval = max(int(rule.get("INTERVAL", "1") or 1), 1)
    byday = {code.strip()[-2:].upper() for code in rule.get("BYDAY", "").split(",") if code}
    months = _int_set(rule, "BYMONTH")
    if months and day.month not in months:
        return False
    weekday = _WEEKDAYS[day.weekday()]
    if freq == "DAILY":
        if "BYMONTHDAY" in rule and not _month_day_matches(rule, start, day):
            return False
        return (day - start).days % interval == 0 and (not byday or weekday in byday)
    if freq == "WEEKLY":
        week_start = _WEEKDAYS.index(rule.get("WKST", "MO").upper()[:2] or "MO")

        def week_of(value: date) -> date:
            return value - timedelta(days=(value.weekday() - week_start) % 7)

        weeks = (week_of(day) - week_of(start)).days // 7
        wanted = byday or {_WEEKDAYS[start.weekday()]}
        return weeks % interval == 0 and weekday in wanted
    if freq == "MONTHLY":
        elapsed = (day.year - start.year) * 12 + day.month - start.month
        return elapsed % interval == 0 and _month_day_matches(rule, start, day)
    if freq == "YEARLY":
        if not months and day.month != start.month:
            return False
        return (day.year - start.year) % interval == 0 and _month_day_matches(rule, start, day)
    return False


def _occurs_on(event: CalendarEvent, day: date) -> bool:
    rule = event.rrule or {}
    start = event.start.date() if isinstance(event.start, datetime) else event.start
    if day < start or day in event.exdates:
        return False
    if not _supported(rule):
        return day == start  # cannot expand this rule exactly: keep only the first instance
    until_date, _ = _until(event)
    if until_date is not None and day > until_date:
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
