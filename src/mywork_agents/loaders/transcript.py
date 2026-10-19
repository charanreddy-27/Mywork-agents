"""Normalise meeting transcripts (WebVTT, SRT or plain text) into compact speaker lines."""

from __future__ import annotations

import re

_TIMING = re.compile(
    r"^\s*(\d{1,2}:)?\d{1,2}:\d{2}([.,]\d{1,3})?\s*-->\s*(\d{1,2}:)?\d{1,2}:\d{2}([.,]\d{1,3})?"
)
_VOICE = re.compile(r"<v(?:\.[^\s>]+)?\s+([^>]+)>(.*?)(?:</v>|$)")
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_SPEAKER = re.compile(r"^([^:]{1,60}):\s+(.*)$")


def is_subtitle_format(text: str) -> bool:
    if text.lstrip().startswith("WEBVTT"):
        return True
    return any(_TIMING.match(line) for line in text[:4000].splitlines())


def clean_transcript(text: str) -> str:
    """Strip cue numbers, timestamps and markup; merge consecutive lines by one speaker."""
    if not is_subtitle_format(text):
        return text.strip()

    raw_lines = text.splitlines()
    lines: list[str] = []
    skipping_note = False
    for index, raw in enumerate(raw_lines):
        line = raw.strip()
        if not line:
            skipping_note = False
            continue
        if line.startswith("WEBVTT") or line.startswith(("STYLE", "REGION")):
            continue
        if line.startswith("NOTE"):
            skipping_note = True
            continue
        if skipping_note or _TIMING.match(line):
            continue
        following = raw_lines[index + 1].strip() if index + 1 < len(raw_lines) else ""
        if (line.isdigit() or _looks_like_cue_id(line)) and _TIMING.match(following):
            continue
        voice = _VOICE.match(line)
        if voice:
            line = f"{voice.group(1).strip()}: {voice.group(2)}"
        line = _TAG.sub("", line).strip()
        if line:
            lines.append(line)
    return "\n".join(_merge_speakers(lines))


def _looks_like_cue_id(line: str) -> bool:
    return len(line) < 64 and " " not in line and "-" in line


def _merge_speakers(lines: list[str]) -> list[str]:
    merged: list[str] = []
    current_speaker: str | None = None
    for line in lines:
        match = _SPEAKER.match(line)
        if match and match.group(1) == current_speaker and merged:
            merged[-1] += " " + match.group(2)
        elif match:
            current_speaker = match.group(1)
            merged.append(line)
        elif merged and current_speaker is not None:
            merged[-1] += " " + line
        else:
            merged.append(line)
    return merged
