"""Load emails from .eml, .mbox and plain-text files (or folders of them)."""

from __future__ import annotations

import mailbox
from collections.abc import Sequence
from dataclasses import dataclass
from email import policy
from email.message import EmailMessage, MIMEPart
from email.parser import BytesParser
from pathlib import Path

from mywork_agents.errors import InputError
from mywork_agents.tools.web import html_to_text

EMAIL_SUFFIXES = (".eml", ".mbox", ".txt", ".md")


@dataclass(frozen=True)
class Email:
    sender: str
    to: str
    date: str
    subject: str
    body: str
    source: str


def _mbox_bundle(path: Path) -> Path | None:
    """The mailbox file inside an Apple Mail export (a folder named ``Inbox.mbox``)."""
    inner = path / "mbox"
    if path.is_dir() and path.suffix.lower() == ".mbox" and inner.is_file():
        return inner
    return None


def load_emails(paths: Sequence[Path]) -> list[Email]:
    emails: list[Email] = []
    for path in paths:
        bundle = _mbox_bundle(path)
        if bundle is not None:
            emails.extend(_load_mbox(bundle, path.name))
        elif path.is_dir():
            found = False
            for entry in sorted(path.iterdir()):
                inner = _mbox_bundle(entry)
                if inner is not None:
                    emails.extend(_load_mbox(inner, entry.name))
                    found = True
                elif entry.is_file() and entry.suffix.lower() in EMAIL_SUFFIXES:
                    emails.extend(_load_file(entry))
                    found = True
            if not found:
                raise InputError(f"No .eml, .mbox, .txt or .md files in {path}")
        elif path.is_file():
            emails.extend(_load_file(path))
        else:
            raise InputError(f"No such file or directory: {path}")
    if not emails:
        raise InputError("No emails found in the given input.")
    return emails


def _load_file(path: Path) -> list[Email]:
    suffix = path.suffix.lower()
    if suffix == ".eml":
        with path.open("rb") as fh:
            message = BytesParser(policy=policy.default).parse(fh)
        return [_from_message(message, path.name)]
    if suffix == ".mbox":
        return _load_mbox(path, path.name)
    return [from_text(path.read_text(encoding="utf-8", errors="replace"), path.name)]


def _load_mbox(path: Path, label: str) -> list[Email]:
    box = mailbox.mbox(path, create=False)
    try:
        return [
            _from_message(
                BytesParser(policy=policy.default).parsebytes(item.as_bytes()),
                f"{label}#{index}",
            )
            for index, item in enumerate(box, 1)
        ]
    finally:
        box.close()


def from_text(text: str, source: str = "stdin") -> Email:
    """A pasted email or thread; headers (if any) are left in the body for Claude to read."""
    if not text.strip():
        raise InputError(f"{source} is empty.")
    return Email(sender="", to="", date="", subject="", body=text.strip(), source=source)


def _from_message(message: EmailMessage, source: str) -> Email:
    return Email(
        sender=str(message.get("From", "")),
        to=str(message.get("To", "")),
        date=str(message.get("Date", "")),
        subject=str(message.get("Subject", "")),
        body=_body(message),
        source=source,
    )


def _body(message: EmailMessage, depth: int = 0) -> str:
    text = _part_text(message.get_body(preferencelist=("plain", "html")))
    if not text:  # e.g. an empty text/plain alternative next to the real HTML body
        text = _part_text(message.get_body(preferencelist=("html",)))
    if depth < 3:  # forwarded-as-attachment messages carry the real content
        for attachment in message.iter_attachments():
            if attachment.get_content_type() != "message/rfc822":
                continue
            inner = attachment.get_content()
            if isinstance(inner, EmailMessage):
                forwarded = (
                    "--- Forwarded message ---\n"
                    f"From: {inner.get('From', '')}\nSubject: {inner.get('Subject', '')}\n\n"
                    f"{_body(inner, depth + 1)}"
                )
                text = f"{text}\n\n{forwarded}" if text else forwarded
    return text.strip()


def _part_text(part: MIMEPart | None) -> str:
    if part is None:
        return ""
    try:
        content = part.get_content()
    except (LookupError, ValueError):  # unknown charset or broken encoding
        payload = part.get_payload(decode=True)
        content = payload.decode("utf-8", errors="replace") if isinstance(payload, bytes) else ""
    if not isinstance(content, str):
        return ""
    if part.get_content_subtype() == "html":
        _, content = html_to_text(content)
    return content.strip()


def emails_to_prompt(emails: Sequence[Email]) -> str:
    blocks = []
    for ref, mail in enumerate(emails, 1):
        headers = [
            f"{label}: {value}"
            for label, value in (
                ("From", mail.sender),
                ("To", mail.to),
                ("Date", mail.date),
                ("Subject", mail.subject),
            )
            if value
        ]
        head = "\n".join(headers)
        blocks.append(
            f'<email ref="{ref}">\n{head}\n\n{mail.body}\n</email>'.replace("\n\n\n", "\n\n")
        )
    return "\n\n".join(blocks)
