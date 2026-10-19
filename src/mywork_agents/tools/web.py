"""Web access: Anthropic's server-side search/fetch tools and a local page fetcher."""

from __future__ import annotations

import ipaddress
import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

import httpx2
from pydantic import BaseModel, Field

from mywork_agents._version import __version__
from mywork_agents.core.tools import ToolContext, tool
from mywork_agents.errors import ToolError
from mywork_agents.models import model_info

USER_AGENT = f"mywork-agents/{__version__} (+https://github.com/charanreddy-27/Mywork-agents)"
FETCH_TIMEOUT_S = 20.0
MAX_DOWNLOAD_BYTES = 5_000_000


def web_server_tools(model: str, *, max_uses: int, fetch: bool = True) -> list[dict[str, Any]]:
    """Server-side web tools in the newest version ``model`` supports.

    Models with dynamic filtering get ``web_search_20260209`` and ``web_fetch_20260209``;
    older models get basic search only (pair it with the local :func:`fetch_url` tool).
    """
    if model_info(model).dynamic_web_tools:
        tools: list[dict[str, Any]] = [
            {"type": "web_search_20260209", "name": "web_search", "max_uses": max_uses}
        ]
        if fetch:
            tools.append({"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": max_uses})
        return tools
    return [{"type": "web_search_20250305", "name": "web_search", "max_uses": max_uses}]


def supports_server_fetch(model: str) -> bool:
    return model_info(model).dynamic_web_tools


# ------------------------------------------------------------ HTML -> text


class _TextExtractor(HTMLParser):
    SKIP = frozenset({"script", "style", "noscript", "svg", "template", "iframe", "head"})
    BLOCK = frozenset(
        {
            "p",
            "div",
            "section",
            "article",
            "main",
            "header",
            "footer",
            "br",
            "tr",
            "table",
            "ul",
            "ol",
            "pre",
            "blockquote",
            "hr",
            "nav",
            "aside",
            "figure",
        }
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
        if tag in self.SKIP:
            self._skip_depth += 1
        elif re.fullmatch(r"h[1-6]", tag):
            self.parts.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self.BLOCK or re.fullmatch(r"h[1-6]", tag):
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skip_depth:
            self.parts.append(data)


def html_to_text(html: str) -> tuple[str, str]:
    """Return ``(title, readable_text)`` for an HTML document."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    text = "".join(parser.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return parser.title.strip(), text.strip()


# ---------------------------------------------------------------- fetch_url


def _check_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ToolError(f"Only http(s) URLs can be fetched: {url!r}")
    host = parsed.hostname.lower()
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        raise ToolError("Refusing to fetch a local address.")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return
    if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved:
        raise ToolError("Refusing to fetch a private or local IP address.")


def download(
    url: str, *, client: httpx2.Client | None = None, max_bytes: int = MAX_DOWNLOAD_BYTES
) -> tuple[str, str, str]:
    """Fetch ``url`` and return ``(final_url, title, text)``."""
    _check_url(url)
    owns_client = client is None
    http = client or httpx2.Client(
        follow_redirects=True, timeout=FETCH_TIMEOUT_S, headers={"User-Agent": USER_AGENT}
    )
    try:
        with http.stream("GET", url) as response:
            _check_url(str(response.url))  # redirects must not land somewhere local
            if response.status_code >= 400:
                raise ToolError(f"HTTP {response.status_code} fetching {url}")
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > max_bytes:
                    raise ToolError(f"{url} is larger than {max_bytes // 1_000_000} MB.")
                chunks.append(chunk)
            content_type = response.headers.get("content-type", "").lower()
            body = b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")
            final_url = str(response.url)
    except httpx2.HTTPError as exc:
        raise ToolError(f"Could not fetch {url}: {exc}") from exc
    finally:
        if owns_client:
            http.close()

    if "html" in content_type or body.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
        title, text = html_to_text(body)
        return final_url, title, text
    if (
        content_type.startswith(("text/", "application/json", "application/xml"))
        or not content_type
    ):
        return final_url, "", body
    raise ToolError(f"Unsupported content type {content_type!r} at {url}")


class FetchUrlArgs(BaseModel):
    url: str = Field(description="The http(s) URL to fetch.")


@tool()
def fetch_url(ctx: ToolContext, args: FetchUrlArgs) -> str:
    """Download a web page and return its readable text (HTML is converted to plain text).

    Use this to read a specific URL the user gave you or one you need to verify.
    """
    client = ctx.extra.get("http_client")
    final_url, title, text = download(args.url, client=client)
    header = f"URL: {final_url}\n" + (f"Title: {title}\n" if title else "")
    return f"{header}\n{text}" if text else f"{header}\n[The page has no readable text.]"
