"""Web access: Anthropic's server-side search/fetch tools and a local page fetcher."""

from __future__ import annotations

import ipaddress
import re
import socket
from collections.abc import Iterable
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

import httpcore2
import httpx2
from pydantic import BaseModel, Field

from mywork_agents._version import __version__
from mywork_agents.core.tools import ToolContext, tool
from mywork_agents.errors import ToolError
from mywork_agents.models import model_info

USER_AGENT = f"mywork-agents/{__version__} (+https://github.com/charanreddy-27/Mywork-agents)"
FETCH_TIMEOUT_S = 20.0
MAX_DOWNLOAD_BYTES = 5_000_000
MAX_REDIRECTS = 5
_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})


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
    SKIP = frozenset({"script", "style", "noscript", "svg", "template", "iframe"})
    # Tags that can only appear in the body: they implicitly close an unclosed <head>.
    BODY_ONLY = frozenset({"body", "p", "div", "main", "article", "section", "table", "ul", "ol"})
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
        self._in_head = False
        self._in_title = False

    @property
    def _hidden(self) -> bool:
        return self._skip_depth > 0 or self._in_head

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "head":
            self._in_head = True
        elif tag in self.BODY_ONLY or re.fullmatch(r"h[1-6]", tag):
            self._in_head = False  # </head> is optional in HTML5
        if tag == "title" and not self._skip_depth:
            self._in_title = True  # ignores <title> inside inline <svg> icons
        if tag in self.SKIP:
            self._skip_depth += 1
        elif self._hidden:
            return
        elif re.fullmatch(r"h[1-6]", tag):
            self.parts.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in ("td", "th"):
            self.parts.append(" | ")
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag == "head":
            self._in_head = False
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif not self._hidden and (tag in self.BLOCK or re.fullmatch(r"h[1-6]", tag)):
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._hidden:
            self.parts.append(data)


def html_to_text(html: str) -> tuple[str, str]:
    """Return ``(title, readable_text)`` for an HTML document."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    text = "".join(parser.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"(?m)^\| ", "", text)  # table rows start with a cell separator
    text = re.sub(r"\n{3,}", "\n\n", text)
    return parser.title.strip(), text.strip()


# ---------------------------------------------------------------- fetch_url


def _check_url(url: str) -> None:
    """Reject non-http(s) URLs and obviously local hosts before any network activity."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname
    except ValueError as exc:
        raise ToolError(f"Not a valid URL: {url!r} ({exc})") from exc
    if parsed.scheme not in ("http", "https") or not host:
        raise ToolError(f"Only http(s) URLs can be fetched: {url!r}")
    host = host.lower().rstrip(".")
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise ToolError("Refusing to fetch a local address.")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return  # a name: vetted when it is resolved, at connect time
    _require_public(address)


def _require_public(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    if not address.is_global:
        raise ToolError("Refusing to fetch a private or local IP address.")


def _resolve_public(host: str, port: int) -> str:
    """Resolve ``host`` and return an address only if every address it maps to is public."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ToolError(f"Could not resolve {host}: {exc}") from exc
    addresses = [str(info[4][0]).split("%", 1)[0] for info in infos]
    if not addresses:
        raise ToolError(f"Could not resolve {host}")
    for address in addresses:
        _require_public(ipaddress.ip_address(address))
    return addresses[0]


class _PublicOnlyBackend(httpcore2.SyncBackend):
    """Resolve, vet and connect in one step.

    The address that passed the check is the one dialled, so neither numeric host
    forms (``http://2130706433/``), internal DNS names nor DNS rebinding can reach a
    private network. TLS still verifies the certificate against the original host.
    """

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore2.NetworkStream:
        return super().connect_tcp(
            _resolve_public(host, port),
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )


def _public_client() -> httpx2.Client:
    # A normal client keeps its HTTP(S)_PROXY / NO_PROXY mounts; only its default
    # (direct-connection) transport gets the guard. Requests through a proxy are
    # resolved by the proxy, so only the literal-address checks apply to them.
    client = httpx2.Client(timeout=FETCH_TIMEOUT_S, headers={"User-Agent": USER_AGENT})
    pool = getattr(getattr(client, "_transport", None), "_pool", None)
    if pool is None or not hasattr(pool, "_network_backend"):  # pragma: no cover
        client.close()
        raise ToolError("Cannot install the network guard on this httpx2 version.")
    pool._network_backend = _PublicOnlyBackend()
    return client


def download(
    url: str,
    *,
    client: httpx2.Client | None = None,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
    allow_private: bool = False,
) -> tuple[str, str, str]:
    """Fetch ``url`` and return ``(final_url, title, text)``.

    Model-chosen URLs must be public. ``allow_private`` is for URLs the user typed
    themselves (such as a self-hosted calendar on the LAN).
    """
    if not allow_private:
        _check_url(url)
    owns_client = client is None
    if client is not None:
        http = client
    elif allow_private:
        http = httpx2.Client(timeout=FETCH_TIMEOUT_S, headers={"User-Agent": USER_AGENT})
    else:
        http = _public_client()
    try:
        current = url
        for _hop in range(MAX_REDIRECTS + 1):
            with http.stream("GET", current, follow_redirects=False) as response:
                location = response.headers.get("location")
                if response.status_code in _REDIRECT_CODES and location:
                    current = str(httpx2.URL(current).join(location))
                    if not allow_private:
                        _check_url(current)  # vet every hop before requesting it
                    continue
                if response.status_code >= 400:
                    raise ToolError(f"HTTP {response.status_code} fetching {current}")
                chunks: list[bytes] = []
                size = 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > max_bytes:
                        raise ToolError(f"{url} is larger than {max_bytes // 1_000_000} MB.")
                    chunks.append(chunk)
                content_type = response.headers.get("content-type", "").lower()
                body = b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")
                break
        else:
            raise ToolError(f"Too many redirects fetching {url}")
    except (httpx2.HTTPError, httpx2.InvalidURL) as exc:
        raise ToolError(f"Could not fetch {url}: {exc}") from exc
    finally:
        if owns_client:
            http.close()

    if "html" in content_type or body.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
        title, text = html_to_text(body)
        return current, title, text
    if (
        content_type.startswith(("text/", "application/json", "application/xml"))
        or not content_type
    ):
        return current, "", body
    raise ToolError(f"Unsupported content type {content_type!r} at {current}")


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
