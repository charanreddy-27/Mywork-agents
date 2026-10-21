"""Research brief: web search plus page reading, returned as a cited Markdown report."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from mywork_agents.config import Effort
from mywork_agents.core.agent import Agent
from mywork_agents.core.llm import LLMResponse, Source
from mywork_agents.core.tools import Tool
from mywork_agents.render import join_sections, section
from mywork_agents.tools.web import fetch_url, supports_server_fetch, web_server_tools

Depth = Literal["quick", "standard", "deep"]

# depth -> (effort, maximum web searches/fetches)
DEPTHS: dict[Depth, tuple[Effort, int]] = {
    "quick": ("low", 3),
    "standard": ("medium", 6),
    "deep": ("high", 15),
}


class SourceRef(BaseModel):
    url: str
    title: str | None = None


class ResearchReport(BaseModel):
    question: str
    report: str
    sources: list[SourceRef]


INSTRUCTIONS = """\
You are a careful research analyst. Answer the user's question with a brief they can act on.

Research well: search before answering anything time-sensitive, prefer primary and \
authoritative sources (official docs, filings, papers, first-hand data) over aggregators, \
cross-check important claims against a second source, and note how recent the evidence is. \
Distinguish established facts from estimates, opinions and marketing claims, and say plainly \
when sources disagree or the evidence is thin.

Write the brief in Markdown with these sections:
## TL;DR — two to four bullets with the answer.
## Key findings — the evidence, with specifics (numbers, dates, names).
## Details — only if the question needs more depth.
## Caveats & open questions — limits of the evidence and what would change the conclusion.

Do not add a sources list; one is appended automatically from your citations. If you cannot \
search the web, answer from what you know, state that it reflects your training data and may \
be out of date, and point out which facts the user should verify."""


class ResearchAgent(Agent[ResearchReport]):
    name = "research"
    title = "Research brief"
    description = "Search the web and write a cited brief on any question."
    group = "Research"
    instructions = INSTRUCTIONS
    default_max_tokens = 64_000
    default_max_turns = 16

    def __init__(self, *, depth: Depth = "standard", search: bool = True) -> None:
        self.depth = depth
        self.search = search
        self.question = ""
        self.urls: list[str] = []

    def effort_override(self) -> Effort:
        return DEPTHS[self.depth][0]

    def server_tools(self, model: str) -> list[dict[str, Any]]:
        if not self.search:
            return []
        return web_server_tools(model, max_uses=DEPTHS[self.depth][1])

    def active_tools(self, model: str) -> tuple[Tool, ...]:
        # Use Anthropic's web_fetch when it is available; otherwise fetch pages locally.
        if self.search and supports_server_fetch(model):
            return ()
        return (fetch_url,)

    def build_task(self, question: str, *, urls: list[str] | None = None) -> str:
        self.question = question.strip()
        self.urls = list(urls or [])
        # The question is the user's request, so it is not wrapped in a data tag.
        parts = [f"Research question: {self.question}"]
        if urls:
            listed = "\n".join(f"- {u}" for u in urls)
            parts.append(f"Start by reading these pages:\n{listed}")
        if not self.search:
            parts.append("Web search is turned off for this run.")
        return "\n\n".join(parts)

    def task_from_text(self, text: str) -> str:
        return self.build_task(text)

    def finalize(self, response: LLMResponse, sources: list[Source]) -> ResearchReport:
        # Without citations (e.g. a summary of one page), list the pages the user gave.
        refs = [SourceRef(url=s.url, title=s.title) for s in sources]
        known = {ref.url for ref in refs}
        refs += [SourceRef(url=url) for url in self.urls if url not in known]
        return ResearchReport(question=self.question, report=response.text.strip(), sources=refs)

    def render(self, output: ResearchReport) -> str:
        sources = "\n".join(
            f"{i}. [{s.title or s.url}]({s.url})" for i, s in enumerate(output.sources, 1)
        )
        title = output.question if len(output.question) <= 90 else output.question[:89] + "…"
        return join_sections(
            f"# {title}",
            output.report,
            section("Sources", sources) if sources else "",
        )
