"""README writer: explores a project with file tools and documents what it finds."""

from __future__ import annotations

import re

from mywork_agents.core.agent import Agent
from mywork_agents.core.llm import LLMResponse, Source
from mywork_agents.tools.files import FILE_TOOLS

INSTRUCTIONS = """\
You write accurate, genuinely useful README files by exploring the project first.

Explore efficiently: list the tree, then read the manifests and entry points (pyproject.toml, \
package.json, Cargo.toml, go.mod, Makefile, Dockerfile, CI workflows, main modules) and grep for \
configuration such as environment variables. Only document what you verified in the files: \
install and run commands must come from the project's own scripts and manifests, and you must \
not invent features, badges, URLs or benchmarks.

A good README answers, in this order: what the project is and who it is for (one or two \
sentences), key features, quick start (prerequisites, install, first run), usage with real \
examples, configuration, project structure (brief), development (tests, lint) and license (only \
if a license file exists). Prefer short sections and copy-pasteable commands. If an existing \
README is provided, keep what is still accurate and its voice, and fix what is outdated or \
missing.

Reply with the README Markdown only: no preamble and no surrounding code fence."""

_FENCED = re.compile(r"^```(?:markdown|md)?\s*\n(.*)\n```\s*$", re.DOTALL)


class ReadmeAgent(Agent[str]):
    name = "readme"
    title = "README writer"
    description = "Explore a project and write (or refresh) its README."
    group = "Development"
    instructions = INSTRUCTIONS
    tools = FILE_TOOLS
    default_max_tokens = 64_000
    default_max_turns = 30

    def build_task(self, *, existing: str | None = None, notes: str | None = None) -> str:
        parts = ["Write the README for the project in the working directory."]
        if notes:
            parts.append(f"Notes from the author: {notes}")
        if existing and existing.strip():
            parts.append(f"<existing_readme>\n{existing.strip()}\n</existing_readme>")
        return "\n\n".join(parts)

    def task_from_text(self, text: str) -> str:
        return self.build_task(notes=text)

    def finalize(self, response: LLMResponse, sources: list[Source]) -> str:
        text = response.text.strip()
        match = _FENCED.match(text)
        return (match.group(1) if match else text).strip() + "\n"

    def render(self, output: str) -> str:
        return output
