"""The Python examples in the docs must keep working."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from mywork_agents import AgentContext, Settings
from mywork_agents.testing import FakeLLM

ROOT = Path(__file__).resolve().parent.parent


def python_blocks(path: Path) -> list[str]:
    return re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), re.DOTALL)


def test_readme_custom_agent_example() -> None:
    namespace: dict[str, Any] = {}
    exec(python_blocks(ROOT / "README.md")[0], namespace)
    agent = namespace["ChangelogAgent"]()

    llm = FakeLLM([FakeLLM.json({"version": "1.3.0", "added": ["Dark mode"], "fixed": []})])
    result = agent.run(AgentContext(settings=Settings(), make_llm=lambda model: llm), "abc feat")

    assert result.markdown.startswith("## 1.3.0")
    assert "- Dark mode" in result.markdown
    assert {t["name"] for t in llm.requests[0].tools} == {"read_file", "search_files"}


def test_custom_agents_guide_examples() -> None:
    namespace: dict[str, Any] = {"__name__": "docs_example"}
    for block in python_blocks(ROOT / "docs" / "custom-agents.md"):
        exec(block, namespace)
    namespace["test_retro"]()
    assert namespace["get_ticket"].name == "get_ticket"
