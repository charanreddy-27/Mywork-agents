# Writing your own agents

An agent is a subclass of `mywork_agents.Agent` with a few class attributes. Everything else
(streaming, the tool loop, structured output, retries, history, the CLI) comes for free.

## 1. The agent

```python
from pydantic import BaseModel, Field

from mywork_agents import Agent
from mywork_agents.render import bullets, join_sections


class Retro(BaseModel):
    went_well: list[str]
    to_improve: list[str]
    experiments: list[str] = Field(description="Concrete things to try next sprint.")


class RetroAgent(Agent[Retro]):
    name = "retro"                       # used in config ([agents.retro]) and `mywork run retro`
    title = "Sprint retro"
    description = "Turn sprint notes into a retrospective."
    group = "Planning"                   # shown in `mywork agents`
    instructions = "You facilitate blameless sprint retrospectives. Be specific and kind."
    output_model = Retro                 # omit for a free-text (Markdown) answer
    default_effort = "low"               # low | medium | high | xhigh | max

    def task_from_text(self, text: str) -> str:
        return f"Write a retro from these notes.\n\n<notes>\n{text}\n</notes>"

    def render(self, output: Retro) -> str:
        return join_sections(
            "# Retro",
            "**Went well**\n\n" + bullets(output.went_well),
            "**To improve**\n\n" + bullets(output.to_improve),
            "**Experiments**\n\n" + bullets(output.experiments),
        )
```

Tips:

- **Wrap user content in tags** (`<notes>`, `<diff>`, ...). The base system prompt tells Claude
  to treat tagged content as data, which protects against prompt injection.
- **Describe fields.** `Field(description=...)` text is sent to Claude as part of the schema.
- **Avoid `dict` fields** in output models; structured outputs need fixed properties. Use a list
  of small models instead.

## 2. Tools

Tools are typed functions. The docstring is the description Claude sees, so say *when* to use it.

```python
import subprocess

from pydantic import BaseModel, Field

from mywork_agents import ToolContext, tool
from mywork_agents.errors import ToolError


class TicketArgs(BaseModel):
    key: str = Field(description="Ticket key such as PAY-123.")


@tool()
def get_ticket(ctx: ToolContext, args: TicketArgs) -> str:
    """Fetch a ticket's title and status. Use it whenever notes mention a ticket key."""
    result = subprocess.run(["jira", "view", args.key], capture_output=True, text=True)
    if result.returncode != 0:
        raise ToolError(f"Ticket {args.key} not found")  # reported to Claude as is_error
    return result.stdout
```

Then add `tools = (get_ticket,)` to the agent. Inputs are validated against `TicketArgs` before
your function runs; exceptions become error results that Claude can react to. Built-in tools
live in `mywork_agents.tools`: `read_file`, `list_dir`, `search_files`, `git_log`, `git_show`,
`fetch_url` and `recent_runs` (earlier outputs from history).

Define argument models at module level so their type hints can be resolved.

## 3. Test it offline

`FakeLLM` replays scripted responses, so tests need no network or API key:

```python
from mywork_agents import AgentContext, Settings
from mywork_agents.testing import FakeLLM


def test_retro():
    llm = FakeLLM([FakeLLM.json({"went_well": ["CI"], "to_improve": [], "experiments": []})])
    ctx = AgentContext(settings=Settings(), make_llm=lambda model: llm)

    result = RetroAgent().run(ctx, RetroAgent().task_from_text("CI was fast"))

    assert "- CI" in result.markdown
    assert "<notes>" in llm.requests[0].messages[0]["content"]
```

Script tool calls with `FakeLLM.tool_use("get_ticket", {"key": "PAY-1"})`, followed by the
final answer.

## 4. Ship it as a plugin

Expose the class through an entry point in your package:

```toml
[project.entry-points."mywork_agents.agents"]
retro = "my_package.agents:RetroAgent"
```

After `pip install`, `mywork agents` lists it and you can run it on files, arguments or stdin:

```bash
mywork run retro -f sprint-notes.md
cat notes.md | mywork run retro --json
```

Configure it like any built-in agent:

```toml
[agents.retro]
model = "claude-sonnet-5-5"
instructions = "Our team uses the 'start, stop, continue' format."
```
