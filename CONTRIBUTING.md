# Contributing

Thanks for helping improve mywork-agents!

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make check   # format check, lint, strict type check, tests
```

The test suite never calls the Claude API: agent runs use `mywork_agents.testing.FakeLLM`, and
the SDK integration tests use a mocked HTTP transport. No API key is needed.

## Conventions

- **Type hints everywhere**; `mypy --strict` must pass.
- **Format and lint with ruff** (`make fmt`).
- **Errors shown to users** are `MyworkError` subclasses with a message that says what to do next.
- **Prompts** say what good output looks like and why; avoid long lists of rules. Wrap user
  content in XML-style tags.
- **Tools** are read-only and validate their inputs. Anything that writes (commits, files) belongs
  in the CLI, behind an explicit flag and a confirmation.

## Adding a built-in agent

1. Create `src/mywork_agents/agents/<name>.py` with the output model, instructions, `build_task`,
   `task_from_text` and `render` (see [docs/custom-agents.md](docs/custom-agents.md)).
2. Register it in `BUILTIN_AGENTS` in `agents/__init__.py`.
3. Add a CLI command in `cli.py` that gathers inputs and calls `_execute`.
4. Add tests in `tests/test_agents.py` and `tests/test_cli.py`, and document it in the README.
