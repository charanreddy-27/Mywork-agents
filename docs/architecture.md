# Architecture

mywork-agents is a small framework (`mywork_agents.core`) plus eight agents built on it and a
CLI that wires inputs to agents.

```
src/mywork_agents/
├── cli.py              Typer app: one command per agent, plus run/agents/history/config/doctor
├── config.py           Settings, profile, [files] defaults, per-agent overrides, TOML + env loading
├── models.py           Model capabilities (thinking, effort, web tool versions, fallbacks) and prices
├── history.py          Local JSON log of runs; feeds earlier plans/standups back into prompts
├── render.py           Markdown helpers used by agents' render()
├── testing.py          FakeLLM: scripted responses for offline tests
├── core/
│   ├── agent.py        Agent base class and the tool-use loop
│   ├── llm.py          LLM protocol and AnthropicLLM (the only module that calls the API)
│   ├── tools.py        @tool decorator, argument validation, output limits
│   ├── schema.py       Pydantic model -> strict JSON Schema for structured outputs
│   └── events.py       Progress events rendered by the CLI
├── tools/              Built-in read-only tools: files, git, web, history
├── loaders/            Email (.eml/.mbox), transcripts (VTT/SRT), iCalendar (with recurrence)
└── agents/             plan, standup, inbox, meeting, research, review, commit, readme
```

## Life of a run

1. **The CLI gathers inputs deterministically.** Files are read, transcripts cleaned, calendars
   expanded for the target day, git logs and diffs collected, and recent history attached.
   Doing this in code (rather than asking Claude to fetch it) makes runs faster, cheaper and
   reproducible; `--dry-run` shows the result.
2. **The agent builds the request.** `system` = agent instructions + a data-handling rule + the
   user's profile + per-agent instructions from config. The first user message = a context
   header (current date, time zone, working directory) + the task, with user content wrapped in
   XML-style tags such as `<diff>` and `<email>`.
3. **`AnthropicLLM` streams it.** Every request is streamed (no HTTP timeouts on long turns) and
   uses:
   - adaptive thinking, with `output_config.effort` as the depth/cost dial;
   - `output_config.format` (JSON Schema) for agents with an output model;
   - top-level `cache_control` so each tool-loop turn reuses the cached prefix;
   - `fallbacks: "default"` (beta `server-side-fallback-2026-07-01`) on models that support it,
     so a declined request is retried on Anthropic's recommended fallback model;
   - `eager_input_streaming` on client tools, with inputs validated by Pydantic before use.
4. **The loop handles every stop reason.**
   - `tool_use`: run all requested tools in parallel and return every `tool_result` in one
     user message, in order. Invalid input or a failing tool becomes an `is_error` result, so
     Claude can correct itself instead of the run crashing.
   - `pause_turn` (long server-side web searches): append the turn and continue.
   - `refusal`: raise `RefusalError` with the category.
   - `max_tokens`: raise a clear error instead of returning a truncated answer.
   - unparseable streamed tool JSON: re-issue the turn (bounded).
   - `end_turn`: validate against the output model; on a mismatch, send one repair request.
5. **The CLI renders and records.** Markdown to stdout (or JSON), a usage/cost line to stderr,
   and the run saved to history.

## Design decisions

**A hand-written loop instead of the SDK tool runner.** The loop is ~150 lines and gives us
things the runner doesn't: `pause_turn` resumption for server tools, structured-output
validation with a repair turn, progress events, and an `LLM` protocol small enough that every
agent can be tested offline with `FakeLLM`.

**Structured outputs, not "please reply in JSON".** Agents declare a Pydantic model;
`core/schema.py` converts it to the strict subset the API accepts (inlined `$ref`s,
`additionalProperties: false`, unsupported numeric/length constraints removed). Pydantic then
validates the full model client-side, so nothing is lost.

**Free text where citations matter.** The research agent returns Markdown rather than JSON,
because citations from web search are attached to text blocks. Sources are collected from the
citations (falling back to search results).

**Model-aware requests.** `models.py` records what each model supports, so switching `--model`
to an older model drops adaptive thinking and effort, and picks basic web search plus the local
`fetch_url` tool instead of the dynamic-filtering web tools.

## Security model

- **Read-only tools.** Agents can list, read and search files, and read git history. The only
  write operations are performed by the CLI on your explicit request: `commit --apply` and
  `readme --write` ask for confirmation first (unless `--yes`); `-o FILE` writes or overwrites
  the file you name.
- **Workspace sandbox.** File paths are resolved and must stay inside the workspace (symlinks
  included, both for reading and for listing). Version-control folders (`.git`, `.hg`, `.svn`)
  are off limits, and likely secrets (`.env*`, `*.env`, `.env*/` folders, keys,
  `credentials.json`, `.npmrc`, ...) are never read or searched, whether reached by their own
  name or through a symlink.
- **Secrets stay out of diffs.** `git_show` cannot print file contents (`rev:path`), is limited to
  the workspace, and leaves secret files out of patches; the diffs `review` and `commit` send to
  Claude exclude them too.
- **No option injection.** Git revisions coming from the model are validated and paths are
  passed after `--`.
- **SSRF guard.** `fetch_url` only reaches public addresses. Host names are resolved and vetted
  at connect time and the vetted address is the one dialled, so numeric host forms, internal
  DNS names and DNS rebinding cannot reach private, loopback, link-local or shared (100.64/10)
  ranges. Redirects are followed manually and every hop is checked before it is requested;
  downloads are capped at 5 MB. (Behind an HTTP proxy, name resolution happens at the proxy, so
  only literal addresses can be checked.) URLs you type yourself, such as a calendar on your LAN,
  are not restricted.
- **Prompt injection.** User-supplied content is wrapped in tags, and the system prompt tells
  Claude to treat it as data and to flag embedded instructions (the bundled
  `examples/emails/03-suspicious.eml` contains one).
- **Bounded tool output.** Tool results over 30,000 characters are cut with an explicit note
  telling Claude how to page (`offset`/`limit`), never silently.
- **Untrusted text in the terminal.** Anything that came from the model or your files is
  escaped before it is printed with Rich, so it is shown literally rather than parsed as markup.
