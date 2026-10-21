# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-19

First release.

### Added

- `mywork` CLI with eight agents: `plan`, `standup`, `inbox`, `meeting`, `research`, `review`,
  `commit` and `readme`, plus `run` for any agent (including plugins), `agents`, `history`,
  `config` and `doctor`.
- Agent framework: streamed requests, parallel tool calls, `pause_turn` handling, structured
  outputs validated with Pydantic (with one repair round-trip), adaptive thinking with per-agent
  effort, prompt caching and server-side refusal fallbacks.
- Read-only tools sandboxed to the workspace: `read_file`, `list_dir`, `search_files`, `git_log`,
  `git_show`, `fetch_url`, `recent_runs`; web research through Anthropic's server-side web search
  and fetch.
- Loaders for `.eml`/`.mbox` email, WebVTT/SRT transcripts and iCalendar files or URLs with
  recurring events.
- Local run history that lets `plan` and `standup` build on earlier runs.
- Configuration through TOML files, `MYWORK_*` environment variables and flags, including a user
  profile, default input files and per-agent overrides.
- `--json`, `--output`, `--dry-run` on every agent command; `review --fail-on` for CI.
- Plugin agents through the `mywork_agents.agents` entry point group and `FakeLLM` for offline
  tests.

### Security

- Before release, an independent multi-agent review reproduced and verified every finding. Fixed:
  SSRF bypasses in `fetch_url` (numeric hosts, internal DNS names, redirects, 100.64/10), secret
  exposure through `.git/config`, `.env*` variants, symlinks and `git show rev:path`, and secrets in
  review diffs. The same review also covered calendar recurrences and time zones, email and
  transcript parsing, CLI precedence and output safety, history clean-up, commit message wrapping
  and the docs.
