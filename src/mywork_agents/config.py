"""Configuration: defaults < user config file < project ``mywork.toml`` < env vars < CLI flags."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mywork_agents.errors import ConfigError
from mywork_agents.models import DEFAULT_MODEL

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.10
    import tomli as tomllib

Effort = Literal["low", "medium", "high", "xhigh", "max"]
EFFORT_LEVELS: tuple[Effort, ...] = ("low", "medium", "high", "xhigh", "max")

PROJECT_CONFIG_NAME = "mywork.toml"


class Profile(BaseModel):
    """Who you are. Agents use this to personalise plans, replies and standups."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    role: str | None = None
    team: str | None = None
    timezone: str | None = None
    working_hours: str = "09:00-17:30"
    focus: list[str] = Field(default_factory=list)
    writing_style: str | None = None

    def as_prompt(self) -> str:
        lines = []
        if self.name:
            lines.append(f"- Name: {self.name}")
        if self.role:
            lines.append(f"- Role: {self.role}")
        if self.team:
            lines.append(f"- Team: {self.team}")
        if self.timezone:
            lines.append(f"- Time zone: {self.timezone}")
        lines.append(f"- Working hours: {self.working_hours}")
        if self.focus:
            lines.append(f"- Current focus areas: {', '.join(self.focus)}")
        if self.writing_style:
            lines.append(f"- Preferred writing style: {self.writing_style}")
        return "About the user you are working for:\n" + "\n".join(lines)


class Files(BaseModel):
    """Default inputs, so `mywork plan` and `mywork standup` work without flags."""

    model_config = ConfigDict(extra="forbid")

    tasks: list[str] = Field(default_factory=list)
    notes: str | None = None
    # A local .ics file or an http(s)/webcal URL (e.g. a calendar's secret iCal address).
    calendar: str | None = None


class AgentSettings(BaseModel):
    """Per-agent overrides, configured under ``[agents.<name>]``."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    effort: Effort | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    max_turns: int | None = Field(default=None, gt=0)
    instructions: str | None = None


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = DEFAULT_MODEL
    effort: Effort | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    max_turns: int | None = Field(default=None, gt=0)
    fallbacks: bool = True
    eager_tool_streaming: bool = True
    timeout: float = Field(default=600.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    history: bool = True
    history_dir: Path | None = None
    profile: Profile = Field(default_factory=Profile)
    files: Files = Field(default_factory=Files)
    agents: dict[str, AgentSettings] = Field(default_factory=dict)

    def resolve(
        self,
        agent: str,
        *,
        default_effort: Effort,
        default_max_tokens: int,
        default_max_turns: int,
    ) -> ResolvedSettings:
        """Settings for one agent run: agent override > global setting > agent default."""
        override = self.agents.get(agent, AgentSettings())
        return ResolvedSettings(
            model=override.model or self.model,
            effort=override.effort or self.effort or default_effort,
            max_tokens=override.max_tokens or self.max_tokens or default_max_tokens,
            max_turns=override.max_turns or self.max_turns or default_max_turns,
            instructions=override.instructions,
        )

    @property
    def resolved_history_dir(self) -> Path:
        return self.history_dir.expanduser() if self.history_dir else data_dir() / "history"


class ResolvedSettings(BaseModel):
    model: str
    effort: Effort
    max_tokens: int
    max_turns: int
    instructions: str | None = None


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "mywork"


def data_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(base) / "mywork"


def user_config_path() -> Path:
    return config_dir() / "config.toml"


def config_files(cwd: Path, explicit: Path | None = None) -> list[Path]:
    """Config files that apply, lowest precedence first."""
    if explicit is not None:
        explicit = explicit.expanduser()
        if not explicit.is_file():
            raise ConfigError(f"Config file not found: {explicit}")
        return [explicit]
    env_path = os.environ.get("MYWORK_CONFIG")
    if env_path:
        path = Path(env_path).expanduser()
        if not path.is_file():
            raise ConfigError(f"MYWORK_CONFIG points to a missing file: {path}")
        return [path]
    return [p for p in (user_config_path(), cwd / PROJECT_CONFIG_NAME) if p.is_file()]


_PER_AGENT_KEYS = ("model", "effort", "max_tokens", "max_turns")
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _env_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ConfigError(f"{name} must be true or false, got {raw!r}")


def _env_overrides() -> dict[str, Any]:
    data: dict[str, Any] = {}
    for key in ("model", "effort", "max_tokens", "max_turns", "history_dir"):
        value = os.environ.get(f"MYWORK_{key.upper()}")
        if value:
            data[key] = value
    for key in ("history", "fallbacks"):
        flag = _env_bool(f"MYWORK_{key.upper()}")
        if flag is not None:
            data[key] = flag
    return data


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path} is not valid TOML: {exc}") from exc


def load_settings(
    cwd: Path | None = None,
    *,
    config_file: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    cwd = cwd or Path.cwd()
    data: dict[str, Any] = {}
    sources: list[str] = []
    for path in config_files(cwd, config_file):
        data = _deep_merge(data, _read_toml(path))
        sources.append(str(path))
    env = _env_overrides()
    # Env vars outrank every config file, including per-agent sections in them.
    per_agent = {k: env[k] for k in _PER_AGENT_KEYS if k in env}
    if per_agent and isinstance(data.get("agents"), dict):
        data["agents"] = {
            name: {**section, **per_agent} if isinstance(section, dict) else section
            for name, section in data["agents"].items()
        }
    data = _deep_merge(data, env)
    data = _deep_merge(data, {k: v for k, v in (overrides or {}).items() if v is not None})
    try:
        return Settings.model_validate(data)
    except ValidationError as exc:
        where = f" (from {', '.join(sources)})" if sources else ""
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise ConfigError(f"Invalid configuration{where}: {problems}") from exc


CONFIG_TEMPLATE = """\
# mywork-agents configuration
# Location: ~/.config/mywork/config.toml (user) or ./mywork.toml (project).
# Precedence: defaults < user file < project file < MYWORK_* env vars < CLI flags.

model = "claude-opus-5-5"   # any Claude model ID
# effort = "medium"         # low | medium | high | xhigh | max (default: per agent)
# max_tokens = 32000        # per response; thinking counts toward this
# max_turns = 16            # tool-use round trips per run
fallbacks = true            # retry declined requests on Anthropic's recommended fallback model
history = true              # keep a local log of runs (used by `plan` and `standup`)

[profile]
# name = "Your Name"
# role = "Backend engineer"
# team = "Payments"
# timezone = "Asia/Kolkata"
working_hours = "09:00-17:30"
# focus = ["Q4 launch", "on-call hygiene"]
# writing_style = "concise, friendly, no emojis"

[files]
# Defaults for `mywork plan` / `mywork standup` (each can be overridden with a flag).
# tasks = ["~/notes/todo.md"]
# notes = "~/notes/today.md"
# calendar = "~/calendar.ics"   # or your calendar's secret iCal URL (https:// or webcal://)

# Per-agent overrides: model, effort, max_tokens, max_turns, instructions
# [agents.review]
# effort = "high"
# instructions = "We use Django and pytest. Flag missing migrations."

# [agents.standup]
# model = "claude-haiku-5-5"
# instructions = "Keep each bullet under 15 words."
"""


def project_template() -> str:
    """CONFIG_TEMPLATE with every setting commented out.

    A project file outranks the user's own config, so it must not silently reset
    their choices to the defaults; it only documents what can be set.
    """
    lines = []
    for line in CONFIG_TEMPLATE.splitlines():
        stripped = line.lstrip()
        is_setting = stripped and not stripped.startswith(("#", "[")) and "=" in stripped
        lines.append(f"# {line}" if is_setting else line)
    return (
        "\n".join(lines).replace(
            "# mywork-agents configuration",
            "# mywork-agents project configuration: uncomment what this project should override",
            1,
        )
        + "\n"
    )
