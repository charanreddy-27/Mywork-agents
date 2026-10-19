from __future__ import annotations

from pathlib import Path

import pytest

from mywork_agents.config import (
    CONFIG_TEMPLATE,
    Profile,
    Settings,
    load_settings,
    user_config_path,
)
from mywork_agents.errors import ConfigError
from mywork_agents.models import DEFAULT_MODEL


def write_user_config(text: str) -> Path:
    path = user_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_defaults(isolated_env: Path) -> None:
    settings = load_settings(isolated_env)
    assert settings.model == DEFAULT_MODEL
    assert settings.history is True
    assert settings.fallbacks is True


def test_precedence_user_project_env_overrides(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_user_config('model = "user-model"\neffort = "low"\nmax_turns = 5\n')
    (isolated_env / "mywork.toml").write_text('model = "project-model"\n')

    assert load_settings(isolated_env).model == "project-model"
    assert load_settings(isolated_env).max_turns == 5

    monkeypatch.setenv("MYWORK_MODEL", "env-model")
    assert load_settings(isolated_env).model == "env-model"

    settings = load_settings(isolated_env, overrides={"model": "cli-model", "effort": None})
    assert settings.model == "cli-model"
    assert settings.effort == "low"  # None overrides are ignored


def test_explicit_config_file_is_used_alone(isolated_env: Path, tmp_path: Path) -> None:
    write_user_config('model = "user-model"\n')
    explicit = tmp_path / "other.toml"
    explicit.write_text('model = "explicit-model"\n')
    assert load_settings(isolated_env, config_file=explicit).model == "explicit-model"

    with pytest.raises(ConfigError, match="not found"):
        load_settings(isolated_env, config_file=tmp_path / "missing.toml")


def test_invalid_toml_and_values_are_reported(isolated_env: Path) -> None:
    path = write_user_config("model = \n")
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_settings(isolated_env)

    path.write_text('effort = "extreme"\n')
    with pytest.raises(ConfigError, match=r"effort"):
        load_settings(isolated_env)

    path.write_text("unknown_key = 1\n")
    with pytest.raises(ConfigError, match=str(path)):
        load_settings(isolated_env)


def test_env_booleans(isolated_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MYWORK_HISTORY", "off")
    assert load_settings(isolated_env).history is False
    monkeypatch.setenv("MYWORK_HISTORY", "maybe")
    with pytest.raises(ConfigError, match="MYWORK_HISTORY"):
        load_settings(isolated_env)


def test_resolve_prefers_agent_then_global_then_default() -> None:
    settings = Settings.model_validate(
        {"effort": "high", "agents": {"review": {"model": "claude-sonnet-5-5", "max_turns": 7}}}
    )
    review = settings.resolve(
        "review", default_effort="low", default_max_tokens=1000, default_max_turns=3
    )
    assert (review.model, review.effort, review.max_tokens, review.max_turns) == (
        "claude-sonnet-5-5",
        "high",
        1000,
        7,
    )
    other = settings.resolve(
        "plan", default_effort="low", default_max_tokens=1, default_max_turns=2
    )
    assert other.model == DEFAULT_MODEL


def test_profile_prompt() -> None:
    prompt = Profile(name="Asha", role="SRE", focus=["latency"]).as_prompt()
    assert "Name: Asha" in prompt
    assert "Current focus areas: latency" in prompt
    assert "Working hours: 09:00-17:30" in prompt


def test_template_is_valid_config(isolated_env: Path) -> None:
    write_user_config(CONFIG_TEMPLATE)
    assert load_settings(isolated_env).model == DEFAULT_MODEL
