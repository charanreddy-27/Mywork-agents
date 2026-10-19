"""What mywork-agents knows about Claude models: request capabilities and list prices.

Unknown model IDs are assumed to be current-generation models (adaptive thinking,
effort control); they just get no cost estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mywork_agents.core.llm import Usage

DEFAULT_MODEL = "claude-opus-5-5"


@dataclass(frozen=True)
class ModelInfo:
    id: str
    input_per_mtok: float | None = None
    output_per_mtok: float | None = None
    cache_read_per_mtok: float | None = None
    adaptive_thinking: bool = True
    effort: bool = True
    # web_search_20260209 / web_fetch_20260209 (dynamic filtering)
    dynamic_web_tools: bool = False
    # server-side refusal fallbacks (`fallbacks: "default"`)
    refusal_fallbacks: bool = False

    @property
    def known(self) -> bool:
        return self.input_per_mtok is not None


_MODELS: dict[str, ModelInfo] = {
    m.id: m
    for m in (
        ModelInfo("claude-fable-5-1", 10.0, 50.0, 0.25, refusal_fallbacks=True),
        ModelInfo("claude-fable-5", 10.0, 50.0),
        ModelInfo(
            "claude-opus-5-5", 4.0, 20.0, 0.20, dynamic_web_tools=True, refusal_fallbacks=True
        ),
        ModelInfo("claude-opus-5", 5.0, 25.0, dynamic_web_tools=True, refusal_fallbacks=True),
        ModelInfo("claude-opus-4-8", 5.0, 25.0, dynamic_web_tools=True),
        ModelInfo("claude-opus-4-7", 5.0, 25.0, dynamic_web_tools=True),
        ModelInfo("claude-opus-4-6", 5.0, 25.0, dynamic_web_tools=True),
        ModelInfo(
            "claude-sonnet-5-5", 2.0, 10.0, 0.20, dynamic_web_tools=True, refusal_fallbacks=True
        ),
        ModelInfo("claude-sonnet-5", 2.0, 10.0, dynamic_web_tools=True),
        ModelInfo("claude-sonnet-4-6", 3.0, 15.0, dynamic_web_tools=True),
        ModelInfo("claude-haiku-5-5", 0.10, 0.50),
        ModelInfo("claude-haiku-4-5", 1.0, 5.0, adaptive_thinking=False, effort=False),
    )
}

# Models older than the 4.6 generation take neither adaptive thinking nor effort.
_LEGACY_PREFIXES = (
    "claude-3",
    "claude-opus-4-0",
    "claude-opus-4-1",
    "claude-opus-4-5",
    "claude-sonnet-4-0",
    "claude-sonnet-4-5",
    # dated Claude 4.0 IDs such as claude-sonnet-4-20250514
    "claude-opus-4-2",
    "claude-sonnet-4-2",
)


def model_info(model_id: str) -> ModelInfo:
    """Return capabilities for ``model_id``, falling back to sensible defaults."""
    if model_id in _MODELS:
        return _MODELS[model_id]
    if model_id.startswith(_LEGACY_PREFIXES):
        return ModelInfo(model_id, adaptive_thinking=False, effort=False)
    return ModelInfo(model_id)


def known_models() -> list[ModelInfo]:
    return list(_MODELS.values())


def estimate_cost(model_id: str, usage: Usage) -> float | None:
    """Estimate the USD cost of ``usage`` from list prices, or None for unknown models.

    Cache writes are priced at 1.25x input (5-minute TTL). Server-tool fees such as
    web search are not included.
    """
    info = model_info(model_id)
    if info.input_per_mtok is None or info.output_per_mtok is None:
        return None
    cache_read = (
        info.cache_read_per_mtok
        if info.cache_read_per_mtok is not None
        else info.input_per_mtok * 0.1
    )
    total = (
        usage.input_tokens * info.input_per_mtok
        + usage.output_tokens * info.output_per_mtok
        + usage.cache_read_input_tokens * cache_read
        + usage.cache_creation_input_tokens * info.input_per_mtok * 1.25
    )
    return total / 1_000_000
