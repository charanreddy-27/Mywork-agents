"""Built-in agents and plugin discovery.

Third-party packages can add agents by exposing an entry point in the
``mywork_agents.agents`` group that points at an :class:`~mywork_agents.Agent` subclass::

    [project.entry-points."mywork_agents.agents"]
    changelog = "my_package.agents:ChangelogAgent"
"""

from __future__ import annotations

import warnings
from importlib.metadata import entry_points
from typing import Any

from mywork_agents.agents.commit import CommitAgent
from mywork_agents.agents.inbox import InboxAgent
from mywork_agents.agents.meeting import MeetingAgent
from mywork_agents.agents.plan import PlanAgent
from mywork_agents.agents.readme import ReadmeAgent
from mywork_agents.agents.research import ResearchAgent
from mywork_agents.agents.review import ReviewAgent
from mywork_agents.agents.standup import StandupAgent
from mywork_agents.core.agent import Agent
from mywork_agents.errors import InputError

ENTRY_POINT_GROUP = "mywork_agents.agents"

BUILTIN_AGENTS: dict[str, type[Agent[Any]]] = {
    cls.name: cls
    for cls in (
        PlanAgent,
        StandupAgent,
        InboxAgent,
        MeetingAgent,
        ResearchAgent,
        ReviewAgent,
        CommitAgent,
        ReadmeAgent,
    )
}


def discover_agents() -> dict[str, type[Agent[Any]]]:
    """Built-in agents plus any installed plugins (built-ins win name clashes)."""
    found: dict[str, type[Agent[Any]]] = {}
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            cls = ep.load()
        except Exception as exc:  # a broken plugin must not break the CLI
            warnings.warn(f"Could not load agent plugin {ep.name!r}: {exc}", stacklevel=2)
            continue
        if isinstance(cls, type) and issubclass(cls, Agent):
            found[getattr(cls, "name", ep.name)] = cls
        else:
            warnings.warn(f"Plugin {ep.name!r} is not an Agent subclass; skipped.", stacklevel=2)
    return {**found, **BUILTIN_AGENTS}


def get_agent_class(name: str) -> type[Agent[Any]]:
    agents = discover_agents()
    if name not in agents:
        known = ", ".join(sorted(agents))
        raise InputError(f"Unknown agent {name!r}. Available agents: {known}")
    return agents[name]


__all__ = [
    "BUILTIN_AGENTS",
    "CommitAgent",
    "InboxAgent",
    "MeetingAgent",
    "PlanAgent",
    "ReadmeAgent",
    "ResearchAgent",
    "ReviewAgent",
    "StandupAgent",
    "discover_agents",
    "get_agent_class",
]
