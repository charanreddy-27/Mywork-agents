"""Built-in client-side tools agents can use."""

from mywork_agents.tools.files import FILE_TOOLS, list_dir, read_file, search_files
from mywork_agents.tools.git import GIT_TOOLS, git_log, git_show
from mywork_agents.tools.history import recent_runs
from mywork_agents.tools.web import fetch_url

__all__ = [
    "FILE_TOOLS",
    "GIT_TOOLS",
    "fetch_url",
    "git_log",
    "git_show",
    "list_dir",
    "read_file",
    "recent_runs",
    "search_files",
]
