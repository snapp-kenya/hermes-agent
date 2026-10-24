"""The agent that is currently executing a tool call.

Registry tool handlers receive ``(args, **kwargs)`` -- there is no agent handle in
that signature. Two skills tools need one:

  * ``skill_run`` must delegate, and ``delegate_task`` requires a parent agent.
  * ``skill_view`` must know whether its caller is the top-level agent (which must
    never receive a skill body) or a worker (whose context is disposable).

Rather than change every dispatch signature, ``agent/tool_executor.py`` binds the
executing agent here for the duration of each tool call. Handlers read it back
through :func:`current_agent` / :func:`agent_depth`.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import Any, Iterator, Optional

_current_agent: ContextVar[Optional[Any]] = ContextVar("hermes_current_agent", default=None)


def current_agent() -> Optional[Any]:
    """The agent executing the current tool call, or None outside a tool call."""
    return _current_agent.get()


def agent_depth() -> int:
    """Delegation depth of the current agent: 0 = top level, >=1 = worker."""
    agent = _current_agent.get()
    if agent is None:
        return 0
    return int(getattr(agent, "_delegate_depth", 0) or 0)


@contextlib.contextmanager
def bind_agent(agent: Any) -> Iterator[None]:
    token = _current_agent.set(agent)
    try:
        yield
    finally:
        _current_agent.reset(token)
