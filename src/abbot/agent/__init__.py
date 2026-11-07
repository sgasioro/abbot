"""Agent access to abbot: named, saved campaigns, served over MCP.

:class:`Workspace` is the framework-free core; :mod:`abbot.agent.server` exposes it
as an MCP server (``abbot[mcp]``, ``abbot-mcp``).
"""

from abbot.agent.workspace import PLANNING_GUIDE, SPEC_GUIDE, Workspace

__all__ = ["PLANNING_GUIDE", "SPEC_GUIDE", "Workspace"]
