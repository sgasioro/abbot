"""An MCP server over :class:`~abbot.agent.workspace.Workspace` (``abbot[mcp]``).

Run ``abbot-mcp --dir ./campaigns`` (or ``python -m abbot.agent``) and point an MCP
client at it over stdio. Each tool is one call on the workspace; errors come back
to the agent as tool errors with abbot's own message.
"""

from __future__ import annotations

import argparse
import functools
from collections.abc import Callable
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from abbot import __version__
from abbot.agent.workspace import SPEC_GUIDE, Workspace
from abbot.problem import ProblemSpec

INSTRUCTIONS = """\
abbot chooses which experiment to run next. Read spec_guide first, create a campaign,
then loop: suggest, have the design run, report_result (or report_failure). Campaigns
are saved as they go and keep their names across sessions; list_campaigns shows them.
Ask the person before resolving a halt as a bad region, and before creating a
campaign whose spec you inferred rather than were given."""


def build_server(root: str | Path) -> MCPServer:
    """The MCP server for the campaigns saved under ``root``."""
    ws = Workspace(root)
    server = MCPServer(
        "abbot",
        title="abbot",
        description="Goal-driven Bayesian experimental design for the lab.",
        instructions=INSTRUCTIONS,
        version=__version__,
    )

    @_tool(server)
    def spec_guide() -> str:
        """How to write a spec and a goal, with examples. Read this before
        create_campaign."""
        return SPEC_GUIDE

    @_tool(server)
    def spec_schema() -> dict:
        """The full JSON schema of a spec, for fields spec_guide doesn't cover."""
        return ProblemSpec.model_json_schema()

    @_tool(server)
    def create_campaign(
        name: str,
        spec: dict,
        goal: dict | None = None,
        strategy: str = "SwitchBAX",
        model: str = "default",
        initial_designs: list[dict] | None = None,
        max_retries: int = 1,
    ) -> dict:
        """Create and save a campaign. spec and goal are described by spec_guide.
        initial_designs: designs to run first ({variable: value}); with travel
        limits, the first is the current setting. Without them abbot picks a few
        space-filling designs."""
        return ws.create(name, spec, goal, strategy, model, initial_designs, max_retries)

    @_tool(server)
    def list_campaigns() -> list[dict]:
        """Every saved campaign: runs, outstanding designs, and any halt."""
        return ws.campaigns()

    @_tool(server)
    def suggest(name: str, n: int = 1, fixed: dict | None = None) -> list[dict]:
        """The next n designs to run, each with an id to report its result under.
        fixed: {variable: value} to hold some variables at a setting."""
        return ws.suggest(name, n, fixed)

    @_tool(server)
    def report_result(
        name: str, id: str, outputs: dict[str, float], cost: float | None = None
    ) -> dict:
        """Report what was measured for a suggested design: {output: raw value}
        for every output, and the run's cost if known."""
        return ws.report_result(name, id, outputs, cost)

    @_tool(server)
    def report_failure(name: str, id: str, error: str) -> dict:
        """A suggested design couldn't be run (the apparatus failed). It is retried;
        repeated failures halt the campaign for a person to resolve."""
        return ws.report_failure(name, id, error)

    @_tool(server)
    def cancel(name: str, id: str) -> dict:
        """Withdraw a suggested design that won't be run. Nothing is recorded."""
        return ws.cancel(name, id)

    @_tool(server)
    def add_results(name: str, results: list[dict]) -> dict:
        """Results measured outside the loop: a list of {"design": {...},
        "outputs": {...}} with optional "cost"."""
        return ws.add_results(name, results)

    @_tool(server)
    def resolve_halt(name: str, answer: str, reason: str | None = None) -> dict:
        """Answer a halt: "retry" (the cause is fixed), "exclude" (skip that
        design), or "bad_region" (the design itself gives nothing usable; abbot
        steers away from the region). Confirm with the person first."""
        return ws.resolve_halt(name, answer, reason)

    @_tool(server)
    def status(name: str) -> dict:
        """Runs so far, outstanding designs, any halt, and the recommendation."""
        return ws.status(name)

    @_tool(server)
    def recommend(name: str) -> list[dict]:
        """The recommended design(s) with their measured outputs."""
        return ws.recommend(name)

    @_tool(server)
    def check_model(name: str) -> str:
        """Whether the model fits the data, and what looks off (noise, outliers,
        variables that don't matter)."""
        return ws.check(name)

    @_tool(server)
    def history(name: str) -> str:
        """The spec and every result, as CSV."""
        return ws.history(name)

    return server


def _tool(server: MCPServer) -> Callable[[Callable], Callable]:
    """Register a tool whose errors reach the agent with abbot's message.

    The SDK hides the text of an unexpected exception from the client. This server
    runs locally for the person using it, and abbot's messages say what to fix (a
    name that doesn't exist, a halt to resolve), so every error is passed on.
    """

    def register(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # every error goes to the agent
                text = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
                raise ToolError(f"{type(exc).__name__}: {text}") from exc

        return server.tool()(wrapper)

    return register


def main(argv: list[str] | None = None) -> None:
    """Entry point: ``abbot-mcp [--dir DIR]``, serving over stdio."""
    parser = argparse.ArgumentParser(
        prog="abbot-mcp", description="Serve abbot campaigns to an MCP client over stdio."
    )
    parser.add_argument(
        "--dir", default="abbot-campaigns",
        help="where campaigns are saved (default: ./abbot-campaigns)",
    )
    args = parser.parse_args(argv)
    build_server(args.dir).run("stdio")
