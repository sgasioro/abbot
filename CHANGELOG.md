# Changelog

## Unreleased

- **Setup assistant:** the MCP server's `plan_campaign` prompt has the agent talk
  a campaign through with the person before it writes a spec: the knobs and their
  reachable ranges, what's measured, the goal, limits and failure modes, budget,
  earlier results, and who runs each experiment. `spec_guide` opens with the same
  questions, so an agent that never sees the prompt asks them too. An abbot skill
  for Claude Code (`skills/abbot`) connects the server, then hands off.
- **`abbot-mcp setup` / `abbot-mcp start`:** register the server with Claude Code
  and install the skill (asking first), then open a session on the planning
  prompt. The skill ships in the wheel. A bare `abbot-mcp` still serves.
- `benchmarks/agent`: real agent sessions against a simulated anneal rig the agent
  can't read, with and without the skill.

## 0.2.0

- **UI:** the setup panel covers log-scaled variables, travel limits (with the
  current setting as the start), a min/max per output, an optional source
  (labeled, or a fidelity range) with a cost field on each run, and a choice of
  independent or correlated-output GPs. A running campaign shows
  `campaign.check()`. Spec import keeps these fields. Plot titles and axis labels
  render again.
- **Agent access:** `abbot-mcp` (`abbot[mcp]`) serves named campaigns, saved on
  disk, to an MCP client: create, suggest, report results or failures, resolve
  halts, recommend and check the model, and start a narrowed campaign from an
  earlier one's runs (`results_from`). `abbot.agent.Workspace` is the same
  surface from Python. Tested with headless agent sessions on a simulated lab.

## 0.1.0

First release.
