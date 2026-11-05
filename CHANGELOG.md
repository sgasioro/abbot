# Changelog

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
