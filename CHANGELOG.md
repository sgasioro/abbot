# Changelog

## Unreleased

- **Aim an output at a value:** `OutputSpec(name="size", target=40)` (direction
  `TARGET`) finds the setting nearest the value. The model fits the output as
  measured and the policies score its distance from the target, so it works with
  limits, output constraints and a second objective (`ParetoBO`). The best so far,
  the recommendation, the progress plot and the stopping rules follow it
  (`target_reached` reads its value as a tolerance). In the UI, "Near" beside
  Higher and Lower, with the value to aim for; the CSV header and the agent's spec
  guide carry it too.
- **Guided UI:** `python -m abbot.ui` now asks five short questions (what can be
  set, what each run measures, the goal, extras, who runs the experiments) and
  says in plain words what abbot will do. Running, it shows the next design with
  boxes for what was measured (the one after is suggested as soon as you record),
  how it's going in plain language, and the model with the goal marked (its band
  or threshold, the region the model puts in it, the next design). Finish with the
  current best when you're done.
- **Expert panel:** `{ } Expert` opens every field beside the steps: the spec as
  tables, `campaign.json` (the format `abbot-mcp` saves) and the policy and model,
  following the steps as you edit; once running, every run (suggest several,
  hold variables fixed, CSV in and out, click a run to correct a value), the
  plots, and `check()`.
- **Editing campaign.json:** the expert panel's `campaign.json` is editable. Each
  pause in typing checks it; a valid edit fills the steps, an invalid one says why
  and changes nothing. What the steps can't show (whole-number and simplex knobs,
  linear constraints, boxes, several sources, a list of starting designs,
  `max_retries`) is kept and tagged "set in campaign.json" on the steps (the tag
  opens the editor). The first step offers the other ways in: open a saved
  campaign, load a spec or campaign.json, or write campaign.json.
- **Campaigns shared with the agent:** the UI saves each campaign it starts in
  `~/abbot-campaigns` (`--dir` or `ABBOT_UI_DIR` to choose), the folder
  `abbot-mcp` uses, and "Open a saved campaign" carries on any of them, whichever
  started it. A process holds a campaign while it has it open (a `.lock` with its
  process id), so the UI and the agent's server never write the same one at once;
  `Workspace.open` / `close` / `config` / `free_name` are new, and the MCP
  server's `close_campaign` hands a campaign over to the UI. Simulations aren't
  saved.
- A spec's domain entries (and source domains) may leave out `kind` where it's
  plain: `values` is enumerated, `dim` structured, `lower`/`upper` continuous
  (whole numbers still say `"kind": "integer"`). An entry that can't be told says
  what to add, instead of pydantic's "Unable to extract tag".
- Fixed: the `campaign.json` the expert panel showed left out each variable's
  `kind`, so it didn't load back.

## 0.2.1

- **Setup assistant:** the MCP server's `plan_campaign` prompt has the agent talk
  a campaign through with the person before it writes a spec: the knobs and their
  reachable ranges, what's measured, the goal, limits and failure modes, budget,
  earlier results, and who runs each experiment. `spec_guide` opens with the same
  questions, so an agent that never sees the prompt asks them too. An abbot skill
  for Claude Code (`skills/abbot`) connects the server, then hands off.
- **`abbot-mcp setup` / `abbot-mcp start`:** register the server with Claude Code
  and install the skill (asking first), then open a session on the skill, which
  plans the campaign with you. The skill ships in the wheel. A bare `abbot-mcp` still serves.
- **Plots:** `abbot.plots.model` / `progress` / `check` draw a campaign with
  matplotlib: the model's mean and uncertainty with the runs, the next design and
  the goal (its band or threshold, and the region the model puts in it), adapting
  to one, two or more variables; progress; each run's leave-one-out error. The MCP
  server's `plot` tool saves them in the campaign's folder and returns the image.
  The `mcp` extra now includes matplotlib.
- **Correcting a value:** `Campaign.correct(run, outputs, reason=)` fixes a run
  already recorded, keeping the old values on the record (`Record.corrections`);
  `correct_result` on the MCP server.
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
