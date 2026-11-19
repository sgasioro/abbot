# Changelog

## Unreleased

- **Runs that didn't measure every output:** give the missing one as `None`
  (`ingest({"yield": 0.82, "purity": None})`, `null` from the agent, an empty box
  in the UI, an empty CSV cell). Every output must still be named, so a typo
  can't pass for a missing value; a result with nothing measured is refused. The
  run counts as usual, with NaN stored, and each output's model trains on the
  runs that measured it (every built-in model; a custom `ModelConstructor` opts in
  with `handles_missing = True`, helped by `abbot.model.measured`). The best run
  so far, the observed front and fallbacks without a model skip runs missing an
  objective or a constrained output. `correct` fills a value in later. Output
  constraints ignore a gap in an output they don't use.

- **Neural-network surrogates (experimental):** `abbot.nn` wraps your own
  `torch.nn.Module` as a model: `NetEnsembleConstructor` (a deep ensemble),
  `MCDropoutConstructor` (dropout masks shared across designs, so each sample is
  one function) or `DeepKernelConstructor` (a GP on the network's features). abbot
  scales inputs and outputs, trains with warm starts, skips missing outputs, and
  `featurize=` handles inputs that aren't numbers. InfoBAX and SwitchBAX work with
  the ensemble and dropout models by fine-tuning a copy on each posterior sample's
  values (after nn-bax), measured against a matching fine-tune on the data alone.
- **InfoBAX and SwitchBAX refuse a model that can't be updated** when the campaign
  is set up (a constructor with `supports_conditioning = False`, such as the toy
  `EnsembleConstructor`), instead of failing at the first proposal.
- **InfoBAX with an ensemble** takes distinct members as its samples, not repeated
  draws of the same ones, and uses a model's own `observation_noise_variance` when
  it has one.

## 0.3.0

The UI is rebuilt around a guided setup, with an expert panel for every field and
an editable `campaign.json` it shares with the agent. Your own experiment can run
each design from the UI: Python, a command, or an Xopt evaluator, held to the spec
it declares. Also new: outputs aimed at a value, Steps knobs, PS-BAX, and GPU
support (`device="cuda"`).

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
- **Your own experiment in the UI:** `python -m abbot.ui --experiment rig:Rig`
  (or `rig.py:Rig`) offers your experiment on the Start step as who runs the
  experiments; the run view drives it with run / pause / step / stop, and the
  results are saved as any campaign's are. `Rig` is an object with `run(design)`
  (or `run_batch(designs)`), or a class or function that makes one, given the
  spec if it takes an argument. An experiment that declares its spec (a `spec`
  attribute, or `--spec rig.json`) holds the setup to it: the same knobs and
  measurements, ranges and choices only narrowed, travel limits and design
  constraints kept; the goal is free. `abbot.ui.serve(campaign, experiment)` opens
  the UI from Python on a campaign built in code, with any policy or model, shared
  by every tab.
- **A command as the experiment:** `CommandExperiment("./measure.sh {temp}
  {conc}")` runs a program for each design (a whole-number float fills a
  `{n:d}`) and reads its outputs from the JSON on
  its last line (`"cost"` recorded as the run's cost, other keys kept with the
  run); a non-zero exit, no JSON or a `timeout` fails the run with the reason. In
  the UI: `python -m abbot.ui --command "./measure.sh {temp} {conc}"`, the knobs
  starting as its names (ranges for you to set, or whole-number Steps for a
  `{n:d}`, unless `--spec` gives them) and
  the Start step checking its names against the knobs.
- **An Xopt evaluator as the experiment:** `EvaluatorExperiment(evaluator, vocs)`
  (`abbot.interop.xopt`), its VOCS as the spec it works with; a failed run
  raises (rather than becoming a value) so abbot retries it.
- **Steps in the UI:** a knob can be a grid of numbers, "from 20 to 80 in steps
  of 5", next to Range and Choices. A step of 1 between whole ends gives whole
  numbers (`IntegerDomain`); any other step gives the list of values (an ordered
  `EnumeratedDomain`). It shows how many settings that makes, takes a travel limit,
  and fits inside an experiment's declared range. The Knobs step now stops a
  setup whose listed settings are too many to search beside a range.
- **Aim an output at a value:** `OutputSpec(name="size", target=40)` (direction
  `TARGET`) finds the setting nearest the value. The model fits the output as
  measured and the policies score its distance from the target, so it works with
  limits, output constraints and a second objective (`ParetoBO`). The best so far,
  the recommendation, the progress plot and the stopping rules follow it
  (`target_reached` reads its value as a tolerance). In the UI, "Near" beside
  Higher and Lower, with the value to aim for; the CSV header and the agent's spec
  guide carry it too.
- **PS-BAX:** `BAXPolicy(goal, strategy="PSBAX")`, posterior-sampling BAX (Cheng et
  al., 2024). The goal runs on one joint posterior draw over the grid, and the
  pick is the most uncertain design in that draw's target set (uncertainty
  sampling when it's empty or all measured). Like InfoBAX, it spreads runs to pin
  down the whole target set (its edges, and parts not found yet) rather than
  measuring inside the current best guess of it as SwitchBAX and MeanBAX do, at
  a small multiple of MeanBAX's cost per proposal. Proposes at the target, like
  MeanBAX. The docs and the UI now present the strategies by that choice.
- **GPU support:** `Campaign(spec, device="cuda")` (and `run_campaign(...,
  device=)`) fits the model and runs the policy on an NVIDIA GPU, in float64. The
  default stays the CPU. Only the model and the policy see the device: the history,
  the store, candidates, `state()`, experiments and the UI stay on the CPU, so a
  store written on a GPU resumes on any machine. Without CUDA, `device="cuda"`
  raises at construction and says how to install a CUDA build of torch. The
  built-in constructors, policies, optimizers, the bad-region classifier,
  `recommend`, `abbot.analysis`, `abbot.plots`, the UI's figures and `check()` all
  follow the model's device (`abbot.model.model_device`). A custom
  `ModelConstructor` gets its data on the device. The device isn't saved with a
  campaign. The UI and the MCP server take one for the whole process:
  `python -m abbot.ui --device cuda`, `abbot-mcp --device cuda`, or
  `ABBOT_DEVICE=cuda` (`serve(device=)`, `Workspace(root, device=)`). Tests marked
  `cuda` run only where a GPU is available.
  **Your own functions on a GPU campaign** get designs as GPU tensors: a
  `prior_mean` function, a cost model, `BAXPolicy(noise_at=)`, a `FunctionGoal`'s
  function or your own `Goal`. One that calls `.numpy()` on its input fails there;
  write it in torch, or call `.cpu()` first and return a tensor on the input's
  device.
- **Small fixes:**
  - `BAXPolicy(requery=True)` lets a measured design be proposed again (only
    designs in flight are left out), for repeating a noisy measurement.
  - `Campaign(hook_errors=)`: what an `on_step` hook that raises does. `"log"`, the
    default and the behavior so far, logs it and carries on. `"raise"`, opt-in,
    raises it after the result is recorded and saved.
  - `Campaign(on_model_error=)`: what a failed fit or a numerical failure while
    proposing does. `"probe"`, the default and the behavior so far, takes one
    random design and tries again next step. `"raise"`, opt-in, raises instead.
    New in the default: a failed fit now says so in `last_warning`, where before its
    random probe looked like a cold start's.
  - `add_hook` and the `Campaign` docstring say that a registered hook makes each
    result cost a model fit and a recommendation.
- **A run that asks for a knob the design doesn't have** records the reason with
  the error ("asked for 'temp', which isn't a variable here; the design has x1").
- A spec's domain entries (and source domains) may leave out `kind` where it's
  plain: `values` is enumerated, `dim` structured, `lower`/`upper` continuous
  (whole numbers still say `"kind": "integer"`). An entry that can't be told says
  what to add, instead of pydantic's "Unable to extract tag".
- **Fix:** InfoBAX no longer keeps every posterior sample's fantasy model until
  the proposal ends (a GPyTorch model is in a reference cycle, so only Python's
  cyclic collector frees one). Its memory no longer grows with
  `n_posterior_samples`: 100 samples at 1000 runs peaked at 766 MB, now 95 MB, and
  at 3000 runs no longer need over 7 GB.
- **Fix:** an output bounded only from above at or below zero
  (`OutputSpec(max_val=0)`) no longer fails when the model's chance of satisfying
  the constraints is computed.
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
