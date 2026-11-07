# How do I…

Short answers, each with a pointer to the details. `spec` is a `ProblemSpec` and
`campaign` a `Campaign(spec)` throughout. Names come from `abbot` unless a module
is shown.

## Setting up a problem

| How do I… | |
| --- | --- |
| describe knobs and measurements | `ProblemSpec(domain=[ContinuousDomain(name="temp", lower=20, upper=80)], outputs=[OutputSpec(name="yield", direction="MAXIMIZE")])`. See the [README quickstart](../README.md#quickstart). |
| use whole numbers, or a fixed list of values | `IntegerDomain(name=, lower=, upper=)`; `EnumeratedDomain(name=, values=[0.1, 0.5, 1.0])` (an ordered grid). |
| use categories (solvents, materials) | `EnumeratedDomain(name="solvent", values=["water", "ethanol"])`. Designs carry the labels. [Categories](API.md#categories-labeled-values) |
| search a knob over decades | `ContinuousDomain(..., log=True)`. [Names, not positions](API.md#names-not-positions) |
| record something without optimizing it | `OutputSpec(name=, direction="OBSERVE")`. |
| require an output to stay in a range | `OutputSpec(..., min_val=, max_val=)`; for a combination of outputs, `output_constraints=[LinearConstraint({...}, min_val=)]`. [Constraints](API.md#constraints) |
| limit how far a knob moves per run | `ContinuousDomain(..., max_step=0.5)`, in the knob's units; start the campaign at the current settings with `initial_designs=[{...}]`. [Travel limits](API.md#travel-limits) |
| forbid some combinations of settings | `domain_constraints=[LinearConstraint({"temp": 1, "conc": 2}, max_val=100)]`. [Constraints](API.md#constraints) |
| optimize several outputs at once | Give two or more outputs a direction; `Campaign(spec)` picks `ParetoBO`, and `result().recommended` is the Pareto set. |
| find a region, not an optimum | Mark outputs `OBSERVE`, pick a goal (`BandIntersection`, `LevelSet`, `TopK`, …) and use `BAXPolicy(goal)`. [BAX details](API.md#bax-details) |
| combine cheap and accurate runs (fidelities, two codes) | `ProblemSpec(sources=[Source(domain=..., target=...)])`. [Column contract → source variables](API.md#column-contract) |
| save or share a spec | `abbot.io.spec_to_json` / `spec_from_json`, or `spec_history_to_csv` for a spec plus its runs in one CSV. |

## Running a campaign

| How do I… | |
| --- | --- |
| run by hand (ask, measure, tell) | `x = campaign.suggest(1)[0]`, measure, `campaign.ingest({"yield": v})`. [README quickstart](../README.md#quickstart) |
| let code run the experiment | `run_campaign(spec, experiment, n_shots=40)`, with `experiment.run(design) -> outputs`; or `drive(campaign, experiment, n_total=...)` for a campaign you built. |
| run several at once | `campaign.suggest(4)`, then `ingest(result, id=c.id)` as each finishes. [Outstanding candidates](API.md#outstanding-candidates) |
| hold a knob at a value for now | `campaign.suggest(1, fixed={"temp": 30})`. |
| start from my own designs, or none | `Campaign(spec, initial_designs=[{...}, ...])`; an empty tensor for a cold start. [Defaults](API.md#defaults) |
| use results I already have | `campaign.record({"temp": 40}, {"yield": 0.7})` for each, before the first `suggest`. |
| report a run that failed | `campaign.fail(c.id, "pump jammed")`; it's retried up to `max_retries`, then the campaign halts. [Faults and bad regions](API.md#faults-and-bad-regions) |
| carry on after a halt | `campaign.resolve("retry")` once fixed; `"exclude"` to skip the design; `"bad_region"` if the design itself is the cause. Or the halt banner's buttons in the UI. [Faults and bad regions](API.md#faults-and-bad-regions) |
| never retry, ask me at once | `Campaign(spec, max_retries=0)`. |
| tell abbot a region gives nothing usable | `per_shot(is_dark, bad_region="dark")` in `faults=`, or a rule returning `Verdict(bad_region=True)`; proposals then steer away. [Faults and bad regions](API.md#faults-and-bad-regions) |
| withdraw a design I won't run | `campaign.cancel(c.id)`. |
| fix a value I recorded wrong | `campaign.correct(c.id, {"yield": 0.65}, reason="typo")`; the old value is kept on the record. [Outstanding candidates](API.md#outstanding-candidates) |
| block unsafe designs before they run | `Campaign(..., pre_run=check)`, with `check(design)` returning `None` or a reason. [Failure behavior](API.md#failure-behavior) |
| save progress and resume after a crash | `Campaign(spec, store=DataStore("runs/today"))`; building it again on the same folder resumes, outstanding designs included. [Outstanding candidates](API.md#outstanding-candidates) |
| catch bad shots and a drifting instrument | `supervisor=standard_supervisor(spec.bounds_tensor(), detector)` from `abbot.supervision`; for your own mix, `Supervisor(faults=[per_shot(detector), ...])`. [Supervisors](API.md#supervisors-faults-and-stops) |
| use the web interface | `python -m abbot.ui`. [README](../README.md#run-the-frontend) |
| let an AI agent run the campaign | `pip install "abbot[mcp]"`; in Claude Code, `abbot-mcp start` sets up and opens a session that plans the campaign with you. Other clients: add `abbot-mcp --dir ~/abbot-campaigns` as an MCP server. [Agent access](API.md#agent-access-mcp) |

## Knowing when to stop

| How do I… | |
| --- | --- |
| stop at a target value | `Campaign(spec, supervisor=Supervisor(stops=[target_reached(0.95)]))`; the rules are in `abbot.supervision`. [Diagnostics and stopping](API.md#diagnostics-and-stopping) |
| stop when progress stalls | `stops=[no_improvement(15, min_delta=...)]`. |
| stop at a budget | `stops=[cost_cap(budget)]`, in the units of the recorded costs (run time by default). [Column contract → cost and stopping](API.md#column-contract) |
| combine stopping rules | List them: `stops=[target_reached(0.95), no_improvement(15), cost_cap(100)]` stops on the first. |
| stop on a rule, but not too early | `stops=[all_of(no_improvement(10), min_runs(30))]`. [Supervisors](API.md#supervisors-faults-and-stops) |
| catch faults and stop, together | `Supervisor(faults=[...], stops=[...])`. |
| tell whether the run stopped | `campaign.stopped` holds the reason (`drive` and `run_campaign` end there). |

## Checking and handing over

| How do I… | |
| --- | --- |
| check that the model makes sense | `print(campaign.check())`: leave-one-out fit, suspicious runs, noise, lengthscales. [Diagnostics and stopping](API.md#diagnostics-and-stopping) |
| get the best design | `campaign.result().best` (a `Record`; its design is `spec.as_dict(best.candidate)`). |
| plot the model or progress | `abbot.plots.model(campaign)` (mean, uncertainty, runs and the goal), `plots.progress(campaign)`, `plots.check(campaign)`; each returns a matplotlib figure. [Plots](API.md#plots) |
| export the runs | `abbot.io.history_to_csv(campaign.history, spec)`. |

## Going further

| How do I… | |
| --- | --- |
| start the model from a physics model | `default_constructor(spec, prior_mean={"yield": fn})`, with `fn({name: tensor})` in raw units, written in torch. [Prior means](API.md#prior-means) |
| let outputs inform each other (yield and selectivity, summaries of one posterior) | `default_constructor(spec, correlated=True)`. Not the same as sources: those are about where a measurement came from. [Correlated outputs](API.md#correlated-outputs) |
| reuse data from an earlier, similar system | A labeled source for the system, with the old runs loaded under its old label; or a prior mean from the old campaign's model. [Prior means](API.md#prior-means) |
| use my own model | Anything with `build(X, Y, Yvar) -> Model` and `provides_uq`; pass it as `Campaign(spec, policy, model_constructor)`. See `examples/custom_models.py`. |
| fit the model to something derived from the raw data | `Campaign(..., targets=fn)`, with `fn(history, spec) -> Y`. |
| use an Xopt generator (TuRBO, Nelder-Mead, …) in abbot | `Campaign(spec, GestGeneratorPolicy(NelderMeadGenerator))` from `abbot.interop.xopt`. [Xopt's generators](API.md#xopts-generators-in-abbots-loop) |
| use abbot inside Xopt | `CorePolicyGenerator(spec)` from `abbot.interop.xopt`. |
| account for runs that cost different amounts | `SimpleBO(cost_model=...)`. [Column contract → cost and stopping](API.md#column-contract) |
