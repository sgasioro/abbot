# Architecture

How abbot is put together: the modules, the seams an application plugs into,
and the conventions the engine keeps. For using abbot, start with the
[README](../README.md) and [`HOWTO.md`](HOWTO.md); for what's stable to build
on, see [`API.md`](API.md).

## Layout

```
src/abbot/
  problem.py      # ProblemSpec (type-polymorphic domains + OutputSpec: objectives
                  #   MAX/MIN vs OBSERVE-only, output bounds) + domain/output constraints
  feasibility.py  # output constraints: observed feasibility, the model's probability
  model.py        # ModelConstructor seam: (X,Y,Yvar) -> botorch.Model  [the SBI seam]
  models/
    ensemble.py   # ToyEnsembleModel — the non-GP smoke test for the ModelConstructor seam
  acq_opt.py      # CandidateOptimizer seam: Continuous | DiscreteArgmax
  policy.py       # Policy (ask/tell) + SimpleBO, ParetoBO, UncertaintySampling, BAXPolicy; uq_requirement
  session.py      # Campaign — the step-based ask/tell session (suggest/ingest/step/state/
                  #   recommend), resume, on_step hook; shared engine helpers; CampaignResult
  loop.py         # run_campaign / drive — thin blocking drivers over a Campaign
  analysis.py     # live-analysis built-ins (convergence, hypervolume, recommended_set)
  history.py      # Record{candidate, observation, outputs, aux, fault, bad_region} + History + DataStore
  experiment.py   # Experiment protocols, Verdict, ManualExperiment
  config.py       # name -> class registries
  goal.py         # Goal (BAX): PercentileSet, BandIntersection, Wishlist, LevelSet, TopK,
                  #   FunctionGoal (arbitrary fn(f, X) -> indices)
  composite.py    # composite objectives: composite_evaluate, Rescore (targets hook),
                  #   the ObjectiveReducer protocol
  supervision.py  # Supervisor: fault checks (per-shot, stall, drift) + stop rules
  usability.py    # where runs give usable results: a classifier over bad-region runs
  diagnostics.py  # campaign.check(): leave-one-out fit, noise, lengthscales
  ui/             # NiceGUI web client [extra: ui] — thin over Campaign
    controller.py # RunController: auto/manual run-state machine (framework-free, testable)
    builder.py    # form rows -> ProblemSpec; optimizer selection
    app.py        # single-dashboard: Define / Run / Progress; `python -m abbot.ui`
  interop/
    xopt.py       # gest VOCS <-> spec + CorePolicyGenerator (runs under Xopt)
    sbi.py        # SBI as a composite objective: PosteriorSampler, sbi_evaluate,
                  #   TrainedPosterior [sbi]
```

## Composite objectives and SBI
A measurement is often richer than the number being optimized — a spectrum,
replicate measurements, posterior samples. `composite_evaluate(analysis, reducer,
spec)` maps each observation to a payload (`analysis`), reduces it to the outputs
the surrogate sees (an `ObjectiveReducer`), and caches the payload on
`Record.aux`. For an objective that changes during the run,
`Campaign(..., targets=Rescore(reducer, schedule))` re-scores every clean shot's
cached payload at the current schedule step (a score that sharpens, a tolerance
that tightens), and the model, the policy and the hand-over all see the re-scored
values — so any policy (`SimpleBO`, BAX, ...) can anneal. `targets=` itself is
general: any `(history, spec) -> outputs` function redefines what the surrogate
trains on. The reducer is the application's own score. Simulation-based inference (SBI) is the case where the analysis
draws posterior samples over θ: `abbot.interop.sbi` has `sbi_evaluate(sampler,
reducer, spec)` and `TrainedPosterior`, which wraps a loaded `sbi` posterior
(`abbot[sbi]`) with the application's observation preprocessing. An SBI design
application is then thin: its loop is a `Campaign`, and its targets, scores and
θ-distribution goals plug into these seams.

## Session API and frontend
The optimization engine is exposed as a headless, step-based `Campaign`
(`suggest` / `ingest` / `step` / `state` / `recommend`) — the same ask/tell surface
a web UI, a notebook, or an agent drives. `run_campaign` is a thin blocking driver
over it (`drive(campaign, ...)` for a campaign you built yourself); its budget is a
total, so rerunning the same call after a halt finishes the remainder. Only a spec and an experiment are required — `run_campaign(spec, experiment)`
or `Campaign(spec)`; the policy, model, and space-filling seed designs default from
the spec (`default_policy` / `default_constructor` / `default_initial_designs`), and
each is an overridable keyword. The **NiceGUI** web client (`abbot[ui]`, `python -m abbot.ui`) is a thin
layer over `Campaign`, laid out as a **single dashboard**: left column = a
**type-polymorphic variable editor** (Continuous | Enumerated) + run controls; right column = live visualization. Two run modes
share one `RunController`: **manual** (suggest → enter measurement → ingest) and
**mock auto-run** (a `MockAnalyticExperiment` drives `Campaign.step` on a timer with
run/pause/step/stop). Manual vs. auto is a UI-driver concern, kept out of the core.

Live visualization (via `abbot/analysis.py`, computed from history + the fitted
model): an **adaptive model view** — 1-D posterior mean ± uncertainty band, 2-D
posterior-mean contour, or a 1-D/2-D slice through the best point for 3-D+ (with
variable selectors) — plus a dimension-agnostic **parity plot** (predicted vs
observed), a **convergence / hypervolume** trace, the recommended design(s), and
the history table. Per-step the plots **update in place** (`update_figure`),
rebuilt only on create/reset/selector-change, so auto-run stays smooth.

**Import / export** (via `abbot/io.py`, pure functions). Two tiers:
- **Self-describing CSV** (`spec_history_to_csv` / `spec_history_from_csv`) — one
  lab-friendly file carrying the whole setup in `#` comment rows (variables,
  objectives, boxes, version) *plus* the measurement table. Opens in Excel/pandas
  (`read_csv(comment='#')`); refuses (not silently drops) parts it can't hold
  (structured domains, constraints). The portable "email a campaign" format.
- **Canonical JSON** (`spec_to_json`) — lossless `ProblemSpec` round-trip for the
  cases CSV can't express. *(A run-level JSON serializer incl. the SBI `aux`
  payload is a planned follow-on.)*

## Outputs and signs

A `ProblemSpec` declares **outputs** (`OutputSpec(name, direction)`): the surrogate
models every output (one column per output, in `spec.outputs` order — the
`evaluate`/history contract). `direction` is `MAXIMIZE`/`MINIMIZE` for an *optimized
objective*, or `OBSERVE` for a **modeled-but-not-optimized** output (a feature a BAX
`Goal` consumes, or an output that only has to stay within bounds). Any output can
carry bounds, `OutputSpec(name, direction, min_val=, max_val=)`, which make it a
constraint; `output_constraints=` and `domain_constraints=` take
`LinearConstraint`s over output and design-variable names (see
[`API.md`](API.md#constraints)). `spec.objectives` is the non-OBSERVE
subset; BO/MO optimize it (selecting its columns from the possibly-wider model),
BAX reads all outputs. (`objectives=[...]` is still accepted as a constructor
alias; `ObjectiveSpec` aliases `OutputSpec`.)

The engine is **natively multi-objective** and **maximize-canonical**: the loop
applies per-output signs internally (`output_signs`) — `evaluate` returns raw
physical measurements, no caller-side negation. `run_campaign` returns a
`CampaignResult` whose `.recommended` is a single index (scalar) or the Pareto set
(multi-objective); `.best` is the single-objective convenience.

## Ready for agents

Three design properties are kept so that agents can drive abbot without
architecture changes: **headless-first**,
**typed JSON-serializable `ProblemSpec` + run state** (no live-object references),
and **ask/tell as the public contract**.
