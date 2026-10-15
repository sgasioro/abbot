# abbot

**A goal-driven (a to b) Bayesian guidance (BO) framework (bot) targeting lab experiments.**


`abbot` is an active-learning-for-labs layer on top of [BoTorch](https://botorch.org).
It is **not** a competing optimization framework: BoTorch already provides the
surrogate-agnostic `Model`/`Posterior` layer, the BO/uncertainty-sampling loop
primitives, async/batch, and serialization. `abbot` adds the two things that are
genuinely ours on top of that substrate:

1. the **composite-SBI objective + cached-posterior (`aux`) layer**, and
2. the slow / human-in-the-loop / fault-supervised **lab-experiment orchestration**.

It unifies the method that is currently reimplemented across several SLAC repos
(sbido, BAXstics, nn-bax) — surrogate + acquisition/BAX + propose→evaluate→retrain
— behind one serializable `ProblemSpec` and an ask/tell loop.

## Quickstart

```python
from abbot import (
    ProblemSpec, OutputSpec, ContinuousDomain,
    Campaign, default_initial_designs,
)

# 1. Define the problem: design knobs + what you measure.
spec = ProblemSpec(
    domain=[ContinuousDomain(name="temp", lower=20.0, upper=80.0)],
    outputs=[OutputSpec(name="yield", direction="MAXIMIZE")],
)

# 2. Open a session (policy + surrogate are picked from the spec) and seed it
#    with a few space-filling starting points — or your own (n, d) tensor.
campaign = Campaign(spec)
campaign.seed(default_initial_designs(spec))

# 3. Ask → run in the lab → tell. Repeat.
for _ in range(8):
    x = campaign.suggest(1)[0]                 # e.g. {"temp": 54.2}
    measured_yield = run_in_lab(x["temp"])     # your measurement here
    campaign.ingest({"yield": measured_yield})

# 4. Hand over the recommended design.
best = campaign.result().best                  # a Record; best.candidate is the setting
```

You report **raw physical values** (`direction` handles maximize vs. minimize for
you — never pre-negate). `Campaign(spec)` picks `SimpleBO` for one objective and
`ParetoBO` for several, with a GP surrogate; pass `Campaign(spec, policy,
model_constructor)` to choose your own. A few variations:

- **Programmatic experiment** — if measurements come from code (an instrument
  driver, a simulator), hand over the whole loop: `run_campaign(spec, experiment,
  n_shots=8)`, where `experiment` is any object with `run(design) -> outputs`.
  `design` is the same `{"temp": 54.2}` dict `suggest` gives you; return the
  outputs as a `{"yield": ...}` dict (or a list in `spec.outputs` order).
- **Multi-objective** — give the spec two+ directional outputs; the same calls
  pick `ParetoBO`, and `result().recommended` is the Pareto set.
- **Find a region, not an optimum (BAX)** — mark outputs `direction="OBSERVE"`, pick
  a `Goal` (e.g. `BandIntersection`, `LevelSet`), and use `BAXPolicy(goal)`.
- **No code** — `python -m abbot.ui` launches the lab web frontend over the same
  `Campaign`.

See [`docs/API.md`](docs/API.md) for the stability contract (what's safe to build
on) and [`examples/`](examples/) for a runnable notebook per capability.

## Status: M1 core + M3 lab frontend

```
src/abbot/
  problem.py      # ProblemSpec (type-polymorphic domains + OutputSpec: objectives
                  #   MAX/MIN vs OBSERVE-only) + bounds/constraints
  model.py        # ModelConstructor seam: (X,Y,Yvar) -> botorch.Model  [the SBI seam]
  models/
    ensemble.py   # ToyEnsembleModel — the non-GP smoke test for the ModelConstructor seam
  acq_opt.py      # CandidateOptimizer seam: Continuous | DiscreteArgmax
  policy.py       # Policy (ask/tell) + SimpleBO, ParetoBO, UncertaintySampling, BAXPolicy; uq_requirement
  session.py      # Campaign — the step-based ask/tell session (suggest/ingest/step/state/
                  #   recommend), resume, on_step hook; shared engine helpers; CampaignResult
  loop.py         # run_campaign(...) -> CampaignResult — thin blocking driver over a Campaign
  analysis.py     # live-analysis built-ins (convergence, hypervolume, recommended_set)
  history.py      # Record{candidate, observation, outputs, aux, fault} + History + DataStore
  experiment.py   # Experiment / PosteriorSampler / Supervisor protocols, Verdict, ManualExperiment
  config.py       # name -> class registries
  goal.py         # Goal (BAX): PercentileSet, BandIntersection, Wishlist, LevelSet, TopK,
                  #   FunctionGoal (arbitrary fn(f, X) -> indices — nn-bax-style flexibility)
  reduce.py       # ObjectiveReducer + BoxMembershipReducer (composite SBI, M2)
  sbi.py          # sbi_evaluate: PosteriorSampler + reducer -> (F, theta) evaluate seam; rescore
  ui/             # NiceGUI web client [extra: ui] — thin over Campaign
    controller.py # RunController: auto/manual run-state machine (framework-free, testable)
    builder.py    # form rows -> ProblemSpec; optimizer selection (surfaces mixed-space limit)
    app.py        # single-dashboard: Define / Run / Progress; `python -m abbot.ui`
  interop/
    sbido.py      # spec_from_sbido + protocol-conformance (lazy; co-dev drift sentinel) [M2]
    xopt.py       # gest VOCS <-> spec + CorePolicyGenerator (runs under Xopt) [F3]
```

### Composite-SBI (M2)
Each observation → posterior samples over θ → a scalar objective F via an
`ObjectiveReducer` (`BoxMembershipReducer` ports sbido's `selection_score`); the
GP sees `(xi, F)` and the θ samples are cached on `Record.aux` so an annealed
reducer can `rescore` without re-sampling. `sbi_evaluate(sampler, reducer, spec)`
builds the loop's `evaluate` callback. sbido co-develops via structural protocols
+ a lazy `interop/sbido.py` adapter — validated against sbido's analytic
`AnalyticalGaussianPosterior` (no `sbi` install); the mat_finder benchmark
reproduction is a follow-on.

### Session API & frontend
The optimization engine is exposed as a headless, step-based `Campaign`
(`suggest` / `ingest` / `step` / `state` / `recommend`) — the same ask/tell surface
a web UI, a notebook, or an agent drives. `run_campaign` is a thin blocking driver
over it. Only a spec and an experiment are required — `run_campaign(spec, experiment)`
or `Campaign(spec)`; the policy, model, and space-filling seed designs default from
the spec (`default_policy` / `default_constructor` / `default_initial_designs`), and
each is an overridable keyword. The **NiceGUI** web client (`abbot[ui]`, `python -m abbot.ui`) is a thin
layer over `Campaign`, laid out as a **single dashboard**: left column = a
**type-polymorphic variable editor** (Continuous | Enumerated — the Badger
differentiator) + run controls; right column = live visualization. Two run modes
share one `RunController`: **manual** (suggest → enter measurement → ingest) and
**mock auto-run** (a `SyntheticExperiment` drives `Campaign.step` on a timer with
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

A `ProblemSpec` declares **outputs** (`OutputSpec(name, direction)`): the surrogate
models every output (one column per output, in `spec.outputs` order — the
`evaluate`/history contract). `direction` is `MAXIMIZE`/`MINIMIZE` for an *optimized
objective*, or `OBSERVE` for a **modeled-but-not-optimized** output (a feature a BAX
`Goal` consumes, or a future constraint). `spec.objectives` is the non-OBSERVE
subset; BO/MO optimize it (selecting its columns from the possibly-wider model),
BAX reads all outputs. (`objectives=[...]` is still accepted as a constructor
alias; `ObjectiveSpec` aliases `OutputSpec`.)

The engine is **natively multi-objective** and **maximize-canonical**: the loop
applies per-output signs internally (`output_signs`) — `evaluate` returns raw
physical measurements, no caller-side negation. `run_campaign` returns a
`CampaignResult` whose `.recommended` is a single index (scalar) or the Pareto set
(multi-objective); `.best` is the single-objective convenience.

Three design properties are preserved from day one so agent integration is later
a ~day of wrapper work (not an architecture change): **headless-first**,
**typed JSON-serializable `ProblemSpec` + run state** (no live-object references),
and **ask/tell as the public contract**.

## Local setup

```bash
# Option A — conda (reproducible; CPU by default):
conda env create -f environment.yml
conda activate abbot
pip install -e '.[ui,dev]'    # editable install + UI (nicegui, plotly) + dev (pytest, ruff)

# Option B — venv + pip (3.11 or 3.12; pyproject requires <3.13):
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e '.[ui,dev]'    # pulls torch/botorch/pydantic/gest-api + nicegui/plotly
```

`gest-api` is on PyPI, so no git/`--no-deps` tricks are needed. Torch installs the
CPU wheel by default (all the UI and toy tests need); for GPU, follow the comment
in `environment.yml`.

## Examples

Four runnable notebooks in [`examples/`](examples/) (install `pip install -e
'.[examples]'`), one per capability: lab BO, multi-objective (`ParetoBO`),
composite-SBI, and goal-driven BAX. See [`examples/README.md`](examples/README.md).

## Run the frontend

```bash
python -m abbot.ui                 # http://127.0.0.1:8080, opens a browser tab
python -m abbot.ui --port 9000     # pick a port (auto-picks a free one if taken)
python -m abbot.ui --no-show       # headless (e.g. on a cluster) — tunnel the port:
                                   #   ssh -L 8080:localhost:8080 <cluster>
```
Flags also read env vars: `ABBOT_UI_HOST`, `ABBOT_UI_PORT`, `ABBOT_UI_SHOW`.

**Mock mode:** on the Define tab, toggle *Mock mode* before creating the campaign
to attach a spec-adaptive `SyntheticExperiment` (smooth, seeded landscape over
whatever variables/objectives you defined). This enables **auto-run**
(run / pause / step / stop) on the Run tab, so a whole campaign drives itself
with no typed measurements — the quickest way to see the loop and plots working.

## Develop

```bash
pytest -m "not slow"   # fast unit tier (~15s, no GP fits) — run on every change
pytest                 # full suite incl. GP-fitting integration tests (minutes;
                       # run before commits / on a compute node, not a login node)
PYTHONPATH=~/sbido/src pytest tests/test_interop_sbido.py   # real sbido integration
ruff check src/ tests/
```

Tests are tiered with a `slow` marker: the fast layer uses canned tensors (goal
logic, reducer scoring, io round-trips, spec/VOCS maps, UI builders); the `slow`
layer fits GPs / runs policies / drives end-to-end loops (botorch `optimize_acqf`,
ParetoBO hypervolume, InfoBAX fantasy conditioning, the Xopt run).

## Roadmap

- **M1** — shared core skeleton on botorch *(done)*.
- **M3** — non-expert lab BO tool: NiceGUI web frontend on the shared `Campaign`
  session API, with a type-polymorphic variable editor *(done; FastAPI+React
  split and `optimize_acqf_mixed` are follow-ons)*.
- **M2** — composite-SBI app: `BoxMembershipReducer` + `sbi_evaluate` + aux
  caching + the lazy `interop/sbido.py` adapter, validated on sbido's analytic
  posterior *(done)*. Follow-on: install `sbi` + reproduce sbido's mat_finder
  benchmark (regret-parity) on the abbot core.
- **F1** — goal-BAX layer so sbido's BAX designers run on abbot (retires the
  multibax-sklearn dep): `Goal` (`PercentileSet`/`BandIntersection`/`Wishlist`) +
  `BAXPolicy` (US/MeanBAX/SwitchBAX/InfoBAX over a discrete grid, botorch-native
  fantasy conditioning). Validated behaviorally on sbido's analytic benchmark
  *(done)*. Follow-on: continuous BAX (pathwise / seeded-ascent); the sbido-side
  cutover PR.
- **F3** — Xopt / gest-api interop: `spec_to_vocs`/`vocs_to_spec` field map
  (`xopt.VOCS is gest_api.VOCS`, so one adapter) + `CorePolicyGenerator`, a gest
  `Generator` wrapping an abbot `Campaign` so a problem runs under
  `Xopt(generator=...)` on Xopt's parallel evaluator — the "never the spine" proof
  (`abbot[xopt]`) *(done)*. This is also the path for expensive-parallel/DFT
  campaigns: abbot defines the problem, Xopt orchestrates evaluation.
- **Now** (pilot campaigns: one simulation, one human-in-the-loop; and a
  composite-SBI + BAX paper on sbido):
  - Batch / async: candidate ids and a persisted outstanding queue, and
    pending-aware policies (`pending=`; kriging believer for BAX) *(done)*;
    optional `Experiment.run_batch` and an outstanding-candidates list in the UI.
  - Composite SBI + BAX on sbido: the sbido cutover PR, then whatever the paper
    benchmark needs (mat_finder regret-parity, continuous BAX).
  - Layering: abbot is the general-purpose engine (what multibax-sklearn was to
    sbido, and more); sbido becomes a thin SBI layer that imports abbot. Generic
    plumbing moves down into abbot — sbido's loop/resume (`measure_and_optimize`
    onto `Campaign`), `initial_designs`, and the supervision monitors (stall /
    drift / consecutive-fault). SBI-specific pieces stay in sbido: posteriors,
    theta-distribution scores, mat_finder, benchmarks, configs. The
    `interop/sbido.py` adapter then moves to sbido's side.
  - Agent setup: an MCP server over `Campaign`, then a setup copilot that writes
    the spec / goal. The UI stays the run console and the human-in-the-loop
    entry point.
- **Later** — mixed continuous + categorical spaces (`optimize_acqf_mixed`), output
  constraints, BAXstics goals on abbot, nn-bax productionization + upstreaming to
  Xopt, Xopt/Badger and Bluesky interop.
