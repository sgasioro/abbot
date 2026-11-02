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
from abbot import ProblemSpec, OutputSpec, ContinuousDomain, Campaign

# 1. Define the problem: design knobs + what you measure.
spec = ProblemSpec(
    domain=[ContinuousDomain(name="temp", lower=20.0, upper=80.0)],
    outputs=[OutputSpec(name="yield", direction="MAXIMIZE")],
)

# 2. Open a session. The policy, surrogate and a few space-filling starting
#    points are picked from the spec (pass initial_designs= for your own).
campaign = Campaign(spec)

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
  a `Goal` (e.g. `BandIntersection({"yield": (0.4, 0.6)})`,
  `LevelSet(0.9, output="purity")`), and use `BAXPolicy(goal)`.
- **Hold a knob fixed** — `campaign.suggest(1, fixed={"temp": 30})` proposes with
  `temp` at 30 and chooses the rest.
- **Categories** — `EnumeratedDomain(name="solvent", values=["water", "ethanol"])`;
  designs carry the labels, and the model learns how alike they are.
- **Know when to stop** — `Campaign(spec, supervisor=Supervisor(stops=[
  target_reached(0.95), no_improvement(15)]))` (rules from `abbot.supervision`),
  and `print(campaign.check())` to see whether the model makes sense.
- **No code** — `python -m abbot.ui` launches the lab web frontend over the same
  `Campaign`.

See [`docs/HOWTO.md`](docs/HOWTO.md) for "how do I…" answers, [`docs/API.md`](docs/API.md)
for the stability contract (what's safe to build on) and [`examples/`](examples/)
for a runnable notebook per capability.

See [`docs/ECOSYSTEM.md`](docs/ECOSYSTEM.md) for where abbot sits next to BoTorch,
Ax, Xopt and Bluesky/blop.

## Status: M1 core + M3 lab frontend

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
  history.py      # Record{candidate, observation, outputs, aux, fault} + History + DataStore
  experiment.py   # Experiment protocols, Verdict, ManualExperiment
  config.py       # name -> class registries
  goal.py         # Goal (BAX): PercentileSet, BandIntersection, Wishlist, LevelSet, TopK,
                  #   FunctionGoal (arbitrary fn(f, X) -> indices — nn-bax-style flexibility)
  composite.py    # composite objectives: composite_evaluate, Rescore (targets hook),
                  #   the ObjectiveReducer protocol
  supervision.py  # Supervisor: fault checks (per-shot, stall, drift) + stop rules
  diagnostics.py  # campaign.check(): leave-one-out fit, noise, lengthscales
  ui/             # NiceGUI web client [extra: ui] — thin over Campaign
    controller.py # RunController: auto/manual run-state machine (framework-free, testable)
    builder.py    # form rows -> ProblemSpec; optimizer selection
    app.py        # single-dashboard: Define / Run / Progress; `python -m abbot.ui`
  interop/
    xopt.py       # gest VOCS <-> spec + CorePolicyGenerator (runs under Xopt) [F3]
    sbi.py        # SBI as a composite objective: PosteriorSampler, sbi_evaluate,
                  #   TrainedPosterior [sbi]
```

### Composite objectives and SBI (M2)
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
trains on. The reducer is the application's own score. Simulation-based inference is the case where the analysis
draws posterior samples over θ: `abbot.interop.sbi` has `sbi_evaluate(sampler,
reducer, spec)` and `TrainedPosterior`, which wraps a loaded `sbi` posterior
(`abbot[sbi]`) with the application's observation preprocessing. sbido is a thin application on top: its
loop is a `Campaign`, and its targets, scores and theta-distribution goals plug
into these seams.

### Session API & frontend
The optimization engine is exposed as a headless, step-based `Campaign`
(`suggest` / `ingest` / `step` / `state` / `recommend`) — the same ask/tell surface
a web UI, a notebook, or an agent drives. `run_campaign` is a thin blocking driver
over it (`drive(campaign, ...)` for a campaign you built yourself); its budget is a
total, so rerunning the same call after a halt finishes the remainder. Only a spec and an experiment are required — `run_campaign(spec, experiment)`
or `Campaign(spec)`; the policy, model, and space-filling seed designs default from
the spec (`default_policy` / `default_constructor` / `default_initial_designs`), and
each is an overridable keyword. The **NiceGUI** web client (`abbot[ui]`, `python -m abbot.ui`) is a thin
layer over `Campaign`, laid out as a **single dashboard**: left column = a
**type-polymorphic variable editor** (Continuous | Enumerated — the Badger
differentiator) + run controls; right column = live visualization. Two run modes
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

A `ProblemSpec` declares **outputs** (`OutputSpec(name, direction)`): the surrogate
models every output (one column per output, in `spec.outputs` order — the
`evaluate`/history contract). `direction` is `MAXIMIZE`/`MINIMIZE` for an *optimized
objective*, or `OBSERVE` for a **modeled-but-not-optimized** output (a feature a BAX
`Goal` consumes, or an output that only has to stay within bounds). Any output can
carry bounds, `OutputSpec(name, direction, min_val=, max_val=)`, which make it a
constraint; `output_constraints=` and `domain_constraints=` take
`LinearConstraint`s over output and design-variable names (see
[`docs/API.md`](docs/API.md#constraints)). `spec.objectives` is the non-OBSERVE
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
to attach a spec-adaptive `MockAnalyticExperiment` (smooth, seeded landscape over
whatever variables/objectives you defined). This enables **auto-run**
(run / pause / step / stop) on the Run tab, so a whole campaign drives itself
with no typed measurements — the quickest way to see the loop and plots working.

## Develop

```bash
pytest -m "not slow"   # fast unit tier (~15s, no GP fits) — run on every change
pytest                 # full suite incl. GP-fitting integration tests (minutes;
                       # run before commits / on a compute node, not a login node)
ruff check src/ tests/
```

Tests are tiered with a `slow` marker: the fast layer uses canned tensors (goal
logic, reducer scoring, io round-trips, spec/VOCS maps, UI builders); the `slow`
layer fits GPs / runs policies / drives end-to-end loops (botorch `optimize_acqf`,
ParetoBO hypervolume, InfoBAX fantasy conditioning, the Xopt run).

## Roadmap

- **M1** — shared core skeleton on botorch *(done)*.
- **M3** — non-expert lab BO tool: NiceGUI web frontend on the shared `Campaign`
  session API, with a type-polymorphic variable editor *(done; a FastAPI+React
  split is a follow-on)*.
- **M2** — composite-SBI app: `BoxMembershipReducer` + `sbi_evaluate` + aux
  caching + a lazy `interop/sbido.py` adapter (since removed), validated on sbido's analytic
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
- **Batch / async** — candidate ids and a persisted outstanding queue,
  pending-aware policies (BoTorch q-batches for BO; greedy kriging believer for
  BAX / US), synchronous batches in the blocking loop (`batch_size=`,
  `BatchExperiment`), hooks once per batch, `MockAnalyticExperiment` /
  `MockDatasetExperiment` *(done)*.
- **Layering** — abbot is the general engine; sbido and BAXstics are thin
  applications on it. Problem-specific targets stay in the applications and plug
  into abbot's seams (`ObjectiveReducer`, `Goal`, `targets=` / `Rescore`); the
  loop (`Campaign`, supervisors, resume, fingerprints) replaces sbido's
  `measure_and_optimize` *(done; sbido cutover PR in review, BAXstics port on a
  branch)*.
- **Standing: UX audits.** Ease of use is a deliberate priority. For each core
  workflow (single- and multi-objective BO, BAX goals, batches and async
  ask/tell, resume, cost and stopping, sources / multi-fidelity), periodically
  set the same problem up in abbot, plain BoTorch, Ax and Xopt (and blop where
  it applies), and compare lines of code, concepts a newcomer must learn, what
  must be gotten right by hand, error messages on common mistakes, and how
  quickly the docs answer "how do I...". The point is to find what to improve in
  abbot; these are internal checks, not published claims.
- **Now**, in order:
  1. **BoTorch 0.18** *(done)*: abbot now requires it, the same line as blop and
     Ax.
  2. **Multi-fidelity, cost-aware BO** (pilot: calibrating an expensive simulation;
     fidelity is system size and other knobs; runs 30 s to 1 h).
     - Phase A, cost *(done)*: per-record cost (wall-clock or caller-given), cost
       models (`FixedCost`, `CallableCost`, `PowerLawCost`), cost-aware `SimpleBO`
       / `ParetoBO`, and `Verdict.stop` with `cost_cap` / `acquisition_monitor`.
     - Phase B, source variables *(done)*: fidelities and tasks / sources as one
       spec concept (`Source`: ordered or not, with a target), `IntegerDomain`,
       mixed-space proposal (`optimize_acqf_mixed`), policies pinned at the
       target, recommendation scored at the target.
     - Phase C1 *(done)*: `MultiFidelityGPConstructor` (picked by default with
       sources) and `Source(accurate=)`; `SimpleBO` chooses sources (cost-weighted
       multi-fidelity max-value entropy search, no separate policy) with a learned
       default cost, `SourceCost`.
     - Phase C2 *(done)*: multi-fidelity InfoBAX (source levels on the candidate
       grid, information per unit cost); fixes for two BoTorch multi-fidelity bugs.
     - Phase D *(done)*: `benchmarks/multi_fidelity.py` (cost to a good answer at
       the target, against target-only BO and BoTorch's MF-KG; it kept the
       linear-truncated kernel as the default), the `05_multi_fidelity` example,
       `MockAnalyticExperiment` with sources, and `Experiment.cost(design)`.
     - Later: a BAX-shaped goal (every setting that matches the data); a noise level
       that varies with the source, without per-run variances; UI support.
  3. **Peer audit** *(done; internal)*: functionality and UX against BoTorch,
     Ax, Xopt and blop. What to build and what to borrow is in
     [`docs/ECOSYSTEM.md`](docs/ECOSYSTEM.md#build-or-borrow). From it, in order:
     - **Output constraints** *(done)*: `OutputSpec(min_val=, max_val=)`,
       `output_constraints` and named `domain_constraints` (`LinearConstraint`);
       feasibility-weighted BO, feasibility search until a run satisfies them, BAX
       target sets and multi-fidelity entropy search that respect them, and
       recommending only designs likely to satisfy them. Later: UI support and a
       feasibility stopping rule.
     - **Spec ergonomics** *(done)*: names instead of positions in boxes and
       goals, `log=True` on design variables, `suggest(fixed=...)`, and labeled
       categories (`values=["water", "ethanol"]`) with a categorical kernel.
     - **Xopt generators in abbot's loop** *(done)*: `GestGeneratorPolicy`
       (TuRBO, Nelder-Mead, RCDS, genetic algorithms, time-dependent BO), with
       policies able to skip abbot's model fit (`needs_model = False`).
     - **Diagnostics and stopping** *(done)*: `campaign.check()` (leave-one-out
       fit, suspicious runs, noise, lengthscales, runs vs. inputs),
       `Supervisor(faults=, stops=)` with `target_reached`, `no_improvement` and
       `all_of` floors, and a "how do I…" index
       ([`docs/HOWTO.md`](docs/HOWTO.md)). Later: the check in the UI.
     - **Travel limits** per variable (with or before the blop backend), a
       prior-mean option, a high-dimensional (SAAS) default.
  4. **Correlated outputs:** `MultiTaskGPConstructor` (plan written). Then
     related tasks / data sources, sharing the spec design from 2.
  5. **Small fixes:** a dimension-scaled prior for `kernel="matern52"` (it fails
     to fit sparse, partly zero data; check sbido's kernel option), flagged-shot
     handling, an outstanding-candidates list in the UI, and reporting
     multibax's shared random draws across outputs upstream.
  6. **blop backend:** a `Campaign`-backed blop `Optimizer` for Bluesky
     beamlines (`abbot[blop]`; see [`docs/ECOSYSTEM.md`](docs/ECOSYSTEM.md)).
  7. **Agent setup:** an MCP server over `Campaign`, then a setup copilot that
     writes the spec / goal. The UI stays the run console and the
     human-in-the-loop entry point.
- **Deferred: integration.** Building out abbot comes first; then back to the
  applications. sbido: Sam's review of the cutover PR, port master's BAX fallback
  option (e6a5d77) into abbot or sbido, bump the abbot pin, then what the paper
  benchmark needs (mat_finder regret-parity, continuous BAX). BAXstics: push the
  `abbot-port` branch, bump its abbot pin, and merge to its main once abbot is
  public.
- **Later** — output constraints, nn-bax productionization (upstream to Xopt where
  it fits).
