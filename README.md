# abbot

**A goal-driven (a to b) Bayesian guidance (BO) framework (bot) targeting lab experiments.**


`abbot` runs experiment campaigns on top of [BoTorch](https://botorch.org): BoTorch
supplies the models and acquisition functions, and abbot supplies the campaign
around them. It is built for lab work, where:

1. **the goal isn't always an optimum:** besides standard and multi-objective BO,
   goal-driven BAX finds a region, a level set or the best point per band;
2. **a measurement isn't always a number:** composite objectives reduce a spectrum,
   replicates or an SBI posterior to outputs, keeping the raw observation;
3. **runs are slow, supervised and sometimes fail:** an ask/tell loop with retries,
   halts a person resolves, bad regions, resume, and a web UI.

It grew out of several lab projects that each reimplemented the same method —
surrogate + acquisition/BAX + propose→evaluate→retrain — and puts it behind one
serializable `ProblemSpec` and an ask/tell loop.

**Status: alpha.** abbot is in active development. Anything marked *Stable* in
[`docs/API.md`](docs/API.md) is safe to build on; *Provisional* parts may change
between releases (renamed names keep working as aliases for a while). Issues and
pull requests are welcome.

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
- **Limit how far a knob moves** — `ContinuousDomain(..., max_step=0.5)`: every
  proposal is within 0.5 of the last one, starting from the `initial_designs` you
  give (the current settings).
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

### Composite objectives and SBI
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
(`abbot[sbi]`) with the application's observation preprocessing. An SBI design
application is then thin: its loop is a `Campaign`, and its targets, scores and
θ-distribution goals plug into these seams.

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

Three design properties are kept so that agents can drive abbot without
architecture changes: **headless-first**,
**typed JSON-serializable `ProblemSpec` + run state** (no live-object references),
and **ask/tell as the public contract**.

## Local setup

```bash
# Option A — conda:
conda env create -f environment.yml
conda activate abbot

# Option B — venv (Python 3.11 or 3.12; pyproject requires <3.13):
python3.12 -m venv .venv && source .venv/bin/activate

# Then, either way: torch first, from PyTorch's index, then abbot.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[ui,dev]'    # editable install + UI (nicegui, plotly) + dev (pytest, ruff)
```

The CPU build of torch is all the UI and the tests need. For an NVIDIA GPU,
install a CUDA build instead, e.g. `pip install torch --index-url
https://download.pytorch.org/whl/cu128`; the right tag for your driver is listed
at [pytorch.org/get-started](https://pytorch.org/get-started). Installing torch
with pip from PyTorch's index is more reliable than conda's CUDA packages.
Everything else (BoTorch, `gest-api`, ...) comes from PyPI with `pip install -e`.

## Examples

Five runnable notebooks in [`examples/`](examples/) (install `pip install -e
'.[examples]'`), one per capability: lab BO, multi-objective (`ParetoBO`),
composite-SBI, goal-driven BAX, and multi-fidelity. See [`examples/README.md`](examples/README.md).

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
pytest -m "not slow"   # fast unit tier (no GP fits) — run on every change
pytest                 # full suite incl. GP-fitting integration tests (minutes;
                       # run before commits / on a compute node, not a login node)
ruff check src/ tests/
```

Tests are tiered with a `slow` marker: the fast layer uses canned tensors (goal
logic, reducer scoring, io round-trips, spec/VOCS maps, UI builders); the `slow`
layer fits GPs / runs policies / drives end-to-end loops (botorch `optimize_acqf`,
ParetoBO hypervolume, InfoBAX fantasy conditioning, the Xopt run).

## Roadmap

**Built:**
- The ask/tell `Campaign`, blocking drivers, resume, batches and asynchronous
  runs with pending designs.
- Single- and multi-objective BO, uncertainty sampling, and goal-driven BAX
  (InfoBAX, SwitchBAX, MeanBAX) over discrete grids.
- Composite objectives and SBI (`composite_evaluate`, `Rescore`,
  `abbot.interop.sbi`).
- Multi-fidelity and multi-source BO with costs (`Source`,
  `MultiSourceGPConstructor`, cost-aware `SimpleBO` and InfoBAX).
- Output and domain constraints; names, `log=True` inputs and labeled categories
  in the spec; `suggest(fixed=...)`; travel limits (`max_step`).
- Prior means, and correlated outputs (`default_constructor(spec,
  correlated=True)`).
- Supervision: fault checks, retries and halts, bad regions with a usability
  classifier, stopping rules, and `campaign.check()` diagnostics.
- Xopt interop both ways: abbot as an Xopt generator, and Xopt generators (TuRBO,
  Nelder-Mead, ...) in abbot's loop.
- The NiceGUI lab frontend (`python -m abbot.ui`).

**Next:**
1. **The UI catches up with the engine:** constraints, travel limits, sources,
   `check()`, and a model choice in the builder.
2. **Agent setup:** an MCP server over `Campaign`, then a setup assistant that
   writes the spec and goal. The UI stays the run console and the
   human-in-the-loop entry point.
3. **GPU support:** one setting (`Campaign(..., device="cuda")`) that fits models
   and runs policies on the GPU, while the history, saved files, experiments and
   the UI stay on the CPU. The default stays CPU, since lab-scale GPs are usually
   faster there. CUDA only: abbot fits in float64, which Apple's MPS doesn't
   support.

**Later:**
- Continuous BAX (no candidate grid).
- Correlated outputs together with sources or prior means.
- A BAX-shaped multi-fidelity goal; noise that varies with the source.
- Check shots at a known-good design.
- A Bluesky route: abbot as a blop `Optimizer` backend
  (see [`docs/ECOSYSTEM.md`](docs/ECOSYSTEM.md)).
- A run-level JSON format that includes cached SBI payloads.

**Ease of use is a priority.** For each core workflow we periodically set the
same problem up in abbot and in neighboring tools, and compare how much a
newcomer must learn and get right by hand, to find what to improve in abbot.

## License

Apache-2.0; see [`LICENSE`](LICENSE).
