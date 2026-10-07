# abbot

**From A to B — a Bayesian guide for lab experiments.**

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

## Status: M1 core + M3 lab frontend

```
src/abbot/
  problem.py      # ProblemSpec (pydantic, type-polymorphic domains) + bounds/constraints
  model.py        # ModelConstructor seam: (X,Y,Yvar) -> botorch.Model  [the SBI seam]
  models/
    ensemble.py   # ToyEnsembleModel — the non-GP smoke test for the ModelConstructor seam
  acq_opt.py      # CandidateOptimizer seam: Continuous | DiscreteArgmax
  policy.py       # Policy (ask/tell) + SimpleBO, ParetoBO, UncertaintySampling; uq_requirement
  session.py      # Campaign — the step-based ask/tell session (suggest/ingest/step/state/
                  #   recommend), resume, on_step hook; shared engine helpers; CampaignResult
  loop.py         # run_campaign(...) -> CampaignResult — thin blocking driver over a Campaign
  analysis.py     # live-analysis built-ins (convergence, hypervolume, recommended_set)
  history.py      # Record{candidate, observation, outputs, aux, fault} + History + DataStore
  experiment.py   # Experiment / PosteriorSampler / Supervisor protocols, Verdict, ManualExperiment
  config.py       # name -> class registries
  goal.py         # Goal protocol (BAX goal-as-algorithm)   [stub — F1]
  reduce.py       # ObjectiveReducer protocol (composite SBI) [stub — M2]
  ui/             # NiceGUI web client [extra: ui] — thin over Campaign
    controller.py # RunController: auto/manual run-state machine (framework-free, testable)
    builder.py    # form rows -> ProblemSpec; optimizer selection (surfaces mixed-space limit)
    app.py        # Define -> Run -> Progress screens; `python -m abbot.ui`
  interop/
    xopt.py       # gest-api VOCS <-> spec + CorePolicyGenerator [stub — F3]
```

### Session API & frontend
The optimization engine is exposed as a headless, step-based `Campaign`
(`suggest` / `ingest` / `step` / `state` / `recommend`) — the same ask/tell surface
a web UI, a notebook, or an agent drives. `run_campaign` is a thin blocking driver
over it. The **NiceGUI** web client (`abbot[ui]`, `python -m abbot.ui`) is a thin
layer over `Campaign` with three screens: a **type-polymorphic variable editor**
(Continuous | Enumerated — the Badger differentiator), a manual run loop
(suggest → enter measurement → ingest), and live progress plots fed by the
`on_step` hook. Auto-run (controller drives `Campaign.step` with run/pause/step/stop)
is available in the library when a programmatic `Experiment` is attached; manual vs.
auto is a UI-driver concern, kept out of the core.

The engine is **natively multi-objective** and **maximize-canonical**: an
`evaluate` callback returns raw physical measurements (one column per objective,
in `spec.objectives` order), `ObjectiveSpec.direction` (`MINIMIZE`/`MAXIMIZE`)
declares intent, and the loop applies the signs internally — no caller-side
negation. `run_campaign` returns a `CampaignResult` whose `.recommended` is a
single index (scalar) or the Pareto set (multi-objective); `.best` is the
single-objective convenience.

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
pytest        # engine (optimum recovery, MO, direction), session API, analysis,
              # controller, UI builder/import  (~3 min; many GP fits)
ruff check src/ tests/
```

## Roadmap

- **M1** — shared core skeleton on botorch *(done)*.
- **M3** — non-expert lab BO tool: NiceGUI web frontend on the shared `Campaign`
  session API, with a type-polymorphic variable editor *(done; FastAPI+React
  split and `optimize_acqf_mixed` are follow-ons)*.
- **M2** — SBI design-optimization app (the lead build): composite objective +
  `SBIPosteriorSampler` + real experiments. Acceptance = reproduce sbido's
  benchmark entirely on the `abbot` core.
- **F-tier** — BAXstics goal-BAX port, full nn-bax productionization + upstreaming
  to Xopt, Xopt/Badger interop via gest-api, and agent integration (setup-copilot,
  then an MCP server).
