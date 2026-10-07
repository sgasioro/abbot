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

## Status: M1 (shared core skeleton)

```
src/abbot/
  problem.py      # ProblemSpec (pydantic, type-polymorphic domains) + bounds/constraints
  model.py        # ModelConstructor seam: (X,Y,Yvar) -> botorch.Model  [the SBI seam]
  models/
    ensemble.py   # ToyEnsembleModel — the non-GP smoke test for the ModelConstructor seam
  acq_opt.py      # CandidateOptimizer seam: Continuous | DiscreteArgmax
  policy.py       # Policy (ask/tell) + SimpleBO, UncertaintySampling; uq_requirement capability
  loop.py         # run_campaign(...) — the ask/tell driver (resume, retry/halt, denoised recommend)
  history.py      # Record{candidate, observation, outputs, aux, fault} + History + DataStore
  experiment.py   # Experiment / PosteriorSampler / Supervisor protocols, Verdict
  config.py       # name -> class registries
  goal.py         # Goal protocol (BAX goal-as-algorithm)   [stub — F1]
  reduce.py       # ObjectiveReducer protocol (composite SBI) [stub — M2]
  interop/
    xopt.py       # gest-api VOCS <-> spec + CorePolicyGenerator [stub — F3]
```

Three design properties are preserved from day one so agent integration is later
a ~day of wrapper work (not an architecture change): **headless-first**,
**typed JSON-serializable `ProblemSpec` + run state** (no live-object references),
and **ask/tell as the public contract**.

## Develop

```bash
conda activate abbot          # torch, botorch, pydantic, gest-api, pytest, ruff
pip install -e .
pytest        # SimpleBO recovers a known optimum; EnsembleModel passes the Model
              # contract; ProblemSpec round-trips to JSON
ruff check src/ tests/
```

## Roadmap

- **M1** — shared core skeleton on botorch *(current)*.
- **M2** — SBI design-optimization app (the lead build): composite objective +
  `SBIPosteriorSampler` + real experiments. Acceptance = reproduce sbido's
  benchmark entirely on the `abbot` core.
- **M3** — non-expert lab BO tool: new web frontend on the shared core, with a
  type-polymorphic variable editor.
- **F-tier** — BAXstics goal-BAX port, full nn-bax productionization + upstreaming
  to Xopt, Xopt/Badger interop via gest-api, and agent integration (setup-copilot,
  then an MCP server).
