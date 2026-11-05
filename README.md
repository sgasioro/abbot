# abbot

**Goal-driven Bayesian experimental design for the lab.**

abbot helps you choose which experiment to run next. You describe the knobs you
can turn, what you measure, and what you want to achieve; abbot suggests a
setting, you (or your instrument, or a simulation) run it and report the result,
and it learns from every run. The goal can be:

- **an optimum:** single- or multi-objective Bayesian optimization;
- **a region:** every setting that hits a target band, a level set, the best point
  per band (Bayesian algorithm execution, BAX);
- **anything you can write as code** over the model's predictions.

The aim is to make sophisticated methods (goal-driven search, multiple fidelities
or sources, constraints, supervision) available through a short, readable spec
with safe defaults, so a lab gets them without writing BoTorch code or tuning an
algorithm.

It's built for how lab work goes: a measurement that isn't one number (a
spectrum, replicates, a posterior from simulation-based inference), runs that
are slow or fail, and a person who stays in charge of halts, retries and
overrides. A web UI runs a campaign with no code. Built on [BoTorch](https://botorch.org):
abbot uses its models and standard acquisition functions, and adds goal-driven
(BAX) acquisitions and the campaign around them.

**Status: alpha.** APIs marked *Stable* in [`docs/API.md`](https://github.com/sgasioro/abbot/blob/main/docs/API.md) are safe to
build on; *Provisional* ones may change between releases. Issues and pull requests
are welcome.

## Install

abbot needs Python 3.11 or later (tested on 3.11–3.13). Install torch first, from
PyTorch's index (the CPU build is enough; see [Local setup](https://github.com/sgasioro/abbot#local-setup)
for a GPU):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "abbot[ui]"
```

The `ui` extra adds the web frontend, `mcp` the agent server; `xopt` and `sbi` add
those integrations.

## Quickstart

```python
from abbot import ProblemSpec, OutputSpec, ContinuousDomain, Campaign

def run_in_lab(temp):                          # a stand-in: your measurement here
    return 1.0 - ((temp - 55.0) / 30.0) ** 2

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
    measured_yield = run_in_lab(x["temp"])
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
  `LevelSet(0.9, output="purity")`), and pass it in: `Campaign(spec,
  BAXPolicy(goal))`.
- **Keep an output in range** — `OutputSpec(name="purity", direction="OBSERVE",
  min_val=0.9)`: proposals and the recommendation favor designs likely to meet it.
- **Cheap and accurate runs** — add `sources=[Source(domain=EnumeratedDomain(
  name="code", values=["sim", "experiment"]), target="experiment")]` to the spec:
  abbot chooses which to run, trading cost against information, and recommends
  at the target.
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
- **Runs that fail, or regions that give nothing** — a failed run is retried, then
  halts for a person (`campaign.resolve("retry" | "exclude" | "bad_region")`); a
  check like `Supervisor(faults=[per_shot(is_dark, bad_region="dark")])` marks a
  region as unusable, and proposals steer away from it.
- **No code** — `python -m abbot.ui` launches the lab web frontend over the same
  `Campaign`.
- **From an AI agent** — `abbot-mcp --dir ~/abbot-campaigns` (with the `mcp` extra)
  serves named, saved campaigns over MCP. The agent first talks the campaign
  through with you (ranges, limits, goal, budget), then suggests, records results
  and hands a halt back to you. See
  [Agent access](https://github.com/sgasioro/abbot/blob/main/docs/API.md#agent-access-mcp).

See [`docs/HOWTO.md`](https://github.com/sgasioro/abbot/blob/main/docs/HOWTO.md) for "how do I…" answers, [`docs/API.md`](https://github.com/sgasioro/abbot/blob/main/docs/API.md)
for the stability contract (what's safe to build on), [`examples/`](https://github.com/sgasioro/abbot/tree/main/examples)
for a runnable notebook per capability, and
[`docs/ARCHITECTURE.md`](https://github.com/sgasioro/abbot/blob/main/docs/ARCHITECTURE.md) for how abbot is put together
(modules, the seams an application plugs into, the session API).

## Where abbot fits

abbot brings together methods that had been spread across several
problem-specific packages. Labs work in different modes (a control room, a
beamline, a bench), and good tools already exist for some of them. abbot aims to
cover the gaps, following the [gest-api](https://pypi.org/project/gest-api/)
generator standard to promote interoperability and a coherent ecosystem. As it
develops, the emphasis is on what can be shared with neighboring tools and what
is better kept separate; [`docs/ECOSYSTEM.md`](https://github.com/sgasioro/abbot/blob/main/docs/ECOSYSTEM.md) has a first pass
at that split.

## Local setup

```bash
# Option A — conda:
conda env create -f environment.yml
conda activate abbot

# Option B — venv (Python 3.11 or later):
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

Five runnable notebooks in [`examples/`](https://github.com/sgasioro/abbot/tree/main/examples) (install `pip install -e
'.[examples]'`), one per capability: lab BO, multi-objective (`ParetoBO`),
composite-SBI, goal-driven BAX, and multi-fidelity. See [`examples/README.md`](https://github.com/sgasioro/abbot/blob/main/examples/README.md).

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
                       # run before committing)
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
- The NiceGUI lab frontend (`python -m abbot.ui`), with output ranges, travel
  limits, sources, a model choice and `check()`.
- An MCP server (`abbot-mcp`) for agents, over named campaigns saved on disk, with
  a setup assistant: the agent asks what it needs before writing the spec
  (`plan_campaign`, and a Claude Code skill).

**Next:**
1. **A guided UI:** step-by-step setup and plain-language progress, with an
   expert panel that shows (and edits) the whole spec, in the same
   `campaign.json` format the agent server writes.
2. **Your own experiment in the UI:** `python -m abbot.ui --experiment
   module:Class`, then a command adapter (`./measure.sh {temp} {conc}` printing
   JSON) so a script can run the experiment without writing Python.
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
  (see [`docs/ECOSYSTEM.md`](https://github.com/sgasioro/abbot/blob/main/docs/ECOSYSTEM.md)).
- A run-level JSON format that includes cached SBI payloads.

**Ease of use is a priority.** For each core workflow we periodically set the
same problem up in abbot and in neighboring tools, and compare how much a
newcomer must learn and get right by hand, to find what to improve in abbot.

## Citing

If abbot helps your work, please cite the repository (see
[`CITATION.cff`](https://github.com/sgasioro/abbot/blob/main/CITATION.cff)). Its goal-driven policies follow:

- W. Neiswanger, K. A. Wang and S. Ermon, "Bayesian Algorithm Execution:
  Estimating Computable Properties of Black-box Functions Using Mutual
  Information," *ICML* 2021, [arXiv:2104.09460](https://arxiv.org/abs/2104.09460)
  (InfoBAX).
- S. R. Chitturi *et al.*, "Targeted materials discovery using Bayesian
  algorithm execution," *npj Computational Materials* 10, 156 (2024),
  [doi:10.1038/s41524-024-01326-2](https://doi.org/10.1038/s41524-024-01326-2)
  (MeanBAX, SwitchBAX and target-subset goals).

and its models and standard acquisitions come from BoTorch: M. Balandat *et al.*,
"BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization,"
*NeurIPS* 2020, [arXiv:1910.06403](https://arxiv.org/abs/1910.06403).

## License

Apache-2.0; see [`LICENSE`](https://github.com/sgasioro/abbot/blob/main/LICENSE).
