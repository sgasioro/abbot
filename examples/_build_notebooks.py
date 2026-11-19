"""Build the example notebooks programmatically with nbformat.

Run: ``python examples/_build_notebooks.py`` then execute them (the Makefile-free
way is ``jupyter nbconvert --to notebook --execute --inplace examples/*.ipynb``).
Kept in-repo so the notebooks are regenerable rather than hand-edited JSON. Budgets
are deliberately tiny so each notebook runs in seconds (they are CI-smoke-tested).

NOTE: this OVERWRITES every notebook (clearing committed outputs), so always
re-execute ALL of them after running this, not just the one you edited.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent


def _nb(cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


# --------------------------------------------------------------------------- #
# 01 — Lab BO (single objective, mixed continuous + enumerated)
# --------------------------------------------------------------------------- #

lab_bo = _nb([
    md("""
# 1 · Lab Bayesian optimization

The simplest abbot workflow: define a design space, pick an objective, and let
abbot propose experiments that climb toward the optimum. This is the "give a lab a
BO tool" case — a non-expert describes the knobs and the goal, nothing more.
"""),
    code("""
import torch
from abbot import (
    ProblemSpec, ContinuousDomain, OutputSpec, run_campaign, MockAnalyticExperiment,
)
torch.manual_seed(0)
"""),
    md("""
### Define the problem
Two continuous reaction knobs (`temp`, `conc`) and a single objective to
**maximize** (`yield`). An output's `direction` makes it an objective
(`MAXIMIZE`/`MINIMIZE`); `OBSERVE` would model it without optimizing it.
"""),
    code("""
spec = ProblemSpec(
    domain=[
        ContinuousDomain(name="temp", lower=20.0, upper=80.0),
        ContinuousDomain(name="conc", lower=0.0, upper=1.0),
    ],
    outputs=[OutputSpec(name="yield", direction="MAXIMIZE")],
)
spec.variable_names
"""),
    md("""
### A stand-in experiment
`MockAnalyticExperiment` is a seeded, smooth virtual lab — it stands in for a real
bench measurement so the notebook is self-contained. In practice you'd pass a
`ManualExperiment` (a human enters the measured value) or any object with a
`run(design) -> value` method, where `design` is a `{variable name: value}` dict
like `{"temp": 54.2, "conc": 0.31}`.
"""),
    code("""
experiment = MockAnalyticExperiment(spec, seed=1)
"""),
    md("""
### Run the campaign
`run_campaign` drives the propose → evaluate → retrain loop: fit a GP surrogate on
what's been measured, propose the next experiment, evaluate, repeat. Everything
except the spec and the experiment is inferred from the spec: one objective means
`SimpleBO` (noisy expected improvement) on a single-output GP, seeded with a few
space-filling (Sobol) points. Here that's 5 seeds, then 8 informed steps.

Each of those choices is a keyword you can override: `policy=`,
`model_constructor=`, `initial_designs=`, and `evaluate=` (for when the
experiment returns raw data, like a spectrum, that needs reducing to outputs).
"""),
    code("""
import warnings
warnings.filterwarnings("ignore")  # quiet botorch's tiny-data GP-fit warnings

result = run_campaign(spec, experiment, n_shots=8)
"""),
    md("### The recommendation\nabbot hands back the denoised best design (posterior-mean argmax, not the luckiest noisy shot)."),
    code("""
best = result.best
print("recommended:", {n: round(float(v), 3) for n, v in zip(spec.variable_names, best.candidate.tolist())})
print("yield:", round(float(best.outputs), 3))
print("steps taken:", len(result.history))
"""),
    md("### Convergence\nBest objective seen so far, per step."),
    code("""
import matplotlib.pyplot as plt
from abbot.analysis import convergence

ys = convergence(result.history, spec)
plt.figure(figsize=(6, 3.2))
plt.plot(range(len(ys)), ys, "-o", color="#4f46e5", ms=4)
plt.xlabel("step"); plt.ylabel("best yield so far"); plt.title("Lab BO convergence")
plt.tight_layout(); plt.show()
"""),
    md("""
### Categorical / discrete variables
abbot's design space is **type-polymorphic**: an `EnumeratedDomain` holds a finite
set of allowed values (a catalyst choice, a discrete grid). For an *all-enumerated*
space abbot automatically uses the discrete optimizer, which enumerates the
candidate set rather than coercing it to a float range — the call is unchanged:

```python
from abbot import EnumeratedDomain

spec = ProblemSpec(
    domain=[EnumeratedDomain(name="catalyst", values=[0.0, 1.0, 2.0, 3.0])],
    outputs=[OutputSpec(name="yield", direction="MAXIMIZE")],
)
result = run_campaign(spec, experiment, n_shots=8)
```

*(Mixed continuous **and** enumerated spaces in one problem need
`optimize_acqf_mixed`, which is a deferred extension point — abbot raises a clear
error rather than silently relaxing the discrete axis.)*

**What this showed:** a non-expert described continuous knobs and a single
objective; abbot fit a surrogate, proposed experiments, and recovered the best
setting — no ML expertise required in the loop.
"""),
])


# --------------------------------------------------------------------------- #
# 02 — Multi-objective (ParetoBO)
# --------------------------------------------------------------------------- #

multi_obj = _nb([
    md("""
# 2 · Multi-objective optimization

Many lab goals trade off against each other — maximize yield *and* minimize cost.
abbot is **natively multi-objective**: declare several objectives (each with its
own direction) and `ParetoBO` searches for the Pareto front rather than collapsing
everything to one number.
"""),
    code("""
import torch, warnings
warnings.filterwarnings("ignore")
from abbot import (
    ProblemSpec, ContinuousDomain, OutputSpec, run_campaign, MockAnalyticExperiment,
)
torch.manual_seed(0)
"""),
    md("""
### A two-objective problem
`f1` to maximize and `f2` to maximize, over a 2-D design space. (Directions are
declared per output; abbot handles the signs internally — you always report
raw physical values.)
"""),
    code("""
spec = ProblemSpec(
    domain=[
        ContinuousDomain(name="x1", lower=0.0, upper=1.0),
        ContinuousDomain(name="x2", lower=0.0, upper=1.0),
    ],
    outputs=[
        OutputSpec(name="f1", direction="MAXIMIZE"),
        OutputSpec(name="f2", direction="MAXIMIZE"),
    ],
)
experiment = MockAnalyticExperiment(spec, seed=5)  # two bumps peaking in different places
"""),
    md("""
Two objectives, so `run_campaign` picks `ParetoBO` and an independent GP per
output (`ModelListConstructor`) on its own — the call is the same as notebook 1.

_(ParetoBO's hypervolume acquisition is the heaviest policy — a small budget keeps this notebook to a few seconds; raise `n_shots` for a denser front.)_
"""),
    code("""
result = run_campaign(spec, experiment, n_shots=8)
print("Pareto set size:", len(result.recommended))
"""),
    md("### The recovered Pareto front\nEach recommended design is non-dominated: you can't improve one objective without sacrificing the other."),
    code("""
import matplotlib.pyplot as plt

all_Y = result.history.outputs.numpy()
front = result.designs  # recommended designs
front_Y = result.history.outputs[result.recommended].numpy()

plt.figure(figsize=(5.2, 4))
plt.scatter(all_Y[:, 0], all_Y[:, 1], c="#c7d2fe", label="all evaluations")
plt.scatter(front_Y[:, 0], front_Y[:, 1], c="#4f46e5", label="Pareto set", zorder=3)
plt.xlabel("f1"); plt.ylabel("f2"); plt.title("Recovered Pareto front")
plt.legend(); plt.tight_layout(); plt.show()
"""),
    md("""
**What this showed:** with two competing objectives, abbot returns a *set* of
non-dominated designs (the trade-off frontier) rather than a single answer — the
scientist picks the operating point that suits them.
"""),
])


# --------------------------------------------------------------------------- #
# 03 — Composite SBI
# --------------------------------------------------------------------------- #

composite_sbi = _nb([
    md("""
# 3 · Composite simulation-based inference (SBI)

This is abbot's lead differentiator — the case that does **not** fit a plain
BO/DataFrame tool. Each experiment doesn't return a scalar directly; it returns an
*observation*, from which we infer a **posterior over latent parameters** `theta`,
and the objective `F` is a property of that posterior (e.g. "how much posterior
mass lands in a target band").

abbot wires this with `sbi_evaluate(posterior_sampler, reducer, spec)`: it maps
observation → posterior samples over `theta` → a scalar `F`, and caches the `theta`
samples on each record so an annealed objective can be re-scored without
re-sampling.
"""),
    code("""
import torch, warnings
warnings.filterwarnings("ignore")
from abbot import (
    ProblemSpec, ContinuousDomain, OutputSpec, Box,
    Campaign, PosteriorSampler, Rescore, sbi_evaluate,
)
torch.manual_seed(0)
"""),
    md("""
### A toy posterior sampler
In a real SBI app this wraps a trained neural density estimator (e.g. an `sbi`
`DirectPosterior`). Here it's a closed-form Gaussian around the observed value, so
the notebook needs no trained model. The design `x` controls the latent `theta*`;
we want the posterior over `theta` to sit inside the band `[4, 6]`.
"""),
    code("""
class ToyExperiment:
    def run(self, design):                    # design = {"x": 0.42}
        return torch.tensor([design["x"] * 10.0])   # observation centered at theta* = 10*x

class ToyPosterior(PosteriorSampler):
    def sample(self, observation, n=512):
        mean = observation.reshape(-1)
        return mean + 0.5 * torch.randn(n, mean.shape[0])  # posterior over theta

spec = ProblemSpec(
    domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
    outputs=[OutputSpec(name="F", direction="MAXIMIZE")],
    boxes=[Box(lower=[4.0], upper=[6.0])],   # target band on theta
)
"""),
    md("""
### The composite objective
The reducer is yours: it turns a posterior into the number you want to maximize.
Here it is the (soft) fraction of `theta` samples inside the target band, with
`beta` setting how sharp the band edges are. `sbi_evaluate` chains sampler →
reducer and returns `(F, theta)` so `theta` is cached as `aux`.
"""),
    code("""
def reducer(theta, spec, beta=5.0):
    lo, hi = spec.boxes[0].lower[0], spec.boxes[0].upper[0]
    return (torch.sigmoid(beta * (theta - lo)) * torch.sigmoid(beta * (hi - theta))).mean()

evaluate = sbi_evaluate(ToyPosterior(), reducer, spec, n_samples=256)

campaign = Campaign(spec)  # SimpleBO + single-output GP, inferred from the spec
campaign.seed(torch.tensor([[0.05], [0.5], [0.95]], dtype=torch.double))
for _ in range(8):
    campaign.step(ToyExperiment(), evaluate)
"""),
    code("""
rec = campaign.recommend()
best = campaign.history.records[rec[0]]
print("recommended x:", round(float(best.candidate), 3), "(theta* lands in [4,6] near x=0.5)")
print("cached theta samples per record (aux):", campaign.history.records[0].aux.shape)
"""),
    md("""
### Re-scoring from cached posteriors
Because `theta` is cached, the objective can be re-reduced — here with a sharper
`beta` — without re-running the (expensive) inference. `Rescore(reducer, schedule)`
does this; passed as `Campaign(spec, targets=Rescore(...))` it re-scores every
clean shot at each step, so the model, the proposals and the hand-over follow an
annealed objective.
"""),
    code("""
sharp = Rescore(reducer, lambda t: {"beta": 50.0})(campaign.history, spec)
print("re-scored F (sharper beta), first 5:", [round(v, 3) for v in sharp.reshape(-1)[:5].tolist()])
"""),
    md("""
**What this showed:** the objective is a functional of an inferred posterior, not a
direct measurement — and the posterior samples are cached so the objective can be
re-annealed cheaply. This composite-SBI pattern is the thing abbot owns that a
tabular BO tool structurally cannot represent.
"""),
])


# --------------------------------------------------------------------------- #
# 04 — Goal-driven BAX
# --------------------------------------------------------------------------- #

goal_bax = _nb([
    md("""
# 4 · Goal-driven search (BAX)

Sometimes the goal isn't "find the maximum" but "find the **region** of design
space where the outputs satisfy a target." Bayesian Algorithm Execution (BAX) runs
a *goal algorithm* on sampled surrogate functions and focuses sampling to resolve
that target region.

abbot's `BAXPolicy` ports the InfoBAX / SwitchBAX / MeanBAX / US strategies onto
botorch; `Goal` objects (`PercentileSet`, `BandIntersection`, `Wishlist`) define
the target. Here we look for designs whose two outputs land inside a target box.
"""),
    code("""
import torch, warnings
warnings.filterwarnings("ignore")
from abbot import (
    ProblemSpec, ContinuousDomain, OutputSpec, Box,
    BAXPolicy, goals_from_spec, Campaign,
)
torch.manual_seed(0)
"""),
    md("""
In BAX the outputs `a` and `b` are **modeled features a Goal consumes**, not things
to maximize — so we declare them `OBSERVE` (modeled, not optimized). The *target
region* on `(a, b)` is what encodes the intent.
"""),
    code("""
spec = ProblemSpec(
    domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
    outputs=[
        OutputSpec(name="a", direction="OBSERVE"),
        OutputSpec(name="b", direction="OBSERVE"),
    ],
    boxes=[Box(lower=[0.4, 0.4], upper=[0.6, 0.6])],  # target region on (a, b)
)

class TargetExperiment:
    # a(x), b(x) both pass through the [0.4,0.6] box near x=0.5
    def run(self, design):
        return {"a": design["x"], "b": design["x"]}   # outputs by name

goal = goals_from_spec(spec)              # Wishlist built from the target box
policy = BAXPolicy(goal, strategy="MeanBAX", n_candidates=200)
"""),
    md("### Run the goal-driven campaign\nBAXPolicy proposes points from a candidate grid, concentrating where the model believes the target region is."),
    code("""
campaign = Campaign(spec, policy)  # model defaults to a GP per output
campaign.seed(policy.candidate_grid(spec)[torch.randperm(200)[:6]])
for _ in range(10):
    campaign.step(TargetExperiment())
"""),
    code("""
# How many sampled designs landed in the target region vs. the base rate?
def in_box(x):
    a, b = float(x), float(x)
    return 0.4 <= a <= 0.6 and 0.4 <= b <= 0.6

shots = [r.candidate for r in campaign.history.records[6:]]  # exclude seeds
hit = sum(in_box(x) for x in shots) / len(shots)
print(f"fraction of BAX shots in the target region: {hit:.2f}")
print("(the target box is ~20% of the x-range, so > 0.2 means BAX concentrated there)")
"""),
    md("""
### Beyond boxes: arbitrary goals
`Wishlist`/`BandIntersection` cover target boxes, but a goal is just a function
from predicted outputs to target indices. `FunctionGoal` wraps **any**
`fn(f, X) -> indices`, so goals that aren't boxes — level sets, custom top-k,
multi-output logic — plug into the same `BAXPolicy`:

```python
from abbot import FunctionGoal, LevelSet, TopK

# ready-made non-box goals:
goal = LevelSet(threshold=0.5, output=0)      # super-level set on output 0
goal = TopK(k=5, output=0)                    # the 5 best candidates by output 0

# or anything you can express as (values, candidates) -> indices:
goal = FunctionGoal(lambda f, X: (f[:, 0] > f[:, 1]).nonzero(as_tuple=True)[0].tolist())
```

**What this showed:** given a *target region* (or any goal-as-code) rather than a
maximum, BAX steers sampling toward designs that satisfy it. This is the
goal-as-algorithm capability abbot ports from multibax, now on botorch, with
`FunctionGoal` for any goal you can write as code.
"""),
])


# --------------------------------------------------------------------------- #
# 05 — Multi-fidelity: calibrating an expensive simulation
# --------------------------------------------------------------------------- #

multi_fidelity = _nb([
    md("""
# 5 · Multi-fidelity: calibrating an expensive simulation

A simulation has two parameters to calibrate against data, and its accuracy depends
on a knob — here the number of atoms. Small systems run in seconds but carry a
finite-size bias; the size you trust takes an hour. Cheap runs are still
informative: they show roughly where the good parameters are.

Declaring the knob as a **source**, with the value you trust as its **target**, is
all abbot needs. The model learns how cheap runs relate to runs at the target,
`SimpleBO` picks which size to run by information per unit of cost, and the cost
itself is learned from what runs report.
"""),
    code("""
import torch
from abbot import (
    Campaign, ContinuousDomain, IntegerDomain, OutputSpec, ProblemSpec, SimpleBO,
    Source,
)
from abbot.supervision import Supervisor, cost_cap
torch.manual_seed(0)
"""),
    md("""
### Define the problem
`eps` and `sigma` are the parameters to calibrate; `misfit` (to minimize) is the
discrepancy between simulation and data. `atoms` is a source: an input the
simulation needs, but not something to recommend. `log=True` because cost and bias
scale with size, and `target=256` is the size we trust.
"""),
    code("""
spec = ProblemSpec(
    domain=[
        ContinuousDomain(name="eps", lower=0.0, upper=1.0),
        ContinuousDomain(name="sigma", lower=0.0, upper=1.0),
    ],
    sources=[Source(domain=IntegerDomain(name="atoms", lower=32, upper=256),
                    target=256, log=True)],
    outputs=[OutputSpec(name="misfit", direction="MINIMIZE")],
)
"""),
    md("""
### The simulation
A stand-in: the misfit has a good basin near `(0.3, 0.7)` and a broad, worse one
near `(0.8, 0.2)`, plus a finite-size bias that shrinks as the system grows.
`cost(design)` reports what a run cost (here, core-hours that grow as atoms²).
Without it abbot records each run's wall-clock time, which on a cluster would
include queueing.
"""),
    code("""
import math

def misfit(eps, sigma, atoms):
    good = math.exp(-((eps - 0.3) ** 2 + (sigma - 0.7) ** 2) / 0.05)
    broad = 0.5 * math.exp(-((eps - 0.8) ** 2 + (sigma - 0.2) ** 2) / 0.1)
    finite_size = 4.0 / atoms * (1 + 2 * eps)       # vanishes for large systems
    return 1 - good - broad + finite_size

class Simulation:
    def run(self, design):
        return [misfit(design["eps"], design["sigma"], design["atoms"])]

    def cost(self, design):
        return (design["atoms"] / 32) ** 2 / 60     # core-hours

sim = Simulation()
target_run = sim.cost({"atoms": 256})
print(f"one run at the target: {target_run:.2f} core-hours; at 32 atoms: {sim.cost({'atoms': 32}):.3f}")
"""),
    md("""
### Run until a budget is spent
`cost_cap` stops the campaign once the recorded cost reaches a budget: here, the
price of 5 runs at the target.
"""),
    code("""
budget = 5 * target_run
campaign = Campaign(spec, supervisor=Supervisor(stops=[cost_cap(budget)]))
while not campaign.stopped:
    campaign.step(sim)

atoms = [int(r.candidate[2]) for r in campaign.history.records]
print("atoms per run:", atoms)
print(f"{len(atoms)} runs for {campaign.history.cost_spent:.1f} core-hours")
print("learned cost:", campaign.policy.source_cost)
"""),
    md("""
The campaign seeded itself with a few space-filling runs, two of them at the
target. After that most runs are small. The learned cost recovers the atoms² scaling
from what the runs reported.

### The answer, at the target
Recommendations are judged at the target: the model's prediction at 256 atoms, even
for a design that was only ever run small. `run_at_target` says whether it was.
"""),
    code("""
result = campaign.result()
best = campaign.history.records[result.recommended[0]]
eps, sigma = best.candidate[:2].tolist()
print(f"recommended eps={eps:.3f}, sigma={sigma:.3f}  (good basin: 0.3, 0.7)")
print("run at the target:", result.run_at_target[0], "| model std there:", result.target_std[0])
"""),
    md("""
### The same budget at the target only
For comparison, plain BO with every run at 256 atoms gets 5 runs for the same
budget.
"""),
    code("""
target_only = Campaign(spec, policy=SimpleBO(choose_sources=False),
                       supervisor=Supervisor(stops=[cost_cap(budget)]),
                       initial_designs=spec.at_target(spec.sample_designs(3)))
while not target_only.stopped:
    target_only.step(sim)

rec = target_only.history.records[target_only.result().recommended[0]].candidate
best = min(misfit(e / 100, s / 100, 256) for e in range(101) for s in range(101))
print(f"best possible:  misfit {best:.3f} at the target")
print(f"multi-fidelity: misfit {misfit(eps, sigma, 256):.3f}")
print(f"target only:    misfit {misfit(*rec[:2].tolist(), 256):.3f}")
"""),
    md("""
### Other kinds of source
- **A knob where smaller is more accurate** (a convergence tolerance, a mesh
  spacing): `Source(..., accurate="low")`.
- **Different codes, or simulation vs experiment** (unordered): an enumerated
  source with whole-number labels, e.g.
  `Source(domain=EnumeratedDomain(name="code", values=[0, 1]), target=1)`.
- **Goal-driven search**: `BAXPolicy(goal, strategy="InfoBAX")` chooses sources the
  same way, by information about the goal per unit of cost.
- **Known noise per run** (e.g. a standard error from block averaging): pass it as
  `output_var` and the model uses it instead of one shared noise level.

**What this showed:** with the fidelity knob declared as a source, abbot spends
most of the budget on cheap runs, learns their cost and bias, and still recommends
for the size you trust. With five target runs' worth of budget, the cheap runs found
the good basin; five runs at the target alone usually don't (this holds across
seeds, not just this one).
"""),
])


# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# 06 — Your own neural network as the model (abbot.nn, experimental)
# --------------------------------------------------------------------------- #

neural_surrogates = _nb([
    md("""
# 6 · Your own neural network as the model

abbot's default model is a Gaussian process. When you'd rather use a neural
network (lots of runs, inputs that aren't a handful of numbers, a network you've
already designed), `abbot.nn` wraps it so every policy can use it. You write the
network; abbot scales the inputs and outputs, trains it, starts each fit from the
last one, and turns it into a model with uncertainty.

`abbot.nn` is **experimental**: its options and defaults may change.
"""),
    code("""
import torch
from abbot import BAXPolicy, Campaign, ContinuousDomain, LevelSet, OutputSpec, ProblemSpec
from abbot.nn import DeepKernelConstructor, MCDropoutConstructor, NetEnsembleConstructor
torch.manual_seed(0)
"""),
    md("""
### The problem
Two knobs and one measured property. We want the region where the property is at
least 0.5, which here is two separate patches.
"""),
    code("""
spec = ProblemSpec(
    domain=[
        ContinuousDomain(name="x1", lower=0.0, upper=1.0),
        ContinuousDomain(name="x2", lower=0.0, upper=1.0),
    ],
    outputs=[OutputSpec(name="signal", direction="OBSERVE")],
)

def signal(x1, x2):
    a = torch.exp(-((x1 - 0.25) ** 2 + (x2 - 0.7) ** 2) / 0.02)
    b = torch.exp(-((x1 - 0.75) ** 2 + (x2 - 0.3) ** 2) / 0.03)
    return a + 0.8 * b

class Experiment:
    @staticmethod
    def run(design):
        x1, x2 = torch.tensor(design["x1"]), torch.tensor(design["x2"])
        return [float(signal(x1, x2))]

goal = LevelSet(threshold=0.5, output="signal")
"""),
    md("""
### Your network
`make_net(d_in, d_out)` returns any `torch.nn.Module` from `(N, d_in)` to
`(N, d_out)`. abbot calls it once per ensemble member, converts it to float64 and
puts it on the campaign's device. Inputs arrive scaled to [0, 1] when you pass
`bounds`.
"""),
    code("""
def make_net(d_in, d_out):
    return torch.nn.Sequential(
        torch.nn.Linear(d_in, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, d_out),
    )
"""),
    md("""
### An ensemble, in a campaign
`NetEnsembleConstructor` trains five copies from different starting weights; where
they disagree, the model is unsure. It plugs into `Campaign` like any model. Here
SwitchBAX measures inside the region the model currently predicts, and when there's
nothing new there, asks where a run would say most about the region (InfoBAX,
which updates the network by fine-tuning a copy on each guess).
"""),
    code("""
model = NetEnsembleConstructor(make_net, n_members=5, bounds=spec.bounds_tensor())
policy = BAXPolicy(goal, "SwitchBAX", n_candidates=400)
campaign = Campaign(spec, policy, model)
for _ in range(25):
    campaign.step(Experiment)
print(len(campaign.history), "runs")
"""),
    md("""
### How well does it know the region?
Compare the region the model predicts (on its mean) with the true one, over the
candidate grid. Jaccard is the overlap divided by the union: 1 is a perfect match.
"""),
    code("""
grid = policy.candidate_grid(spec)
truth = set(goal.bind(spec).identify_subspace(signal(grid[:, 0], grid[:, 1]).unsqueeze(-1)))

def jaccard(campaign):
    with torch.no_grad():
        mean = campaign.current_model().posterior(grid).mean
    found = set(goal.bind(spec).identify_subspace(mean))
    return len(found & truth) / len(found | truth)

print(f"Jaccard: {jaccard(campaign):.2f}")
"""),
    code("""
import matplotlib.pyplot as plt

with torch.no_grad():
    post = campaign.current_model().posterior(grid)
runs = campaign.history.designs
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, values, title in [(ax[0], post.mean[:, 0], "mean"), (ax[1], post.variance[:, 0].sqrt(), "spread across members")]:
    s = a.scatter(grid[:, 0], grid[:, 1], c=values, s=12)
    a.scatter(runs[:, 0], runs[:, 1], c="red", s=15, label="runs")
    a.set_title(title); fig.colorbar(s, ax=a)
ax[0].legend()
plt.show()
"""),
    md("""
### Other ways to get uncertainty
Swap the constructor; nothing else changes.

- **MC dropout:** one network with `Dropout` layers, sampled with dropout on.
  Each sample is one dropout mask, shared by every design, so a sample is one
  function.
- **Deep kernel:** the network maps designs to a few features and a GP works on
  them. It's a GP, so it gives calibrated uncertainty and works with every policy.

For comparison, the default GP runs on the same budget. On a smooth problem with a
few dozen runs, like this one, expect the GP to do better: networks earn their
place with many runs, many inputs, or inputs that aren't numbers.
"""),
    code("""
def make_dropout_net(d_in, d_out):
    return torch.nn.Sequential(
        torch.nn.Linear(d_in, 64), torch.nn.ReLU(), torch.nn.Dropout(0.1),
        torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Dropout(0.1),
        torch.nn.Linear(64, d_out),
    )

for name, model in [
    ("MC dropout", MCDropoutConstructor(make_dropout_net, n_samples=16, bounds=spec.bounds_tensor())),
    ("deep kernel", DeepKernelConstructor(make_net, n_features=4, bounds=spec.bounds_tensor())),
    ("GP (the default)", None),
]:
    torch.manual_seed(0)
    policy = BAXPolicy(goal, "SwitchBAX", n_candidates=400)
    other = Campaign(spec, policy, model)
    for _ in range(25):
        other.step(Experiment)
    print(f"{name}: Jaccard {jaccard(other):.2f}")
"""),
    md("""
### Training: yours to tune
The defaults are a starting point, checked only on small problems like this one.

- `epochs` (first fit), `warm_epochs` (later fits, starting from the last weights;
  `None` starts fresh each time), `refit_epochs` (InfoBAX's fine-tunes);
- `lr`, `weight_decay`, `batch_size` for the default trainer (Adam on the squared
  error), or `train=` for your own: `train(net, X, Y, epochs)`, where `net` maps raw
  designs to standardized outputs and `Y` has NaN where a run didn't measure an
  output;
- `noise=` when you know the measurement noise (a variance per output).
"""),
    code("""
def train(net, X, Y, epochs):
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    seen = ~Y.isnan()
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.huber_loss(net(X)[seen], Y[seen])
        loss.backward()
        opt.step()

model = NetEnsembleConstructor(make_net, bounds=spec.bounds_tensor(), train=train, epochs=800)
"""),
    md("""
### Inputs that aren't numbers
When a design is a structure (a molecule, a crystal), give each one an index and
let `featurize` turn indices into whatever your network takes. abbot passes the raw
designs to `featurize` and doesn't scale them.
"""),
    code("""
from abbot import IntegerDomain

structures = torch.randn(50, 8, dtype=torch.double)   # stand-in descriptors, one row per structure
indexed = ProblemSpec(
    domain=[IntegerDomain(name="structure", lower=0, upper=49)],
    outputs=[OutputSpec(name="signal", direction="OBSERVE")],
)

def featurize(X):                                     # (N, 1) indices -> (N, 8) inputs
    return structures[X[:, 0].long()]

model = NetEnsembleConstructor(lambda d_in, d_out: make_net(8, d_out), featurize=featurize)
"""),
    md("""
### When a network helps
A standard GP assumes the output varies about equally fast everywhere, with one
lengthscale per input. Here the output **jumps** by 3 across a wavy boundary in
`x1`, `x2`, and a small **bump** in `x1`, `x3` sits on the high side. The goal is
the high side near the top of the bump (y ≥ 3.4), about 3% of the space: finding
it needs both the jump and the bump.
"""),
    code("""
import logging
import math
import warnings

# Quiet for a short output: the policy says, while the model predicts no target
# anywhere, that it explores by uncertainty instead; GPyTorch notes numerical jitter.
logging.getLogger("abbot").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*added jitter.*")

jump_spec = ProblemSpec(
    domain=[ContinuousDomain(name=n, lower=0.0, upper=1.0) for n in ("x1", "x2", "x3")],
    outputs=[OutputSpec(name="y", direction="OBSERVE")],
)

def jump_and_bump(X):
    x1, x2, x3 = X[..., 0], X[..., 1], X[..., 2]
    high = (x2 > 0.5 + 0.2 * torch.sin(2 * math.pi * x1)).double()
    return 3 * high + 0.8 * torch.exp(-((x1 - 0.3) ** 2 + (x3 - 0.6) ** 2) / 0.04)

class JumpExperiment:
    @staticmethod
    def run(design):
        x = torch.tensor([design["x1"], design["x2"], design["x3"]], dtype=torch.double)
        return [float(jump_and_bump(x))]

jump_goal = LevelSet(threshold=3.4, output="y")

def wider_net(d_in, d_out):
    return torch.nn.Sequential(
        torch.nn.Linear(d_in, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, d_out),
    )

found = {}
for name, model in [
    ("GP (the default)", None),
    ("ensemble", NetEnsembleConstructor(wider_net, bounds=jump_spec.bounds_tensor())),
]:
    torch.manual_seed(0)
    policy = BAXPolicy(jump_goal, "InfoBAX", n_candidates=1000)
    found[name] = Campaign(jump_spec, policy, model)
    for _ in range(80):
        found[name].step(JumpExperiment)
    grid = policy.candidate_grid(jump_spec)
    truth = set(jump_goal.bind(jump_spec).identify_subspace(jump_and_bump(grid).unsqueeze(-1)))
    with torch.no_grad():
        mean = found[name].current_model().posterior(grid).mean
    est = set(jump_goal.bind(jump_spec).identify_subspace(mean))
    print(f"{name}: Jaccard {len(est & truth) / len(est | truth):.2f}")
"""),
    md("""
The GP finds none of the region. Its check shows why: it follows the jump with a
short lengthscale in `x2`, and gives `x1` and `x3` lengthscales longer than their
whole range, in effect deciding they barely matter. So it misses both the wavy
boundary and the bump, and reads part of what it can't fit as noise. It also flags
runs it can't explain ("a sharp feature"), which is the hint to try something
else.

The ensemble has no single lengthscale to compromise on. Over several seeds it
found most of the region (Jaccard about 0.5 to 1.0) while the GP found almost
none; the seed here is one of those runs. InfoBAX mattered too: with MeanBAX or
PS-BAX, the ensemble sometimes never found the region at all.
"""),
    code("""
for fit in found["GP (the default)"].check().fits:
    print(fit.name, {k: round(v, 2) for k, v in fit.lengthscales.items()})
print(found["GP (the default)"].check())
"""),
    md("""
### Which policies
- **MeanBAX, PS-BAX, SwitchBAX, InfoBAX** use the ensemble's and dropout's samples
  directly.
- **US and the BO policies** (`SimpleBO`, `ParetoBO`) run too, with a warning: they
  read the spread across members as calibrated uncertainty, which it isn't.
- **The deep kernel** works with everything.

More in the API docs: *Neural-network surrogates (experimental)*.
"""),
])

NOTEBOOKS = {
    "01_lab_bo.ipynb": lab_bo,
    "02_multi_objective.ipynb": multi_obj,
    "03_composite_sbi.ipynb": composite_sbi,
    "04_goal_bax.ipynb": goal_bax,
    "05_multi_fidelity.ipynb": multi_fidelity,
    "06_neural_surrogates.ipynb": neural_surrogates,
}


def main() -> None:
    for name, nb in NOTEBOOKS.items():
        path = HERE / name
        nbf.write(nb, path)
        print("wrote", path)


if __name__ == "__main__":
    main()
