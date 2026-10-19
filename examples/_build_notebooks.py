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
    ProblemSpec, ContinuousDomain, OutputSpec, run_campaign, SyntheticExperiment,
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
`SyntheticExperiment` is a seeded, smooth virtual lab — it stands in for a real
bench measurement so the notebook is self-contained. In practice you'd pass a
`ManualExperiment` (a human enters the measured value) or any object with a
`run(design) -> value` method, where `design` is a `{variable name: value}` dict
like `{"temp": 54.2, "conc": 0.31}`.
"""),
    code("""
experiment = SyntheticExperiment(spec, seed=1)
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
    ProblemSpec, ContinuousDomain, OutputSpec, run_campaign, SyntheticExperiment,
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
experiment = SyntheticExperiment(spec, seed=5)  # two bumps peaking in different places
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
    campaign.step(TargetExperiment(), lambda obs: obs)
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
goal-as-algorithm capability abbot ports from multibax — the same engine that
drives sbido's BAX designers — with `FunctionGoal` giving the arbitrary-goal
flexibility of nn-bax's `FunctionAlgorithm`, now on botorch.
"""),
])


# --------------------------------------------------------------------------- #

NOTEBOOKS = {
    "01_lab_bo.ipynb": lab_bo,
    "02_multi_objective.ipynb": multi_obj,
    "03_composite_sbi.ipynb": composite_sbi,
    "04_goal_bax.ipynb": goal_bax,
}


def main() -> None:
    for name, nb in NOTEBOOKS.items():
        path = HERE / name
        nbf.write(nb, path)
        print("wrote", path)


if __name__ == "__main__":
    main()
