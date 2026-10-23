"""Multi-fidelity benchmark: cost to a good answer at the target.

Each problem has an expensive target source and cheaper, biased ones. Each method
runs a campaign until it has spent ``budget`` (in units of one run at the target) or
made ``max_evals`` runs. After its seeds and after every run, the method's
recommendation is scored by the true value *at the target*: simple regret against
the target's optimum. The regret at a cost checkpoint is the last one at or below
that spend (blank if the seeds alone cost more); a method that stopped below a
checkpoint keeps its last regret there.

Methods:
  target    single-fidelity BO at the target (the spec without sources)
  mf-lt     abbot's default: SimpleBO choosing sources, linear-truncated kernel
  mf-prod   the same with the product kernel (``linear_truncated=False``)
  mf-yvar   mf-lt / mf-prod given each run's noise variance (noisy problem only)
  mf-kg     BoTorch's multi-fidelity knowledge gradient (its tutorial's reference
            setup) on abbot's model and learned cost
  xopt      Xopt's approach, rebuilt in abbot: the source as a design variable
            s in [0, 1] that is also maximized as an objective, with independent
            GPs, cost-weighted NEHVI, the true cost given, and Xopt's default
            reference point; recommends the best posterior mean at s = 1

    python benchmarks/multi_fidelity.py --problems wiggly smooth --seeds 5 --workers 4

Results go to ``--out`` (JSON lines, one per run) and a summary table is printed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import torch

CHECKPOINTS = (1, 2, 5, 10, 20, 30)


# -- problems -------------------------------------------------------------------------


@dataclass
class Problem:
    """``truth(design) -> float`` is the noiseless output, ``cost(design)`` a run's
    cost, ``noise_sd(design)`` the observation noise (or ``None``). ``best`` is the
    optimum of ``truth`` at the target, ``budget`` in target-run units."""

    name: str
    spec_kwargs: dict
    truth: Callable[[dict], float]
    cost: Callable[[dict], float]
    best: float
    budget: float
    max_evals: int
    noise_sd: Callable[[dict], float] | None = None
    methods: tuple[str, ...] = ("target", "mf-lt", "mf-prod", "mf-kg", "xopt")

    def spec(self):
        from abbot import ProblemSpec

        return ProblemSpec(**self.spec_kwargs)

    @property
    def source(self):
        (source,) = self.spec_kwargs["sources"]
        return source

    @property
    def minimize(self) -> bool:
        return self.spec_kwargs["outputs"][0].direction == "MINIMIZE"

    def regret(self, design: dict) -> float:
        value = self.truth({**design, self.source.name: self.source.target})
        return value - self.best if self.minimize else self.best - value


def _grid_best(truth, source, n_design, minimize, n=4001):
    """The optimum at the target of a 1-D problem, on a fine grid."""
    xs = torch.linspace(0, 1, n).tolist()
    vals = [truth({"x": x, source.name: source.target}) for x in xs]
    return min(vals) if minimize else max(vals)


def _atoms_problem(name, bias, noise=None, methods=None, upper=256):
    from abbot import ContinuousDomain, IntegerDomain, OutputSpec, Source

    # By default the target is the top of the range, as Xopt's formulation requires
    # (s = 1); ``upper=512`` puts it mid-range (abbot only).
    source = Source(domain=IntegerDomain(name="atoms", lower=32, upper=upper), target=256,
                    log=True)

    def truth(d):
        return math.sin(6 * d["x"]) + bias(d["x"], d["atoms"])

    kwargs = {
        "domain": [ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        "sources": [source],
        "outputs": [OutputSpec(name="y", direction="MAXIMIZE")],
    }
    return Problem(
        name=name, spec_kwargs=kwargs, truth=truth,
        cost=lambda d: (d["atoms"] / 32.0) ** 2,  # 64x from 32 to the target
        best=_grid_best(truth, source, 1, minimize=False),
        budget=15, max_evals=60, noise_sd=noise,
        methods=methods or ("target", "mf-lt", "mf-prod", "mf-kg", "xopt"),
    )


def _hartmann():
    from botorch.test_functions.multi_fidelity import AugmentedHartmann

    from abbot import ContinuousDomain, OutputSpec, Source

    fn = AugmentedHartmann()  # minimize; optimum -3.32237 at s = 1
    names = [f"x{i}" for i in range(6)]

    def truth(d):
        return float(fn.evaluate_true(torch.tensor([[*(d[n] for n in names), d["s"]]],
                                                   dtype=torch.double)))

    kwargs = {
        "domain": [ContinuousDomain(name=n, lower=0.0, upper=1.0) for n in names],
        "sources": [Source(domain=ContinuousDomain(name="s", lower=0.0, upper=1.0),
                           target=1.0)],
        "outputs": [OutputSpec(name="y", direction="MINIMIZE")],
    }
    return Problem(name="hartmann", spec_kwargs=kwargs, truth=truth,
                   cost=lambda d: 100.0 ** d["s"], best=-3.32237, budget=30, max_evals=150)


def _coded():
    """Two codes: the target (code 1) and a cheap code (0) that's biased."""
    from abbot import ContinuousDomain, EnumeratedDomain, OutputSpec, Source

    def bump(x, y, cx, cy, w):
        return math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * w * w))

    def truth(d):
        base = bump(d["x"], d["y"], 0.7, 0.3, 0.2) + 0.5 * bump(d["x"], d["y"], 0.2, 0.8, 0.15)
        return base + (0.3 * bump(d["x"], d["y"], 0.4, 0.5, 0.3) if d["code"] == 0 else 0.0)

    kwargs = {
        "domain": [ContinuousDomain(name="x", lower=0.0, upper=1.0),
                   ContinuousDomain(name="y", lower=0.0, upper=1.0)],
        "sources": [Source(domain=EnumeratedDomain(name="code", values=[0.0, 1.0]),
                           target=1.0)],
        "outputs": [OutputSpec(name="f", direction="MAXIMIZE")],
    }
    grid = torch.linspace(0, 1, 201).tolist()
    best = max(truth({"x": x, "y": y, "code": 1.0}) for x in grid for y in grid)
    return Problem(name="coded", spec_kwargs=kwargs, truth=truth,
                   cost=lambda d: 50.0 if d["code"] == 1.0 else 1.0, best=best,
                   budget=15, max_evals=60, methods=("target", "mf-lt"))


def _basins():
    """The example notebook's calibration: a good, narrower basin and a broad, worse
    one, with a finite-size bias. Target-only BO with a small budget tends to settle
    in the broad one."""
    from abbot import ContinuousDomain, IntegerDomain, OutputSpec, Source

    def truth(d):
        e, s, a = d["eps"], d["sigma"], d["atoms"]
        good = math.exp(-((e - 0.3) ** 2 + (s - 0.7) ** 2) / 0.05)
        broad = 0.5 * math.exp(-((e - 0.8) ** 2 + (s - 0.2) ** 2) / 0.1)
        return 1 - good - broad + 4.0 / a * (1 + 2 * e)

    kwargs = {
        "domain": [ContinuousDomain(name="eps", lower=0.0, upper=1.0),
                   ContinuousDomain(name="sigma", lower=0.0, upper=1.0)],
        "sources": [Source(domain=IntegerDomain(name="atoms", lower=32, upper=256),
                           target=256, log=True)],
        "outputs": [OutputSpec(name="misfit", direction="MINIMIZE")],
    }
    grid = [i / 200 for i in range(201)]
    best = min(truth({"eps": e, "sigma": s, "atoms": 256}) for e in grid for s in grid)
    return Problem(name="basins", spec_kwargs=kwargs, truth=truth,
                   cost=lambda d: (d["atoms"] / 32.0) ** 2, best=best, budget=10,
                   max_evals=60)


def problems() -> dict[str, Problem]:
    return {
        "basins": _basins(),
        "wiggly-mid": _atoms_problem(
            "wiggly-mid", lambda x, a: 40.0 / a * math.cos(9 * x), upper=512,
            methods=("target", "mf-lt", "mf-prod")),
        "wiggly": _atoms_problem("wiggly", lambda x, a: 40.0 / a * math.cos(9 * x)),
        "smooth": _atoms_problem("smooth", lambda x, a: 40.0 / a * (1 + 2 * x)),
        "noisy": _atoms_problem(
            "noisy", lambda x, a: 40.0 / a * (1 + 2 * x),
            noise=lambda d: 0.15 * math.sqrt(32.0 / d["atoms"]),  # small systems noisier
            methods=("target", "mf-lt", "mf-prod", "mf-yvar-lt", "mf-yvar-prod")),
        "hartmann": _hartmann(),
        "coded": _coded(),
    }


# -- methods --------------------------------------------------------------------------


class MultiFidelityKG:
    """BoTorch's multi-fidelity knowledge gradient, as in its tutorial: the current
    value is the best posterior mean at the target, the cost is abbot's learned
    ``SourceCost`` (relative to a target run), and candidates are projected to the
    target."""

    uq_requirement = "calibrated_variance"

    def __init__(self, num_fantasies: int = 32) -> None:
        self.num_fantasies = num_fantasies
        self.last_acquisition_value = None
        self._cost = None

    def suggest(self, history, spec, model, n=1, *, pending=None):
        from botorch.acquisition import PosteriorMean, qMultiFidelityKnowledgeGradient
        from botorch.acquisition.cost_aware import InverseCostWeightedUtility
        from botorch.models.deterministic import GenericDeterministicModel
        from botorch.optim import optimize_acqf

        from abbot.acq_opt import Continuous, optimize
        from abbot.cost import SourceCost

        if self._cost is None:
            self._cost = SourceCost(spec)
        cost = self._cost
        cost.fit(history, spec)
        bounds = spec.bounds_tensor()
        best_at_target = optimize(Continuous(num_restarts=4, raw_samples=128),
                                  PosteriorMean(model), spec, 1, spec.target_features())
        with torch.no_grad():
            current = model.posterior(best_at_target).mean.max()
        scale = cost(spec.at_target(spec.sample_designs(256))).mean()
        acq = qMultiFidelityKnowledgeGradient(
            model, num_fantasies=self.num_fantasies, current_value=current,
            cost_aware_utility=InverseCostWeightedUtility(
                GenericDeterministicModel(lambda X: (cost(X) / scale).clamp_min(1e-12).unsqueeze(-1))
            ),
            project=spec.at_target,
        )
        x, value = optimize_acqf(acq, bounds, q=1, num_restarts=4, raw_samples=128,
                                 options={"batch_limit": 4, "maxiter": 100})
        x = x.detach()
        for j, d in enumerate(spec.inputs):
            if d.kind == "integer":
                x[..., j] = x[..., j].round()
        self.last_acquisition_value = float(value)
        return x

    def ingest(self, results):
        return None


def _accuracy(source, value: float) -> float:
    """Where ``value`` sits between the source's cheap end (0) and accurate end (1),
    on its model scale: Xopt's ``s``."""
    d = source.domain
    lo, hi, v = float(d.lower), float(d.upper), float(value)
    if source.log:
        lo, hi, v = math.log(lo), math.log(hi), math.log(v)
    u = (v - lo) / (hi - lo)
    return u if source.accurate == "high" else 1 - u


def _from_accuracy(source, u: float) -> float:
    d = source.domain
    lo, hi = float(d.lower), float(d.upper)
    u = u if source.accurate == "high" else 1 - u
    v = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))) if source.log else lo + u * (hi - lo)
    return float(round(v)) if d.kind == "integer" else v


def _xopt_campaign(problem: Problem):
    """The source as an input s in [0, 1] that is also maximized, as Xopt does."""
    from abbot import (
        CallableCost,
        Campaign,
        ContinuousDomain,
        OutputSpec,
        ParetoBO,
        ProblemSpec,
    )

    base = problem.spec()
    source = problem.source
    spec = ProblemSpec(
        domain=[*base.domain, ContinuousDomain(name="s", lower=0.0, upper=1.0)],
        outputs=[*base.outputs, OutputSpec(name="s_out", direction="MAXIMIZE")],
    )

    def cost_of_s(X):
        return torch.stack([
            torch.as_tensor(problem.cost({source.name: _from_accuracy(source, float(s))}),
                            dtype=X.dtype)
            for s in X[..., -1].reshape(-1)
        ]).reshape(X.shape[:-1])

    # Xopt's default reference point: -100 for a maximized objective (+100 raw for a
    # minimized one, i.e. -100 canonical), 0 for s.
    policy = ParetoBO(ref_point=[-100.0, 0.0], cost_model=CallableCost(cost_of_s))
    return spec, Campaign(spec, policy=policy)


def _recommend_xopt(problem, spec, campaign):
    """Xopt's ``get_optimum``: the best posterior mean at s = 1, over evaluated designs."""
    model = campaign.current_model()
    X = campaign.history.clean.designs.double().clone()
    u_target = _accuracy(problem.source, problem.source.target)
    X[:, -1] = u_target
    with torch.no_grad():
        mean = model.posterior(X).mean[..., 0]  # canonical objective column
    x = X[int(mean.argmax())]
    return dict(zip(spec.design_names[:-1], x[:-1].tolist()))


def run_one(problem_name: str, method: str, seed: int) -> dict:
    warnings.filterwarnings("ignore")
    torch.set_num_threads(2)
    from abbot import (
        Campaign,
        MultiFidelityGPConstructor,
        ProblemSpec,
        SimpleBO,
        default_initial_designs,
    )

    problem = problems()[problem_name]
    spec = problem.spec()
    source = problem.source
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(10_000 + seed)
    xopt = method == "xopt"

    def observe(design):
        y = problem.truth(design)
        var = None
        if problem.noise_sd is not None:
            sd = problem.noise_sd(design)
            y += sd * float(torch.randn(1, generator=gen, dtype=torch.double))
            var = sd * sd
        return y, var

    if method == "target":
        run_spec = ProblemSpec(**{**problem.spec_kwargs, "sources": None})
        campaign = Campaign(run_spec)
    elif xopt:
        run_spec, campaign = _xopt_campaign(problem)
    else:
        run_spec = spec
        linear_truncated = not method.endswith("prod")
        policy = MultiFidelityKG() if method == "mf-kg" else SimpleBO()
        campaign = Campaign(spec, policy=policy, model_constructor=MultiFidelityGPConstructor(
            spec, linear_truncated=linear_truncated))
    yvar = method.startswith("mf-yvar")

    def to_problem(design):
        """A campaign design as the problem's design dict (with the source)."""
        if method == "target":
            return {**design, source.name: source.target}
        if xopt:
            d = {k: v for k, v in design.items() if k != "s"}
            return {**d, source.name: _from_accuracy(source, design["s"])}
        return dict(design)

    def record(design):
        full = to_problem(design)
        y, var = observe(full)
        outputs = [y, design["s"]] if xopt else [y]
        kw = {"output_var": [var] if (yvar and var is not None) else None}
        if xopt and kw["output_var"] is not None:
            kw["output_var"] = [var, 1e-8]
        campaign.record(design, outputs, cost=problem.cost(full), **kw)
        return problem.cost(full)

    target_cost = problem.cost({source.name: source.target})

    def recommendation():
        if xopt:
            return _recommend_xopt(problem, run_spec, campaign)
        result = campaign.result()
        rec = campaign.history.records[result.recommended[0]].candidate.tolist()
        return dict(zip(run_spec.design_names, rec))

    trace, spent, start = [], 0.0, time.time()  # (spend in target runs, regret)
    seeds = default_initial_designs(run_spec, seed=seed)
    for x in seeds:
        spent += record(dict(zip(run_spec.variable_names, x.tolist())))
    trace.append((spent / target_cost, problem.regret(recommendation())))
    evals = len(seeds)
    error = None
    while spent / target_cost < problem.budget and evals < problem.max_evals:
        try:
            (cand,) = campaign.suggest(1)
        except Exception as exc:  # noqa: BLE001 - a method failing is a result
            error = f"{type(exc).__name__}: {exc}"
            break
        spent += record(dict(cand))
        evals += 1
        trace.append((spent / target_cost, problem.regret(recommendation())))
    final = trace[-1][1]
    curve = []
    for c in (c for c in CHECKPOINTS if c <= problem.budget):
        within = [r for x, r in trace if x <= c]
        if within:
            curve.append((c, within[-1]))
    sources_used = (
        [round(_from_accuracy(source, r.candidate[-1].item()), 3) for r in campaign.history.records]
        if xopt else
        [r.candidate[len(spec.design_names)].item() for r in campaign.history.records]
        if method != "target" else []
    )
    return {
        "problem": problem_name, "method": method, "seed": seed, "curve": curve,
        "trace": trace,
        "final_regret": final, "spent": spent / target_cost, "evals": evals,
        "seconds": time.time() - start, "error": error, "sources": sources_used,
    }


# -- driver ---------------------------------------------------------------------------


def summarize(rows: list[dict]) -> str:
    import statistics

    lines = []
    by = {}
    for r in rows:
        by.setdefault((r["problem"], r["method"]), []).append(r)
    for (pname, method), runs in sorted(by.items()):
        budget = problems()[pname].budget
        cells = []
        for cp in CHECKPOINTS:
            if cp > budget:
                cells.append("")
                continue
            vals = [dict(map(tuple, r["curve"])).get(cp) for r in runs]
            vals = [v for v in vals if v is not None]
            # A median only when most seeds got under this spend.
            cells.append(f"{statistics.median(vals):.3f}" if len(vals) > len(runs) / 2 else "-")
        errors = sum(1 for r in runs if r["error"])
        evals = statistics.median(r["evals"] for r in runs)
        secs = statistics.median(r["seconds"] for r in runs)
        lines.append(f"{pname:9s} {method:12s} " + " ".join(f"{c:>7s}" for c in cells)
                     + f"   evals {evals:5.0f}  {secs:6.0f}s" + (f"  errors {errors}" if errors else ""))
    header = f"{'problem':9s} {'method':12s} " + " ".join(f"{'@' + str(c):>7s}" for c in CHECKPOINTS)
    return header + "\n" + "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--problems", nargs="+", default=list(problems()))
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="multi_fidelity_results.jsonl")
    args = parser.parse_args()

    jobs = [(p, m, s) for p in args.problems for m in (args.methods or problems()[p].methods)
            if m in problems()[p].methods for s in range(args.seeds)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool, open(args.out, "a") as out:
        futures = [pool.submit(run_one, *job) for job in jobs]
        for fut in futures:
            row = fut.result()
            rows.append(row)
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"{row['problem']:9s} {row['method']:12s} seed {row['seed']}: "
                  f"regret {row['final_regret']:.3f} at {row['spent']:.1f} target runs "
                  f"({row['evals']} evals, {row['seconds']:.0f}s)"
                  + (f"  ERROR {row['error']}" if row["error"] else ""), flush=True)
    print()
    print(summarize(rows))


if __name__ == "__main__":
    main()
