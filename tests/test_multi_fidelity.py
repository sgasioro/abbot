"""Multi-fidelity: the source-aware model, the learned default cost, and SimpleBO
choosing sources."""

import logging

import pytest
import torch
from botorch.models import ModelListGP, MultiTaskGP, SingleTaskMultiFidelityGP
from pydantic import ValidationError

from abbot import (
    BAXPolicy,
    Campaign,
    ContinuousDomain,
    EnumeratedDomain,
    IntegerDomain,
    MultiFidelityGPConstructor,
    OutputSpec,
    ProblemSpec,
    SimpleBO,
    Source,
    SourceCost,
    TopK,
    default_constructor,
    default_initial_designs,
)
from abbot.history import History, Record
from abbot.model import _fidelity_transform
from abbot.policy import _condition, _MultiFidelityMES


def _spec(*sources, outputs=None):
    return ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        sources=list(sources),
        outputs=outputs or [OutputSpec(name="f", direction="MAXIMIZE")],
    )


def _atoms(**kw):
    return Source(domain=IntegerDomain(name="atoms", lower=32, upper=512), target=256, **kw)


def _code(target=1.0):
    return Source(domain=EnumeratedDomain(name="code", values=[0.0, 1.0]), target=target)


def _records(rows, costs=None):
    costs = costs or [None] * len(rows)
    return History(records=[
        Record(candidate=torch.tensor(x, dtype=torch.double), observation=None,
               outputs=torch.tensor(y, dtype=torch.double).reshape(-1), cost=c)
        for (x, y), c in zip(rows, costs)
    ])


# -- the spec ---------------------------------------------------------------------


def test_accurate_and_unordered_validation():
    assert _atoms().accurate == "high"
    assert _atoms(accurate="low").accurate == "low"
    with pytest.raises(ValidationError, match="must be enumerated"):
        Source(domain=IntegerDomain(name="n", lower=1, upper=4), target=2, ordered=False)
    with pytest.raises(ValidationError, match="whole numbers"):
        Source(domain=EnumeratedDomain(name="code", values=[0.5, 1.0]), target=1.0)
    with pytest.raises(ValidationError, match="ordered sources only"):
        Source(domain=EnumeratedDomain(name="code", values=[0.0, 1.0]), target=1.0,
               accurate="low")
    # An enumerated source declared ordered is a fidelity ladder; any values are fine.
    Source(domain=EnumeratedDomain(name="mesh", values=[0.5, 0.25]), target=0.25,
           ordered=True, accurate="low")


# -- the model --------------------------------------------------------------------


def test_fidelity_transform_puts_the_accurate_end_at_one():
    spec = _spec(
        _atoms(log=True),
        Source(domain=ContinuousDomain(name="tol", lower=1e-3, upper=1e-1), target=1e-2, log=True,
               accurate="low"),
    )
    tf = _fidelity_transform(spec, spec.bounds_tensor())
    X = torch.tensor([[0.0, 32.0, 1e-1], [1.0, 512.0, 1e-3], [0.5, 128.0, 1e-2]],
                     dtype=torch.double)
    Z = tf(X)
    assert torch.allclose(Z[0], torch.tensor([0.0, 0.0, 0.0], dtype=torch.double))
    assert torch.allclose(Z[1], torch.tensor([1.0, 1.0, 1.0], dtype=torch.double))
    assert torch.allclose(Z[2], torch.tensor([0.5, 0.5, 0.5], dtype=torch.double))  # log midpoints


def test_default_constructor_picks_the_multi_fidelity_model():
    assert isinstance(default_constructor(_spec(_atoms())), MultiFidelityGPConstructor)
    with pytest.raises(ValueError, match="together"):
        MultiFidelityGPConstructor(_spec(_atoms(), _code()))
    two_codes = _spec(_code(), Source(domain=EnumeratedDomain(name="lab", values=[0, 1]),
                                      target=0))
    with pytest.raises(ValueError, match="one unordered"):
        MultiFidelityGPConstructor(two_codes)
    with pytest.raises(ValueError, match="not an ordered source"):
        MultiFidelityGPConstructor(_spec(_atoms()), iteration="x")


def _fidelity_data(n=16, m=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, generator=g, dtype=torch.double)
    atoms = torch.randint(32, 513, (n, 1), generator=g).double()
    y = torch.sin(6 * x) + 4.0 / atoms
    Y = torch.cat([y, x * 2][:m], -1) + 0.01 * torch.randn(n, m, generator=g, dtype=torch.double)
    return torch.cat([x, atoms], -1), Y


@pytest.mark.parametrize("kwargs", [{}, {"linear_truncated": False, "iteration": "atoms"}])
def test_ordered_sources_build_a_fidelity_gp(kwargs):
    spec = _spec(_atoms(log=True), outputs=[OutputSpec(name="f"), OutputSpec(name="g")])
    X, Y = _fidelity_data(m=2)
    gp = MultiFidelityGPConstructor(spec, **kwargs).build(X, Y)
    assert isinstance(gp, SingleTaskMultiFidelityGP)
    assert gp.posterior(X[:5]).mean.shape == (5, 2)
    gp_var = MultiFidelityGPConstructor(spec, **kwargs).build(X, Y, torch.full_like(Y, 1e-4))
    assert gp_var.posterior(X[:5]).mean.shape == (5, 2)


def test_cheap_runs_inform_the_target():
    """With a shared shape across fidelities, low-fidelity runs on one side of the
    space improve the target prediction there over the target runs alone."""
    spec = _spec(_atoms(log=True))
    g = torch.Generator().manual_seed(1)
    x_hi = torch.linspace(0, 0.4, 5, dtype=torch.double).unsqueeze(-1)
    x_lo = torch.linspace(0.5, 1, 12, dtype=torch.double).unsqueeze(-1)
    truth = lambda x, a: torch.sin(6 * x) + 4.0 / a
    X_hi = torch.cat([x_hi, torch.full_like(x_hi, 512.0)], -1)
    X_lo = torch.cat([x_lo, torch.full_like(x_lo, 64.0)], -1)
    X = torch.cat([X_hi, X_lo])
    Y = truth(X[:, :1], X[:, 1:]) + 0.005 * torch.randn(len(X), 1, generator=g, dtype=torch.double)
    gp = MultiFidelityGPConstructor(spec).build(X, Y)
    test = torch.cat([x_lo, torch.full_like(x_lo, 512.0)], -1)
    err_mf = (gp.posterior(test).mean - truth(x_lo, 512.0)).abs().mean()
    only_target = MultiFidelityGPConstructor(spec).build(X_hi, Y[:5])
    err_hi = (only_target.posterior(test).mean - truth(x_lo, 512.0)).abs().mean()
    assert err_mf < 0.5 * err_hi


def _code_data(n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, generator=g, dtype=torch.double)
    code = torch.randint(0, 2, (n, 1), generator=g).double()
    Y = torch.sin(6 * x) + 0.3 * x * (code == 0) + 0.02 * torch.randn(n, 1, generator=g,
                                                                      dtype=torch.double)
    return torch.cat([x, code], -1), Y


def test_an_unordered_source_builds_a_multitask_gp_at_the_target():
    spec = _spec(Source(domain=EnumeratedDomain(name="code", values=[3.0, 7.0]), target=7.0))
    X, Y = _code_data()
    X[:, 1] = torch.where(X[:, 1] == 0, 3.0, 7.0)
    gp = MultiFidelityGPConstructor(spec).build(X, Y)
    assert isinstance(gp, MultiTaskGP) and gp.num_outputs == 1
    assert gp.posterior(X[:4]).mean.shape == (4, 1)
    # InfoBAX's batched, near-noise-free fantasy conditioning works.
    Xf = torch.rand(3, 2, 2, dtype=torch.double)
    Xf[..., 1] = 7.0
    cond = _condition(gp, Xf, torch.randn(3, 2, 1, dtype=torch.double), noise=1e-6)
    assert cond.posterior(Xf).variance.max() < 1e-3
    two = MultiFidelityGPConstructor(_spec(_code(), outputs=[OutputSpec(name="f"),
                                                             OutputSpec(name="g")]))
    X, Y = _code_data()
    model = two.build(X, torch.cat([Y, -Y], -1), torch.full((len(X), 2), 1e-4, dtype=torch.double))
    assert isinstance(model, ModelListGP) and model.posterior(X[:3]).mean.shape == (3, 2)


def test_linear_truncated_power_is_pinned_so_the_kernel_stays_valid():
    from botorch.models.kernels import LinearTruncatedFidelityKernel

    spec = _spec(_atoms(log=True))
    X, Y = _fidelity_data(n=24)
    gp = MultiFidelityGPConstructor(spec).build(X, Y)
    (kernel,) = [k for k in gp.covar_module.modules()
                 if isinstance(k, LinearTruncatedFidelityKernel)]
    assert kernel.power.item() == pytest.approx(1.0) and not kernel.raw_power.requires_grad
    Z = torch.rand(300, 2, dtype=torch.double)
    assert torch.linalg.eigvalsh(kernel(Z).to_dense()).min() > -1e-8


# -- the default cost -------------------------------------------------------------


def test_source_cost_learns_a_power_law_and_level_factors():
    spec = _spec(_atoms(), _code())
    rows, costs = [], []
    for atoms in (32, 64, 128, 256, 512):
        for code in (0.0, 1.0):
            rows.append(([0.5, atoms, code], [0.0]))
            costs.append(3e-4 * atoms**2 * (10.0 if code == 1.0 else 1.0))
    cost = SourceCost(spec)
    cost.fit(_records(rows[:2], costs[:2]), spec)  # too few records: stays flat
    assert torch.allclose(cost(torch.tensor([[0.1, 32.0, 0.0], [0.9, 512.0, 1.0]],
                                            dtype=torch.double)),
                          torch.ones(2, dtype=torch.double))
    cost.fit(_records(rows, costs), spec)
    assert cost.params["p_atoms"].item() == pytest.approx(2.0, abs=1e-3)
    # The factor is for the non-target level, relative to the target (code 1).
    assert cost.params["f_code_0"].exp().item() == pytest.approx(0.1, rel=1e-3)
    X = torch.tensor([[0.2, 100.0, 1.0]], dtype=torch.double)
    assert cost(X).item() == pytest.approx(3e-4 * 100**2 * 10, rel=1e-3)


def test_source_cost_is_exponential_for_a_source_that_reaches_zero():
    spec = _spec(Source(domain=ContinuousDomain(name="level", lower=0.0, upper=4.0), target=4.0))
    rows = [([0.5, v], [0.0]) for v in (0.0, 1.0, 2.0, 3.0, 4.0)]
    cost = SourceCost(spec)
    cost.fit(_records(rows, [2.0 * 3.0**v for v in (0, 1, 2, 3, 4)]), spec)
    assert cost(torch.tensor([[0.0, 2.5]], dtype=torch.double)).item() == pytest.approx(
        2.0 * 3.0**2.5, rel=1e-3)


def test_source_cost_warns_once_without_costs(caplog):
    spec = _spec(_atoms())
    cost = SourceCost(spec)
    with caplog.at_level(logging.WARNING, logger="abbot.cost"):
        cost.fit(_records([([0.5, 64.0], [0.0])]), spec)
        cost.fit(_records([([0.5, 64.0], [0.0])]), spec)
    assert sum("No run has a recorded cost" in r.message for r in caplog.records) == 1


# -- SimpleBO choosing sources ------------------------------------------------------


def _calibration(design):
    """A misfit to minimize whose cheap runs are biased, costing ~atoms**2."""
    x, atoms = design["x"], design["atoms"]
    misfit = (x - 0.3) ** 2 + 3.0 / atoms * (1 + x)
    return misfit, (atoms / 32.0) ** 2


def _run(policy, steps, seed=0, spec=None, sim=_calibration):
    torch.manual_seed(seed)
    spec = spec or _spec(_atoms(log=True), outputs=[OutputSpec(name="misfit",
                                                               direction="MINIMIZE")])
    c = Campaign(spec, policy=policy)
    for x in default_initial_designs(spec, seed=seed):
        d = dict(zip(spec.variable_names, x.tolist()))
        y, cost = sim(d)
        c.record(d, [y], cost=cost)
    for _ in range(steps):
        (d,) = c.suggest(1)
        y, cost = sim(d)
        c.record(d, [y], cost=cost)
    return c


@pytest.mark.slow
def test_simple_bo_chooses_cheap_sources_and_learns_the_cost():
    c = _run(SimpleBO(), steps=6)
    proposed = c.history.records[-6:]
    atoms = [r.candidate[1].item() for r in proposed]
    assert min(atoms) < 256  # it used cheaper runs than the target
    assert all(a == round(a) for a in atoms)  # integer source, rounded
    assert c.policy.source_cost.params["p_atoms"].item() == pytest.approx(2.0, abs=1e-3)
    assert all(r.acquisition is not None and r.acquisition > 0 for r in proposed)
    result = c.result()
    assert len(result.recommended) == 1 and len(result.run_at_target) == 1
    # A batch is built pick by pick, each accounting for the ones before.
    batch = c.suggest(2)
    assert len(batch) == 2 and (batch[0]["x"], batch[0]["atoms"]) != (batch[1]["x"], batch[1]["atoms"])


@pytest.mark.parametrize("linear_truncated", [True, False])
def test_entropy_search_handles_runs_at_a_log_sources_lower_bound(linear_truncated):
    """BoTorch evaluates the model's transformed training inputs as raw designs; a
    32-atom run (transformed to 0) then hits log(0). The wrapper avoids that."""
    from botorch.acquisition.cost_aware import InverseCostWeightedUtility
    from botorch.models.deterministic import GenericDeterministicModel

    spec = _spec(_atoms(log=True))
    X, Y = _fidelity_data(n=12)
    X[:3, 1] = 32.0
    gp = MultiFidelityGPConstructor(spec, linear_truncated=linear_truncated).build(X, Y)
    candidates = spec.at_target(spec.sample_designs(200, seed=0))
    acq = _MultiFidelityMES(
        gp, candidates, train_inputs=spec.at_target(X),
        cost_aware_utility=InverseCostWeightedUtility(
            GenericDeterministicModel(lambda Z: Z[..., 1:2] / 32.0)),
        project=spec.at_target,
    )
    assert torch.isfinite(acq(X.unsqueeze(1))).all()


@pytest.mark.slow
def test_choose_sources_off_keeps_proposals_at_the_target():
    c = _run(SimpleBO(choose_sources=False), steps=3)
    assert all(r.candidate[1].item() == 256.0 for r in c.history.records[-3:])


@pytest.mark.slow
def test_an_explicit_cost_model_overrides_the_learned_one():
    from abbot import CallableCost

    # Claim the cheap end is expensive: it should then stay near the top.
    flipped = CallableCost(lambda X: (544.0 - X[..., 1]) ** 2)
    c = _run(SimpleBO(cost_model=flipped), steps=4)
    assert c.policy._source_cost is None and c.policy.source_cost is flipped
    assert all(r.candidate[1].item() > 256 for r in c.history.records[-4:])


@pytest.mark.slow
def test_simple_bo_chooses_an_unordered_source_with_an_observe_output():
    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        sources=[_code(target=1.0)],
        outputs=[OutputSpec(name="f", direction="MAXIMIZE"),
                 OutputSpec(name="aux", direction="OBSERVE")],
    )

    def sim(d):
        x, code = d["x"], d["code"]
        f = -((x - 0.6) ** 2) + (0.05 * x if code == 0 else 0.0)
        return [f, x], (1.0 if code == 0 else 50.0)

    torch.manual_seed(0)
    c = Campaign(spec)
    for x in (0.1, 0.4, 0.7, 0.9):
        for code in (0.0, 1.0):
            y, cost = sim({"x": x, "code": code})
            c.record({"x": x, "code": code}, y, cost=cost)
    picks = c.suggest(2)
    assert any(p["code"] == 0.0 for p in picks)
    assert all(p["code"] in (0.0, 1.0) for p in picks)


@pytest.mark.slow
def test_infobax_can_stay_at_the_target():
    spec = _spec(_atoms(log=True), outputs=[OutputSpec(name="y", direction="OBSERVE")])
    grid = torch.linspace(0, 1, 21, dtype=torch.double).reshape(-1, 1)
    c = Campaign(spec, policy=BAXPolicy(TopK(k=2, output=0), strategy="InfoBAX",
                                        candidates=grid, choose_sources=False))
    torch.manual_seed(0)
    for x, a in [(0.0, 32), (0.3, 64), (0.5, 512), (0.8, 256), (1.0, 128), (0.6, 256)]:
        c.record({"x": x, "atoms": a}, [torch.sin(torch.tensor(6 * x)).item() + 1.0 / a])
    picks = c.suggest(2)
    assert all(p["atoms"] == 256.0 for p in picks)
    assert not c.last_warning


# -- BAX choosing sources -------------------------------------------------------------


def test_bax_source_levels_and_proposal_set():
    spec = _spec(_atoms(log=True), outputs=[OutputSpec(name="y", direction="OBSERVE")])
    grid = torch.linspace(0, 1, 11, dtype=torch.double).reshape(-1, 1)
    policy = BAXPolicy(TopK(k=2, output=0), strategy="InfoBAX", candidates=grid)
    assert policy._levels(spec) == [[256.0, 32.0, 64.0, 128.0, 512.0]]
    proposals = policy.proposal_set(spec)
    assert proposals.shape == (55, 2) and torch.equal(proposals[:11], policy.candidate_grid(spec))
    assert BAXPolicy(TopK(k=2, output=0), strategy="InfoBAX", candidates=grid,
                     source_levels={"atoms": [32, 100]})._levels(spec) == [[256.0, 32.0, 100.0]]
    # Strategies that don't choose sources, or choose_sources=False, keep the grid.
    for kw in ({"strategy": "MeanBAX"}, {"strategy": "US"},
               {"strategy": "InfoBAX", "choose_sources": False}):
        p = BAXPolicy(TopK(k=2, output=0), candidates=grid, **kw)
        assert p.proposal_set(spec).shape == (11, 2)
    with pytest.raises(ValueError, match="not sources"):
        BAXPolicy(TopK(k=2, output=0), candidates=grid, source_levels={"n": [1]})._levels(spec)
    with pytest.raises(ValueError, match="at least 2"):
        BAXPolicy(TopK(k=2, output=0), candidates=grid, source_levels=1)._levels(spec)
    unordered = _spec(_code(), outputs=[OutputSpec(name="y", direction="OBSERVE")])
    assert policy._levels(unordered) == [[1.0, 0.0]]


def _bax_campaign(strategy, steps, seed=0):
    from abbot.goal import BandIntersection

    spec = _spec(_atoms(log=True), outputs=[OutputSpec(name="y", direction="OBSERVE")])
    grid = torch.linspace(0, 1, 101, dtype=torch.double).reshape(-1, 1)

    def sim(d):
        x, atoms = d["x"], d["atoms"]
        return torch.sin(torch.tensor(6 * x)).item() + 40.0 / atoms * (1 + 2 * x), (atoms / 32) ** 2

    torch.manual_seed(seed)
    policy = BAXPolicy(BandIntersection([(0.5, 1.2)]), strategy=strategy, candidates=grid,
                       n_posterior_samples=10)
    c = Campaign(spec, policy=policy)
    for x in default_initial_designs(spec, seed=seed):
        d = dict(zip(spec.variable_names, x.tolist()))
        y, cost = sim(d)
        c.record(d, [y], cost=cost)
    for _ in range(steps):
        (d,) = c.suggest(1)
        y, cost = sim(d)
        c.record(d, [y], cost=cost)
    return c


@pytest.mark.slow
def test_infobax_chooses_cheap_sources_by_information_per_cost():
    c = _bax_campaign("InfoBAX", steps=4)
    proposed = c.history.records[-4:]
    assert all(r.candidate[1].item() < 256 for r in proposed)
    assert all(r.acquisition is not None and r.acquisition > 0 for r in proposed)
    assert c.policy._source_cost.params["p_atoms"].item() == pytest.approx(2.0, abs=1e-3)
    batch = c.suggest(2)
    assert len(batch) == 2
    assert (batch[0]["x"], batch[0]["atoms"]) != (batch[1]["x"], batch[1]["atoms"])
    assert torch.isfinite(c.policy.last_acquisition_value).all()


@pytest.mark.slow
def test_switchbax_proposes_at_the_target_until_it_falls_back_to_infobax():
    c = _bax_campaign("SwitchBAX", steps=3)
    for r in c.history.records[-3:]:
        assert r.candidate[1].item() == 256.0 or c.policy.switched


# -- experiments reporting cost; the analytic mock with sources -----------------------


def test_an_experiments_own_cost_replaces_the_measured_time():
    from abbot import TensorExperiment

    spec = _spec(_atoms())

    class _Sim:
        def run(self, design):
            if design["x"] > 0.9:
                raise RuntimeError("diverged")
            return [design["x"]]

        def cost(self, design):
            return None if design["atoms"] == 32 else design["atoms"] / 10.0

    c = Campaign(spec, max_retries=5)
    c.seed(torch.tensor([[0.1, 64.0], [0.2, 32.0], [0.95, 128.0]], dtype=torch.double))
    c.step(_Sim(), lambda y: y, n=3)
    by_design = {tuple(r.candidate.tolist()): r for r in c.history.records}
    assert by_design[(0.1, 64.0)].cost == 6.4
    measured = by_design[(0.2, 32.0)].cost
    assert measured is not None and measured < 1.0  # wall-clock seconds
    failed = by_design[(0.95, 128.0)]
    assert failed.outputs is None and failed.cost == 12.8  # failed runs are paid for too

    class _Vector:
        def run(self, x):
            return [float(x[0])]

        def cost(self, x):
            return float(x[1])

    wrapped = TensorExperiment(_Vector())
    assert wrapped.cost({"x": 0.5, "atoms": 64.0}) == 64.0


def test_mock_sources_bias_cheap_runs_and_report_cost():
    from abbot import MockAnalyticExperiment

    spec = _spec(_atoms(log=True))
    mock = MockAnalyticExperiment(spec, seed=1, source_bias=0.5)
    plain = MockAnalyticExperiment(_spec(), seed=1)
    design = {"x": 0.4}
    accurate = mock.run({**design, "atoms": 512}).item()
    assert accurate == pytest.approx(plain.run(design).item())  # no bias at the accurate end
    assert abs(mock.run({**design, "atoms": 32}).item() - accurate) > 1.0
    assert mock.cost({**design, "atoms": 32}) == pytest.approx(1.0)
    assert mock.cost({**design, "atoms": 512}) == pytest.approx(100.0)
    assert mock.cost({**design, "atoms": 128}) == pytest.approx(10.0)  # log midpoint
    assert plain.cost(design) is None
    tol = _spec(Source(domain=ContinuousDomain(name="tol", lower=0.0, upper=1.0), target=0.1,
                       accurate="low"))
    flipped = MockAnalyticExperiment(tol, seed=1, cost_ratio=10.0)
    assert flipped.cost({**design, "tol": 0.0}) == pytest.approx(10.0)
    coded = MockAnalyticExperiment(_spec(_code(target=1.0)), seed=1)
    assert coded.cost({**design, "code": 1.0}) == pytest.approx(100.0)
    assert coded.run({**design, "code": 1.0}).item() == pytest.approx(plain.run(design).item())
    assert coded.run({**design, "code": 0.0}).item() != pytest.approx(plain.run(design).item())


@pytest.mark.slow
def test_a_mock_driven_campaign_learns_the_cost_through_step():
    from abbot import MockAnalyticExperiment

    torch.manual_seed(0)
    spec = _spec(_atoms(log=True))
    mock = MockAnalyticExperiment(spec, seed=1)
    c = Campaign(spec)
    c.seed(default_initial_designs(spec, seed=0))
    for _ in range(9):
        c.step(mock, lambda y: y)
    assert all(r.cost == pytest.approx(mock.cost(dict(zip(spec.variable_names, r.candidate.tolist()))))
               for r in c.history.records)
    # cost = (atoms / 32) ** (log 100 / log 16)
    assert c.policy._source_cost.params["p_atoms"].item() == pytest.approx(1.661, abs=1e-3)
    assert min(r.candidate[1].item() for r in c.history.records[7:]) < 256
