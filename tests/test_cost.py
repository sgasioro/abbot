"""Cost: per-record cost, cost models, cost-aware BO, and stopping on cost or convergence."""

import pytest
import torch

from abbot import (
    CallableCost,
    Campaign,
    ContinuousDomain,
    CostModel,
    FixedCost,
    OutputSpec,
    PowerLawCost,
    ProblemSpec,
    SimpleBO,
    UncertaintySampling,
    drive,
)
from abbot.experiment import Verdict
from abbot.history import DataStore, History, Record
from abbot.policy import _acq_value, _CostWeighted
from abbot.supervision import acquisition_monitor, any_of, cost_cap


def _spec(names=("x",), lower=0.0, upper=1.0):
    return ProblemSpec(
        domain=[ContinuousDomain(name=n, lower=lower, upper=upper) for n in names],
        outputs=[OutputSpec(name="f", direction="MAXIMIZE")],
    )


def _identity(obs):
    return obs


class _Quadratic:
    def run(self, design):
        return torch.tensor([-(design["x"] - 0.3) ** 2])


class _Failing:
    def run(self, design):
        raise RuntimeError("solver diverged")


class _Plate:
    def run_batch(self, designs):
        return [torch.tensor([d["x"]]) for d in designs]


def _seeded(n=3, **kw):
    c = Campaign(_spec(), **kw)
    c.seed(torch.linspace(0.1, 0.9, n, dtype=torch.double).reshape(-1, 1))
    return c


# -- cost models --------------------------------------------------------------


def test_fixed_and_callable_costs_have_the_design_batch_shape():
    X = torch.rand(4, 3, 2, dtype=torch.double)
    assert FixedCost(2.0)(X).shape == (4, 3)
    assert torch.all(FixedCost(2.0)(X) == 2.0)
    c = CallableCost(lambda X: 1 + X[..., 0])
    assert torch.allclose(c(X), 1 + X[..., 0])
    assert isinstance(FixedCost(), CostModel) and isinstance(c, CostModel)
    with pytest.raises(ValueError, match="positive"):
        FixedCost(0.0)


def _history(spec, xs, costs):
    return History(records=[
        Record(candidate=torch.as_tensor(x, dtype=torch.double), observation=None, cost=c)
        for x, c in zip(xs, costs)
    ])


def test_power_law_recovers_exponent_and_prefactor():
    spec = _spec(("atoms", "temp"), lower=1.0, upper=500.0)
    n = torch.tensor([8.0, 16.0, 32.0, 64.0, 128.0], dtype=torch.double)
    xs = torch.stack([n, torch.full_like(n, 300.0)], -1)
    costs = (2.0 * n**3).tolist()
    model = PowerLawCost(["atoms"])
    model.fit(_history(spec, xs, costs), spec)
    assert float(model.exponents[0]) == pytest.approx(3.0, abs=1e-6)
    assert float(model.log_a.exp()) == pytest.approx(2.0, rel=1e-6)
    assert float(model(torch.tensor([[256.0, 300.0]], dtype=torch.double))) == pytest.approx(2.0 * 256**3, rel=1e-6)


def test_power_law_with_fixed_exponents_fits_only_the_prefactor():
    spec = _spec(("atoms",), lower=1.0, upper=500.0)
    model = PowerLawCost(["atoms"], exponents=[3.0])
    model.fit(_history(spec, [[10.0]], [5000.0]), spec)
    assert float(model.exponents[0]) == 3.0
    assert float(model.log_a.exp()) == pytest.approx(5.0)


def test_power_law_predicts_initial_until_it_has_costs_and_checks_names():
    spec = _spec(("atoms",), lower=1.0, upper=500.0)
    model = PowerLawCost(["atoms"], initial=7.0)
    with pytest.raises(RuntimeError, match="fit"):
        model(torch.ones(1, 1, dtype=torch.double))
    model.fit(History(), spec)
    assert float(model(torch.tensor([[50.0]], dtype=torch.double))) == pytest.approx(7.0)
    with pytest.raises(ValueError, match="not in the spec"):
        PowerLawCost(["mesh"]).fit(History(), spec)


# -- costs on records -----------------------------------------------------------


def test_step_records_wall_clock_seconds_for_runs_and_failures():
    c = _seeded()
    c.step(_Quadratic(), _identity)
    assert c.history.records[0].cost is not None and c.history.records[0].cost >= 0
    c.step(_Failing(), _identity)
    failed = c.history.records[-1]
    assert failed.fault and failed.cost is not None
    assert c.state().cost_spent == pytest.approx(c.history.cost_spent)


def test_a_batch_splits_its_time_evenly():
    c = _seeded()
    c.step(_Plate(), _identity, n=3)
    costs = [r.cost for r in c.history.records]
    assert len(set(costs)) == 1 and costs[0] >= 0


def test_ingest_record_and_fail_take_caller_costs():
    c = _seeded()
    a, b, d = c.suggest(3)
    c.ingest([[0.1], [0.2]], ids=[a.id, b.id], costs=[3.0, 4.0])
    c.fail(d.id, "lost sample", cost=1.5)
    c.record({"x": 0.5}, [0.0], cost=2.5)
    assert [r.cost for r in c.history.records] == [3.0, 4.0, 1.5, 2.5]
    state = c.state()
    assert state.cost_spent == pytest.approx(11.0) and state.costs == [3.0, 4.0, 1.5, 2.5]
    with pytest.raises(ValueError, match="cost"):
        e, f = c.suggest(2)
        c.ingest([[0.0], [0.0]], ids=[e.id, f.id], costs=[1.0])


def test_records_pickled_before_cost_existed_read_as_none():
    rec = Record(candidate=torch.zeros(1), observation=None)
    del rec.__dict__["cost"], rec.__dict__["acquisition"]  # what an old pickle restores
    assert rec.cost is None and rec.acquisition is None
    assert History(records=[rec]).cost_spent == 0.0


# -- stopping -------------------------------------------------------------------


def test_any_of_merges_stop():
    stop = any_of(lambda h: Verdict(), lambda h: Verdict(stop="done"), lambda h: Verdict(stop="later"))
    assert stop(History()).stop == "done"


def test_cost_cap_ends_drive_normally():
    c = _seeded(n=6, supervisor=cost_cap(10.0))
    calls = []

    class _Priced:
        def run(self, design):
            calls.append(design)
            return torch.tensor([0.0])

    # Ask/tell costs are explicit; drive's are wall-clock, so price via record.
    for x in (0.1, 0.2, 0.3):
        c.record({"x": x}, [0.0], cost=4.0)
    assert c.stopped and "cost budget" in c.stopped
    drive(c, _Priced(), _identity, n_total=20)
    assert calls == [] and not c.halted
    assert c.state().stopped == c.stopped


@pytest.mark.slow
def test_acquisition_values_reach_records_and_the_monitor_stops():
    torch.manual_seed(0)
    c = _seeded(policy=UncertaintySampling(), supervisor=acquisition_monitor(1e9, window=2))
    drive(c, _Quadratic(), _identity, n_total=20)
    seeds, proposed = c.history.records[:3], c.history.records[3:]
    assert all(r.acquisition is None for r in seeds)
    assert len(proposed) == 2 and all(r.acquisition is not None for r in proposed)
    assert c.stopped and "converged" in c.stopped


@pytest.mark.slow
def test_acquisition_values_survive_a_resume(tmp_path):
    torch.manual_seed(0)
    c = _seeded(policy=UncertaintySampling(), store=DataStore(tmp_path))
    for cand in c.suggest(3):
        c.ingest([0.0], id=cand.id)
    (pending,) = c.suggest(1)
    value = c._inflight[pending.id].acquisition
    assert value is not None
    resumed = Campaign(_spec(), policy=UncertaintySampling(), store=DataStore(tmp_path))
    assert resumed._inflight[pending.id].acquisition == pytest.approx(value)
    resumed.ingest([0.0], id=pending.id)
    assert resumed.history.records[-1].acquisition == pytest.approx(value)


# -- cost-aware BO --------------------------------------------------------------


def test_cost_weighting_subtracts_log_batch_cost():
    class _Const(torch.nn.Module):
        model = None

        def forward(self, X):
            return torch.zeros(X.shape[:-2], dtype=X.dtype)

    acq = _CostWeighted(_Const(), CallableCost(lambda X: 1 + X[..., 0]))
    X = torch.tensor([[0.5], [1.0]], dtype=torch.double)  # one batch of q=2
    assert _acq_value(acq, X) == pytest.approx(-torch.log(torch.tensor(1.5 + 2.0)).item())


@pytest.mark.slow
def test_simple_bo_with_a_cost_model_prefers_the_cheap_side():
    torch.manual_seed(0)
    spec = _spec()
    # Symmetric objective with optima at 0.25 and 0.75; running above 0.5 is 1000x dearer.
    f = lambda x: torch.sin(2 * torch.pi * x) ** 2
    cost = CallableCost(lambda X: torch.where(X[..., 0] > 0.5, 1000.0, 1.0).to(X))
    picks = {}
    for name, policy in [("plain", SimpleBO()), ("costed", SimpleBO(cost_model=cost))]:
        c = Campaign(spec, policy=policy)
        for x in (0.0, 0.1, 0.4, 0.5, 0.6, 0.9, 1.0):
            c.record({"x": x}, [float(f(torch.tensor(x)))])
        picks[name] = [cand["x"] for cand in c.suggest(4)]
        assert policy.last_acquisition_value is not None
    assert any(x > 0.5 for x in picks["plain"])  # without cost, both optima are fair game
    assert all(x <= 0.5 for x in picks["costed"])
