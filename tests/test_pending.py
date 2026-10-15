"""Policies see designs in flight (``pending=``) and steer away from duplicating them."""

import logging

import pytest
import torch

from abbot import (
    BAXPolicy,
    Campaign,
    ContinuousDomain,
    OutputSpec,
    ParetoBO,
    ProblemSpec,
    SimpleBO,
    UncertaintySampling,
)
from abbot.goal import BandIntersection, PercentileSet
from abbot.history import History, Record
from abbot.model import (
    EnsembleConstructor,
    ModelListConstructor,
    SingleTaskGPConstructor,
)
from abbot.policy import condition_on_pending


def _spec(n_outputs: int = 1, direction: str = "MAXIMIZE") -> ProblemSpec:
    return ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        outputs=[OutputSpec(name=f"y{i}", direction=direction) for i in range(n_outputs)],
    )


def _data(n: int = 6, m: int = 1):
    torch.manual_seed(0)
    X = torch.linspace(0.05, 0.95, n, dtype=torch.double).reshape(-1, 1)
    Y = torch.cat([torch.sin(6 * X + i) for i in range(m)], dim=-1)
    return X, Y


def _history(X, Y) -> History:
    return History(records=[Record(candidate=x, observation=None, outputs=y) for x, y in zip(X, Y)])


# --- kriging believer --------------------------------------------------------


@pytest.mark.parametrize(
    "ctor, m, yvar",
    [
        (SingleTaskGPConstructor(), 1, False),
        (SingleTaskGPConstructor(), 1, True),  # fixed-noise likelihood
        (ModelListConstructor(), 2, False),
        (ModelListConstructor(), 2, True),
    ],
    ids=["gp", "gp-yvar", "list", "list-yvar"],
)
def test_condition_on_pending_keeps_mean_and_shrinks_variance(ctor, m, yvar):
    X, Y = _data(m=m)
    model = ctor.build(X, Y, torch.full_like(Y, 1e-3) if yvar else None)
    P = torch.tensor([[0.5]], dtype=torch.double)
    cond = condition_on_pending(model, P)
    before, after = model.posterior(P), cond.posterior(P)
    assert torch.allclose(before.mean, after.mean, atol=1e-6)
    assert (after.variance < 0.8 * before.variance).all()


def test_condition_on_pending_degrades_for_models_that_cannot(caplog):
    X, Y = _data()
    model = EnsembleConstructor().build(X, Y, None)
    with caplog.at_level(logging.INFO, logger="abbot.policy"):
        assert condition_on_pending(model, torch.tensor([[0.5]], dtype=torch.double)) is model
    assert "can't condition" in caplog.text


def test_nothing_pending_is_a_no_op():
    X, Y = _data()
    model = SingleTaskGPConstructor().build(X, Y, None)
    assert condition_on_pending(model, None) is model
    assert condition_on_pending(model, torch.empty(0, 1, dtype=torch.double)) is model


# --- policies ----------------------------------------------------------------


def _fit(m: int = 1, n: int = 6):
    X, Y = _data(n=n, m=m)
    ctor = SingleTaskGPConstructor() if m == 1 else ModelListConstructor()
    return ctor.build(X, Y, None), _history(X, Y)


@pytest.mark.slow
@pytest.mark.parametrize("policy", [SimpleBO(), UncertaintySampling()], ids=["bo", "us"])
def test_policy_moves_away_from_a_pending_design(policy):
    torch.manual_seed(0)
    spec = _spec()
    # Few points: with a well-resolved peak, EI's next-best spot is legitimately
    # right beside the pending one.
    model, h = _fit(n=4)
    first = policy.suggest(h, spec, model, n=1)
    torch.manual_seed(0)
    again = policy.suggest(h, spec, model, n=1)
    torch.manual_seed(0)
    second = policy.suggest(h, spec, model, n=1, pending=first)
    assert abs(float(again - first)) < 0.02  # deterministic without pending
    assert abs(float(second - first)) > 0.05


@pytest.mark.slow
def test_pareto_bo_accepts_pending():
    torch.manual_seed(0)
    spec = _spec(n_outputs=2)
    model, h = _fit(m=2)
    pol = ParetoBO(min_clean=2)
    first = pol.suggest(h, spec, model, n=1)
    second = pol.suggest(h, spec, model, n=1, pending=first)
    assert second.shape == (1, 1)


@pytest.mark.slow
@pytest.mark.parametrize("strategy", ["US", "InfoBAX"])
def test_bax_excludes_and_conditions_on_pending(strategy):
    torch.manual_seed(0)
    spec = _spec(direction="OBSERVE")
    model, h = _fit()
    pol = BAXPolicy(PercentileSet(70.0), strategy=strategy, n_candidates=128,
                    n_posterior_samples=8)
    first = pol.suggest(h, spec, model, n=1)
    second = pol.suggest(h, spec, model, n=1, pending=first)
    assert not torch.equal(first, second)
    if strategy == "US":  # conditioning, not just exclusion: moves well away
        assert abs(float(second - first)) > 0.05


@pytest.mark.slow
def test_infobax_fantasies_work_on_a_model_list(caplog):
    """Regression: ModelListGP rejected the fantasy call, so every sample fell back
    and EIG was identically zero."""
    torch.manual_seed(0)
    spec = _spec(n_outputs=2, direction="OBSERVE")
    model, _ = _fit(m=2)
    pol = BAXPolicy(BandIntersection([(0.0, 2.0), (-2.0, 0.5)]), strategy="InfoBAX",
                    n_candidates=64, n_posterior_samples=4)
    grid = pol.candidate_grid(spec)
    with caplog.at_level(logging.ERROR, logger="abbot.policy"):
        eig = pol._acquisition(spec, model, grid)
    assert "fantasy conditioning failed" not in caplog.text
    assert eig.abs().max() > 0


# --- campaign wiring ---------------------------------------------------------


class _Spy:
    uq_requirement = "point"

    def __init__(self):
        self.pending: list = []

    def suggest(self, history, spec, model, n=1, *, pending=None):
        self.pending.append(None if pending is None else pending.clone())
        return torch.full((n, 1), 0.1 * (len(self.pending)), dtype=torch.double)

    def ingest(self, results):
        return None


class _Legacy(_Spy):
    def suggest(self, history, spec, model, n=1):
        return super().suggest(history, spec, model, n)


def _with_history(policy, seeds=None) -> Campaign:
    c = Campaign(_spec(), policy)
    if seeds is not None:
        c.seed(torch.tensor(seeds, dtype=torch.double))  # before any record
    c.record({"x": 0.2}, [0.3])
    c.record({"x": 0.8}, [0.1])
    return c


def test_campaign_passes_everything_in_flight():
    spy = _Spy()
    c = _with_history(spy, seeds=[[0.5]])
    c.suggest(1)  # the seed
    assert spy.pending == []  # no proposal needed
    c.suggest(1)  # proposal; the seed is in flight
    assert spy.pending[-1].tolist() == [[0.5]]
    c.fail(c.outstanding()[0].id, "spilled")  # queues a retry of 0.5
    c.suggest(2)  # the retry is issued alongside one proposal
    assert sorted(spy.pending[-1].reshape(-1).tolist()) == [0.1, 0.5]


def test_legacy_policy_without_pending_still_works(caplog):
    legacy = _Legacy()
    c = _with_history(legacy)
    c.suggest(1)
    with caplog.at_level(logging.WARNING, logger="abbot.session"):
        c.suggest(1)
    assert "takes no pending" in caplog.text
    assert len(c.outstanding()) == 2
