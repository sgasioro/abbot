"""RescoredBO (annealed composite SBI), TrainedPosterior, and the resume fingerprint."""

import json
import pickle

import pytest
import torch

from abbot import (
    Box,
    Campaign,
    ContinuousDomain,
    DataStore,
    OutputSpec,
    ProblemSpec,
    RescoredBO,
    ResumeMismatchError,
    TrainedPosterior,
    sbi_evaluate,
)
from abbot.reduce import BoxMembershipReducer


def _spec(lo: float = 0.4) -> ProblemSpec:
    return ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        outputs=[OutputSpec(name="F", direction="MAXIMIZE")],
        boxes=[Box(lower=[lo], upper=[0.6])],
    )


class _Sampler:
    """theta | x ~ N(x, 0.05^2): the target box is hit near x = 0.5."""

    def sample(self, observation, n):
        g = torch.Generator().manual_seed(0)
        return observation + 0.05 * torch.randn(n, 1, generator=g, dtype=torch.double)


class _Exp:
    def run(self, design):
        return torch.tensor([design["x"]], dtype=torch.double)


def _schedule_log():
    calls = []

    def schedule(t):
        calls.append(t)
        return {"beta": 2.0 * (t + 1)}

    return calls, schedule


def test_rescored_bo_reruns_the_reducer_on_cached_theta_each_step():
    spec = _spec()
    calls, schedule = _schedule_log()
    pol = RescoredBO(BoxMembershipReducer("indicator"), schedule, n_initial=3)
    c = Campaign(spec, pol)
    c.seed(torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.double))
    ev = sbi_evaluate(_Sampler(), BoxMembershipReducer("indicator"), spec, n_samples=64)
    for _ in range(5):
        c.step(_Exp(), ev)
    # one schedule call per proposal; the clock starts at 0 after the seeds
    assert calls == [0, 1]
    assert pol.last_model is not None
    X, F = pol.targets(c.history, spec)
    assert X.shape == (5, 1) and F.shape == (5, 1)
    assert float(F[1]) > float(F[0])  # x=0.5 is in the box, x=0.1 is not


def test_rescored_bo_refuses_records_without_cached_theta():
    spec = _spec()
    c = Campaign(spec, RescoredBO(BoxMembershipReducer()))
    c.record({"x": 0.2}, [0.0])
    with pytest.raises(ValueError, match="cached posterior samples"):
        c.policy.targets(c.history, spec)


class _FakePosterior:
    def sample(self, shape, x):
        return torch.full((*shape, 2), 0.5) + 0 * x.sum()


def test_trained_posterior_loads_and_unscales(tmp_path):
    (tmp_path / "posterior.pkl").write_bytes(pickle.dumps(_FakePosterior()))
    (tmp_path / "y_scaler.json").write_text(json.dumps({"y_min": [0, 10], "y_max": [2, 30]}))
    post = TrainedPosterior.from_dir(tmp_path, transform=lambda o: torch.log10(o["obs"]))
    theta = post.sample({"obs": torch.ones(4)}, 3)
    assert theta.shape == (3, 2)
    assert theta[0].tolist() == [1.0, 20.0]


def test_resume_refuses_a_different_spec(tmp_path):
    store = DataStore(tmp_path)
    c = Campaign(_spec(), store=store, fingerprint={"seed": 1})
    c.record({"x": 0.2}, [0.0])
    Campaign(_spec(), store=store, fingerprint={"seed": 1})  # same problem: fine
    with pytest.raises(ResumeMismatchError, match=r"spec\.boxes.*0\.4 -> 0\.3"):
        Campaign(_spec(lo=0.3), store=store, fingerprint={"seed": 1})
    with pytest.raises(ResumeMismatchError, match=r"extra\.seed: 1 -> 2"):
        Campaign(_spec(), store=store, fingerprint={"seed": 2})
    r = Campaign(_spec(), store=store, fingerprint={"seed": 2}, allow_resume_mismatch=True)
    assert len(r.history) == 1
    Campaign(_spec(), store=store, fingerprint={"seed": 2})  # the override was recorded
