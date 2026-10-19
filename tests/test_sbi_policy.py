"""The targets hook + Rescore (annealed composite SBI), TrainedPosterior, and the
resume fingerprint."""

import pytest
import torch

from abbot import (
    Box,
    Campaign,
    ContinuousDomain,
    DataStore,
    OutputSpec,
    ProblemSpec,
    Rescore,
    ResumeMismatchError,
    SimpleBO,
    TrainedPosterior,
    sbi_evaluate,
)


def _spec(lo: float = 0.4) -> ProblemSpec:
    return ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        outputs=[OutputSpec(name="F", direction="MAXIMIZE")],
        boxes=[Box(lower=[lo], upper=[0.6])],
    )


def band_score(theta, spec, beta: float = 50.0):
    """Soft fraction of theta samples inside the spec's first target box."""
    lo, hi = (torch.tensor(v, dtype=theta.dtype) for v in (spec.boxes[0].lower, spec.boxes[0].upper))
    return (torch.sigmoid(beta * (theta - lo)) * torch.sigmoid(beta * (hi - theta))).prod(-1).mean()


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


def test_rescore_rescores_cached_theta_each_step():
    spec = _spec()
    calls, schedule = _schedule_log()
    rescore = Rescore(band_score, schedule, n_initial=3)
    c = Campaign(spec, SimpleBO(), targets=rescore)
    c.seed(torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.double))
    ev = sbi_evaluate(_Sampler(), band_score, spec, n_samples=64)
    for _ in range(5):
        c.step(_Exp(), ev)
    # one schedule call per proposal; the clock starts at 0 after the seeds
    assert calls == [0, 1]
    scored = c.scored_history()  # t = 2 now
    assert calls == [0, 1, 2]
    F = scored.clean.outputs
    assert F.shape == (5, 1)
    assert float(F[1]) > float(F[0])  # x=0.5 is in the box, x=0.1 is not
    # the stored history keeps what was measured; only the view is re-scored
    assert not torch.equal(F, c.history.clean.outputs)


def test_rescore_refuses_records_without_cached_theta():
    spec = _spec()
    c = Campaign(spec, SimpleBO(), targets=Rescore(band_score))
    c.record({"x": 0.2}, [0.0])
    with pytest.raises(ValueError, match="cached payload"):
        c.scored_history()


class _FakePosterior:
    def sample(self, shape, x):
        return torch.full((*shape, 2), 0.5) + 0 * x.sum()


def test_trained_posterior_transforms_and_unscales():
    post = TrainedPosterior(
        _FakePosterior(),
        transform=lambda o: torch.log10(o["obs"]),
        theta_min=torch.tensor([0.0, 10.0]),
        theta_max=torch.tensor([2.0, 30.0]),
    )
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


def test_resume_ignores_fields_the_saved_fingerprint_lacks(tmp_path):
    # A newer version adding a spec/fingerprint key must not orphan old stores.
    store = DataStore(tmp_path)
    Campaign(_spec(), store=store).record({"x": 0.2}, [0.0])
    r = Campaign(_spec(), store=store, fingerprint={"new_key": 1})
    assert len(r.history) == 1


def test_mismatch_with_no_saved_steps_just_starts_over(tmp_path):
    store = DataStore(tmp_path)
    Campaign(_spec(), store=store)  # crashed before its first shot
    c = Campaign(_spec(lo=0.3), store=store)
    assert len(c.history) == 0

