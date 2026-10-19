"""Composite-SBI wiring: sbi_evaluate + aux caching + Rescore, driving a Campaign.

Self-contained (no sbido/sbi): a toy analytic posterior stands in for the learned
one. The mechanism — observation -> posterior samples -> reduce -> (F, theta),
theta cached on aux, loop fits GP over (xi, F) — is identical to the real case.
"""

import pytest
import torch

from abbot.composite import Rescore
from abbot.interop.sbi import PosteriorSampler, sbi_evaluate
from abbot.model import SingleTaskGPConstructor
from abbot.policy import SimpleBO
from abbot.problem import Box, ContinuousDomain, ObjectiveSpec, ProblemSpec

pytestmark = pytest.mark.slow  # Campaign GP fits
from abbot.session import Campaign


def band_score(theta, spec, beta: float = 5.0):
    """Soft fraction of theta samples inside the spec's first target box."""
    lo, hi = (torch.tensor(v, dtype=theta.dtype) for v in (spec.boxes[0].lower, spec.boxes[0].upper))
    return (torch.sigmoid(beta * (theta - lo)) * torch.sigmoid(beta * (hi - theta))).prod(-1).mean()


class _ToyExperiment:
    """design x in [0,1] -> observation centered at theta* = 10*x (1-D theta)."""

    def run(self, design: dict) -> torch.Tensor:
        return torch.tensor([design["x"] * 10.0])  # the 'observation' is the latent mean


class _ToyPosterior(PosteriorSampler):
    """Gaussian posterior around the observed mean (analytic stand-in)."""

    def __init__(self, sigma: float = 0.5) -> None:
        self.sigma = sigma

    def sample(self, observation, n: int = 512) -> torch.Tensor:
        mean = observation.reshape(-1)
        return mean + self.sigma * torch.randn(n, mean.shape[0])


def _spec():
    # Target band on theta is [4, 6] -> best design is x ~ 0.5 (theta* = 5).
    return ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        objectives=[ObjectiveSpec(name="F", direction="MAXIMIZE")],
        boxes=[Box(lower=[4.0], upper=[6.0])],
    )


def test_sbi_evaluate_returns_f_and_theta():
    spec = _spec()
    ev = sbi_evaluate(_ToyPosterior(), band_score, spec, n_samples=256)
    obs = _ToyExperiment().run({"x": 0.5})
    out = ev(obs)
    assert isinstance(out, tuple)
    f, theta = out
    assert f.shape == (1,)
    assert theta.shape == (256, 1)
    assert 0.0 <= float(f) <= 1.0


def test_campaign_caches_theta_on_aux_and_finds_box():
    torch.manual_seed(0)
    spec = _spec()
    reducer = band_score
    ev = sbi_evaluate(_ToyPosterior(), reducer, spec, n_samples=256)
    c = Campaign(spec, SimpleBO(), SingleTaskGPConstructor(bounds=spec.bounds_tensor()))
    c.seed(torch.tensor([[0.05], [0.5], [0.95]], dtype=torch.double))
    exp = _ToyExperiment()
    for _ in range(12):
        c.step(exp, ev)

    # aux (theta samples) cached on every record.
    assert all(r.aux is not None and r.aux.shape[0] == 256 for r in c.history.records)
    # best design should sit near x=0.5 (theta* in the [4,6] band).
    rec = c.recommend()
    best_x = float(c.history.records[rec[0]].candidate)
    assert abs(best_x - 0.5) < 0.2


def test_rescore_uses_cached_theta_without_resampling():
    torch.manual_seed(0)
    spec = _spec()
    reducer = band_score
    ev = sbi_evaluate(_ToyPosterior(), reducer, spec, n_samples=128)
    c = Campaign(spec, SimpleBO(), SingleTaskGPConstructor(bounds=spec.bounds_tensor()))
    c.seed(torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.double))
    exp = _ToyExperiment()
    for _ in range(3):
        c.step(exp, ev)
    # Re-reduce with a different beta (annealing) off cached theta.
    def scores(beta):
        return Rescore(reducer, lambda t: {"beta": beta})(c.history, spec).reshape(-1).tolist()

    sharp, soft = scores(100.0), scores(1.0)
    assert len(sharp) == len(c.history.clean) == len(soft)
    # The record whose theta is most inside the box scores highest under both.
    assert sharp.index(max(sharp)) == soft.index(max(soft))
