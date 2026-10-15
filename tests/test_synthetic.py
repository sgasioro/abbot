"""SyntheticExperiment: deterministic, spec-adaptive, drives an auto-run campaign."""

import pytest
import torch

from abbot.experiment import SyntheticExperiment
from abbot.model import default_constructor
from abbot.policy import ParetoBO, SimpleBO
from abbot.problem import ContinuousDomain, ObjectiveSpec, ProblemSpec
from abbot.session import Campaign
from abbot.ui.controller import RunController, RunStatus

pytestmark = pytest.mark.slow  # auto-run campaigns + GP fits


def _spec(m=1, d=2):
    return ProblemSpec(
        domain=[ContinuousDomain(name=f"x{i}", lower=0.0, upper=1.0) for i in range(d)],
        objectives=[ObjectiveSpec(name=f"y{j}", direction="MAXIMIZE") for j in range(m)],
    )


def test_output_width_matches_objectives():
    for m in (1, 2, 3):
        spec = _spec(m=m, d=2)
        exp = SyntheticExperiment(spec, seed=0)
        out = exp.run({"x0": 0.3, "x1": 0.7})
        assert out.shape == (m,)
        assert torch.all(out > 0) and torch.all(out <= 100.0 + 1e-6)


def test_deterministic_and_noise_repeats():
    spec = _spec(m=1, d=2)
    x = {"x0": 0.4, "x1": 0.6}
    # No noise -> identical across experiments with the same seed.
    a = SyntheticExperiment(spec, seed=1).run(x)
    b = SyntheticExperiment(spec, seed=1).run(x)
    assert torch.allclose(a, b)
    # With noise -> re-measuring the SAME point repeats exactly (seeded by x).
    noisy = SyntheticExperiment(spec, seed=1, noise=5.0)
    assert torch.allclose(noisy.run(x), noisy.run(x))


def test_different_seeds_differ():
    spec = _spec(m=1, d=2)
    x = {"x0": 0.2, "x1": 0.8}
    assert not torch.allclose(
        SyntheticExperiment(spec, seed=1).run(x), SyntheticExperiment(spec, seed=2).run(x)
    )


def test_auto_run_with_synthetic_experiment():
    torch.manual_seed(0)
    spec = _spec(m=1, d=2)
    exp = SyntheticExperiment(spec, seed=3)
    campaign = Campaign(spec, SimpleBO(), default_constructor(spec, bounds=spec.bounds_tensor()))
    campaign.seed(torch.rand(4, 2, dtype=torch.double))
    ctrl = RunController(campaign, experiment=exp, evaluate=lambda o: o, max_steps=10)
    assert ctrl.can_auto is True
    ctrl.run()
    assert len(campaign.history) == 10
    assert ctrl.status == RunStatus.DONE


def test_multiobjective_synthetic_has_tradeoff():
    torch.manual_seed(0)
    spec = _spec(m=2, d=2)
    exp = SyntheticExperiment(spec, seed=5)
    campaign = Campaign(spec, ParetoBO(), default_constructor(spec, bounds=spec.bounds_tensor()))
    campaign.seed(torch.rand(6, 2, dtype=torch.double))
    ctrl = RunController(campaign, experiment=exp, evaluate=lambda o: o, max_steps=12)
    ctrl.run()
    assert len(campaign.recommend()) >= 2  # a real front emerges
