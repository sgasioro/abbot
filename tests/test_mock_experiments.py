"""Mock experiments: MockAnalyticExperiment (a seeded formula) and
MockDatasetExperiment (replaying known results)."""

import pytest
import torch

from abbot.experiment import MockAnalyticExperiment, MockDatasetExperiment
from abbot.goal import TopK
from abbot.history import History, Record
from abbot.loop import run_campaign
from abbot.model import default_constructor
from abbot.policy import BAXPolicy, ParetoBO, SimpleBO
from abbot.problem import ContinuousDomain, ObjectiveSpec, OutputSpec, ProblemSpec
from abbot.session import Campaign
from abbot.ui.controller import RunController, RunStatus


def _spec(m=1, d=2):
    return ProblemSpec(
        domain=[ContinuousDomain(name=f"x{i}", lower=0.0, upper=1.0) for i in range(d)],
        objectives=[ObjectiveSpec(name=f"y{j}", direction="MAXIMIZE") for j in range(m)],
    )


def test_output_width_matches_objectives():
    for m in (1, 2, 3):
        spec = _spec(m=m, d=2)
        exp = MockAnalyticExperiment(spec, seed=0)
        out = exp.run({"x0": 0.3, "x1": 0.7})
        assert out.shape == (m,)
        assert torch.all(out > 0) and torch.all(out <= 100.0 + 1e-6)


def test_deterministic_and_noise_repeats():
    spec = _spec(m=1, d=2)
    x = {"x0": 0.4, "x1": 0.6}
    # No noise -> identical across experiments with the same seed.
    a = MockAnalyticExperiment(spec, seed=1).run(x)
    b = MockAnalyticExperiment(spec, seed=1).run(x)
    assert torch.allclose(a, b)
    # With noise -> re-measuring the SAME point repeats exactly (seeded by x).
    noisy = MockAnalyticExperiment(spec, seed=1, noise=5.0)
    assert torch.allclose(noisy.run(x), noisy.run(x))


def test_different_seeds_differ():
    spec = _spec(m=1, d=2)
    x = {"x0": 0.2, "x1": 0.8}
    assert not torch.allclose(
        MockAnalyticExperiment(spec, seed=1).run(x), MockAnalyticExperiment(spec, seed=2).run(x)
    )


@pytest.mark.slow
def test_auto_run_with_analytic_experiment():
    torch.manual_seed(0)
    spec = _spec(m=1, d=2)
    exp = MockAnalyticExperiment(spec, seed=3)
    campaign = Campaign(spec, SimpleBO(), default_constructor(spec, bounds=spec.bounds_tensor()))
    campaign.seed(torch.rand(4, 2, dtype=torch.double))
    ctrl = RunController(campaign, experiment=exp, evaluate=lambda o: o, max_steps=10)
    assert ctrl.can_auto is True
    ctrl.run()
    assert len(campaign.history) == 10
    assert ctrl.status == RunStatus.DONE


@pytest.mark.slow
def test_multiobjective_analytic_has_tradeoff():
    torch.manual_seed(0)
    spec = _spec(m=2, d=2)
    exp = MockAnalyticExperiment(spec, seed=5)
    campaign = Campaign(spec, ParetoBO(), default_constructor(spec, bounds=spec.bounds_tensor()))
    campaign.seed(torch.rand(6, 2, dtype=torch.double))
    ctrl = RunController(campaign, experiment=exp, evaluate=lambda o: o, max_steps=12)
    ctrl.run()
    assert len(campaign.recommend()) >= 2  # a real front emerges


# --- MockDatasetExperiment ---------------------------------------------------


def _grid_spec():
    return ProblemSpec(
        domain=[ContinuousDomain(name=n, lower=0.0, upper=1.0) for n in ("a", "b")],
        outputs=[OutputSpec(name=n, direction="OBSERVE") for n in ("y", "z")],
    )


def _grid():
    g = torch.linspace(0, 1, 5, dtype=torch.double)
    X = torch.cartesian_prod(g, g)
    Y = torch.stack([torch.sin(3 * X[:, 0]) + X[:, 1], X.sum(-1)], dim=-1)
    return X, Y


def test_dataset_returns_the_row_for_a_design():
    X, Y = _grid()
    exp = MockDatasetExperiment(_grid_spec(), X, Y)
    assert torch.equal(exp.candidates, X)
    assert torch.equal(exp.run({"a": float(X[7, 0]), "b": float(X[7, 1])}), Y[7])
    # Order comes from the spec, not the dict.
    assert torch.equal(exp.run({"b": float(X[7, 1]), "a": float(X[7, 0])}), Y[7])


def test_dataset_refuses_a_design_it_does_not_have():
    X, Y = _grid()
    exp = MockDatasetExperiment(_grid_spec(), X, Y)
    with pytest.raises(ValueError, match="not in this dataset"):
        exp.run({"a": 0.1, "b": 0.1})


def test_dataset_checks_shapes_against_the_spec():
    X, Y = _grid()
    with pytest.raises(ValueError, match="X must be"):
        MockDatasetExperiment(_grid_spec(), X[:, :1], Y)
    with pytest.raises(ValueError, match="Y must be"):
        MockDatasetExperiment(_grid_spec(), X, Y[:, :1])


def test_dataset_from_history_uses_clean_measured_records():
    X, Y = _grid()
    records = [Record(candidate=x, observation=None, outputs=y) for x, y in zip(X[:3], Y[:3])]
    records.append(Record(candidate=X[3], observation=None, fault=True))
    exp = MockDatasetExperiment.from_history(History(records=records), _grid_spec())
    assert torch.equal(exp.candidates, X[:3])
    with pytest.raises(ValueError):
        exp.run({"a": float(X[3, 0]), "b": float(X[3, 1])})


@pytest.mark.slow
def test_dataset_drives_a_batched_bax_campaign():
    torch.manual_seed(0)
    X, Y = _grid()
    spec = _grid_spec()
    exp = MockDatasetExperiment(spec, X, Y)
    result = run_campaign(
        spec,
        exp,
        n_shots=8,
        batch_size=2,
        policy=BAXPolicy(TopK(k=3, output=0), strategy="MeanBAX", candidates=exp.candidates),
        initial_designs=X[[0, 12, 24]],
    )
    designs = result.history.designs
    assert len(designs) == 11
    # Every proposal came from the dataset, and none was measured twice.
    assert len({tuple(d.tolist()) for d in designs}) == 11
