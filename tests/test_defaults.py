"""Spec-derived defaults: policy, model, seed designs, and the minimal run_campaign."""

import pytest
import torch

from abbot import (
    BAXPolicy,
    Campaign,
    ContinuousDomain,
    EnumeratedDomain,
    InequalitySpec,
    ModelListConstructor,
    ObjectiveSpec,
    OutputSpec,
    ParetoBO,
    ProblemSpec,
    SimpleBO,
    SingleTaskGPConstructor,
    default_constructor,
    default_initial_designs,
    default_policy,
    run_campaign,
)
from abbot.acq_opt import Continuous, DiscreteArgmax
from abbot.goal import TopK
from abbot.history import DataStore


def _cont(m: int = 1) -> ProblemSpec:
    return ProblemSpec(
        domain=[
            ContinuousDomain(name="temp", lower=20.0, upper=80.0),
            ContinuousDomain(name="conc", lower=0.0, upper=1.0),
        ],
        objectives=[ObjectiveSpec(name=f"f{i}") for i in range(m)],
    )


def _enum() -> ProblemSpec:
    return ProblemSpec(
        domain=[EnumeratedDomain(name="cat", values=[0.0, 1.0, 5.0])],
        objectives=[ObjectiveSpec(name="f")],
    )


# --- default_policy ----------------------------------------------------------


def test_default_policy_by_objective_count():
    assert isinstance(default_policy(_cont(1)), SimpleBO)
    assert isinstance(default_policy(_cont(2)), ParetoBO)


def test_default_policy_picks_optimizer_from_domain():
    assert isinstance(default_policy(_cont()).optimizer, Continuous)
    assert isinstance(default_policy(_enum()).optimizer, DiscreteArgmax)


def test_default_policy_takes_mixed_spaces_and_refuses_no_objectives():
    mixed = ProblemSpec(
        domain=[
            ContinuousDomain(name="x", lower=0.0, upper=1.0),
            EnumeratedDomain(name="c", values=[0.0, 1.0]),
        ],
        objectives=[ObjectiveSpec(name="f")],
    )
    assert isinstance(default_policy(mixed).optimizer, Continuous)
    observe_only = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        outputs=[OutputSpec(name="y", direction="OBSERVE")],
    )
    with pytest.raises(ValueError, match="no objectives"):
        default_policy(observe_only)


# --- default_constructor -----------------------------------------------------


def test_default_constructor_uses_spec_bounds():
    spec = _cont(1)
    mc = default_constructor(spec)
    assert isinstance(mc, SingleTaskGPConstructor)
    assert torch.equal(mc.bounds, spec.bounds_tensor())
    assert isinstance(default_constructor(_cont(2)), ModelListConstructor)


# --- sample_designs / default_initial_designs --------------------------------


def test_sample_designs_respects_bounds_and_enumerated_values():
    X = _cont().sample_designs(16, seed=0)
    assert X.shape == (16, 2)
    assert (X[:, 0] >= 20.0).all() and (X[:, 0] <= 80.0).all()
    assert (X[:, 1] >= 0.0).all() and (X[:, 1] <= 1.0).all()
    E = _enum().sample_designs(16, seed=0)
    assert set(E.reshape(-1).tolist()) == {0.0, 1.0, 5.0}


def test_sample_designs_meets_constraints():
    spec = ProblemSpec(
        domain=[
            ContinuousDomain(name="a", lower=0.0, upper=1.0),
            ContinuousDomain(name="b", lower=0.0, upper=1.0),
        ],
        objectives=[ObjectiveSpec(name="f")],
        constraints=[InequalitySpec(indices=[0, 1], coeffs=[1.0, 1.0], rhs=1.5)],
    )
    X = spec.sample_designs(10, seed=0)
    assert X.shape == (10, 2)
    assert (X.sum(dim=-1) >= 1.5).all()


def test_sample_designs_follows_torch_manual_seed():
    torch.manual_seed(7)
    a = _cont().sample_designs(5)
    torch.manual_seed(7)
    b = _cont().sample_designs(5)
    assert torch.equal(a, b)


def test_default_initial_designs_count():
    assert default_initial_designs(_cont(1)).shape == (5, 2)  # 2d + 1
    one_d_three_obj = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        objectives=[ObjectiveSpec(name=f"f{i}") for i in range(3)],
    )
    assert default_initial_designs(one_d_three_obj).shape == (6, 1)  # 2m floor


# --- Campaign / run_campaign -------------------------------------------------


def test_campaign_needs_only_a_spec():
    c = Campaign(_cont())
    assert isinstance(c.policy, SimpleBO)
    (pending,) = c.suggest(1)  # the first default seed
    assert set(pending) == {"temp", "conc"}
    assert 20.0 <= pending["temp"] <= 80.0


def test_campaign_queues_default_seeds_on_first_suggest(caplog):
    import logging

    torch.manual_seed(0)
    c = Campaign(_cont())
    with caplog.at_level(logging.INFO, logger="abbot.session"):
        c.suggest(1)
    assert "Queued 5 default initial designs" in caplog.text
    assert len(c.state().pending) == 1 and len(c._seed_queue) == 4  # 2d + 1 in all


def test_initial_designs_choose_skip_or_yield_to_data():
    mine = torch.tensor([[30.0, 0.1], [70.0, 0.9]], dtype=torch.double)
    c = Campaign(_cont(), initial_designs=mine)
    assert [d["temp"] for d in c.suggest(2)] == [30.0, 70.0]
    assert not c._seed_queue
    cold = Campaign(_cont(), initial_designs=torch.empty(0, 2))
    cold.suggest(1)
    assert not cold._seed_queue  # a random cold-start probe, nothing queued
    seeded = Campaign(_cont())
    seeded.seed(mine[:1])
    seeded.suggest(1)
    assert not seeded._seed_queue  # your own seed wins
    recorded = Campaign(_cont())
    recorded.record({"temp": 50.0, "conc": 0.5}, [1.0])
    recorded.suggest(1)
    assert not recorded._seed_queue  # data first: no default seeds


@pytest.mark.slow
def test_step_without_evaluate_takes_the_outputs_as_returned():
    torch.manual_seed(0)
    c = Campaign(_peak_spec())
    for _ in range(4):
        c.step(_Peak())
    assert len(c.history) == 4
    assert c.history.records[0].outputs.item() == pytest.approx(
        _Peak().run({"x": c.history.records[0].candidate.item()}))


def test_campaign_explicit_policy_still_wins():
    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        outputs=[OutputSpec(name="y", direction="OBSERVE")],
    )
    c = Campaign(spec, BAXPolicy(TopK(k=1)))
    assert isinstance(c.policy, BAXPolicy)


class _Peak:
    """Max at x=0.3 on [0, 1]."""

    def run(self, design: dict) -> float:
        return -((design["x"] - 0.3) ** 2)


def _peak_spec() -> ProblemSpec:
    return ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        objectives=[ObjectiveSpec(name="f")],
    )


@pytest.mark.slow
def test_minimal_run_campaign_finds_optimum():
    torch.manual_seed(0)
    result = run_campaign(_peak_spec(), _Peak(), n_shots=8)
    assert len(result.history) == 3 + 8  # 2d + 1 default seeds
    assert abs(float(result.best.candidate) - 0.3) < 0.1


@pytest.mark.slow
def test_default_seeds_not_requeued_on_resume(tmp_path):
    torch.manual_seed(0)
    store = DataStore(tmp_path / "run")
    run_campaign(_peak_spec(), _Peak(), n_shots=2, store=store)
    result = run_campaign(_peak_spec(), _Peak(), n_shots=4, store=store)
    assert len(result.history) == 3 + 4


def test_sample_designs_reaches_a_thin_feasible_region():
    # ~1e-6 of the box is feasible: rejection sampling can't find it, hit-and-run can.
    spec = ProblemSpec(
        domain=[
            ContinuousDomain(name="a", lower=0.0, upper=1.0),
            ContinuousDomain(name="b", lower=0.0, upper=1.0),
        ],
        objectives=[ObjectiveSpec(name="f")],
        constraints=[InequalitySpec(indices=[0, 1], coeffs=[1.0, 1.0], rhs=1.998)],
    )
    X = spec.sample_designs(8, seed=0)
    assert X.shape == (8, 2)
    assert (X.sum(dim=-1) >= 1.998 - 1e-9).all() and (X <= 1.0).all()


def test_targets_hook_redefines_the_objective_after_the_fact():
    """A non-SBI use of ``targets``: the stored outputs score one quantity, the
    lab later decides another matters; the hand-over follows the new definition
    without re-measuring."""
    from abbot import Campaign, apply_targets

    spec = _peak_spec()
    measured = {0.1: (0.1, 0.9), 0.5: (0.5, 0.5), 0.9: (0.9, 0.1)}  # x -> (a, b)

    def score_b(history, spec):
        return torch.tensor([[r.observation[1]] for r in history.clean.records], dtype=torch.double)

    stored, rescored = Campaign(spec), Campaign(spec, targets=score_b)
    for c in (stored, rescored):
        for x, (a, b) in measured.items():
            c.record({"x": x}, [a], observation=(a, b))  # stored output: a
    def best(c):
        return float(c.history.records[c.recommend()[0]].candidate)

    assert best(stored) == 0.9 and best(rescored) == 0.1
    assert apply_targets(score_b, rescored.history, spec).clean.outputs.reshape(-1).tolist() == [
        0.9, 0.5, 0.1,
    ]
