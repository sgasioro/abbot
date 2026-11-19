"""BAXPolicy strategies on a toy GP (botorch, self-contained)."""

import pytest
import torch

from abbot.goal import BandIntersection, FunctionGoal, PercentileSet
from abbot.history import History, Record
from abbot.loop import check_uq_pairing
from abbot.model import SingleTaskGPConstructor
from abbot.policy import BAXPolicy
from abbot.problem import Box, ContinuousDomain, ObjectiveSpec, ProblemSpec

pytestmark = pytest.mark.slow  # GP fits + fantasy conditioning


def _spec():
    return ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        objectives=[ObjectiveSpec(name="y", direction="MAXIMIZE")],
        boxes=[Box(lower=[0.4], upper=[0.6])],
    )


def _fit(spec, xs, ys):
    """Fit a single-output GP in canonical space (MAXIMIZE -> raw == canonical)."""
    h = History()
    for x, y in zip(xs, ys):
        h.append(Record(candidate=torch.tensor([x]), observation=None, outputs=torch.tensor([y])))
    ctor = SingleTaskGPConstructor(bounds=spec.bounds_tensor())
    return ctor.build(h.designs.double(), h.outputs.double()), h


def test_us_picks_high_uncertainty_point():
    torch.manual_seed(0)
    spec = _spec()
    # Train points clustered near 0; the grid far from them should be most uncertain.
    model, h = _fit(spec, [0.0, 0.05, 0.1], [0.1, 0.1, 0.1])
    pol = BAXPolicy(PercentileSet(80.0), strategy="US", n_candidates=200)
    x = pol.suggest(h, spec, model, n=1)
    assert x.shape == (1, 1)
    assert float(x[0, 0]) > 0.5  # away from the training cluster


def test_meanbax_exploits_in_predicted_target_then_prevents_requery():
    torch.manual_seed(0)
    spec = _spec()
    # A bump peaking in [0.4,0.6] so the mean-execution target set is non-empty.
    xs = [0.1, 0.3, 0.5, 0.7, 0.9]
    ys = [0.0, 0.4, 1.0, 0.4, 0.0]
    model, h = _fit(spec, xs, ys)
    pol = BAXPolicy(BandIntersection([(0.3, 1.2)]), strategy="MeanBAX", n_candidates=200)
    x0 = pol.suggest(h, spec, model, n=1)
    assert pol.collected_count == 1
    # prevent-requery: a pick still in flight is not proposed again.
    idx0 = pol._collected[0]
    x1 = pol.suggest(h, spec, model, n=1, pending=x0)
    assert pol._collected[0] == idx0 and len(pol._collected) == 2
    assert pol._collected[1] != idx0 and not torch.equal(x0, x1)
    # ...nor once it has been measured.
    h.append(Record(candidate=x0[0], observation=None, outputs=torch.tensor([1.0])))
    x2 = pol.suggest(h, spec, model, n=1)
    assert not torch.equal(x0, x2)


def test_meanbax_falls_back_to_us_when_target_empty():
    torch.manual_seed(0)
    spec = _spec()
    model, h = _fit(spec, [0.1, 0.5, 0.9], [0.0, 0.0, 0.0])
    # An impossible band (never satisfied by the mean) -> stuck -> US fallback.
    pol = BAXPolicy(BandIntersection([(1e6, 1e7)]), strategy="MeanBAX", n_candidates=200)
    pol.suggest(h, spec, model, n=1)
    assert pol.switched is True


def test_switchbax_falls_back_to_infobax_when_stuck():
    torch.manual_seed(0)
    spec = _spec()
    model, h = _fit(spec, [0.1, 0.5, 0.9], [0.2, 0.2, 0.2])
    pol = BAXPolicy(
        BandIntersection([(1e6, 1e7)]), strategy="SwitchBAX",
        n_candidates=120, n_posterior_samples=5,
    )
    pol.suggest(h, spec, model, n=1)
    assert pol.switched is True  # routed into InfoBAX


def test_infobax_eig_finite_and_prefers_boundary():
    torch.manual_seed(0)
    spec = _spec()
    xs = [0.1, 0.3, 0.5, 0.7, 0.9]
    ys = [0.0, 0.4, 1.0, 0.4, 0.0]
    model = _fit(spec, xs, ys)[0]
    pol = BAXPolicy(BandIntersection([(0.5, 1.2)]), strategy="InfoBAX",
                    n_candidates=120, n_posterior_samples=8)
    grid = pol.candidate_grid(spec)
    eig = pol._acquisition(spec, model, grid)
    assert torch.isfinite(eig).all()
    # EIG should be larger near the target boundary than at the far edge x~0.
    near_boundary = eig[(grid[:, 0] > 0.35) & (grid[:, 0] < 0.65)].mean()
    far = eig[grid[:, 0] < 0.1].mean()
    assert float(near_boundary) >= float(far) - 1e-6


def test_infobax_frees_each_fantasy_before_the_next(monkeypatch):
    # A fantasy model is in a reference cycle, so only the cyclic collector frees it;
    # InfoBAX collects after each sample, or every fantasy is kept until the loop
    # ends (memory n² per sample). With automatic collection off, only that does.
    import gc
    import weakref

    import abbot.policy as P

    torch.manual_seed(0)
    spec = _spec()
    model = _fit(spec, [0.1, 0.3, 0.5, 0.7, 0.9], [0.0, 0.4, 1.0, 0.4, 0.0])[0]
    pol = BAXPolicy(BandIntersection([(0.5, 1.2)]), strategy="InfoBAX",
                    n_candidates=60, n_posterior_samples=6)
    alive, condition = [], P._condition

    def tracked(*args, **kwargs):
        fant = condition(*args, **kwargs)
        alive.append(weakref.ref(fant))
        return fant

    monkeypatch.setattr(P, "_condition", tracked)
    gc.collect()
    gc.disable()
    try:
        pol._acquisition(spec, model, pol.candidate_grid(spec))
        assert len(alive) > 1
        assert sum(r() is not None for r in alive) == 0
    finally:
        gc.enable()


def test_infobax_refuses_when_every_fantasy_fails(monkeypatch):
    import abbot.policy

    torch.manual_seed(0)
    spec = _spec()
    model = _fit(spec, [0.1, 0.3, 0.5, 0.7, 0.9], [0.0, 0.4, 1.0, 0.4, 0.0])[0]
    pol = BAXPolicy(BandIntersection([(0.5, 1.2)]), strategy="InfoBAX",
                    n_candidates=40, n_posterior_samples=4)

    def refuse(*args, **kwargs):
        raise IndexError("can't condition")

    monkeypatch.setattr(abbot.policy, "_condition", refuse)
    with pytest.raises(RuntimeError, match="failed for every posterior sample"):
        pol._acquisition(spec, model, pol.candidate_grid(spec))


def test_coherent_samples_are_joint():
    torch.manual_seed(0)
    spec = _spec()
    model = _fit(spec, [0.1, 0.5, 0.9], [0.1, 1.0, 0.1])[0]
    grid = torch.linspace(0, 1, 50, dtype=torch.double).unsqueeze(-1)
    with torch.no_grad():
        s = model.posterior(grid).rsample(torch.Size([200]))  # (200, 50, 1)
    s = s.squeeze(-1)
    # Adjacent grid points should correlate strongly under a smooth kernel;
    # distant points much less (coherence = joint, not per-point marginal).
    adj = torch.corrcoef(torch.stack([s[:, 0], s[:, 1]]))[0, 1]
    distant = torch.corrcoef(torch.stack([s[:, 0], s[:, 49]]))[0, 1]
    assert float(adj) > float(distant)


def test_function_goal_drives_bax():
    """An arbitrary FunctionGoal works as a BAXPolicy goal (the flexibility hatch)."""
    torch.manual_seed(0)
    spec = _spec()
    model = _fit(spec, [0.1, 0.3, 0.5, 0.7, 0.9], [0.0, 0.4, 1.0, 0.4, 0.0])[0]
    # Custom goal: top third of candidates by the (single) output.
    goal = FunctionGoal(
        lambda f, X: (f[:, 0] >= f[:, 0].quantile(0.66)).nonzero(as_tuple=True)[0].tolist()
    )
    pol = BAXPolicy(goal, strategy="MeanBAX", n_candidates=150)
    x = pol.suggest(History(), spec, model, n=1)
    assert x.shape == (1, 1)
    assert pol.collected_count == 1


def test_uq_pairing_refuses_point_model_with_infobax():
    class _PointConstructor:
        provides_uq = "point"

    pol = BAXPolicy(PercentileSet(80.0), strategy="InfoBAX")
    with pytest.raises(ValueError, match="coherent posterior samples"):
        check_uq_pairing(pol, _PointConstructor())
    # MeanBAX only needs a point prediction -> fine on a point model.
    check_uq_pairing(BAXPolicy(PercentileSet(80.0), strategy="MeanBAX"), _PointConstructor())


def test_requery_lets_a_measured_design_come_back():
    torch.manual_seed(0)
    spec = _spec()
    xs, ys = [0.1, 0.3, 0.5, 0.7, 0.9], [0.0, 0.4, 1.0, 0.4, 0.0]
    model, h = _fit(spec, xs, ys)
    grid = torch.tensor([*xs, 0.2], dtype=torch.double).reshape(-1, 1)
    plain = BAXPolicy(PercentileSet(50.0), strategy="US", candidates=grid)
    assert float(plain.suggest(h, spec, model)[0, 0]) == pytest.approx(0.2)  # only new
    again = BAXPolicy(PercentileSet(50.0), strategy="US", candidates=grid, requery=True)
    x = again.suggest(h, spec, model, n=6)
    # Every design, measured or not, and none twice in one batch (in flight).
    assert sorted(x[:, 0].tolist()) == pytest.approx(sorted(grid[:, 0].tolist()))
    assert again.collected_count == 6  # measured ones still count (MeanBAX's "stuck")


def test_psbax_picks_in_a_sampled_target_set_and_explores():
    spec = _spec()
    xs, ys = [0.1, 0.3, 0.5, 0.7, 0.9], [0.0, 0.4, 1.0, 0.4, 0.0]
    model, h = _fit(spec, xs, ys)
    grid = torch.linspace(0, 1, 101, dtype=torch.double).reshape(-1, 1)
    goal = BandIntersection([(0.6, 1.5)])  # around the peak at 0.5
    picks = []
    for seed in range(8):
        torch.manual_seed(seed)
        pol = BAXPolicy(goal, strategy="PSBAX", candidates=grid)
        (x,) = pol.suggest(h, spec, model)
        picks.append(float(x[0]))
        assert not pol.switched and pol.uq_requirement == "samples"
    assert all(0.3 < p < 0.7 for p in picks)  # inside a plausible target set
    # A batch: a fresh draw per pick, none twice.
    torch.manual_seed(0)
    batch = BAXPolicy(goal, strategy="PSBAX", candidates=grid).suggest(h, spec, model, n=3)
    assert len({float(v) for v in batch[:, 0]}) == 3


def test_psbax_falls_back_to_us_when_the_draw_selects_nothing():
    torch.manual_seed(0)
    spec = _spec()
    model, h = _fit(spec, [0.1, 0.5, 0.9], [0.0, 0.0, 0.0])
    pol = BAXPolicy(BandIntersection([(1e6, 1e7)]), strategy="PSBAX", n_candidates=200)
    pol.suggest(h, spec, model, n=1)
    assert pol.switched is True


# -- in_set: how MeanBAX / SwitchBAX / PS-BAX choose inside the target set -------


def _in_set_campaign(in_set):
    from abbot import Campaign, ContinuousDomain, OutputSpec, ProblemSpec
    from abbot.goal import TopK

    spec = ProblemSpec(
        domain=[ContinuousDomain(name="a", lower=0, upper=1), ContinuousDomain(name="b", lower=0, upper=1)],
        outputs=[OutputSpec(name="f", direction="OBSERVE")],
    )
    policy = BAXPolicy(TopK(k=30, output="f"), "MeanBAX", n_candidates=256, in_set=in_set)
    c = Campaign(spec, policy, on_model_error="raise")

    class Run:
        @staticmethod
        def run(d):
            return [-(d["a"] - 0.3) ** 2 - (d["b"] - 0.6) ** 2]

    torch.manual_seed(0)
    for _ in range(6):
        c.step(Run)
    return c, policy, spec


def test_in_set_takes_a_function_of_the_set_the_runs_and_the_uncertainty():
    seen = {}

    def first_coordinate(X, taken, uncertainty):
        seen.update(k=len(X), taken=None if taken is None else len(taken), u=uncertainty.shape)
        return X[:, 0]

    c, policy, spec = _in_set_campaign(first_coordinate)
    (cand,) = c.suggest(1)
    assert seen["k"] >= 1 and seen["taken"] == len(c.history) and seen["u"] == (seen["k"],)
    # The pick is the in-set design with the largest a (among those not taken).
    mean = c.current_model().posterior(policy.candidate_grid(spec)).mean.detach()
    target = policy._goal_for(spec).identify_subspace(mean)
    free = [i for i in target if i not in policy._collected[:-1]]
    a_max = max(float(policy.candidate_grid(spec)[i, 0]) for i in free)
    assert cand["a"] == pytest.approx(a_max)


def test_in_set_farthest_and_validation():
    c, policy, _ = _in_set_campaign("farthest")
    c.suggest(1)
    assert float(policy.last_acquisition_value[0]) >= 1.0
    with pytest.raises(ValueError, match="in_set must be one of"):
        BAXPolicy(None, "MeanBAX", in_set="nearest")
    with pytest.raises(ValueError, match="scores for"):
        c, _, _ = _in_set_campaign(lambda X, taken, u: torch.zeros(len(X) + 1))
        c.suggest(1)
