"""Correlated outputs: one GP over every output (MultiTaskGPConstructor)."""

import time

import pytest
import torch

from abbot import (
    Campaign,
    ContinuousDomain,
    EnumeratedDomain,
    ModelListConstructor,
    MultiTaskGPConstructor,
    OutputSpec,
    ProblemSpec,
    Source,
    default_constructor,
)
from abbot.goal import TopK
from abbot.loop import drive
from abbot.policy import BAXPolicy, ParetoBO, SimpleBO, UncertaintySampling, _condition


def _spec(directions=("MAXIMIZE", "MINIMIZE", "OBSERVE"), **kw) -> ProblemSpec:
    return ProblemSpec(
        domain=[
            ContinuousDomain(name="a", lower=0.0, upper=2.0),
            ContinuousDomain(name="b", lower=-1.0, upper=1.0),
        ],
        outputs=[OutputSpec(name=f"y{i}", direction=d) for i, d in enumerate(directions)],
        **kw,
    )


def _truth(X: torch.Tensor) -> torch.Tensor:
    """Three outputs on different scales; the first two move together."""
    a = torch.sin(2 * X[..., 0]) + X[..., 1]
    return torch.stack([a, 10 * a + 0.5 * X[..., 0], -0.1 * a + X[..., 1] ** 2], -1)


def _data(n=12, seed=0, spec=None):
    spec = spec or _spec()
    torch.manual_seed(seed)
    X = spec.sample_designs(n)
    return spec, X, _truth(X)


# -- contract --


def test_the_posterior_is_one_column_per_output_in_raw_units():
    spec, X, Y = _data()
    canonical = Y * spec.output_signs()
    gp = MultiTaskGPConstructor(spec).build(X, canonical)
    Xt = spec.sample_designs(20, seed=1)
    post = gp.posterior(Xt)
    assert post.mean.shape == post.variance.shape == (20, 3)
    assert gp.posterior(Xt.unsqueeze(1)).mean.shape == (20, 1, 3)
    assert gp.posterior(Xt).rsample(torch.Size([4])).shape == (4, 20, 3)
    # Each output predicted in its own units, the 10x one included.
    err = (post.mean.detach() - _truth(Xt) * spec.output_signs()).abs().mean(0)
    scale = _truth(Xt).std(0)
    assert (err < 0.3 * scale).all(), (err, scale)


def test_output_vars_set_the_noise():
    spec, X, Y = _data()
    Xt = spec.sample_designs(5, seed=2)
    per_output = torch.tensor([1e-4, 1e-2, 1e-5], dtype=torch.double)  # each its own
    small = MultiTaskGPConstructor(spec).build(X, Y, per_output.expand_as(Y))
    large = MultiTaskGPConstructor(spec).build(X, Y, 10 * per_output.expand_as(Y))
    assert (small.posterior(X[:5]).variance < large.posterior(X[:5]).variance).all()
    noisy = small.posterior(Xt, observation_noise=True).variance - small.posterior(Xt).variance
    assert torch.allclose(noisy, per_output.expand_as(noisy), rtol=1e-2)


def test_correlation_helps_a_sparsely_measured_output():
    # y1 is measured on only 4 of 20 runs; y0, which it follows, on all of them.
    spec = _spec(("MAXIMIZE", "OBSERVE"))
    torch.manual_seed(3)
    X = spec.sample_designs(20)
    Y = _truth(X)[:, :2]
    sparse = Y.clone()
    sparse[4:, 1] = float("nan")
    Xt = spec.sample_designs(50, seed=4)
    joint = MultiTaskGPConstructor(spec).build(X, sparse)
    alone = ModelListConstructor(bounds=spec.bounds_tensor()).build(X[:4], Y[:4])
    rmse = lambda m: float((m.posterior(Xt).mean[:, 1].detach() - _truth(Xt)[:, 1]).pow(2).mean().sqrt())
    assert rmse(joint) < 0.5 * rmse(alone), (rmse(joint), rmse(alone))


def test_minimizing_only_flips_the_learned_correlation():
    # The campaign negates a minimized output before fitting. The model should
    # learn the same thing with that correlation's sign flipped (BoTorch's default
    # output kernel allows only positive correlations, so it couldn't). The start is
    # taken from the data, so this holds exactly, not just in distribution.
    spec, X, Y = _data()
    flip = torch.tensor([1.0, -1.0, 1.0], dtype=torch.double)
    torch.manual_seed(1)  # the random state doesn't matter
    raw = MultiTaskGPConstructor(spec).build(X, Y)
    torch.manual_seed(2)
    flipped = MultiTaskGPConstructor(spec).build(X, Y * flip)
    Xt = spec.sample_designs(10, seed=5)
    scale = Y.std(0)
    gap = ((raw.posterior(Xt).mean * flip - flipped.posterior(Xt).mean).abs() / scale).max()
    assert gap < 1e-4, gap
    corr = lambda B: B[0, 1] / (B[0, 0] * B[1, 1]).sqrt()
    assert corr(_output_covariance(raw)) > 0.9
    assert corr(_output_covariance(flipped)) < -0.9


def _output_covariance(gp) -> torch.Tensor:
    from gpytorch.kernels import IndexKernel

    (index,) = [k for k in gp.covar_module.modules() if isinstance(k, IndexKernel)]
    return index.covar_matrix.to_dense().detach()


@pytest.mark.parametrize("with_vars", [False, True])
def test_conditioning_on_wide_designs(with_vars):
    spec, X, Y = _data()
    gp = MultiTaskGPConstructor(spec).build(X, Y, torch.full_like(Y, 1e-4) if with_vars else None)
    Xc = spec.sample_designs(2, seed=6)
    Yc = gp.posterior(Xc).mean.detach()
    fant = _condition(gp, Xc, Yc, noise=1e-6)  # InfoBAX's near-noise-free fantasy
    assert fant.posterior(Xc).variance.max() < 0.05 * gp.posterior(Xc).variance.max()
    S = 4  # per-sample batches
    batched = gp.condition_on_observations(Xc.expand(S, 2, 2), Yc.expand(S, 2, 3),
                                           noise=torch.full((S, 2, 3), 1e-6, dtype=torch.double))
    assert batched.posterior(Xc).mean.shape == (S, 2, 3)
    if with_vars:  # the fixed-noise path takes the model's own noise level
        assert _condition(gp, Xc, Yc).posterior(Xc).mean.shape == (2, 3)


# -- entry points and refusals --


def test_default_constructor_correlated():
    spec = _spec()
    assert isinstance(default_constructor(spec, correlated=True), MultiTaskGPConstructor)
    with pytest.raises(ValueError, match="at least two outputs"):
        default_constructor(_spec(("MAXIMIZE",)), correlated=True)
    with pytest.raises(ValueError, match="prior_mean"):
        default_constructor(spec, correlated=True, prior_mean={"y0": lambda d: d["a"]})


def test_sources_are_refused_and_explained():
    spec = _spec(sources=[Source(domain=EnumeratedDomain(name="code", values=[0, 1]), target=1)])
    with pytest.raises(ValueError, match="MultiSourceGPConstructor") as err:
        MultiTaskGPConstructor(spec)
    assert "Correlated outputs and sources in one model" in str(err.value)
    with pytest.raises(ValueError, match="rank"):
        MultiTaskGPConstructor(_spec(), rank=4)


def test_labeled_and_log_inputs():
    spec = ProblemSpec(
        domain=[
            ContinuousDomain(name="a", lower=0.01, upper=10.0, log=True),
            EnumeratedDomain(name="solvent", values=["water", "ethanol"]),
        ],
        outputs=[OutputSpec(name="y0"), OutputSpec(name="y1")],
    )
    X = spec.sample_designs(10, seed=7)
    Y = torch.stack([X[:, 0].log() + X[:, 1], 2 * X[:, 0].log() - X[:, 1]], -1)
    gp = MultiTaskGPConstructor(spec).build(X, Y)
    assert gp.posterior(spec.sample_designs(4, seed=8)).mean.shape == (4, 2)


# -- in a campaign --


class _Toy:
    def __init__(self, m: int = 3) -> None:
        self.m = m

    def run(self, design):
        x = torch.tensor([design["a"], design["b"]], dtype=torch.double)
        return _truth(x)[: self.m].tolist()


ONE_OBJECTIVE = ("MAXIMIZE", "OBSERVE", "OBSERVE")


def _campaign(policy, spec=None, n=10):
    spec = spec or _spec(ONE_OBJECTIVE)
    c = Campaign(spec, policy, default_constructor(spec, correlated=True),
                 initial_designs=spec.sample_designs(6, seed=9))
    drive(c, _Toy(spec.n_outputs), n_total=n)
    return c


@pytest.mark.parametrize(
    "policy",
    [
        SimpleBO(),
        UncertaintySampling(),
        BAXPolicy(TopK(k=3, output="y0"), "InfoBAX", n_candidates=128),
        BAXPolicy(TopK(k=3, output="y0"), "SwitchBAX", n_candidates=128),
    ],
    ids=["SimpleBO", "US", "InfoBAX", "SwitchBAX"],
)
def test_campaigns_run_with_correlated_outputs(policy):
    c = _campaign(policy)
    assert len(c.history.clean) == 10
    assert type(c.current_model()).__name__ == "_OutputsMultiTaskGP"
    assert c.recommend() if c.spec.n_objectives else True


def test_pareto_and_output_constraints():
    spec = _spec(("MAXIMIZE", "MINIMIZE", "OBSERVE"))
    _campaign(ParetoBO(), spec)
    constrained = ProblemSpec(
        domain=spec.domain,
        outputs=[OutputSpec(name="y0"), OutputSpec(name="y1", direction="OBSERVE", max_val=5.0)],
    )
    c = _campaign(SimpleBO(), constrained)
    assert c.result().p_feasible


def test_check_holds_out_whole_runs():
    c = _campaign(SimpleBO(), n=12)
    report = c.check()
    assert [f.name for f in report.fits] == ["y0", "y1", "y2"]
    for fit in report.fits:
        assert fit.r2 is not None and len(fit.z) == 12
        assert set(fit.lengthscales) == {"a", "b"}


@pytest.mark.slow
def test_infobax_cost_against_independent_models(capsys):
    """Not a pass/fail on speed: prints the cost of one InfoBAX proposal on a
    1000-point grid with three outputs, joint vs. independent."""
    spec, X, Y = _data(n=20)
    times = {}
    for name, constructor in [
        ("independent", ModelListConstructor(bounds=spec.bounds_tensor())),
        ("correlated", MultiTaskGPConstructor(spec)),
    ]:
        model = constructor.build(X, Y * spec.output_signs())
        policy = BAXPolicy(TopK(k=5, output="y0"), "InfoBAX", n_candidates=1000)
        from abbot.history import History, Record

        history = History([Record(candidate=x, observation=None, outputs=y) for x, y in zip(X, Y)])
        start = time.perf_counter()
        policy.suggest(history, spec, model, 1)
        times[name] = time.perf_counter() - start
    with capsys.disabled():
        print(f"\nInfoBAX, 1000 candidates, 3 outputs: {times}")
    assert all(t < 600 for t in times.values())
