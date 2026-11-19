"""Neural-network surrogates (abbot.nn, experimental): ensembles, MC dropout and
deep kernels on your own network, with InfoBAX by fine-tuning."""

import copy
import warnings

import pytest
import torch

from abbot import BAXPolicy, Campaign, ContinuousDomain, OutputSpec, ProblemSpec
from abbot.goal import TopK
from abbot.model import EnsembleConstructor
from abbot.nn import DeepKernelConstructor, MCDropoutConstructor, NetEnsembleConstructor
from abbot.policy import ParetoBO, SimpleBO, UncertaintySampling
from abbot.session import check_uq_pairing

pytestmark = pytest.mark.slow  # network training

SPEC = ProblemSpec(
    domain=[
        ContinuousDomain(name="a", lower=-2, upper=2),
        ContinuousDomain(name="b", lower=-2, upper=2),
    ],
    outputs=[OutputSpec(name="f", direction="OBSERVE"), OutputSpec(name="g", direction="OBSERVE")],
)
BOUNDS = SPEC.bounds_tensor()


def mlp(d, m):
    return torch.nn.Sequential(
        torch.nn.Linear(d, 32), torch.nn.Tanh(), torch.nn.Linear(32, m)
    )


def dropout_mlp(d, m):
    return torch.nn.Sequential(
        torch.nn.Linear(d, 64), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(64, m)
    )


def _truth(X):
    a, b = X[:, 0], X[:, 1]
    return torch.stack([(a - 0.5) ** 2 + (b + 0.5) ** 2, a * b], dim=-1)


def _data(n=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, 2, generator=g, dtype=torch.double) * 4 - 2
    return X, _truth(X)


class _Experiment:
    @staticmethod
    def run(design):
        X = torch.tensor([[float(design["a"]), float(design["b"])]], dtype=torch.double)
        return _truth(X)[0].tolist()


CONSTRUCTORS = {
    "ensemble": lambda **kw: NetEnsembleConstructor(mlp, n_members=3, epochs=200, bounds=BOUNDS, **kw),
    "dropout": lambda **kw: MCDropoutConstructor(dropout_mlp, n_samples=8, epochs=200, bounds=BOUNDS, **kw),
}


# -- the model -----------------------------------------------------------------


@pytest.mark.parametrize("kind", ["ensemble", "dropout"])
def test_a_network_model_is_a_botorch_model_with_samples(kind):
    X, Y = _data()
    model = CONSTRUCTORS[kind]().build(X, Y)
    post = model.posterior(X)
    assert post.mean.shape == (12, 2) and post.variance.shape == (12, 2)
    assert (post.variance > 0).all()
    assert post.rsample(torch.Size([4])).shape == (4, 12, 2)
    # It fits the data it was trained on (relative to the outputs' spread).
    err = ((post.mean - Y) ** 2).mean(0) / Y.var(0)
    assert (err < 0.5).all()
    noise = model.observation_noise_variance(X)
    assert noise.shape == (12, 2) and (noise > 0).all()


def test_a_dropout_sample_is_one_function_wherever_it_is_evaluated():
    X, Y = _data()
    model = CONSTRUCTORS["dropout"]().build(X, Y)
    grid = torch.rand(50, 2, dtype=torch.double) * 4 - 2
    together = model(grid)  # (samples, 50, m)
    alone = torch.stack([model(grid[i : i + 1])[:, 0] for i in range(5)], dim=1)
    assert torch.allclose(together[:, :5], alone)
    assert not torch.allclose(together[0], together[1])  # the masks differ


@pytest.mark.parametrize("kind", ["ensemble", "dropout"])
def test_prediction_in_chunks_matches_all_at_once(kind):
    torch.manual_seed(0)
    X, Y = _data()
    chunked = CONSTRUCTORS[kind](chunk_size=7).build(X, Y)
    whole = copy.deepcopy(chunked)
    for net in whole.nets:
        net.chunk_size = None
    grid = torch.rand(50, 2, dtype=torch.double) * 4 - 2
    assert torch.allclose(chunked(grid), whole(grid))  # dropout: the same masks per chunk


def test_identical_members_warn():
    pretrained = mlp(2, 2).double()

    def same_every_time(d, m):
        return copy.deepcopy(pretrained)  # the same weights for every member

    X, Y = _data()
    with pytest.warns(UserWarning, match="no uncertainty"):
        NetEnsembleConstructor(same_every_time, n_members=3, epochs=20, bounds=BOUNDS).build(X, Y)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        CONSTRUCTORS["ensemble"]().build(X, Y)  # independent members: no warning


def test_dropout_needs_a_dropout_layer():
    X, Y = _data()
    with pytest.raises(ValueError, match="no torch.nn.Dropout layer"):
        MCDropoutConstructor(mlp, epochs=5).build(X, Y)


def test_one_network_is_a_point_model_for_meanbax_only():
    one = NetEnsembleConstructor(mlp, n_members=1, epochs=50, bounds=BOUNDS)
    assert one.provides_uq == "point" and one.supports_conditioning is False
    assert NetEnsembleConstructor(mlp).provides_uq == "samples"  # the class is unchanged
    check_uq_pairing(BAXPolicy(TopK(k=3, output="f"), "MeanBAX"), one)
    for strategy in ("PSBAX", "SwitchBAX", "InfoBAX", "US"):
        with pytest.raises(ValueError):
            check_uq_pairing(BAXPolicy(TopK(k=3, output="f"), strategy), one)
    X, Y = _data()
    post = one.build(X, Y).posterior(X)
    assert (post.variance == 0).all()
    with pytest.raises(ValueError, match="at least 1"):
        NetEnsembleConstructor(mlp, n_members=0)


def test_meanbax_on_one_network_picks_at_random_inside_the_set():
    torch.manual_seed(0)
    policy = BAXPolicy(TopK(k=20, output="f", largest=False), "MeanBAX", n_candidates=256)
    c = Campaign(SPEC, policy, NetEnsembleConstructor(mlp, n_members=1, epochs=100, bounds=BOUNDS),
                 on_model_error="raise")
    for _ in range(8):
        c.step(_Experiment)
    assert not policy.switched  # inside the predicted set
    assert policy._no_spread
    value = float(policy.last_acquisition_value[0])
    assert 1.0 <= value <= 2.0  # a random score in [0, 1), shifted above the rest


def test_with_nothing_new_in_the_set_one_network_measures_the_farthest_design():
    from abbot import LevelSet

    torch.manual_seed(0)
    policy = BAXPolicy(LevelSet(threshold=1e6, output="f"), "MeanBAX", n_candidates=128)
    c = Campaign(SPEC, policy, NetEnsembleConstructor(mlp, n_members=1, epochs=50, bounds=BOUNDS),
                 on_model_error="raise")
    for _ in range(5):
        c.step(_Experiment)
    taken = c.history.designs.double()
    (cand,) = c.suggest(1)
    assert policy.switched  # the goal selects nothing: the fallback
    grid = policy.candidate_grid(SPEC)
    span = BOUNDS[1] - BOUNDS[0]
    dist = torch.cdist((grid - BOUNDS[0]) / span, (taken - BOUNDS[0]) / span).min(-1).values
    best = grid[int(dist.argmax())]
    assert torch.allclose(torch.tensor([cand["a"], cand["b"]], dtype=torch.double), best)


@pytest.mark.parametrize("kind", ["ensemble", "dropout"])
def test_conditioning_fine_tunes_a_copy_toward_the_new_points(kind):
    torch.manual_seed(0)  # dropout's training masks
    X, Y = _data()
    model = CONSTRUCTORS[kind](refit_epochs=200).build(X, Y)
    Xn = torch.tensor([[1.5, 1.5], [-1.5, 1.5]], dtype=torch.double)
    Yn = torch.tensor([[5.0, -3.0], [4.0, 2.0]], dtype=torch.double)
    before = model.posterior(Xn).mean
    cond = model.condition_on_observations(Xn, Yn)
    after = cond.posterior(Xn).mean
    assert ((after - Yn).abs() < (before - Yn).abs()).all()
    assert torch.equal(model.posterior(Xn).mean, before)  # the original is untouched
    control = model.refit_control()
    assert control is model.refit_control()  # one per model
    # As InfoBAX does: one sample's own values. Fitting them, the samples usually
    # agree more there than after the same fine-tune on the data alone (not always:
    # the other members don't always reach that sample's function), so the median.
    ratios = []
    for seed in range(5):
        g = torch.Generator().manual_seed(seed)
        grid = torch.rand(10, 2, generator=g, dtype=torch.double) * 4 - 2
        own = model(grid)[0].detach()
        cond = model.condition_on_observations(grid, own)
        ratios.append(float(cond.posterior(grid).variance.mean() / control.posterior(grid).variance.mean()))
    assert torch.tensor(ratios).median() < 1
    # In a no-grad block too (the batch's kriging believer conditions in one).
    with torch.no_grad():
        model.condition_on_observations(Xn, Yn)


def test_control_false_leaves_no_control():
    X, Y = _data()
    assert CONSTRUCTORS["ensemble"](control=False).build(X, Y).refit_control() is None


def test_the_next_fit_starts_from_the_last_one():
    calls = []

    def train(net, X, Y, epochs):
        calls.append(epochs)

    ctor = NetEnsembleConstructor(mlp, n_members=2, epochs=50, warm_epochs=7, train=train)
    X, Y = _data()
    first = ctor.build(X, Y)
    ctor.build(X, Y)
    assert calls == [50, 50, 7, 7]
    calls.clear()
    first.condition_on_observations(X[:1], Y[:1])
    assert calls == [100, 100]  # refit_epochs, with the custom trainer


def test_a_gap_is_left_out_of_the_loss():
    X, Y = _data()
    Y[:6, 1] = float("nan")
    for ctor in (CONSTRUCTORS["ensemble"](), DeepKernelConstructor(mlp, bounds=BOUNDS, epochs=50)):
        post = ctor.build(X, Y).posterior(X)
        assert torch.isfinite(post.mean).all() and torch.isfinite(post.variance).all()
    dkl = DeepKernelConstructor(mlp, bounds=BOUNDS, epochs=50).build(X, Y)
    assert [len(m.train_targets) for m in dkl.models] == [12, 6]


def test_noise_can_be_given():
    X, Y = _data()
    model = CONSTRUCTORS["ensemble"](noise=[0.04, 0.01]).build(X, Y)
    assert torch.allclose(model.observation_noise_variance(X[:3])[0], torch.tensor([0.04, 0.01]).double())


def test_featurize_turns_an_index_into_the_network_input():
    """An index variable into a table of structures: ``featurize`` looks them up."""
    table = torch.randn(20, 6, dtype=torch.double)

    def featurize(X):
        return table[X[:, 0].long()]

    X = torch.arange(0, 20, 2, dtype=torch.double).unsqueeze(-1)
    Y = table[X[:, 0].long()].sum(-1, keepdim=True)
    model = NetEnsembleConstructor(lambda d, m: mlp(6, m), featurize=featurize, epochs=300).build(X, Y)
    grid = torch.arange(20, dtype=torch.double).unsqueeze(-1)
    assert model.posterior(grid).mean.shape == (20, 1)
    err = ((model.posterior(X).mean - Y) ** 2).mean() / Y.var()
    assert err < 0.5


# -- deep kernel ---------------------------------------------------------------


def test_a_deep_kernel_is_a_gp_with_exact_conditioning():
    X, Y = _data()
    ctor = DeepKernelConstructor(mlp, bounds=BOUNDS, epochs=100)
    assert ctor.provides_uq == "calibrated_variance"
    model = ctor.build(X, Y[:, :1])
    Xn = torch.tensor([[1.5, 1.5]], dtype=torch.double)
    model.posterior(Xn)  # a GP conditions once it has predicted (as in a campaign)
    cond = model.condition_on_observations(Xn, torch.tensor([[3.0]], dtype=torch.double))
    assert cond.posterior(Xn).variance < model.posterior(Xn).variance
    assert torch.isfinite(model.likelihood.noise).all() and (model.likelihood.noise > 0).all()


def test_check_reads_a_deep_kernel_without_input_lengthscales():
    c = Campaign(SPEC, BAXPolicy(TopK(k=5, output="f"), "PSBAX", n_candidates=128),
                 DeepKernelConstructor(mlp, bounds=BOUNDS, epochs=50), on_model_error="raise")
    for _ in range(7):
        c.step(_Experiment)
    report = c.check()
    assert len(report.fits) == 2
    assert all(fit.lengthscales == {} for fit in report.fits)  # features, not inputs


# -- pairing -------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["InfoBAX", "SwitchBAX"])
def test_a_model_that_cant_be_updated_is_refused_for_infobax(strategy):
    policy = BAXPolicy(TopK(k=3, output="f"), strategy)
    with pytest.raises(ValueError, match="can't be updated"):
        check_uq_pairing(policy, EnsembleConstructor(n_members=5))
    check_uq_pairing(policy, CONSTRUCTORS["ensemble"]())
    check_uq_pairing(policy, CONSTRUCTORS["dropout"]())


def test_a_toy_ensemble_still_pairs_with_strategies_that_dont_update():
    for strategy in ("MeanBAX", "PSBAX"):
        check_uq_pairing(BAXPolicy(TopK(k=3, output="f"), strategy), EnsembleConstructor(n_members=5))


def test_policies_wanting_calibrated_variance_warn_with_samples():
    for kind in ("ensemble", "dropout"):
        with pytest.warns(UserWarning, match="isn't"):
            check_uq_pairing(UncertaintySampling(), CONSTRUCTORS[kind]())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_uq_pairing(UncertaintySampling(), DeepKernelConstructor(mlp))
    with pytest.raises(ValueError, match="calibrated"):  # a single network: refused
        check_uq_pairing(UncertaintySampling(), EnsembleConstructor(n_members=1))


@pytest.mark.parametrize("kind", ["ensemble", "dropout"])
@pytest.mark.parametrize("policy", ["SimpleBO", "ParetoBO", "UncertaintySampling", "US"])
def test_bo_and_us_run_on_a_network_model(kind, policy):
    spec = SPEC
    if policy == "SimpleBO":
        spec = ProblemSpec(domain=SPEC.domain, outputs=[OutputSpec(name="f", direction="MINIMIZE")])
    if policy == "ParetoBO":
        spec = ProblemSpec(domain=SPEC.domain, outputs=[
            OutputSpec(name="f", direction="MINIMIZE"), OutputSpec(name="g")])
    pol = {
        "SimpleBO": SimpleBO, "ParetoBO": ParetoBO, "UncertaintySampling": UncertaintySampling,
        "US": lambda: BAXPolicy(TopK(k=3, output="f"), "US", n_candidates=128),
    }[policy]()

    class Run:
        @staticmethod
        def run(design):
            return _Experiment.run(design)[: spec.n_outputs]

    torch.manual_seed(0)
    with pytest.warns(UserWarning, match="calibrated"):
        c = Campaign(spec, pol, CONSTRUCTORS[kind](), on_model_error="raise")
    for _ in range(5):
        c.step(Run)
    c.step(Run, n=2)
    assert len(c.history) == 7


# -- in a campaign ---------------------------------------------------------------


def test_infobax_scores_each_member_once_against_the_control():
    calls = {"cond": 0, "control": 0}
    ctor = CONSTRUCTORS["ensemble"](refit_epochs=20)
    built = []
    orig_build = ctor.build

    def build(*a, **k):
        model = orig_build(*a, **k)
        cond, ctrl = model.condition_on_observations, model.refit_control

        def counted_cond(*a, **k):
            calls["cond"] += 1
            return cond(*a, **k)

        def counted_control():
            calls["control"] += 1
            return ctrl()

        model.condition_on_observations = counted_cond
        model.refit_control = counted_control
        built.append(model)
        return model

    ctor.build = build
    policy = BAXPolicy(TopK(k=5, output="f", largest=False), "InfoBAX",
                       n_candidates=128, n_posterior_samples=20)
    c = Campaign(SPEC, policy, ctor, on_model_error="raise")
    while not built:
        c.step(_Experiment)
    assert calls == {"cond": 3, "control": 1}  # 3 members, not 20 repeats
    assert c.last_warning is None


@pytest.mark.parametrize("kind", ["ensemble", "dropout", "deep kernel"])
@pytest.mark.parametrize("strategy", ["MeanBAX", "PSBAX", "SwitchBAX", "InfoBAX"])
def test_every_strategy_runs_a_campaign(kind, strategy):
    torch.manual_seed(0)
    ctor = (
        DeepKernelConstructor(mlp, bounds=BOUNDS, epochs=100)
        if kind == "deep kernel" else CONSTRUCTORS[kind](refit_epochs=20)
    )
    policy = BAXPolicy(TopK(k=5, output="f", largest=False), strategy,
                       n_candidates=128, n_posterior_samples=4)
    c = Campaign(SPEC, policy, ctor, on_model_error="raise")
    for _ in range(4):
        c.step(_Experiment)
    c.step(_Experiment, n=2)  # a batch conditions on its earlier picks
    assert c.current_model() is not None
    assert len({tuple(r.candidate.tolist()) for r in c.history.records}) == len(c.history)
