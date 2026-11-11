"""Spec ergonomics: inputs held fixed (``suggest(fixed=)``), goals and boxes by
output name, and ``log=True`` design variables."""

import math

import pytest
import torch

from abbot import (
    BandIntersection,
    BAXPolicy,
    Box,
    Campaign,
    ContinuousDomain,
    EnumeratedDomain,
    IntegerDomain,
    LevelSet,
    LinearConstraint,
    OutputSpec,
    ParetoBO,
    PercentileSet,
    ProblemSpec,
    SimpleBO,
    Source,
    TensorExperiment,
    TopK,
    UncertaintySampling,
    Wishlist,
    goals_from_spec,
)
from abbot.acq_opt import Continuous
from abbot.history import History, Record
from abbot.model import MultiFidelityGPConstructor, default_constructor
from abbot.session import canonical_Y


def _spec(**kw):
    domain = kw.pop("domain", [
        ContinuousDomain(name="temp", lower=20, upper=80),
        IntegerDomain(name="n", lower=1, upper=5),
        EnumeratedDomain(name="cat", values=[0.0, 1.0]),
    ])
    outputs = kw.pop("outputs", [OutputSpec(name="yield"), OutputSpec(name="purity", direction="OBSERVE")])
    return ProblemSpec(domain=domain, outputs=outputs, **kw)


def _history(spec, X, f):
    h = History()
    for x in X:
        h.append(Record(candidate=x.double(), observation=None, outputs=f(x.double())))
    return h


def _f(x):
    return torch.stack([-((x[0] - 50) / 30) ** 2 - 0.1 * (x[1] - 3) ** 2 + 0.2 * x[2], x[0] / 80])


class _Experiment:
    def run(self, x):
        return _f(x)


# --- fixed inputs --------------------------------------------------------------------


def test_resolve_fixed_checks_names_and_values():
    spec = _spec()
    assert spec.resolve_fixed({"temp": 30, "cat": 1}) == {0: 30.0, 2: 1.0}
    assert spec.resolve_fixed(None) == {}
    for fixed, match in [
        ({"tmp": 30}, "'tmp' is not a variable"),
        ({"yield": 1}, "an output; only inputs can be fixed"),
        ({"temp": 90}, "outside"),
        ({"n": 2.5}, "whole number"),
        ({"cat": 0.5}, "not one of"),
    ]:
        with pytest.raises(ValueError, match=match):
            spec.resolve_fixed(fixed)


def test_sample_designs_holds_fixed_inputs_and_domain_constraints():
    spec = _spec()
    X = spec.sample_designs(32, seed=0, fixed={"temp": 30.0, "cat": 1.0})
    assert (X[:, 0] == 30).all() and (X[:, 2] == 1).all()
    assert len(X[:, 1].unique()) > 1  # the rest still varies
    poly = ProblemSpec(
        domain=[ContinuousDomain(name="a", lower=0, upper=1), ContinuousDomain(name="b", lower=0, upper=1)],
        outputs=[OutputSpec(name="y")],
        domain_constraints=[LinearConstraint({"a": 1, "b": 1}, max_val=1.0)],
    )
    X = poly.sample_designs(16, seed=0, fixed={"a": 0.7})
    assert torch.allclose(X[:, 0], torch.tensor(0.7, dtype=X.dtype))
    assert (X.sum(-1) <= 1 + 1e-9).all()


@pytest.mark.slow
def test_campaign_holds_fixed_inputs_in_seeds_and_proposals():
    torch.manual_seed(0)
    spec = _spec()
    campaign = Campaign(spec)
    experiment = TensorExperiment(_Experiment())
    for _ in range(9):  # seeds, then policy proposals
        campaign.step(experiment, fixed={"temp": 30.0})
        assert campaign.history.records[-1].candidate[0] == 30.0
    assert campaign.history.records[-1].acquisition is not None  # the policy proposed it
    c = campaign.suggest(2, fixed={"n": 4, "cat": 0.0})
    assert all(d["n"] == 4 and d["cat"] == 0.0 for d in c)


def test_cold_start_draws_hold_fixed_inputs():
    campaign = Campaign(_spec(), initial_designs=torch.empty(0, 3))
    (c,) = campaign.suggest(1, fixed={"temp": 42.0})
    assert c["temp"] == 42.0


def test_a_retry_waits_for_a_suggest_it_matches():
    spec = _spec()
    campaign = Campaign(spec, initial_designs=torch.tensor([[60.0, 2.0, 0.0]]))
    (c,) = campaign.suggest(1)
    campaign.fail(c.id, "beam dump")
    (other,) = campaign.suggest(1, fixed={"temp": 30.0})
    assert other["temp"] == 30.0  # not the queued retry at 60
    (retry,) = campaign.suggest(1)
    assert retry["temp"] == 60.0


def test_a_policy_without_fixed_is_refused():
    class Old:
        uq_requirement = "point"

        def suggest(self, history, spec, model, n=1):
            return spec.sample_designs(n)

        def ingest(self, results):
            return None

    campaign = Campaign(_spec(), Old())
    with pytest.raises(ValueError, match="takes no fixed= argument"):
        campaign.suggest(1, fixed={"temp": 30.0})


@pytest.mark.slow
@pytest.mark.parametrize("policy", [SimpleBO(), UncertaintySampling()])
def test_policies_hold_fixed_inputs(policy):
    torch.manual_seed(0)
    spec = _spec()
    h = _history(spec, spec.sample_designs(10, seed=1), _f)
    model = default_constructor(spec).build(h.designs.double(), canonical_Y(h, spec))
    X = policy.suggest(h, spec, model, n=2, fixed={"temp": 33.0, "cat": 1.0})
    assert (X[:, 0] == 33).all() and (X[:, 2] == 1).all()


@pytest.mark.slow
def test_pareto_bo_holds_fixed_inputs():
    torch.manual_seed(0)
    spec = _spec(outputs=[OutputSpec(name="yield"), OutputSpec(name="purity")])
    h = _history(spec, spec.sample_designs(10, seed=1), _f)
    model = default_constructor(spec).build(h.designs.double(), canonical_Y(h, spec))
    X = ParetoBO().suggest(h, spec, model, n=1, fixed={"temp": 33.0})
    assert X[0, 0] == 33


@pytest.mark.slow
def test_simple_bo_with_sources_holds_fixed_inputs():
    torch.manual_seed(0)
    spec = ProblemSpec(
        domain=[ContinuousDomain(name="a", lower=0, upper=1), ContinuousDomain(name="b", lower=0, upper=1)],
        outputs=[OutputSpec(name="y")],
        sources=[Source(domain=ContinuousDomain(name="s", lower=0, upper=1), target=1.0)],
    )
    X = torch.rand(10, 3)
    h = History()
    for x in X:
        h.append(Record(candidate=x.double(), observation=None,
                        outputs=-((x[:2] - 0.3) ** 2).sum(-1, keepdim=True).double(), cost=1 + 9 * float(x[2])))
    model = default_constructor(spec).build(h.designs.double(), canonical_Y(h, spec))
    x = SimpleBO(n_candidates=200).suggest(h, spec, model, n=1, fixed={"a": 0.8})
    assert x[0, 0] == pytest.approx(0.8)
    x = SimpleBO(n_candidates=200).suggest(h, spec, model, n=1, fixed={"s": 0.0})
    assert x[0, 2] == 0.0  # fixing a source overrides the policy's choice


@pytest.mark.slow
def test_bax_holds_fixed_integer_and_enumerated_inputs():
    torch.manual_seed(0)
    spec = _spec()
    policy = BAXPolicy(TopK(k=5, output="yield"), strategy="MeanBAX", n_candidates=256)
    grid = policy.candidate_grid(spec)
    assert (grid[:, 1] == grid[:, 1].round()).all()  # integers take whole values
    assert set(grid[:, 2].tolist()) <= {0.0, 1.0}  # enumerated take allowed values
    h = _history(spec, spec.sample_designs(10, seed=1), _f)
    model = default_constructor(spec).build(h.designs.double(), canonical_Y(h, spec))
    X = policy.suggest(h, spec, model, n=2, fixed={"n": 2, "cat": 1.0})
    assert (X[:, 1] == 2).all() and (X[:, 2] == 1).all()
    with pytest.raises(ValueError, match="no candidate has"):
        policy.suggest(h, spec, model, n=1, fixed={"temp": 33.3})


# --- goals and boxes by output name ----------------------------------------------------


def test_goals_by_output_name():
    spec = _spec()
    f = torch.tensor([[1.0, 0.1], [3.0, 0.9], [2.0, 0.5], [0.0, 0.95]])
    assert TopK(k=2, output="purity").bind(spec).identify_subspace(f) == [3, 1]
    assert LevelSet(1.5, output="yield").bind(spec).identify_subspace(f) == [1, 2]
    assert PercentileSet(50, output="purity").bind(spec).identify_subspace(f) == [1, 3]
    assert BandIntersection({"purity": (0.4, 1.0)}).bind(spec).identify_subspace(f) == [1, 2, 3]
    wish = Wishlist([{"yield": (2.5, 4.0)}, {"purity": (0.0, 0.2)}]).bind(spec)
    assert wish.identify_subspace(f) == [0, 1]
    # Positions still work, and binding leaves the original goal unchanged.
    assert TopK(k=1, output=1).bind(spec).identify_subspace(f) == [3]
    goal = TopK(k=1, output="purity")
    goal.bind(spec)
    assert goal.output == "purity"


def test_goal_names_get_pointed_errors():
    spec = _spec()
    with pytest.raises(ValueError, match="'purty' is not an output"):
        TopK(output="purty").bind(spec)
    with pytest.raises(ValueError, match="use it through BAXPolicy, or call goal.bind"):
        LevelSet(0.5, output="purity").identify_subspace(torch.rand(4, 2))
    with pytest.raises(ValueError, match="say which"):
        PercentileSet(80).identify_subspace(torch.rand(4, 2))


def test_named_boxes():
    spec = _spec(boxes=[Box({"purity": (0.4, 1.0)}), Box(lower=[2.5, 0.0], upper=[4.0, 1.0])])
    f = torch.tensor([[1.0, 0.1], [3.0, 0.9], [2.0, 0.5]])
    assert goals_from_spec(spec).identify_subspace(f) == [1, 2]
    assert ProblemSpec.model_validate_json(spec.model_dump_json()) == spec
    with pytest.raises(ValueError, match="'purty' is not an output"):
        _spec(boxes=[Box({"purty": (0, 1)})])
    with pytest.raises(ValueError, match="named bands or lower/upper, not both"):
        Box({"purity": (0, 1)}, lower=[0], upper=[1])
    with pytest.raises(ValueError, match="must exceed"):
        Box({"purity": (1, 0)})
    with pytest.raises(ValueError, match="name them instead"):
        Box(lower=[0.0], upper=[1.0]).band_dict(spec.output_names)


def test_named_boxes_and_log_flags_round_trip_through_csv():
    from abbot.io import spec_history_from_csv, spec_history_to_csv

    spec = _spec(
        domain=[ContinuousDomain(name="conc", lower=1e-3, upper=1.0, log=True),
                IntegerDomain(name="n", lower=1, upper=100, log=True)],
        boxes=[Box({"purity": (0.4, 1.0)})],
    )
    back, _ = spec_history_from_csv(spec_history_to_csv(spec))
    assert back.domain == spec.domain
    # Written positionally, the unbanded output as (-inf, inf).
    assert back.boxes[0].band_dict(back.output_names) == {
        "yield": (-math.inf, math.inf), "purity": (0.4, 1.0)
    }


@pytest.mark.slow
def test_bax_policy_binds_named_goals():
    torch.manual_seed(0)
    spec = _spec()
    h = _history(spec, spec.sample_designs(10, seed=1), _f)
    model = default_constructor(spec).build(h.designs.double(), canonical_Y(h, spec))
    for strategy in ("MeanBAX", "InfoBAX"):
        policy = BAXPolicy(LevelSet(0.6, output="purity"), strategy=strategy, n_candidates=128,
                           n_posterior_samples=4)
        assert policy.suggest(h, spec, model, n=1).shape == (1, 3)


# --- log=True design variables -----------------------------------------------------------


def _log_spec():
    return ProblemSpec(
        domain=[ContinuousDomain(name="conc", lower=1e-4, upper=1.0, log=True)],
        outputs=[OutputSpec(name="y")],
    )


def test_log_needs_a_positive_lower_bound():
    with pytest.raises(ValueError, match="positive lower bound"):
        ContinuousDomain(name="x", lower=0, upper=1, log=True)
    with pytest.raises(ValueError, match="positive lower bound"):
        IntegerDomain(name="n", lower=0, upper=10, log=True)
    assert Source(domain=ContinuousDomain(name="s", lower=1, upper=10, log=True), target=10).is_log


def test_log_variables_are_sampled_across_decades():
    spec = _log_spec()
    assert spec.log_indices() == [0]
    X = spec.sample_designs(256, seed=0)[:, 0]
    assert 0.4 < float((X < 1e-2).double().mean()) < 0.6  # half below the middle decade
    grid = BAXPolicy(LevelSet(0.0), n_candidates=256).candidate_grid(spec)[:, 0]
    assert 0.4 < float((grid < 1e-2).double().mean()) < 0.6


@pytest.mark.slow
def test_log_variables_are_modeled_on_a_log_scale():
    from botorch.models.transforms.input import Log10

    torch.manual_seed(0)
    spec = _log_spec()
    X = spec.sample_designs(12, seed=0)
    Y = torch.sin(3 * X.log10())  # smooth in log10(conc), wiggly in conc
    model = default_constructor(spec).build(X, Y)
    assert any(isinstance(m, Log10) for m in model.input_transform.modules())
    Xt = torch.logspace(-4, 0, 50, dtype=torch.double).unsqueeze(-1)
    with torch.no_grad():
        err = (model.posterior(Xt).mean - torch.sin(3 * Xt.log10())).abs().max()
    assert err < 0.2


@pytest.mark.slow
def test_optimizer_searches_every_decade_of_a_log_variable():
    spec = _log_spec()

    class Bump(torch.nn.Module):  # narrow around conc = 10^-3.5, flat elsewhere
        def forward(self, X):
            return torch.exp(-(((X[..., 0].clamp_min(1e-12).log10() + 3.5) / 0.2) ** 2)).sum(-1)

    # Starting from uniform draws on [1e-4, 1], the search misses it most of the time.
    for seed in range(3):
        torch.manual_seed(seed)
        x = Continuous().optimize(Bump(), spec, 1)
        assert abs(float(x[0, 0].log10()) + 3.5) < 0.05


@pytest.mark.slow
def test_multi_fidelity_model_with_a_log_design_variable():
    torch.manual_seed(0)
    spec = ProblemSpec(
        domain=[ContinuousDomain(name="conc", lower=1e-3, upper=1.0, log=True)],
        outputs=[OutputSpec(name="y")],
        sources=[Source(domain=IntegerDomain(name="atoms", lower=8, upper=512), target=512, log=True)],
    )
    X = spec.sample_designs(12, seed=0)
    model = MultiFidelityGPConstructor(spec).build(X, X[:, :1].log10())
    with torch.no_grad():
        assert torch.isfinite(model.posterior(X).mean).all()


def test_a_plain_kind_can_be_left_out_of_json():
    spec = ProblemSpec.model_validate({
        "domain": [{"name": "a", "lower": 0, "upper": 1},
                   {"name": "b", "values": ["x", "y"]},
                   {"name": "mix", "dim": 3},
                   {"kind": "integer", "name": "n", "lower": 1, "upper": 4}],
        "outputs": [{"name": "y"}],
        "sources": [{"domain": {"name": "code", "values": ["sim", "lab"]}, "target": "lab"}],
    })
    assert [d.kind for d in spec.domain] == ["continuous", "enumerated", "structured", "integer"]
    assert spec.sources[0].domain.kind == "enumerated"
    with pytest.raises(ValueError, match='add "kind"'):
        ProblemSpec.model_validate({"domain": [{"name": "n"}], "outputs": [{"name": "y"}]})
