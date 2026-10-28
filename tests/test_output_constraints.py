"""Output constraints: the spec (``OutputSpec`` bounds, ``LinearConstraint``, the
two constraint lists), feasibility, constrained policies and the recommendation."""

import json
import math

import pytest
import torch

from abbot import (
    BAXPolicy,
    Campaign,
    ContinuousDomain,
    InequalitySpec,
    LevelSet,
    LinearConstraint,
    OutputSpec,
    ParetoBO,
    ProblemSpec,
    SimpleBO,
    Source,
    TensorExperiment,
)
from abbot.analysis import convergence
from abbot.feasibility import (
    LogProbFeasible,
    feasible,
    log_prob_feasible_from,
    violation,
)
from abbot.history import History, Record
from abbot.model import default_constructor
from abbot.session import canonical_Y, feasibility, recommend


def _domain():
    return [ContinuousDomain(name="a", lower=0, upper=1), ContinuousDomain(name="b", lower=0, upper=1)]


def _spec(**kw):
    """Maximize y, subject to c = a + b >= 1 (an OBSERVE output)."""
    outputs = kw.pop("outputs", [OutputSpec(name="y"), OutputSpec(name="c", direction="OBSERVE", min_val=1.0)])
    return ProblemSpec(domain=_domain(), outputs=outputs, **kw)


def _f(X):
    """y peaks at (0.2, 0.2), where c = 0.4 breaks the constraint; the best
    feasible design is (0.5, 0.5)."""
    X = X.double().reshape(-1, 2)
    return torch.stack([-((X - 0.2) ** 2).sum(-1), X.sum(-1)], -1)


def _history(X, f=_f):
    h = History()
    for x, y in zip(X, f(X)):
        h.append(Record(candidate=x.double(), observation=None, outputs=y))
    return h


# --- the spec ------------------------------------------------------------------


def test_linear_constraint_takes_coeffs_positionally_and_checks_bounds():
    c = LinearConstraint({"a": 1, "b": 2}, max_val=1.5)
    assert c.coeffs == {"a": 1.0, "b": 2.0} and c.min_val is None
    assert LinearConstraint(coeffs={"a": 1}, min_val=0) == LinearConstraint({"a": 1}, min_val=0)
    with pytest.raises(ValueError, match="min_val, max_val or both"):
        LinearConstraint({"a": 1})
    with pytest.raises(ValueError, match="must exceed"):
        LinearConstraint({"a": 1}, min_val=2, max_val=1)
    with pytest.raises(ValueError, match="at least one name"):
        LinearConstraint({}, min_val=0)


def test_output_bounds_are_checked():
    assert OutputSpec(name="p", min_val=0.9).is_constrained
    assert not OutputSpec(name="p").is_constrained
    with pytest.raises(ValueError, match="must exceed"):
        OutputSpec(name="p", min_val=1, max_val=1)


def test_constraints_is_the_old_name_for_domain_constraints():
    old = InequalitySpec(indices=[0, 1], coeffs=[1.0, 1.0], rhs=0.5)
    spec = _spec(constraints=[old])
    assert spec.domain_constraints == [old] and spec.constraints == [old]
    with pytest.raises(ValueError, match="not both"):
        _spec(constraints=[old], domain_constraints=[old])
    # Old JSON (and old stores' specs) still load.
    data = json.loads(_spec().model_dump_json())
    data["constraints"] = data.pop("domain_constraints")
    data["constraints"] = [old.model_dump()]
    assert ProblemSpec.model_validate(data).domain_constraints == [old]


def test_names_in_the_wrong_list_get_a_pointed_error():
    with pytest.raises(ValueError, match="'c' is an output: put it in output_constraints"):
        _spec(domain_constraints=[LinearConstraint({"a": 1, "c": 1}, max_val=1)])
    with pytest.raises(ValueError, match="'a' is a design variable: put it in domain_constraints"):
        _spec(output_constraints=[LinearConstraint({"y": 1, "a": 1}, max_val=1)])
    with pytest.raises(ValueError, match="'zz' is not a known output"):
        _spec(output_constraints=[LinearConstraint({"zz": 1}, max_val=1)])
    with pytest.raises(ValueError, match="out of range"):
        _spec(domain_constraints=[InequalitySpec(indices=[5], coeffs=[1.0], rhs=0)])


def test_domain_constraints_by_name_bound_proposals_and_seeds():
    spec = _spec(domain_constraints=[LinearConstraint({"a": 1, "b": 1}, min_val=0.5, max_val=1.2)])
    cons = spec.inequality_constraints()
    assert len(cons) == 2  # one entry per bound, botorch's >= form
    (i0, c0, r0), (i1, c1, r1) = cons
    assert i0.tolist() == [0, 1] and c0.tolist() == [1, 1] and r0 == 0.5
    assert i1.tolist() == [0, 1] and c1.tolist() == [-1, -1] and r1 == -1.2
    s = spec.sample_designs(64, seed=0).sum(-1)
    assert ((s >= 0.5 - 1e-9) & (s <= 1.2 + 1e-9)).all()


def test_spec_round_trips_and_fingerprint_keeps_the_old_key():
    spec = _spec(
        domain_constraints=[LinearConstraint({"a": 1, "b": -1}, max_val=0.5)],
        output_constraints=[LinearConstraint({"y": 1, "c": 1}, min_val=0)],
    )
    assert ProblemSpec.model_validate_json(spec.model_dump_json()) == spec
    fp = spec.fingerprint()
    assert "constraints" in fp and "domain_constraints" not in fp
    assert _spec().fingerprint()["constraints"] is None  # same as before the rename


def test_output_constraint_rows():
    spec = _spec(
        outputs=[OutputSpec(name="y", direction="MINIMIZE", max_val=3), OutputSpec(name="c", direction="OBSERVE")],
        output_constraints=[LinearConstraint({"y": 1, "c": 2}, min_val=0, max_val=4)],
    )
    A, lo, hi = spec.output_constraint_rows()
    assert A.tolist() == [[1, 0], [1, 2]]
    assert lo.tolist() == [-math.inf, 0] and hi.tolist() == [3, 4]
    assert spec.has_output_constraints and not _spec(outputs=[OutputSpec(name="y")]).has_output_constraints


# --- feasibility -----------------------------------------------------------------


def test_feasible_and_violation_use_raw_units():
    spec = _spec(
        outputs=[OutputSpec(name="y", direction="MINIMIZE", max_val=3), OutputSpec(name="c", direction="OBSERVE")],
        output_constraints=[LinearConstraint({"y": 1, "c": 1}, min_val=0)],
    )
    Y = torch.tensor([[2.0, 0.0], [4.0, 0.0], [1.0, -2.0], [5.0, -9.0]])
    assert feasible(spec, Y).tolist() == [True, False, False, False]
    assert violation(spec, Y).tolist() == [0.0, 1.0, 1.0, 6.0]


def test_log_prob_feasible_matches_the_normal_cdf_with_finite_gradients():
    spec = _spec(outputs=[OutputSpec(name="y", direction="MINIMIZE", min_val=-1, max_val=1),
                          OutputSpec(name="c", direction="OBSERVE", min_val=0.5)])
    # Canonical y = -raw y, so a canonical mean of -0.5 is a raw mean of 0.5.
    mean = torch.tensor([[-0.5, 1.0]], dtype=torch.double, requires_grad=True)
    var = torch.tensor([[0.25, 4.0]], dtype=torch.double)
    log_p = log_prob_feasible_from(spec, mean, var)
    N = torch.distributions.Normal(0.0, 1.0)
    expected = (N.cdf(torch.tensor(1.0)) - N.cdf(torch.tensor(-3.0))) * N.cdf(torch.tensor(0.25))
    assert float(log_p.exp()) == pytest.approx(float(expected), rel=1e-6)
    log_p.sum().backward()
    assert torch.isfinite(mean.grad).all()


# --- policies (GP fits) -----------------------------------------------------------


@pytest.mark.slow
def test_simple_bo_searches_for_feasibility_until_a_run_satisfies_the_constraints():
    torch.manual_seed(0)
    spec = _spec()
    h = _history(torch.rand(8, 2) * 0.4)  # c <= 0.8: nothing feasible yet
    model = default_constructor(spec).build(h.designs.double(), canonical_Y(h, spec))
    x = SimpleBO().suggest(h, spec, model, n=2)
    assert x.shape == (2, 2)
    assert (x.sum(-1) > 0.8).all()  # heads for a + b >= 1, away from y's peak


@pytest.mark.slow
def test_campaign_finds_the_best_feasible_design():
    torch.manual_seed(0)
    spec = _spec()
    campaign = Campaign(spec)
    experiment = TensorExperiment(type("E", (), {"run": staticmethod(lambda x: _f(x)[0])})())
    for _ in range(16):
        campaign.step(experiment)
    result = campaign.result()
    best = result.best.candidate
    assert result.feasible == [True] and result.p_feasible[0] >= 0.5
    assert float(best.sum()) >= 1.0 - 0.02
    assert torch.allclose(best, torch.tensor([0.5, 0.5], dtype=best.dtype), atol=0.1)
    # The unconstrained peak is infeasible; convergence never reports it.
    assert convergence(campaign.history, spec)[-1] <= -0.18 + 1e-3


@pytest.mark.slow
def test_recommendation_when_nothing_is_feasible():
    torch.manual_seed(0)
    spec = _spec()
    X = torch.tensor([[0.1, 0.1], [0.3, 0.3], [0.45, 0.45], [0.2, 0.0]])
    h = _history(X)
    # From observed values: the smallest violation (c = 0.9).
    assert recommend(h, spec, denoise=False) == [2]
    assert feasibility(h, spec, [2]) == [0.0]
    model = default_constructor(spec).build(h.designs.double(), canonical_Y(h, spec))
    rec = recommend(h, spec, model)
    assert rec == [2]
    assert feasibility(h, spec, rec, model)[0] < 0.5
    campaign = Campaign(spec, initial_designs=torch.empty(0, 2))
    campaign.load_history(h)
    result = campaign.result()
    assert result.feasible == [False]
    assert "No design is likely to satisfy" in campaign.last_warning


@pytest.mark.slow
def test_recommendation_skips_infeasible_designs():
    spec = _spec()
    h = _history(torch.tensor([[0.2, 0.2], [0.6, 0.6], [0.5, 0.5], [0.9, 0.1]]))
    assert recommend(h, spec, denoise=False) == [2]  # (0.2, 0.2) has the best y but c = 0.4


@pytest.mark.slow
def test_pareto_bo_with_an_output_constraint():
    torch.manual_seed(0)
    outputs = [OutputSpec(name="y1"), OutputSpec(name="y2"), OutputSpec(name="c", direction="OBSERVE", min_val=1.0)]
    spec = _spec(outputs=outputs)

    def f(X):
        X = X.double().reshape(-1, 2)
        return torch.stack([-((X - 0.2) ** 2).sum(-1), -((X - 0.4) ** 2).sum(-1), X.sum(-1)], -1)

    h = _history(torch.rand(10, 2), f)
    model = default_constructor(spec).build(h.designs.double(), canonical_Y(h, spec))
    x = ParetoBO().suggest(h, spec, model, n=2)
    assert x.shape == (2, 2)
    rec = recommend(h, spec, denoise=False)
    assert rec and feasible(spec, h.outputs[rec].double()).all()
    # Nothing feasible: it searches for feasibility instead of an inferred front.
    low = _history(torch.rand(10, 2) * 0.4, f)
    model = default_constructor(spec).build(low.designs.double(), canonical_Y(low, spec))
    x = ParetoBO().suggest(low, spec, model, n=1)
    assert float(x.sum()) > 0.8


@pytest.mark.slow
def test_bax_target_set_keeps_only_feasible_designs():
    torch.manual_seed(0)
    spec = _spec()
    policy = BAXPolicy(LevelSet(-0.5, output=0), strategy="InfoBAX", n_candidates=64, n_posterior_samples=4)
    Y = torch.tensor([[0.0, 1.5], [0.0, 0.5], [-1.0, 1.5]])
    assert policy._identify(spec, Y, torch.zeros(3, 2)) == [0]
    h = _history(torch.rand(8, 2))
    model = default_constructor(spec).build(h.designs.double(), canonical_Y(h, spec))
    x = policy.suggest(h, spec, model, n=1)
    assert x.shape == (1, 2)


@pytest.mark.slow
def test_simple_bo_with_sources_and_an_output_constraint():
    torch.manual_seed(0)
    spec = ProblemSpec(
        domain=_domain(),
        outputs=[OutputSpec(name="y"), OutputSpec(name="c", direction="OBSERVE", min_val=1.0)],
        sources=[Source(domain=ContinuousDomain(name="s", lower=0, upper=1), target=1.0)],
    )

    def f(X):
        X = X.double().reshape(-1, 3)
        return _f(X[:, :2]) + 0.1 * (1 - X[:, 2:])

    X = torch.rand(10, 3)
    X[:3, 2] = 1.0
    h = _history(X, f)
    for r in h.records:
        r.cost = 1 + 10 * float(r.candidate[2])
    model = default_constructor(spec).build(h.designs.double(), canonical_Y(h, spec))
    policy = SimpleBO(n_candidates=200)
    x = policy.suggest(h, spec, model, n=1)
    assert x.shape == (1, 3)
    # Nothing feasible at the target: the search stays at the target.
    X[:3, :2] = 0.1
    low = _history(X, f)
    model = default_constructor(spec).build(low.designs.double(), canonical_Y(low, spec))
    x = SimpleBO().suggest(low, spec, model, n=1)
    assert float(x[0, 2]) == 1.0


@pytest.mark.slow
def test_log_prob_feasible_acquisition_shapes():
    spec = _spec()
    h = _history(torch.rand(6, 2))
    model = default_constructor(spec).build(h.designs.double(), canonical_Y(h, spec))
    X = torch.rand(5, 1, 2, dtype=torch.double)
    assert LogProbFeasible(model, spec)(X).shape == (5,)
    assert LogProbFeasible(model, spec, explore=True)(X).shape == (5,)


# --- formats ---------------------------------------------------------------------


def test_output_bounds_round_trip_through_csv():
    from abbot.io import spec_history_from_csv, spec_history_to_csv

    spec = _spec(outputs=[OutputSpec(name="y", max_val=2.0), OutputSpec(name="c", direction="OBSERVE", min_val=1.0, max_val=3.0), OutputSpec(name="z", direction="OBSERVE")])
    back, _ = spec_history_from_csv(spec_history_to_csv(spec))
    assert back.outputs == spec.outputs
    with pytest.raises(ValueError, match="output constraints over several outputs"):
        spec_history_to_csv(_spec(output_constraints=[LinearConstraint({"y": 1, "c": 1}, min_val=0)]))


def test_output_bounds_round_trip_through_vocs():
    from gest_api.vocs import (
        BoundsConstraint,
        GreaterThanConstraint,
        LessThanConstraint,
    )

    from abbot.interop.xopt import spec_to_vocs, vocs_to_spec

    spec = _spec(outputs=[
        OutputSpec(name="y", max_val=2.0),
        OutputSpec(name="c", direction="OBSERVE", min_val=1.0),
        OutputSpec(name="d", direction="OBSERVE", min_val=0.0, max_val=1.0),
        OutputSpec(name="z", direction="OBSERVE"),
    ])
    vocs = spec_to_vocs(spec)
    assert isinstance(vocs.constraints["y"], LessThanConstraint)
    assert isinstance(vocs.constraints["c"], GreaterThanConstraint)
    assert isinstance(vocs.constraints["d"], BoundsConstraint)
    assert set(vocs.observables) == {"z"}
    assert vocs_to_spec(vocs).outputs == spec.outputs
    for kw in (
        {"domain_constraints": [LinearConstraint({"a": 1}, max_val=0.5)]},
        {"output_constraints": [LinearConstraint({"y": 1, "c": 1}, min_val=0)]},
    ):
        with pytest.raises(NotImplementedError, match="no linear constraints"):
            spec_to_vocs(_spec(**kw))


@pytest.mark.slow
@pytest.mark.parametrize("ordered", [True, False])
def test_simple_bo_chooses_sources_with_an_observe_output(ordered):
    """Entropy search takes single-output models only; a spec with an OBSERVE output
    (as most constrained specs have) still chooses sources."""
    from abbot import EnumeratedDomain

    torch.manual_seed(0)
    source = (
        Source(domain=ContinuousDomain(name="s", lower=0, upper=1), target=1.0)
        if ordered else Source(domain=EnumeratedDomain(name="s", values=[0, 1]), target=1)
    )
    spec = ProblemSpec(
        domain=_domain(),
        outputs=[OutputSpec(name="y"), OutputSpec(name="c", direction="OBSERVE")],
        sources=[source],
    )
    X = torch.rand(12, 3)
    X[:, 2] = X[:, 2].round()
    X[:4, 2] = 1.0
    h = _history(X, lambda X: _f(X[:, :2]) + 0.1 * (1 - X[:, 2:]))
    model = default_constructor(spec).build(h.designs.double(), canonical_Y(h, spec))
    x = SimpleBO(n_candidates=200).suggest(h, spec, model, n=1)
    assert x.shape == (1, 3)
