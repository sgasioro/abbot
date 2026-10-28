"""Labeled values (categories): ``EnumeratedDomain(values=["water", ...])``.

Designs carry the labels; tensors inside the loop carry each label's code (its
position in ``values``); the model gets a categorical kernel.
"""

import pytest
import torch

from abbot import (
    BAXPolicy,
    Campaign,
    ContinuousDomain,
    EnumeratedDomain,
    LinearConstraint,
    MockAnalyticExperiment,
    MockDatasetExperiment,
    OutputSpec,
    ParetoBO,
    ProblemSpec,
    SimpleBO,
    Source,
    TensorExperiment,
    TopK,
)
from abbot.history import History, Record
from abbot.io import (
    history_from_csv,
    history_to_csv,
    spec_history_from_csv,
    spec_history_to_csv,
)
from abbot.model import MultiFidelityGPConstructor, default_constructor
from abbot.policy import _condition
from abbot.session import canonical_Y

SOLVENTS = ["water", "ethanol", "dmso"]
OFFSET = {"water": 0.0, "ethanol": 1.0, "dmso": 0.3}


def _spec(**kw):
    domain = kw.pop("domain", [
        ContinuousDomain(name="x", lower=0, upper=1),
        EnumeratedDomain(name="solvent", values=SOLVENTS),
    ])
    outputs = kw.pop("outputs", [OutputSpec(name="yield")])
    return ProblemSpec(domain=domain, outputs=outputs, **kw)


def _yield(design):
    return -((design["x"] - 0.3) ** 2) * 4 + OFFSET[design["solvent"]]


class _Experiment:
    def __init__(self):
        self.seen = []

    def run(self, design):
        self.seen.append(dict(design))
        return [_yield(design)]


def _history(spec, n=12, seed=0):
    h = History()
    for x in spec.sample_designs(n, seed=seed):
        design = spec.as_dict(x)
        h.append(Record(candidate=x, observation=None,
                        outputs=torch.tensor([_yield(design)], dtype=torch.double)))
    return h


# --- the spec --------------------------------------------------------------------------


def test_labels_are_categories_with_codes():
    d = EnumeratedDomain(name="solvent", values=SOLVENTS)
    assert d.is_categorical and d.codes == [0.0, 1.0, 2.0]
    assert d.encode("dmso") == 2.0 and d.decode(1.0) == "ethanol"
    with pytest.raises(ValueError, match="not one of"):
        d.encode("acetone")
    numbers = EnumeratedDomain(name="n", values=[1, 2.5])
    assert not numbers.is_categorical and numbers.codes == [1.0, 2.5]
    assert numbers.encode(2.5) == 2.5 and numbers.decode(2.5) == 2.5
    with pytest.raises(ValueError, match="repeated"):
        EnumeratedDomain(name="s", values=["a", "a"])


def test_spec_converts_between_designs_and_tensors():
    spec = _spec()
    assert spec.categorical_indices() == [1]
    x = spec.as_tensor({"x": 0.5, "solvent": "dmso"})
    assert x.tolist() == [0.5, 2.0]
    assert spec.as_dict(x) == {"x": 0.5, "solvent": "dmso"}
    assert spec.bounds_tensor()[:, 1].tolist() == [0.0, 2.0]
    with pytest.raises(ValueError, match="missing"):
        spec.as_tensor({"x": 0.5})
    with pytest.raises(ValueError, match="not one of"):
        spec.as_tensor({"x": 0.5, "solvent": "acetone"})
    with pytest.raises(ValueError, match="expected a number"):
        spec.as_tensor({"x": "high", "solvent": "water"})


def test_draws_and_fixed_inputs_use_labels():
    spec = _spec()
    X = spec.sample_designs(30, seed=0)
    assert set(X[:, 1].tolist()) == {0.0, 1.0, 2.0}
    assert spec.resolve_fixed({"solvent": "ethanol"}) == {1: 1.0}
    assert (spec.sample_designs(5, fixed={"solvent": "dmso"})[:, 1] == 2).all()
    with pytest.raises(ValueError, match="fixed: solvent='acetone' is not one of"):
        spec.resolve_fixed({"solvent": "acetone"})


def test_linear_constraints_refuse_labeled_variables():
    with pytest.raises(ValueError, match="take labels"):
        _spec(domain_constraints=[LinearConstraint({"x": 1, "solvent": 1}, max_val=1)])


def test_sources_can_be_labeled():
    s = Source(domain=EnumeratedDomain(name="code", values=["dft", "xtb"]), target="dft")
    assert not s.is_ordered and s.target_code == 0.0
    spec = _spec(sources=[s])
    assert spec.target_features() == {2: 0.0}
    assert spec.categorical_indices() == [1]  # the source is a source, not a category
    with pytest.raises(ValueError, match="not one of"):
        Source(domain=EnumeratedDomain(name="code", values=["dft", "xtb"]), target="cc")
    with pytest.raises(ValueError, match="labels are unordered"):
        Source(domain=EnumeratedDomain(name="code", values=["dft", "xtb"]), target="dft",
               ordered=True)
    with pytest.raises(ValueError, match="must be a number"):
        Source(domain=ContinuousDomain(name="f", lower=0, upper=1), target="high")


def test_spec_json_round_trip_keeps_labels():
    spec = _spec()
    again = ProblemSpec.model_validate_json(spec.model_dump_json())
    assert again.domain[1].values == SOLVENTS and again.categorical_indices() == [1]


# --- the model ---------------------------------------------------------------------------


@pytest.mark.slow
def test_default_model_has_a_categorical_kernel():
    from botorch.models import MixedSingleTaskGP

    torch.manual_seed(0)
    spec = _spec(outputs=[OutputSpec(name="yield"), OutputSpec(name="p", direction="OBSERVE")])
    h = History()
    for x in spec.sample_designs(15, seed=0):
        d = spec.as_dict(x)
        h.append(Record(candidate=x, observation=None,
                        outputs=torch.tensor([_yield(d), d["x"]], dtype=torch.double)))
    model = default_constructor(spec).build(h.designs.double(), canonical_Y(h, spec))
    assert all(isinstance(m, MixedSingleTaskGP) for m in model.models)
    # InfoBAX's batched fantasy conditioning
    Xc = spec.sample_designs(2, seed=1).expand(3, 2, 2)
    model.posterior(Xc)  # conditioning extends the caches a prediction builds
    conditioned = _condition(model, Xc, torch.zeros(3, 2, 2, dtype=torch.double), 1e-6)
    assert conditioned.posterior(Xc).variance.max() < 1e-3


@pytest.mark.slow
@pytest.mark.parametrize("source", [
    Source(domain=ContinuousDomain(name="fid", lower=0.1, upper=1), target=1.0),
    Source(domain=EnumeratedDomain(name="code", values=["dft", "xtb"]), target="dft"),
])
@pytest.mark.parametrize("linear_truncated", [True, False])
def test_multi_fidelity_models_with_a_labeled_variable(source, linear_truncated):
    from botorch.models.kernels.categorical import CategoricalKernel

    if not source.is_ordered and not linear_truncated:
        pytest.skip("one model for an unordered source")
    torch.manual_seed(0)
    spec = _spec(outputs=[OutputSpec(name="yield"), OutputSpec(name="p", direction="OBSERVE")],
                 sources=[source])
    X = spec.sample_designs(20, seed=0)
    Y = torch.stack([torch.sin(5 * X[:, 0]) + X[:, 1], X[:, 1] ** 2], -1)
    model = MultiFidelityGPConstructor(spec, linear_truncated=linear_truncated).build(X, Y)
    assert any(isinstance(k, CategoricalKernel) for k in model.modules())
    assert model.posterior(X[:3]).mean.shape == (3, 2)
    assert model.subset_output([0]).posterior(X[:3]).mean.shape == (3, 1)  # entropy search
    Xc = X[:2].expand(4, 2, 3)
    conditioned = _condition(model, Xc, torch.zeros(4, 2, 2, dtype=torch.double), 1e-6)
    assert conditioned.posterior(Xc).variance.max() < 1e-3


# --- the loop ------------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1])
def test_campaign_finds_the_best_label(seed):
    torch.manual_seed(seed)
    campaign = Campaign(_spec(), SimpleBO())
    experiment = _Experiment()
    for _ in range(14):
        campaign.step(experiment)
    assert {d["solvent"] for d in experiment.seen} <= set(SOLVENTS)  # labels, not codes
    best = campaign.spec.as_dict(campaign.result().best.candidate)
    assert best["solvent"] == "ethanol"
    assert best["x"] == pytest.approx(0.3, abs=0.1)


def test_campaign_takes_and_hands_out_labels():
    spec = _spec()
    campaign = Campaign(spec, initial_designs=[{"x": 0.1, "solvent": "dmso"}])
    (c,) = campaign.suggest(1)
    assert c == {"x": 0.1, "solvent": "dmso"}
    assert campaign.state().pending == [{"x": 0.1, "solvent": "dmso"}]
    campaign.ingest([_yield(c)], id=c.id)
    campaign.record({"x": 0.5, "solvent": "water"}, [_yield({"x": 0.5, "solvent": "water"})])
    assert campaign.history.records[-1].candidate.tolist() == [0.5, 0.0]
    (c,) = campaign.suggest(1, fixed={"solvent": "ethanol"})
    assert c["solvent"] == "ethanol"


@pytest.mark.slow
def test_policies_with_labels():
    torch.manual_seed(0)
    spec = _spec(outputs=[OutputSpec(name="yield"), OutputSpec(name="other")])
    h = History()
    for x in spec.sample_designs(12, seed=0):
        d = spec.as_dict(x)
        h.append(Record(candidate=x, observation=None,
                        outputs=torch.tensor([_yield(d), d["x"]], dtype=torch.double)))
    model = default_constructor(spec).build(h.designs.double(), canonical_Y(h, spec))
    X = ParetoBO().suggest(h, spec, model, n=1, fixed={"solvent": "water"})
    assert X[0, 1] == 0
    policy = BAXPolicy(TopK(k=3, output="yield"), strategy="InfoBAX", n_candidates=128)
    assert set(policy.candidate_grid(spec)[:, 1].tolist()) <= {0.0, 1.0, 2.0}
    X = policy.suggest(h, spec, model, n=2, fixed={"solvent": "dmso"})
    assert (X[:, 1] == 2).all()


@pytest.mark.slow
def test_source_aware_campaign_with_labels():
    torch.manual_seed(0)
    spec = _spec(sources=[Source(domain=EnumeratedDomain(name="code", values=["dft", "xtb"]),
                                 target="dft")])
    experiment = MockAnalyticExperiment(spec, seed=0)
    campaign = Campaign(spec, SimpleBO(n_candidates=100))
    for _ in range(7):
        campaign.step(experiment)
    for r in campaign.history.records:
        d = spec.as_dict(r.candidate)
        assert d["solvent"] in SOLVENTS and d["code"] in ("dft", "xtb")


# --- adapters and files -------------------------------------------------------------------


def test_tensor_experiment_needs_the_spec_for_labels():
    spec = _spec()

    class Vector:
        def run(self, x):
            return x[:1]

    with pytest.raises(TypeError, match="spec=spec"):
        TensorExperiment(Vector()).run({"x": 0.2, "solvent": "dmso"})
    out = TensorExperiment(Vector(), spec=spec).run({"x": 0.2, "solvent": "dmso"})
    assert out.tolist() == [0.2]


def test_dataset_experiment_takes_labels():
    spec = _spec()
    X = spec.sample_designs(4, seed=0)
    experiment = MockDatasetExperiment(spec, X, X[:, :1])
    assert experiment.run(spec.as_dict(X[2])).tolist() == X[2, :1].tolist()


def test_csv_round_trips_labels():
    spec = _spec(domain=[
        ContinuousDomain(name="x", lower=0, upper=1),
        EnumeratedDomain(name="solvent", values=["water", "acetic acid, glacial"]),
    ])
    h = History()
    h.append(Record(candidate=torch.tensor([0.2, 1.0], dtype=torch.double), observation=None,
                    outputs=torch.tensor([1.5], dtype=torch.double)))
    flat = history_to_csv(h, spec)
    assert "acetic acid, glacial" in flat
    assert history_from_csv(flat, spec).records[0].candidate.tolist() == [0.2, 1.0]
    spec2, h2 = spec_history_from_csv(spec_history_to_csv(spec, h))
    assert spec2.domain[1].values == ["water", "acetic acid, glacial"]
    assert h2.records[0].candidate.tolist() == [0.2, 1.0]


def test_vocs_round_trip_keeps_labels_in_a_stable_order():
    pytest.importorskip("gest_api")
    from abbot.interop.xopt import spec_to_vocs, vocs_to_spec

    spec = vocs_to_spec(spec_to_vocs(_spec()))
    assert spec.domain[1].values == sorted(SOLVENTS)
