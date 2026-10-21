"""Source variables (fidelities and information sources), integer inputs, and
proposals in mixed spaces."""

from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from abbot import (
    BAXPolicy,
    Campaign,
    ContinuousDomain,
    EnumeratedDomain,
    IntegerDomain,
    OutputSpec,
    ProblemSpec,
    SimpleBO,
    Source,
    TopK,
    default_initial_designs,
)
from abbot.acq_opt import Continuous, DiscreteArgmax
from abbot.history import DataStore, History, Record, ResumeMismatchError
from abbot.io import (
    spec_from_json,
    spec_history_from_csv,
    spec_history_to_csv,
    spec_to_json,
)
from abbot.session import recommend


def _spec(target=256, lower=32, upper=512, **source_kw):
    return ProblemSpec(
        domain=[
            ContinuousDomain(name="eps", lower=0.0, upper=1.0),
            ContinuousDomain(name="sigma", lower=0.0, upper=1.0),
        ],
        sources=[Source(domain=IntegerDomain(name="atoms", lower=lower, upper=upper), target=target, **source_kw)],
        outputs=[OutputSpec(name="fit", direction="MAXIMIZE")],
    )


def _plain():
    return ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        outputs=[OutputSpec(name="f", direction="MAXIMIZE")],
    )


# -- the spec -------------------------------------------------------------------


def test_sources_are_inputs_after_the_design_variables():
    spec = _spec()
    assert spec.variable_names == ["eps", "sigma", "atoms"]
    assert spec.design_names == ["eps", "sigma"] and spec.source_names == ["atoms"]
    assert spec.source_indices() == [2] and spec.target_features() == {2: 256.0}
    assert spec.bounds_tensor().tolist() == [[0.0, 0.0, 32.0], [1.0, 1.0, 512.0]]
    X = torch.tensor([[0.1, 0.2, 64.0], [0.3, 0.4, 128.0]], dtype=torch.double)
    assert spec.at_target(X)[:, 2].tolist() == [256.0, 256.0]
    assert X[:, 2].tolist() == [64.0, 128.0]  # a copy


def test_without_sources_the_views_coincide():
    spec = _plain()
    assert spec.design_names == spec.variable_names and spec.source_names == []
    assert spec.target_features() == {}
    X = torch.rand(3, 1, dtype=torch.double)
    assert torch.equal(spec.at_target(X), X)


def test_source_validation():
    with pytest.raises(ValidationError, match="outside"):
        _spec(target=1000)
    with pytest.raises(ValidationError, match="whole number"):
        _spec(target=100.5)
    with pytest.raises(ValidationError, match="positive"):
        _spec(lower=0, log=True)
    with pytest.raises(ValidationError, match="not one of"):
        Source(domain=EnumeratedDomain(name="code", values=[0.0, 1.0]), target=2.0)
    with pytest.raises(ValidationError, match="Duplicate"):
        ProblemSpec(
            domain=[ContinuousDomain(name="atoms", lower=0.0, upper=1.0)],
            sources=[Source(domain=IntegerDomain(name="atoms", lower=1, upper=9), target=4)],
            outputs=[OutputSpec(name="f")],
        )
    assert _spec().sources[0].is_ordered
    unordered = Source(domain=EnumeratedDomain(name="code", values=[0.0, 1.0]), target=1.0)
    assert not unordered.is_ordered


def test_spec_round_trips_and_the_fingerprint_ignores_the_target():
    spec = _spec(log=True)
    assert spec_from_json(spec_to_json(spec)) == spec
    assert spec.fingerprint() == _spec(target=128, log=True).fingerprint()
    assert spec.fingerprint() != _spec(upper=256, target=128, log=True).fingerprint()


def test_sampling_integers_and_log_sources():
    X = _spec(log=True).sample_designs(256, seed=0)
    atoms = X[:, 2]
    assert torch.equal(atoms, atoms.round()) and atoms.min() >= 32 and atoms.max() <= 512
    assert (atoms < 128).float().mean() > 0.4  # log-uniform: half the draws below sqrt(32*512)
    plain = ProblemSpec(
        domain=[IntegerDomain(name="n", lower=1, upper=3)], outputs=[OutputSpec(name="f")]
    )
    assert set(plain.sample_designs(64, seed=0)[:, 0].tolist()) == {1.0, 2.0, 3.0}


def test_initial_designs_include_two_at_the_target():
    X = default_initial_designs(_spec(), seed=0)
    assert len(X) == 7 and (X[:, 2] == 256).sum() >= 2
    assert torch.equal(default_initial_designs(_plain(), seed=0), _plain().sample_designs(3, seed=0))


def test_csv_holds_integers_but_refuses_sources():
    spec = ProblemSpec(
        domain=[IntegerDomain(name="n", lower=1, upper=9), ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        outputs=[OutputSpec(name="f")],
    )
    back, _ = spec_history_from_csv(spec_history_to_csv(spec))
    assert back.domain == spec.domain
    with pytest.raises(ValueError, match="source"):
        spec_history_to_csv(_spec())


def test_resume_survives_a_target_change_but_not_new_bounds(tmp_path):
    c = Campaign(_spec(), store=DataStore(tmp_path))
    c.record({"eps": 0.1, "sigma": 0.2, "atoms": 64}, [0.5])
    Campaign(_spec(target=128), store=DataStore(tmp_path))  # fine
    with pytest.raises(ResumeMismatchError):
        Campaign(_spec(upper=1024), store=DataStore(tmp_path))


# -- proposals ------------------------------------------------------------------


class _Mean(torch.nn.Module):
    """A stand-in acquisition: the sum of the inputs' first column."""

    model = None

    def forward(self, X):
        return X[..., 0].sum(-1)


def test_continuous_rounds_integers_and_honors_fixed_features():
    spec = _spec()
    x = Continuous(num_restarts=2, raw_samples=16).optimize(_Mean(), spec, 1, fixed_features={2: 128.0})
    assert x[0, 2].item() == 128.0
    spec_int = ProblemSpec(domain=[IntegerDomain(name="n", lower=1, upper=9)], outputs=[OutputSpec(name="f")])
    (x,) = Continuous(num_restarts=2, raw_samples=16).optimize(_Mean(), spec_int, 1)
    assert x.item() == 9.0


def test_discrete_argmax_keeps_only_matching_choices():
    choices = torch.tensor([[0.0, 1.0], [1.0, 1.0], [5.0, 2.0]], dtype=torch.double)
    spec = ProblemSpec(
        domain=[EnumeratedDomain(name="a", values=[0.0, 1.0, 5.0])],
        sources=[Source(domain=EnumeratedDomain(name="code", values=[1.0, 2.0]), target=1.0)],
        outputs=[OutputSpec(name="f")],
    )
    (x,) = DiscreteArgmax(choices).optimize(_Mean(), spec, 1, fixed_features={1: 1.0})
    assert x.tolist() == [1.0, 1.0]


def test_a_custom_optimizer_without_fixed_features_still_works_without_sources():
    class _Old:
        def optimize(self, acq, spec, n):
            return torch.full((n, 1), 0.5, dtype=torch.double)

    c = Campaign(_plain(), policy=SimpleBO(_Old()))
    for x in (0.1, 0.5, 0.9):
        c.record({"x": x}, [x])
    assert [cand["x"] for cand in c.suggest(1)] == [0.5]


@pytest.mark.slow
def test_simple_bo_proposes_at_the_target_and_the_experiment_sees_the_source():
    torch.manual_seed(0)
    spec = _spec()
    seen = []

    class _Sim:
        def run(self, design):
            seen.append(dict(design))
            return torch.tensor([-(design["eps"] - 0.3) ** 2 - 1e-4 * design["atoms"]])

    c = Campaign(spec)
    c.seed(default_initial_designs(spec, seed=0))
    for _ in range(9):
        c.step(_Sim(), lambda y: y)
    assert set(seen[0]) == {"eps", "sigma", "atoms"}
    proposed = c.history.records[7:]
    assert proposed and all(r.candidate[2].item() == 256.0 for r in proposed)
    assert not c.last_warning


@pytest.mark.slow
def test_bax_grid_without_source_columns_proposes_at_the_target():
    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        sources=[Source(domain=IntegerDomain(name="n", lower=1, upper=8), target=8)],
        outputs=[OutputSpec(name="y", direction="OBSERVE")],
    )
    grid = torch.linspace(0, 1, 11, dtype=torch.double).reshape(-1, 1)
    policy = BAXPolicy(TopK(k=2, output=0), strategy="MeanBAX", candidates=grid)
    assert policy.candidate_grid(spec).shape == (11, 2)
    c = Campaign(spec, policy=policy)
    for x, n in [(0.0, 1), (0.5, 2), (1.0, 8), (0.3, 8)]:
        c.record({"x": x, "n": n}, [x])
    picks = c.suggest(2)
    assert all(p["n"] == 8.0 for p in picks)


# -- recommendation -------------------------------------------------------------


class _Fake:
    """A model whose posterior mean is the first input column (no uncertainty)."""

    def posterior(self, X):
        return SimpleNamespace(mean=X[..., :1], variance=torch.zeros_like(X[..., :1]))


def _history(rows):
    return History(records=[
        Record(candidate=torch.tensor(x, dtype=torch.double), observation=None,
               outputs=torch.tensor([y], dtype=torch.double))
        for x, y in rows
    ])


def test_recommend_scores_designs_at_the_target_once_each():
    spec = _spec()
    h = _history([
        ([0.9, 0.5, 64.0], 10.0),   # best design, cheap run (looked great there)
        ([0.9, 0.5, 256.0], 0.1),   # the same design at the target
        ([0.2, 0.5, 256.0], 5.0),
    ])
    # Model-scored at target: design 0.9 wins, under its at-target record.
    assert recommend(h, spec, _Fake()) == [1]
    # Observed fallback: only at-target records are comparable.
    assert recommend(h, spec, None, denoise=False) == [2]


def test_without_sources_recommend_is_unchanged():
    h = _history([([0.2], 1.0), ([0.8], 0.0)])
    assert recommend(h, _plain(), _Fake()) == [1]
    assert recommend(h, _plain(), None, denoise=False) == [0]


@pytest.mark.slow
def test_result_reports_run_at_target_and_uncertainty():
    torch.manual_seed(0)
    spec = _spec()
    c = Campaign(spec)
    for x, n in [(0.1, 64), (0.5, 64), (0.9, 128), (0.4, 256), (0.7, 32), (0.2, 512)]:
        c.record({"eps": x, "sigma": 0.5, "atoms": n}, [-(x - 0.6) ** 2])
    result = c.result()
    assert len(result.recommended) == 1
    assert len(result.run_at_target) == 1 and len(result.target_std[0]) == 1
    rec = c.history.records[result.recommended[0]]
    assert result.run_at_target[0] == (rec.candidate[2].item() == 256.0)
