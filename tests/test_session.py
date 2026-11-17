"""The Campaign session API: ask/tell parity, validation, persistence, hooks."""

import json

import pytest
import torch

from abbot.experiment import ManualExperiment
from abbot.history import DataStore
from abbot.model import (
    ModelListConstructor,
    default_constructor,
)
from abbot.policy import ParetoBO, SimpleBO
from abbot.problem import ContinuousDomain, ObjectiveSpec, ProblemSpec
from abbot.session import Campaign

pytestmark = pytest.mark.slow  # GP fits via Campaign suggest/ingest


def _scalar_spec():
    return ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        objectives=[ObjectiveSpec(name="f", direction="MAXIMIZE")],
    )


def _make_campaign(spec, store=None, on_step=None):
    return Campaign(
        spec,
        SimpleBO() if spec.n_objectives == 1 else ParetoBO(),
        default_constructor(spec, bounds=spec.bounds_tensor()),
        store=store,
        on_step=on_step,
    )


def test_suggest_ingest_recovers_optimum():
    torch.manual_seed(0)
    opt = 0.3
    spec = _scalar_spec()
    c = _make_campaign(spec)
    c.seed(torch.tensor([[0.05], [0.5], [0.95]], dtype=torch.double))

    for _ in range(15):
        s = c.suggest(1)
        assert set(s[0].keys()) == {"x"}
        x = s[0]["x"]
        c.ingest(-((x - opt) ** 2))  # bare scalar accepted for one pending candidate

    rec = c.recommend()
    assert len(rec) == 1
    assert abs(float(c.history.records[rec[0]].candidate) - opt) < 0.12


def test_ingest_validates_width():
    spec = _scalar_spec()
    c = _make_campaign(spec)
    c.suggest(1)
    with pytest.raises(ValueError, match="output"):
        c.ingest(torch.tensor([1.0, 2.0]))  # 2 cols, spec declares 1


def test_minimize_handled_canonically():
    torch.manual_seed(0)
    minimizer = 0.4
    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        objectives=[ObjectiveSpec(name="cost", direction="MINIMIZE")],
    )
    c = _make_campaign(spec)
    c.seed(torch.tensor([[0.1], [0.6], [0.9]], dtype=torch.double))
    for _ in range(12):
        x = c.suggest(1)[0]["x"]
        c.ingest(torch.tensor([(x - minimizer) ** 2 + 0.5]))  # raw positive cost
    rec = c.recommend()
    best = c.history.records[rec[0]]
    assert abs(float(best.candidate) - minimizer) < 0.15
    assert float(best.outputs) > 0.0  # reported raw, not negated


def test_multiobjective_recommend_non_dominated():
    torch.manual_seed(0)
    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        objectives=[
            ObjectiveSpec(name="f1", direction="MAXIMIZE"),
            ObjectiveSpec(name="f2", direction="MAXIMIZE"),
        ],
    )
    c = Campaign(
        spec, ParetoBO(), ModelListConstructor(bounds=spec.bounds_tensor())
    )
    c.seed(torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.double))
    for _ in range(10):
        x = c.suggest(1)[0]["x"]
        c.ingest(torch.tensor([x, 1.0 - x]))
    rec = c.recommend()
    assert len(rec) >= 2


def test_step_cycle_parity_and_persistence(tmp_path):
    torch.manual_seed(0)
    spec = _scalar_spec()
    opt = 0.3
    exp = ManualExperiment(responder=lambda d: -((d["x"] - opt) ** 2))

    c = _make_campaign(spec, store=DataStore(tmp_path))
    c.seed(torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.double))
    for _ in range(5):
        c.step(exp, lambda obs: obs)
    assert len(c.history) == 5

    # Resume from disk into a fresh campaign.
    c2 = _make_campaign(spec, store=DataStore(tmp_path))
    assert len(c2.history) == 5
    c2.seed(torch.tensor([[0.2]], dtype=torch.double))  # ignored on resume
    c2.step(exp, lambda obs: obs)
    assert len(c2.history) == 6


def test_state_round_trips_to_json():
    torch.manual_seed(0)
    spec = _scalar_spec()
    c = _make_campaign(spec)
    c.seed(torch.tensor([[0.2], [0.8]], dtype=torch.double))
    c.step(
        ManualExperiment(responder=lambda d: -(d["x"] ** 2)),
        lambda obs: obs,
    )
    state = c.state()
    restored = json.loads(state.to_json())
    assert restored["n_steps"] == 1
    assert restored["variable_names"] == ["x"]
    assert restored["objective_names"] == ["f"]


def test_on_step_hook_fires():
    torch.manual_seed(0)
    spec = _scalar_spec()
    seen = []

    def hook(state, model):
        seen.append(state.n_steps)

    c = _make_campaign(spec, on_step=hook)
    c.seed(torch.tensor([[0.2], [0.8]], dtype=torch.double))
    c.step(ManualExperiment(responder=lambda d: -(d["x"] ** 2)), lambda obs: obs)
    c.step(ManualExperiment(responder=lambda d: -(d["x"] ** 2)), lambda obs: obs)
    assert seen == [1, 2]


# --- API-freeze additions: current_model cache, record(), pending_count --------


def test_current_model_caches_until_history_mutates():
    torch.manual_seed(0)
    spec = _scalar_spec()
    c = _make_campaign(spec)
    c.seed(torch.tensor([[0.2], [0.5], [0.8]], dtype=torch.double))
    for _ in range(3):  # drain seeds into history
        x = c.suggest(1)[0]["x"]
        c.ingest(torch.tensor([-((x - 0.3) ** 2)]))
    m1 = c.current_model()
    assert m1 is not None
    assert c.current_model() is m1  # cached: same object, no refit
    # A new measurement invalidates the cache -> a fresh fit.
    x = c.suggest(1)[0]["x"]
    c.ingest(torch.tensor([-((x - 0.3) ** 2)]))
    assert c.current_model() is not m1


def test_record_matches_ingest_and_persists(tmp_path):
    torch.manual_seed(0)
    spec = _scalar_spec()
    c = _make_campaign(spec, store=DataStore(tmp_path))
    # record() folds an externally-chosen (x, y) in with no prior suggest().
    c.record(torch.tensor([0.3]), torch.tensor([0.9]))
    c.record(torch.tensor([0.7]), torch.tensor([0.1]))
    assert len(c.history) == 2
    assert abs(float(c.history.records[0].candidate) - 0.3) < 1e-6
    assert abs(float(c.history.records[0].outputs) - 0.9) < 1e-6
    # Persisted: a fresh campaign on the same store resumes both.
    c2 = _make_campaign(spec, store=DataStore(tmp_path))
    assert len(c2.history) == 2


def test_pending_count_tracks_seed_queue():
    spec = _scalar_spec()
    c = _make_campaign(spec)
    assert c.pending_count() == 0
    c.seed(torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.double))
    assert c.pending_count() == 3
    c.suggest(1)
    c.ingest(torch.tensor([0.5]))
    assert c.pending_count() == 2  # one seed consumed


def test_misconfig_raises_not_silent_random():
    """A policy that can't serve the spec must fail loud on suggest, not silently
    fall back to random draws (the hybrid fail-mode guarantee)."""
    from abbot.policy import ParetoBO

    spec = _scalar_spec()  # single objective
    c = Campaign(
        spec, ParetoBO(), default_constructor(spec, bounds=spec.bounds_tensor())
    )
    c.seed(torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.double))
    for _ in range(3):  # drain seeds so the next suggest asks the policy
        x = c.suggest(1)[0]["x"]
        c.ingest(torch.tensor([x]))
    with pytest.raises(ValueError, match="ParetoBO"):
        c.suggest(1)  # ParetoBO on a 1-objective spec -> loud, not random


# --- what a failing hook, fit or proposal does ------------------------------


class _Broken(ModelListConstructor):
    def build(self, X, Y, Yvar=None):
        raise RuntimeError("Cholesky blew up")


def _seeded(**kwargs):
    spec = _scalar_spec()
    c = Campaign(spec, SimpleBO(), _Broken(), initial_designs=torch.tensor(
        [[0.2], [0.5], [0.8]], dtype=torch.double), **kwargs)
    for _ in range(3):
        c.suggest(1)
        c.ingest(0.5)
    return c


def test_a_failed_fit_is_a_random_probe_that_says_so_or_raises():
    torch.manual_seed(0)
    c = _seeded()
    assert len(c.suggest(1)) == 1  # the run goes on, on a random design
    assert "The model fit failed (RuntimeError: Cholesky blew up)" in c.last_warning
    strict = _seeded(on_model_error="raise")
    with pytest.raises(RuntimeError, match="Cholesky blew up"):
        strict.suggest(1)
    assert strict.outstanding() == []  # nothing handed out


def test_a_failed_proposal_raises_when_asked():
    class Flaky(SimpleBO):
        def suggest(self, *args, **kwargs):
            raise RuntimeError("NaN in the acquisition")

    torch.manual_seed(0)
    spec = _scalar_spec()
    seeds = torch.tensor([[0.2], [0.5], [0.8]], dtype=torch.double)
    for mode, ok in (("probe", True), ("raise", False)):
        c = Campaign(spec, Flaky(), initial_designs=seeds, on_model_error=mode)
        for _ in range(3):
            c.suggest(1)
            c.ingest(float(torch.rand(1)))
        if ok:
            assert len(c.suggest(1)) == 1 and "transient numerical" in c.last_warning
        else:
            with pytest.raises(RuntimeError, match="NaN in the acquisition"):
                c.suggest(1)


def test_a_hook_error_is_logged_or_raised_after_the_result_is_saved(tmp_path, caplog):
    def hook(state, model):
        raise KeyError("plot")

    torch.manual_seed(0)
    spec = _scalar_spec()
    seeds = torch.tensor([[0.2], [0.8]], dtype=torch.double)
    c = Campaign(spec, on_step=hook, initial_designs=seeds)
    c.suggest(1)
    c.ingest(0.1)  # logged; the campaign carries on
    assert len(c.history) == 1 and "on_step hook failed" in caplog.text

    store = DataStore(tmp_path / "s")
    c = Campaign(spec, on_step=hook, initial_designs=seeds, store=store,
                 hook_errors="raise")
    c.suggest(1)
    with pytest.raises(KeyError, match="plot"):
        c.ingest(0.1)
    assert len(c.history) == 1 and len(store.load()) == 1  # recorded and saved first


@pytest.mark.parametrize(("kwargs", "words"), [
    ({"hook_errors": "ignore"}, "hook_errors must be one of"),
    ({"on_model_error": "retry"}, "on_model_error must be one of"),
])
def test_bad_error_modes_are_refused(kwargs, words):
    with pytest.raises(ValueError, match=words):
        Campaign(_scalar_spec(), **kwargs)
