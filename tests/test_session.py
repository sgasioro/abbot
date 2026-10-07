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
    with pytest.raises(ValueError, match="objective"):
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
    exp = ManualExperiment(responder=lambda cand: -((cand.reshape(-1) - opt) ** 2))

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
        ManualExperiment(responder=lambda cand: -(cand.reshape(-1) ** 2)),
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
    c.step(ManualExperiment(responder=lambda cand: -(cand.reshape(-1) ** 2)), lambda obs: obs)
    c.step(ManualExperiment(responder=lambda cand: -(cand.reshape(-1) ** 2)), lambda obs: obs)
    assert seen == [1, 2]
