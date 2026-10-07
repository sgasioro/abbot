"""RunController auto/manual sequencing + controls, headless (no browser)."""

import pytest
import torch

from abbot.experiment import ManualExperiment
from abbot.model import default_constructor
from abbot.policy import SimpleBO
from abbot.problem import ContinuousDomain, ObjectiveSpec, ProblemSpec
from abbot.session import Campaign
from abbot.ui.controller import RunController, RunStatus


def _campaign():
    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        objectives=[ObjectiveSpec(name="f", direction="MAXIMIZE")],
    )
    c = Campaign(spec, SimpleBO(), default_constructor(spec, bounds=spec.bounds_tensor()))
    c.seed(torch.tensor([[0.2], [0.5], [0.8]], dtype=torch.double))
    return c


def test_manual_mode_requires_no_experiment():
    ctrl = RunController(_campaign())
    assert ctrl.can_auto is False
    s = ctrl.suggest(1)
    assert len(s) == 1
    ctrl.submit(torch.tensor([0.5]))
    assert ctrl.campaign.history.records[0].outputs is not None


def test_auto_run_stops_at_max_steps():
    exp = ManualExperiment(responder=lambda c: -((c.reshape(-1) - 0.3) ** 2))
    ctrl = RunController(_campaign(), experiment=exp, evaluate=lambda o: o, max_steps=6)
    assert ctrl.can_auto is True
    ctrl.run()
    assert len(ctrl.campaign.history) == 6
    assert ctrl.status == RunStatus.DONE


def test_should_continue_gate_pauses():
    exp = ManualExperiment(responder=lambda c: -((c.reshape(-1) - 0.3) ** 2))
    ctrl = RunController(_campaign(), experiment=exp, evaluate=lambda o: o, max_steps=100)
    # Gate allows exactly 3 steps then pauses.
    budget = {"n": 3}

    def gate():
        if budget["n"] <= 0:
            return False
        budget["n"] -= 1
        return True

    ctrl.run(should_continue=gate)
    assert len(ctrl.campaign.history) == 3
    assert ctrl.status == RunStatus.PAUSED


def test_auto_run_refused_without_experiment():
    ctrl = RunController(_campaign())
    with pytest.raises(RuntimeError, match="auto mode"):
        ctrl.run()
    with pytest.raises(RuntimeError, match="requires a programmatic"):
        ctrl.step_once()


def test_manual_round_feeds_progress_analysis():
    """The exact wiring the UI depends on: manual suggest -> submit several rounds,
    then convergence() reflects progress. (The browser layer is thin over this.)
    """
    from abbot.analysis import convergence

    torch.manual_seed(0)
    opt = 0.3
    ctrl = RunController(_campaign())
    spec = ctrl.campaign.spec
    for _ in range(8):
        cand = ctrl.suggest(1)[0]
        x = cand["x"]
        ctrl.submit(torch.tensor([-((x - opt) ** 2)], dtype=torch.double))
    series = convergence(ctrl.campaign.history, spec)
    assert len(series) == len(ctrl.campaign.history)
    # Running best is non-decreasing for a MAXIMIZE objective.
    assert all(series[i + 1] >= series[i] - 1e-9 for i in range(len(series) - 1))
