"""Synchronous batches: Campaign.step(n=...), Experiment.run_batch, batch_size=."""

import pytest
import torch

from abbot import (
    Campaign,
    ContinuousDomain,
    OutputSpec,
    ProblemSpec,
    TensorExperiment,
    run_campaign,
)


def _spec() -> ProblemSpec:
    return ProblemSpec(
        domain=[ContinuousDomain(name="temp", lower=20.0, upper=80.0)],
        outputs=[OutputSpec(name="yield", direction="MAXIMIZE")],
    )


def _identity(obs):
    return obs


class _OneAtATime:
    """Only ``run``: the campaign loops over the batch."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def run(self, design):
        self.calls.append(design["temp"])
        return torch.tensor([design["temp"] / 10.0])


class _Plate:
    """``run_batch``: the whole batch in one call. ``fail_slot`` returns an
    exception for that design; ``raise_all`` makes the call itself fail."""

    def __init__(self, fail_slot=None, raise_all=False, short=False) -> None:
        self.batches: list[int] = []
        self.fail_slot, self.raise_all, self.short = fail_slot, raise_all, short

    def run(self, design):
        raise AssertionError("an experiment with run_batch is never run one at a time")

    def run_batch(self, designs):
        self.batches.append(len(designs))
        if self.raise_all:
            raise ConnectionError("plate reader offline")
        out = [torch.tensor([d["temp"] / 10.0]) for d in designs]
        if self.fail_slot is not None:
            out[self.fail_slot] = RuntimeError("well 2 empty")
        return out[:-1] if self.short else out


def _seeded(n=4):
    c = Campaign(_spec())
    c.seed(torch.linspace(25.0, 75.0, n, dtype=torch.double).reshape(-1, 1))
    return c


def test_step_runs_n_designs_one_at_a_time():
    c, exp = _seeded(), _OneAtATime()
    c.step(exp, _identity, n=3)
    assert exp.calls == pytest.approx([25.0, 25.0 + 50 / 3, 25.0 + 100 / 3])
    assert len(c.history) == 3 and not c.outstanding()
    assert [float(r.outputs) for r in c.history.records] == pytest.approx(
        [x / 10 for x in exp.calls]
    )


def test_step_uses_run_batch_when_defined():
    c, exp = _seeded(), _Plate()
    c.step(exp, _identity, n=3)
    assert exp.batches == [3]
    assert len(c.history) == 3


def test_a_single_design_still_goes_to_run_batch():
    c, exp = _seeded(), _Plate()
    c.step(exp, _identity)
    assert exp.batches == [1] and len(c.history) == 1


class _VectorPlate:
    def __init__(self) -> None:
        self.shapes: list[tuple[int, ...]] = []

    def run(self, x):
        raise AssertionError("run_batch is defined")

    def run_batch(self, X):
        self.shapes.append(tuple(X.shape))
        return list(X / 10.0)


def test_tensor_experiment_forwards_run_batch():
    c, inner = _seeded(), _VectorPlate()
    c.step(TensorExperiment(inner), _identity, n=3)
    assert inner.shapes == [(3, 1)] and len(c.history) == 3
    assert not hasattr(TensorExperiment(_OneAtATime()), "run_batch")


def test_a_failed_slot_fails_only_that_design():
    c, exp = _seeded(), _Plate(fail_slot=1)
    c.step(exp, _identity, n=3)
    faults = [r.fault for r in c.history.records]
    assert faults == [False, True, False]
    assert "well 2 empty" in c.history.records[1].error
    assert c.halted is None  # retried next step (max_retries=1)
    c.step(exp, _identity, n=1)
    assert c.history.records[-1].retry_of == 1


def test_a_raising_batch_fails_every_design():
    c, exp = _seeded(), _Plate(raise_all=True)
    c.step(exp, _identity, n=2)
    assert [r.fault for r in c.history.records] == [True, True]
    assert all("plate reader offline" in r.error for r in c.history.records)


def test_a_short_batch_fails_every_design():
    c, exp = _seeded(), _Plate(short=True)
    c.step(exp, _identity, n=2)
    assert [r.fault for r in c.history.records] == [True, True]
    assert "returned 1 observation(s) for 2" in c.history.records[0].error


def test_pre_run_blocks_the_whole_batch():
    c = Campaign(_spec(), pre_run=lambda d: "too hot" if d["temp"] > 50 else None)
    c.seed(torch.tensor([[30.0], [70.0]], dtype=torch.double))
    exp = _OneAtATime()
    c.step(exp, _identity, n=2)
    assert exp.calls == [] and len(c.history) == 0
    assert "too hot" in c.halted
    assert len(c.outstanding()) == 2


def test_n_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        _seeded().step(_OneAtATime(), _identity, n=0)


@pytest.mark.slow
def test_run_campaign_batches_without_overshooting_the_budget():
    torch.manual_seed(0)
    exp = _Plate()
    seeds = torch.tensor([[25.0], [50.0], [75.0]], dtype=torch.double)
    result = run_campaign(_spec(), exp, n_shots=7, batch_size=3, initial_designs=seeds)
    assert len(result.history) == 10
    assert exp.batches == [3, 3, 3, 1]  # the budget's last design is a batch of one
