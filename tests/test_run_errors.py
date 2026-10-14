"""Unattended-run safety: experiment.run errors retry then halt; pre_run can block."""

import pytest
import torch

from abbot import (
    Campaign,
    ContinuousDomain,
    DataStore,
    OutputSpec,
    ProblemSpec,
    run_campaign,
)
from abbot.session import PersistentFaultError


def _spec() -> ProblemSpec:
    return ProblemSpec(
        domain=[ContinuousDomain(name="temp", lower=20.0, upper=80.0)],
        outputs=[OutputSpec(name="yield", direction="MAXIMIZE")],
    )


class _Flaky:
    """Raises on the first ``n_failures`` calls, then returns a fixed yield."""

    def __init__(self, n_failures: int) -> None:
        self.n_failures = n_failures
        self.calls: list[torch.Tensor] = []

    def run(self, candidate: torch.Tensor) -> torch.Tensor:
        self.calls.append(candidate.clone())
        if len(self.calls) <= self.n_failures:
            raise TimeoutError("serial port timed out")
        return torch.tensor([1.0])


def _identity(obs):
    return obs


def _seeded(**kw) -> Campaign:
    c = Campaign(_spec(), **kw)
    c.seed(torch.tensor([[30.0], [50.0]], dtype=torch.double))
    return c


def test_transient_run_error_is_recorded_and_retried():
    c = _seeded()
    exp = _Flaky(n_failures=1)
    c.step(exp, _identity)
    assert c.halted is None
    failed = c.history.records[0]
    assert failed.fault and failed.outputs is None
    assert "TimeoutError" in failed.error
    assert "retrying" in c.last_warning

    c.step(exp, _identity)  # the retry runs the same candidate
    assert torch.equal(exp.calls[0], exp.calls[1])
    ok = c.history.records[1]
    assert not ok.fault and ok.retry_of == 0 and ok.error is None
    assert len(c.history.clean) == 1


def test_persistent_run_error_halts():
    c = _seeded(max_retries=1)
    exp = _Flaky(n_failures=10)
    c.step(exp, _identity)
    c.step(exp, _identity)
    assert c.halted is not None
    assert "2 time(s)" in c.halted and "serial port timed out" in c.halted
    with pytest.raises(RuntimeError, match="halted"):
        c.suggest(1)


def test_run_campaign_raises_and_resumes_after_fault(tmp_path):
    store = DataStore(tmp_path / "run")
    seeds = torch.tensor([[30.0], [50.0]], dtype=torch.double)
    with pytest.raises(PersistentFaultError, match="serial port"):
        run_campaign(_spec(), _Flaky(n_failures=10), initial_designs=seeds, store=store)
    resumed = Campaign(_spec(), store=store)
    assert len(resumed.history) == 2
    assert all(r.fault and r.error for r in resumed.history.records)
    assert resumed.halted is None  # a fresh process starts un-halted


def test_pre_run_check_blocks_before_running():
    seen: list[dict] = []

    def check(candidate: dict) -> str | None:
        seen.append(candidate)
        return "too hot" if candidate["temp"] > 40.0 else None

    c = _seeded(pre_run=check)
    exp = _Flaky(n_failures=0)
    c.step(exp, _identity)  # temp=30 passes
    assert len(exp.calls) == 1 and c.halted is None

    state = c.step(exp, _identity)  # temp=50 is blocked
    assert len(exp.calls) == 1  # never ran
    assert "too hot" in c.halted
    assert state.pending == [{"temp": 50.0}]  # left for a human to inspect
    assert seen == [{"temp": 30.0}, {"temp": 50.0}]


def test_evaluate_errors_propagate():
    c = _seeded()

    def bad_evaluate(obs):
        raise ValueError("analysis bug")

    with pytest.raises(ValueError, match="analysis bug"):
        c.step(_Flaky(n_failures=0), bad_evaluate)
