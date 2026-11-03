"""Apparatus faults (retry, then a person decides) vs. bad regions (the model learns)."""

import pytest
import torch

from abbot import (
    Campaign,
    ContinuousDomain,
    DataStore,
    OutputSpec,
    ProblemSpec,
    Supervisor,
    Verdict,
)
from abbot.goal import TopK
from abbot.history import History, Record
from abbot.loop import drive
from abbot.policy import BAXPolicy, SimpleBO, UncertaintySampling
from abbot.session import PersistentFaultError, recommend
from abbot.supervision import consecutive_fault_monitor, per_shot, target_reached
from abbot.usability import fit_usability

DARK = 0.6  # runs with a > DARK give no usable result


def _spec() -> ProblemSpec:
    return ProblemSpec(
        domain=[
            ContinuousDomain(name="a", lower=0.0, upper=1.0),
            ContinuousDomain(name="b", lower=0.0, upper=1.0),
        ],
        outputs=[OutputSpec(name="f", direction="MAXIMIZE")],
    )


def _f(design: dict) -> float:
    """Best at a = 1, inside the dark region."""
    return design["a"] - (design["b"] - 0.5) ** 2


class _Dark:
    """A flat (zero) profile when a > DARK; else the value."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def run(self, design: dict) -> list[float]:
        self.calls.append(dict(design))
        return [0.0] if design["a"] > DARK else [_f(design)]


class _Failing:
    """Raises on every call (a broken instrument), or on the first ``n``."""

    def __init__(self, n: int = 10**6) -> None:
        self.n, self.calls = n, []

    def run(self, design: dict) -> list[float]:
        self.calls.append(dict(design))
        if len(self.calls) <= self.n:
            raise TimeoutError("serial port timed out")
        return [_f(design)]


def _flat(observation) -> bool:
    return float(observation[0]) == 0.0


SEEDS = torch.tensor([[0.2, 0.3], [0.5, 0.8], [0.3, 0.6]], dtype=torch.double)


def _campaign(**kw) -> Campaign:
    kw.setdefault("initial_designs", SEEDS)
    return Campaign(_spec(), **kw)


# -- apparatus faults: retries, then a person --


@pytest.mark.parametrize("max_retries", [0, 1, 2])
def test_a_run_that_keeps_failing_halts_after_max_retries(max_retries):
    c = _campaign(max_retries=max_retries)
    exp = _Failing()
    for _ in range(max_retries + 1):
        assert c.halted is None
        c.step(exp)
    assert c.halted is not None and f"{max_retries + 1} time(s)" in c.halted
    assert "resolve()" in c.halted
    assert len({tuple(d.values()) for d in exp.calls}) == 1  # the same design each time
    assert c.halted_on == [exp.calls[0]]
    assert c.state().halted_on == [exp.calls[0]]


@pytest.mark.parametrize("max_retries", [0, 2])
def test_a_flagged_shot_halts_once_its_retries_are_used(max_retries):
    # Before, a shot flagged again was dropped and the campaign moved on.
    c = _campaign(supervisor=per_shot(_flat), max_retries=max_retries,
                  initial_designs=torch.tensor([[0.9, 0.5]], dtype=torch.double))
    for _ in range(max_retries + 1):
        c.step(_Dark())
    assert "rejected the run" in c.halted
    assert c.halted_on == [{"a": 0.9, "b": 0.5}]
    assert all(r.fault for r in c.history.records)


def test_resolve_retry_runs_the_design_next_with_fresh_retries():
    c = _campaign(max_retries=1)
    c.step(_Failing())
    c.step(_Failing())
    halted_at = c.halted_on[0]
    c.resolve("retry")
    assert c.halted is None and c.halted_on == []
    exp = _Failing(n=1)  # one more failure, then fixed
    c.step(exp)
    assert c.halted is None  # the retry count started over
    c.step(exp)
    assert exp.calls == [halted_at, halted_at]
    assert c.history.records[-1].usable and c.history.records[-1].retry_of == 0


def test_rebuilding_on_the_store_resumes_as_retry(tmp_path):
    store = DataStore(tmp_path / "run")
    c = _campaign(store=store, max_retries=0)
    with pytest.raises(PersistentFaultError):
        drive(c, _Failing(), n_total=5)
    halted_at = c.halted_on[0]
    resumed = Campaign(_spec(), store=store)
    assert resumed.halted is None  # rebuilding means the cause is fixed
    exp = _Failing(n=0)
    resumed.step(exp)
    assert exp.calls == [halted_at]  # before, this design was lost


def test_resolve_exclude_skips_the_design_and_steers_away():
    seen = []

    class Spy(SimpleBO):
        def suggest(self, *args, pending=None, **kw):
            seen.append(pending)
            return super().suggest(*args, pending=pending, **kw)

    c = _campaign(policy=Spy(), max_retries=0)
    exp = _Failing(n=1)
    c.step(exp)
    c.resolve("exclude")
    assert c.halted is None and c.pending_count() == 2  # the other seeds, no retry
    assert c.history.records[0].fault  # stays a fault
    for _ in range(3):
        c.step(exp)
    assert exp.calls[0] not in exp.calls[1:]
    c.suggest(1)  # the policy now proposes; the excluded design counts as in flight
    excluded = torch.tensor(list(exp.calls[0].values()), dtype=torch.double)
    assert any(torch.allclose(row, excluded) for row in seen[-1])


def test_excluded_designs_survive_a_resume(tmp_path):
    store = DataStore(tmp_path / "run")
    c = _campaign(store=store, max_retries=0)
    c.step(_Failing())
    c.resolve("exclude")
    resumed = Campaign(_spec(), store=store)
    assert len(resumed._excluded) == 1 and resumed.pending_count() == 2


def test_resolve_bad_region_turns_the_runs_into_bad_region_runs(tmp_path):
    store = DataStore(tmp_path / "run")
    c = _campaign(store=store, supervisor=per_shot(_flat), max_retries=1,
                  initial_designs=torch.tensor([[0.9, 0.5], [0.2, 0.2]], dtype=torch.double))
    c.step(_Dark())
    c.step(_Dark())
    c.resolve("bad_region", reason="dark: flat profile")
    first, retry = c.history.records
    for r in (first, retry):
        assert not r.fault and r.bad_region == "dark: flat profile" and not r.usable
    assert len(c.history.clean) == 0 and len(c.history.judged) == 2
    assert c.pending_count() == 1  # not retried
    # Persisted: a resumed campaign sees bad-region runs.
    resumed = Campaign(_spec(), store=store)
    assert [r.bad_region for r in resumed.history.records] == ["dark: flat profile"] * 2


def test_resolve_refusals():
    c = _campaign()
    with pytest.raises(RuntimeError, match="isn't halted"):
        c.resolve()
    stall = Campaign(_spec(), supervisor=lambda h: Verdict(halt="stuck"),
                     initial_designs=SEEDS)
    stall.step(_Dark())
    with pytest.raises(ValueError, match="isn't about a design"):
        stall.resolve("exclude")
    with pytest.raises(ValueError, match="one of"):
        stall.resolve("ignore")
    stall.resolve("retry")
    assert stall.halted is None
    c = _campaign(max_retries=0)
    c.step(_Failing())
    with pytest.raises(ValueError, match="reason= goes with 'bad_region'"):
        c.resolve("exclude", reason="x")


def test_drive_continue_excludes_the_failed_design():
    c = _campaign(max_retries=0)
    exp = _Failing(n=1)
    drive(c, exp, n_total=4, on_persistent_fault="continue")
    assert len(c.history) == 4 and c.halted is None
    assert exp.calls[0] not in exp.calls[1:]
    assert len(c._excluded) == 1


# -- bad regions: a verdict --


def test_a_bad_region_verdict_is_recorded_not_retried():
    c = _campaign(supervisor=per_shot(_flat, bad_region="dark: flat profile"),
                  initial_designs=torch.tensor([[0.9, 0.5], [0.2, 0.2]], dtype=torch.double))
    c.step(_Dark())
    r = c.history.records[0]
    assert r.bad_region == "dark: flat profile" and not r.fault and not r.usable
    assert c.halted is None and c.pending_count() == 1  # no retry queued
    assert c.state().bad_regions == ["dark: flat profile"]
    c.step(_Dark())
    assert len(c.history.clean) == 1 and len(c.history.judged) == 2


def test_bad_region_true_gets_a_generic_reason():
    c = _campaign(supervisor=lambda h: Verdict(bad_region=True, retry=True))
    c.step(_Dark())
    r = c.history.records[0]
    assert r.bad_region == "bad region" and not r.fault  # outranks the retry
    assert c.pending_count() == 2


def test_stop_rules_ignore_bad_region_runs():
    # The dark run's flat 0.0 would otherwise "reach" a target of <= 0.
    spec = ProblemSpec(
        domain=_spec().domain, outputs=[OutputSpec(name="f", direction="MINIMIZE")]
    )
    c = Campaign(
        spec,
        supervisor=Supervisor(faults=per_shot(_flat, bad_region=True), stops=target_reached(0.0)),
        initial_designs=torch.tensor([[0.9, 0.5]], dtype=torch.double),
    )
    c.step(_Dark())
    assert c.stopped is None


def test_consecutive_fault_monitor_ignores_bad_region_runs():
    monitor = consecutive_fault_monitor(_flat, k=2)
    records = [
        Record(candidate=torch.zeros(2), observation=[0.0], bad_region="dark"),
        Record(candidate=torch.zeros(2), observation=[0.0], fault=True),
    ]
    assert monitor(History(records)).halt is None


# -- bad regions: the usability model --


def _with_dark_runs(n: int = 16, seed: int = 0) -> History:
    torch.manual_seed(seed)
    X = torch.rand(n, 2, dtype=torch.double)
    return History([
        Record(
            candidate=x, observation=None,
            outputs=torch.tensor([0.0 if x[0] > DARK else _f({"a": x[0], "b": x[1]})]),
            bad_region="dark" if x[0] > DARK else None,
        )
        for x in X
    ])


def test_usability_learns_the_dark_region():
    u = fit_usability(_with_dark_runs(), _spec())
    p = u(torch.tensor([[0.2, 0.5], [0.9, 0.5]], dtype=torch.double))
    assert p[0] > 0.9 and p[1] < 0.1
    assert fit_usability(History([r for r in _with_dark_runs().records if r.usable]), _spec()) is None


def test_recommend_ranks_only_usable_designs():
    spec = _spec()
    history = _with_dark_runs()
    # Pretend the dark runs were usable with a great value: recommend must still
    # skip designs the usability model calls bad.
    u = fit_usability(history, spec)
    for r in history.records:
        if r.bad_region:
            r.bad_region, r.outputs = None, torch.tensor([5.0])
    best = recommend(history, spec, usability=u, denoise=False)
    assert float(history.records[best[0]].candidate[0]) <= DARK


def _dark_campaign(policy, n_steps=6):
    c = Campaign(
        _spec(), policy,
        supervisor=per_shot(_flat, bad_region="dark"),
        initial_designs=torch.tensor(
            [[0.1, 0.2], [0.3, 0.8], [0.5, 0.5], [0.7, 0.3], [0.8, 0.7], [0.95, 0.5]],
            dtype=torch.double,
        ),
    )
    exp = _Dark()
    drive(c, exp, n_total=6 + n_steps)
    return c, exp.calls[6:]


@pytest.mark.parametrize(
    "policy",
    [SimpleBO(), UncertaintySampling(), BAXPolicy(TopK(k=3, output="f"), "MeanBAX", n_candidates=256)],
    ids=["SimpleBO", "UncertaintySampling", "MeanBAX"],
)
def test_policies_steer_away_from_the_dark_region(policy):
    # Without the usability model, each of these proposes in the dark region nearly
    # every time (SimpleBO repeats one dark design). With it, proposals may refine
    # the edge (weighting, not a hard cut) but rarely go deep into it.
    torch.manual_seed(0)
    c, proposed = _dark_campaign(policy, n_steps=8)
    deep = sum(d["a"] > DARK + 0.15 for d in proposed)
    assert deep <= 2, proposed
    result = c.result()
    assert result.p_usable and result.p_usable[0] >= 0.5
    assert float(c.history.records[result.recommended[0]].candidate[0]) <= DARK
    assert "gave no usable result" in str(c.check())


def test_a_policy_without_usability_is_warned_once(caplog):
    class Plain:
        uq_requirement = "calibrated_variance"

        def suggest(self, history, spec, model, n=1, *, pending=None):
            return spec.sample_designs(n)

        def ingest(self, results):
            return None

    _dark_campaign(Plain(), n_steps=2)
    assert sum("usability" in m for m in caplog.messages) == 1
