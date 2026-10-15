"""Supervisor monitors: per-shot lifting, composition, stall / drift / consecutive faults."""

import torch

from abbot import Campaign, ContinuousDomain, OutputSpec, ProblemSpec, Verdict
from abbot.history import History, Record
from abbot.supervision import (
    any_of,
    consecutive_fault_monitor,
    default_signal,
    drift_monitor,
    per_shot,
    stall_monitor,
    standard_supervisor,
)

BOUNDS = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)


def _hist(designs, signals, **kw) -> History:
    return History(records=[
        Record(candidate=torch.as_tensor(x, dtype=torch.double),
               observation=torch.as_tensor(s, dtype=torch.double), **kw)
        for x, s in zip(designs, signals)
    ])


def _spread(n):  # designs that move across the box
    return [[(i % 3) / 2, (i % 2)] for i in range(n)]


def _still(n):  # designs that sit on one spot (a converged optimizer)
    return [[0.5 + 1e-3 * i, 0.5] for i in range(n)]


def test_per_shot_sees_only_the_latest_observation():
    seen = []
    h = _hist(_still(3), [[0.0], [1.0], [2.0]])
    v = per_shot(lambda obs: seen.append(float(obs[0])) or True)(h)
    assert seen == [2.0] and v == Verdict(retry=True)
    assert per_shot(lambda obs: True)(History()) == Verdict()


def test_any_of_ors_retry_and_keeps_the_first_halt():
    a = lambda h: Verdict(retry=True)
    b = lambda h: Verdict(halt="first")
    c = lambda h: Verdict(halt="second")
    assert any_of(a, b, c)(History()) == Verdict(retry=True, halt="first")
    assert any_of()(History()) == Verdict()


def test_stall_fires_when_designs_move_but_observations_do_not():
    stuck = _hist(_spread(6), [[1.0, 2.0, 3.0]] * 6)
    healthy = _hist(_spread(6), [[float(i), 2.0 * i, 1.0] for i in range(1, 7)])
    mon = stall_monitor(BOUNDS)
    assert "stall_monitor" in mon(stuck).halt
    assert mon(healthy).halt is None
    assert mon(_hist(_still(6), [[1.0, 2.0, 3.0]] * 6)).halt is None  # converged
    assert mon(_hist(_spread(5), [[1.0, 2.0, 3.0]] * 5)).halt is None  # short window


def test_drift_fires_on_a_monotone_slide_at_a_fixed_design():
    sliding = _hist(_still(8), [[10.0 - i] for i in range(8)])
    noisy = _hist(_still(8), [[float(v)] for v in [3, 1, 4, 1, 5, 9, 2, 6]])
    mon = drift_monitor(BOUNDS)
    assert "drift_monitor" in mon(sliding).halt
    assert mon(noisy).halt is None
    assert mon(_hist(_spread(8), [[10.0 - i] for i in range(8)])).halt is None  # exploring


def test_consecutive_faults_escalate_and_a_clean_shot_resets():
    bad = lambda obs: float(obs[0]) < 0
    mon = consecutive_fault_monitor(bad, k=3)
    assert mon(_hist(_still(3), [[-1.0]] * 3)).halt
    assert mon(_hist(_still(3), [[-1.0], [1.0], [-1.0]])).halt is None
    # a run that raised counts as rejected
    h = _hist(_still(2), [[-1.0]] * 2)
    h.append(Record(candidate=torch.zeros(2), observation=None, fault=True, error="boom"))
    assert mon(h).halt


def test_default_signal_falls_back_to_outputs():
    r = Record(candidate=torch.zeros(1), observation={"obs": 1}, outputs=torch.tensor([2.0]))
    assert default_signal(r).tolist() == [2.0]
    r = Record(candidate=torch.zeros(1), observation=[1.0, 2.0])
    assert default_signal(r).tolist() == [1.0, 2.0]


def test_a_campaign_halts_on_a_stall_and_resumes(tmp_path):
    from abbot import DataStore

    spec = ProblemSpec(
        domain=[ContinuousDomain(name=n, lower=0.0, upper=1.0) for n in ("a", "b")],
        outputs=[OutputSpec(name="y", direction="MAXIMIZE")],
    )
    sup = standard_supervisor(spec.bounds_tensor(), lambda obs: False)
    c = Campaign(spec, store=DataStore(tmp_path), supervisor=sup)
    for x in _spread(6):
        c.record(dict(zip("ab", x)), [1.0], observation=torch.tensor([1.0, 2.0]))
    assert "stall_monitor" in c.halted
    assert len(Campaign(spec, store=DataStore(tmp_path)).history) == 6
