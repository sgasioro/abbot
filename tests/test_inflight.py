"""Outstanding candidates carry ids, accumulate across suggest() calls, and survive a restart."""

import json

import pytest
import torch

from abbot import Campaign, ContinuousDomain, DataStore, OutputSpec, ProblemSpec


def _spec() -> ProblemSpec:
    return ProblemSpec(
        domain=[ContinuousDomain(name="temp", lower=20.0, upper=80.0)],
        outputs=[OutputSpec(name="yield", direction="MAXIMIZE")],
    )


SEEDS = torch.tensor([[30.0], [40.0], [50.0], [60.0], [70.0]], dtype=torch.double)


def _seeded(**kw) -> Campaign:
    c = Campaign(_spec(), **kw)
    c.seed(SEEDS)
    return c


def test_suggest_returns_ids_and_accumulates():
    c = _seeded()
    a = c.suggest(2)
    b = c.suggest(1)
    assert [x.id for x in a + b] == ["c0000", "c0001", "c0002"]
    assert a[0] == {"temp": 30.0}  # still a plain {name: value} dict
    assert [x.id for x in c.outstanding()] == ["c0000", "c0001", "c0002"]
    st = c.state()
    assert st.pending_ids == ["c0000", "c0001", "c0002"]
    assert st.pending == [{"temp": 30.0}, {"temp": 40.0}, {"temp": 50.0}]
    json.loads(st.to_json())


def test_ingest_by_id_in_any_order():
    c = _seeded()
    a, b = c.suggest(2)
    c.ingest({"yield": 2.0}, id=b.id)
    assert [x.id for x in c.outstanding()] == [a.id]
    c.ingest([1.0], id=a.id)
    assert [r.id for r in c.history.records] == [b.id, a.id]
    assert [r.candidate.item() for r in c.history.records] == [40.0, 30.0]
    assert c.outstanding() == []


def test_ingest_without_ids_covers_all_outstanding_in_order():
    c = _seeded()
    c.suggest(1)
    c.suggest(1)
    with pytest.raises(ValueError, match="2 outstanding"):
        c.ingest([1.0])
    c.ingest([[1.0], [2.0]])
    assert c.history.outputs.reshape(-1).tolist() == [1.0, 2.0]


def test_ingest_ids_and_validation_is_all_or_nothing():
    c = _seeded()
    a, b = c.suggest(2)
    with pytest.raises(ValueError):
        c.ingest([{"yield": 1.0}, {"wrong": 2.0}], ids=[a.id, b.id])
    assert len(c.history) == 0 and len(c.outstanding()) == 2
    c.ingest([{"yield": 1.0}, {"yield": 2.0}], ids=[b.id, a.id])
    assert [r.outputs.item() for r in c.history.records] == [1.0, 2.0]


def test_unknown_id_is_a_clear_error():
    c = _seeded()
    (a,) = c.suggest(1)
    c.ingest([1.0], id=a.id)
    with pytest.raises(KeyError, match="not outstanding"):
        c.ingest([1.0], id=a.id)
    with pytest.raises(RuntimeError, match="nothing outstanding"):
        c.ingest([1.0])


def test_fail_retries_with_a_new_id_then_halts():
    c = _seeded(max_retries=1)
    (a,) = c.suggest(1)
    c.fail(a.id, "vial cracked")
    rec = c.history.records[0]
    assert rec.fault and rec.error == "vial cracked" and rec.id == a.id
    (retry,) = c.suggest(1)
    assert retry == a and retry.id != a.id
    c.fail(retry.id, RuntimeError("cracked again"))
    assert c.history.records[1].retry_of == 0
    assert "RuntimeError: cracked again" in c.halted


def test_cancel_drops_without_recording():
    c = _seeded()
    a, b = c.suggest(2)
    c.cancel(a.id)
    assert len(c.history) == 0
    assert [x.id for x in c.outstanding()] == [b.id]


def test_record_closes_a_matching_outstanding_design():
    c = _seeded()
    a, b = c.suggest(2)
    c.record({"temp": 40.0}, {"yield": 3.0})  # an external evaluator hands back b
    assert [x.id for x in c.outstanding()] == [a.id]
    assert c.history.records[0].id == b.id
    c.record({"temp": 33.0}, {"yield": 1.0})  # chosen elsewhere: new id, a untouched
    assert c.history.records[1].id == "c0002"
    assert [x.id for x in c.outstanding()] == [a.id]


def test_outstanding_retries_and_seeds_survive_a_restart(tmp_path):
    store = DataStore(tmp_path / "run")
    c = _seeded(store=store, max_retries=1)
    a, b, d = c.suggest(3)
    c.ingest([1.0], id=a.id)
    c.fail(b.id, "lost")  # queues a retry

    r = Campaign(_spec(), store=store, max_retries=1)
    r.seed(torch.tensor([[25.0]], dtype=torch.double))  # ignored: resumed
    assert [x.id for x in r.outstanding()] == [d.id]
    assert r.outstanding()[0] == {"temp": 50.0}
    assert r.pending_count() == 1 + 2  # the retry + the two unissued seeds
    (retry, s4) = r.suggest(2)
    assert retry == {"temp": 40.0} and s4 == {"temp": 60.0}
    assert retry.id == "c0003"  # numbering continues
    r.ingest([2.0], id=d.id)
    assert r.history.records[-1].id == d.id


def test_seeds_survive_a_crash_before_any_result(tmp_path):
    store = DataStore(tmp_path / "run")
    _seeded(store=store)
    r = Campaign(_spec(), store=store)
    assert r.pending_count() == len(SEEDS)


def test_record_written_but_queue_not_is_not_rerun(tmp_path):
    store = DataStore(tmp_path / "run")
    c = _seeded(store=store)
    (a,) = c.suggest(1)
    saved = store.load_queue()
    c.ingest([1.0], id=a.id)
    store.save_queue(saved)  # simulate a crash between the two writes
    r = Campaign(_spec(), store=store)
    assert r.outstanding() == []


class _Exp:
    def __init__(self):
        self.seen: list[dict] = []

    def run(self, design):
        self.seen.append(design)
        return [design["temp"] / 10.0]


def test_step_finishes_an_outstanding_candidate_first(tmp_path):
    store = DataStore(tmp_path / "run")
    c = _seeded(store=store)
    (a,) = c.suggest(1)  # handed out, then the process died mid-experiment

    r = Campaign(_spec(), store=store)
    exp = _Exp()
    r.step(exp, lambda o: o)
    assert exp.seen == [{"temp": 30.0}]
    assert r.history.records[0].id == a.id
