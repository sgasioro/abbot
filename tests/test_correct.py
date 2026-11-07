"""Campaign.correct: fixing a value reported wrong, kept for the record."""

import pytest
import torch

from abbot import Campaign, ProblemSpec
from abbot.history import DataStore

SPEC = ProblemSpec.model_validate({
    "domain": [{"kind": "continuous", "name": "x", "lower": 0, "upper": 4}],
    "outputs": [{"name": "y", "direction": "MAXIMIZE"},
                {"name": "z", "direction": "OBSERVE"}],
})


def _campaign(tmp_path):
    c = Campaign(SPEC, store=DataStore(tmp_path / "store"))
    for x in (0.5, 1.5, 2.5):
        c.record({"x": x}, {"y": x, "z": -x})
    return c


def test_correct_by_id_changes_only_the_named_output_and_keeps_the_old(tmp_path):
    c = _campaign(tmp_path)
    run = c.history.records[0].id
    c.correct(run, {"y": 0.65}, reason="typed wrong")
    r = c.history.records[0]
    assert r.outputs.tolist() == [0.65, -0.5]
    (old,) = r.corrections
    assert old["outputs"] == [0.5, -0.5] and old["reason"] == "typed wrong"


def test_a_correction_is_saved_and_survives_a_restart(tmp_path):
    c = _campaign(tmp_path)
    c.correct(1, [9.0, 9.5])
    again = Campaign(SPEC, store=DataStore(tmp_path / "store"))
    assert again.history.records[1].outputs.tolist() == [9.0, 9.5]
    assert again.history.records[1].corrections[0]["outputs"] == [1.5, -1.5]
    assert len(again.history) == 3


def test_a_correction_refits_the_model(tmp_path):
    c = _campaign(tmp_path)
    c._model, c._model_valid = object(), True  # a cached fit
    c.correct(2, {"z": 0.0})
    assert not c._model_valid


def test_correct_refuses_what_it_cant_correct(tmp_path):
    c = _campaign(tmp_path)
    (pending,) = c.suggest()
    with pytest.raises(KeyError, match="outstanding"):
        c.correct(pending.id, {"y": 1.0})
    with pytest.raises(KeyError, match="No recorded run 'c9999'"):
        c.correct("c9999", {"y": 1.0})
    with pytest.raises(IndexError, match="out of range"):
        c.correct(7, {"y": 1.0})
    with pytest.raises(ValueError, match="unknown output"):
        c.correct(0, {"yeild": 1.0})
    c.fail(pending.id, "lamp died")
    with pytest.raises(ValueError, match="failed"):
        c.correct(pending.id, {"y": 1.0})
    assert torch.equal(c.history.records[0].outputs, torch.tensor([0.5, -0.5], dtype=torch.double))
