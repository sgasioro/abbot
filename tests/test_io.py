"""Spec JSON and history CSV import/export round-trips."""

import pytest
import torch

from abbot.io import (
    history_from_csv,
    history_to_csv,
    spec_from_json,
    spec_to_json,
)
from abbot.model import default_constructor
from abbot.policy import SimpleBO
from abbot.problem import ContinuousDomain, EnumeratedDomain, ObjectiveSpec, ProblemSpec
from abbot.session import Campaign


def _spec():
    return ProblemSpec(
        domain=[
            ContinuousDomain(name="temp", lower=20.0, upper=80.0),
            EnumeratedDomain(name="catalyst", values=[0.0, 1.0, 2.0]),
        ],
        objectives=[
            ObjectiveSpec(name="yield", direction="MAXIMIZE"),
            ObjectiveSpec(name="cost", direction="MINIMIZE"),
        ],
    )


def test_spec_json_round_trip():
    spec = _spec()
    restored = spec_from_json(spec_to_json(spec))
    assert restored == spec


def test_history_csv_round_trip():
    spec = _spec()
    campaign = Campaign(spec, SimpleBO(), default_constructor(spec, bounds=spec.bounds_tensor()))
    # Hand-build a couple of measured records via load_history.
    from abbot.history import History, Record

    h = History()
    h.append(Record(candidate=torch.tensor([30.0, 1.0]), observation=None, outputs=torch.tensor([42.0, 5.0])))
    h.append(Record(candidate=torch.tensor([60.0, 2.0]), observation=None, outputs=torch.tensor([55.0, 3.0])))
    campaign.load_history(h)

    csv_text = history_to_csv(campaign.history, spec)
    assert "step,temp,catalyst,yield,cost" in csv_text.replace(" ", "")

    reparsed = history_from_csv(csv_text, spec)
    assert len(reparsed) == 2
    assert torch.allclose(reparsed.records[0].candidate, torch.tensor([30.0, 1.0], dtype=torch.double))
    assert torch.allclose(reparsed.records[1].outputs, torch.tensor([55.0, 3.0], dtype=torch.double))


def test_history_from_csv_missing_column():
    spec = _spec()
    bad = "step,temp,yield,cost\n0,30,42,5\n"  # missing 'catalyst'
    with pytest.raises(ValueError, match="missing required column"):
        history_from_csv(bad, spec)


def test_history_from_csv_bad_number():
    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        objectives=[ObjectiveSpec(name="y", direction="MAXIMIZE")],
    )
    bad = "x,y\n0.5,notanumber\n"
    with pytest.raises(ValueError, match="row 2"):
        history_from_csv(bad, spec)


def test_load_history_feeds_the_fit(tmp_path):
    from abbot.history import DataStore, History, Record

    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        objectives=[ObjectiveSpec(name="y", direction="MAXIMIZE")],
    )
    campaign = Campaign(
        spec, SimpleBO(), default_constructor(spec, bounds=spec.bounds_tensor()),
        store=DataStore(tmp_path),
    )
    h = History()
    for xv in (0.1, 0.4, 0.7):
        h.append(Record(candidate=torch.tensor([xv]), observation=None, outputs=torch.tensor([xv])))
    campaign.load_history(h)
    assert len(campaign.history) == 3
    # Persisted: a fresh campaign on the same store resumes the imported rows.
    resumed = Campaign(
        spec, SimpleBO(), default_constructor(spec, bounds=spec.bounds_tensor()),
        store=DataStore(tmp_path),
    )
    assert len(resumed.history) == 3
