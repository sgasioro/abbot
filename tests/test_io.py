"""Spec JSON and history CSV import/export round-trips."""

import pytest
import torch

from abbot.history import History, Record
from abbot.io import (
    history_from_csv,
    history_to_csv,
    spec_from_json,
    spec_history_from_csv,
    spec_history_to_csv,
    spec_to_json,
)
from abbot.model import default_constructor
from abbot.policy import SimpleBO
from abbot.problem import (
    Box,
    ContinuousDomain,
    EnumeratedDomain,
    ObjectiveSpec,
    ProblemSpec,
)
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
    from abbot.history import DataStore

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


# --- self-describing CSV (spec + history in one file) -------------------------


def _history_for(spec):
    h = History()
    h.append(Record(candidate=torch.tensor([30.0, 1.0]), observation=None, outputs=torch.tensor([42.0, 5.0])))
    h.append(Record(candidate=torch.tensor([60.0, 2.0]), observation=None, outputs=torch.tensor([55.0, 3.0])))
    return h


def test_self_describing_round_trip():
    spec = _spec()  # continuous + enumerated vars, MAX + MIN objectives
    hist = _history_for(spec)
    text = spec_history_to_csv(spec, hist)

    # Config lives in `#` rows; the data table is pandas/Excel-readable.
    assert "# version" in text
    assert "# continuous,temp" in text
    assert "# enumerated,catalyst" in text

    spec2, hist2 = spec_history_from_csv(text)
    assert spec2 == spec                       # spec round-trips exactly
    assert len(hist2) == 2
    assert torch.allclose(hist2.records[1].outputs, torch.tensor([55.0, 3.0], dtype=torch.double))


def test_self_describing_observe_round_trip():
    from abbot import OutputSpec

    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        outputs=[
            OutputSpec(name="a", direction="OBSERVE"),
            OutputSpec(name="b", direction="OBSERVE"),
        ],
        boxes=[Box(lower=[0.4, 0.4], upper=[0.6, 0.6])],
    )
    text = spec_history_to_csv(spec)
    assert "# output,a,OBSERVE" in text  # OBSERVE direction emitted
    spec2, _ = spec_history_from_csv(text)
    assert spec2.n_objectives == 0 and spec2.n_outputs == 2
    assert all(o.direction == "OBSERVE" for o in spec2.outputs)


def test_self_describing_reads_legacy_objective_rows():
    # A file written by the old format (# objective rows) still loads.
    legacy = (
        "# version,1\n"
        "# continuous,x,0.0,1.0\n"
        "# objective,y,MAXIMIZE\n"
        "step,x,y\n0,0.5,1.0\n"
    )
    spec, hist = spec_history_from_csv(legacy)
    assert spec.output_names == ["y"]
    assert spec.n_objectives == 1
    assert len(hist) == 1


def test_self_describing_with_boxes():
    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        objectives=[ObjectiveSpec(name="a", direction="MAXIMIZE"),
                    ObjectiveSpec(name="b", direction="MAXIMIZE")],
        boxes=[Box(lower=[4.0, 0.0], upper=[6.0, 1.0])],
    )
    spec2, _ = spec_history_from_csv(spec_history_to_csv(spec))
    assert spec2.boxes[0].lower == [4.0, 0.0]
    assert spec2.boxes[0].upper == [6.0, 1.0]


def test_self_describing_refuses_constraints():
    from abbot.problem import InequalitySpec

    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        objectives=[ObjectiveSpec(name="y", direction="MAXIMIZE")],
        constraints=[InequalitySpec(indices=[0], coeffs=[1.0], rhs=0.5)],
    )
    with pytest.raises(ValueError, match="constraints"):
        spec_history_to_csv(spec)


def test_self_describing_malformed_box_row_is_clear():
    text = "# continuous,x,0,1\n# output,y,MAXIMIZE\n# box,4.0,6.0\nstep,x,y\n"
    with pytest.raises(ValueError, match="box"):
        spec_history_from_csv(text)


def test_self_describing_short_continuous_row_is_clear():
    text = "# continuous,x,0\n# output,y,MAXIMIZE\nstep,x,y\n"
    with pytest.raises(ValueError, match="continuous"):
        spec_history_from_csv(text)


def test_self_describing_ignores_unknown_comment_rows():
    spec = _spec()
    text = "# some note,ignored\n# version,1\n" + spec_history_to_csv(spec)
    spec2, _ = spec_history_from_csv(text)
    assert spec2.variable_names == ["temp", "catalyst"]


def test_pandas_can_read_the_data_table():
    pd = pytest.importorskip("pandas")
    import io as _io
    text = spec_history_to_csv(_spec(), _history_for(_spec()))
    df = pd.read_csv(_io.StringIO(text), comment="#")
    assert list(df.columns) == ["step", "temp", "catalyst", "yield", "cost"]
    assert len(df) == 2
