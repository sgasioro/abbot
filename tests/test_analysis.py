"""The analysis.py built-ins compute correctly on a canned History (no UI)."""

import pytest
import torch

from abbot.analysis import convergence, hypervolume, recommended_set
from abbot.history import History, Record
from abbot.problem import ContinuousDomain, ObjectiveSpec, ProblemSpec


def _hist(outputs, faults=None):
    h = History()
    faults = faults or [False] * len(outputs)
    for i, (y, f) in enumerate(zip(outputs, faults)):
        h.append(
            Record(
                candidate=torch.tensor([float(i)], dtype=torch.double),
                observation=None,
                outputs=torch.tensor(y, dtype=torch.double),
                fault=f,
            )
        )
    return h


def test_convergence_maximize_is_running_max():
    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=5.0)],
        objectives=[ObjectiveSpec(name="f", direction="MAXIMIZE")],
    )
    h = _hist([[1.0], [3.0], [2.0], [5.0], [4.0]])
    assert convergence(h, spec) == [1.0, 3.0, 3.0, 5.0, 5.0]


def test_convergence_minimize_is_running_min():
    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=5.0)],
        objectives=[ObjectiveSpec(name="cost", direction="MINIMIZE")],
    )
    h = _hist([[5.0], [3.0], [4.0], [1.0]])
    assert convergence(h, spec) == [5.0, 3.0, 3.0, 1.0]


def test_convergence_rejects_multiobjective():
    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=5.0)],
        objectives=[
            ObjectiveSpec(name="a", direction="MAXIMIZE"),
            ObjectiveSpec(name="b", direction="MAXIMIZE"),
        ],
    )
    with pytest.raises(ValueError, match="single-objective"):
        convergence(_hist([[1.0, 2.0]]), spec)


def test_hypervolume_non_decreasing():
    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        objectives=[
            ObjectiveSpec(name="f1", direction="MAXIMIZE"),
            ObjectiveSpec(name="f2", direction="MAXIMIZE"),
        ],
    )
    # A spread-out trade-off front: HV should grow monotonically as points arrive.
    h = _hist([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1], [0.3, 0.7]])
    hv = hypervolume(h, spec)
    assert len(hv) == 4
    assert all(hv[i + 1] >= hv[i] - 1e-9 for i in range(len(hv) - 1))
    assert hv[-1] > 0.0


def test_recommended_set_matches_observed_front():
    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        objectives=[ObjectiveSpec(name="f", direction="MAXIMIZE")],
    )
    h = _hist([[1.0], [3.0], [2.0]])
    assert recommended_set(h, spec) == [1]  # argmax index


# --- model visualization (needs a fitted model) ------------------------------


def _fitted_1d_minimize():
    """A fitted model on a MINIMIZE problem, to check RAW-unit reporting."""
    from abbot.analysis import canonical_Y
    from abbot.model import SingleTaskGPConstructor

    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        objectives=[ObjectiveSpec(name="cost", direction="MINIMIZE")],
    )
    # Raw cost samples (positive); minimum near x=0.3.
    xs = [0.0, 0.3, 0.6, 1.0]
    h = History()
    for x in xs:
        h.append(
            Record(
                candidate=torch.tensor([x], dtype=torch.double),
                observation=None,
                outputs=torch.tensor([(x - 0.3) ** 2 + 1.0], dtype=torch.double),
            )
        )
    model = SingleTaskGPConstructor(bounds=spec.bounds_tensor()).build(
        h.designs.double(), canonical_Y(h, spec)
    )
    return h, spec, model


@pytest.mark.slow
def test_parity_reports_raw_units():
    from abbot.analysis import parity

    h, spec, model = _fitted_1d_minimize()
    p = parity(h, spec, model)
    assert "cost" in p
    obs = p["cost"]["observed"]
    pred = p["cost"]["predicted"]
    assert len(obs) == len(pred) == 4
    # Observed costs are the raw positive values; predictions should be positive
    # too (RAW units), i.e. the canonical sign was un-applied for display.
    assert all(o > 0 for o in obs)
    assert all(pr > 0 for pr in pred)


@pytest.mark.slow
def test_slice_1d_shape_and_raw_minimum():
    from abbot.analysis import slice_1d

    h, spec, model = _fitted_1d_minimize()
    s = slice_1d(h, spec, model, var=0, output=0, n=50)
    assert len(s["x"]) == len(s["mean"]) == len(s["std"]) == 50
    assert s["var_name"] == "x" and s["objective_name"] == "cost"
    # Raw mean should be lowest near the true minimizer (~0.3), not highest.
    lo_idx = min(range(len(s["mean"])), key=lambda i: s["mean"][i])
    assert 0.1 < s["x"][lo_idx] < 0.5


@pytest.mark.slow
def test_slice_2d_shape():
    from abbot.analysis import canonical_Y, slice_2d
    from abbot.model import ModelListConstructor

    spec = ProblemSpec(
        domain=[
            ContinuousDomain(name="a", lower=0.0, upper=1.0),
            ContinuousDomain(name="b", lower=0.0, upper=1.0),
        ],
        objectives=[ObjectiveSpec(name="y", direction="MAXIMIZE")],
    )
    h = History()
    for a, b in [(0.1, 0.2), (0.5, 0.5), (0.9, 0.3), (0.3, 0.8)]:
        h.append(
            Record(
                candidate=torch.tensor([a, b], dtype=torch.double),
                observation=None,
                outputs=torch.tensor([a + b], dtype=torch.double),
            )
        )
    model = ModelListConstructor(bounds=spec.bounds_tensor()).build(
        h.designs.double(), canonical_Y(h, spec)
    )
    s = slice_2d(h, spec, model, var_x=0, var_y=1, n=12)
    assert len(s["x"]) == len(s["y"]) == 12
    assert len(s["z"]) == 12 and len(s["z"][0]) == 12
