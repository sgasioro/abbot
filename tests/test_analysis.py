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
