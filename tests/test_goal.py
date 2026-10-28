"""Goal subset-identification logic on canned f (exact, no GP)."""

import pytest
import torch

from abbot.goal import (
    BandIntersection,
    FunctionGoal,
    LevelSet,
    PercentileSet,
    TopK,
    Wishlist,
    goals_from_spec,
)
from abbot.problem import Box, ContinuousDomain, ObjectiveSpec, ProblemSpec


def test_percentile_set_top_fraction():
    f = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    # 80th percentile of [1..5] = 4.2 -> only value 5 is >= it.
    assert PercentileSet(80.0).identify_subspace(f) == [4]
    # 50th percentile = 3.0 -> values 3,4,5.
    assert PercentileSet(50.0).identify_subspace(f) == [2, 3, 4]


def test_percentile_set_rejects_multioutput():
    f = torch.rand(5, 2)
    with pytest.raises(ValueError, match="scores one output"):
        PercentileSet(80.0).identify_subspace(f)


def test_band_intersection_is_and_across_outputs():
    # out0 in [0,1], out1 in [10,11]; only rows inside BOTH count.
    f = torch.tensor([[0.5, 10.5], [0.5, 20.0], [5.0, 10.5], [0.9, 10.9]])
    goal = BandIntersection([(0.0, 1.0), (10.0, 11.0)])
    assert goal.identify_subspace(f) == [0, 3]


def test_wishlist_is_or_of_boxes():
    f = torch.tensor([[0.5], [5.5], [9.0]])
    # box A: [0,1]; box B: [5,6]; row 2 (9.0) in neither.
    goal = Wishlist([[(0.0, 1.0)], [(5.0, 6.0)]])
    assert goal.identify_subspace(f) == [0, 1]


def test_band_bounds_length_mismatch():
    with pytest.raises(ValueError, match="outputs"):
        BandIntersection([(0.0, 1.0)]).identify_subspace(torch.rand(3, 2))


def test_goals_from_spec_unions_boxes():
    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        objectives=[
            ObjectiveSpec(name="a", direction="MAXIMIZE"),
            ObjectiveSpec(name="b", direction="MAXIMIZE"),
        ],
        boxes=[
            Box(lower=[0.0, 10.0], upper=[1.0, 11.0]),
            Box(lower=[5.0, 20.0], upper=[6.0, 21.0]),
        ],
    )
    goal = goals_from_spec(spec)
    f = torch.tensor([[0.5, 10.5], [5.5, 20.5], [3.0, 3.0]])
    assert goal.identify_subspace(f) == [0, 1]


def test_function_goal_wraps_arbitrary_fn():
    f = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.5, 0.5]])
    # "output 0 greater than output 1" — not expressible as a box.
    goal = FunctionGoal(lambda f, X: (f[:, 0] > f[:, 1]).nonzero(as_tuple=True)[0].tolist())
    assert goal.identify_subspace(f) == [1]


def test_function_goal_receives_X():
    f = torch.zeros(3, 1)
    X = torch.tensor([[0.1], [0.5], [0.9]])
    # x-dependent goal: candidates with x > 0.4.
    goal = FunctionGoal(lambda f, X: (X[:, 0] > 0.4).nonzero(as_tuple=True)[0].tolist())
    assert goal.identify_subspace(f, X) == [1, 2]


def test_level_set():
    f = torch.tensor([[1.0], [5.0], [3.0]])
    assert LevelSet(threshold=3.0).identify_subspace(f) == [1, 2]       # >= 3
    assert LevelSet(threshold=3.0, upper=True).identify_subspace(f) == [0, 2]  # <= 3


def test_top_k():
    f = torch.tensor([[1.0], [5.0], [3.0], [2.0]])
    assert sorted(TopK(k=2).identify_subspace(f)) == [1, 2]            # two largest
    assert sorted(TopK(k=2, largest=False).identify_subspace(f)) == [0, 3]  # two smallest


def test_goals_from_spec_requires_boxes():
    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        objectives=[ObjectiveSpec(name="a", direction="MAXIMIZE")],
    )
    with pytest.raises(ValueError, match="boxes"):
        goals_from_spec(spec)
