"""UI builder helpers: form rows -> ProblemSpec, and optimizer selection."""

import pytest

from abbot.acq_opt import Continuous, DiscreteArgmax
from abbot.goal import BandIntersection, LevelSet, PercentileSet, TopK
from abbot.problem import ContinuousDomain, EnumeratedDomain
from abbot.ui.builder import (
    BAX_STRATEGY_LABELS,
    build_goal,
    build_spec,
    choose_optimizer,
)


def test_build_spec_mixed_variable_types():
    spec = build_spec(
        variable_rows=[
            {"kind": "continuous", "name": "temp", "lower": 20, "upper": 80},
            {"kind": "enumerated", "name": "catalyst", "values": "0, 1, 2"},
        ],
        output_rows=[{"name": "yield", "direction": "MAXIMIZE"}],
    )
    assert spec.variable_names == ["temp", "catalyst"]
    assert isinstance(spec.domain[0], ContinuousDomain)
    assert isinstance(spec.domain[1], EnumeratedDomain)
    assert spec.domain[1].values == [0.0, 1.0, 2.0]


def test_build_spec_surfaces_validation_errors():
    with pytest.raises(ValueError):
        build_spec(
            variable_rows=[{"kind": "continuous", "name": "x", "lower": 5, "upper": 1}],
            output_rows=[{"name": "f", "direction": "MAXIMIZE"}],
        )


def test_choose_optimizer_by_space():
    cont = build_spec(
        [{"kind": "continuous", "name": "x", "lower": 0, "upper": 1}],
        [{"name": "f", "direction": "MAXIMIZE"}],
    )
    enum = build_spec(
        [{"kind": "enumerated", "name": "c", "values": [0, 1, 2]}],
        [{"name": "f", "direction": "MAXIMIZE"}],
    )
    mixed = build_spec(
        [
            {"kind": "continuous", "name": "x", "lower": 0, "upper": 1},
            {"kind": "enumerated", "name": "c", "values": [0, 1]},
        ],
        [{"name": "f", "direction": "MAXIMIZE"}],
    )
    assert isinstance(choose_optimizer(cont), Continuous)
    assert isinstance(choose_optimizer(enum), DiscreteArgmax)
    with pytest.raises(ValueError, match="Mixed"):
        choose_optimizer(mixed)


# --- BAX goal construction (goal-type dropdown + params) ---------------------


def _bax_spec(output_names):
    return build_spec(
        [{"kind": "continuous", "name": "x", "lower": 0, "upper": 1}],
        [{"name": n, "direction": "OBSERVE"} for n in output_names],
    )


def test_build_goal_band_pads_to_full_width():
    spec = _bax_spec(["a", "b"])
    # Only 'a' carries a band; 'b' is unconstrained -> padded to (-inf, inf).
    goal = build_goal({"kind": "band", "bounds": {"a": (0.0, 1.0)}}, spec)
    assert isinstance(goal, BandIntersection)
    assert goal.bounds[0] == (0.0, 1.0)
    assert goal.bounds[1] == (float("-inf"), float("inf"))


def test_build_goal_band_requires_a_band():
    spec = _bax_spec(["a"])
    with pytest.raises(ValueError, match="at least one output"):
        build_goal({"kind": "band", "bounds": {}}, spec)


def test_build_goal_band_rejects_inverted_bounds():
    spec = _bax_spec(["a"])
    with pytest.raises(ValueError, match="must exceed"):
        build_goal({"kind": "band", "bounds": {"a": (1.0, 0.0)}}, spec)


def test_build_goal_percentile_single_output_only():
    assert isinstance(
        build_goal({"kind": "percentile", "percentile": 90.0}, _bax_spec(["a"])),
        PercentileSet,
    )
    with pytest.raises(ValueError, match="single-output"):
        build_goal({"kind": "percentile"}, _bax_spec(["a", "b"]))


def test_build_goal_level_and_topk_resolve_output_names():
    spec = _bax_spec(["a", "b"])
    lvl = build_goal({"kind": "level", "threshold": 2.0, "output": "b", "upper": True}, spec)
    assert isinstance(lvl, LevelSet)
    assert lvl.output == 1 and lvl.upper is True and lvl.threshold == 2.0

    top = build_goal({"kind": "topk", "k": 3, "output": "b", "largest": False}, spec)
    assert isinstance(top, TopK)
    assert top.k == 3 and top.output == 1 and top.largest is False


def test_build_goal_unknown_output_name_errors():
    with pytest.raises(ValueError, match="unknown output"):
        build_goal({"kind": "level", "threshold": 1.0, "output": "nope"}, _bax_spec(["a"]))


def test_bax_strategy_labels_cover_the_engine_strategies():
    from abbot.policy import _BAX_UQ

    assert set(BAX_STRATEGY_LABELS) == set(_BAX_UQ)
    assert BAX_STRATEGY_LABELS["SwitchBAX"].startswith("Balanced")
