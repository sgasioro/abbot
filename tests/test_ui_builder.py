"""UI builder helpers: form rows -> ProblemSpec, and optimizer selection."""

import pytest

from abbot.acq_opt import Continuous, DiscreteArgmax, default_optimizer
from abbot.goal import BandIntersection, LevelSet, PercentileSet, TopK
from abbot.problem import ContinuousDomain, EnumeratedDomain
from abbot.ui.builder import (
    BAX_STRATEGY_LABELS,
    build_constructor,
    build_goal,
    build_spec,
    build_start,
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


# --- lab-friendly error messages (cryptic native errors -> named ValueError) --


def test_build_spec_blank_continuous_bound_names_the_variable():
    with pytest.raises(ValueError, match="temp"):
        build_spec(
            [{"kind": "continuous", "name": "temp", "lower": None, "upper": 80}],
            [{"name": "f", "direction": "MAXIMIZE"}],
        )


def test_build_spec_non_numeric_enumerated_values_are_labels():
    spec = build_spec(
        [{"kind": "enumerated", "name": "solvent", "values": "water, acetic acid, 1"}],
        [{"name": "f", "direction": "MAXIMIZE"}],
    )
    assert spec.domain[0].values == ["water", "acetic acid", "1"]
    assert spec.categorical_indices() == [0]


def test_build_spec_empty_enumerated_values_errors():
    with pytest.raises(ValueError, match="at least one"):
        build_spec(
            [{"kind": "enumerated", "name": "c", "values": ""}],
            [{"name": "f", "direction": "MAXIMIZE"}],
        )


def test_build_goal_level_missing_threshold_is_clear():
    spec = _bax_spec(["a"])
    with pytest.raises(ValueError, match="threshold"):
        build_goal({"kind": "level", "output": "a"}, spec)


def test_default_optimizer_by_space():
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
    assert isinstance(default_optimizer(cont), Continuous)
    assert isinstance(default_optimizer(enum), DiscreteArgmax)
    assert isinstance(default_optimizer(mixed), Continuous)  # enumerated axis enumerated


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


# --- the rest of the spec: log scale, travel limits, ranges, sources, model ---


def test_build_spec_log_scale_travel_limit_and_output_ranges():
    spec = build_spec(
        [{"kind": "continuous", "name": "conc", "lower": 1e-3, "upper": 1.0, "log": True,
          "max_step": "0.2"}],
        [{"name": "yield", "direction": "MAXIMIZE", "min_val": None, "max_val": ""},
         {"name": "purity", "direction": "OBSERVE", "min_val": 0.9}],
    )
    assert spec.domain[0].log and spec.domain[0].max_step == 0.2
    assert spec.outputs[0].min_val is None and spec.outputs[0].max_val is None
    assert spec.outputs[1].min_val == 0.9


def test_build_spec_labeled_and_ranged_sources():
    rows = [{"kind": "continuous", "name": "x", "lower": 0, "upper": 1}]
    out = [{"name": "f", "direction": "MAXIMIZE"}]
    labeled = build_spec(rows, out, {"kind": "enumerated", "name": "code",
                                     "values": "sim, experiment", "target": "experiment"})
    assert labeled.sources[0].target == "experiment"
    assert labeled.variable_names == ["x", "code"]
    ranged = build_spec(rows, out, {"kind": "continuous", "name": "mesh", "lower": 0.1,
                                    "upper": 1.0, "target": "0.1", "log": True,
                                    "accurate": "low"})
    assert ranged.sources[0].target == 0.1 and ranged.sources[0].accurate == "low"
    assert build_spec(rows, out, {"kind": "enumerated", "name": " "}).sources is None
    with pytest.raises(ValueError, match="target"):
        build_spec(rows, out, {"kind": "enumerated", "name": "code", "values": "a, b"})


def test_build_start_only_with_travel_limits():
    rows = [{"kind": "continuous", "name": "x", "lower": 0, "upper": 10, "max_step": 1,
             "start": 2.0},
            {"kind": "continuous", "name": "y", "lower": 0, "upper": 4},
            {"kind": "enumerated", "name": "c", "values": "a, b"}]
    out = [{"name": "f", "direction": "MAXIMIZE"}]
    assert build_start(build_spec(rows, out), rows) == [{"x": 2.0, "y": 2.0, "c": "a"}]
    free = [{**rows[0], "max_step": None}] + rows[1:]
    assert build_start(build_spec(free, out), free) is None
    missing = [{**rows[0], "start": None}] + rows[1:]
    with pytest.raises(ValueError, match="where it starts"):
        build_start(build_spec(missing, out), missing)


def test_build_constructor_choices():
    rows = [{"kind": "continuous", "name": "x", "lower": 0, "upper": 1}]
    one = build_spec(rows, [{"name": "f", "direction": "MAXIMIZE"}])
    two = build_spec(rows, [{"name": "f", "direction": "MAXIMIZE"},
                            {"name": "g", "direction": "OBSERVE"}])
    assert build_constructor(two) is None
    assert build_constructor(two, "correlated") is not None
    with pytest.raises(ValueError, match="two outputs"):
        build_constructor(one, "correlated")
    sourced = build_spec(rows, [{"name": "f", "direction": "MAXIMIZE"},
                                {"name": "g", "direction": "OBSERVE"}],
                         {"kind": "enumerated", "name": "code", "values": "a, b", "target": "b"})
    with pytest.raises(ValueError, match="sources yet"):
        build_constructor(sourced, "correlated")


def test_spec_to_rows_round_trips_the_editor_fields():
    from abbot.ui.builder import spec_to_rows

    spec = build_spec(
        [{"kind": "continuous", "name": "conc", "lower": 0.01, "upper": 1.0, "log": True,
          "max_step": 0.2},
         {"kind": "enumerated", "name": "solvent", "values": "water, ethanol"}],
        [{"name": "yield", "direction": "MAXIMIZE"},
         {"name": "purity", "direction": "OBSERVE", "min_val": 0.9}],
        {"kind": "continuous", "name": "mesh", "lower": 0.1, "upper": 1.0, "target": "0.1",
         "log": True, "accurate": "low"},
    )
    again = build_spec(*spec_to_rows(spec))
    assert again == spec


def test_spec_to_rows_refuses_what_the_editor_cant_show():
    from abbot.problem import LinearConstraint
    from abbot.ui.builder import spec_to_rows

    spec = build_spec(
        [{"kind": "continuous", "name": "a", "lower": 0, "upper": 1},
         {"kind": "continuous", "name": "b", "lower": 0, "upper": 1}],
        [{"name": "f", "direction": "MAXIMIZE"}],
    ).model_copy(update={"domain_constraints": [LinearConstraint({"a": 1, "b": 1}, max_val=1)]})
    with pytest.raises(ValueError, match="linear constraints"):
        spec_to_rows(spec)
