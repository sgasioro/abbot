"""UI builder helpers: form rows -> ProblemSpec, and optimizer selection."""

import pytest

from abbot.acq_opt import Continuous, DiscreteArgmax
from abbot.problem import ContinuousDomain, EnumeratedDomain
from abbot.ui.builder import build_spec, choose_optimizer


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
