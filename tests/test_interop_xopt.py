"""gest-api / Xopt interop: spec<->VOCS field map + CorePolicyGenerator.

The standard-conformance tests need only gest-api (always installed). The
end-to-end ``Xopt(generator=...)`` test needs xopt (importorskip) — it is the
"never the spine" proof: an abbot policy drives a full Xopt run on Xopt's
evaluator, and a user's ProblemSpec can always walk back to Xopt.
"""

import pytest
import torch

from abbot.interop.xopt import CorePolicyGenerator, spec_to_vocs, vocs_to_spec
from abbot.problem import (
    ContinuousDomain,
    EnumeratedDomain,
    ObjectiveSpec,
    ProblemSpec,
)


def _spec():
    return ProblemSpec(
        domain=[
            ContinuousDomain(name="temp", lower=20.0, upper=80.0),
            EnumeratedDomain(name="catalyst", values=[0.0, 1.0, 2.0]),
        ],
        objectives=[ObjectiveSpec(name="yield", direction="MAXIMIZE")],
    )


# --- spec <-> VOCS (gest-api only) --------------------------------------------


def test_spec_to_vocs_variable_and_objective_types():
    from gest_api.vocs import ContinuousVariable, DiscreteVariable, MaximizeObjective

    vocs = spec_to_vocs(_spec())
    assert isinstance(vocs.variables["temp"], ContinuousVariable)
    assert vocs.variables["temp"].domain == [20.0, 80.0]
    assert isinstance(vocs.variables["catalyst"], DiscreteVariable)
    assert isinstance(vocs.objectives["yield"], MaximizeObjective)


def test_vocs_round_trips_direction():
    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        objectives=[ObjectiveSpec(name="cost", direction="MINIMIZE")],
    )
    back = vocs_to_spec(spec_to_vocs(spec))
    assert back.objectives[0].direction == "MINIMIZE"
    assert back.variable_names == ["x"]


@pytest.mark.slow
def test_core_policy_generator_is_a_gest_generator():
    import gest_api

    gen = CorePolicyGenerator(_spec())
    assert isinstance(gen, gest_api.Generator)
    # ask/tell contract works standalone (no Xopt).
    gen.campaign.seed(torch.tensor([[30.0, 1.0], [50.0, 2.0]], dtype=torch.double))
    s = gen.suggest(1)
    assert set(s[0].keys()) == {"temp", "catalyst"}
    gen.ingest([{**s[0], "yield": 42.0}])
    assert len(gen.campaign.history) == 1


# --- end-to-end under real Xopt (needs xopt) ----------------------------------


@pytest.mark.slow
def test_abbot_policy_drives_an_xopt_run():
    pytest.importorskip("xopt")
    import warnings

    from xopt import Evaluator, Xopt

    torch.manual_seed(0)
    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        objectives=[ObjectiveSpec(name="y", direction="MAXIMIZE")],
    )
    gen = CorePolicyGenerator(spec)

    def f(inputs):  # objective peaks at x = 0.3
        return {"y": -((inputs["x"] - 0.3) ** 2)}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xo = Xopt(generator=gen, evaluator=Evaluator(function=f, max_workers=1))
        # Xopt copies the generator; seed via Xopt and inspect the LIVE one.
        xo.random_evaluate(3)
        for _ in range(10):
            xo.step()

    live = xo.generator
    assert len(xo.data) == 13
    # abbot's campaign (inside the live generator) saw every evaluated point.
    assert len(live.campaign.history) == 13
    # Both Xopt's data and abbot's recommendation locate the optimum ~0.3.
    best_xopt = float(xo.data.loc[xo.data["y"].idxmax(), "x"])
    assert abs(best_xopt - 0.3) < 0.1
    rec = live.campaign.recommend()
    assert rec and abs(float(live.campaign.history.records[rec[0]].candidate) - 0.3) < 0.15
