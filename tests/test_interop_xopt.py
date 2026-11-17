"""gest-api / Xopt interop: spec<->VOCS field map + CorePolicyGenerator.

The standard-conformance tests need only gest-api (always installed). The
end-to-end ``Xopt(generator=...)`` test needs xopt (importorskip) — it is the
"never the spine" proof: an abbot policy drives a full Xopt run on Xopt's
evaluator, and a user's ProblemSpec can always walk back to Xopt.
"""

import pytest
import torch

from abbot.interop.xopt import CorePolicyGenerator, spec_to_vocs, vocs_to_spec
from abbot.policy import SimpleBO
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


def test_observe_outputs_map_to_vocs_observables():
    from abbot import OutputSpec

    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        outputs=[
            OutputSpec(name="yield", direction="MAXIMIZE"),
            OutputSpec(name="feat", direction="OBSERVE"),
        ],
    )
    vocs = spec_to_vocs(spec)
    assert "yield" in vocs.objectives
    assert "feat" in vocs.observables and "feat" not in vocs.objectives
    # round-trip: observables come back as OBSERVE outputs
    back = vocs_to_spec(vocs)
    assert back.n_outputs == 2 and back.n_objectives == 1
    assert any(o.name == "feat" and o.direction == "OBSERVE" for o in back.outputs)


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

    # _spec() is a mixed space, which has no default policy; this test only
    # exercises the gest contract on seeded points, so any policy will do.
    gen = CorePolicyGenerator(_spec(), SimpleBO())
    assert isinstance(gen, gest_api.Generator)
    # ask/tell contract works standalone (no Xopt).
    gen.campaign.seed(torch.tensor([[30.0, 1.0], [50.0, 2.0]], dtype=torch.double))
    s = gen.suggest(1)
    assert set(s[0].keys()) == {"temp", "catalyst"}
    gen.ingest([{**s[0], "yield": 42.0}])
    assert len(gen.campaign.history) == 1
    # The result came back by value, not id; it still closes the suggestion.
    assert gen.campaign.outstanding() == []
    assert gen.campaign.history.records[0].id == s[0].id


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
    # Every suggestion came back through Xopt's DataFrame and closed by exact value.
    assert live.campaign.outstanding() == []
    # Both Xopt's data and abbot's recommendation locate the optimum ~0.3.
    best_xopt = float(xo.data.loc[xo.data["y"].idxmax(), "x"])
    assert abs(best_xopt - 0.3) < 0.1
    rec = live.campaign.recommend()
    assert rec and abs(float(live.campaign.history.records[rec[0]].candidate) - 0.3) < 0.15


# -- EvaluatorExperiment: an Xopt evaluator as an abbot experiment --


def _peak(inputs, scale=1.0):
    if inputs["temp"] > 79:
        raise RuntimeError("interlock")
    return {"yield": scale * (100 - (inputs["temp"] - 55) ** 2 / 10), "flux": 3.0,
            "xopt_runtime": 0.01}


def test_an_evaluator_runs_as_an_experiment_held_to_its_vocs():
    xopt = pytest.importorskip("xopt")
    from abbot import Campaign
    from abbot.interop.xopt import EvaluatorExperiment

    vocs = xopt.VOCS(variables={"temp": [20, 80]}, objectives={"yield": "MAXIMIZE"},
                     observables=["flux"])
    e = EvaluatorExperiment(xopt.Evaluator(function=_peak, function_kwargs={"scale": 2.0}),
                            vocs)
    assert e.spec.variable_names == ["temp"] and e.spec.output_names == ["yield", "flux"]
    assert e.run({"temp": 55.0}) == ({"yield": 200.0, "flux": 3.0}, {"xopt_runtime": 0.01})
    # A raise reaches abbot (Evaluator.evaluate would have caught it): a fault, retried.
    with pytest.raises(RuntimeError, match="interlock"):
        e.run({"temp": 80.0})
    torch.manual_seed(0)
    campaign = Campaign(e.spec)
    for _ in range(3):
        campaign.step(e)
    assert len(campaign.history) == 3
    assert campaign.history.records[-1].aux == {"xopt_runtime": 0.01}


def test_a_plain_function_or_a_vectorized_evaluator():
    xopt = pytest.importorskip("xopt")
    from abbot import Campaign
    from abbot.interop.xopt import EvaluatorExperiment

    spec = ProblemSpec(domain=[ContinuousDomain(name="temp", lower=20, upper=80)],
                       objectives=[ObjectiveSpec(name="yield", direction="MAXIMIZE")])
    plain = EvaluatorExperiment(lambda inputs: {"yield": inputs["temp"]}, spec=spec)
    assert plain.run({"temp": 30.0}) == {"yield": 30.0}

    def batch(df):
        return {"yield": list(df["temp"] / 10), "xopt_runtime": [0.0] * len(df)}

    vec = EvaluatorExperiment(xopt.Evaluator(function=batch, vectorized=True), spec=spec)
    assert not hasattr(vec, "run")
    assert vec.run_batch([{"temp": 30.0}, {"temp": 50.0}]) == [
        ({"yield": 3.0}, {"xopt_runtime": 0.0}), ({"yield": 5.0}, {"xopt_runtime": 0.0})]
    campaign = Campaign(spec)
    campaign.step(vec, n=3)
    assert len(campaign.history) == 3
    with pytest.raises(RuntimeError, match=r"without \['yield'\]"):
        EvaluatorExperiment(lambda inputs: {"y": 1}, spec=spec).run({"temp": 30.0})
    with pytest.raises(ValueError, match="vocs or spec, not both"):
        EvaluatorExperiment(batch, spec_to_vocs(spec), spec=spec)
