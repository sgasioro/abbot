"""GestGeneratorPolicy: gest-api / Xopt generators as abbot policies."""

import pytest
import torch

pytest.importorskip("xopt")

from abbot import (
    Campaign,
    ContinuousDomain,
    EnumeratedDomain,
    OutputSpec,
    ProblemSpec,
    Source,
)
from abbot.interop.xopt import (
    CorePolicyGenerator,
    GestGeneratorPolicy,
)
from abbot.model import default_constructor


def _spec(**kw):
    domain = kw.pop("domain", [
        ContinuousDomain(name="a", lower=-2, upper=2),
        ContinuousDomain(name="b", lower=-2, upper=2),
    ])
    return ProblemSpec(domain=domain, outputs=[OutputSpec(name="f", direction="MINIMIZE")], **kw)


def _f(design):
    return [(design["a"] - 0.5) ** 2 + (design["b"] + 0.3) ** 2]


class _Bowl:
    def run(self, design):
        return _f(design)


class _CountingConstructor:
    """The default model, counting its fits."""

    provides_uq = "calibrated_variance"

    def __init__(self, spec):
        self.inner = default_constructor(spec)
        self.builds = 0

    def build(self, X, Y, Yvar=None):
        self.builds += 1
        return self.inner.build(X, Y, Yvar)


def _nelder_mead():
    from xopt.generators.sequential.neldermead import NelderMeadGenerator

    return NelderMeadGenerator


def test_nelder_mead_minimizes_without_abbot_fitting_a_model():
    torch.manual_seed(0)
    spec = _spec()
    model = _CountingConstructor(spec)
    campaign = Campaign(spec, GestGeneratorPolicy(_nelder_mead()), model)
    for _ in range(60):
        campaign.step(_Bowl())
    assert model.builds == 0  # proposals never waited on abbot's fit
    best = min(float(r.outputs) for r in campaign.history.clean.records)
    assert best < 1e-3
    assert campaign.result().best is not None  # recommendations still work


def test_nelder_mead_starts_from_the_best_seed():
    spec = _spec()
    seeds = [{"a": 2.0, "b": 2.0}, {"a": 0.4, "b": -0.2}, {"a": -2.0, "b": 1.0}]
    campaign = Campaign(spec, GestGeneratorPolicy(_nelder_mead()), initial_designs=seeds)
    for _ in range(3):
        campaign.step(_Bowl())
    (c,) = campaign.suggest(1)
    assert abs(c["a"] - 0.4) < 0.3 and abs(c["b"] + 0.2) < 0.3  # near the best seed


def test_sequential_generators_go_one_at_a_time():
    spec = _spec()
    campaign = Campaign(spec, GestGeneratorPolicy(_nelder_mead()),
                        initial_designs=[{"a": 1.0, "b": 1.0}])
    campaign.step(_Bowl())
    with pytest.raises(ValueError, match="one design at a time; ask for n=1"):
        campaign.suggest(2)
    campaign.suggest(1)
    with pytest.raises(ValueError, match="still outstanding"):
        campaign.suggest(1)


def test_an_outside_result_resets_a_waiting_sequential_generator():
    spec = _spec()
    campaign = Campaign(spec, GestGeneratorPolicy(_nelder_mead()),
                        initial_designs=[{"a": 1.0, "b": 1.0}])
    campaign.step(_Bowl())
    (c,) = campaign.suggest(1)
    campaign.cancel(c.id)
    campaign.record({"a": 0.0, "b": 0.0}, _f({"a": 0.0, "b": 0.0}))
    campaign.step(_Bowl())
    assert "reset" in campaign.last_warning
    for _ in range(5):
        campaign.step(_Bowl())  # and carries on


def test_a_resumed_campaign_hands_a_new_generator_the_history():
    spec = _spec()
    first = Campaign(spec, GestGeneratorPolicy(_nelder_mead()))
    for _ in range(12):
        first.step(_Bowl())
    second = Campaign(spec, GestGeneratorPolicy(_nelder_mead()), initial_designs=torch.empty(0, 2))
    second.load_history(first.history)
    second.step(_Bowl())
    assert len(second.policy.generator.data) == len(first.history.clean)


def test_a_batch_generator_with_labels():
    from xopt.generators.random import RandomGenerator

    spec = _spec(domain=[
        ContinuousDomain(name="a", lower=-2, upper=2),
        ContinuousDomain(name="b", lower=-2, upper=2),
        EnumeratedDomain(name="solvent", values=["water", "ethanol"]),
    ])
    campaign = Campaign(spec, GestGeneratorPolicy(RandomGenerator), initial_designs=torch.empty(0, 3))
    designs = campaign.suggest(3)  # a batch, and the generator starts with no data
    assert len(designs) == 3 and {d["solvent"] for d in designs} <= {"water", "ethanol"}
    campaign.ingest([_f(d) for d in designs])
    assert len(campaign.history.clean) == 3


@pytest.mark.slow
def test_xopt_bayesian_generator_with_turbo():
    from xopt.generators.bayesian.expected_improvement import (
        ExpectedImprovementGenerator,
    )

    torch.manual_seed(0)
    campaign = Campaign(_spec(), GestGeneratorPolicy(ExpectedImprovementGenerator,
                                                     turbo_controller="optimize"))
    for _ in range(8):
        campaign.step(_Bowl())
    assert campaign.policy.generator.turbo_controller is not None
    assert len(campaign.history.clean) == 8


def test_what_a_generator_policy_refuses():
    from xopt.generators.random import RandomGenerator

    spec = _spec()
    with pytest.raises(TypeError, match="generator class"):
        GestGeneratorPolicy(RandomGenerator(vocs=GestGeneratorPolicy.vocs_for(spec)), seed=1)
    other = _spec(domain=[ContinuousDomain(name="z", lower=0, upper=1)])
    campaign = Campaign(spec, GestGeneratorPolicy(RandomGenerator(vocs=GestGeneratorPolicy.vocs_for(other))),
                        initial_designs=torch.empty(0, 2))
    with pytest.raises(ValueError, match="don't match"):
        campaign.suggest(1)
    campaign = Campaign(spec, GestGeneratorPolicy(RandomGenerator), initial_designs=torch.empty(0, 2))
    with pytest.raises(ValueError, match="takes no fixed= argument"):
        campaign.suggest(1, fixed={"a": 0.0})
    with_source = _spec(sources=[Source(domain=ContinuousDomain(name="s", lower=0, upper=1), target=1.0)])
    campaign = Campaign(with_source, GestGeneratorPolicy(RandomGenerator),
                        initial_designs=torch.empty(0, 3))
    with pytest.raises(ValueError, match="source variables"):
        campaign.suggest(1)


def test_core_policy_generator_ingests_labels():
    spec = _spec(domain=[
        ContinuousDomain(name="a", lower=-2, upper=2),
        EnumeratedDomain(name="solvent", values=["water", "ethanol"]),
    ])
    gen = CorePolicyGenerator(spec)
    gen.ingest([{"a": 0.5, "solvent": "ethanol", "f": 1.0}])
    assert gen.campaign.history.records[0].candidate.tolist() == [0.5, 1.0]
