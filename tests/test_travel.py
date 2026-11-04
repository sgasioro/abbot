"""Travel limits: max_step on design variables."""

import pytest
import torch

from abbot import (
    BAXPolicy,
    Campaign,
    ContinuousDomain,
    EnumeratedDomain,
    IntegerDomain,
    OutputSpec,
    ProblemSpec,
    Source,
    UncertaintySampling,
)
from abbot.acq_opt import DiscreteArgmax
from abbot.goal import TopK
from abbot.io import (
    spec_from_json,
    spec_history_from_csv,
    spec_history_to_csv,
    spec_to_json,
)
from abbot.policy import SimpleBO


def _spec(*outputs, **kw):
    outputs = outputs or (OutputSpec(name="f", direction="MINIMIZE"),)
    domain = kw.pop("domain", [
        ContinuousDomain(name="a", lower=-2, upper=2, max_step=0.3),
        ContinuousDomain(name="b", lower=-2, upper=2, max_step=0.3),
    ])
    return ProblemSpec(domain=domain, outputs=list(outputs), **kw)


def _bowl(design):
    return [(design["a"] - 1.5) ** 2 + (design["b"] - 1.5) ** 2]


class _Bowl:
    def run(self, design):
        return _bowl(design)


START = [{"a": -1.8, "b": -1.8}]


def _walk(campaign):
    """Every issued design, in issue order."""
    records = sorted(campaign.history.records, key=lambda r: r.issued_at or r.finished_at)
    return [r.candidate for r in records] + [c.x for c in campaign._inflight.values()]


def _assert_in_reach(spec, designs):
    for i in range(1, len(designs)):
        assert not spec.out_of_reach(designs[i - 1], designs[i]), (i, designs[i - 1], designs[i])


# -- the spec --


def test_max_step_validation():
    with pytest.raises(ValueError, match="greater than 0"):
        ContinuousDomain(name="a", lower=0, upper=1, max_step=0)
    with pytest.raises(ValueError, match="at least 1"):
        IntegerDomain(name="n", lower=0, upper=5, max_step=0.5)
    with pytest.raises(ValueError, match="labels have no distance"):
        EnumeratedDomain(name="s", values=["x", "y"], max_step=1)
    with pytest.raises(ValueError, match="travel limits are for design variables"):
        _spec(sources=[Source(domain=ContinuousDomain(name="s", lower=0, upper=1, max_step=0.1),
                              target=1.0)])


def test_reach_narrows_each_kind_of_variable():
    spec = _spec(domain=[
        ContinuousDomain(name="a", lower=-5, upper=5, max_step=0.5),
        IntegerDomain(name="n", lower=0, upper=10, max_step=2),
        EnumeratedDomain(name="e", values=[0.1, 0.5, 1.0, 2.0], max_step=0.5),
        EnumeratedDomain(name="s", values=["x", "y"]),
        ContinuousDomain(name="free", lower=0, upper=1),
    ])
    reach = spec.reach(torch.tensor([4.8, 1.0, 0.5, 1.0, 0.3]))
    lower, upper = reach.bounds_tensor()
    assert lower.tolist() == pytest.approx([4.3, 0, 0.1, 0, 0])
    assert upper.tolist() == pytest.approx([5.0, 3, 1.0, 1, 1])  # clipped at the bounds
    assert reach.domain[2].values == [0.1, 0.5, 1.0]
    assert spec.out_of_reach(torch.tensor([0.0, 0, 0.1, 0, 0]),
                             torch.tensor([1.0, 3, 0.5, 1, 1])) == {"a": 1.0, "n": 3.0}
    moved = reach._into(torch.tensor([9.0, 7.4, 1.9, 1.0, 0.3]))
    assert moved.tolist() == pytest.approx([5.0, 3.0, 1.0, 1.0, 0.3])


def test_max_step_round_trips_through_json_and_csv():
    spec = _spec(domain=[
        ContinuousDomain(name="a", lower=0.1, upper=5, log=True, max_step=0.5),
        IntegerDomain(name="n", lower=0, upper=10, max_step=2),
        EnumeratedDomain(name="e", values=[0.1, 0.5, 1.0], max_step=0.5),
    ])
    assert spec_from_json(spec_to_json(spec)) == spec
    back, _ = spec_history_from_csv(spec_history_to_csv(spec))
    assert [d.max_step for d in back.domain] == [0.5, 2, 0.5]
    assert back.domain[0].log and back.domain[2].values == [0.1, 0.5, 1.0]


def test_vocs_warns_that_it_drops_travel_limits(caplog):
    pytest.importorskip("gest_api")
    from abbot.interop.xopt import spec_to_vocs

    spec_to_vocs(_spec())
    assert "no travel limits" in caplog.text


# -- the campaign --


def test_a_start_is_required():
    with pytest.raises(ValueError, match="needs to know where the hardware starts"):
        Campaign(_spec()).suggest(1)
    campaign = Campaign(_spec())
    campaign.record({"a": 0.0, "b": 0.0}, [1.0])  # a recorded run is a start too
    (c,) = campaign.suggest(1)
    assert abs(c["a"]) <= 0.3 + 1e-9 and abs(c["b"]) <= 0.3 + 1e-9


@pytest.mark.slow
def test_bo_walks_to_the_optimum_one_step_at_a_time():
    torch.manual_seed(0)
    spec = _spec()
    campaign = Campaign(spec, initial_designs=START)
    for _ in range(25):
        campaign.step(_Bowl())
    designs = _walk(campaign)
    _assert_in_reach(spec, designs)
    best = min(float(r.outputs) for r in campaign.history.records)
    assert best < 0.1  # from (-1.8, -1.8) to near (1.5, 1.5): at least 11 steps


@pytest.mark.slow
def test_batches_and_outstanding_designs_walk_in_order():
    torch.manual_seed(0)
    spec = _spec()
    campaign = Campaign(spec, initial_designs=START)
    campaign.step(_Bowl())
    first = campaign.suggest(3)  # a batch: each a step from the one before
    second = campaign.suggest(2)  # more while those are out: from the last issued
    designs = [campaign.history.records[0].candidate] + [
        spec.as_tensor(c) for c in first + second
    ]
    _assert_in_reach(spec, designs)
    campaign.ingest([_bowl(c) for c in first + second])
    for _ in range(3):
        campaign.step(_Bowl(), n=2)
    _assert_in_reach(spec, _walk(campaign))


def test_seeds_must_walk():
    with pytest.raises(ValueError, match="seed 1 is out of reach of seed 0: a moves 1"):
        Campaign(_spec(), initial_designs=[{"a": 0.0, "b": 0.0}, {"a": 1.0, "b": 0.1}])
    Campaign(_spec(), initial_designs=[{"a": 0.0, "b": 0.0}, {"a": 0.3, "b": 0.1}])


def test_fixed_out_of_reach_is_refused():
    campaign = Campaign(_spec(), initial_designs=torch.empty(0, 2))
    campaign.record({"a": 0.0, "b": 0.0}, [1.0])
    with pytest.raises(ValueError, match="fixed= puts the design out of reach"):
        campaign.suggest(1, fixed={"a": 1.0})
    (c,) = campaign.suggest(1, fixed={"a": 0.2})
    assert c["a"] == pytest.approx(0.2)


def test_a_retry_out_of_reach_is_dropped():
    campaign = Campaign(_spec(), initial_designs=torch.empty(0, 2))
    campaign.record({"a": 0.0, "b": 0.0}, [1.0])
    (c,) = campaign.suggest(1)
    campaign.fail(c.id, "jammed")  # queued for a retry at c
    campaign.record({"a": -1.0, "b": -1.0}, [1.0])  # the hardware moved on
    (d,) = campaign.suggest(1)
    assert "dropped a retry" in campaign.last_warning
    assert d.id != c.id and not campaign._retry_queue
    assert abs(d["a"] + 1.0) <= 0.3 + 1e-9


def test_a_policy_without_within_is_moved_into_reach():
    class Far:
        uq_requirement = "point"

        def suggest(self, history, spec, model, n=1, *, pending=None):
            return torch.full((n, 2), 1.9, dtype=torch.double)

        def ingest(self, results):
            return None

    campaign = Campaign(_spec(), Far(), initial_designs=START)
    campaign.step(_Bowl())
    (c,) = campaign.suggest(1)
    assert (c["a"], c["b"]) == pytest.approx((-1.5, -1.5))
    assert "out of reach" in campaign.last_warning and "moved it" in campaign.last_warning


@pytest.mark.slow
@pytest.mark.parametrize("policy", ["US", "BAX", "discrete"])
def test_other_policies_stay_in_reach(policy):
    torch.manual_seed(0)
    spec = _spec()
    if policy == "US":
        campaign = Campaign(spec, UncertaintySampling(), initial_designs=START)
    elif policy == "BAX":
        spec = _spec(OutputSpec(name="f", direction="OBSERVE"))
        campaign = Campaign(spec, BAXPolicy(TopK(k=3, output="f"), n_candidates=400),
                            initial_designs=START)
    else:
        grid = torch.linspace(-2, 2, 21, dtype=torch.double)
        choices = torch.cartesian_prod(grid, grid)
        campaign = Campaign(spec, SimpleBO(DiscreteArgmax(choices)), initial_designs=START)
    for _ in range(10):
        campaign.step(_Bowl())
    _assert_in_reach(spec, _walk(campaign))
    if policy == "BAX":  # a sparse grid: it runs out of new points in reach, and walks on
        assert "stepping toward" in campaign.last_warning
    else:
        assert campaign.last_warning is None
