"""An output aimed at a value (direction TARGET): as close to ``target`` as it goes."""

import pytest
import torch

from abbot import Campaign, ContinuousDomain, OutputSpec, ProblemSpec
from abbot.analysis import convergence
from abbot.io import spec_history_from_csv, spec_history_to_csv
from abbot.loop import drive
from abbot.session import recommend
from abbot.supervision import no_improvement, target_reached

DOMAIN = [ContinuousDomain(name="x", lower=0, upper=1),
          ContinuousDomain(name="w", lower=0, upper=1)]


def _spec(*outputs):
    return ProblemSpec(domain=DOMAIN, outputs=list(outputs or [OutputSpec(name="y", target=1.2)]))


def _campaign(spec, values, **kw):
    campaign = Campaign(spec, initial_designs=torch.empty(0, 2), **kw)
    for i, v in enumerate(values):
        campaign.record({"x": i / 100, "w": 0.0}, v if isinstance(v, list) else [v])
    return campaign


class _Curve:  # y = 3x² + w: every point on a curve reaches 1.2
    def run(self, design):
        return [3 * design["x"] ** 2 + design["w"]]


def test_a_target_implies_the_direction_and_needs_a_value():
    assert OutputSpec(name="y", target=1.2).direction == "TARGET"
    assert ProblemSpec.model_validate({
        "domain": [{"name": "x", "lower": 0, "upper": 1}],
        "outputs": [{"name": "y", "target": 2}],
    }).outputs[0].direction == "TARGET"
    with pytest.raises(ValueError, match="give the value to aim for"):
        OutputSpec(name="y", direction="TARGET")
    with pytest.raises(ValueError, match="has a target but is MAXIMIZE"):
        OutputSpec(name="y", direction="MAXIMIZE", target=1.0)


def test_a_target_is_scored_by_its_distance():
    spec = _spec(OutputSpec(name="y", target=1.2), OutputSpec(name="z", direction="MINIMIZE"),
                 OutputSpec(name="t", direction="OBSERVE"))
    assert spec.has_target and spec.n_objectives == 2
    raw = torch.tensor([[1.0, 3.0, 9.0], [1.5, 1.0, 9.0]], dtype=torch.double)
    assert torch.allclose(spec.objective_view(raw),
                          torch.tensor([[-0.2, -3.0], [-0.3, -1.0]], dtype=torch.double))
    # The model sees a TARGET column unflipped, so the scores of canonical samples agree.
    canonical = raw * spec.output_signs()
    assert torch.allclose(spec.objective_scores(canonical), spec.objective_view(raw))
    assert spec.outputs[0].score(1.1) == pytest.approx(-0.1)


def test_the_best_so_far_and_the_recommendation_are_the_nearest_run():
    spec = _spec()
    campaign = _campaign(spec, [0.0, 2.0, 1.3, 1.6, 1.15])
    assert convergence(campaign.history, spec) == [0.0, 2.0, 1.3, 1.3, 1.15]
    assert recommend(campaign.history, spec, denoise=False) == [4]


@pytest.mark.slow
def test_simple_bo_reaches_the_target():
    torch.manual_seed(0)
    campaign = Campaign(_spec())
    drive(campaign, _Curve(), n_total=14)
    best = convergence(campaign.history, campaign.spec)[-1]
    assert abs(best - 1.2) < 0.05
    rec = campaign.history.records[campaign.recommend()[0]]
    assert abs(float(rec.outputs.reshape(-1)[0]) - 1.2) < 0.05


@pytest.mark.slow
def test_a_target_works_with_a_constraint_and_with_a_second_objective():
    torch.manual_seed(0)
    capped = _spec(OutputSpec(name="y", target=1.2),
                   OutputSpec(name="z", direction="OBSERVE", max_val=0.3))
    campaign = Campaign(capped)
    for _ in range(14):
        d = campaign.suggest(1)[0]
        campaign.ingest({"y": 3 * d["x"] ** 2 + d["w"], "z": d["x"]})
    y, z = campaign.history.records[campaign.recommend()[0]].outputs.reshape(-1).tolist()
    assert abs(y - 1.2) < 0.1 and z <= 0.3

    torch.manual_seed(0)
    two = _spec(OutputSpec(name="y", target=1.2), OutputSpec(name="cost", direction="MINIMIZE"))
    campaign = Campaign(two)
    for _ in range(14):
        d = campaign.suggest(1)[0]
        campaign.ingest({"y": 3 * d["x"] ** 2 + d["w"], "cost": d["w"]})
    assert type(campaign.policy).__name__ == "ParetoBO"
    front = [campaign.history.records[i].outputs.reshape(-1).tolist()
             for i in campaign.recommend()]
    assert any(abs(y - 1.2) < 0.1 and cost < 0.1 for y, cost in front)


def test_stopping_rules_read_a_target():
    spec = _spec()
    # target_reached's value is a tolerance for a TARGET output
    assert _campaign(spec, [0.0, 1.5], supervisor=target_reached(0.1)).stopped is None
    stopped = _campaign(spec, [0.0, 1.25], supervisor=target_reached(0.1)).stopped
    assert "y = 1.25 (within 0.1 of 1.2)" in stopped
    with pytest.raises(ValueError, match="how near is near enough"):
        Campaign(spec, supervisor=target_reached(-0.1))
    # no_improvement: nearer counts as better, whichever side
    rule = no_improvement(2)
    assert _campaign(spec, [0.0, 1.3, 1.1, 1.15], supervisor=rule).stopped is None
    stopped = _campaign(spec, [0.0, 1.25, 1.0, 1.4], supervisor=no_improvement(2)).stopped
    assert "the best y is 1.25" in stopped


def test_the_self_describing_csv_keeps_the_target():
    spec = _spec(OutputSpec(name="y", target=1.2, max_val=2.0))
    text = spec_history_to_csv(spec)
    assert "# output,y,TARGET,,2.0,target=1.2" in text
    assert spec_history_from_csv(text)[0] == spec


def test_xopt_export_refuses_a_target():
    pytest.importorskip("xopt")
    from abbot.interop.xopt import spec_to_vocs

    with pytest.raises(NotImplementedError, match="TARGET"):
        spec_to_vocs(_spec())
