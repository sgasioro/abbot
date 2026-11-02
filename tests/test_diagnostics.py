"""Stopping on a target or on no improvement, and Campaign.check()."""

import pytest
import torch

from abbot import (
    Campaign,
    ContinuousDomain,
    EnumeratedDomain,
    OutputSpec,
    ProblemSpec,
    Source,
    Verdict,
)
from abbot.diagnostics import check
from abbot.loop import drive
from abbot.supervision import (
    all_of,
    any_of,
    bind,
    cost_cap,
    min_runs,
    no_improvement,
    per_shot,
    target_reached,
)


def _spec(*outputs, **kw):
    outputs = outputs or (OutputSpec(name="f", direction="MINIMIZE"),)
    domain = kw.pop("domain", [
        ContinuousDomain(name="a", lower=-2, upper=2),
        ContinuousDomain(name="b", lower=-2, upper=2),
    ])
    return ProblemSpec(domain=domain, outputs=list(outputs), **kw)


def _bowl(design):
    return (design["a"] - 0.5) ** 2 + (design["b"] + 0.3) ** 2


class _Bowl:
    def __init__(self, noise=0.0):
        self.noise = noise

    def run(self, design):
        return [_bowl(design) + self.noise * float(torch.randn(()))]


def _campaign(spec, supervisor, values):
    """A campaign that has recorded ``values`` (one output each) in order."""
    campaign = Campaign(spec, supervisor=supervisor, initial_designs=torch.empty(0, 2))
    for i, v in enumerate(values):
        campaign.record({"a": i / 100, "b": 0.0}, v if isinstance(v, list) else [v])
    return campaign


# -- target_reached --


def test_target_reached_stops_drive_at_the_target():
    torch.manual_seed(0)
    campaign = Campaign(_spec(), supervisor=target_reached(0.05))
    drive(campaign, _Bowl(), n_total=40)
    assert campaign.stopped is not None and "target reached" in campaign.stopped
    best = min(float(r.outputs) for r in campaign.history.records)
    assert best <= 0.05
    assert float(campaign.history.records[-1].outputs) <= 0.05  # stopped on that run
    assert len(campaign.history) < 40


def test_target_reached_respects_direction_names_and_constraints():
    up = _spec(OutputSpec(name="y", direction="MAXIMIZE"))
    assert _campaign(up, target_reached(1.0), [0.5, 0.9]).stopped is None
    assert _campaign(up, target_reached(1.0), [0.5, 1.0]).stopped is not None

    two = _spec(OutputSpec(name="y", direction="MAXIMIZE"),
                OutputSpec(name="cost", direction="MINIMIZE"))
    rule = target_reached({"y": 1.0, "cost": 2.0})
    assert _campaign(two, rule, [[1.5, 3.0], [0.5, 1.0]]).stopped is None  # never both at once
    stopped = _campaign(two, rule, [[1.5, 3.0], [1.2, 1.5]]).stopped
    assert "run 1" in stopped and "y = 1.2" in stopped

    capped = _spec(OutputSpec(name="y", direction="MAXIMIZE"),
                   OutputSpec(name="t", direction="OBSERVE", max_val=1.0))
    assert _campaign(capped, target_reached(1.0), [[2.0, 5.0]]).stopped is None  # infeasible
    assert _campaign(capped, target_reached(1.0), [[2.0, 0.5]]).stopped is not None


def test_target_reached_refusals():
    two = _spec(OutputSpec(name="y", direction="MAXIMIZE"),
                OutputSpec(name="z", direction="MAXIMIZE"))
    with pytest.raises(ValueError, match="2 objectives; name the outputs"):
        Campaign(two, supervisor=target_reached(1.0))
    with pytest.raises(ValueError, match="no output named 'w'"):
        Campaign(two, supervisor=target_reached({"w": 1.0}))
    observed = _spec(OutputSpec(name="y", direction="MAXIMIZE"),
                     OutputSpec(name="t", direction="OBSERVE"))
    with pytest.raises(ValueError, match="'t' is OBSERVE"):
        Campaign(observed, supervisor=target_reached({"t": 1.0}))
    with pytest.raises(TypeError, match="needs the problem spec"):
        target_reached(1.0)(Campaign(_spec()).history)


def test_a_rejected_run_does_not_reach_the_target():
    def glitch(observation):
        return observation is not None and observation < -50  # a nonsense reading

    rule = any_of(per_shot(glitch), target_reached(0.0), cost_cap(1e9))
    campaign = Campaign(_spec(), supervisor=rule, initial_designs=torch.empty(0, 2))
    campaign.record({"a": 0.0, "b": 0.0}, [-100.0], observation=-100.0)
    assert campaign.stopped is None
    assert campaign.history.records[-1].fault
    campaign.record({"a": 0.0, "b": 0.0}, [-1.0], observation=-1.0)
    assert "target reached" in campaign.stopped


def test_bind_outside_a_campaign():
    spec = _spec()
    supervisor = bind(any_of(target_reached(1.0), lambda h: Verdict()), spec)
    assert supervisor(Campaign(spec).history).stop is None


# -- no_improvement --


def test_no_improvement_counts_runs_since_the_best():
    rule = no_improvement(3)
    assert _campaign(_spec(), rule, [5, 4, 3, 3.5, 3.2]).stopped is None  # 2 runs since
    assert "no improvement" in _campaign(_spec(), rule, [5, 4, 3, 3.5, 3.2, 3.1]).stopped
    # min_delta: small gains don't count
    assert _campaign(_spec(), no_improvement(3), [5, 3, 2.99, 2.98, 2.97]).stopped is None
    assert _campaign(_spec(), no_improvement(3, min_delta=0.1),
                     [5, 3, 2.99, 2.98, 2.97]).stopped is not None


def test_no_improvement_waits_for_a_feasible_run_and_refuses_several_objectives():
    capped = _spec(OutputSpec(name="y", direction="MAXIMIZE"),
                   OutputSpec(name="t", direction="OBSERVE", max_val=1.0))
    assert _campaign(capped, no_improvement(2), [[1.0, 5.0]] * 6).stopped is None
    two = _spec(OutputSpec(name="y", direction="MAXIMIZE"),
                OutputSpec(name="z", direction="MAXIMIZE"))
    with pytest.raises(ValueError, match="use acquisition_monitor"):
        Campaign(two, supervisor=no_improvement(5))
    with pytest.raises(ValueError, match="patience"):
        no_improvement(0)


def test_no_improvement_ends_a_converged_campaign():
    torch.manual_seed(0)
    campaign = Campaign(_spec(), supervisor=no_improvement(8, min_delta=1e-3))
    drive(campaign, _Bowl(), n_total=80)
    assert campaign.stopped is not None and "no improvement" in campaign.stopped
    assert len(campaign.history) < 80


# -- all_of: floors --


def test_all_of_stops_only_when_every_rule_agrees():
    values = [5, 4, 3, 3.5, 3.2, 3.1]  # no improvement over the last 3
    assert _campaign(_spec(), all_of(no_improvement(3), min_runs(8)), values).stopped is None
    stopped = _campaign(_spec(), all_of(no_improvement(3), min_runs(8)), values + [3.3, 3.4]).stopped
    assert "no improvement" in stopped and "at least 8 runs" in stopped
    # a cost floor
    campaign = Campaign(_spec(), supervisor=all_of(no_improvement(3), cost_cap(10.0)),
                        initial_designs=torch.empty(0, 2))
    for i, v in enumerate(values):
        campaign.record({"a": i / 100, "b": 0.0}, [v], cost=1.0)
    assert campaign.stopped is None
    for i in range(4):
        campaign.record({"a": 0.5 + i / 100, "b": 0.0}, [4.0], cost=1.0)
    assert campaign.stopped is not None


def test_all_of_passes_faults_through_and_a_rejected_run_does_not_stop():
    def glitch(observation):
        return observation is not None and observation < -50

    rule = any_of(per_shot(glitch), all_of(target_reached(0.0), min_runs(2)))
    campaign = Campaign(_spec(), supervisor=rule, initial_designs=torch.empty(0, 2))
    campaign.record({"a": 0.0, "b": 0.0}, [1.0], observation=1.0)
    campaign.record({"a": 0.1, "b": 0.0}, [-100.0], observation=-100.0)
    assert campaign.history.records[-1].fault and campaign.stopped is None
    campaign.record({"a": 0.2, "b": 0.0}, [-1.0], observation=-1.0)
    assert "target reached" in campaign.stopped

    plain = Campaign(_spec(), initial_designs=torch.empty(0, 2))
    plain.record({"a": 0.0, "b": 0.0}, [-100.0], observation=-100.0)
    detecting = all_of(per_shot(glitch), min_runs(1))  # retries still pass through
    assert detecting(plain.history).retry
    with pytest.raises(ValueError, match="at least one"):
        all_of()
    with pytest.raises(ValueError, match="at least 1"):
        min_runs(0)


# -- check --


def _fitted(spec, experiment, n, seed=0):
    torch.manual_seed(seed)
    campaign = Campaign(spec)
    for _ in range(n):
        campaign.step(experiment)
    return campaign


def test_check_on_a_good_fit():
    spec = _spec(domain=[
        ContinuousDomain(name="a", lower=-2, upper=2),
        ContinuousDomain(name="b", lower=-2, upper=2),
        ContinuousDomain(name="junk", lower=0, upper=1),
    ])
    report = _fitted(spec, _Bowl(noise=0.01), 20).check()
    assert report.ok, str(report)
    (fit,) = report.fits
    assert fit.r2 > 0.6 and len(fit.z) == 20
    assert fit.lengthscales["junk"] > 10 * fit.lengthscales["a"]
    assert any("dependence on junk" in f.message for f in report.findings)
    assert "leave-one-out R²" in str(report)


def test_check_flags_an_outlier_with_its_run_and_sign():
    campaign = _fitted(_spec(), _Bowl(noise=0.01), 20)
    campaign.record({"a": 0.4, "b": -0.2}, [_bowl({"a": 0.4, "b": -0.2}) + 1.0])
    report = campaign.check()
    assert not report.ok
    assert report.fits[0].z[20] > 3  # higher than predicted, though f is minimized
    assert any("run 20 (+" in f.message for f in report.warnings), str(report)


def test_check_flags_an_output_that_is_only_noise():
    spec = _spec(OutputSpec(name="f", direction="MINIMIZE"),
                 OutputSpec(name="g", direction="OBSERVE"))

    class Noise:
        def run(self, design):
            return [_bowl(design), float(torch.randn(()))]

    report = _fitted(spec, Noise(), 15).check()
    g = [f for f in report.findings if f.output == "g"]
    assert any(f.level == "warning" for f in g), str(report)
    assert not [f for f in report.findings if f.output == "f" and f.level == "warning"]


def test_check_runs_vs_inputs_and_no_model():
    spec = _spec(domain=[ContinuousDomain(name=f"x{i}", lower=0, upper=1) for i in range(4)])
    report = Campaign(spec).check()
    assert report.model is None and not report.ok
    assert "0 clean runs for 4 inputs" in str(report)
    campaign = Campaign(spec, initial_designs=torch.empty(0, 4))
    for i in range(6):
        campaign.record({f"x{j}": (i * 0.17 + j * 0.3) % 1 for j in range(4)}, [float(i)])
    assert any("expect a rough model until about 8" in f.message
               for f in campaign.check().findings)


@pytest.mark.parametrize("kind", ["categories", "ordered source", "unordered source"])
def test_check_reads_the_other_builtin_models(kind):
    if kind == "categories":
        spec = _spec(domain=[ContinuousDomain(name="a", lower=-2, upper=2),
                             EnumeratedDomain(name="s", values=["x", "y", "z"])])

        def f(d):
            return d["a"] ** 2 + {"x": 0, "y": 1, "z": 2}[d["s"]]
    elif kind == "ordered source":
        spec = _spec(domain=[ContinuousDomain(name="a", lower=-2, upper=2)],
                     sources=[Source(domain=ContinuousDomain(name="fid", lower=0, upper=1),
                                     target=1.0)])

        def f(d):
            return d["a"] ** 2 + 0.1 * (1 - d["fid"])
    else:
        spec = _spec(domain=[ContinuousDomain(name="a", lower=-2, upper=2)],
                     sources=[Source(domain=EnumeratedDomain(name="code", values=["dft", "xtb"]),
                                     target="dft")])

        def f(d):
            return d["a"] ** 2 + (0.3 if d["code"] == "xtb" else 0.0)

    class E:
        def run(self, design):
            return [f(design)]

    report = _fitted(spec, E(), 12).check()
    (fit,) = report.fits
    assert fit.r2 is not None and "a" in fit.lengthscales
    assert "s" not in fit.lengthscales  # a categorical kernel's lengthscale isn't a distance


def test_check_on_a_model_that_is_not_a_gp():
    class Mean:
        provides_uq = "calibrated_variance"

        def build(self, X, Y, Yvar=None):
            return object()

    campaign = Campaign(_spec(), model_constructor=Mean(), initial_designs=torch.empty(0, 2))
    for i in range(5):
        campaign.record({"a": i / 5, "b": 0.0}, [float(i)])
    report = check(campaign.history, campaign.spec, campaign.current_model())
    assert report.fits == [] and "isn't one" in str(report)
