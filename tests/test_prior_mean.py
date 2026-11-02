"""Prior means: a GP that starts from a prior model."""

import math

import pytest
import torch

from abbot import (
    BAXPolicy,
    Campaign,
    ContinuousDomain,
    EnumeratedDomain,
    OutputSpec,
    ProblemSpec,
    Source,
    default_constructor,
)
from abbot.goal import TopK

SPEC = ProblemSpec(
    domain=[ContinuousDomain(name="t", lower=0, upper=10),
            ContinuousDomain(name="p", lower=0.01, upper=10, log=True)],
    outputs=[OutputSpec(name="y", direction="MINIMIZE")],
)


def physics(x):
    return 3 * torch.sin(x["t"]) + torch.log10(x["p"])


def truth(t, p, offset=0.0, scale=1.0):
    return scale * (3 * math.sin(t) + math.log10(p)) + offset + 0.3 * math.cos(2 * t)


def _rmse(n=6, offset=0.0, scale=1.0, **kw):
    """Fit on ``n`` runs; RMSE of the raw prediction over the space, and the model."""
    campaign = Campaign(SPEC, model_constructor=default_constructor(SPEC, **kw),
                        initial_designs=torch.empty(0, 2))
    for t, p in SPEC.sample_designs(n, seed=1).tolist():
        campaign.record({"t": t, "p": p}, [truth(t, p, offset, scale)])
    model = campaign.current_model()
    X = SPEC.sample_designs(300, seed=3)
    y = torch.tensor([truth(t, p, offset, scale) for t, p in X.tolist()])
    raw = -model.posterior(X).mean.squeeze(-1).detach()  # MINIMIZE: canonical -> raw
    return float(((raw - y) ** 2).mean().sqrt()), model


@pytest.mark.slow
def test_a_prior_helps_with_few_runs_and_learns_an_offset():
    plain, _ = _rmse()
    with_prior, _ = _rmse(prior_mean={"y": physics})
    assert with_prior < 0.6 * plain
    shifted, model = _rmse(offset=5.0, prior_mean={"y": physics})
    assert shifted < 0.6 * plain  # the offset is learned
    assert float(model.mean_module.scale) == 1.0


@pytest.mark.slow
def test_fit_prior_scale_shrinks_a_prior_off_in_amplitude():
    fixed, _ = _rmse(scale=0.4, prior_mean={"y": physics})
    fitted, model = _rmse(scale=0.4, prior_mean={"y": physics}, fit_prior_scale=True)
    assert fitted < fixed
    assert 0.2 < float(model.mean_module.scale) < 0.7


@pytest.mark.slow
def test_a_prior_on_one_output_with_labels_under_infobax():
    spec = ProblemSpec(
        domain=[ContinuousDomain(name="a", lower=-2, upper=2),
                EnumeratedDomain(name="s", values=["x", "y"])],
        outputs=[OutputSpec(name="f", direction="OBSERVE"),
                 OutputSpec(name="g", direction="OBSERVE")],
    )
    model = default_constructor(spec, prior_mean={"f": lambda x: x["a"] ** 2 + x["s"]})

    class Experiment:
        def run(self, d):
            return [d["a"] ** 2 + {"x": 0, "y": 1}[d["s"]] + 0.5, d["a"]]

    torch.manual_seed(0)
    campaign = Campaign(spec, BAXPolicy(TopK(k=3, output="f"), n_candidates=200), model)
    for _ in range(8):
        campaign.step(Experiment())  # InfoBAX conditions the model on its samples
    means = [type(gp.mean_module).__name__ for gp in campaign.current_model().models]
    assert means == ["_PriorMean", "ConstantMean"]
    assert campaign.check().fits[0].r2 is not None


def test_a_prior_with_known_noise():
    spec = ProblemSpec(domain=[ContinuousDomain(name="a", lower=-2, upper=2)],
                       outputs=[OutputSpec(name="f", direction="MAXIMIZE")])
    model = default_constructor(spec, prior_mean={"f": lambda x: -x["a"] ** 2})
    campaign = Campaign(spec, model_constructor=model, initial_designs=torch.empty(0, 1))
    for a in (-1.5, 0.3, 1.2):
        campaign.record({"a": a}, [-a * a + 0.2], output_var=[0.01])
    gp = campaign.current_model()
    assert type(gp.likelihood).__name__ == "FixedNoiseGaussianLikelihood"
    mean = float(gp.posterior(torch.tensor([[0.0]], dtype=torch.double)).mean)
    assert mean == pytest.approx(0.2, abs=0.15)  # the prior's peak, lifted by the offset


@pytest.mark.slow
def test_warm_start_from_an_earlier_campaign():
    """The recipe in docs/API.md: an earlier campaign's model as the prior."""
    torch.manual_seed(0)
    old = Campaign(SPEC, initial_designs=torch.empty(0, 2))
    for t, p in SPEC.sample_designs(30, seed=5).tolist():
        old.record({"t": t, "p": p}, [truth(t, p)])
    earlier = old.current_model()
    names = SPEC.variable_names
    sign = float(SPEC.output_signs()[0])

    def prior(x):
        X = torch.stack([x[n] for n in names], dim=-1)
        return sign * earlier.posterior(X).mean[..., 0]

    plain, _ = _rmse(offset=1.0)
    warm, _ = _rmse(offset=1.0, prior_mean={"y": prior})
    assert warm < 0.5 * plain


def test_what_prior_mean_refuses():
    with pytest.raises(ValueError, match="no output named"):
        default_constructor(SPEC, prior_mean={"z": physics})
    with pytest.raises(ValueError, match="failed on sample designs: KeyError"):
        default_constructor(SPEC, prior_mean={"y": lambda x: x["nope"]})
    with pytest.raises(ValueError, match="non-finite"):
        default_constructor(SPEC, prior_mean={"y": lambda x: x["t"] * float("nan")})
    sourced = SPEC.model_copy(update={"sources": [
        Source(domain=ContinuousDomain(name="fid", lower=0, upper=1), target=1.0)]})
    with pytest.raises(ValueError, match="source variables"):
        default_constructor(sourced, prior_mean={"y": physics})
