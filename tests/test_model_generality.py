"""Model-seam generality: per-point noise (Yvar) threading + the EnsembleConstructor
capability contract. Fast cases need no GP fit; the end-to-end fits are slow."""

import pytest
import torch

from abbot.history import History, Record
from abbot.model import EnsembleConstructor
from abbot.problem import ContinuousDomain, OutputSpec, ProblemSpec
from abbot.session import canonical_Y, canonical_Yvar


def _spec(direction="MINIMIZE"):
    return ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        outputs=[OutputSpec(name="cost", direction=direction)],
    )


def _hist_with_var(rows):
    """rows: list of (x, y, var|None)."""
    h = History()
    for x, y, v in rows:
        h.append(
            Record(
                candidate=torch.tensor([x], dtype=torch.double),
                observation=None,
                outputs=torch.tensor([y], dtype=torch.double),
                output_var=None if v is None else torch.tensor([v], dtype=torch.double),
            )
        )
    return h


def test_canonical_yvar_is_sign_invariant():
    # MINIMIZE flips Y to canonical (negates), but VARIANCE must NOT flip (sign^2=1).
    spec = _spec("MINIMIZE")
    h = _hist_with_var([(0.1, 5.0, 0.25), (0.5, 3.0, 0.16)])
    Y = canonical_Y(h, spec)
    Yvar = canonical_Yvar(h, spec)
    assert (Y < 0).all()  # raw-positive costs negated to canonical
    assert Yvar is not None
    assert (Yvar > 0).all()  # variances stay positive (unflipped)
    assert torch.allclose(Yvar.reshape(-1), torch.tensor([0.25, 0.16], dtype=torch.double))


def test_canonical_yvar_all_or_nothing():
    spec = _spec("MAXIMIZE")
    # One record missing its variance -> the whole fit infers noise (None).
    h = _hist_with_var([(0.1, 1.0, 0.1), (0.5, 2.0, None)])
    assert canonical_Yvar(h, spec) is None
    # All present -> a stacked tensor.
    h2 = _hist_with_var([(0.1, 1.0, 0.1), (0.5, 2.0, 0.2)])
    assert canonical_Yvar(h2, spec) is not None


def test_ensemble_constructor_provides_uq_is_honest():
    assert EnsembleConstructor(n_members=5).provides_uq == "samples"
    assert EnsembleConstructor(n_members=1).provides_uq == "point"


def test_ensemble_point_model_refused_by_variance_policy():
    from abbot.policy import UncertaintySampling
    from abbot.session import check_uq_pairing

    # A size-1 ensemble is a point model; US needs calibrated variance -> refused.
    with pytest.raises(ValueError, match="calibrated uncertainty"):
        check_uq_pairing(UncertaintySampling(), EnsembleConstructor(n_members=1))


# --- end-to-end fits (slow) ---------------------------------------------------


@pytest.mark.slow
def test_yvar_threads_into_a_campaign_fit():
    from abbot.model import SingleTaskGPConstructor
    from abbot.policy import SimpleBO
    from abbot.session import Campaign

    torch.manual_seed(0)
    spec = _spec("MAXIMIZE")
    c = Campaign(spec, SimpleBO(), SingleTaskGPConstructor(bounds=spec.bounds_tensor()))
    # Supply per-point noise with every measurement.
    for xv in [0.1, 0.4, 0.6, 0.9]:
        c.record(torch.tensor([xv]), torch.tensor([-((xv - 0.5) ** 2)]),
                 output_var=torch.tensor([0.01]))
    # The clean history carries variance everywhere -> a fitted model exists.
    assert canonical_Yvar(c.history, spec) is not None
    assert c.current_model() is not None


@pytest.mark.slow
def test_ensemble_constructor_drives_meanbax():
    """A deep ensemble (point-UQ path) drives a full MeanBAX campaign — the seam is
    genuinely non-GP."""
    from abbot.goal import LevelSet
    from abbot.policy import BAXPolicy
    from abbot.session import Campaign

    torch.manual_seed(0)
    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        outputs=[OutputSpec(name="y", direction="OBSERVE")],
    )
    # MeanBAX needs only a point prediction -> a 1-member ensemble is a valid pairing.
    ctor = EnsembleConstructor(n_members=1, epochs=50)
    pol = BAXPolicy(LevelSet(threshold=0.0), strategy="MeanBAX", n_candidates=100)
    c = Campaign(spec, pol, ctor)
    c.seed(torch.tensor([[0.2], [0.5], [0.8]], dtype=torch.double))
    for _ in range(4):
        x = c.suggest(1)[0]["x"]
        c.ingest(torch.tensor([-((x - 0.5) ** 2)]))
    assert len(c.history) == 4


def test_matern52_has_a_dimension_scaled_lengthscale_prior():
    """Without one it failed to fit sparse data with many exact zeros (BAXstics'
    clipped yields); the prior keeps the fit well posed."""
    from gpytorch.kernels import MaternKernel

    from abbot import ModelListConstructor, SingleTaskGPConstructor

    torch.manual_seed(0)
    X = torch.rand(10, 3, dtype=torch.double)
    Y = torch.clamp(10 * (X[:, :1] * X[:, 1:2]) - 3, min=0)  # mostly zero
    for ctor in (SingleTaskGPConstructor, ModelListConstructor):
        model = ctor(kernel="matern52").build(X, torch.cat([Y, X[:, 2:]], -1)
                                              if ctor is ModelListConstructor else Y)
        gps = model.models if ctor is ModelListConstructor else [model]
        for gp in gps:
            matern = next(m for m in gp.covar_module.modules() if isinstance(m, MaternKernel))
            assert matern.nu == 2.5 and matern.lengthscale_prior is not None
