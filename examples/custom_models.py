"""Swapping the surrogate: the ``ModelConstructor`` seam IS the model-swap point.

abbot does not ship a model zoo — it ships a seam. Any botorch ``Model`` fits behind
``ModelConstructor.build(X, Y, Yvar) -> Model``; the policies compose over
``.posterior(X)`` unchanged. This file shows three swaps against the same
``Campaign``:

  1. a different KERNEL (Matern-5/2 instead of the default RBF),
  2. a fully-Bayesian GP (SAAS) for high-dimensional / few-data regimes,
  3. a deep ensemble (non-GP), via the builtin ``EnsembleConstructor``.

The one rule: set ``provides_uq`` to what your model's posterior ACTUALLY provides
("calibrated_variance" for a GP, "samples" for an ensemble, "point" for a bare net).
The loop checks it against the policy so you can't silently pair, e.g., a point net
with a variance-hungry acquisition.

Research models (PFN, deep-kernel) are the SAME seam — a ``build`` that returns a
botorch ``Model`` — and are deliberately not built here (added when a problem
needs them). For outputs that inform each other, abbot ships one:
``default_constructor(spec, correlated=True)`` (``MultiTaskGPConstructor``).

Run: ``python examples/custom_models.py``
"""

from __future__ import annotations

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from abbot import (
    BAXPolicy,
    Campaign,
    ContinuousDomain,
    EnsembleConstructor,
    LevelSet,
    OutputSpec,
    ProblemSpec,
    SimpleBO,
)


class MaternConstructor:
    """SingleTaskGP with a Matern-5/2 ARD kernel instead of the botorch-default RBF.

    This is the whole kernel-swap story: one ``covar_module``, everything else is the
    standard GP build. A GP posterior is a full Gaussian -> calibrated variance.
    """

    provides_uq = "calibrated_variance"

    def __init__(self, bounds: torch.Tensor | None = None) -> None:
        self.bounds = bounds

    def build(self, X, Y, Yvar=None):
        X, Y = X.double(), Y.double()
        d = X.shape[-1]
        gp = SingleTaskGP(
            X, Y,
            train_Yvar=Yvar.double() if Yvar is not None else None,
            covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d)),
            input_transform=(
                Normalize(d=d, bounds=self.bounds.double())
                if self.bounds is not None else Normalize(d=d)
            ),
        )
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        return gp


class SaasConstructor:
    """A fully-Bayesian SAAS GP (sparse-axis-aligned priors + NUTS) — strong when the
    design space is high-D and data is scarce. Still a GP posterior -> calibrated
    variance. ``warmup``/``num_samples`` kept small here so the example runs quickly.
    """

    provides_uq = "calibrated_variance"

    def __init__(self, bounds: torch.Tensor | None = None, warmup=16, samples=16) -> None:
        self.bounds = bounds
        self.warmup = warmup
        self.samples = samples

    def build(self, X, Y, Yvar=None):
        from botorch.fit import fit_fully_bayesian_model_nuts
        from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP

        X, Y = X.double(), Y.double()
        d = X.shape[-1]
        gp = SaasFullyBayesianSingleTaskGP(
            X, Y,
            train_Yvar=Yvar.double() if Yvar is not None else None,
            input_transform=(
                Normalize(d=d, bounds=self.bounds.double())
                if self.bounds is not None else Normalize(d=d)
            ),
        )
        fit_fully_bayesian_model_nuts(
            gp, warmup_steps=self.warmup, num_samples=self.samples, thinning=4,
            disable_progbar=True,
        )
        return gp


def _bo_demo(name: str, constructor) -> None:
    """Optimize a 1-D objective with SimpleBO on a swapped surrogate (needs a GP:
    log-EI wants calibrated variance, which the guardrail enforces)."""
    torch.manual_seed(0)
    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        outputs=[OutputSpec(name="y", direction="MAXIMIZE")],
    )
    campaign = Campaign(spec, SimpleBO(), constructor)
    campaign.seed(torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.double))
    for _ in range(6):
        x = campaign.suggest(1)[0]["x"]
        campaign.ingest(torch.tensor([-((x - 0.35) ** 2)]))  # peak near x=0.35
    best = float(campaign.result().best.candidate)
    print(f"{name:>22}: recommended x = {best:.3f}  (target 0.35)")
    # NB: SAAS here uses a tiny warmup/sample count so the example runs fast; it is
    # under-sampled on this toy and may miss — raise warmup/samples for real use. The
    # point is that it plugs into the same Campaign, not that it wins a 1-D toy.


def _ensemble_demo() -> None:
    """A deep ensemble provides SAMPLES, not calibrated variance — so it pairs with a
    sample-based BAX policy, NOT SimpleBO. (Try SimpleBO here and the guardrail
    refuses it: capability must match the policy. That refusal is the feature.)"""
    torch.manual_seed(0)
    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        outputs=[OutputSpec(name="y", direction="OBSERVE")],
    )
    policy = BAXPolicy(LevelSet(threshold=-0.05), strategy="SwitchBAX", n_candidates=100)
    campaign = Campaign(spec, policy, EnsembleConstructor(n_members=5, epochs=120))
    campaign.seed(torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.double))
    for _ in range(6):
        x = campaign.suggest(1)[0]["x"]
        campaign.ingest(torch.tensor([-((x - 0.35) ** 2)]))
    region = campaign.policy.candidate_grid(spec).shape[0]
    print(f"{'deep ensemble (BAX)':>22}: ran {len(campaign.history)} steps over a "
          f"{region}-point grid (samples-UQ pairing)")


if __name__ == "__main__":
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    _bo_demo("Matern-5/2 GP", MaternConstructor(bounds=bounds))
    _bo_demo("SAAS fully-Bayesian", SaasConstructor(bounds=bounds))
    _ensemble_demo()
