"""The ModelConstructor seam: ``(X, Y, Yvar) -> botorch.Model``.

This is the key SBI seam. BoTorch's ``Model``/``Posterior`` ABC already *is* the
surrogate-agnostic layer, so abbot does NOT define a bespoke Posterior protocol —
a constructor just builds a botorch ``Model`` and every policy composes over its
``.posterior(X)`` unchanged.

Builtins cover single-output GP and multi-output ``ModelListGP``. Involved SBI
construction (correlated ``MultiTaskGP``/LCM, per-shot posterior variance via
``train_Yvar``, custom GPyTorch likelihoods) plugs in here WITHOUT touching the
loop or the policy. NN surrogates are botorch ``EnsembleModel`` subclasses (see
``abbot.models.ensemble``), not a parallel interface.

Basis: ``~/sbido/src/sbido/methods/sbido.py::SBIdODesigner.step`` GP build, and
Xopt's ``ModelConstructor`` ABC.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.model import Model
from botorch.models.transforms.input import Normalize
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood

# The UQ quality a model's posterior actually provides. This is a property of the
# surrogate, so it lives on the constructor that builds it (NOT a string the caller
# passes alongside — that could silently disagree with the real model). The loop
# checks ``policy.uq_requirement`` against this at fit time (see
# ``session.check_uq_pairing``). Ordering (weakest->strongest): a point estimate, a
# full set of posterior samples, calibrated variance.
UQCapability = str  # "point" | "samples" | "calibrated_variance"


@runtime_checkable
class ModelConstructor(Protocol):
    """Build a fitted botorch ``Model`` from training data.

    X:    (n, d) design inputs
    Y:    (n, m) observed outputs (m=1 for scalar objectives)
    Yvar: (n, m) optional observation-noise variances (-> train_Yvar). When None
          the model infers the noise level (sbido's choice: F is genuinely noisy).

    ``provides_uq`` declares the UQ quality the built model's posterior supplies, so
    the loop can refuse a policy that needs more than the surrogate can give.
    """

    provides_uq: UQCapability

    def build(
        self, X: torch.Tensor, Y: torch.Tensor, Yvar: torch.Tensor | None = None
    ) -> Model: ...


Kernel = Literal["default", "matern52"]


def _covar_module(kernel: Kernel, d: int):
    """``None`` keeps BoTorch's default kernel; ``"matern52"`` is a scaled ARD
    Matern-5/2 (the sklearn ``ConstantKernel * Matern(nu=2.5)`` multibax used)."""
    if kernel == "default":
        return None
    if kernel == "matern52":
        from gpytorch.kernels import MaternKernel, ScaleKernel

        return ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d))
    raise ValueError(f"kernel must be 'default' or 'matern52', got {kernel!r}")


class SingleTaskGPConstructor:
    """Scalar / independent-output GP with input normalization and MLL fit.

    Mirrors ``SBIdODesigner.step``: let the GP infer the noise level unless Yvar
    is supplied, normalize inputs to the problem bounds. A GP posterior is a full
    Gaussian, so it provides calibrated variance (the strongest UQ tier).
    """

    provides_uq = "calibrated_variance"

    def __init__(
        self, bounds: torch.Tensor | None = None, kernel: Kernel = "default"
    ) -> None:
        # bounds: (2, d) for input Normalize; None -> normalize from data.
        self.bounds = bounds
        self.kernel = kernel
        _covar_module(kernel, 1)  # validate early

    def build(
        self, X: torch.Tensor, Y: torch.Tensor, Yvar: torch.Tensor | None = None
    ) -> SingleTaskGP:
        X = X.double()
        Y = Y.double()
        d = X.shape[-1]
        input_transform = (
            Normalize(d=d, bounds=self.bounds.double())
            if self.bounds is not None
            else Normalize(d=d)
        )
        gp = SingleTaskGP(
            X,
            Y,
            train_Yvar=Yvar.double() if Yvar is not None else None,
            input_transform=input_transform,
            covar_module=_covar_module(self.kernel, d),
        )
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        return gp


class ModelListConstructor:
    """Independent GP per output -> ``ModelListGP`` (multi-output / multibax MGPR).

    Each column of Y gets its own ``SingleTaskGP``. This is the rewrite target for
    multibax's independent-sklearn-GP ``MGPR`` and the base for multi-output θ
    models in the SBI app. Each GP posterior is Gaussian -> calibrated variance.
    """

    provides_uq = "calibrated_variance"

    def __init__(
        self, bounds: torch.Tensor | None = None, kernel: Kernel = "default"
    ) -> None:
        self.bounds = bounds
        self.kernel = kernel
        _covar_module(kernel, 1)  # validate early

    def build(
        self, X: torch.Tensor, Y: torch.Tensor, Yvar: torch.Tensor | None = None
    ) -> ModelListGP:
        X = X.double()
        Y = Y.double()
        d = X.shape[-1]
        gps = []
        for j in range(Y.shape[-1]):
            input_transform = (
                Normalize(d=d, bounds=self.bounds.double())
                if self.bounds is not None
                else Normalize(d=d)
            )
            gps.append(
                SingleTaskGP(
                    X,
                    Y[..., j : j + 1],
                    train_Yvar=Yvar[..., j : j + 1].double() if Yvar is not None else None,
                    input_transform=input_transform,
                    covar_module=_covar_module(self.kernel, d),
                )
            )
        model = ModelListGP(*gps)
        fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
        return model


class EnsembleConstructor:
    """A deep-ensemble surrogate -> botorch ``EnsembleModel`` (the non-GP proof).

    Builds ``n_members`` small MLPs and trains each (independent init + a short Adam
    fit) so the spread across members is the uncertainty. This is the generality
    PROOF that the ``ModelConstructor`` seam is not GP-coupled — NOT nn-bax
    productionization (MC-dropout, deep kernels, calibration are F-tier).

    ``provides_uq`` is honest and load-bearing (it feeds ``check_uq_pairing``):
      - ``n_members > 1`` -> ``"samples"`` (coherent member draws; the InfoBAX/
        SwitchBAX contract), which also exposes a (member-spread) variance.
      - ``n_members == 1`` -> ``"point"`` (a single deterministic net; only MeanBAX/
        greedy may pair with it).
    An ensemble's spread is NOT calibrated variance, so it never claims that tier —
    pairing one with ``UncertaintySampling`` is correctly refused.

    ``Yvar`` is ignored: an ensemble represents noise through member disagreement,
    not an explicit per-point variance.
    """

    def __init__(
        self,
        n_members: int = 5,
        *,
        width: int = 16,
        epochs: int = 200,
        lr: float = 0.05,
        seed: int = 0,
    ) -> None:
        self.n_members = n_members
        self.width = width
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.provides_uq = "samples" if n_members > 1 else "point"

    def build(
        self, X: torch.Tensor, Y: torch.Tensor, Yvar: torch.Tensor | None = None
    ):
        from abbot.models.ensemble import ToyEnsembleModel, make_toy_mlp

        X = X.double()
        Y = Y.double()
        d, m = X.shape[-1], Y.shape[-1]
        g = torch.Generator().manual_seed(self.seed)
        members = []
        for _ in range(self.n_members):
            # Independent init per member: reseed torch's global RNG from our
            # generator (avoids Date/random bans elsewhere; deterministic by seed).
            torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=g)))
            net = make_toy_mlp(d, m, width=self.width)
            opt = torch.optim.Adam(net.parameters(), lr=self.lr)
            for _ in range(self.epochs):
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(net(X), Y)
                loss.backward()
                opt.step()
            members.append(net)
        return ToyEnsembleModel(members, num_outputs=m)


def default_constructor(spec, bounds: torch.Tensor | None = None) -> ModelConstructor:
    """Pick a sensible builtin: single-output GP for one OUTPUT, else ModelList.

    Keyed on ``n_outputs`` (every output is modeled, incl. OBSERVE), not objective
    count — a single-objective spec with an extra OBSERVE output needs a ModelList.
    Callers may still pass their own ``ModelConstructor`` (e.g. a correlated
    MultiTaskGP for SBI). ``bounds`` defaults to the spec's design bounds.
    """
    if bounds is None:
        bounds = spec.bounds_tensor()
    if spec.n_outputs == 1:
        return SingleTaskGPConstructor(bounds=bounds)
    return ModelListConstructor(bounds=bounds)
