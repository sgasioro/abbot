"""Where runs give usable results: a classifier learned from bad-region runs.

A bad-region run (``Record.bad_region``) says the design itself gave no usable
result: a dark region, a sample that decomposes. Its outputs mean nothing, so the
output models never see it. Instead, a separate classifier is trained on every run
that wasn't an apparatus fault (:attr:`History.judged`): usable or not. Proposals
weigh designs by its ``P(usable)``, and recommendations rank only designs with
``P(usable) >= 0.5``.

The classifier is a GP with Dirichlet-based classification (Milios et al., 2018,
"Dirichlet-based Gaussian processes for large-scale calibrated classification", as
GPyTorch's ``DirichletClassificationLikelihood``): each class's label is turned
into a regression target with its own noise, and an exact GP is fitted per class.
So it is fitted, conditioned and saved like abbot's other models. blop 0.7 used
the same model for the same purpose.
"""

from __future__ import annotations

import math

import torch

from abbot.history import History
from abbot.problem import ProblemSpec

# Designs with at least this chance of a usable result count as usable (for
# recommendations and BAX's candidates).
USABLE_P = 0.5

# The Dirichlet pseudo-count for the classes a run isn't in (GPyTorch's default).
ALPHA_EPSILON = 0.01


class Usability:
    """``P(usable)`` at designs, from a two-class Dirichlet GP.

    Call it on designs ``(..., d)`` for probabilities ``(...)``; it is
    differentiable, so acquisition functions can use it. ``log_prob`` gives the
    log.
    """

    def __init__(self, model) -> None:
        self.model = model  # a 2-output GP: latent (bad, usable)

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return self.log_prob(X).exp()

    def log_prob(self, X: torch.Tensor) -> torch.Tensor:
        """``log P(usable)`` at designs ``(..., d)``: ``(...)``.

        With two classes, the Dirichlet GP's class probability is ``E[sigmoid(f_usable
        - f_bad)]``; this uses the probit approximation of that expectation
        (MacKay, 1992), which is smooth in the posterior mean and variance.
        """
        post = self.model.posterior(X.double())
        mean, var = post.mean, post.variance  # (..., 2)
        diff = mean[..., 1] - mean[..., 0]
        spread = (var[..., 0] + var[..., 1]).clamp_min(0)
        return torch.nn.functional.logsigmoid(diff / torch.sqrt(1 + math.pi * spread / 8))


def dirichlet_targets(usable: torch.Tensor, alpha_epsilon: float = ALPHA_EPSILON):
    """Regression targets and noise variances ``(n, 2)`` for labels ``usable`` ``(n,)``:
    columns (bad, usable). As GPyTorch's ``DirichletClassificationLikelihood``."""
    onehot = torch.stack([~usable, usable], dim=-1).double()
    alpha = onehot + alpha_epsilon
    noise = torch.log(1 / alpha + 1)
    return alpha.log() - noise / 2, noise


def fit_usability(history: History, spec: ProblemSpec) -> Usability | None:
    """Fit the classifier on ``history``'s judged runs, or ``None`` when there's
    nothing to learn: no bad-region run, or no usable one."""
    judged = history.judged.records
    usable = torch.tensor([r.bad_region is None for r in judged], dtype=torch.bool)
    if not len(judged) or usable.all() or not usable.any():
        return None
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Normalize, Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    X = torch.stack([r.candidate.double().reshape(-1) for r in judged])
    Y, Yvar = dirichlet_targets(usable)
    model = SingleTaskGP(
        X, Y, Yvar,
        input_transform=Normalize(X.shape[-1], bounds=spec.bounds_tensor().double()),
        outcome_transform=Standardize(m=2),
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    model.eval()
    return Usability(model)
