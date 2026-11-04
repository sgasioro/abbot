"""Toy NN ensemble as a botorch ``EnsembleModel`` — the extensibility smoke test.

This is not a production neural-network surrogate. Its only job is to prove
the ModelConstructor seam is not GP-coupled: a non-GP surrogate expressed as a
botorch ``EnsembleModel`` satisfies ``.posterior(X)`` and yields coherent (fixed-
index) function samples, exactly as a GP does via pathwise/rsample. MC-dropout
maps onto this 1:1 — each masked forward pass is one ensemble member;
coherent BAX samples come from fixing the member index.

Scope reminder: MeanBAX/greedy runs on the posterior *mean* and needs only a
point prediction, so a deterministic version of this (ensemble size 1) is a
legitimate ``uq_requirement="point"`` model. Calibrated variance is a separate
(DKL) escalation, deferred.
"""

from __future__ import annotations

import torch
from botorch.models.ensemble import EnsembleModel


class ToyEnsembleModel(EnsembleModel):
    """A fixed bag of tiny MLPs. Each member is one ensemble sample.

    ``forward(X)`` returns (num_models, batch..., n, m): botorch's
    ``EnsemblePosterior`` treats the leading ``num_models`` dim as the sample set,
    so ``.posterior(X).mean`` averages members and ``.rsample`` with fixed base
    samples selects coherent members — the coherent-sample contract.
    """

    def __init__(self, models: list[torch.nn.Module], num_outputs: int = 1) -> None:
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self._num_outputs = num_outputs

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Stack per-member predictions on a new leading dim.
        preds = [m(X) for m in self.models]  # each (batch..., n, m)
        return torch.stack(preds, dim=0)  # (num_models, batch..., n, m)


def make_toy_mlp(d_in: int, d_out: int = 1, width: int = 16) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(d_in, width),
        torch.nn.Tanh(),
        torch.nn.Linear(width, d_out),
    ).double()
