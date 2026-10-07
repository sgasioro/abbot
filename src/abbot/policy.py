"""Acquisition policies: ask/tell over a botorch ``Model``.

The public contract is gest-api-aligned ask/tell (``suggest`` / ``ingest``), which
IS the agent loop and the Xopt-interop surface. A policy operates on a botorch
``Model`` (not a custom posterior), so GP / NN-ensemble surrogates are
interchangeable.

Each policy declares a ``uq_requirement`` capability:
  - "point"               -> needs only a point prediction (MeanBAX/greedy; a
                             deterministic NN satisfies it)
  - "samples"             -> needs coherent posterior function samples
                             (InfoBAX/SwitchBAX)
  - "calibrated_variance" -> miscalibration is fatal (uncertainty sampling / pure
                             exploration)
The loop validates the model<->policy pairing against this (see
``abbot.loop.check_uq_pairing``), turning the empirical "MeanBAX works on
deterministic NNs" finding into a guardrail so a non-expert (or agent) can't
silently pair a point-estimate model with a variance-hungry acquisition.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import torch
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.models.model import Model

from abbot.acq_opt import CandidateOptimizer, Continuous
from abbot.history import History
from abbot.problem import ProblemSpec

UQRequirement = Literal["point", "samples", "calibrated_variance"]


@runtime_checkable
class Policy(Protocol):
    """ask/tell acquisition policy over a botorch Model (gest-api-aligned)."""

    uq_requirement: UQRequirement

    def suggest(
        self, history: History, spec: ProblemSpec, model: Model, n: int = 1
    ) -> torch.Tensor:
        """Propose ``n`` candidates: (n, d)."""
        ...

    def ingest(self, results) -> None:
        """Fold evaluated results back in (stateful policies override)."""
        ...

    def recommend(
        self, history: History, spec: ProblemSpec, model: Model
    ) -> int | None:
        """Index of the design to hand over at end-of-run (None -> argmax)."""
        ...


class SimpleBO:
    """Single-objective BO: noisy log-EI + a CandidateOptimizer.

    Mirrors ``SBIdODesigner.step`` acquisition (qLogNoisyEI, X_baseline=history).
    Defaults to maximization; the loop/caller is responsible for sign-flipping a
    MINIMIZE objective before fitting (kept out of the policy to stay symmetric).
    """

    uq_requirement: UQRequirement = "calibrated_variance"

    def __init__(self, optimizer: CandidateOptimizer | None = None) -> None:
        self.optimizer = optimizer or Continuous()

    def suggest(
        self, history: History, spec: ProblemSpec, model: Model, n: int = 1
    ) -> torch.Tensor:
        X = history.clean.designs.double()
        acq = qLogNoisyExpectedImprovement(model, X_baseline=X)
        return self.optimizer.optimize(acq, spec, n)

    def ingest(self, results) -> None:  # stateless: history carries the state
        return None

    def recommend(
        self, history: History, spec: ProblemSpec, model: Model
    ) -> int | None:
        return None


class UncertaintySampling:
    """Pure exploration: propose where the posterior variance is largest.

    All variance -> miscalibration is fatal, hence ``calibrated_variance``.
    """

    uq_requirement: UQRequirement = "calibrated_variance"

    def __init__(self, optimizer: CandidateOptimizer | None = None) -> None:
        self.optimizer = optimizer or Continuous()

    def suggest(
        self, history: History, spec: ProblemSpec, model: Model, n: int = 1
    ) -> torch.Tensor:
        from botorch.acquisition import PosteriorStandardDeviation

        acq = PosteriorStandardDeviation(model)
        return self.optimizer.optimize(acq, spec, n)

    def ingest(self, results) -> None:
        return None

    def recommend(
        self, history: History, spec: ProblemSpec, model: Model
    ) -> int | None:
        return None
