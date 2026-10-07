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

import logging
from typing import Literal, Protocol, runtime_checkable

import torch
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.models.model import Model
from botorch.utils.multi_objective.pareto import is_non_dominated

from abbot.acq_opt import CandidateOptimizer, Continuous
from abbot.history import History
from abbot.problem import ProblemSpec

logger = logging.getLogger(__name__)

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
    ) -> list[int]:
        """Indices of the design(s) to hand over at end-of-run.

        Returns a list into ``history.records``: a single index for a scalar
        objective, the Pareto set for multi-objective. An empty list means "no
        opinion" (the loop falls back to the observed non-dominated set). This is
        the same set-of-indices shape a future BAX ``Goal``-identified subset uses.
        """
        ...


def _denoised_recommendation(
    history: History, spec: ProblemSpec, model: Model
) -> list[int]:
    """Pick handover index/indices by the canonical posterior MEAN, not the raw
    (noisy) observed values — the sbido denoising idea, generalized to m objectives.

    Scores the clean designs through ``model`` in maximize-canonical space:
      - m == 1: argmax of the posterior mean -> single index.
      - m  > 1: non-dominated set of the posterior-mean vectors -> index list.
    Indices are mapped back to full-history coordinates (faulted shots excluded).
    """
    keep = [i for i, r in enumerate(history.records) if not r.fault]
    if not keep:
        return []
    X = torch.stack([history.records[i].candidate for i in keep]).double()
    with torch.no_grad():
        mean = model.posterior(X).mean  # (len(keep), m) in canonical space
    if mean.shape[-1] == 1:
        best_local = int(mean.reshape(-1).argmax())
        return [keep[best_local]]
    nd = is_non_dominated(mean)  # bool (len(keep),)
    return [keep[i] for i in range(len(keep)) if bool(nd[i])]


class SimpleBO:
    """Single-objective BO: noisy log-EI + a CandidateOptimizer.

    Mirrors ``SBIdODesigner.step`` acquisition (qLogNoisyEI, X_baseline=history).
    Always maximizes: the loop fits the model in maximize-canonical space (it
    applies ``spec.objective_signs``), so a MINIMIZE objective is handled upstream
    and the caller never pre-negates. Refuses a multi-objective spec (use ParetoBO).
    """

    uq_requirement: UQRequirement = "calibrated_variance"

    def __init__(self, optimizer: CandidateOptimizer | None = None) -> None:
        self.optimizer = optimizer or Continuous()

    def suggest(
        self, history: History, spec: ProblemSpec, model: Model, n: int = 1
    ) -> torch.Tensor:
        if spec.n_objectives != 1:
            raise ValueError(
                f"SimpleBO is single-objective but the spec has {spec.n_objectives} "
                "objectives; use ParetoBO for multi-objective optimization."
            )
        X = history.clean.designs.double()
        acq = qLogNoisyExpectedImprovement(model, X_baseline=X)
        return self.optimizer.optimize(acq, spec, n)

    def ingest(self, results) -> None:  # stateless: history carries the state
        return None

    def recommend(
        self, history: History, spec: ProblemSpec, model: Model
    ) -> list[int]:
        return _denoised_recommendation(history, spec, model)


class ParetoBO:
    """Multi-objective BO via noisy expected hypervolume improvement (qLogNEHVI).

    Operates on the maximize-canonical model (the loop applies objective signs), so
    the reference point and hypervolume live in canonical space. ``ref_point`` is
    optional: by default it is inferred from the observed canonical outputs each
    step. Inference is made LOUD (not silent) to avoid the vanishing-hypervolume
    failure mode where an inferred ref point lands inside the data cloud and NEHVI
    stops directing the search:
      - below a cold-start floor of clean points it falls back to uncertainty
        sampling and logs that it did;
      - it logs the inferred ref point (and how to pin it);
      - it warns if the non-dominated set is empty / hypervolume is ~0.
    """

    uq_requirement: UQRequirement = "calibrated_variance"

    def __init__(
        self,
        optimizer: CandidateOptimizer | None = None,
        ref_point: torch.Tensor | list[float] | None = None,
        min_clean: int | None = None,
    ) -> None:
        self.optimizer = optimizer or Continuous()
        self.ref_point = (
            None if ref_point is None else torch.as_tensor(ref_point, dtype=torch.double)
        )
        self.min_clean = min_clean  # None -> max(2, 2*m) resolved at suggest time

    def _canonical_Y(self, history: History, spec: ProblemSpec) -> torch.Tensor:
        return history.clean.outputs.double() * spec.objective_signs()

    def suggest(
        self, history: History, spec: ProblemSpec, model: Model, n: int = 1
    ) -> torch.Tensor:
        m = spec.n_objectives
        if m < 2:
            raise ValueError(
                f"ParetoBO needs >=2 objectives but the spec has {m}; use SimpleBO."
            )
        Y = self._canonical_Y(history, spec)  # (n_clean, m) canonical
        floor = self.min_clean if self.min_clean is not None else max(2, 2 * m)
        if Y.shape[0] < floor:
            logger.info(
                "ParetoBO: only %d clean points (<%d floor); proposing via "
                "uncertainty sampling until the Pareto front is seeded.",
                Y.shape[0],
                floor,
            )
            return UncertaintySampling(self.optimizer).suggest(history, spec, model, n)

        if self.ref_point is not None:
            ref_point = self.ref_point
        else:
            from botorch.utils.multi_objective.hypervolume import (
                infer_reference_point,
            )

            nd_mask = is_non_dominated(Y)
            pareto_Y = Y[nd_mask]
            if pareto_Y.numel() == 0:
                logger.warning(
                    "ParetoBO: no non-dominated points to infer a reference point "
                    "from; proposals may be undirected. Supply ParetoBO(ref_point=...) "
                    "or add more seed points."
                )
                return UncertaintySampling(self.optimizer).suggest(history, spec, model, n)
            ref_point = infer_reference_point(pareto_Y)
            logger.info(
                "ParetoBO: inferred canonical ref_point=%s from %d non-dominated "
                "points. Pass ParetoBO(ref_point=...) to pin it.",
                ref_point.tolist(),
                int(nd_mask.sum()),
            )

        from botorch.acquisition.multi_objective.logei import (
            qLogNoisyExpectedHypervolumeImprovement,
        )

        X = history.clean.designs.double()
        acq = qLogNoisyExpectedHypervolumeImprovement(
            model=model, ref_point=ref_point, X_baseline=X
        )
        return self.optimizer.optimize(acq, spec, n)

    def ingest(self, results) -> None:
        return None

    def recommend(
        self, history: History, spec: ProblemSpec, model: Model
    ) -> list[int]:
        return _denoised_recommendation(history, spec, model)


class UncertaintySampling:
    """Pure exploration: propose where the posterior variance is largest.

    All variance -> miscalibration is fatal, hence ``calibrated_variance``.
    Multi-output models are scalarized as the SUM of per-output posterior standard
    deviations (direction-agnostic, so no objective-sign handling is needed).
    """

    uq_requirement: UQRequirement = "calibrated_variance"

    def __init__(self, optimizer: CandidateOptimizer | None = None) -> None:
        self.optimizer = optimizer or Continuous()

    def suggest(
        self, history: History, spec: ProblemSpec, model: Model, n: int = 1
    ) -> torch.Tensor:
        from botorch.acquisition import PosteriorStandardDeviation

        if spec.n_objectives == 1:
            acq = PosteriorStandardDeviation(model)
        else:
            acq = _SumPosteriorStd(model)
        return self.optimizer.optimize(acq, spec, n)

    def ingest(self, results) -> None:
        return None

    def recommend(
        self, history: History, spec: ProblemSpec, model: Model
    ) -> list[int]:
        return _denoised_recommendation(history, spec, model)


class _SumPosteriorStd(torch.nn.Module):
    """Acquisition: sum over outputs of the posterior std at each candidate.

    A minimal multi-output analogue of botorch's PosteriorStandardDeviation; used
    by UncertaintySampling when the model has >1 output. Expects the standard
    acquisition input shape ``(b, q, d)`` and returns ``(b,)``.
    """

    def __init__(self, model: Model) -> None:
        super().__init__()
        self.model = model

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        posterior = self.model.posterior(X)
        std = posterior.variance.clamp_min(1e-12).sqrt()  # (b, q, m)
        return std.sum(dim=-1).squeeze(-1)  # (b,)
