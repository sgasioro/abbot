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


# --- BAX: goal-as-algorithm over a discrete candidate grid --------------------

BAXStrategy = Literal["US", "MeanBAX", "SwitchBAX", "InfoBAX"]

_BAX_UQ: dict[str, UQRequirement] = {
    "US": "calibrated_variance",
    "MeanBAX": "point",  # runs the goal on the posterior mean; a point net suffices
    "SwitchBAX": "samples",  # stuck -> InfoBAX, which needs coherent samples
    "InfoBAX": "samples",
}


class BAXPolicy:
    """Bayesian Algorithm Execution over a discrete candidate grid.

    One policy, strategy-dispatched over ``US`` / ``MeanBAX`` / ``SwitchBAX`` /
    ``InfoBAX`` — ports multibax-sklearn's acquisition layer onto a botorch
    ``Model`` (a multi-output ``ModelListGP`` for the theta-feature case). The
    ``Goal`` identifies the target subset of the grid; the acquisition focuses
    sampling to resolve that subset.

    Discrete: the acquisition is a length-N vector over a Sobol candidate grid
    (built once from the spec), selection is argmax, and already-collected grid
    indices are masked to ``-inf`` (prevent-requery). Continuous proposal
    (pathwise / seeded-ascent) is a deferred extension point.

    Outputs are read in RAW units before the goal runs (the model is canonical;
    ``objective_signs`` is un-applied), matching multibax's unnormalize step.
    """

    def __init__(
        self,
        goal,
        strategy: BAXStrategy = "SwitchBAX",
        *,
        n_candidates: int = 1000,
        n_posterior_samples: int = 20,
        candidate_seed: int = 0,
    ) -> None:
        if strategy not in _BAX_UQ:
            raise ValueError(f"strategy must be one of {list(_BAX_UQ)}, got {strategy!r}")
        self.goal = goal
        self.strategy = strategy
        self.n_candidates = n_candidates
        self.n_posterior_samples = n_posterior_samples
        self.candidate_seed = candidate_seed
        self.uq_requirement: UQRequirement = _BAX_UQ[strategy]
        self._grid: torch.Tensor | None = None
        self._collected: list[int] = []
        self.switched: bool = False  # last suggest fell back to the explore strategy

    # -- grid --

    def _ensure_grid(self, spec: ProblemSpec) -> torch.Tensor:
        if self._grid is None:
            from torch.quasirandom import SobolEngine

            bounds = spec.bounds_tensor()
            d = bounds.shape[-1]
            sobol = SobolEngine(dimension=d, scramble=True, seed=self.candidate_seed)
            draw = sobol.draw(self.n_candidates).double()
            grid = bounds[0] + draw * (bounds[1] - bounds[0])
            # Respect sparse inequality constraints (sum(coef*x[idx]) >= rhs).
            cons = spec.inequality_constraints()
            if cons:
                keep = torch.ones(grid.shape[0], dtype=torch.bool)
                for idx, coef, rhs in cons:
                    keep &= (grid[:, idx] * coef).sum(dim=-1) >= rhs
                grid = grid[keep]
            self._grid = grid
        return self._grid

    @property
    def collected_count(self) -> int:
        """How many grid points have been proposed so far (prevent-requery state)."""
        return len(self._collected)

    def _raw_mean_std(self, model, X, spec):
        """Posterior (mean, std) at X in RAW units: (N, d_y)."""
        signs = spec.objective_signs()
        with torch.no_grad():
            post = model.posterior(X.double())
            mean = post.mean.reshape(X.shape[0], -1) * signs
            std = post.variance.clamp_min(1e-12).sqrt().reshape(X.shape[0], -1)
        return mean, std

    # -- Policy API --

    def suggest(
        self, history: History, spec: ProblemSpec, model: Model, n: int = 1
    ) -> torch.Tensor:
        grid = self._ensure_grid(spec)
        acq = self._acquisition(spec, model, grid)
        acq = acq.clone()
        acq[self._collected] = float("-inf")
        picks = torch.topk(acq, k=min(n, acq.numel())).indices.tolist()
        self._collected.extend(picks)
        return grid[picks]

    def ingest(self, results) -> None:
        return None

    def recommend(
        self, history: History, spec: ProblemSpec, model: Model
    ) -> list[int]:
        # Hand over the observed non-dominated / best set (BAX's job is subset
        # discovery, not single-point optimization); defer to the shared helper.
        return _denoised_recommendation(history, spec, model)

    # -- strategy dispatch (mirrors multibax acquisition.py) --

    def _acquisition(self, spec, model, grid) -> torch.Tensor:
        self.switched = False
        mean, std = self._raw_mean_std(model, grid, spec)
        if self.strategy == "US":
            return std.mean(dim=-1)
        # MeanBAX / SwitchBAX gate on the mean-execution target set.
        if self.strategy in ("MeanBAX", "SwitchBAX"):
            target = self.goal.identify_subspace(mean, grid)
            if self._stuck(target):
                self.switched = True
                if self.strategy == "MeanBAX":
                    return std.mean(dim=-1)  # fall back to US
                return self._infobax(spec, model, grid, std)  # SwitchBAX -> InfoBAX
            acq = torch.zeros(grid.shape[0], dtype=torch.double)
            acq[target] = std.mean(dim=-1)[target]
            return acq
        # InfoBAX
        return self._infobax(spec, model, grid, std)

    def _stuck(self, target: list[int]) -> bool:
        """multibax ``meanbax_stuck``: empty target set, or already all collected."""
        return len(set(target)) == 0 or set(target).issubset(self._collected)

    def _infobax(self, spec, model, grid, std) -> torch.Tensor:
        """EIG via fantasy conditioning (multibax ``multiproperty_infobax``).

        acq = H[current] - E_sample[ H[posterior | execution(sample)] ], averaged
        over outputs. Coherent joint samples over the grid; condition the model on
        each sample's goal-target points; recompute summed latent entropy.
        """
        term1 = _entropy(std)  # (N, d_y) current predictive entropy
        signs = spec.objective_signs()
        with torch.no_grad():
            post = model.posterior(grid.double())
            # (n_samples, N, d_y) coherent joint draws; raw units for the goal.
            samples = post.rsample(torch.Size([self.n_posterior_samples]))
        term2 = torch.zeros_like(term1)
        n = self.n_posterior_samples
        for i in range(n):
            f_canon = samples[i].reshape(grid.shape[0], -1)
            f_raw = f_canon * signs
            target = self.goal.identify_subspace(f_raw, grid)
            if not target:
                term2 += term1  # no execution info -> entropy unchanged
                continue
            try:
                fant = model.condition_on_observations(
                    grid[target].double(), f_canon[target]
                )
                _, fstd = self._raw_mean_std(fant, grid, spec)
                term2 += _entropy(fstd)
            except Exception:
                logger.exception("InfoBAX fantasy conditioning failed for a sample.")
                term2 += term1
        term2 = term2 / n
        eig = (term1 - term2).mean(dim=-1)  # (N,) averaged over outputs
        return eig


def _entropy(std: torch.Tensor) -> torch.Tensor:
    """Gaussian differential entropy per candidate per output (multibax formula)."""
    import math

    var = std.clamp_min(1e-12) ** 2
    return 0.5 * torch.log(2 * math.pi * var) + 0.5
