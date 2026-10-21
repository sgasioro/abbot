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
    """ask/tell acquisition policy over a botorch Model (gest-api-aligned).

    Optionally, a policy sets ``last_acquisition_value`` after each ``suggest``: its
    acquisition value per pick, ``(n,)`` (or one value for a joint batch), in its
    own units. The campaign stores it on each record (``Record.acquisition``), which
    is what :func:`abbot.supervision.acquisition_monitor` stops on. Every built-in
    policy sets it.
    """

    uq_requirement: UQRequirement

    def suggest(
        self,
        history: History,
        spec: ProblemSpec,
        model: Model,
        n: int = 1,
        *,
        pending: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Propose ``n`` candidates: (n, d).

        ``pending`` ``(k, d)`` holds designs already handed out but not yet
        measured; a policy should steer away from duplicating them. ``None`` or
        empty means nothing is outstanding.
        """
        ...

    def ingest(self, results) -> None:
        """Fold evaluated results back in (stateful policies override)."""
        ...


# End-of-run handover ("which design do we recommend?") is NOT a policy concern:
# every policy wanted the same rule, so it lives once in ``session.recommend``.


def _acq_value(acq, X: torch.Tensor) -> float:
    """The acquisition value of the joint batch ``X (q, d)``."""
    with torch.no_grad():
        return float(acq(X.double().unsqueeze(0)).reshape(-1)[0])


class _CostWeighted(torch.nn.Module):
    """A log-space acquisition per unit cost: ``acq(X) - log(cost of the batch)``.

    For the log-valued acquisitions abbot uses (log-NEI, log-NEHVI), this is the
    expected improvement per unit cost. ``cost_model`` sees raw designs.
    """

    def __init__(self, acq, cost_model) -> None:
        super().__init__()
        self.acq = acq
        self.cost_model = cost_model
        self.model = acq.model

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        cost = self.cost_model(X).reshape(X.shape[:-1]).sum(dim=-1)
        return self.acq(X) - cost.clamp_min(1e-12).log()


def _cost_weighted(acq, cost_model, history: History, spec: ProblemSpec):
    if cost_model is None:
        return acq
    cost_model.fit(history, spec)
    return _CostWeighted(acq, cost_model)


def _objective_transform(spec: ProblemSpec):
    """A ``ScalarizedPosteriorTransform`` selecting the single objective column
    from a (possibly wider) model, or ``None`` when the model is already exactly
    the one objective (no OBSERVE columns) — the common case, avoids overhead.
    """
    if spec.n_outputs == 1:
        return None
    from botorch.acquisition.objective import ScalarizedPosteriorTransform

    weights = torch.zeros(spec.n_outputs, dtype=torch.double)
    weights[spec.objective_indices()[0]] = 1.0
    return ScalarizedPosteriorTransform(weights=weights)


def _pending_or_none(pending: torch.Tensor | None) -> torch.Tensor | None:
    if pending is None or pending.numel() == 0:
        return None
    return pending.double().reshape(pending.shape[0], -1)


def _extend_pending(
    pending: torch.Tensor | None, picks: list[torch.Tensor]
) -> torch.Tensor | None:
    """``pending`` plus this batch's earlier picks, ``(k, d)``, or None if both empty."""
    parts = ([] if pending is None else [pending]) + [p.double().reshape(len(p), -1) for p in picks]
    return torch.cat(parts) if parts else None


def condition_on_pending(model: Model, pending: torch.Tensor | None) -> Model:
    """Kriging believer: condition ``model`` on ``pending`` designs at its own
    posterior mean.

    The mean is unchanged; the variance shrinks at and near the pending designs, so
    a variance-driven acquisition stops proposing duplicates of work in flight.
    Returns ``model`` unchanged when nothing is pending or the model can't be
    conditioned (e.g. an ensemble), logging the latter.
    """
    pending = _pending_or_none(pending)
    if pending is None:
        return model
    try:
        with torch.no_grad():
            mean = model.posterior(pending).mean.reshape(pending.shape[0], -1)
            return _condition(model, pending, mean)
    except Exception as exc:  # noqa: BLE001 - any model may refuse; degrade gracefully
        logger.info(
            "Model %s can't condition on pending designs (%s: %s); proposals only "
            "exclude the exact pending points.",
            type(model).__name__, type(exc).__name__, exc,
        )
        return model


def _condition(
    model: Model, X: torch.Tensor, Y: torch.Tensor, noise: float | None = None
) -> Model:
    """``model.condition_on_observations(X, Y)`` for the surrogates abbot builds.

    Covers the two cases the plain call refuses: a ``ModelListGP`` (wants one X per
    output) is conditioned submodel by submodel, and a fixed-noise GP (fitted with
    per-point ``Yvar``) is given its mean training noise for the new points.
    ``noise`` instead sets the new points' noise variance explicitly, in the
    model's outcome-transformed units (InfoBAX's near-noise-free fantasies).
    """
    from botorch.models import ModelListGP
    from gpytorch.likelihoods import FixedNoiseGaussianLikelihood

    if isinstance(model, ModelListGP):
        subs, col = [], 0
        for sub in model.models:
            k = sub.num_outputs
            subs.append(_condition(sub, X, Y[..., col : col + k], noise))
            col += k
        return ModelListGP(*subs)
    kwargs = {}
    lik = getattr(model, "likelihood", None)
    if noise is not None:
        kwargs["noise"] = torch.full_like(Y, noise)
    elif isinstance(lik, FixedNoiseGaussianLikelihood):
        # Already in the model's outcome-transformed space, as botorch expects.
        level = lik.noise.reshape(-1, lik.noise.shape[-1]).mean(-1)  # (m,) or (1,)
        kwargs["noise"] = level.to(Y).expand(Y.shape)
    return model.condition_on_observations(X, Y, **kwargs)


class SimpleBO:
    """Single-objective BO: noisy log-EI + a CandidateOptimizer.

    The acquisition is qLogNoisyEI with the clean history as ``X_baseline``. With a
    ``cost_model`` (:mod:`abbot.cost`) it becomes log-EI minus log cost: expected
    improvement per unit cost, refitted to the recorded costs each step.
    Always maximizes: the loop fits the model in maximize-canonical space (it
    applies ``spec.output_signs``), so a MINIMIZE objective is handled upstream and
    the caller never pre-negates. When the model has extra OBSERVE outputs, a
    ``ScalarizedPosteriorTransform`` selects the single objective column. Refuses a
    multi-objective spec (use ParetoBO).
    """

    uq_requirement: UQRequirement = "calibrated_variance"

    def __init__(
        self, optimizer: CandidateOptimizer | None = None, *, cost_model=None
    ) -> None:
        self.optimizer = optimizer or Continuous()
        self.cost_model = cost_model
        self.last_acquisition_value: float | None = None

    def suggest(
        self,
        history: History,
        spec: ProblemSpec,
        model: Model,
        n: int = 1,
        *,
        pending: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if spec.n_objectives != 1:
            raise ValueError(
                f"SimpleBO is single-objective but the spec has {spec.n_objectives} "
                "objective(s); use ParetoBO for multi-objective optimization."
            )
        X = history.clean.designs.double()
        acq = qLogNoisyExpectedImprovement(
            model,
            X_baseline=X,
            posterior_transform=_objective_transform(spec),
            X_pending=_pending_or_none(pending),
        )
        acq = _cost_weighted(acq, self.cost_model, history, spec)
        picks = self.optimizer.optimize(acq, spec, n)
        self.last_acquisition_value = _acq_value(acq, picks)
        return picks

    def ingest(self, results) -> None:  # stateless: history carries the state
        return None


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

    ``cost_model`` (:mod:`abbot.cost`) makes it log-NEHVI minus log cost, as for
    :class:`SimpleBO`.
    """

    uq_requirement: UQRequirement = "calibrated_variance"

    def __init__(
        self,
        optimizer: CandidateOptimizer | None = None,
        ref_point: torch.Tensor | list[float] | None = None,
        min_clean: int | None = None,
        *,
        cost_model=None,
    ) -> None:
        self.optimizer = optimizer or Continuous()
        self.cost_model = cost_model
        self.last_acquisition_value: float | torch.Tensor | None = None
        self.ref_point = (
            None if ref_point is None else torch.as_tensor(ref_point, dtype=torch.double)
        )
        self.min_clean = min_clean  # None -> max(2, 2*m) resolved at suggest time

    def _canonical_Y(self, history: History, spec: ProblemSpec) -> torch.Tensor:
        """Objective columns only, in canonical space: (n_clean, n_objectives)."""
        return spec.objective_view(history.clean.outputs)

    def suggest(
        self,
        history: History,
        spec: ProblemSpec,
        model: Model,
        n: int = 1,
        *,
        pending: torch.Tensor | None = None,
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
            return self._explore(history, spec, model, n, pending)

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
                return UncertaintySampling(self.optimizer).suggest(
                    history, spec, model, n, pending=pending
                )
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
        # Model is full-width over all outputs; select the objective columns so
        # NEHVI's hypervolume is over objectives only (OBSERVE columns ignored).
        objective = None
        if spec.n_outputs != spec.n_objectives:
            from botorch.acquisition.multi_objective.objective import (
                IdentityMCMultiOutputObjective,
            )

            objective = IdentityMCMultiOutputObjective(outcomes=spec.objective_indices())
        acq = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=X,
            objective=objective,
            X_pending=_pending_or_none(pending),
        )
        acq = _cost_weighted(acq, self.cost_model, history, spec)
        picks = self.optimizer.optimize(acq, spec, n)
        self.last_acquisition_value = _acq_value(acq, picks)
        return picks

    def _explore(self, history, spec, model, n, pending) -> torch.Tensor:
        """Uncertainty sampling until the front can direct the search."""
        us = UncertaintySampling(self.optimizer)
        picks = us.suggest(history, spec, model, n, pending=pending)
        self.last_acquisition_value = us.last_acquisition_value
        return picks

    def ingest(self, results) -> None:
        return None


class UncertaintySampling:
    """Pure exploration: propose where the posterior variance is largest.

    All variance -> miscalibration is fatal, hence ``calibrated_variance``.
    Multi-output models are scalarized as the SUM of per-output posterior standard
    deviations (direction-agnostic, so no objective-sign handling is needed).
    A batch of ``n`` is built greedily, conditioning on each pick at the model's
    own mean (kriging believer) before the next.
    """

    uq_requirement: UQRequirement = "calibrated_variance"

    def __init__(self, optimizer: CandidateOptimizer | None = None) -> None:
        self.optimizer = optimizer or Continuous()
        self.last_acquisition_value: torch.Tensor | None = None

    def suggest(
        self,
        history: History,
        spec: ProblemSpec,
        model: Model,
        n: int = 1,
        *,
        pending: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from botorch.acquisition import PosteriorStandardDeviation

        # Pure exploration over ALL modeled outputs (direction-agnostic, incl.
        # OBSERVE columns) — keyed on n_outputs, not objective count. The std
        # acquisition is single-point, so a batch is built greedily: each pick joins
        # the pending set the model is conditioned on before the next.
        pending = _pending_or_none(pending)
        picks: list[torch.Tensor] = []
        values: list[float] = []
        for _ in range(n):
            cond = condition_on_pending(model, _extend_pending(pending, picks))
            if spec.n_outputs == 1:
                acq = PosteriorStandardDeviation(cond)
            else:
                acq = _SumPosteriorStd(cond)
            picks.append(self.optimizer.optimize(acq, spec, 1))
            values.append(_acq_value(acq, picks[-1]))
        self.last_acquisition_value = torch.tensor(values, dtype=torch.double)
        return torch.cat(picks)

    def ingest(self, results) -> None:
        return None


def default_policy(spec: ProblemSpec) -> Policy:
    """Pick the builtin policy a spec implies: SimpleBO for one objective, ParetoBO
    for several, each with the optimizer the domain supports (``default_optimizer``).

    A spec with no objectives (all OBSERVE) has nothing to optimize — that is a
    goal-driven (``BAXPolicy``) or exploration (``UncertaintySampling``) problem,
    so it is refused rather than guessed.
    """
    from abbot.acq_opt import default_optimizer

    m = spec.n_objectives
    if m == 0:
        raise ValueError(
            "The spec has no objectives (every output is OBSERVE), so there is no "
            "default policy. Pass one explicitly, e.g. BAXPolicy(goal) or "
            "UncertaintySampling()."
        )
    optimizer = default_optimizer(spec)  # may raise on mixed spaces
    return SimpleBO(optimizer) if m == 1 else ParetoBO(optimizer)


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
    indices are masked to ``-inf`` (prevent-requery). A batch of ``n`` is built
    greedily: each pick is added to the pending designs, which the model is
    conditioned on at its own mean (kriging believer) before the next pick. Continuous proposal
    (pathwise / seeded-ascent) is a deferred extension point.

    Outputs are read in RAW units before the goal runs (the model is canonical;
    ``output_signs`` is un-applied), matching multibax's unnormalize step.
    """

    def __init__(
        self,
        goal,
        strategy: BAXStrategy = "SwitchBAX",
        *,
        n_candidates: int = 1000,
        n_posterior_samples: int = 20,
        candidate_seed: int = 0,
        candidates: torch.Tensor | None = None,
        noise_at=None,
    ) -> None:
        if strategy not in _BAX_UQ:
            raise ValueError(f"strategy must be one of {list(_BAX_UQ)}, got {strategy!r}")
        self.goal = goal
        self.strategy = strategy
        self.n_candidates = n_candidates
        self.n_posterior_samples = n_posterior_samples
        self.candidate_seed = candidate_seed
        self.uq_requirement: UQRequirement = _BAX_UQ[strategy]
        # A caller-supplied candidate set (e.g. an application's own design grid) replaces
        # the Sobol grid; n_candidates / candidate_seed are then unused.
        self._grid: torch.Tensor | None = (
            None if candidates is None else candidates.double().reshape(len(candidates), -1)
        )
        # Grid indices measured or pending at the last suggest, plus its picks.
        # Recomputed every call from history + pending (nothing to lose on resume).
        self._collected: list[int] = []
        self.switched: bool = False  # last suggest left MeanBAX for any pick (SwitchBAX -> InfoBAX, MeanBAX -> US)
        self.last_acquisition_value: torch.Tensor | None = None  # per pick, last suggest
        # Optional ``noise_at(X) -> (N, m)`` measurement-noise variance (raw units) at
        # candidates, for InfoBAX; default: the model's own observation noise.
        self.noise_at = noise_at
        self._out_scale = torch.ones(1, dtype=torch.double)  # per-output, set by suggest

    # -- grid --

    def candidate_grid(self, spec: ProblemSpec) -> torch.Tensor:
        """The candidate grid ``(N, d)`` this policy proposes over.

        The ``candidates`` passed at construction, else a Sobol grid built once from
        the spec and cached (honoring sparse inequality constraints).
        Exposed because callers legitimately seed a campaign from the same grid the
        policy searches, so fits and proposals share one candidate set.
        """
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

    def _ensure_grid(self, spec: ProblemSpec) -> torch.Tensor:
        """Deprecated alias for :meth:`candidate_grid` (kept for one release)."""
        return self.candidate_grid(spec)

    @property
    def collected_count(self) -> int:
        """How many grid points were measured, pending or picked as of the last
        ``suggest`` (the prevent-requery set)."""
        return len(self._collected)

    def _raw_mean_std(self, model, X, spec):
        """Posterior (mean, std) at X in RAW units: (N, d_y).

        BAX reads ALL outputs raw (the goal consumes them); ``spec.to_raw`` applies
        the canonical->raw flip, the same rule the analysis module uses.
        """
        with torch.no_grad():
            post = model.posterior(X.double())
            mean = spec.to_raw(post.mean.reshape(X.shape[0], -1))
            std = post.variance.clamp_min(1e-12).sqrt().reshape(X.shape[0], -1)
        return mean, std

    # -- Policy API --

    def suggest(
        self,
        history: History,
        spec: ProblemSpec,
        model: Model,
        n: int = 1,
        *,
        pending: torch.Tensor | None = None,
    ) -> torch.Tensor:
        grid = self.candidate_grid(spec)
        pending = _pending_or_none(pending)
        self._out_scale = _output_scale(history)
        # A batch is built greedily (kriging believer): each pick joins the pending
        # designs, the model is conditioned on them at its own mean, and the
        # acquisition is recomputed before the next pick.
        picks: list[int] = []
        values: list[float] = []
        switched = False
        for _ in range(min(n, grid.shape[0])):
            inflight = _extend_pending(pending, [grid[picks]] if picks else [])
            # Prevent-requery: grid points already measured (any record) or in flight.
            taken = [history.designs.double()] if len(history) else []
            if inflight is not None:
                taken.append(inflight)
            self._collected = _grid_matches(grid, torch.cat(taken)) if taken else []
            acq = self._acquisition(spec, condition_on_pending(model, inflight), grid)
            switched |= self.switched
            acq = acq.clone()
            acq[self._collected] = float("-inf")
            picks.append(int(acq.argmax()))
            values.append(float(acq[picks[-1]]))
        self._collected.append(picks[-1])
        self.switched = switched
        self.last_acquisition_value = torch.tensor(values, dtype=torch.double)
        return grid[picks]

    def ingest(self, results) -> None:
        return None

    # -- strategy dispatch (mirrors multibax acquisition.py) --

    def _acquisition(self, spec, model, grid) -> torch.Tensor:
        self.switched = False
        mean, std = self._raw_mean_std(model, grid, spec)
        # Average uncertainty across outputs in scale-free units (multibax fits on
        # min-max-scaled outputs); raw stds would let the largest-unit output win.
        rel = (std / self._out_scale.to(std)).mean(dim=-1)
        if self.strategy == "US":
            return rel
        # MeanBAX / SwitchBAX gate on the mean-execution target set.
        if self.strategy in ("MeanBAX", "SwitchBAX"):
            target = self.goal.identify_subspace(mean, grid)
            if self._stuck(target):
                self.switched = True
                if self.strategy == "MeanBAX":
                    return rel  # fall back to US
                return self._infobax(spec, model, grid, std)  # SwitchBAX -> InfoBAX
            acq = torch.zeros(grid.shape[0], dtype=torch.double)
            acq[target] = rel[target]
            return acq
        # InfoBAX
        return self._infobax(spec, model, grid, std)

    def _stuck(self, target: list[int]) -> bool:
        """multibax ``meanbax_stuck``: empty target set, or already all collected."""
        return len(set(target)) == 0 or set(target).issubset(self._collected)

    def _measurement_noise(self, model, grid) -> torch.Tensor:
        """Measurement-noise variance ``(N, m)`` in raw units at each candidate:
        ``noise_at`` if given, else the model's (learned, or mean per-point) noise.
        """
        N = grid.shape[0]
        if self.noise_at is not None:
            return torch.as_tensor(self.noise_at(grid), dtype=torch.double).reshape(N, -1)
        try:
            with torch.no_grad():
                noisy = model.posterior(grid.double(), observation_noise=True).variance
                latent = model.posterior(grid.double()).variance
            return (noisy - latent).clamp_min(0).reshape(N, -1)
        except Exception as exc:  # noqa: BLE001 - e.g. a model with no likelihood
            logger.info(
                "InfoBAX: no observation noise from %s (%s); using none.",
                type(model).__name__, exc,
            )
            return torch.zeros(N, 1, dtype=torch.double)

    def _infobax(self, spec, model, grid, std) -> torch.Tensor:
        """EIG via fantasy conditioning (multibax ``multiproperty_infobax``).

        acq = H[y_x] - E_sample[ H[y_x | execution(sample)] ], averaged over
        outputs, where y_x is a noisy MEASUREMENT at x: both entropies include the
        measurement noise at x. Coherent joint samples over the grid; the model is
        conditioned (hyperparameters fixed) on each sample's goal-target values,
        which are function values, so with only a numerical jitter as noise.
        """
        noise = self._measurement_noise(model, grid)  # (N, m) or (N, 1)
        term1 = _entropy((std**2 + noise).sqrt())  # (N, d_y)
        with torch.no_grad():
            post = model.posterior(grid.double())
            # (n_samples, N, d_y) coherent joint draws; raw units for the goal.
            samples = post.rsample(torch.Size([self.n_posterior_samples]))
        term2 = torch.zeros_like(term1)
        n = self.n_posterior_samples
        for i in range(n):
            f_canon = samples[i].reshape(grid.shape[0], -1)
            f_raw = spec.to_raw(f_canon)  # BAX reads ALL outputs raw (goal consumes them)
            target = self.goal.identify_subspace(f_raw, grid)
            if not target:
                term2 += term1  # no execution info -> entropy unchanged
                continue
            try:
                fant = _condition(
                    model, grid[target].double(), f_canon[target], noise=_FANTASY_JITTER
                )
                _, fstd = self._raw_mean_std(fant, grid, spec)
                term2 += _entropy((fstd**2 + noise).sqrt())
            except Exception:
                logger.exception("InfoBAX fantasy conditioning failed for a sample.")
                term2 += term1
        term2 = term2 / n
        eig = (term1 - term2).mean(dim=-1)  # (N,) averaged over outputs
        return eig


# Noise variance (outcome-transformed units) for InfoBAX's fantasy points. In
# theory they are exact function values (zero noise); the jitter only keeps the
# update well-conditioned when the target set has many nearby grid points under
# a smooth kernel.
_FANTASY_JITTER = 1e-6


def _entropy(std: torch.Tensor) -> torch.Tensor:
    """Gaussian differential entropy per candidate per output (multibax formula)."""
    import math

    var = std.clamp_min(1e-12) ** 2
    return 0.5 * torch.log(2 * math.pi * var) + 0.5


def _grid_matches(grid: torch.Tensor, X: torch.Tensor) -> list[int]:
    """Indices of ``grid`` rows that equal some row of ``X``.

    Equal up to float32 round-off: callers that store designs in single precision
    must still match the double grid point they came from.
    """
    X = X.reshape(X.shape[0], -1).to(grid)
    close = torch.isclose(grid.unsqueeze(1), X.unsqueeze(0), rtol=1e-6, atol=1e-9)
    return close.all(dim=-1).any(dim=-1).nonzero().reshape(-1).tolist()


def _output_scale(history: History) -> torch.Tensor:
    """Per-output spread of the clean observed outputs ``(m,)`` (1 where unknown),
    used to put posterior stds of different outputs on a common scale."""
    clean = [r.outputs.reshape(-1) for r in history.clean.records if r.outputs is not None]
    if len(clean) < 2:
        return torch.ones(1, dtype=torch.double)
    sd = torch.stack(clean).double().std(dim=0)
    return torch.where(sd > 1e-12, sd, torch.ones_like(sd))
