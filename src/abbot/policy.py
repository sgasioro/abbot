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
from botorch.acquisition import (
    qLogNoisyExpectedImprovement,
    qMultiFidelityMaxValueEntropy,
)
from botorch.models.model import Model
from botorch.utils.multi_objective.pareto import is_non_dominated

from abbot.acq_opt import CandidateOptimizer, Continuous, optimize
from abbot.feasibility import (
    FEASIBLE_P,
    LogProbFeasible,
    any_feasible,
    feasible,
    mc_constraints,
    prob_feasible,
    row_scale,
)
from abbot.history import History
from abbot.problem import ContinuousDomain, ProblemSpec

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

        A policy may also take ``fixed: dict[str, float] | None``: inputs to hold at
        given values in every proposal (``Campaign.suggest(fixed=...)``). The
        builtins do; a policy without it is refused when ``fixed`` is used.
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

    # Sequential batch builders (``optimize_acqf_mixed``) set pending points.
    @property
    def X_pending(self):
        return self.acq.X_pending

    def set_X_pending(self, X_pending=None) -> None:
        self.acq.set_X_pending(X_pending)


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


def _objective_kwargs(spec: ProblemSpec, history: History) -> dict:
    """How a single-objective MC acquisition reads the model: through
    ``_objective_transform``, or with output constraints, an MC objective selecting
    the objective column plus the constraints (BoTorch applies constraints to the
    model's outputs, which a posterior transform would already have reduced)."""
    if not spec.has_output_constraints:
        return {"posterior_transform": _objective_transform(spec)}
    from botorch.acquisition.objective import GenericMCObjective

    kwargs = {"constraints": mc_constraints(spec, row_scale(history, spec))}
    if spec.n_outputs > 1:
        j = spec.objective_indices()[0]
        kwargs["objective"] = GenericMCObjective(lambda Z, X=None: Z[..., j])
    return kwargs


def _pinned(spec: ProblemSpec, fixed: dict[str, float] | None) -> dict[int, float]:
    """Inputs a proposal holds fixed, ``{position: value}``: sources at their
    target, overridden by ``fixed`` (``{name: value}``)."""
    return {**spec.target_features(), **spec.resolve_fixed(fixed)}


def _find_feasible(policy, history, spec, model, n, pending, fixed=None) -> torch.Tensor:
    """Propose ``n`` designs at the target by the probability of satisfying every
    output constraint, while no run at the target has. Improvement and
    hypervolume have no feasible reference yet, so they would ignore the
    constraints. Sets ``policy.last_acquisition_value``."""
    logger.info(
        "%s: no run satisfies the output constraints yet; searching for designs that "
        "will.", type(policy).__name__,
    )
    scale = row_scale(history, spec)
    pending = _pending_or_none(pending)
    picks, values = [], []
    for _ in range(n):
        cond = condition_on_pending(model, _extend_pending(pending, picks))
        acq = LogProbFeasible(cond, spec, explore=True, scale=scale)
        picks.append(optimize(policy.optimizer, acq, spec, 1, _pinned(spec, fixed)))
        values.append(_acq_value(acq, picks[-1]))
    policy.last_acquisition_value = torch.tensor(values, dtype=torch.double)
    return torch.cat(picks)


class _FeasibilityWeighted(torch.nn.Module):
    """An acquisition times the probability that the design, moved to the target,
    satisfies every output constraint. For a positive acquisition (entropy search)."""

    def __init__(self, acq, model, spec: ProblemSpec) -> None:
        super().__init__()
        self.acq = acq
        self.model = model
        self.spec = spec

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        pof = LogProbFeasible(self.model, self.spec)(self.spec.at_target(X)).exp()
        return self.acq(X) * pof


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

    **With sources** it chooses them too: multi-fidelity max-value entropy search,
    the information a run gives about the best value *at the target*, per unit of
    cost. The cost is ``cost_model`` if given, else a :class:`abbot.cost.SourceCost`
    learned from the recorded run times. A batch is built one pick at a time, each
    accounting for the earlier ones. ``choose_sources=False`` keeps every proposal
    at the target instead (cheap runs then only inform the model).
    ``n_candidates`` sets how many target designs sample the best value.

    **With output constraints** (``OutputSpec`` bounds or ``output_constraints``),
    the improvement only counts where the constraints hold, weighted by their
    probability (constrained log-NEI). Until some run satisfies them, it proposes by
    the probability that a design will, favoring designs where that is uncertain.
    With sources, the entropy search is weighted by that probability at the target
    and the best value is sampled over designs likely to satisfy the constraints (an
    approximation).
    """

    uq_requirement: UQRequirement = "calibrated_variance"

    def __init__(
        self,
        optimizer: CandidateOptimizer | None = None,
        *,
        cost_model=None,
        choose_sources: bool = True,
        n_candidates: int = 1000,
    ) -> None:
        self.optimizer = optimizer or Continuous()
        self.cost_model = cost_model
        self.choose_sources = choose_sources
        self.n_candidates = n_candidates
        self._source_cost = None
        self.last_acquisition_value: float | torch.Tensor | None = None

    def suggest(
        self,
        history: History,
        spec: ProblemSpec,
        model: Model,
        n: int = 1,
        *,
        pending: torch.Tensor | None = None,
        fixed: dict[str, float] | None = None,
    ) -> torch.Tensor:
        if spec.n_objectives != 1:
            raise ValueError(
                f"SimpleBO is single-objective but the spec has {spec.n_objectives} "
                "objective(s); use ParetoBO for multi-objective optimization."
            )
        if spec.has_output_constraints and not any_feasible(history, spec):
            return _find_feasible(self, history, spec, model, n, pending, fixed)
        if spec.sources and self.choose_sources:
            return self._suggest_with_sources(history, spec, model, n, pending, fixed)
        # Improvement is judged at the target: a lucky cheap-source run must not set
        # the bar. Without sources this is the clean designs unchanged.
        X = spec.at_target(history.clean.designs.double()).unique(dim=0)
        acq = qLogNoisyExpectedImprovement(
            model,
            X_baseline=X,
            X_pending=_pending_or_none(pending),
            **_objective_kwargs(spec, history),
        )
        acq = _cost_weighted(acq, self.cost_model, history, spec)
        picks = optimize(self.optimizer, acq, spec, n, _pinned(spec, fixed))
        self.last_acquisition_value = _acq_value(acq, picks)
        return picks

    @property
    def source_cost(self):
        """The cost model choosing sources: ``cost_model`` if given, else the
        :class:`abbot.cost.SourceCost` learned from the records (``None`` until the
        first proposal with sources)."""
        return self.cost_model if self.cost_model is not None else self._source_cost

    def _suggest_with_sources(self, history, spec, model, n, pending, fixed=None):
        from botorch.acquisition.cost_aware import InverseCostWeightedUtility
        from botorch.models.deterministic import GenericDeterministicModel

        cost_model = self.cost_model
        if cost_model is None:
            from abbot.cost import SourceCost

            if self._source_cost is None:
                self._source_cost = SourceCost(spec)
            cost_model = self._source_cost
        cost_model.fit(history, spec)
        candidates = spec.at_target(spec.sample_designs(self.n_candidates))
        if spec.has_output_constraints:
            # The best value worth learning about is among designs that satisfy
            # the constraints: sample it over the likely ones (at least a few).
            p = prob_feasible(spec, model, candidates)
            likely = p >= FEASIBLE_P
            if int(likely.sum()) < 10:
                likely = p >= p.topk(min(10, len(p))).values[-1]
            candidates = candidates[likely]
        # Costs relative to a typical target run, so the values read as information
        # per target-run-equivalent.
        scale = cost_model(candidates).mean().clamp_min(1e-300)
        utility = InverseCostWeightedUtility(
            GenericDeterministicModel(
                lambda X: (cost_model(X) / scale).clamp_min(1e-12).unsqueeze(-1)
            )
        )
        train_inputs = spec.at_target(history.clean.designs.double()).unique(dim=0)
        objective_model = _objective_model(model, spec)
        picks, values = [], []
        for _ in range(n):  # entropy search proposes one design per call
            acq = _MultiFidelityMES(
                objective_model,
                candidates,
                X_pending=_extend_pending(_pending_or_none(pending), picks),
                cost_aware_utility=utility,
                project=spec.at_target,
                train_inputs=train_inputs,
            )
            if spec.has_output_constraints:
                acq = _FeasibilityWeighted(acq, model, spec)
            x = optimize(self.optimizer, acq, spec, 1, spec.resolve_fixed(fixed))
            picks.append(x)
            values.append(_acq_value(acq, x))
        self.last_acquisition_value = torch.tensor(values, dtype=torch.double)
        return torch.cat(picks)

    def ingest(self, results) -> None:  # stateless: history carries the state
        return None


def _objective_model(model: Model, spec: ProblemSpec) -> Model:
    """The single-output model of the objective, for entropy search, which refuses
    multi-output models even with a posterior transform (a spec with OBSERVE or
    constrained outputs has one)."""
    if model.num_outputs == 1:
        return model
    j = spec.objective_indices()[0]
    try:
        return model.subset_output([j])
    except NotImplementedError as exc:
        raise ValueError(
            f"Choosing sources needs a model that can give the objective's output "
            f"alone (subset_output); {type(model).__name__} can't."
        ) from exc


class _MultiFidelityMES(qMultiFidelityMaxValueEntropy):
    """BoTorch's multi-fidelity max-value entropy search, with the max values drawn
    once, over the candidates and the raw training designs at the target.

    Upstream (0.18) appends the model's *transformed* training inputs to the
    candidates and draws before ``project`` is set, so they are evaluated as raw
    designs off the target (NaN for a log-scaled source at its lower bound, whose
    transformed value is 0).
    """

    def __init__(self, model, candidate_set, *, train_inputs, **kwargs) -> None:
        self._ready = False
        super().__init__(model, candidate_set, **kwargs)
        self.candidate_set = self.project(torch.cat([candidate_set, train_inputs]))
        self._ready = True
        self._sample_max_values(num_samples=self.num_mv_samples, X_pending=kwargs.get("X_pending"))

    def _sample_max_values(self, num_samples, X_pending=None):
        if self._ready:
            super()._sample_max_values(num_samples, X_pending=X_pending)


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

    **With output constraints**, only runs that satisfy them form the front and set
    the inferred reference point, and the hypervolume improvement is weighted by
    the probability they hold. Until some run satisfies them (and without a
    ``ref_point``), it proposes by that probability, as :class:`SimpleBO` does.
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
        fixed: dict[str, float] | None = None,
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
            return self._explore(history, spec, model, n, pending, fixed)
        constrained = spec.has_output_constraints
        if constrained:
            if self.ref_point is None and not any_feasible(history, spec):
                return _find_feasible(self, history, spec, model, n, pending, fixed)
            Y = Y[feasible(spec, history.clean.outputs.double())]

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
                    history, spec, model, n, pending=pending, fixed=fixed
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

        X = spec.at_target(history.clean.designs.double()).unique(dim=0)
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
            constraints=mc_constraints(spec, row_scale(history, spec)) if constrained else None,
        )
        acq = _cost_weighted(acq, self.cost_model, history, spec)
        picks = optimize(self.optimizer, acq, spec, n, _pinned(spec, fixed))
        self.last_acquisition_value = _acq_value(acq, picks)
        return picks

    def _explore(self, history, spec, model, n, pending, fixed=None) -> torch.Tensor:
        """Uncertainty sampling until the front can direct the search."""
        us = UncertaintySampling(self.optimizer)
        picks = us.suggest(history, spec, model, n, pending=pending, fixed=fixed)
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
        fixed: dict[str, float] | None = None,
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
            picks.append(optimize(self.optimizer, acq, spec, 1, _pinned(spec, fixed)))
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
    optimizer = default_optimizer(spec)
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

    **With sources**, the goal always runs at the target, and InfoBAX (also as
    SwitchBAX's fallback) chooses sources: it scores every grid design at a ladder of
    source levels by the information a run there gives about the goal's target set,
    per unit of cost. The cost is ``cost_model`` if given, else a
    :class:`abbot.cost.SourceCost` learned from the recorded costs. ``source_levels``
    is how many levels each ordered source gets (evenly spaced on the model's scale,
    plus the target), or ``{name: [values]}``; an unordered source uses all its
    values. US, MeanBAX and SwitchBAX's MeanBAX phase propose at the target.
    ``choose_sources=False`` keeps InfoBAX at the target too.

    **With output constraints** and ``respect_constraints=True`` (the default), the
    goal only sees the designs whose outputs satisfy them: on the posterior mean for
    MeanBAX, and on each posterior sample for InfoBAX. The goal runs on those rows,
    so ``TopK(k=5)`` means the five best designs that satisfy the constraints.
    ``respect_constraints=False`` lets the goal see every design, for bounds you want
    to map rather than filter by. When the constraints leave the goal nothing to
    select, the policy says so in ``last_warning`` (which ``Campaign`` passes on).

    If the goal selects nothing on any posterior sample, InfoBAX has no
    information to rank by; it proposes by uncertainty instead and says so in
    ``last_warning``.
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
        cost_model=None,
        choose_sources: bool = True,
        source_levels: int | dict[str, list[float]] = 5,
        respect_constraints: bool = True,
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
        self._candidates: torch.Tensor | None = (
            None if candidates is None else candidates.double().reshape(len(candidates), -1)
        )
        self._grid: torch.Tensor | None = None  # built from the spec on first use
        self._proposals: torch.Tensor | None = None  # the grid at each source level
        self.cost_model = cost_model
        self.choose_sources = choose_sources
        self.source_levels = source_levels
        self._source_cost = None
        self._cost: torch.Tensor | None = None  # relative cost per proposal, this suggest
        # Grid indices measured or pending at the last suggest, plus its picks.
        # Recomputed every call from history + pending (nothing to lose on resume).
        self._collected: list[int] = []
        self.switched: bool = False  # last suggest left MeanBAX for any pick (SwitchBAX -> InfoBAX, MeanBAX -> US)
        self.last_acquisition_value: torch.Tensor | None = None  # per pick, last suggest
        # Optional ``noise_at(X) -> (N, m)`` measurement-noise variance (raw units) at
        # candidates, for InfoBAX; default: the model's own observation noise.
        self.noise_at = noise_at
        self.respect_constraints = respect_constraints
        self._bound = None  # (spec, goal with names resolved), see _goal_for
        self.last_warning: str | None = None  # set by suggest when the goal is empty
        self._out_scale = torch.ones(1, dtype=torch.double)  # per-output, set by suggest

    # -- grid --

    def candidate_grid(self, spec: ProblemSpec) -> torch.Tensor:
        """The candidate grid ``(N, d)`` this policy proposes over.

        The ``candidates`` passed at construction, else a Sobol grid built once from
        the spec and cached (honoring the domain constraints; integer and enumerated
        variables at allowed values; ``log=True`` ones log-uniform). With sources,
        candidates are proposed at the target: a grid over the design variables only
        gets the target columns appended, and a full-width one is moved to the target
        (MeanBAX / SwitchBAX / US don't choose sources).
        Exposed because callers legitimately seed a campaign from the same grid the
        policy searches, so fits and proposals share one candidate set.
        """
        if self._grid is None and self._candidates is not None:
            self._grid = self._at_target(spec, self._candidates)
        if self._grid is None:
            from torch.quasirandom import SobolEngine

            n_design = len(spec.design_names)
            sobol = SobolEngine(dimension=n_design, scramble=True, seed=self.candidate_seed)
            draw = sobol.draw(self.n_candidates).double()
            # Integer and enumerated variables take allowed values; log=True ones
            # are spread log-uniformly.
            grid = spec._from_unit(draw, spec.domain)
            if any(not isinstance(d, ContinuousDomain) for d in spec.domain):
                grid = grid.unique(dim=0)  # snapping can repeat a design
            grid = self._at_target(spec, grid)
            # Respect sparse inequality constraints (sum(coef*x[idx]) >= rhs).
            cons = spec.inequality_constraints()
            if cons:
                keep = torch.ones(grid.shape[0], dtype=torch.bool)
                for idx, coef, rhs in cons:
                    keep &= (grid[:, idx] * coef).sum(dim=-1) >= rhs
                grid = grid[keep]
            self._grid = grid
        return self._grid

    @staticmethod
    def _at_target(spec: ProblemSpec, grid: torch.Tensor) -> torch.Tensor:
        """A design-only or full-width grid as full-width rows at the target."""
        if not spec.sources:
            return grid
        n_design = len(spec.design_names)
        if grid.shape[-1] == n_design:
            target = torch.tensor(list(spec.target_features().values()), dtype=grid.dtype)
            grid = torch.cat([grid, target.expand(grid.shape[0], -1)], dim=-1)
        elif grid.shape[-1] != len(spec.variable_names):
            raise ValueError(
                f"candidates have {grid.shape[-1]} column(s); expected {n_design} "
                f"(design variables {spec.design_names}) or {len(spec.variable_names)} "
                f"(with sources {spec.source_names})"
            )
        return spec.at_target(grid).unique(dim=0)

    def proposal_set(self, spec: ProblemSpec) -> torch.Tensor:
        """What this policy proposes from: the candidate grid, or with sources (when
        InfoBAX may choose them) the grid at every combination of source levels. The
        first ``len(candidate_grid(spec))`` rows are the grid at the target."""
        grid = self.candidate_grid(spec)
        if not self._chooses_sources(spec):
            return grid
        if self._proposals is None:
            from itertools import product

            columns = spec.source_indices()
            blocks = []
            for combo in product(*self._levels(spec)):  # the all-target combo first
                block = grid.clone()
                block[:, columns] = torch.tensor(combo, dtype=grid.dtype)
                blocks.append(block)
            self._proposals = torch.cat(blocks)
        return self._proposals

    def _chooses_sources(self, spec: ProblemSpec) -> bool:
        return bool(spec.sources) and self.choose_sources and self.strategy in (
            "InfoBAX", "SwitchBAX"
        )

    def _levels(self, spec: ProblemSpec) -> list[list[float]]:
        """Each source's levels, the target first."""
        import math

        from abbot.problem import EnumeratedDomain, IntegerDomain

        explicit = self.source_levels if isinstance(self.source_levels, dict) else {}
        unknown = set(explicit) - set(spec.source_names)
        if unknown:
            raise ValueError(f"source_levels names {sorted(unknown)}, not sources {spec.source_names}")
        levels = []
        for source in spec.sources:
            d = source.domain
            if source.name in explicit:
                values = [
                    d.encode(v) if isinstance(d, EnumeratedDomain) else float(v)
                    for v in explicit[source.name]
                ]
            elif isinstance(d, EnumeratedDomain):
                values = list(d.codes)
            else:
                k = 5 if isinstance(self.source_levels, dict) else int(self.source_levels)
                if k < 2:
                    raise ValueError(f"source_levels must be at least 2, got {k}")
                t = [i / (k - 1) for i in range(k)]
                if source.is_log:
                    lo, hi = math.log(d.lower), math.log(d.upper)
                    values = [math.exp(lo + u * (hi - lo)) for u in t]
                else:
                    values = [d.lower + u * (d.upper - d.lower) for u in t]
                if isinstance(d, IntegerDomain):
                    values = [float(round(v)) for v in values]
            target = source.target_code
            levels.append([target, *sorted({v for v in values if v != target})])
        return levels

    @property
    def source_cost(self):
        """The cost model choosing sources: ``cost_model`` if given, else the
        :class:`abbot.cost.SourceCost` learned from the records (``None`` until the
        first proposal with sources)."""
        return self.cost_model if self.cost_model is not None else self._source_cost

    def _relative_cost(self, history: History, spec: ProblemSpec, proposals) -> torch.Tensor:
        """Each proposal's predicted cost relative to a typical run at the target."""
        cost_model = self.cost_model
        if cost_model is None:
            from abbot.cost import SourceCost

            if self._source_cost is None:
                self._source_cost = SourceCost(spec)
            cost_model = self._source_cost
        cost_model.fit(history, spec)
        with torch.no_grad():
            cost = cost_model(proposals).reshape(-1).double()
        n_target = len(self.candidate_grid(spec))
        return (cost / cost[:n_target].mean().clamp_min(1e-300)).clamp_min(1e-12)

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
        fixed: dict[str, float] | None = None,
    ) -> torch.Tensor:
        grid = self.proposal_set(spec)
        self.last_warning = None
        allowed = self._matching(spec, grid, fixed)
        pending = _pending_or_none(pending)
        self._out_scale = _output_scale(history)
        self._cost = (
            self._relative_cost(history, spec, grid) if self._chooses_sources(spec) else None
        )
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
            if allowed is not None:
                acq[~allowed] = float("-inf")
                if not torch.isfinite(acq[allowed]).any():
                    raise ValueError(
                        f"BAXPolicy: every candidate with {fixed} is already measured "
                        "or in flight"
                    )
            picks.append(int(acq.argmax()))
            values.append(float(acq[picks[-1]]))
        self._collected.append(picks[-1])
        self.switched = switched
        self.last_acquisition_value = torch.tensor(values, dtype=torch.double)
        return grid[picks]

    @staticmethod
    def _matching(spec, grid, fixed) -> torch.Tensor | None:
        """Rows of ``grid`` with every ``fixed`` input at its value (None: all)."""
        pinned = spec.resolve_fixed(fixed)
        if not pinned:
            return None
        ok = torch.ones(grid.shape[0], dtype=torch.bool)
        for j, v in pinned.items():
            ok &= torch.isclose(grid[:, j], torch.tensor(v, dtype=grid.dtype))
        if not ok.any():
            names = spec.variable_names
            raise ValueError(
                f"BAXPolicy proposes from its candidate grid, and no candidate has "
                f"{ {names[j]: v for j, v in pinned.items()} }. Fix an integer or "
                "enumerated variable (the grid takes their values), or pass "
                "candidates= that include the fixed values."
            )
        return ok

    def ingest(self, results) -> None:
        return None

    # -- strategy dispatch (mirrors multibax acquisition.py) --

    def _acquisition(self, spec, model, grid) -> torch.Tensor:
        """The acquisition over ``grid`` (the proposal set). Its first
        ``len(candidate_grid)`` rows are at the target; only InfoBAX scores the rest
        (the other strategies leave them at ``-inf``)."""
        self.switched = False
        n_target = len(self.candidate_grid(spec))
        at_target = grid[:n_target]

        def at_target_only(acq):
            rest = torch.full((grid.shape[0] - n_target,), float("-inf"), dtype=acq.dtype)
            return torch.cat([acq, rest])

        mean, std = self._raw_mean_std(model, at_target, spec)
        # Average uncertainty across outputs in scale-free units (multibax fits on
        # min-max-scaled outputs); raw stds would let the largest-unit output win.
        rel = (std / self._out_scale.to(std)).mean(dim=-1)
        if self.strategy == "US":
            return at_target_only(rel)
        # MeanBAX / SwitchBAX gate on the mean-execution target set.
        if self.strategy in ("MeanBAX", "SwitchBAX"):
            target = self._identify(spec, mean, at_target)
            if not target and self._constrains(spec) and len(self._goal_for(spec).identify_subspace(mean, at_target)):
                self._warn(
                    "the output constraints exclude every design the goal selects on the "
                    "posterior mean"
                )
            if self._stuck(target):
                self.switched = True
                if self.strategy == "MeanBAX":
                    return at_target_only(rel)  # fall back to US
                return self._infobax(spec, model, grid, std)  # SwitchBAX -> InfoBAX
            acq = torch.zeros(n_target, dtype=torch.double)
            acq[target] = rel[target]
            return at_target_only(acq)
        # InfoBAX
        return self._infobax(spec, model, grid, std)

    def _goal_for(self, spec):
        """The goal with output names resolved against ``spec`` (``bind``), once."""
        if self._bound is None or self._bound[0] is not spec:
            bind = getattr(self.goal, "bind", None)
            self._bound = (spec, bind(spec) if bind is not None else self.goal)
        return self._bound[1]

    def _constrains(self, spec) -> bool:
        return self.respect_constraints and spec.has_output_constraints

    def _identify(self, spec, Y, X) -> list[int]:
        """The goal's target set on RAW outputs ``Y`` at designs ``X``. With output
        constraints (and ``respect_constraints``), the goal runs on only the rows
        that satisfy them, and its indices are mapped back to ``Y``'s rows."""
        goal = self._goal_for(spec)
        if not self._constrains(spec):
            return list(goal.identify_subspace(Y, X))
        rows = feasible(spec, Y).nonzero(as_tuple=True)[0]
        if not len(rows):
            return []
        target = goal.identify_subspace(Y[rows], X[rows])
        return [int(rows[i]) for i in target]

    def _warn(self, msg: str) -> None:
        msg = f"BAXPolicy: {msg}."
        if msg != self.last_warning:
            logger.warning("%s", msg)
        self.last_warning = msg

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
        measurement noise at x. Coherent joint samples over the grid at the target
        (its first ``len(std)`` rows, whose posterior std is ``std``), where the goal
        runs; the model is conditioned (hyperparameters fixed) on each sample's
        goal-target values, which are function values, so with only a numerical
        jitter as noise. Rows at other source levels are scored the same way, and
        with sources every row's EIG is divided by its relative cost.
        """
        n_target = std.shape[0]
        at_target = grid[:n_target]
        if grid.shape[0] > n_target:  # other source levels: their stds too
            _, rest = self._raw_mean_std(model, grid[n_target:], spec)
            std = torch.cat([std, rest])
        noise = self._measurement_noise(model, grid)  # (N, m) or (N, 1)
        term1 = _entropy((std**2 + noise).sqrt())  # (N, d_y)
        with torch.no_grad():
            post = model.posterior(at_target.double())
            # (n_samples, N, d_y) coherent joint draws; raw units for the goal.
            samples = post.rsample(torch.Size([self.n_posterior_samples]))
        term2 = torch.zeros_like(term1)
        n = self.n_posterior_samples
        conditioned = failed = 0
        unconstrained = 0  # samples where only the constraints emptied the goal
        for i in range(n):
            f_canon = samples[i].reshape(n_target, -1)
            f_raw = spec.to_raw(f_canon)  # BAX reads ALL outputs raw (goal consumes them)
            target = self._identify(spec, f_raw, at_target)
            if not target:
                if self._constrains(spec) and len(self._goal_for(spec).identify_subspace(f_raw, at_target)):
                    unconstrained += 1
                term2 += term1  # no execution info -> entropy unchanged
                continue
            conditioned += 1
            try:
                fant = _condition(
                    model, at_target[target].double(), f_canon[target], noise=_FANTASY_JITTER
                )
                _, fstd = self._raw_mean_std(fant, grid, spec)
                term2 += _entropy((fstd**2 + noise).sqrt())
            except Exception:
                logger.exception("InfoBAX fantasy conditioning failed for a sample.")
                failed += 1
                term2 += term1
        if not conditioned:
            # No sample gives the goal anything to select, so the information gain
            # is zero everywhere and the pick would be arbitrary. Explore instead.
            why = (
                ", once the output constraints are applied" if unconstrained else ""
            )
            self._warn(
                f"the goal selects nothing on any of {n} posterior samples{why}; "
                "proposing by uncertainty instead"
            )
            rel = (std[:n_target] / self._out_scale.to(std)).mean(dim=-1)
            rest = torch.full((grid.shape[0] - n_target,), float("-inf"), dtype=rel.dtype)
            return torch.cat([rel, rest])
        if conditioned and failed == conditioned:
            # Every fantasy failed, so the information gain is zero everywhere and
            # the pick would be arbitrary. Say so instead.
            raise RuntimeError(
                f"InfoBAX: conditioning {type(model).__name__} failed for every "
                "posterior sample, so there is no information gain to rank by"
            )
        term2 = term2 / n
        eig = (term1 - term2).mean(dim=-1)  # (N,) averaged over outputs
        if self._cost is not None and len(self._cost) == len(eig):
            eig = eig / self._cost  # information per target-run-equivalent of cost
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
