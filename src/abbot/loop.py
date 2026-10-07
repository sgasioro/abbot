"""``run_campaign`` — the ask/tell driver.

Merge basis: ``~/sbido/src/sbido/optimizer.py::measure_and_optimize``. Handles
resume, a seed queue, supervisor retry/halt, and a denoised end-of-run
recommendation. The loop is headless: it only ever calls ``experiment.run`` and
the policy's ask/tell surface, so a GUI, a simulator, or an agent are all just
``Experiment`` implementations.

The engine is natively multi-objective and maximize-canonical: ``evaluate``
returns RAW physical measurements (one column per objective, in
``spec.objectives`` order), and the loop applies per-objective signs
(``spec.objective_signs``) so the surrogate/acquisition always see a
"bigger is better" space. The composite-SBI reduction (posterior -> F) and cached
``aux`` payload are layered in by M2 (see ``abbot.reduce``), which supplies
``outputs`` and ``aux`` per record without changing this driver.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from botorch.utils.multi_objective.pareto import is_non_dominated

from abbot.experiment import Experiment, Supervisor, Verdict
from abbot.history import DataStore, History, Record
from abbot.model import ModelConstructor
from abbot.policy import Policy
from abbot.problem import ProblemSpec

logger = logging.getLogger(__name__)


@dataclass
class CampaignResult:
    """The outcome of a campaign.

    ``recommended`` holds the handover index/indices into ``history.records``: a
    single index for a scalar objective, the Pareto set for multi-objective. Use
    ``.best`` for the single-objective convenience (it refuses to collapse a true
    multi-point front to one answer).
    """

    history: History
    recommended: list[int] = field(default_factory=list)

    @property
    def designs(self) -> torch.Tensor:
        """The recommended design(s), stacked: (k, d)."""
        if not self.recommended:
            return torch.empty(0)
        return torch.stack([self.history.records[i].candidate for i in self.recommended])

    @property
    def best(self) -> Record:
        """The single recommended record. Errors if the front has 0 or >1 points."""
        if len(self.recommended) != 1:
            raise ValueError(
                f"best is only defined for a single recommended design, but "
                f"{len(self.recommended)} were recommended (a multi-objective front). "
                "Use .recommended / .designs instead."
            )
        return self.history.records[self.recommended[0]]

# A model's capability: what UQ quality its posterior actually provides.
ModelUQ = str  # "point" | "samples" | "calibrated_variance"

# Which model capabilities satisfy each policy requirement (ordered by strength).
_SATISFIES: dict[str, set[str]] = {
    "point": {"point", "samples", "calibrated_variance"},
    "samples": {"samples", "calibrated_variance"},
    "calibrated_variance": {"calibrated_variance"},
}


class PersistentFaultError(RuntimeError):
    """The apparatus needs a human. Every completed step is already on disk."""


def check_uq_pairing(policy: Policy, model_uq: ModelUQ) -> None:
    """Refuse a model<->policy pairing the surrogate can't support.

    e.g. deterministic-NN ("point") + UncertaintySampling ("calibrated_variance")
    is refused with a clear error; deterministic-NN + MeanBAX ("point") is fine.
    """
    required = policy.uq_requirement
    if model_uq not in _SATISFIES[required]:
        raise ValueError(
            f"{type(policy).__name__} requires uq_requirement={required!r}, but the "
            f"model only provides {model_uq!r}. Pair it with a model that provides "
            f"at least {required!r} UQ, or choose a lower-UQ policy."
        )


def run_campaign(
    *,
    experiment: Experiment,
    spec: ProblemSpec,
    policy: Policy,
    model_constructor: ModelConstructor,
    evaluate: Callable[[object], object],
    initial_designs: torch.Tensor,
    n_shots: int = 40,
    store: DataStore | None = None,
    supervisor: Supervisor | None = None,
    max_retries: int = 1,
    on_persistent_fault: str = "halt",
    model_uq: ModelUQ = "calibrated_variance",
) -> CampaignResult:
    """Run the propose -> evaluate -> retrain loop.

    ``evaluate`` maps a raw observation to the RAW physical objective value(s) the
    surrogate is fit on (one column per objective, in ``spec.objectives`` order).
    It may return either:
      - a tensor ((m,) / (1,) / 0-d)  — aux is None; or
      - a ``(outputs, aux)`` tuple    — the composite-SBI case, where ``aux`` is an
        opaque payload (e.g. cached theta posterior samples) persisted on the
        Record so a re-scorable reducer avoids re-sampling.
    This single seam is what lets the SAME loop drive the lab-BO tool (M3) and the
    SBI app (M2), scalar or multi-objective, without change. The loop never
    pre-negates: objective direction is applied internally via
    ``spec.objective_signs``. Returns a ``CampaignResult`` whose ``recommended``
    holds the handover index/indices (single point, or Pareto set for MO).
    """
    check_uq_pairing(policy, model_uq)

    history = History()

    # Resume from disk.
    resumed = 0
    if store is not None:
        history = store.load()
        resumed = len(history)
        if resumed:
            logger.info("Resumed %d steps", resumed)

    seed_queue = list(initial_designs) if resumed == 0 else []
    n_total = len(seed_queue) + n_shots
    pending: tuple[torch.Tensor, int, int] | None = None  # (x, retry_of, attempt)

    for i in range(n_total):
        step = resumed + i
        if pending is not None:
            x, retry_of, attempt = pending
            pending = None
        else:
            retry_of, attempt = None, 0
            if seed_queue:
                x = seed_queue.pop(0)
            else:
                x = _propose(policy, spec, model_constructor, history, step)

        observation = experiment.run(x)
        y, aux = _split_eval(evaluate(observation), spec)
        record = Record(
            candidate=x, observation=observation, outputs=y, aux=aux, retry_of=retry_of
        )
        history.append(record)

        verdict = supervisor(history) if supervisor is not None else Verdict()
        record.fault = verdict.retry

        if store is not None:
            store.save_step(step, record)

        if verdict.halt is not None:
            if on_persistent_fault == "halt":
                raise PersistentFaultError(
                    f"{verdict.halt}\n{step + 1} shots completed. Fix the hardware and "
                    "rerun to resume."
                )
            logger.warning("Persistent fault at step %d (%s); continuing.", step, verdict.halt)

        if verdict.retry and attempt < max_retries:
            pending = (x, step if retry_of is None else retry_of, attempt + 1)

    # End-of-run recommendation: fit one final canonical model and let the policy
    # pick the denoised handover (single index, or Pareto set for MO). Fall back to
    # the observed non-dominated set if the policy declines or fitting fails.
    recommended: list[int] = []
    if len(history.clean):
        try:
            model = _fit(model_constructor, history, spec)
            recommended = policy.recommend(history, spec, model)
        except Exception:
            logger.exception("recommend failed; using the observed non-dominated set.")
            recommended = []
        if not recommended:
            recommended = _observed_front(history, spec)

    return CampaignResult(history=history, recommended=recommended)


def _split_eval(result: object, spec: ProblemSpec) -> tuple[torch.Tensor, object | None]:
    """Normalize an ``evaluate`` return into ``(raw_outputs, aux)``.

    Accepts a bare tensor (aux=None) or an ``(outputs, aux)`` tuple. Validates the
    number of objective columns against the spec so a wrong-width return fails here
    with a clear message instead of crashing downstream.
    """
    if isinstance(result, tuple):
        outputs, aux = result
    else:
        outputs, aux = result, None
    outputs = outputs.reshape(-1).double()
    if outputs.shape[0] != spec.n_objectives:
        raise ValueError(
            f"evaluate() returned {outputs.shape[0]} value(s) but the spec declares "
            f"{spec.n_objectives} objective(s); return one column per objective in "
            "spec.objectives order."
        )
    return outputs, aux


def _canonical_Y(history: History, spec: ProblemSpec) -> torch.Tensor:
    """Clean raw outputs mapped to maximize-canonical space via objective signs."""
    return history.clean.outputs.double() * spec.objective_signs()


def _observed_front(history: History, spec: ProblemSpec) -> list[int]:
    """Fallback handover: non-dominated set of the OBSERVED (noisy) canonical
    outputs. Scalar case collapses to the single argmax.
    """
    keep = [i for i, r in enumerate(history.records) if not r.fault]
    if not keep:
        return []
    Y = torch.stack([history.records[i].outputs.reshape(-1) for i in keep]).double()
    Y = Y * spec.objective_signs()
    if Y.shape[-1] == 1:
        return [keep[int(Y.reshape(-1).argmax())]]
    nd = is_non_dominated(Y)
    return [keep[i] for i in range(len(keep)) if bool(nd[i])]


def _fit(model_constructor: ModelConstructor, history: History, spec: ProblemSpec):
    X = history.clean.designs.double()
    Y = _canonical_Y(history, spec)
    return model_constructor.build(X, Y)


def _propose(policy, spec, model_constructor, history, step):
    """Fit a model on the clean history and ask the policy for the next design."""
    try:
        model = _fit(model_constructor, history, spec)
        return policy.suggest(history, spec, model, n=1).squeeze(0)
    except Exception:
        logger.exception(
            "proposal failed at step %d; falling back to a uniform random draw.", step
        )
        bounds = spec.bounds_tensor()
        lo, hi = bounds[0], bounds[1]
        return lo + (hi - lo) * torch.rand(bounds.shape[-1], dtype=torch.double)
