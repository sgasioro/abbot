"""``run_campaign`` — the ask/tell driver.

Merge basis: ``~/sbido/src/sbido/optimizer.py::measure_and_optimize``. Handles
resume, a seed queue, supervisor retry/halt, and a denoised end-of-run
recommendation. The loop is headless: it only ever calls ``experiment.run`` and
the policy's ask/tell surface, so a GUI, a simulator, or an agent are all just
``Experiment`` implementations.

M1 scope: a scalar single-output objective fit by a ModelConstructor and proposed
by a Policy. The composite-SBI reduction (posterior -> F) and cached ``aux``
payload are layered in by M2 (see ``abbot.reduce``), which supplies ``outputs``
and ``aux`` per record without changing this driver.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch

from abbot.experiment import Experiment, Supervisor, Verdict
from abbot.history import DataStore, History, Record
from abbot.model import ModelConstructor
from abbot.policy import Policy
from abbot.problem import ProblemSpec

logger = logging.getLogger(__name__)

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
) -> tuple[torch.Tensor, float, History]:
    """Run the propose -> evaluate -> retrain loop.

    ``evaluate`` maps a raw observation to the objective value(s) the surrogate is
    fit on. It may return either:
      - a tensor (0-d / (1,) / (m,))  — the plain lab-BO case; aux is None; or
      - a ``(outputs, aux)`` tuple    — the composite-SBI case, where ``aux`` is an
        opaque payload (e.g. cached theta posterior samples) persisted on the
        Record so a re-scorable reducer avoids re-sampling.
    This single seam is what lets the SAME loop drive both the lab-BO tool (M3)
    and the SBI app (M2) without change. Returns the recommended best design, its
    score, and the full history.
    """
    check_uq_pairing(policy, model_uq)

    history = History()
    best_x = initial_designs[0].clone()
    best_y = float("-inf")

    # Resume from disk.
    resumed = 0
    if store is not None:
        history = store.load()
        resumed = len(history)
        for r in history.records:
            if r.outputs is not None:
                y = float(r.outputs.reshape(-1)[0])
                if y > best_y:
                    best_y, best_x = y, r.candidate.clone()
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
        y, aux = _split_eval(evaluate(observation))
        record = Record(
            candidate=x, observation=observation, outputs=y, aux=aux, retry_of=retry_of
        )
        history.append(record)

        verdict = supervisor(history) if supervisor is not None else Verdict()
        record.fault = verdict.retry

        y_scalar = float(y[0])
        if not record.fault and y_scalar > best_y:
            best_y, best_x = y_scalar, x.clone()

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

    # Denoised recommendation: let the policy pick a better handover than the raw
    # observed argmax if it can (fit one final model on the clean history).
    if len(history.clean):
        try:
            model = _fit(model_constructor, history)
            idx = policy.recommend(history, spec, model)
        except Exception:
            logger.exception("recommend failed; using observed argmax.")
            idx = None
        if idx is not None:
            best_x = history.records[idx].candidate.clone()
            best_y = float(history.records[idx].outputs.reshape(-1)[0])

    return best_x, best_y, history


def _split_eval(result: object) -> tuple[torch.Tensor, object | None]:
    """Normalize an ``evaluate`` return into ``(outputs, aux)``.

    Accepts a bare tensor (aux=None) or an ``(outputs, aux)`` tuple.
    """
    if isinstance(result, tuple):
        outputs, aux = result
    else:
        outputs, aux = result, None
    return outputs.reshape(-1).double(), aux


def _fit(model_constructor: ModelConstructor, history: History):
    usable = history.clean
    X = usable.designs.double()
    Y = usable.outputs.double()
    return model_constructor.build(X, Y)


def _propose(policy, spec, model_constructor, history, step):
    """Fit a model on the clean history and ask the policy for the next design."""
    try:
        model = _fit(model_constructor, history)
        return policy.suggest(history, spec, model, n=1).squeeze(0)
    except Exception:
        logger.exception(
            "proposal failed at step %d; falling back to a uniform random draw.", step
        )
        bounds = spec.bounds_tensor()
        lo, hi = bounds[0], bounds[1]
        return lo + (hi - lo) * torch.rand(bounds.shape[-1], dtype=torch.double)
