"""Composite objectives: score a rich per-shot payload, and re-score it later.

Many lab measurements are richer than the number being optimized: a spectrum, an
image, replicate measurements, posterior samples from an inference step. A
composite objective splits evaluation in two:

  * an *analysis* maps each observation to a payload (``analysis(observation)``),
  * a *reducer* maps the payload to the outputs the surrogate trains on
    (``reducer(payload, spec, **settings)``).

:func:`composite_evaluate` builds the loop's ``evaluate`` callback from the two
and caches the payload on each record's ``aux``. :class:`Rescore` is the matching
``Campaign(targets=...)`` hook: it re-reduces the cached payloads every step under
a schedule, so the objective can change during the run (a score that sharpens, a
tolerance that tightens) without re-measuring. :class:`BoxMembershipReducer`
scores samples by membership of the spec's target boxes.

Simulation-based inference is one instance — the analysis draws posterior samples
(see :mod:`abbot.interop.sbi`) — but nothing here depends on it.

The reducer mounts pre-GP (reduce to F, then fit the surrogate on F). Reducing
inside the acquisition from a multi-output surrogate (the Astudillo-Frazier
composite-BO pattern) is a possible follow-on.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Protocol, runtime_checkable

import torch

from abbot.history import History
from abbot.problem import Box, ProblemSpec


@runtime_checkable
class ObjectiveReducer(Protocol):
    def __call__(self, payload: Any, spec: ProblemSpec) -> torch.Tensor:
        """Map one shot's payload to its outputs: a scalar, or one value per
        ``spec`` output. May accept extra keyword settings (e.g. ``beta=``) that a
        :class:`Rescore` schedule supplies."""
        ...


def composite_evaluate(
    analysis: Callable[[Any], Any],
    reducer: ObjectiveReducer,
    spec: ProblemSpec,
    *,
    cache: bool = True,
) -> Callable[[object], object]:
    """Build an ``evaluate(observation) -> (outputs, payload)`` callback.

    ``analysis`` turns the raw observation into the payload; ``reducer`` turns the
    payload into the outputs. With ``cache`` the payload is returned alongside, so
    the campaign stores it on the record's ``aux`` (which :class:`Rescore` needs);
    without it only the outputs are returned. Pass the result as
    ``run_campaign(evaluate=...)`` or ``Campaign.step(evaluate=...)``.
    """

    def evaluate(observation: object) -> object:
        payload = analysis(observation)
        outputs = torch.as_tensor(reducer(payload, spec)).reshape(-1)
        return (outputs, payload) if cache else outputs

    return evaluate


class Rescore:
    """A ``Campaign(targets=...)`` hook: re-reduce every clean record's cached payload.

    Each time the campaign needs its surrogate, the clean (un-faulted) records'
    cached payloads (``aux``, see :func:`composite_evaluate`) are reduced afresh
    with ``reducer(payload, spec, **schedule(t))``. The surrogate, the policy and
    the hand-over all see these values; the stored outputs are left as measured.
    So a schedule — a soft indicator that sharpens, a target band that narrows, a
    switch from a distance to a hit/miss score — changes the objective without
    re-measuring, and every past shot is judged by the current setting.

    ``t`` (:meth:`clock`) counts records after the first ``n_initial`` (the
    seeds), faulted ones included since they spend budget: 0 on the first
    policy step, and never negative. The reducer may return one value per record
    or one per spec output.
    """

    def __init__(
        self,
        reducer: ObjectiveReducer,
        schedule: Callable[[int], dict] | None = None,
        *,
        n_initial: int = 0,
    ) -> None:
        self.reducer = reducer
        self.schedule = schedule
        self.n_initial = n_initial

    def clock(self, history: History) -> int:
        """The schedule step ``t`` for this history (0 until the seeds are done)."""
        return max(len(history) - self.n_initial, 0)

    def __call__(self, history: History, spec: ProblemSpec) -> torch.Tensor:
        clean = history.clean.records
        if any(r.aux is None for r in clean):
            raise ValueError(
                "Rescore needs a cached payload (aux) on every clean record; "
                "evaluate with composite_evaluate(..., cache=True) or return "
                "(outputs, payload) from evaluate."
            )
        kwargs = self.schedule(self.clock(history)) if self.schedule is not None else {}
        rows = [torch.as_tensor(self.reducer(r.aux, spec, **kwargs)).reshape(-1) for r in clean]
        if not rows:
            return torch.empty(0, spec.n_outputs, dtype=torch.double)
        return torch.stack(rows).double()


# --- per-sample scoring ------------------------------------------------------


def _per_sample_score(
    theta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: torch.Tensor | None,
    beta: float | None,
) -> torch.Tensor:
    """Per-sample score against one box. Distance mode (beta=None) or soft-indicator.

    Identical to sbido's ``objectives._per_sample_score``.
    """
    if beta is None:
        below = (a - theta).clamp(min=0)
        above = (theta - b).clamp(min=0)
        return -torch.sqrt((((below + above) / scale) ** 2).sum(-1) + 1e-12)
    return (
        torch.sigmoid(beta * (theta - a)) * torch.sigmoid(beta * (b - theta))
    ).prod(-1)


def _or_score(
    theta: torch.Tensor,
    boxes: list[tuple[torch.Tensor, torch.Tensor]],
    beta: float | None,
) -> torch.Tensor:
    """Mean of the max per-sample score across all boxes (an OR over boxes)."""
    per_box = []
    for a, b in boxes:
        scale = (b - a).clamp(min=1e-6)
        per_box.append(
            _per_sample_score(theta, a, b, scale if beta is None else None, beta)
        )
    return torch.stack(per_box).max(dim=0).values.mean()


SelectionObjective = Literal["indicator", "distance", "or_indicator", "or_distance"]


def boxes_to_tensors(
    boxes: list[Box], dtype=None, device=None
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Turn a ``Box`` list into ``(a, b)`` lower/upper tensor pairs."""
    dt = dtype or torch.double
    return [
        (
            torch.tensor(box.lower, dtype=dt, device=device),
            torch.tensor(box.upper, dtype=dt, device=device),
        )
        for box in boxes
    ]


# --- the box-membership reducer ---------------------------------------------


class BoxMembershipReducer:
    """Score a set of samples by membership of the spec's target box(es).

    The samples are any ``(n_samples, d)`` draws — posterior samples over theta, a
    particle-size distribution, replicate measurements — in the coordinates of
    ``spec.boxes``. The scoring rule is sbido's ``selection_score``, kept identical
    (and tested against it) so an oracle and the optimizer objective can be the
    same rule.

    ``objective``:
      * ``"indicator"`` / ``"distance"`` — single-box scoring against ``boxes[0]``.
      * ``"or_indicator"`` / ``"or_distance"`` — OR scoring across all boxes.
    Note the different ranges: indicator is in [0, 1] (MAXIMIZE), distance is <= 0
    (MAXIMIZE toward 0). ``beta`` is the soft-indicator sharpness; a caller running
    an annealing schedule (see :class:`Rescore`) overrides it per step
    via ``beta=`` on the call.
    """

    def __init__(
        self, objective: SelectionObjective = "indicator", beta: float = 1e3
    ) -> None:
        self.objective = objective
        self.beta = beta

    def __call__(
        self,
        samples: torch.Tensor,
        spec: ProblemSpec,
        beta: float | None = None,
    ) -> torch.Tensor:
        if not spec.boxes:
            raise ValueError(
                "BoxMembershipReducer requires ProblemSpec.boxes (target bands); none set."
            )
        boxes = boxes_to_tensors(spec.boxes, dtype=samples.dtype)
        beta = self.beta if beta is None else beta
        theta = samples
        if self.objective == "indicator":
            a, b = boxes[0]
            return _per_sample_score(theta, a, b, None, beta).mean()
        if self.objective == "distance":
            a, b = boxes[0]
            scale = (b - a).clamp(min=1e-6)
            return _per_sample_score(theta, a, b, scale, None).mean()
        if self.objective == "or_indicator":
            return _or_score(theta, boxes, beta=beta)
        if self.objective == "or_distance":
            return _or_score(theta, boxes, beta=None)
        raise ValueError(f"unknown selection objective {self.objective!r}")
