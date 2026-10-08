"""ObjectiveReducer — the composite-SBI reduction (M2).

Reduce posterior samples over theta to a scalar/vector objective F. This is the
genuinely-ours composite-SBI layer. The same reducer logic mounts at EITHER of
two points:
  - pre-GP:  reduce to a scalar F(xi) before the surrogate (status quo, simple
             cases) — supply it as the loop's ``evaluate`` callback. **Implemented
             here (M2).**
  - in-acqf: keep a multi-output theta surrogate and reduce inside a botorch
             ``MCAcquisitionObjective`` / posterior transform (information-rich
             SBI; the Astudillo-Frazier composite-BO pattern). **Follow-on.**

Basis: ``~/sbido/src/sbido/objectives.py::selection_score`` — ported verbatim in
scoring so abbot's numbers match sbido's (the oracle and the optimizer objective
are only comparable if they are literally the same rule). abbot reads target
bands from ``ProblemSpec.boxes`` (``Box.lower``/``Box.upper`` lists) rather than
sbido's ``(a, b)`` tensor tuples; ``boxes_to_tensors`` bridges the two.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import torch

from abbot.problem import Box, ProblemSpec


@runtime_checkable
class ObjectiveReducer(Protocol):
    def __call__(
        self, posterior_samples: torch.Tensor, spec: ProblemSpec
    ) -> torch.Tensor:
        """Map (n_samples, d_theta) posterior samples to scalar/vector F."""
        ...


# --- per-sample scoring (ported from sbido/objectives.py) ---------------------


def _per_sample_score(
    theta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: torch.Tensor | None,
    beta: float | None,
) -> torch.Tensor:
    """Per-sample score against one box. Distance mode (beta=None) or soft-indicator.

    Verbatim from sbido ``objectives._per_sample_score`` so scores are identical.
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
    """Mean of the max per-sample score across all boxes (sbido ``or_score``)."""
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
    """Turn abbot ``Box`` list into sbido-style ``(a, b)`` tensor tuples."""
    dt = dtype or torch.double
    return [
        (
            torch.tensor(box.lower, dtype=dt, device=device),
            torch.tensor(box.upper, dtype=dt, device=device),
        )
        for box in boxes
    ]


# --- the reducer --------------------------------------------------------------


class BoxMembershipReducer:
    """Composite-SBI reducer: posterior samples over theta -> scalar F.

    Mirrors ``sbido.objectives.selection_score``. ``objective``:
      * ``"indicator"`` / ``"distance"`` — single-box scoring against ``boxes[0]``.
      * ``"or_indicator"`` / ``"or_distance"`` — OR scoring across all boxes.
    Note the different ranges: indicator is in [0, 1] (MAXIMIZE), distance is <= 0
    (MAXIMIZE toward 0). ``beta`` is the soft-indicator sharpness; a caller running
    the sbido annealing schedule overrides it per step via ``beta=`` on the call.
    """

    def __init__(
        self, objective: SelectionObjective = "indicator", beta: float = 1e3
    ) -> None:
        self.objective = objective
        self.beta = beta

    def __call__(
        self,
        posterior_samples: torch.Tensor,
        spec: ProblemSpec,
        beta: float | None = None,
    ) -> torch.Tensor:
        if not spec.boxes:
            raise ValueError(
                "BoxMembershipReducer requires ProblemSpec.boxes (target bands); none set."
            )
        boxes = boxes_to_tensors(spec.boxes, dtype=posterior_samples.dtype)
        beta = self.beta if beta is None else beta
        theta = posterior_samples
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
