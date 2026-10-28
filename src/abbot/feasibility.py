"""Output constraints: which designs satisfy the spec's output bounds.

An output constraint (``OutputSpec(min_val=..., max_val=...)`` or an entry of
``ProblemSpec.output_constraints``) is a row ``lower <= a @ y <= upper`` over the RAW
outputs ``y`` (``spec.output_constraint_rows``). Observed outputs are feasible or
not; the model gives a probability. The model is maximize-canonical, so its outputs
are mapped back to raw units (``spec.output_signs``) before a row is applied.

The probability treats the outputs at a design, and the rows, as independent. That
is exact for the default per-output models and a single bound per output, and an
approximation for a linear constraint over correlated outputs.
"""

from __future__ import annotations

import torch

from abbot.history import History
from abbot.problem import ProblemSpec

# A design counts as feasible when the model gives it at least this probability.
FEASIBLE_P = 0.5


def feasible(spec: ProblemSpec, Y: torch.Tensor) -> torch.Tensor:
    """Which rows of RAW outputs ``Y (..., n_outputs)`` satisfy every output
    constraint: a bool tensor ``(...)``. All True without constraints."""
    A, lower, upper = spec.output_constraint_rows(dtype=Y.dtype, device=Y.device)
    value = Y @ A.T  # (..., k)
    return ((value >= lower) & (value <= upper)).all(dim=-1)


def violation(spec: ProblemSpec, Y: torch.Tensor, scale: torch.Tensor | None = None):
    """How far RAW outputs ``Y (..., n_outputs)`` are from satisfying every output
    constraint: the sum over rows of the distance outside ``[lower, upper]``, each
    divided by ``scale (k,)`` (see :func:`row_scale`). 0 when feasible."""
    A, lower, upper = spec.output_constraint_rows(dtype=Y.dtype, device=Y.device)
    value = Y @ A.T
    gap = (lower - value).clamp_min(0) + (value - upper).clamp_min(0)
    if scale is not None:
        gap = gap / scale
    return gap.sum(dim=-1)


def row_scale(history: History, spec: ProblemSpec) -> torch.Tensor:
    """The spread of each constraint row's value over the clean records ``(k,)``,
    1 where it can't be told. Puts rows in different units on one scale."""
    A, _, _ = spec.output_constraint_rows()
    Y = history.clean.outputs.double() if len(history.clean) else None
    if Y is None or Y.shape[0] < 2:
        return torch.ones(A.shape[0], dtype=torch.double)
    sd = (Y @ A.T).std(dim=0)
    return torch.where(sd > 1e-12, sd, torch.ones_like(sd))


def any_feasible(history: History, spec: ProblemSpec) -> bool:
    """Whether some clean record run at the target satisfies every output
    constraint (outputs measured at another source aren't comparable)."""
    records = [
        r for r in history.clean.records
        if torch.equal(r.candidate.double().reshape(-1), spec.at_target(r.candidate.double().reshape(-1)))
    ]
    if not records:
        return False
    Y = torch.stack([r.outputs.double().reshape(-1) for r in records])
    return bool(feasible(spec, Y).any())


def mc_constraints(spec: ProblemSpec, scale: torch.Tensor | None = None) -> list:
    """The output constraints as BoTorch MC constraints: callables on the model's
    (canonical) samples ``Z (..., n_outputs)`` that are <= 0 where satisfied.

    Each bound is divided by its row's ``scale``, so the smoothing BoTorch applies
    at the boundary is in comparable units across constraints.
    """
    A, lower, upper = spec.output_constraint_rows()
    if scale is None:
        scale = torch.ones(A.shape[0], dtype=torch.double)
    a_canonical = A * spec.output_signs()  # y_raw = Z * signs
    out = []
    for a, lo, hi, s in zip(a_canonical, lower, upper, scale):
        if torch.isfinite(lo):
            out.append(lambda Z, a=a, lo=lo, s=s: (lo - Z @ a.to(Z)) / s)
        if torch.isfinite(hi):
            out.append(lambda Z, a=a, hi=hi, s=s: (Z @ a.to(Z) - hi) / s)
    return out


def _row_moments(spec: ProblemSpec, mean: torch.Tensor, variance: torch.Tensor):
    """Mean and std of every constraint row from a canonical posterior's mean and
    variance ``(..., n_outputs)``: two ``(..., k)`` tensors."""
    A, _, _ = spec.output_constraint_rows(dtype=mean.dtype, device=mean.device)
    raw_mean = mean * spec.output_signs(dtype=mean.dtype, device=mean.device)
    row_mean = raw_mean @ A.T
    row_std = (variance.clamp_min(0) @ (A**2).T).clamp_min(1e-18).sqrt()
    return row_mean, row_std


def log_prob_feasible_from(spec: ProblemSpec, mean: torch.Tensor, variance: torch.Tensor):
    """Log probability that every output constraint holds, from a canonical
    posterior's mean and variance ``(..., n_outputs)``: ``(...)``."""
    from botorch.utils.probability.utils import log_ndtr, log_prob_normal_in

    _, lower, upper = spec.output_constraint_rows(dtype=mean.dtype, device=mean.device)
    row_mean, row_std = _row_moments(spec, mean, variance)
    has_lo, has_hi = torch.isfinite(lower), torch.isfinite(upper)
    # Infinite bounds give NaN gradients, so one-sided rows use the normal CDF directly.
    lo = torch.where(has_lo, lower, torch.zeros_like(lower))
    hi = torch.where(has_hi, upper, lo + 1)
    a = (lo - row_mean) / row_std
    b = (hi - row_mean) / row_std
    both = log_prob_normal_in(a, b)
    log_p = torch.where(has_lo, log_ndtr(-a), torch.zeros_like(a))
    log_p = torch.where(has_hi, log_ndtr(b), log_p)
    log_p = torch.where(has_lo & has_hi, both, log_p)
    return log_p.sum(dim=-1)


def prob_feasible(spec: ProblemSpec, model, X: torch.Tensor) -> torch.Tensor:
    """The model's probability that each design in ``X (n, d)`` satisfies every
    output constraint: ``(n,)``. All ones without constraints."""
    if not spec.has_output_constraints:
        return torch.ones(X.shape[0], dtype=torch.double)
    with torch.no_grad():
        post = model.posterior(X.double())
        mean = post.mean.reshape(X.shape[0], -1)
        variance = post.variance.reshape(X.shape[0], -1)
    return log_prob_feasible_from(spec, mean, variance).exp()


class LogProbFeasible(torch.nn.Module):
    """Acquisition: log probability that a design satisfies every output
    constraint. Expects ``(b, 1, d)`` and returns ``(b,)``.

    ``explore=True`` adds the log of the constraint rows' mean posterior std (each
    over its ``scale``): with no feasible design seen yet, this looks for feasibility
    where the model is unsure rather than piling onto one guess, and a batch built
    by conditioning on its earlier picks spreads out.
    """

    def __init__(self, model, spec: ProblemSpec, *, explore: bool = False,
                 scale: torch.Tensor | None = None) -> None:
        super().__init__()
        self.model = model
        self.spec = spec
        self.explore = explore
        self.scale = scale

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        post = self.model.posterior(X)
        mean = post.mean.squeeze(-2)  # (b, m)
        variance = post.variance.squeeze(-2)
        value = log_prob_feasible_from(self.spec, mean, variance)
        if self.explore:
            _, row_std = _row_moments(self.spec, mean, variance)
            scale = torch.ones(row_std.shape[-1]) if self.scale is None else self.scale
            value = value + (row_std / scale.to(row_std)).mean(dim=-1).clamp_min(1e-12).log()
        return value
