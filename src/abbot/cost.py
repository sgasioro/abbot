"""What an evaluation costs, as a function of the design.

Every record carries a ``cost``: the wall-clock seconds ``Campaign.step`` measured
around the run, or whatever the caller passed to ``ingest`` / ``record`` (core-hours,
money). A :class:`CostModel` predicts that cost for designs not yet run, so a policy
can trade value against expense (``SimpleBO(cost_model=...)``) and a supervisor can
stop at a budget (:func:`abbot.supervision.cost_cap`). Use :class:`CallableCost` for a cost
you already know, :class:`FittedCost` for a formula whose parameters should be
learned from the records (:class:`PowerLawCost` is one).

Costs are positive and in the units the records use. Models take designs in raw
(physical) units, the same ``(..., d)`` layout as ``Record.candidate``, and must be
differentiable in ``X`` so a continuous acquisition optimizer can use them.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, runtime_checkable

import torch

from abbot.history import History


@runtime_checkable
class CostModel(Protocol):
    """``cost(X) -> (...)``: the predicted cost of each design in ``X (..., d)``."""

    def __call__(self, X: torch.Tensor) -> torch.Tensor: ...

    def fit(self, history: History, spec) -> None:
        """Update from the costs recorded so far (a no-op for fixed models)."""
        ...


class FixedCost:
    """The same cost for every design (the default: unit cost)."""

    def __init__(self, cost: float = 1.0) -> None:
        if cost <= 0:
            raise ValueError(f"cost must be positive, got {cost}")
        self.cost = float(cost)

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return torch.full(X.shape[:-1], self.cost, dtype=X.dtype, device=X.device)

    def fit(self, history: History, spec) -> None:
        return None


class CallableCost:
    """A cost you already know as a formula: ``fn(X) -> (...)``, raw units in."""

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.fn = fn

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.fn(X), dtype=X.dtype).reshape(X.shape[:-1])

    def fit(self, history: History, spec) -> None:
        return None


class FittedCost:
    """A cost formula with unknown parameters, fitted to the recorded costs: the
    learned counterpart of :class:`CallableCost`.

    ``fn(x, p)`` returns the positive cost. ``x`` maps each spec variable name to a
    ``(...)`` tensor of raw values, and ``p`` maps each parameter name to a scalar
    tensor. ``params`` gives each parameter's starting value. ``fit`` minimizes the
    mean squared error of log cost over the records that carry a positive cost, so
    the fit is in relative terms, as costs spanning orders of magnitude need.
    Parameters named in ``fixed`` keep their values. Until there are at least as many
    costed records as free parameters (or ``min_records``), it keeps its starting
    values.

    See :class:`PowerLawCost` for a complete example.
    """

    def __init__(
        self,
        fn: Callable[[dict[str, torch.Tensor], dict[str, torch.Tensor]], torch.Tensor],
        params: Mapping[str, float],
        *,
        fixed: Sequence[str] = (),
        min_records: int | None = None,
    ) -> None:
        unknown = [name for name in fixed if name not in params]
        if unknown:
            raise ValueError(f"fixed parameter(s) {unknown} not in params {list(params)}")
        self.fn = fn
        self.params = {k: torch.tensor(float(v), dtype=torch.double) for k, v in params.items()}
        self.fixed = set(fixed)
        n_free = len(self.params) - len(self.fixed)
        self.min_records = max(1, n_free) if min_records is None else min_records
        self._names: list[str] | None = None

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return self._evaluate(X, self.params)

    def fit(self, history: History, spec) -> None:
        self._names = list(spec.variable_names)
        rows = [r for r in history.records if r.cost is not None and r.cost > 0]
        free = [k for k in self.params if k not in self.fixed]
        if not free or len(rows) < self.min_records:
            return
        X = torch.stack([r.candidate.double().reshape(-1) for r in rows])
        log_c = torch.tensor([r.cost for r in rows], dtype=torch.double).log()
        theta = {k: v.clone().requires_grad_(k in free) for k, v in self.params.items()}
        opt = torch.optim.LBFGS(
            [theta[k] for k in free], max_iter=200, tolerance_grad=1e-10,
            tolerance_change=1e-12, line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad()
            pred = self._evaluate(X, theta).clamp_min(1e-300).log()
            loss = (pred - log_c).pow(2).mean()
            loss.backward()
            return loss

        opt.step(closure)
        self.params = {k: v.detach() for k, v in theta.items()}

    def _evaluate(self, X: torch.Tensor, params: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self._names is None:
            raise RuntimeError(f"{type(self).__name__}.fit(history, spec) must run before use")
        x = {name: X[..., i] for i, name in enumerate(self._names)}
        p = {k: v.to(X) for k, v in params.items()}
        return torch.as_tensor(self.fn(x, p)).to(X).expand(X.shape[:-1])

    def __repr__(self) -> str:
        shown = ", ".join(f"{k}={float(v):.4g}" for k, v in self.params.items())
        return f"{type(self).__name__}({shown})"


class PowerLawCost(FittedCost):
    """``cost = a * prod_i x_i ** p_i`` over the named ``variables``: run time that
    scales steeply with a size-like knob (atom count, mesh cells, samples). Every
    named variable must be positive. Pass ``exponents`` to fix the ``p_i`` and fit
    only ``a``. Until it has costs it predicts ``initial`` everywhere.

    It is also the worked example of :class:`FittedCost`: the whole model is the
    formula below plus its parameters (``log_a`` and one ``p_<variable>`` each).
    """

    def __init__(
        self,
        variables: Sequence[str],
        *,
        exponents: Sequence[float] | None = None,
        initial: float = 1.0,
    ) -> None:
        if not variables:
            raise ValueError("PowerLawCost needs at least one variable")
        if exponents is not None and len(exponents) != len(variables):
            raise ValueError(f"{len(exponents)} exponent(s) for {len(variables)} variable(s)")
        self.variables = list(variables)
        start = exponents if exponents is not None else [0.0] * len(variables)

        def power_law(x, p):
            log_cost = p["log_a"]
            for v in self.variables:
                log_cost = log_cost + p[f"p_{v}"] * x[v].clamp_min(1e-12).log()
            return log_cost.exp()

        super().__init__(
            power_law,
            {"log_a": math.log(initial), **{f"p_{v}": e for v, e in zip(variables, start)}},
            fixed=[f"p_{v}" for v in variables] if exponents is not None else (),
        )

    def fit(self, history: History, spec) -> None:
        missing = [v for v in self.variables if v not in spec.variable_names]
        if missing:
            raise ValueError(
                f"PowerLawCost: {missing} not in the spec's variables {spec.variable_names}"
            )
        super().fit(history, spec)

    @property
    def exponents(self) -> torch.Tensor:
        return torch.stack([self.params[f"p_{v}"] for v in self.variables])

    @property
    def prefactor(self) -> float:
        return float(self.params["log_a"].exp())
