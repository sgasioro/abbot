"""What an evaluation costs, as a function of the design.

Every record carries a ``cost``: the wall-clock seconds ``Campaign.step`` measured
around the run, or whatever the caller passed to ``ingest`` / ``record`` (core-hours,
money). A :class:`CostModel` predicts that cost for designs not yet run, so a policy
can trade value against expense (``SimpleBO(cost_model=...)``) and a supervisor can
stop at a budget (:func:`abbot.supervision.cost_cap`).

Costs are positive and in the units the records use. Models take designs in raw
(physical) units, the same ``(..., d)`` layout as ``Record.candidate``, and must be
differentiable in ``X`` so a continuous acquisition optimizer can use them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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


class PowerLawCost:
    """``cost = a * prod_i x_i ** p_i`` over the named ``variables``, fitted to the
    recorded costs by least squares in log space.

    Suits costs that scale steeply with a size-like knob (run time against atom
    count, mesh cells, samples). Every named variable must be positive. Pass
    ``exponents`` to fix the ``p_i`` and fit only ``a``. Until two records carry a
    cost (one, with fixed exponents), it predicts ``initial`` everywhere.
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
            raise ValueError(
                f"{len(exponents)} exponent(s) for {len(variables)} variable(s)"
            )
        self.variables = list(variables)
        self.fixed_exponents = exponents is not None
        self.exponents = torch.as_tensor(
            exponents if exponents is not None else [0.0] * len(variables),
            dtype=torch.double,
        )
        self.log_a = torch.tensor(float(initial), dtype=torch.double).log()
        self._columns: list[int] | None = None

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        if self._columns is None:
            raise RuntimeError("PowerLawCost.fit(history, spec) must run before use")
        logx = X[..., self._columns].clamp_min(1e-12).log()
        return (self.log_a.to(X) + logx @ self.exponents.to(X)).exp()

    def fit(self, history: History, spec) -> None:
        names = spec.variable_names
        missing = [v for v in self.variables if v not in names]
        if missing:
            raise ValueError(f"PowerLawCost: {missing} not in the spec's variables {names}")
        self._columns = [names.index(v) for v in self.variables]
        rows = [r for r in history.records if r.cost is not None and r.cost > 0]
        k = len(self.variables)
        if len(rows) < (1 if self.fixed_exponents else 2):
            return
        X = torch.stack([r.candidate.double().reshape(-1) for r in rows])
        logx = X[:, self._columns].clamp_min(1e-12).log()
        logc = torch.tensor([r.cost for r in rows], dtype=torch.double).log()
        if self.fixed_exponents:
            self.log_a = (logc - logx @ self.exponents).mean()
            return
        A = torch.cat([torch.ones(len(rows), 1, dtype=torch.double), logx], dim=-1)
        # min-norm solution: with fewer distinct points than k + 1 the extra
        # exponents stay near zero instead of blowing up.
        sol = torch.linalg.lstsq(A, logc.unsqueeze(-1), driver="gelsd").solution.reshape(-1)
        self.log_a, self.exponents = sol[0], sol[1 : k + 1]

    def __repr__(self) -> str:
        terms = " * ".join(
            f"{v}^{float(p):.3g}" for v, p in zip(self.variables, self.exponents)
        )
        return f"PowerLawCost({float(self.log_a.exp()):.4g} * {terms})"
