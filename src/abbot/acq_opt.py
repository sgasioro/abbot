"""The CandidateOptimizer seam: candidate proposal decoupled from the surrogate.

Basis: nn-bax ``acq_opt.py`` (``SubsetOptimizer`` discrete-argmax vs
``ContinuousOptimizer``). Day-one scope is **continuous** (``optimize_acqf``) +
**enumerated-discrete** (``optimize_acqf_discrete``) — this covers all three
source repos as they run today. True combinatorial (non-enumerable) search is an
extension point, not solved speculatively.

Decoupling proposal from the surrogate is what lets a discrete BAXstics problem
and a continuous accelerator problem share one policy.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf, optimize_acqf_discrete

from abbot.problem import ProblemSpec


@runtime_checkable
class CandidateOptimizer(Protocol):
    """Maximize an acquisition function over the design space, return (n, d)."""

    def optimize(
        self, acq: AcquisitionFunction, spec: ProblemSpec, n: int
    ) -> torch.Tensor: ...


class Continuous:
    """Continuous proposal via botorch ``optimize_acqf``."""

    def __init__(self, num_restarts: int = 10, raw_samples: int = 256) -> None:
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples

    def optimize(
        self, acq: AcquisitionFunction, spec: ProblemSpec, n: int
    ) -> torch.Tensor:
        from abbot.problem import ContinuousDomain

        # Day-one scope is a homogeneous continuous space. A mixed continuous +
        # enumerated space needs optimize_acqf_mixed, which is NOT wired yet — fail
        # loudly rather than silently relaxing the enumerated axis to a float range.
        if any(not isinstance(d, ContinuousDomain) for d in spec.domain):
            kinds = {type(d).__name__ for d in spec.domain}
            raise NotImplementedError(
                "Continuous optimizer requires an all-continuous domain, but got "
                f"{sorted(kinds)}. Mixed continuous+enumerated proposal "
                "(optimize_acqf_mixed) is deferred; for an all-enumerated space use "
                "DiscreteArgmax instead."
            )
        bounds = spec.bounds_tensor()
        candidates, _ = optimize_acqf(
            acq,
            bounds=bounds,
            q=n,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            inequality_constraints=spec.inequality_constraints(),
        )
        return candidates.detach()


class DiscreteArgmax:
    """Enumerated proposal via botorch ``optimize_acqf_discrete``.

    ``choices`` is the (N, d) tensor of enumerable candidates. When omitted it is
    built as the Cartesian product of each EnumeratedDomain's values (small spaces
    only — large products should pass ``choices`` explicitly).
    """

    def __init__(self, choices: torch.Tensor | None = None) -> None:
        self.choices = choices

    def _enumerate(self, spec: ProblemSpec) -> torch.Tensor:
        from itertools import product

        from abbot.problem import EnumeratedDomain

        axes = []
        for d in spec.domain:
            if isinstance(d, EnumeratedDomain):
                axes.append(d.values)
            else:
                raise TypeError(
                    "DiscreteArgmax without explicit `choices` requires every "
                    f"variable to be EnumeratedDomain; {d.name!r} is not."
                )
        return torch.tensor(list(product(*axes)), dtype=torch.double)

    def optimize(
        self, acq: AcquisitionFunction, spec: ProblemSpec, n: int
    ) -> torch.Tensor:
        choices = self.choices if self.choices is not None else self._enumerate(spec)
        candidates, _ = optimize_acqf_discrete(acq, q=n, choices=choices.double())
        return candidates.detach()
