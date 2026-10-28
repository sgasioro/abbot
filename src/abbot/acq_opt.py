"""The CandidateOptimizer seam: candidate proposal decoupled from the surrogate.

Basis: nn-bax ``acq_opt.py`` (``SubsetOptimizer`` discrete-argmax vs
``ContinuousOptimizer``). ``Continuous`` covers any space with a continuous or
integer axis (``optimize_acqf``; integer axes relaxed then rounded; enumerated axes
alongside them enumerated by ``optimize_acqf_mixed``). ``DiscreteArgmax`` covers
all-enumerated spaces (``optimize_acqf_discrete``). True combinatorial
(non-enumerable) search is an extension point, not solved speculatively.

``fixed_features={position: value}`` pins inputs, which is how a policy that doesn't
choose sources proposes at their target (``spec.target_features()``).

Decoupling proposal from the surrogate is what lets a discrete BAXstics problem
and a continuous accelerator problem share one policy.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf, optimize_acqf_discrete, optimize_acqf_mixed

from abbot.problem import ContinuousDomain, EnumeratedDomain, IntegerDomain, ProblemSpec


@runtime_checkable
class CandidateOptimizer(Protocol):
    """Maximize an acquisition function over the inputs, return (n, d).

    ``fixed_features`` pins inputs ``{position: value}``; optimizers that predate it
    may not accept it, and policies only pass it when the spec has sources.
    """

    def optimize(
        self,
        acq: AcquisitionFunction,
        spec: ProblemSpec,
        n: int,
        *,
        fixed_features: dict[int, float] | None = None,
    ) -> torch.Tensor: ...


# Above this many combinations of enumerated values, a mixed space is too big to
# enumerate per proposal; pass the combinations you care about to DiscreteArgmax.
MAX_MIXED_COMBINATIONS = 256


class Continuous:
    """Proposal via botorch ``optimize_acqf`` for any space with a continuous or
    integer input. Integer inputs are optimized as continuous and rounded;
    enumerated inputs alongside them are enumerated (``optimize_acqf_mixed``, one
    sub-problem per combination of their values, up to
    ``MAX_MIXED_COMBINATIONS``). With ``log=True`` inputs (and no enumerated ones
    to enumerate), the restarts start from the best of ``raw_samples`` log-uniform
    designs, so the search covers every decade rather than mostly the top one."""

    def __init__(self, num_restarts: int = 10, raw_samples: int = 256) -> None:
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples

    def optimize(
        self,
        acq: AcquisitionFunction,
        spec: ProblemSpec,
        n: int,
        *,
        fixed_features: dict[int, float] | None = None,
    ) -> torch.Tensor:
        from itertools import product

        inputs = spec.inputs
        unsupported = [
            d.name for d in inputs
            if not isinstance(d, (ContinuousDomain, IntegerDomain, EnumeratedDomain))
        ]
        if unsupported:
            raise NotImplementedError(
                f"Continuous optimizer can't propose structured inputs {unsupported}."
            )
        fixed = dict(fixed_features or {})
        integer = [j for j, d in enumerate(inputs) if isinstance(d, IntegerDomain)]

        def round_integers(X: torch.Tensor) -> torch.Tensor:
            if integer:
                X = X.clone()
                X[..., integer] = X[..., integer].round()
            return X

        enum = [
            j for j, d in enumerate(inputs)
            if isinstance(d, EnumeratedDomain) and j not in fixed
        ]
        kwargs = {
            "acq_function": acq,
            "bounds": spec.bounds_tensor(),
            "q": n,
            "num_restarts": self.num_restarts,
            "raw_samples": self.raw_samples,
            "inequality_constraints": spec.inequality_constraints(),
            "post_processing_func": round_integers,
        }
        if not enum:
            if set(spec.log_indices()) - set(fixed):
                kwargs["batch_initial_conditions"] = self._log_uniform_starts(acq, spec, n, fixed)
                kwargs.pop("raw_samples")
            candidates, _ = optimize_acqf(fixed_features=fixed or None, **kwargs)
            return candidates.detach()
        combos = list(product(*(inputs[j].values for j in enum)))
        if len(combos) > MAX_MIXED_COMBINATIONS:
            raise ValueError(
                f"{len(combos)} combinations of enumerated values "
                f"({[inputs[j].name for j in enum]}) is too many to enumerate per "
                f"proposal (limit {MAX_MIXED_COMBINATIONS}); make some of them "
                "continuous or integer, or use DiscreteArgmax(choices=...)."
            )
        fixed_list = [{**fixed, **dict(zip(enum, map(float, c)))} for c in combos]
        candidates, _ = optimize_acqf_mixed(fixed_features_list=fixed_list, **kwargs)
        return candidates.detach()


    def _log_uniform_starts(self, acq, spec: ProblemSpec, n: int, fixed: dict[int, float]):
        """``(num_restarts, n, d)`` starting batches: the best of ``raw_samples``
        batches drawn by ``spec.sample_designs`` (log-uniform on log axes)."""
        names = spec.variable_names
        X = spec.sample_designs(
            self.raw_samples * n, fixed={names[j]: v for j, v in fixed.items()}
        ).reshape(self.raw_samples, n, -1)
        with torch.no_grad():
            value = acq(X).reshape(-1)
        best = value.nan_to_num(float("-inf")).topk(min(self.num_restarts, len(value))).indices
        return X[best]


class DiscreteArgmax:
    """Enumerated proposal via botorch ``optimize_acqf_discrete``.

    ``choices`` is the (N, d) tensor of enumerable candidates (every input). When
    omitted it is built as the Cartesian product of each enumerated or integer
    input's values (small spaces only — large products should pass ``choices``
    explicitly). ``fixed_features`` keeps only the choices that match.
    """

    def __init__(self, choices: torch.Tensor | None = None) -> None:
        self.choices = choices

    def _enumerate(self, spec: ProblemSpec) -> torch.Tensor:
        from itertools import product

        axes = []
        for d in spec.inputs:
            if isinstance(d, EnumeratedDomain):
                axes.append(d.values)
            elif isinstance(d, IntegerDomain):
                axes.append(list(range(d.lower, d.upper + 1)))
            else:
                raise TypeError(
                    "DiscreteArgmax without explicit `choices` requires every "
                    f"input to be enumerated or integer; {d.name!r} is not."
                )
        return torch.tensor(list(product(*axes)), dtype=torch.double)

    def optimize(
        self,
        acq: AcquisitionFunction,
        spec: ProblemSpec,
        n: int,
        *,
        fixed_features: dict[int, float] | None = None,
    ) -> torch.Tensor:
        choices = self.choices if self.choices is not None else self._enumerate(spec)
        choices = choices.double()
        for j, value in (fixed_features or {}).items():
            choices = choices[torch.isclose(choices[:, j], torch.tensor(value).double())]
        if not len(choices):
            raise ValueError(f"no choices match the fixed inputs {fixed_features}")
        candidates, _ = optimize_acqf_discrete(acq, q=n, choices=choices)
        return candidates.detach()


def optimize(optimizer, acq, spec: ProblemSpec, n: int, fixed_features=None) -> torch.Tensor:
    """``optimizer.optimize``, passing ``fixed_features`` only when there are some,
    so custom optimizers written before it existed keep working without sources."""
    if fixed_features:
        return optimizer.optimize(acq, spec, n, fixed_features=fixed_features)
    return optimizer.optimize(acq, spec, n)


def default_optimizer(spec: ProblemSpec) -> CandidateOptimizer:
    """Pick the CandidateOptimizer the spec supports.

    - all-enumerated inputs -> ``DiscreteArgmax`` (``optimize_acqf_discrete``)
    - otherwise             -> ``Continuous`` (``optimize_acqf``; integers rounded,
      enumerated inputs alongside continuous ones enumerated)
    """
    if all(isinstance(d, EnumeratedDomain) for d in spec.inputs):
        return DiscreteArgmax()
    return Continuous()
