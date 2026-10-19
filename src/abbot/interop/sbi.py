"""Simulation-based inference as a composite objective.

The analysis step of a :mod:`abbot.composite` objective is "draw posterior samples
over theta for this observation": a :class:`PosteriorSampler`. ``sbi_evaluate``
builds the loop's ``evaluate`` from a sampler and a reducer (the samples are
cached on each record's ``aux`` for :class:`abbot.composite.Rescore`), and
``TrainedPosterior`` adapts a trained ``sbi`` posterior object (``abbot[sbi]``) to
the sampler protocol. Loading a posterior from disk is the application's
business: file layouts and pickling differ.

Nothing here imports ``sbi``; a sampler is anything with ``sample(observation, n)``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import torch

from abbot.composite import ObjectiveReducer, composite_evaluate
from abbot.problem import ProblemSpec


@runtime_checkable
class PosteriorSampler(Protocol):
    """Map an observation to posterior samples over theta: ``(n, d_theta)``."""

    def sample(self, observation: Any, n: int) -> torch.Tensor: ...


def sbi_evaluate(
    sampler: PosteriorSampler,
    reducer: ObjectiveReducer,
    spec: ProblemSpec,
    *,
    n_samples: int = 2048,
    cache_theta: bool = True,
) -> Callable[[object], object]:
    """``composite_evaluate`` with posterior sampling as the analysis:
    ``evaluate(observation) -> (F, theta)`` (or just ``F`` without ``cache_theta``).
    """
    return composite_evaluate(
        lambda observation: sampler.sample(observation, n_samples),
        reducer,
        spec,
        cache=cache_theta,
    )


class TrainedPosterior:
    """A trained ``sbi`` posterior as a :class:`PosteriorSampler`.

    ``transform`` maps an observation to the network's input ``x`` (1, d_x) —
    the application's preprocessing. Samples come back in the network's
    parameter space and are mapped to physical units by ``theta_min``/``theta_max``
    when the network was trained on min-max-scaled theta.
    """

    def __init__(
        self,
        posterior: Any,
        transform: Callable[[Any], torch.Tensor] | None = None,
        theta_min: torch.Tensor | None = None,
        theta_max: torch.Tensor | None = None,
    ) -> None:
        self.posterior = posterior
        self.transform = transform
        self.theta_min = theta_min
        self.theta_max = theta_max

    def sample(self, observation: Any, n: int = 2048) -> torch.Tensor:
        x = self.transform(observation) if self.transform is not None else observation
        x = torch.as_tensor(x).float().reshape(1, -1)
        samples = self.posterior.sample((n,), x=x).cpu().float()
        if self.theta_min is not None:
            samples = samples * (self.theta_max - self.theta_min) + self.theta_min
        return samples
