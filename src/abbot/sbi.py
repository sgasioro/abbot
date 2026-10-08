"""Composite-SBI wiring: turn a PosteriorSampler + ObjectiveReducer into an
``evaluate`` callback for the loop / Campaign.

This is the pre-GP mount point (the plan's status quo): each observation is mapped
to posterior samples over theta, then reduced to a scalar F; the GP sees ``(xi, F)``.
The theta samples are returned alongside F so the loop caches them on the Record's
``aux`` payload (``evaluate -> (outputs, aux)`` seam), letting an annealed/re-scorable
reducer avoid re-sampling.

No ``sbi`` / sbido import here — ``PosteriorSampler`` is structural (sbido's
``AnalyticalGaussianPosterior`` and ``SBIPosteriorSampler`` satisfy it by shape),
so this module stays dependency-free; the heavy wrappers live in interop.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from abbot.experiment import PosteriorSampler
from abbot.problem import ProblemSpec
from abbot.reduce import ObjectiveReducer


def sbi_evaluate(
    sampler: PosteriorSampler,
    reducer: ObjectiveReducer,
    spec: ProblemSpec,
    *,
    n_samples: int = 2048,
    cache_theta: bool = True,
) -> Callable[[object], object]:
    """Build an ``evaluate(observation) -> (F, theta)`` callback.

    ``F`` is the reduced scalar objective (shape (1,)); ``theta`` is the cached
    posterior sample set (or None if ``cache_theta`` is False). Pass the result as
    ``run_campaign(evaluate=...)`` or ``Campaign.step(evaluate=...)``.
    """

    def evaluate(observation: object) -> object:
        theta = sampler.sample(observation, n_samples)
        f = reducer(theta, spec).reshape(1)
        return (f, theta) if cache_theta else f

    return evaluate


def rescore(history, reducer: ObjectiveReducer, spec: ProblemSpec, **reducer_kwargs):
    """Re-reduce cached theta on every record (e.g. with a new annealing ``beta``).

    Returns a list of scalar F values. Uses each Record's cached ``aux`` theta, so
    no re-sampling — the whole point of caching. Records without cached theta are
    skipped (None in the output).
    """
    out = []
    for r in history.records:
        if r.aux is None:
            out.append(None)
            continue
        theta = r.aux if isinstance(r.aux, torch.Tensor) else torch.as_tensor(r.aux)
        out.append(float(reducer(theta, spec, **reducer_kwargs).reshape(-1)[0]))
    return out
