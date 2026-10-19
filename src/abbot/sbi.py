"""Composite-SBI wiring: turn a PosteriorSampler + ObjectiveReducer into an
``evaluate`` callback for the loop / Campaign.

This is the pre-GP mount point (the plan's status quo): each observation is mapped
to posterior samples over theta, then reduced to a scalar F; the GP sees ``(xi, F)``.
The theta samples are returned alongside F so the loop caches them on the Record's
``aux`` payload (``evaluate -> (outputs, aux)`` seam), letting an annealed/re-scorable
reducer avoid re-sampling.

``Rescore`` is the matching ``Campaign(targets=...)`` hook: it re-reduces the
cached theta every step, so the objective can change during a run (an annealed
indicator, a target that tightens) without re-sampling.

No ``sbi`` import here — ``PosteriorSampler`` is structural (anything with
``sample(observation, n)`` satisfies it), so this module stays dependency-free.
``TrainedPosterior`` wraps an already-loaded ``sbi`` posterior object; loading one
from disk is the application's business (file layouts and pickling differ).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from abbot.experiment import PosteriorSampler
from abbot.history import History
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


class Rescore:
    """A ``Campaign(targets=...)`` hook: re-reduce every clean record's cached theta.

    Each step the surrogate is fitted to ``reducer(theta, spec, **schedule(t))`` for
    the clean records' cached posterior samples (``aux``, see
    :func:`sbi_evaluate`), so a schedule — an annealed indicator sharpness, a
    switch from a distance score to an indicator — changes the objective the
    policy optimizes and the hand-over is judged on, without re-sampling.

    ``t`` counts records after the first ``n_initial`` (the seeds): 0 on the first
    policy step, and failed shots count, since they spend a step of the run. The
    reducer may return one value per record or one per spec output.
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

    def step(self, history: History) -> int:
        """The schedule step ``t`` for this history."""
        return len(history) - self.n_initial

    def __call__(self, history: History, spec: ProblemSpec) -> torch.Tensor:
        clean = history.clean.records
        if any(r.aux is None for r in clean):
            raise ValueError(
                "Rescore needs cached posterior samples on every clean record "
                "(evaluate with sbi_evaluate(..., cache_theta=True))."
            )
        kwargs = self.schedule(self.step(history)) if self.schedule is not None else {}
        rows = [
            torch.as_tensor(self.reducer(torch.as_tensor(r.aux), spec, **kwargs)).reshape(-1)
            for r in clean
        ]
        if not rows:
            return torch.empty(0, spec.n_outputs, dtype=torch.double)
        return torch.stack(rows).double()


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
