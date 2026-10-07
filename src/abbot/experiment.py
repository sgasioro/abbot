"""Experiment-side protocols: how the world is probed and supervised.

Basis: ``~/sbido/src/sbido/interfaces.py`` (verbatim where sensible). These are
the seams the lab-experiment orchestration layer plugs into — one of the two
genuinely-novel surfaces abbot owns on top of botorch (the other is the
composite-SBI objective + aux layer).

``Experiment`` is deliberately a plain callable contract: a GUI action, an HTTP
reactor call, a simulator, or (later) an agent driving an instrument all satisfy
it. This is the headless-first property — the loop never knows which it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class Experiment(Protocol):
    """Evaluate one candidate. Async-friendly (may block on slow hardware)."""

    def run(self, candidate: torch.Tensor) -> Any: ...


@runtime_checkable
class PosteriorSampler(Protocol):
    """Optional (SBI): map an observation to posterior samples over theta.

    Returns (n_samples, d_theta). The composite-SBI app caches these on the
    Record ``aux`` payload so a re-scorable reducer avoids re-sampling.
    """

    def sample(self, observation: Any, n: int) -> torch.Tensor: ...


@dataclass
class Verdict:
    """What a ``Supervisor`` decided about the run so far (from sbido)."""

    retry: bool = False  # this observation is unusable; re-run the same candidate
    halt: str | None = None  # apparatus is broken; human-readable reason


@runtime_checkable
class Supervisor(Protocol):
    """Called once per shot with the full history. Decides retry vs halt.

    Also the in-the-loop-reasoner / stopping-condition seam for later agent work.
    """

    def __call__(self, history: Any) -> Verdict: ...
