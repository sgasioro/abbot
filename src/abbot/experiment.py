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


class ManualExperiment:
    """Human-in-the-loop Experiment: a person measures the proposed candidate.

    This is the M3 lab-tool's "experiment" — the loop proposes a design, a human
    runs the real-world reaction/synthesis and types back the measured value(s).
    By default it prompts on stdin, but a ``responder`` callable can be injected
    (so the same flow is headless-testable, and so a web frontend or an agent can
    supply the measurement instead of a terminal). The responder receives the
    proposed candidate tensor and returns the measured output(s).
    """

    def __init__(
        self,
        responder: Any | None = None,
        variable_names: list[str] | None = None,
        output_names: list[str] | None = None,
    ) -> None:
        self.responder = responder
        self.variable_names = variable_names
        self.output_names = output_names or ["value"]

    def run(self, candidate: torch.Tensor) -> torch.Tensor:
        if self.responder is not None:
            out = self.responder(candidate)
            return out if isinstance(out, torch.Tensor) else torch.as_tensor(out)
        return self._prompt(candidate)

    def _prompt(self, candidate: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        vals = candidate.reshape(-1).tolist()
        names = self.variable_names or [f"x{i}" for i in range(len(vals))]
        setting = ", ".join(f"{n}={v:.4g}" for n, v in zip(names, vals))
        print(f"\nRun the experiment at: {setting}")
        measured = []
        for name in self.output_names:
            raw = input(f"  measured {name}: ")
            measured.append(float(raw))
        return torch.tensor(measured, dtype=torch.double)
