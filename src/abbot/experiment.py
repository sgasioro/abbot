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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch

# A design as the loop hands it out: {variable name: value}, in spec.variable_names
# order. The same shape ``Campaign.suggest`` returns.
Design = Mapping[str, float]


@runtime_checkable
class Experiment(Protocol):
    """Run one design and return what was measured. May block (slow hardware).

    ``design`` is ``{variable name: value}`` in ``spec.variable_names`` order — the
    same dict ``Campaign.suggest`` returns. Return either the outputs directly (a
    tensor / sequence / float in ``spec.outputs`` order, or a ``{output name:
    value}`` dict) or a raw observation (a spectrum, an image, ...) that the
    ``evaluate`` callable reduces to outputs.

    An experiment written against vectors (a vectorized simulator, sbido) can be
    wrapped in :class:`TensorExperiment`.
    """

    def run(self, design: Design) -> Any: ...


class TensorExperiment:
    """Adapt a vector-input experiment (``run(x: Tensor)``) to the design-dict
    contract. ``x`` is the ``(d,)`` double tensor in ``spec.variable_names`` order.
    """

    def __init__(self, experiment: Any) -> None:
        self.experiment = experiment

    def run(self, design: Design) -> Any:
        x = torch.tensor(list(design.values()), dtype=torch.double)
        return self.experiment.run(x)


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
    design dict and returns the measured output(s).
    ``variable_names`` is accepted for backward compatibility and ignored (the
    design dict carries the names).
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

    def run(self, design: Design) -> Any:
        if self.responder is not None:
            out = self.responder(design)
            if isinstance(out, (torch.Tensor, Mapping)):
                return out
            return torch.as_tensor(out, dtype=torch.double)
        return self._prompt(design)

    def _prompt(self, design: Design) -> torch.Tensor:  # pragma: no cover
        setting = ", ".join(f"{n}={v:.4g}" for n, v in design.items())
        print(f"\nRun the experiment at: {setting}")
        measured = []
        for name in self.output_names:
            raw = input(f"  measured {name}: ")
            measured.append(float(raw))
        return torch.tensor(measured, dtype=torch.double)


class SyntheticExperiment:
    """A spec-adaptive virtual experiment for mock / offline runs.

    Produces smooth, deterministic (seeded) RAW-physical outputs for WHATEVER
    ``ProblemSpec`` you built — any number of variables, any number of outputs
    (one column per ``spec.outputs``), optional observation noise. This is what
    makes the UI self-demonstrating: with
    a programmatic ``Experiment`` attached, auto-run (run/pause/step/stop) becomes
    available and a whole campaign drives itself without typed measurements.

    Each objective is a smooth Gaussian bump over the design space with its own
    (seeded) optimum location, so a single-objective problem has a clear maximum
    and a multi-objective problem has a genuine trade-off (the bumps peak in
    different places). Outputs are RAW and always "larger is better" in shape; the
    engine applies ``objective_signs`` for MINIMIZE objectives, so the mock does
    NOT need to know directions. Values are mapped to a friendly ~[0, 100] range.

    Deterministic: the landscape is fixed by ``seed`` and, when ``noise`` > 0, the
    per-evaluation noise is seeded by the candidate itself (so re-measuring the
    same point gives the same value — honest for a resumable run).
    """

    def __init__(self, spec: Any, *, seed: int = 0, noise: float = 0.0) -> None:
        self.spec = spec
        self.seed = seed
        self.noise = float(noise)
        bounds = spec.bounds_tensor()
        self._lo = bounds[0]
        self._hi = bounds[1]
        d = bounds.shape[-1]
        m = spec.n_outputs
        # Per-objective optima and widths, fixed by the seed.
        g = torch.Generator().manual_seed(seed)
        self._centers = torch.rand(m, d, generator=g, dtype=torch.double)
        self._widths = 0.15 + 0.25 * torch.rand(m, d, generator=g, dtype=torch.double)

    def run(self, design: Design) -> torch.Tensor:
        x = torch.tensor(list(design.values()), dtype=torch.double)
        span = (self._hi - self._lo).clamp(min=1e-9)
        xn = ((x - self._lo) / span).clamp(0.0, 1.0)  # normalized to [0, 1]^d
        # Gaussian bump per objective -> smooth value in (0, 100].
        sq = ((xn.unsqueeze(0) - self._centers) / self._widths) ** 2  # (m, d)
        vals = 100.0 * torch.exp(-0.5 * sq.sum(dim=-1))  # (m,)
        if self.noise > 0.0:
            # Seed noise by the candidate so a re-measurement repeats exactly.
            key = int(abs(hash((self.seed, tuple(round(float(v), 6) for v in xn.tolist())))) % (2**31))
            gn = torch.Generator().manual_seed(key)
            vals = vals + self.noise * torch.randn(vals.shape, generator=gn, dtype=torch.double)
        return vals
