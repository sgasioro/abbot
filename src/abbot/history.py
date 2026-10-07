"""Run history and a resumable DataStore.

Generalizes ``~/sbido/src/sbido/interfaces.py::History`` and the step-file resume
in ``optimizer.py``. Each ``Record`` carries an optional opaque ``aux`` payload
(e.g. cached theta posterior samples) so an annealed/re-scorable reducer re-runs
without re-sampling. The ``aux`` payload is the one piece that is NOT trivially
JSON-serializable; keeping run state round-trippable with no live-object
references is the constraint that serves frontend + resume + agents at once
(see plan open-question #1 for the heavy-aux cap/drop policy, deferred).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class Record:
    """One evaluated candidate and everything we learned from it."""

    candidate: torch.Tensor  # (d,) design point
    observation: Any  # simulator/experiment-dependent payload
    outputs: torch.Tensor | None = None  # (m,) reduced objective value(s), if any
    aux: Any | None = None  # opaque cached payload (e.g. theta samples)
    fault: bool = False  # detector flagged this shot as unusable
    retry_of: int | None = None  # index this shot re-runs, if any


@dataclass
class History:
    records: list[Record] = field(default_factory=list)

    def append(self, record: Record) -> None:
        self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def clean(self) -> History:
        """The records the detector did not reject."""
        return History(records=[r for r in self.records if not r.fault])

    @property
    def designs(self) -> torch.Tensor:
        """All evaluated designs stacked: (n, d)."""
        if not self.records:
            return torch.empty(0)
        return torch.stack([r.candidate for r in self.records])

    @property
    def outputs(self) -> torch.Tensor:
        """Stacked raw outputs: (n, m). Requires every record to carry outputs.

        Outputs are the *raw physical* measurements (the engine applies objective
        signs for maximize-canonical fitting elsewhere, see ``loop._canonical_Y``).
        """
        if not self.records:
            return torch.empty(0)
        vals = [r.outputs for r in self.records]
        if any(v is None for v in vals):
            raise ValueError("History.outputs: some records have no outputs set")
        widths = {int(v.reshape(-1).shape[0]) for v in vals}
        if len(widths) != 1:
            raise ValueError(
                f"History.outputs: records have inconsistent objective counts {sorted(widths)}; "
                "every evaluate() return must have the same number of columns."
            )
        return torch.stack([v.reshape(-1) for v in vals])


class DataStore:
    """Persist/resume a run as per-step files (sbido ``step_NNN.pt`` pattern).

    Minimal M1 implementation: ``torch.save`` each Record to
    ``<root>/steps/step_NNN.pt`` and reload them in order. A JSON sidecar for the
    serializable fields (and an aux cap/drop policy) is a follow-on.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.steps_dir = self.root / "steps"

    def setup(self) -> None:
        self.steps_dir.mkdir(parents=True, exist_ok=True)

    def save_step(self, step: int, record: Record) -> None:
        self.setup()
        torch.save(record, self.steps_dir / f"step_{step:03d}.pt")

    def load(self) -> History:
        history = History()
        if not self.steps_dir.exists():
            return history
        for f in sorted(self.steps_dir.glob("step_*.pt")):
            history.append(torch.load(f, weights_only=False))
        return history
