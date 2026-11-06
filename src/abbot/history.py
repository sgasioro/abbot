"""Run history and a resumable DataStore.

Each ``Record`` carries an optional opaque ``aux`` payload
(a spectrum, replicate measurements, posterior samples) so a re-scoring reducer
(:class:`abbot.composite.Rescore`) runs without re-measuring. The ``aux`` payload is the one piece that is NOT trivially
JSON-serializable; keeping run state round-trippable with no live-object
references is the constraint that serves frontend + resume + agents at once
(see plan open-question #1 for the heavy-aux cap/drop policy, deferred).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class Record:
    """One evaluated candidate and everything we learned from it."""

    candidate: torch.Tensor  # (d,) design point
    observation: Any  # simulator/experiment-dependent payload
    outputs: torch.Tensor | None = None  # (m,) reduced objective value(s), if any
    output_var: torch.Tensor | None = None  # (m,) per-output noise VARIANCE, if known
    aux: Any | None = None  # opaque cached payload (e.g. theta samples)
    fault: bool = False  # the apparatus failed or a fault check rejected the shot
    retry_of: int | None = None  # index this shot re-runs, if any
    error: str | None = None  # experiment.run raised (then fault=True, outputs=None)
    id: str | None = None  # the Campaign candidate id this record closed (e.g. "c0007")
    cost: float | None = None  # what the run cost: caller-given, else step's measured run time
    acquisition: float | None = None  # the policy's acquisition value when it proposed this design
    # The design itself gave no usable result (a dark region, a sample that
    # decomposes): the reason, for people. Not a fault; the usability model learns it.
    bad_region: str | None = None
    # A person answered the halt on this design with "exclude": it isn't run again.
    excluded: bool = False
    # Timing (epoch seconds / seconds). issued_at: handed out by suggest. started_at /
    # finished_at: around the experiment run under ``step``; ask/tell knows only when
    # the result came back (finished_at). fit_seconds / acquisition_seconds: abbot's
    # own time fitting the model and computing the proposal this design came from
    # (shared by the picks of one batch; None for seeds and retries).
    issued_at: float | None = None
    started_at: float | None = None
    finished_at: float | None = None
    fit_seconds: float | None = None
    acquisition_seconds: float | None = None
    # Earlier values of ``outputs`` that Campaign.correct replaced, oldest first:
    # {"outputs": [...], "at": epoch seconds, "reason": str | None}.
    corrections: list[dict] | None = None

    @property
    def usable(self) -> bool:
        """Neither a fault nor a bad-region run: the output models train on it."""
        return not self.fault and self.bad_region is None

    @property
    def duration(self) -> float | None:
        """Experiment run time in seconds, when ``step`` ran it (else None)."""
        if self.started_at is None or self.finished_at is None:
            return None
        return self.finished_at - self.started_at


@dataclass
class History:
    records: list[Record] = field(default_factory=list)

    def append(self, record: Record) -> None:
        self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def clean(self) -> History:
        """The usable records: no fault, no bad region. The output models train on
        these."""
        return History(records=[r for r in self.records if r.usable])

    @property
    def judged(self) -> History:
        """Every record that isn't a fault: the clean ones plus the bad-region runs.
        The usability model trains on these."""
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

    @property
    def cost_spent(self) -> float:
        """Total recorded cost, failed runs included (records without a cost count 0)."""
        return float(sum(r.cost for r in self.records if r.cost is not None))

    @property
    def output_vars(self) -> torch.Tensor | None:
        """Stacked per-output noise variances ``(n, m)``, or ``None`` if ANY record
        lacks them.

        Per-point noise is all-or-nothing for a fit (botorch wants ``train_Yvar`` for
        every row or none), so this returns ``None`` the moment one record has no
        ``output_var`` — the loop then lets the model infer the noise level. These
        are *variances* in raw physical units²; variance is sign-invariant, so no
        canonical sign flip is ever applied to them.
        """
        if not self.records:
            return None
        vals = [r.output_var for r in self.records]
        if any(v is None for v in vals):
            return None
        return torch.stack([v.reshape(-1) for v in vals])


class ResumeMismatchError(Exception):
    """A store's saved fingerprint disagrees with the problem being resumed."""


_MISSING = object()


def _diff(old: Any, new: Any, path: str = "") -> list[tuple[str, Any, Any]]:
    """Leaf-level differences between two JSON values, as (path, old, new).

    A key absent on one side shows up with ``_MISSING`` on that side.
    """
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for k in sorted(set(old) | set(new), key=str):
            out += _diff(
                old.get(k, _MISSING), new.get(k, _MISSING), f"{path}.{k}" if path else str(k)
            )
        return out
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        out = []
        for i, (a, b) in enumerate(zip(old, new)):
            out += _diff(a, b, f"{path}[{i}]")
        return out
    return [] if old == new else [(path or "<root>", old, new)]


class DataStore:
    """Persist/resume a run as per-step files (``steps/step_NNN.pt``).

    Minimal implementation: ``torch.save`` each Record to
    ``<root>/steps/step_NNN.pt`` and reload them in order. A JSON sidecar for the
    serializable fields (and an aux cap/drop policy) is a follow-on.

    ``<root>/queue.json`` holds the work not yet in the history: in-flight
    candidates (with their ids), queued retries and seeds, and the id counter, so a
    resumed ``Campaign`` picks up exactly where it stopped.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.steps_dir = self.root / "steps"
        self.queue_path = self.root / "queue.json"
        self.fingerprint_path = self.root / "fingerprint.json"

    def check_fingerprint(self, fingerprint: dict, *, allow_mismatch: bool = False) -> None:
        """Refuse to resume a store written for a different problem.

        ``fingerprint`` is a JSON-able dict of whatever defines "the same problem"
        (a ``Campaign`` passes its spec plus any caller extras). The first call
        writes it; later calls compare. Keys the saved fingerprint lacks (fields a
        newer version added) are ignored, as is any mismatch while the store has no
        steps yet. A mismatch raises
        :class:`ResumeMismatchError` naming the differing keys, unless
        ``allow_mismatch`` (then it logs, keeps the old steps as valid history,
        and records the new fingerprint). Loading a foreign run's steps into a
        fresh optimization is silent corruption, not a crash, hence the check.
        """
        new = json.loads(json.dumps(fingerprint))  # normalize (tuples, key types)
        if not self.fingerprint_path.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            self.fingerprint_path.write_text(json.dumps(new, indent=1))
            return
        old = json.loads(self.fingerprint_path.read_text())
        if old == new:
            return
        n = len(list(self.steps_dir.glob("step_*.pt"))) if self.steps_dir.exists() else 0
        # Keys the saved fingerprint never had (a field added by a newer version)
        # can't distinguish the problems; nor can anything when no step was saved.
        diffs = [d for d in _diff(old, new) if d[1] is not _MISSING]
        if not diffs or n == 0:
            self.fingerprint_path.write_text(json.dumps(new, indent=1))
            return
        show = lambda v: "<absent>" if v is _MISSING else repr(v)
        lines = "\n".join(f"  {k}: {show(a)} -> {show(b)}" for k, a, b in diffs)
        if allow_mismatch:
            logger.warning(
                "Resuming %s (%d steps) despite a different problem; existing steps "
                "are kept as valid history:\n%s", self.root, n, lines,
            )
            self.fingerprint_path.write_text(json.dumps(new, indent=1))
            return
        raise ResumeMismatchError(
            f"{self.root}/ holds {n} steps from a different problem:\n{lines}\n"
            "Start a fresh store, or pass allow_resume_mismatch=True to keep the "
            "existing steps as valid history."
        )

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

    def save_queue(self, queue: dict) -> None:
        """Atomically write the campaign's outstanding-work queue."""
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.queue_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(queue, indent=1))
        os.replace(tmp, self.queue_path)

    def load_queue(self) -> dict | None:
        if not self.queue_path.exists():
            return None
        return json.loads(self.queue_path.read_text())
