"""Supervisors: per-shot fault detectors and trajectory monitors.

A :class:`~abbot.experiment.Supervisor` is called with the full ``History`` after
every result and returns a :class:`~abbot.experiment.Verdict`: ``retry`` marks the
latest shot unusable (it is kept out of the surrogate and re-run), ``halt`` says the
apparatus needs a human (the campaign stops, resumable), and ``stop`` says the
campaign is done without anything being wrong (a cost budget spent, the search
converged: :func:`cost_cap`, :func:`acquisition_monitor`).

Single-observation checks are written as ``detector(observation) -> bool`` and
lifted with :func:`per_shot`, so their authors never touch the history and cannot
accidentally condition on the trajectory. :func:`any_of` merges several supervisors
into the one a ``Campaign`` takes.

The trajectory monitors read a numeric *signal* per record — by default the raw
observation if it is numeric, else the record's outputs. Pass ``signal=`` to point
them at something else (e.g. ``lambda r: r.observation["obs"]``). Records without a
signal (a run that raised, a result ingested with no observation) are skipped.

The default thresholds were calibrated on one benchmark (sbido's small-angle
scattering problem); treat them as starting points for your apparatus. Calibrate once, offline, on held-out faulted and clean runs — tuning
a threshold on the run it is scored on reports a detection rate the lab can't get.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

from abbot.experiment import Supervisor, Verdict
from abbot.history import History, Record

Detector = Callable[[Any], bool]
Signal = Callable[[Record], Any]

# Below this relative spread, a window of observations counts as "unchanged". On
# sbido's benchmark a stuck episode measured 0.12 (observation noise alone) against
# 4.0 for a healthy run over the same designs.
STALL_OBS_TOL = 0.5

# Above this spread in normalized design coordinates, a window counts as "the
# designs moved". Deliberately large: a converged optimizer proposes near-duplicates
# on purpose, and firing on that would fire exactly when the optimizer works best.
STALL_XI_TOL = 0.15

# Spearman correlation above this, between step index and mean signal level, reads
# as a systematic trend rather than design-driven variation.
DRIFT_RHO_THRESHOLD = 0.9


def default_signal(record: Record) -> torch.Tensor | None:
    """The raw observation if it is numeric, else the record's outputs."""
    obs = record.observation
    if obs is not None and not isinstance(obs, (Mapping, str)):
        try:
            return torch.as_tensor(obs, dtype=torch.double)
        except (TypeError, ValueError, RuntimeError):
            pass
    return record.outputs


def per_shot(detector: Detector) -> Supervisor:
    """Lift a single-observation check into a supervisor.

    The detector judges the shot just taken and nothing else; letting it see earlier
    records would let one old fault flag every later clean shot.
    """

    def supervisor(history: History) -> Verdict:
        if not len(history):
            return Verdict()
        return Verdict(retry=bool(detector(history.records[-1].observation)))

    return supervisor


def any_of(*supervisors: Supervisor) -> Supervisor:
    """Merge supervisors: ``retry`` is an OR; the first non-None ``halt`` and the
    first non-None ``stop`` win."""

    def supervisor(history: History) -> Verdict:
        retry, halt, stop = False, None, None
        for s in supervisors:
            verdict = s(history)
            retry = retry or verdict.retry
            if halt is None:
                halt = verdict.halt
            if stop is None:
                stop = getattr(verdict, "stop", None)
        return Verdict(retry=retry, halt=halt, stop=stop)

    return supervisor


def _window(history: History, window: int, signal: Signal):
    """The last ``window`` records that carry a signal, as (step, design, signal)."""
    out = []
    for step in range(len(history) - 1, -1, -1):
        rec = history.records[step]
        value = signal(rec)
        if value is not None:
            out.append((step, rec.candidate.double(), torch.as_tensor(value).double()))
            if len(out) == window:
                break
    return out[::-1]


def _max_pairwise(values: torch.Tensor) -> float:
    if len(values) < 2:
        return 0.0
    flat = values.reshape(len(values), -1)
    return float(torch.cdist(flat, flat).max())


def _span(bounds: torch.Tensor) -> torch.Tensor:
    bounds = torch.as_tensor(bounds, dtype=torch.double)
    return (bounds[1] - bounds[0]).clamp(min=1e-12)


def stall_monitor(
    bounds: torch.Tensor,
    window: int = 6,
    xi_tol: float = STALL_XI_TOL,
    obs_tol: float = STALL_OBS_TOL,
    *,
    signal: Signal = default_signal,
) -> Supervisor:
    """Halt when the designs moved but the observations didn't (a stuck controller).

    ``bounds`` is the ``(2, d)`` design box (``spec.bounds_tensor()``); design spread
    is measured in units of it.
    """
    span = _span(bounds)

    def supervisor(history: History) -> Verdict:
        win = _window(history, window, signal)
        if len(win) < window:
            # Two shots that happen to look alike are not evidence of a broken rig.
            return Verdict()
        designs = torch.stack([x for _, x, _ in win]) / span
        obs = torch.stack([s.reshape(-1) for _, _, s in win])
        scale = obs.abs().mean().clamp(min=1e-12)
        xi_spread = _max_pairwise(designs)
        obs_spread = _max_pairwise(obs) / float(scale) / max(obs.shape[-1] ** 0.5, 1.0)
        if xi_spread > xi_tol and obs_spread < obs_tol:
            first, last = win[0][0], win[-1][0]
            return Verdict(
                halt=(
                    f"apparatus appears unresponsive (stall_monitor).\n"
                    f"Over steps {first}-{last} the designs varied by up to "
                    f"{xi_spread:.2f} of the box while the observations varied by "
                    f"{100 * obs_spread:.2f}% — consistent with a stuck controller."
                )
            )
        return Verdict()

    return supervisor


def drift_monitor(
    bounds: torch.Tensor,
    window: int = 8,
    xi_tol: float = STALL_XI_TOL,
    rho_threshold: float = DRIFT_RHO_THRESHOLD,
    *,
    signal: Signal = default_signal,
) -> Supervisor:
    """Halt when the signal trends monotonically while the designs sit still.

    Flux decay is the motivating case: no single observation looks wrong, but the
    overall level slides in one direction shot after shot.
    """
    span = _span(bounds)

    def supervisor(history: History) -> Verdict:
        win = _window(history, window, signal)
        if len(win) < window:
            return Verdict()
        designs = torch.stack([x for _, x, _ in win]) / span
        if _max_pairwise(designs) > xi_tol:
            # The designs are still exploring, so level changes are explained.
            return Verdict()
        level = torch.stack([s.mean() for _, _, s in win])
        # Spearman = Pearson on ranks; with distinct values the ranks are the argsort.
        ranks = level.argsort().argsort().double()
        steps = torch.arange(window, dtype=torch.double)
        rho = float(torch.corrcoef(torch.stack([steps, ranks]))[0, 1].nan_to_num(0.0))
        if abs(rho) > rho_threshold:
            return Verdict(
                halt=(
                    f"observations are trending monotonically (drift_monitor): "
                    f"rank correlation {rho:+.2f} with step index over steps "
                    f"{win[0][0]}-{win[-1][0]}, while the designs barely moved — "
                    f"consistent with a decaying source."
                )
            )
        return Verdict()

    return supervisor


def consecutive_fault_monitor(detector: Detector, k: int = 3) -> Supervisor:
    """Halt after ``k`` consecutive shots the detector rejects.

    Without it, a detector that never lets up is a valid trajectory that nothing
    escalates: every shot is retried and the surrogate starves. A record whose run
    raised counts as rejected; one with no observation is not judged.
    """

    def rejected(r: Record) -> bool | None:
        if r.error is not None:
            return True
        if r.observation is None:
            return None
        return bool(detector(r.observation))

    def supervisor(history: History) -> Verdict:
        if len(history) < k:
            return Verdict()
        recent = [rejected(r) for r in history.records[-k:]]
        if all(recent):
            n = len(history)
            return Verdict(
                halt=(
                    f"{k} consecutive shots failed the per-shot detector "
                    f"(consecutive_fault_monitor), steps {n - k}-{n - 1}."
                )
            )
        return Verdict()

    return supervisor


def standard_supervisor(
    bounds: torch.Tensor, detector: Detector, *, signal: Signal = default_signal
) -> Supervisor:
    """The usual stack: the per-shot detector, both trajectory monitors, and the
    consecutive-fault escalation.

    The two trajectory monitors are gated on design spread in opposite directions,
    so at most one can fire in a window; together they cover the optimizer from
    early exploration through convergence. Compose :func:`any_of` directly for
    anything more specific.
    """
    return any_of(
        per_shot(detector),
        stall_monitor(bounds, signal=signal),
        drift_monitor(bounds, signal=signal),
        consecutive_fault_monitor(detector),
    )


def cost_cap(budget: float) -> Supervisor:
    """Stop once the recorded cost reaches ``budget`` (``History.cost_spent``, in the
    records' cost units; failed runs count, since they were paid for).

    A stop, not a halt: nothing is wrong, the campaign is simply done. A run already
    in flight when the cap is crossed still lands.
    """
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")

    def supervisor(history: History) -> Verdict:
        spent = history.cost_spent
        if spent >= budget:
            return Verdict(stop=f"cost budget reached (cost_cap): spent {spent:.4g} of {budget:.4g}.")
        return Verdict()

    return supervisor


def acquisition_monitor(threshold: float, window: int = 3) -> Supervisor:
    """Stop when the last ``window`` policy proposals all had an acquisition value
    below ``threshold``: the policy no longer expects a run to be worth doing.

    Reads ``Record.acquisition`` (set from the policy's ``last_acquisition_value``),
    so it is in the policy's own units: log-EI for ``SimpleBO`` (log-EI minus log
    cost with a ``cost_model``), information gain for InfoBAX, posterior std for
    uncertainty sampling. Records without a value (seeds, random probes, retries)
    are skipped. Pick ``threshold`` per campaign.
    """
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")

    def supervisor(history: History) -> Verdict:
        values = [r.acquisition for r in history.records if r.acquisition is not None]
        recent = values[-window:]
        if len(recent) == window and max(recent) < threshold:
            return Verdict(
                stop=(
                    f"converged (acquisition_monitor): the last {window} proposal(s) "
                    f"had acquisition values below {threshold:.4g} "
                    f"(highest {max(recent):.4g})."
                )
            )
        return Verdict()

    return supervisor
