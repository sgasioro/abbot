"""Supervisors: fault checks and stop rules.

A :class:`Supervisor` watches a campaign. After every result it is called with the
full ``History`` and returns a :class:`~abbot.experiment.Verdict`. It has two kinds
of rule:

- **Fault checks** (``faults=``) judge whether the run is usable. ``retry`` says
  the apparatus may have spoiled the latest run (kept out of the model and re-run,
  up to the campaign's ``max_retries``, then a person decides); ``halt`` says the
  apparatus needs a person (the campaign ends, resumably). ``bad_region`` says the
  design itself gives no usable result (a dark region): no retry, and the
  campaign learns to avoid the region. :func:`per_shot`, :func:`stall_monitor`,
  :func:`drift_monitor`, :func:`consecutive_fault_monitor`;
  :func:`standard_supervisor` bundles them.
- **Stop rules** (``stops=``) say the campaign is done, with nothing wrong:
  :func:`target_reached`, :func:`no_improvement`, :func:`cost_cap`,
  :func:`acquisition_monitor`, :func:`min_runs`. A list of them stops on the first
  that fires; :func:`all_of` stops only when every rule inside agrees (a floor:
  ``all_of(no_improvement(10), min_runs(30))``), and :func:`any_of` is "or", for
  nesting inside it.

::

    Supervisor(
        faults=[per_shot(detector), stall_monitor(bounds)],
        stops=[target_reached(0.95), all_of(no_improvement(10), min_runs(30))],
    )

Fault checks come first: a run they reject can't satisfy a stop rule, and a halt
ends the campaign whatever the stop rules say. Each rule is a plain callable
``(history) -> Verdict``, so writing your own is a function; a plain callable is
also accepted wherever a ``Supervisor`` is.

Single-observation checks are written as ``detector(observation) -> bool`` and
lifted with :func:`per_shot`, so their authors never touch the history and cannot
accidentally condition on the trajectory.

The trajectory monitors read a numeric *signal* per record — by default the raw
observation if it is numeric, else the record's outputs. Pass ``signal=`` to point
them at something else (e.g. ``lambda r: r.observation["obs"]``). Records without a
signal (a run that raised, a result ingested with no observation) are skipped.

The default thresholds were calibrated on one benchmark (a small-angle
scattering problem); treat them as starting points for your apparatus. Calibrate once, offline, on held-out faulted and clean runs — tuning
a threshold on the run it is scored on reports a detection rate the lab can't get.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import torch

from abbot.experiment import Verdict, _bad_region_reason
from abbot.history import History, Record

if TYPE_CHECKING:
    from abbot.problem import ProblemSpec

Detector = Callable[[Any], bool]
Signal = Callable[[Record], Any]
# One fault check or stop rule: called with the history after every result.
Rule = Callable[[History], Verdict]

# Below this relative spread, a window of observations counts as "unchanged". On
# that benchmark a stuck episode measured 0.12 (observation noise alone) against
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


def per_shot(detector: Detector, *, bad_region: str | bool = False) -> Rule:
    """Lift a single-observation check into a supervisor.

    The detector judges the shot just taken and nothing else; letting it see earlier
    records would let one old fault flag every later clean shot.

    By default a flagged shot is an apparatus fault: retried, then a person decides.
    ``bad_region`` (a reason, or True) is for a detector that can only fire when the
    design itself is the cause: a flagged shot is then a bad-region run, not retried.
    A detector that can't tell the two apart (a flat profile from a dark region or a
    closed shutter) should keep the default.
    """

    def supervisor(history: History) -> Verdict:
        if not len(history):
            return Verdict()
        flagged = bool(detector(history.records[-1].observation))
        if flagged and bad_region:
            return Verdict(bad_region=bad_region)
        return Verdict(retry=flagged)

    return supervisor


class Supervisor:
    """What watches a campaign: fault checks, then stop rules.

    ``faults`` and ``stops`` each take a rule or a list of rules (any one of them is
    enough: any fault check can reject a run or halt, any stop rule can end the
    campaign). Fault checks run first and a run they reject is marked before the
    stop rules see the history, so it can't satisfy one; on a halt the stop rules
    aren't asked. The roles set that order and nothing more: whatever a rule
    returns (retry, halt or stop) is honored.

    Pass it as ``Campaign(spec, supervisor=...)`` (or ``run_campaign``).
    """

    def __init__(self, faults: Rule | list[Rule] = (), stops: Rule | list[Rule] = ()) -> None:
        self.faults, self.stops = _rules(faults), _rules(stops)

    def bind(self, spec) -> Rule:
        return Supervisor([bind(r, spec) for r in self.faults], [bind(r, spec) for r in self.stops])

    def __call__(self, history: History) -> Verdict:
        verdict = _merge([r(history) for r in self.faults], every=False)
        if verdict.halt is not None or not self.stops:
            return verdict
        if len(history):  # what the campaign is about to record (a retry outranks)
            if verdict.retry:
                history.records[-1].fault = True
            elif (bad := _bad_region_reason(verdict.bad_region)) is not None:
                history.records[-1].bad_region = bad
        stops = _merge([r(history) for r in self.stops], every=False)
        return Verdict(
            retry=verdict.retry or stops.retry,
            halt=stops.halt,
            stop=verdict.stop if verdict.stop is not None else stops.stop,
            bad_region=verdict.bad_region if verdict.bad_region else stops.bad_region,
        )

    def __repr__(self) -> str:
        return f"Supervisor(faults={self.faults!r}, stops={self.stops!r})"


def _rules(rules) -> list:
    if callable(rules):
        return [rules]
    return list(rules)


def any_of(*rules: Rule) -> Rule:
    """Any one of ``rules``: stops on the first stop, retries if any asks, halts on
    the first halt.

    In a :class:`Supervisor`, a list already means "any", so this is mostly for an
    "or" inside :func:`all_of`: ``all_of(any_of(a, b), min_runs(30))``.
    """
    return _Merge(rules, every=False)


def all_of(*rules: Rule) -> Rule:
    """Stop only when every one of ``rules`` says stop; for floors on a stop rule.

    ``all_of(no_improvement(10), min_runs(30))`` stops on no improvement, but not
    before 30 runs. Put stop rules in it: a fault check never says stop, so it would
    block the stop (fault checks go in ``Supervisor(faults=...)``). A retry or halt
    from a rule inside still passes through.
    """
    return _Merge(rules, every=True)


class _Merge:
    """:func:`any_of` (``every=False``) or :func:`all_of` (``every=True``)."""

    def __init__(self, rules, every: bool) -> None:
        self.rules, self.every = tuple(rules), every
        if every and not self.rules:
            raise ValueError("all_of needs at least one rule.")

    def bind(self, spec) -> _Merge:
        return _Merge((bind(r, spec) for r in self.rules), self.every)

    def __call__(self, history: History) -> Verdict:
        return _merge([r(history) for r in self.rules], self.every)


def _merge(verdicts: list[Verdict], every: bool) -> Verdict:
    """Retry if any asks, the first halt, the first bad region, and the first stop
    (``every``: a stop only if all say stop, their reasons joined)."""
    retry = any(v.retry for v in verdicts)
    halt = next((v.halt for v in verdicts if v.halt is not None), None)
    bad = next((b for v in verdicts if (b := getattr(v, "bad_region", None))), None)
    stops = [getattr(v, "stop", None) for v in verdicts]
    if every:
        agreed = stops and all(x is not None for x in stops)
        stop = " and ".join(x.rstrip(".") for x in stops) + "." if agreed else None
    else:
        stop = next((x for x in stops if x is not None), None)
    return Verdict(retry=retry, halt=halt, stop=stop, bad_region=bad)


def bind(supervisor: Rule, spec) -> Rule:
    """``supervisor`` (or a rule) with the problem spec filled in, for rules that
    read it (:func:`target_reached`, :func:`no_improvement`). ``Campaign`` does this for
    you; call it yourself only to run such a supervisor outside a campaign."""
    return supervisor.bind(spec) if hasattr(supervisor, "bind") else supervisor


class _NeedsSpec:
    """A supervisor that reads the spec (output names, directions, constraints)."""

    def __init__(self, name: str, make: Callable) -> None:
        self.name, self.make = name, make

    def bind(self, spec) -> Rule:
        return self.make(spec)

    def __call__(self, history: History) -> Verdict:
        raise TypeError(
            f"{self.name} needs the problem spec: pass it in Campaign(supervisor=...), "
            "which fills it in, or call abbot.supervision.bind(supervisor, spec) first."
        )


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
) -> Rule:
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
) -> Rule:
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


def consecutive_fault_monitor(detector: Detector, k: int = 3) -> Rule:
    """Halt after ``k`` consecutive shots the detector rejects.

    Without it, a detector that never lets up is a valid trajectory that nothing
    escalates: every shot is retried and the surrogate starves. A record whose run
    raised counts as rejected; one with no observation is not judged, and a
    bad-region run is not an apparatus fault, so it breaks the streak.
    """

    def rejected(r: Record) -> bool | None:
        if r.bad_region is not None:
            return False
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
    early exploration through convergence. Build a :class:`Supervisor` yourself for
    anything more specific; add stop rules with ``Supervisor(faults=[standard_supervisor(
    ...)], stops=[...])``.
    """
    return Supervisor(faults=[
        per_shot(detector),
        stall_monitor(bounds, signal=signal),
        drift_monitor(bounds, signal=signal),
        consecutive_fault_monitor(detector),
    ])


def cost_cap(budget: float) -> Rule:
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


def min_runs(n: int) -> Rule:
    """Stop once ``n`` runs are recorded (every record counts, failures included, as
    for a campaign's budget). Meant as a floor inside :func:`all_of`; on its own it is
    a run budget.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")

    def supervisor(history: History) -> Verdict:
        if len(history) >= n:
            return Verdict(stop=f"at least {n} runs (min_runs): {len(history)} recorded.")
        return Verdict()

    return supervisor


def acquisition_monitor(threshold: float, window: int = 3) -> Rule:
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


def _runs(history: History, spec: ProblemSpec) -> list[tuple[int, torch.Tensor, bool]]:
    """``(step, raw outputs, satisfies the output constraints)`` for each clean,
    measured run."""
    from abbot.feasibility import feasible

    out = []
    for step, r in enumerate(history.records):
        if not r.usable or r.outputs is None:
            continue
        y = r.outputs.double().reshape(-1)
        ok = not spec.has_output_constraints or bool(feasible(spec, y.reshape(1, -1))[0])
        out.append((step, y, ok))
    return out


def _directed(spec: ProblemSpec, name: str, caller: str) -> tuple[int, float]:
    """Column and sign (+1 maximize, -1 minimize) of the output ``name``."""
    names = spec.output_names
    if name not in names:
        raise ValueError(f"{caller}: no output named {name!r}; the outputs are {names}.")
    j = names.index(name)
    direction = spec.outputs[j].direction
    if direction not in ("MAXIMIZE", "MINIMIZE"):
        raise ValueError(
            f"{caller}: output {name!r} is {direction}, so it has no better or worse; "
            "name an output you maximize or minimize."
        )
    return j, 1.0 if direction == "MAXIMIZE" else -1.0


def _the_objective(spec: ProblemSpec, caller: str, hint: str) -> str:
    if spec.n_objectives != 1:
        raise ValueError(f"{caller}: the spec has {spec.n_objectives} objectives; {hint}")
    return spec.objectives[0].name


def target_reached(target: float | Mapping[str, float]) -> Rule:
    """Stop once a run reaches ``target``: at or above it for an output you
    maximize, at or below for one you minimize.

    ``target`` is a number for a spec with one objective, or ``{name: value}``: one
    run must then reach every named value. Only clean runs count, and with output
    constraints only runs that satisfy them. Values are the raw measurements (not
    re-derived by ``targets=``), and a single noisy run can cross the line by luck:
    for a noisy output, pick a target with that margin in mind.

    The spec supplies the directions: pass this to ``Campaign(supervisor=...)``.
    """
    if isinstance(target, Mapping):
        if not target:
            raise ValueError("target_reached: name at least one output.")
        wanted = {str(k): float(v) for k, v in target.items()}
    else:
        wanted = {None: float(target)}

    def make(spec: ProblemSpec) -> Rule:
        if None in wanted:
            name = _the_objective(
                spec, "target_reached",
                "name the outputs, e.g. target_reached({'yield': 0.9, 'purity': 0.95}).",
            )
            goals = {name: wanted[None]}
        else:
            goals = wanted
        cols = {name: (*_directed(spec, name, "target_reached"), value)
                for name, value in goals.items()}

        def meets(y: torch.Tensor) -> bool:
            return all(sign * (float(y[j]) - value) >= 0 for j, sign, value in cols.values())

        def supervisor(history: History) -> Verdict:
            for step, y, ok in _runs(history, spec):
                if ok and meets(y):
                    got = ", ".join(
                        f"{name} = {float(y[j]):.4g} (target {value:.4g})"
                        for name, (j, _, value) in cols.items()
                    )
                    return Verdict(stop=f"target reached (target_reached): run {step} has {got}.")
            return Verdict()

        return supervisor

    return _NeedsSpec("target_reached", make)


def no_improvement(patience: int, min_delta: float = 0.0) -> Rule:
    """Stop when the best value of the objective hasn't improved by more than
    ``min_delta`` over the last ``patience`` runs.

    Counts clean, measured runs, seeds included, so pick ``patience`` well above
    the number of seeds. The best is the best measured value (raw, among runs that
    satisfy any output constraints), so on a noisy output one lucky run counts as
    an improvement: set ``min_delta`` to about the noise level. Nothing stops before
    some run satisfies the constraints. For several objectives, use
    :func:`acquisition_monitor`.

    The spec supplies the direction: pass this to ``Campaign(supervisor=...)``.
    """
    if patience < 1:
        raise ValueError(f"patience must be at least 1, got {patience}")
    if min_delta < 0:
        raise ValueError(f"min_delta must be non-negative, got {min_delta}")

    def make(spec: ProblemSpec) -> Rule:
        name = _the_objective(
            spec, "no_improvement",
            "it watches one, so use acquisition_monitor for several.",
        )
        j, sign = _directed(spec, name, "no_improvement")

        def best(runs) -> float | None:
            values = [sign * float(y[j]) for _, y, ok in runs if ok]
            return max(values) if values else None

        def supervisor(history: History) -> Verdict:
            runs = _runs(history, spec)
            if len(runs) <= patience:
                return Verdict()
            before, now = best(runs[:-patience]), best(runs)
            if before is None or now - before > min_delta:
                return Verdict()
            return Verdict(
                stop=(
                    f"no improvement (no_improvement): the best {name} is "
                    f"{sign * now:.4g}, and the last {patience} runs improved it by "
                    f"{now - before:.4g} (min_delta {min_delta:.4g})."
                )
            )

        return supervisor

    return _NeedsSpec("no_improvement", make)
