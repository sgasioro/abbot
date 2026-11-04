"""The ``Campaign`` session API — the headless, step-based ask/tell surface.

This is the framework-agnostic core that the web UI, a notebook, ``run_campaign``,
and (later) an agent all drive. It exposes the gest-aligned contract as DISCRETE
calls — ``suggest`` / ``ingest`` / ``step`` / ``state`` / ``recommend`` — rather
than a blocking loop, so a human (or a timer, or an agent) decides when each step
fires. The engine logic (fit, propose, canonical-Y, recommendation) lives here as
module helpers that both ``Campaign`` and ``loop.run_campaign`` share — the loop
is just a thin driver over a ``Campaign``.

Auto vs. manual is NOT a concern here: this object only knows how to take one step
when asked. "Who triggers the step" (a human click, an auto-run timer, an agent)
is a caller concern. ``Supervisor`` fault detection is orthogonal and runs inside
``ingest`` in every mode; a halt verdict is surfaced on the state, not raised, so
the caller chooses how to react.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import torch
from botorch.utils.multi_objective.pareto import is_non_dominated

from abbot.experiment import BatchExperiment, Experiment, Verdict, _bad_region_reason
from abbot.feasibility import FEASIBLE_P, feasible, prob_feasible, row_scale, violation
from abbot.history import DataStore, History, Record
from abbot.model import ModelConstructor, default_constructor
from abbot.policy import Policy, default_policy
from abbot.problem import ProblemSpec
from abbot.supervision import Rule, Supervisor, bind
from abbot.usability import USABLE_P, Usability, fit_usability

if TYPE_CHECKING:
    from botorch.models.model import Model

    from abbot.diagnostics import CheckReport

logger = logging.getLogger(__name__)

# A model's capability: what UQ quality its posterior actually provides. Kept as a
# backward-compatible alias of model.UQCapability.
ModelUQ = str  # "point" | "samples" | "calibrated_variance"

_SATISFIES: dict[str, set[str]] = {
    "point": {"point", "samples", "calibrated_variance"},
    "samples": {"samples", "calibrated_variance"},
    "calibrated_variance": {"calibrated_variance"},
}


def default_initial_designs(spec: ProblemSpec, *, seed: int | None = None) -> torch.Tensor:
    """Space-filling seed designs for a fresh campaign: ``2d + 1`` Sobol points over
    every input (at least ``2m`` for ``m`` objectives, ParetoBO's cold-start floor).
    With sources, the last two are moved to the target so the model sees it from the
    start.
    """
    n = max(2 * len(spec.inputs) + 1, 2 * spec.n_objectives)
    X = spec.sample_designs(n, seed=seed)
    if spec.sources:
        X[-2:] = spec.at_target(X[-2:])
    return X


class PersistentFaultError(RuntimeError):
    """The apparatus needs a human. Every completed step is already on disk."""


def check_uq_pairing(policy: Policy, model_constructor: ModelConstructor) -> None:
    """Refuse a policy<->model pairing the surrogate can't support.

    Reads the UQ capability off the actual ``model_constructor`` (``provides_uq``) —
    the object that determines it — so the guardrail can't be defeated by a caller
    passing a string that disagrees with their model. E.g. a point-estimate NN
    constructor + ``UncertaintySampling`` (needs calibrated variance) is refused; the
    same NN + ``MeanBAX`` (needs only a point prediction) is fine.
    """
    required = policy.uq_requirement
    provided = getattr(model_constructor, "provides_uq", "calibrated_variance")
    if provided not in _SATISFIES[required]:
        _NEEDS = {
            "point": "a point prediction",
            "samples": "coherent posterior samples",
            "calibrated_variance": "calibrated uncertainty estimates",
        }
        raise ValueError(
            f"{type(policy).__name__} needs {_NEEDS[required]} from the model, but "
            f"this model only provides {_NEEDS[provided]}. Use a model that provides "
            f"{_NEEDS[required]} (a Gaussian-process surrogate does), or pick a method "
            "that needs less (e.g. a greedy/MeanBAX policy only needs a point prediction)."
        )


# --- shared engine helpers (used by Campaign and run_campaign) ----------------


def split_eval(result: object, spec: ProblemSpec) -> tuple[torch.Tensor, object | None]:
    """Normalize an ``evaluate`` return into ``(raw_outputs, aux)``.

    Accepts a bare tensor / sequence / ``{output name: value}`` dict (aux=None) or
    an ``(outputs, aux)`` tuple. Validates the OUTPUT columns against the spec so a
    wrong-width or misnamed return fails here with a clear message instead of
    crashing downstream.
    """
    if isinstance(result, tuple):
        outputs, aux = result
    else:
        outputs, aux = result, None
    if isinstance(outputs, Mapping):
        outputs = _outputs_from_dict(outputs, spec)
    # dtype up front: python floats would otherwise round-trip through float32.
    outputs = torch.as_tensor(outputs, dtype=torch.double).reshape(-1)
    if outputs.shape[0] != spec.n_outputs:
        raise ValueError(
            f"evaluate() returned {outputs.shape[0]} value(s) but the spec declares "
            f"{spec.n_outputs} output(s); return one column per output in "
            "spec.outputs order."
        )
    return outputs, aux


def _outputs_from_dict(outputs: Mapping, spec: ProblemSpec) -> list[float]:
    """Order a ``{output name: value}`` dict into ``spec.outputs`` order."""
    names = spec.output_names
    missing = [n for n in names if n not in outputs]
    extra = [k for k in outputs if k not in names]
    if missing or extra:
        raise ValueError(
            f"outputs dict keys {sorted(outputs)} don't match the spec's outputs "
            f"{names} (missing {missing}, unexpected {extra})."
        )
    return [float(outputs[n]) for n in names]


def _looks_like_tensor_code(exc: Exception) -> bool:
    """Did ``run`` fail treating its design dict as a tensor (``x.reshape``,
    ``x * 2``, ``x[0]``)? Used only to add a hint to the recorded error.
    """
    if isinstance(exc, (AttributeError, TypeError)) and "'dict'" in str(exc):
        return True
    return isinstance(exc, KeyError) and bool(exc.args) and isinstance(exc.args[0], int)


def _far_text(far: dict[str, float], spec: ProblemSpec) -> str:
    """``{name: distance}`` beyond ``max_step``, as text."""
    steps = {d.name: getattr(d, "max_step", None) for d in spec.domain}
    return ", ".join(f"{k} moves {v:.4g} (max_step {steps[k]:.4g})" for k, v in far.items())


def _accepts(policy: Policy, name: str) -> bool:
    """Does ``policy.suggest`` take keyword ``name``? (Policies written before
    ``pending=`` / ``fixed=`` were added don't.)"""
    import inspect

    try:
        params = inspect.signature(policy.suggest).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


def _accepts_pending(policy: Policy) -> bool:
    return _accepts(policy, "pending")


def canonical_Y(history: History, spec: ProblemSpec) -> torch.Tensor:
    """Clean raw outputs (ALL outputs) mapped to maximize-canonical space.

    The surrogate models every output column; OBSERVE columns pass through
    unflipped (output_signs = +1 for them).
    """
    return history.clean.outputs.double() * spec.output_signs()


def canonical_Yvar(history: History, spec: ProblemSpec) -> torch.Tensor | None:
    """Clean per-output noise VARIANCES ``(n_clean, m)`` for ``train_Yvar``, or None.

    Returns ``None`` unless EVERY clean record carries ``output_var`` (per-point
    noise is all-or-nothing for a fit; otherwise the model infers the level).

    Crucially, NO ``output_signs`` flip is applied: ``canonical_Y`` negates MINIMIZE
    columns to reach maximize-canonical space, but a *variance* scales by sign² = 1,
    so it is sign-invariant. Flipping it would be a silent bug (negative variances).
    """
    vars_ = history.clean.output_vars
    return None if vars_ is None else vars_.double()


def _front_indices(Y: torch.Tensor, keep: list[int]) -> list[int]:
    """Front of a canonical objective matrix ``Y`` (rows aligned to ``keep``).

    Scalar objective (m==1) -> the single argmax; multi-objective -> the
    non-dominated set. Returns indices into the original history (via ``keep``).
    """
    if Y.shape[-1] == 1:
        return [keep[int(Y.reshape(-1).argmax())]]
    nd = is_non_dominated(Y)
    return [keep[i] for i in range(len(keep)) if bool(nd[i])]


def _one_per_design(
    history: History, spec: ProblemSpec, keep: list[int], *, observed: bool
) -> list[int]:
    """One record index per distinct design (design columns only), preferring the
    record run at the target. ``observed`` keeps only at-target records, since
    outputs measured at other sources aren't comparable."""
    n = len(spec.design_names)
    target = spec.at_target
    chosen: dict[tuple, int] = {}
    for i in keep:
        x = history.records[i].candidate.double().reshape(-1)
        at_target = torch.equal(x, target(x))
        if observed and not at_target:
            continue
        key = tuple(x[:n].tolist())
        if key not in chosen or at_target:
            chosen[key] = i
    return sorted(chosen.values())


def recommend(
    history: History, spec: ProblemSpec, model=None, *, denoise: bool = True,
    usability: Usability | None = None,
) -> list[int]:
    """Handover index/indices: the recommended design(s) at end-of-run.

    One rule for every policy — drop faulted shots, take the objective columns in
    canonical space, then argmax (scalar) or non-dominated set (multi-objective),
    mapped back to ``history.records`` coordinates. Two input sources:

    - ``denoise=True`` with a fitted ``model`` scores the clean designs by the
      posterior MEAN (avoids chasing a lucky noisy
      sample). Falls back to observed values when no model is given.
    - ``denoise=False`` (or no model) ranks the OBSERVED (noisy) values directly;
      this is also the loop's model-free fallback.

    With sources, every design is judged at the target: the model scores it with
    its source columns set to the target, a design run at several sources counts
    once (under its at-target record if it has one), and the observed fallback
    ranks only records run at the target.

    With output constraints, only designs that satisfy them are ranked: those the
    model gives at least a 50% chance (or, from observed values, that did). If there
    are none, it returns the single design most likely to (the smallest observed
    violation); :func:`feasibility` tells the two cases apart.

    With a ``usability`` model (the campaign has bad-region runs), only designs it
    gives at least a 50% chance of a usable result are ranked, if there are any.

    Returns ``[]`` for a pure-BAX spec (no objectives — handover is the goal's job)
    or when there are no clean records.
    """
    idx = spec.objective_indices()
    if not idx:
        return []
    keep = [i for i, r in enumerate(history.records) if r.usable]
    use_model = denoise and model is not None
    if spec.sources:
        keep = _one_per_design(history, spec, keep, observed=not use_model)
    if usability is not None and keep:
        X_keep = spec.at_target(torch.stack([history.records[i].candidate for i in keep]).double())
        with torch.no_grad():
            ok = usability(X_keep) >= USABLE_P
        if ok.any():
            keep = [i for i, k in zip(keep, ok.tolist()) if k]
    if not keep:
        return []
    if use_model:
        X = spec.at_target(torch.stack([history.records[i].candidate for i in keep]).double())
        with torch.no_grad():
            # Model is full-width & canonical; select objective columns (already
            # canonical, so no sign flip — unlike raw observed values).
            Y = model.posterior(X).mean.reshape(len(keep), -1)[:, idx]
    else:
        raw = torch.stack([history.records[i].outputs.reshape(-1) for i in keep]).double()
        Y = spec.objective_view(raw)
    if spec.has_output_constraints:
        if use_model:
            p = prob_feasible(spec, model, X)
            ok = p >= FEASIBLE_P
            fallback = int(p.argmax())
        else:
            ok = feasible(spec, raw)
            fallback = int(violation(spec, raw, row_scale(history, spec)).argmin())
        if not ok.any():
            return [keep[fallback]]
        Y = Y[ok]
        keep = [i for i, k in zip(keep, ok.tolist()) if k]
    return _front_indices(Y, keep)


def feasibility(
    history: History, spec: ProblemSpec, indices: list[int], model=None
) -> list[float]:
    """For records ``indices``, the chance their design satisfies every output
    constraint: the model's probability at the target, or without a model 1.0 / 0.0
    from the observed outputs. Empty without output constraints."""
    if not (spec.has_output_constraints and indices):
        return []
    if model is not None:
        X = torch.stack([history.records[i].candidate.double().reshape(-1) for i in indices])
        return prob_feasible(spec, model, spec.at_target(X)).tolist()
    Y = torch.stack([history.records[i].outputs.double().reshape(-1) for i in indices])
    return feasible(spec, Y).double().tolist()


def observed_front(history: History, spec: ProblemSpec) -> list[int]:
    """The observed (noisy) front — ``recommend(..., denoise=False)``.

    Kept as a named helper (the model-free fallback) and for backward compatibility.
    """
    return recommend(history, spec, model=None, denoise=False)


# Re-derives the training targets from the record history (see ``apply_targets``):
# RAW outputs ``(n_clean, n_outputs)`` for the clean records in order, or a
# ``(Y, Yvar)`` tuple to also supply per-point noise variances.
Targets = Callable[[History, ProblemSpec], "torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]"]


def apply_targets(targets: Targets, history: History, spec: ProblemSpec) -> History:
    """The history with every clean record's outputs replaced by ``targets``.

    For objectives defined after the fact: re-weighting a scalarization mid-run,
    a recalibrated analysis of stored raw observations, a score that sharpens over
    the campaign (see :class:`abbot.composite.Rescore`). Faulted and bad-region
    records pass through unchanged; the stored history is not modified.
    """
    clean = [r for r in history.records if r.usable]
    out = targets(history, spec)
    Y, Yvar = out if isinstance(out, tuple) else (out, None)
    Y = torch.as_tensor(Y, dtype=torch.double).reshape(len(clean), -1)
    if Y.shape[-1] != spec.n_outputs:
        raise ValueError(
            f"targets returned {Y.shape[-1]} column(s) for a spec with "
            f"{spec.n_outputs} output(s) {spec.output_names}."
        )
    if Yvar is not None:
        Yvar = torch.as_tensor(Yvar, dtype=torch.double).reshape(Y.shape)
    rows = iter(range(len(clean)))
    records = []
    for r in history.records:
        if not r.usable:
            records.append(r)
            continue
        i = next(rows)
        records.append(
            replace(r, outputs=Y[i], output_var=None if Yvar is None else Yvar[i])
        )
    return History(records=records)


def fit_model(
    model_constructor: ModelConstructor, history: History, spec: ProblemSpec
):
    """Fit the surrogate on the clean history in maximize-canonical space.

    Threads per-point noise through when every clean record carries it
    (``canonical_Yvar`` -> ``train_Yvar``); otherwise ``Yvar`` is ``None`` and the
    model infers the noise level, as before.
    """
    X = history.clean.designs.double()
    Y = canonical_Y(history, spec)
    Yvar = canonical_Yvar(history, spec)
    return model_constructor.build(X, Y, Yvar)


# --- serializable snapshot ----------------------------------------------------


@dataclass
class CampaignState:
    """A plain, JSON-round-trippable snapshot of a campaign (no live objects).

    This is the surface the UI renders and an agent reads/writes — the same
    serializable contract the frontend needs.
    """

    n_steps: int
    variable_names: list[str]
    objective_names: list[str]
    designs: list[list[float]]
    outputs: list[list[float] | None]
    faults: list[bool]
    pending: list[dict] | None  # outstanding (in-flight) designs, oldest first
    recommended: list[int]
    halted: str | None
    pending_ids: list[str] = field(default_factory=list)  # ids aligned with pending
    costs: list[float | None] = field(default_factory=list)  # per record, aligned with designs
    cost_spent: float = 0.0  # total recorded cost, failed runs included
    stopped: str | None = None  # a supervisor's stop reason (done, not a fault)
    bad_regions: list[str | None] = field(default_factory=list)  # per record: the reason, or None
    excluded: list[bool] = field(default_factory=list)  # per record: its design was excluded
    halted_on: list[dict] = field(default_factory=list)  # the designs a halt is about

    def to_json(self) -> str:
        import json
        from dataclasses import asdict

        return json.dumps(asdict(self))


@dataclass
class CampaignResult:
    """The outcome of a campaign.

    ``recommended`` holds the handover index/indices into ``history.records``: a
    single index for a scalar objective, the Pareto set for multi-objective. Use
    ``.best`` for the single-objective convenience (it refuses to collapse a true
    multi-point front to one answer).
    """

    history: History
    recommended: list[int] = field(default_factory=list)
    # With sources: per recommended design, whether it was run at the target, and the
    # model's posterior std of each objective there (raw units). Empty without.
    run_at_target: list[bool] = field(default_factory=list)
    target_std: list[list[float]] = field(default_factory=list)
    # With output constraints: per recommended design, the chance it satisfies them
    # (see ``feasibility``) and whether that is at least 50%. Empty without. All
    # False means nothing is likely to satisfy them yet, and the recommendation is
    # the design closest to doing so.
    p_feasible: list[float] = field(default_factory=list)
    feasible: list[bool] = field(default_factory=list)
    # With bad-region runs: per recommended design, the usability model's chance of
    # a usable result there. Empty without.
    p_usable: list[float] = field(default_factory=list)

    @property
    def designs(self) -> torch.Tensor:
        """The recommended design(s), stacked: (k, d)."""
        if not self.recommended:
            return torch.empty(0)
        return torch.stack([self.history.records[i].candidate for i in self.recommended])

    @property
    def best(self) -> Record:
        """The single recommended record. Errors if the front has 0 or >1 points."""
        if len(self.recommended) != 1:
            raise ValueError(
                f"best is only defined for a single recommended design, but "
                f"{len(self.recommended)} were recommended (a multi-objective front). "
                "Use .recommended / .designs instead."
            )
        return self.history.records[self.recommended[0]]


def _timing(item: _InFlight) -> dict:
    """Record fields carried over from an outstanding item, closing it now."""
    return {
        "acquisition": item.acquisition,
        "issued_at": item.issued_at,
        "started_at": item.started_at,
        "finished_at": item.finished_at if item.finished_at is not None else time.time(),
        "fit_seconds": item.fit_seconds,
        "acquisition_seconds": item.acquisition_seconds,
    }


def _acquisition_values(policy: Policy, n: int) -> list[float | None]:
    """The policy's ``last_acquisition_value`` as ``n`` floats, or Nones if it has
    none (it is optional on the protocol) or the shape doesn't match."""
    value = getattr(policy, "last_acquisition_value", None)
    if value is None:
        return [None] * n
    flat = torch.as_tensor(value, dtype=torch.double).reshape(-1)
    if flat.numel() == 1:
        flat = flat.expand(n)
    if flat.numel() != n:
        logger.debug("last_acquisition_value has %d value(s) for %d pick(s); ignored.", flat.numel(), n)
        return [None] * n
    return [float(v) for v in flat]


# --- the session --------------------------------------------------------------


class Candidate(dict):
    """A suggested design: a ``{variable name: value}`` dict that also carries the
    ``id`` to hand back to ``Campaign.ingest`` / ``fail`` / ``cancel``.
    """

    def __init__(self, *args, id: str | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.id = id

    def __repr__(self) -> str:
        return f"Candidate(id={self.id!r}, {dict.__repr__(self)})"


@dataclass
class _InFlight:
    """A design handed out by ``suggest`` and not yet ingested, failed or canceled."""

    id: str
    x: torch.Tensor
    retry_of: int | None = None
    attempt: int = 0
    acquisition: float | None = None  # the policy's value for this pick, if it proposed it
    issued_at: float | None = None  # epoch seconds suggest handed it out
    fit_seconds: float | None = None  # fit time of the model its proposal used
    acquisition_seconds: float | None = None  # time computing its proposal (shared by a batch)
    started_at: float | None = None  # set by step around the experiment run
    finished_at: float | None = None


# A hook is called after each ingest with (state, fitted_model_or_None); a batched
# ``step`` calls it once, after the whole batch.
StepHook = Callable[[CampaignState, Any], None]

# Called by ``step`` with the named candidate before the experiment runs. Return
# None to proceed, or a reason string to block the run (the campaign halts).
PreRunCheck = Callable[[dict], "str | None"]


class Campaign:
    """A resumable, step-based optimization session over the abbot engine.

    ``policy`` and ``model_constructor`` default to what the spec implies
    (``default_policy`` / ``default_constructor``); pass either to override.

    ``initial_designs`` are measured before the policy proposes. By default (``None``)
    a new campaign queues ``default_initial_designs(spec)`` when it first needs a
    design: space-filling, with two at the target when the spec has sources. That is
    skipped when the history is not empty (a resume, or data you ``record`` first) or
    you ``seed`` your own. Pass a tensor to choose the seeds, or an empty one for a
    cold start: random probes until there are enough results to fit a model.

    When a run fails (the experiment raised, or a fault check rejected the shot), it
    is retried ``max_retries`` times, then the campaign halts on that design and a
    person decides with :meth:`resolve`: retry it, exclude it, or mark it a bad
    region. ``0`` halts on the first failure. A halt is saved with the store, so a
    campaign rebuilt on it is still halted until resolved (rerunning
    ``run_campaign`` answers ``"retry"``).

    Runs a fault check marks ``bad_region`` (or that a person resolves as one) are
    kept out of the output models. A separate classifier learns where runs give
    usable results, and proposals and recommendations steer away from the rest.
    """

    def __init__(
        self,
        spec: ProblemSpec,
        policy: Policy | None = None,
        model_constructor: ModelConstructor | None = None,
        *,
        store: DataStore | None = None,
        supervisor: Supervisor | Rule | None = None,
        max_retries: int = 1,
        on_step: StepHook | list[StepHook] | None = None,
        pre_run: PreRunCheck | None = None,
        fingerprint: dict | None = None,
        allow_resume_mismatch: bool = False,
        targets: Targets | None = None,
        initial_designs: torch.Tensor | list[Mapping] | None = None,
    ) -> None:
        if policy is None:
            policy = default_policy(spec)
            logger.info("Using default policy %s.", type(policy).__name__)
        if model_constructor is None:
            model_constructor = default_constructor(spec)
            logger.info("Using default model %s.", type(model_constructor).__name__)
        # Guardrail reads the UQ capability off the real constructor, not a string.
        check_uq_pairing(policy, model_constructor)
        self.spec = spec
        self.policy = policy
        self.model_constructor = model_constructor
        self.store = store
        self.supervisor = supervisor  # bound to the spec by the property
        self.max_retries = max_retries
        self.pre_run = pre_run
        # Optional re-derivation of the training targets from the history; the
        # surrogate, the policy and the hand-over all see ``scored_history()``.
        self.targets = targets
        self._scored: History | None = None
        self._hooks: list[StepHook] = []
        self._hooks_deferred = False
        if on_step is not None:
            self._hooks.extend(on_step if isinstance(on_step, list) else [on_step])

        # Resume from disk if the store already has steps — but only into the same
        # problem: the spec plus any caller ``fingerprint`` must match what the
        # store was written with (``allow_resume_mismatch`` overrides).
        if store is not None:
            fp = {"spec": spec.fingerprint()}
            if fingerprint:
                fp["extra"] = fingerprint
            store.check_fingerprint(fp, allow_mismatch=allow_resume_mismatch)
        self.history = store.load() if store is not None else History()
        if len(self.history):
            logger.info("Resumed %d steps", len(self.history))

        self._seed_queue: list[torch.Tensor] = []
        self._retry_queue: list[tuple[torch.Tensor, int, int]] = []  # (x, retry_of, attempt)
        # Outstanding designs, in issue order (dicts keep insertion order).
        self._inflight: dict[str, _InFlight] = {}
        self._next_id = 0
        self._recommended: list[int] = []
        self.halted: str | None = None
        # The designs a halt is about, as (design, step of its first run), awaiting
        # ``resolve``.
        self._halted_on: list[tuple[torch.Tensor, int]] = []
        self._usability: Usability | None = None
        self._usability_valid = False
        self._warned_usability = False
        # Set when a supervisor says the campaign is done (Verdict.stop): ``drive``
        # ends there; ask/tell callers may read it and carry on regardless.
        self.stopped: str | None = None
        # The policy's acquisition values for the last proposal, aligned with it, and
        # (fit seconds of the model it used, seconds computing it).
        self._proposal_acq: list[float | None] = []
        self._proposal_seconds: tuple[float | None, float | None] = (None, None)
        self._model_fit_seconds: float | None = None
        # The last non-fatal warning surfaced to a caller (e.g. a transient
        # proposal fallback). A UI reads this after suggest/step and shows a toast.
        self.last_warning: str | None = None
        # Cached fit of the current clean history, invalidated on history mutation.
        self._model = None
        self._model_valid = False
        self._queue_restored = self._load_queue()
        # None: queue the default seeds on the first suggest, if nothing else did.
        self._default_seeds_pending = initial_designs is None
        if initial_designs is not None and len(initial_designs):
            self.seed(initial_designs)

    # -- setup --

    @property
    def supervisor(self) -> Supervisor | Rule | None:
        """What watches the campaign (see :class:`~abbot.supervision.Supervisor`).

        Assigning one, here or through ``Campaign(supervisor=...)``, binds it to the
        spec, so rules that read it (``target_reached``, ``no_improvement``) work and
        a wrong output name fails at assignment, not mid-run.
        """
        return self._supervisor

    @supervisor.setter
    def supervisor(self, supervisor: Supervisor | Rule | None) -> None:
        self._supervisor = bind(supervisor, self.spec) if supervisor is not None else None

    def add_hook(self, hook: StepHook) -> None:
        self._hooks.append(hook)

    def seed(self, designs: torch.Tensor | list[Mapping]) -> None:
        """Queue initial designs to measure before the policy proposes: an ``(n, d)``
        tensor, or a list of ``{name: value}`` dicts (labels for labeled
        variables). No-op (with a log) if the campaign resumed from a store — its
        seeds are already in the history or the saved queue.
        """
        if len(self.history) or self._queue_restored:
            logger.info("seed() ignored: campaign resumed with %d steps.", len(self.history))
            return
        if len(designs) and isinstance(designs[0], Mapping):
            designs = [self.spec.as_tensor(d) for d in designs]
        if self.spec.has_travel_limits:
            # The first seed is where the hardware starts; each next one a step on.
            for i in range(1, len(designs)):
                far = self.spec.out_of_reach(designs[i - 1], designs[i])
                if far:
                    raise ValueError(
                        f"seed {i} is out of reach of seed {i - 1}: "
                        f"{_far_text(far, self.spec)}. With travel limits, seeds are a "
                        "walk from the first one (where the hardware starts)."
                    )
        self._seed_queue.extend(list(designs))
        self._save_queue()

    def _queue_default_seeds(self) -> None:
        """On the first ``suggest`` of a new, unseeded campaign, queue the default
        initial designs (see the class docstring)."""
        if not self._default_seeds_pending:
            return
        self._default_seeds_pending = False
        if len(self.history) or self._queue_restored or self._seed_queue or self._inflight:
            return
        if self.spec.has_travel_limits:
            return  # space-filling seeds can't be reached; _propose asks for a start
        designs = default_initial_designs(self.spec)
        self._seed_queue.extend(list(designs))
        self._save_queue()
        at_target = " (2 at the target)" if self.spec.sources else ""
        logger.info(
            "Queued %d default initial designs%s; pass Campaign(initial_designs=...) to "
            "choose them, or an empty tensor for a random cold start.",
            len(designs), at_target,
        )

    def load_history(self, history: History) -> None:
        """Append already-measured records (e.g. imported from CSV) to the history.

        Unlike ``seed`` (which queues designs to be *run*), these are prior results
        folded straight in and persisted, so the next ``suggest`` fits on them. Any
        unmeasured seed queue is cleared — imported data supersedes it.
        """
        self._seed_queue.clear()
        self._stale()  # history mutates -> cached fit is stale
        for record in history.records:
            step = len(self.history)
            self.history.append(record)
            if self.store is not None:
                self.store.save_step(step, record)
        self._save_queue()

    # -- ask/tell --

    def suggest(self, n: int = 1, *, fixed: dict[str, float] | None = None) -> list[Candidate]:
        """Propose ``n`` more candidates. Retries take priority, then queued seeds,
        then a batch proposal from the policy.

        Each returned :class:`Candidate` is a ``{variable name: value}`` dict with
        an ``.id``. It stays outstanding (``outstanding()``, ``state().pending``)
        until ``ingest``, ``fail`` or ``cancel`` closes it; calling ``suggest``
        again adds to the outstanding set rather than replacing it.

        ``fixed`` (``{name: value}``) holds inputs at given values in these
        candidates: today's sample temperature, a setting you can't change. The
        policy chooses the rest. A queued seed takes the fixed values; a queued
        retry is a re-run of one design, so it waits for a ``suggest`` it matches.
        A source may be fixed too, overriding the policy's choice.
        """
        if self.halted:
            raise RuntimeError(f"campaign is halted: {self.halted}")
        pinned = self.spec.resolve_fixed(fixed)  # a bad name or value raises here
        if pinned and not _accepts(self.policy, "fixed"):
            raise ValueError(
                f"{type(self.policy).__name__}.suggest takes no fixed= argument, so it "
                "can't hold inputs fixed."
            )
        self._queue_default_seeds()
        if self.spec.has_travel_limits:
            self._drop_retries_out_of_reach()
        retries = [i for i, (x, _, _) in enumerate(self._retry_queue) if self._matches(x, pinned)]
        n_retry = min(n, len(retries))
        n_seed = min(n - n_retry, len(self._seed_queue))
        rem = n - n_retry - n_seed
        seeds = [self._pin(x, pinned) for x in self._seed_queue[:n_seed]]
        if self.spec.has_travel_limits:
            self._check_walk([self._retry_queue[i][0] for i in retries[:n_retry]] + seeds, pinned)
        # Propose first: a misconfiguration raises before any queue is touched. The
        # policy sees everything in flight, incl. retries/seeds issued alongside.
        xs = []
        if rem > 0:
            in_flight = [it.x for it in self._inflight.values()]
            in_flight += [self._retry_queue[i][0] for i in retries[:n_retry]]
            in_flight += seeds
            issued_now = [self._retry_queue[i][0] for i in retries[:n_retry]] + seeds
            xs = self._propose(rem, in_flight, fixed, start=issued_now[-1] if issued_now else None)
        issued = [self._issue(*self._retry_queue[i]) for i in retries[:n_retry]]
        for i in reversed(retries[:n_retry]):
            del self._retry_queue[i]
        del self._seed_queue[:n_seed]
        issued += [self._issue(x) for x in seeds]
        fit_s, acq_s = self._proposal_seconds
        issued += [
            self._issue(
                xs[i], acquisition=self._proposal_acq[i], fit_seconds=fit_s,
                acquisition_seconds=acq_s,
            )
            for i in range(rem)
        ]
        self._save_queue()
        return [self._candidate(item) for item in issued]

    def ingest(
        self,
        results: object,
        observations: object | None = None,
        output_vars: object | None = None,
        *,
        id: str | None = None,
        ids: list[str] | None = None,
        costs: object | None = None,
    ) -> CampaignState:
        """Fold measured results for outstanding candidates back in.

        Which candidates the results belong to:

        - ``id="c0003"``: ``results`` is that one candidate's result (and
          ``observations`` / ``output_vars`` its single observation / variance).
        - ``ids=[...]``: one result per listed id.
        - neither: one result per outstanding candidate, oldest first (a bare
          value is accepted when exactly one is outstanding).

        A result is the raw outputs (a tensor, a list in ``spec.outputs`` order, or
        a ``{output name: value}`` dict) or an ``(outputs, aux)`` tuple.
        ``observations`` optionally stores the raw observation per record;
        ``output_vars`` optionally supplies per-output noise VARIANCE ``(m,)`` per
        candidate (e.g. from replicate measurements) — stored on the record and
        threaded to the surrogate's ``train_Yvar`` when every clean record has it.
        ``costs`` optionally gives what each run cost (core-hours, money, seconds),
        one per candidate (a single value with ``id=``); see :mod:`abbot.cost`.
        All results are validated before any is recorded. Runs the supervisor,
        persists, and fires hooks.
        """
        if id is not None and ids is not None:
            raise TypeError("pass id= or ids=, not both")
        if id is not None:
            ids, results = [id], [results]
            observations = [observations]
            output_vars = [output_vars]
            costs = [costs]
        else:
            if ids is None:
                if not self._inflight:
                    raise RuntimeError(
                        "ingest() with nothing outstanding; call suggest() first, or "
                        "record(x, y) for a design you chose yourself."
                    )
                ids = list(self._inflight)
            results = self._as_list(results)
            if len(results) != len(ids):
                raise ValueError(
                    f"ingest got {len(results)} result(s) for {len(ids)} outstanding "
                    f"candidate(s) {ids}; pass id= / ids= to ingest a subset."
                )
        n = len(ids)
        obs_list = observations if observations is not None else [None] * n
        var_list = output_vars if output_vars is not None else [None] * n
        cost_list = list(costs) if costs is not None else [None] * n
        if len(cost_list) != n:
            raise ValueError(f"ingest got {len(cost_list)} cost(s) for {n} result(s)")
        items = [self._outstanding_item(i) for i in ids]
        return self._commit(list(zip(items, results, obs_list, var_list, cost_list)))

    def fail(
        self, id: str, error: str | BaseException, *, cost: float | None = None
    ) -> CampaignState:
        """Close an outstanding candidate whose run failed (the instrument errored,
        the sample was lost). Recorded as a faulted record carrying ``error``, then
        retried up to ``max_retries`` times before the campaign halts — the same
        path ``step`` takes when ``experiment.run`` raises. ``cost`` is what the
        failed run cost, if known (it still counts against a budget).
        """
        return self._record_run_error(self._outstanding_item(id), error, cost=cost)

    def cancel(self, id: str) -> CampaignState:
        """Withdraw an outstanding candidate without running it. Nothing is
        recorded and it is not retried.
        """
        item = self._outstanding_item(id)
        del self._inflight[item.id]
        self._save_queue()
        logger.info("canceled %s at %s", item.id, self._to_dict(item.x))
        return self.state()

    def resolve(self, answer: str = "retry", *, reason: str | None = None) -> CampaignState:
        """Answer a halt, so the campaign can carry on.

        A halt about a design (its runs kept failing, see ``max_retries``) takes one
        of three answers:

        - ``"retry"``: the apparatus was the problem and is fixed. The design runs
          next, with its retries counted afresh.
        - ``"exclude"``: the apparatus is fine; don't run this design again. Its
          failed runs stay in the history as faults, marked ``excluded``, and
          proposals steer away from it.
        - ``"bad_region"``: the apparatus is fine and the design itself is the
          cause (a dark region). Its failed runs become bad-region runs, with
          ``reason`` (for people; the model only sees that they were bad), and the
          campaign learns to avoid the region.

        Any other halt (a pre-run check, a stall or drift monitor) only takes
        ``"retry"``: the cause is fixed, carry on. The halt is saved with the store,
        so a campaign rebuilt on it still needs this answer.
        """
        if self.halted is None:
            raise RuntimeError("resolve(): the campaign isn't halted.")
        answers = ("retry", "exclude", "bad_region")
        if answer not in answers:
            raise ValueError(f"resolve(): answer must be one of {answers}, got {answer!r}")
        if answer != "retry" and not self._halted_on:
            raise ValueError(
                f"resolve({answer!r}): this halt isn't about a design ({self.halted}); "
                "fix the cause and resolve('retry') to carry on."
            )
        if reason is not None and answer != "bad_region":
            raise ValueError("resolve(): reason= goes with 'bad_region' only.")
        if answer == "retry":
            for x, first in reversed(self._halted_on):
                self._retry_queue.insert(0, (x, first, 0))
        elif answer == "exclude":
            self._mark_runs([first for _, first in self._halted_on], excluded=True)
        else:
            reason = _bad_region_reason(reason if reason is not None else True)
            self._mark_runs([first for _, first in self._halted_on], bad_region=reason)
        logger.info("Resolved the halt (%s): %s", answer, self.halted)
        self.halted = None
        self._halted_on = []
        self._save_queue()
        return self.state()

    @property
    def halted_on(self) -> list[dict]:
        """The designs the current halt is about (``{name: value}``), awaiting
        :meth:`resolve`; empty when it isn't about a design, or there's no halt."""
        return [self._to_dict(x) for x, _ in self._halted_on]

    def _halt_on(self, x: torch.Tensor, first: int, reason: str) -> None:
        """Halt on a design whose runs kept failing, for a person to resolve."""
        self._halted_on.append((x.double().reshape(-1), first))
        hint = (
            "Decide with resolve(): 'retry' once the apparatus is fixed, 'exclude' to "
            "skip the design, or 'bad_region' if the design itself is the cause."
        )
        parts = [p for p in (self.halted or "").split("\n") if p and p != hint]
        self.halted = "\n".join([*parts, reason, hint])
        logger.warning("%s", reason)

    def _mark_runs(
        self, firsts: list[int], *, excluded: bool = False, bad_region: str | None = None
    ) -> None:
        """Mark the faulted runs of the designs first run at steps ``firsts``:
        excluded, or turned into bad-region runs. Persisted."""
        for step, r in enumerate(self.history.records):
            if r.fault and (step in firsts or r.retry_of in firsts):
                if bad_region is not None:
                    r.fault, r.bad_region = False, bad_region
                r.excluded = r.excluded or excluded
                if self.store is not None:
                    self.store.save_step(step, r)
        self._stale()

    @property
    def _excluded(self) -> list[torch.Tensor]:
        """The designs a person excluded (one each), from the records."""
        out: list[torch.Tensor] = []
        for r in self.history.records:
            x = r.candidate.double().reshape(-1)
            if r.excluded and not any(torch.equal(x, y) for y in out):
                out.append(x)
        return out

    def step(
        self,
        experiment: Experiment | BatchExperiment,
        evaluate: Callable[[object], object] | None = None,
        n: int = 1,
        *,
        fixed: dict[str, float] | None = None,
    ) -> CampaignState:
        """One full cycle: suggest -> experiment.run -> evaluate -> ingest.

        The single primitive both ``run_campaign`` and the UI auto-runner call.
        If candidates are already outstanding (e.g. resumed after a crash
        mid-experiment), the oldest are run before any new one is suggested.

        ``evaluate`` reduces what the experiment returned to outputs (a spectrum to a
        peak position, say); by default the experiment returns the outputs itself.

        ``n`` > 1 runs a synchronous batch of ``n`` designs (a plate, several
        reactors). A :class:`BatchExperiment` gets every step in one
        ``run_batch(designs)`` call, batches of one included; an
        :class:`Experiment` gets ``run`` once per design. ``run_batch`` returns one
        observation per design; an exception object in a slot fails just that
        design, and raising fails the whole batch.

        Before running, ``pre_run`` (if set) sees each named candidate; a returned
        reason halts the campaign with the batch left outstanding and NOT run. A
        failed run is recorded as a fault (with the error text, no outputs) and
        retried up to ``max_retries`` times, then the campaign halts; the batch's
        other results are still recorded. ``on_step`` hooks run once per step,
        after the whole batch is recorded. Errors from ``evaluate`` propagate (they
        are analysis bugs, and re-running the experiment would not fix them).

        ``fixed`` is passed to ``suggest`` for the new candidates; candidates
        already outstanding run as they were issued.
        """
        if self.halted:
            raise RuntimeError(f"campaign is halted: {self.halted}")
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        if not hasattr(experiment, "run_batch") and not hasattr(experiment, "run"):
            raise TypeError(
                f"{type(experiment).__name__} has neither run(design) nor "
                "run_batch(designs)"
            )
        if len(self._inflight) < n:
            self.suggest(n - len(self._inflight), fixed=fixed)
        items = list(self._inflight.values())[:n]
        designs = [self._to_dict(item.x) for item in items]
        if self.pre_run is not None:
            for design in designs:
                reason = self.pre_run(dict(design))
                if reason:
                    self.halted = f"pre-run check blocked {design}: {reason}"
                    logger.warning("%s", self.halted)
                    self._save_queue()
                    return self.state()
        observations, timings = self._run_designs(experiment, designs)
        reported = getattr(experiment, "cost", None)
        if reported is not None:  # the experiment's own account replaces wall-clock
            timings = [
                (started, finished, measured if (c := reported(design)) is None else float(c))
                for design, (started, finished, measured) in zip(designs, timings, strict=True)
            ]
        # Hooks see the batch once it is all recorded, not each partial state.
        self._hooks_deferred = True
        try:
            for item, design, observation, (started, finished, cost) in zip(
                items, designs, observations, timings, strict=True
            ):
                item.started_at, item.finished_at = started, finished
                if isinstance(observation, BaseException):
                    logger.error("experiment.run failed at %s", design, exc_info=observation)
                    self._record_run_error(item, observation, cost=cost)
                    continue
                out = observation if evaluate is None else evaluate(observation)
                self.ingest(out, observations=observation, id=item.id, costs=cost)
        finally:
            self._hooks_deferred = False
            self._after_ingest()
        return self.state()

    @staticmethod
    def _run_designs(
        experiment: Experiment | BatchExperiment, designs: list[dict]
    ) -> tuple[list[object], list[tuple[float, float, float]]]:
        """One observation per design, or the exception its run raised, and per
        design ``(started_at, finished_at, seconds)``: epoch times around the run and
        its measured duration (a batch's duration is split evenly)."""
        run_batch = getattr(experiment, "run_batch", None)
        if run_batch is not None:
            started, start = time.time(), time.perf_counter()
            try:
                observations = list(run_batch(designs))
            except Exception as exc:
                logger.exception("experiment.run_batch failed")
                observations = [exc] * len(designs)
            share = (time.perf_counter() - start) / len(designs)
            finished = time.time()
            if len(observations) != len(designs):
                err = RuntimeError(
                    f"run_batch returned {len(observations)} observation(s) for "
                    f"{len(designs)} design(s)"
                )
                observations = [err] * len(designs)
            return observations, [(started, finished, share)] * len(designs)
        observations: list[object] = []
        timings: list[tuple[float, float, float]] = []
        for design in designs:
            started, start = time.time(), time.perf_counter()
            try:
                observations.append(experiment.run(design))
            except Exception as exc:  # noqa: BLE001 - recorded as a faulted shot
                observations.append(exc)
            timings.append((started, time.time(), time.perf_counter() - start))
        return observations, timings

    def _record_run_error(
        self, item: _InFlight, exc: str | BaseException, *, cost: float | None = None
    ) -> CampaignState:
        """Persist a failed run as a faulted record, then retry or halt."""
        if isinstance(exc, BaseException):
            error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, Exception) and _looks_like_tensor_code(exc):
                error += (
                    " (Experiment.run receives a {name: value} design dict; wrap a "
                    "tensor-based experiment in abbot.TensorExperiment.)"
                )
        else:
            error = str(exc)
        x, retry_of, attempt = item.x, item.retry_of, item.attempt
        record = Record(
            candidate=x, observation=None, fault=True, retry_of=retry_of, error=error,
            id=item.id, cost=cost, **_timing(item),
        )
        step = len(self.history)
        self.history.append(record)
        if self.store is not None:
            self.store.save_step(step, record)
        self._inflight.pop(item.id, None)
        self._stale()
        where = self._to_dict(x)
        first = step if retry_of is None else retry_of
        if attempt < self.max_retries:
            self._retry_queue.append((x, first, attempt + 1))
            self.last_warning = f"Experiment failed at {where} ({error}); retrying."
        else:
            self._halt_on(
                x, first, f"Experiment failed {attempt + 1} time(s) at {where}; last error: {error}"
            )
        self._save_queue()
        self._after_ingest()
        return self.state()

    def record(
        self,
        x: torch.Tensor | Mapping[str, float],
        y: torch.Tensor | Mapping[str, float],
        *,
        observation: object | None = None,
        aux: object | None = None,
        output_var: object | None = None,
        cost: float | None = None,
    ) -> CampaignState:
        """Fold an already-evaluated ``(candidate, outputs)`` pair in directly.

        For external evaluators (an Xopt/gest generator, a live batch the caller
        measured itself) that hand back a design rather than an id. ``x`` is one
        design (a ``(d,)`` tensor or a ``{variable name: value}`` dict), ``y`` the
        RAW outputs (``(m,)`` in ``spec.outputs`` order, or a ``{output name:
        value}`` dict); ``aux`` is the optional cached payload (see :mod:`abbot.composite`);
        ``output_var`` the optional per-output noise variance ``(m,)``; ``cost``
        what the run cost, if known. If ``x``
        exactly equals an outstanding candidate, that candidate is closed (the
        oldest, if several match); otherwise the design is recorded as new.
        Runs the supervisor, persists, queues a retry if flagged, and fires
        ``on_step`` hooks — identical to ``ingest``.

        Distinct from ``load_history`` (which bulk-imports prior data WITHOUT the
        supervisor/hooks). Returns the resulting :class:`CampaignState`.
        """
        if isinstance(x, Mapping):
            x = self.spec.as_tensor(x)
        xt = torch.as_tensor(x, dtype=torch.double).reshape(-1)
        item = next(
            (it for it in self._inflight.values() if torch.equal(it.x, xt)), None
        )
        if item is None:
            item = _InFlight(id=self._new_id(), x=xt)
        result = (y, aux) if aux is not None else y
        return self._commit([(item, result, observation, output_var, cost)])

    def _commit(
        self, entries: list[tuple[_InFlight, object, object, object, object]]
    ) -> CampaignState:
        """Record ``(item, result, observation, output_var, cost)`` entries: validate
        all, then append, supervise, persist and close each, then fire hooks once.
        """
        parsed = [split_eval(res, self.spec) for _, res, _, _, _ in entries]
        for (item, _, obs, var, cost), (y, aux) in zip(entries, parsed):
            ov = None if var is None else torch.as_tensor(var, dtype=torch.double).reshape(-1)
            record = Record(
                candidate=item.x, observation=obs, outputs=y, output_var=ov,
                aux=aux, retry_of=item.retry_of, id=item.id,
                cost=None if cost is None else float(cost),
                **_timing(item),
            )
            step = len(self.history)
            self.history.append(record)

            verdict = self.supervisor(self.history) if self.supervisor else Verdict()
            # A retry outranks a bad region: while any rule suspects the apparatus,
            # "the design is the cause" can't be trusted.
            record.fault = verdict.retry
            if not verdict.retry:
                record.bad_region = _bad_region_reason(getattr(verdict, "bad_region", None))

            if self.store is not None:
                self.store.save_step(step, record)
            self._inflight.pop(item.id, None)

            if verdict.halt is not None:
                self.halted = verdict.halt
            if getattr(verdict, "stop", None) is not None and self.stopped is None:
                self.stopped = verdict.stop
                logger.info("%s", self.stopped)
            if record.fault:
                first = step if item.retry_of is None else item.retry_of
                if item.attempt < self.max_retries:
                    self._retry_queue.append((item.x, first, item.attempt + 1))
                else:
                    self._halt_on(
                        item.x, first,
                        f"A fault check rejected the run at {self._to_dict(item.x)} "
                        f"{item.attempt + 1} time(s).",
                    )

        self._save_queue()
        self._stale()  # history grew -> cached fit is stale
        self._after_ingest()
        return self.state()

    # -- queries --

    def recommend(self) -> list[int]:
        """Current denoised handover index/indices (fits a final model)."""
        if not len(self.history.clean):
            return []
        model = self.current_model()
        return self._safe_recommend(model)

    def result(self) -> CampaignResult:
        """A ``CampaignResult`` with the end-of-run recommendation (with sources,
        also whether each recommended design was run at the target, and the model's
        uncertainty there)."""
        recommended = self.recommend()
        result = CampaignResult(history=self.history, recommended=recommended)
        usability = self.current_usability()
        if usability is not None and recommended:
            X = torch.stack([self.history.records[i].candidate.double().reshape(-1) for i in recommended])
            with torch.no_grad():
                result.p_usable = usability(self.spec.at_target(X)).tolist()
        if self.spec.has_output_constraints and recommended:
            model = self.current_model()
            result.p_feasible = feasibility(self.scored_history(), self.spec, recommended, model)
            result.feasible = [p >= FEASIBLE_P for p in result.p_feasible]
            if not any(result.feasible):
                msg = (
                    "No design is likely to satisfy the output constraints yet "
                    f"(best chance {max(result.p_feasible):.2f}); the recommendation is "
                    "the design closest to satisfying them."
                )
                logger.warning(msg)
                self.last_warning = msg
        if not (self.spec.sources and recommended):
            return result
        X = torch.stack([self.history.records[i].candidate.double().reshape(-1) for i in recommended])
        X_target = self.spec.at_target(X)
        result.run_at_target = [torch.equal(x, xt) for x, xt in zip(X, X_target)]
        model = self.current_model()
        if model is not None:
            with torch.no_grad():
                var = model.posterior(X_target).variance.reshape(len(X), -1)
            result.target_std = var[:, self.spec.objective_indices()].clamp_min(0).sqrt().tolist()
        return result

    def scored_history(self) -> History:
        """The history the surrogate, the policy and the hand-over see: the stored
        one, or with each clean record's outputs re-derived by ``targets``.

        Cached until the history next changes, so ``targets`` must depend on the
        history (and spec) alone; anything else it reads must not change mid-run.
        """
        if self.targets is None:
            return self.history
        if self._scored is None:
            self._scored = apply_targets(self.targets, self.history, self.spec)
        return self._scored

    def check(self) -> CheckReport:
        """Is the model a fair picture of the data? Leave-one-out fit per output,
        runs far from their prediction, how much is read as noise, lengthscales, and
        runs vs. inputs, as plain-language findings. Print the result; see
        :mod:`abbot.diagnostics`. Uses the cached fit, so it costs one fit at most.
        """
        from abbot.diagnostics import check

        return check(self.scored_history(), self.spec, self.current_model())

    def current_model(self) -> Model | None:
        """The surrogate fitted on the current clean history, or ``None``.

        ``None`` when there are too few clean points to fit, or the fit raised.
        The model lives in maximize-canonical space (objective signs applied); a
        caller that wants RAW posterior values must un-apply ``spec.output_signs``
        (see :func:`abbot.analysis.slice_1d`, which does this).

        The fit is cached and reused until the next ``ingest`` / ``record`` /
        ``load_history`` mutates the history, so repeated reads within one UI
        render (model + parity + BAX-progress panels) cost a single fit.
        """
        if self._model_valid:
            return self._model
        start = time.perf_counter()
        self._model = self._fit_now()
        self._model_fit_seconds = time.perf_counter() - start
        self._model_valid = True
        return self._model

    def current_usability(self) -> Usability | None:
        """The classifier of where runs give usable results, fitted on the runs that
        weren't faults (:attr:`History.judged`), or ``None`` without bad-region runs
        (or if the fit failed). Cached like :meth:`current_model`."""
        if not self._usability_valid:
            try:
                self._usability = fit_usability(self.history, self.spec)
            except Exception:
                logger.exception("usability fit failed; not steering away from bad regions.")
                self._usability = None
            self._usability_valid = True
        return self._usability

    def pending_count(self) -> int:
        """Queued units of work owed before the policy must propose again:
        unmeasured seeds plus supervisor-queued retries. A headless driver
        (``run_campaign``, an agent) uses this to size its budget and drain retries.
        Candidates already handed out are counted separately by ``outstanding()``.
        """
        return len(self._seed_queue) + len(self._retry_queue)

    def outstanding(self) -> list[Candidate]:
        """Candidates handed out by ``suggest`` and not yet closed, oldest first."""
        return [self._candidate(item) for item in self._inflight.values()]

    def state(self) -> CampaignState:
        return CampaignState(
            n_steps=len(self.history),
            variable_names=self.spec.variable_names,
            objective_names=[o.name for o in self.spec.objectives],
            designs=[r.candidate.reshape(-1).tolist() for r in self.history.records],
            outputs=[
                r.outputs.reshape(-1).tolist() if r.outputs is not None else None
                for r in self.history.records
            ],
            faults=[r.fault for r in self.history.records],
            pending=[self._to_dict(it.x) for it in self._inflight.values()] or None,
            recommended=list(self._recommended),
            halted=self.halted,
            pending_ids=list(self._inflight),
            costs=[r.cost for r in self.history.records],
            cost_spent=self.history.cost_spent,
            stopped=self.stopped,
            bad_regions=[r.bad_region for r in self.history.records],
            excluded=[r.excluded for r in self.history.records],
            halted_on=self.halted_on,
        )

    # -- internals --

    # Deterministic misconfiguration (wrong policy for the spec, mixed design
    # space, a non-enumerable discrete axis) fires on the first suggest(). These
    # propagate so the caller/UI sees them immediately — never a silent random run.
    _MISCONFIG_ERRORS = (ValueError, NotImplementedError, TypeError)

    def _propose(
        self, n: int, pending: list[torch.Tensor] | None = None, fixed: dict | None = None,
        *, start: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``n`` proposals. With travel limits they walk from ``start`` (a retry or
        seed issued alongside), else from the last design issued."""
        if not self.spec.has_travel_limits:
            return self._propose_from(n, pending, fixed)
        # Travel limits: one design at a time, each searched within a step of the one
        # before, so the hardware can visit them in order.
        position = start.double().reshape(-1) if start is not None else self._position()
        if position is None:
            raise ValueError(
                "With travel limits (max_step), abbot needs to know where the hardware "
                "starts: pass Campaign(initial_designs=[current settings]) or record a "
                "run first."
            )
        picks, acq, seconds = [], [], [0.0, 0.0]
        for _ in range(n):
            reach = self.spec.reach(position)
            x = self._propose_from(1, [*(pending or []), *picks], fixed, within=reach)[0]
            inside = reach._into(x)
            if not torch.allclose(inside, x.double()):
                self.last_warning = (
                    f"{type(self.policy).__name__} proposed a design out of reach "
                    f"({_far_text(self.spec.out_of_reach(position, x), self.spec)}); "
                    "moved it to the nearest one in reach."
                )
                logger.warning("%s", self.last_warning)
            picks.append(inside)
            position = inside
            acq += self._proposal_acq
            seconds = [a + (b or 0.0) for a, b in zip(seconds, self._proposal_seconds)]
        self._proposal_acq = acq
        self._proposal_seconds = tuple(seconds)
        return torch.stack(picks)

    def _propose_from(
        self, n: int, pending: list[torch.Tensor] | None = None, fixed: dict | None = None,
        within: ProblemSpec | None = None,
    ) -> torch.Tensor:
        self._proposal_acq = [None] * n  # random probes carry no acquisition value
        self._proposal_seconds = (None, None)
        # A policy with its own model (or none) proposes without abbot's fit.
        needs_model = getattr(self.policy, "needs_model", True)
        model = self.current_model() if needs_model else None
        if model is None and needs_model:
            # Cold start: too few clean points to fit yet. A uniform probe is the
            # correct behavior here (not an error) — stay quiet.
            logger.debug("no model yet (too few clean points); drawing a random probe.")
            return self._random_draw(n, fixed, within)
        try:
            start = time.perf_counter()
            xs = self._policy_suggest(model, n, pending or [], fixed, within)
            fit_seconds = self._model_fit_seconds if needs_model else None
            self._proposal_seconds = (fit_seconds, time.perf_counter() - start)
            self._proposal_acq = _acquisition_values(self.policy, len(xs))
            # A policy may explain an odd proposal (e.g. BAX with an empty goal).
            warning = getattr(self.policy, "last_warning", None)
            if warning:
                self.last_warning = warning
            return xs
        except self._MISCONFIG_ERRORS:
            # Setup is wrong (e.g. ParetoBO on a 1-objective spec). Fail loud.
            raise
        except Exception as exc:  # transient numerical hiccup: survive it
            msg = (
                f"Proposal hit a transient numerical error ({type(exc).__name__}); "
                "took a random probe this step and will retry next step."
            )
            logger.warning("%s", msg, exc_info=True)
            self.last_warning = msg
            return self._random_draw(n, fixed, within)

    def _policy_suggest(
        self, model, n: int, pending: list[torch.Tensor], fixed: dict | None = None,
        within: ProblemSpec | None = None,
    ) -> torch.Tensor:
        kwargs = {"fixed": fixed} if fixed else {}
        if within is not None and _accepts(self.policy, "within"):
            kwargs["within"] = within
        usability = self.current_usability()
        if usability is not None:
            if _accepts(self.policy, "usability"):
                kwargs["usability"] = usability
            elif not self._warned_usability:
                self._warned_usability = True
                logger.warning(
                    "%s.suggest takes no usability= argument, so its proposals can't "
                    "steer away from bad regions.", type(self.policy).__name__,
                )
        # Excluded designs count as in flight, so the policy steers away from them.
        pending = [*pending, *self._excluded]
        if not pending:
            return self.policy.suggest(self.scored_history(), self.spec, model, n=n, **kwargs)
        X_pending = torch.stack([x.reshape(-1).double() for x in pending])
        if _accepts_pending(self.policy):
            return self.policy.suggest(
                self.scored_history(), self.spec, model, n=n, pending=X_pending, **kwargs
            )
        logger.warning(
            "%s.suggest takes no pending= argument, so it can't see the %d design(s) "
            "in flight and may propose duplicates.",
            type(self.policy).__name__, len(pending),
        )
        return self.policy.suggest(self.scored_history(), self.spec, model, n=n, **kwargs)

    def _random_draw(
        self, n: int, fixed: dict | None = None, within: ProblemSpec | None = None
    ) -> torch.Tensor:
        # Space-filling, and respects enumerated values + domain constraints.
        return (within or self.spec).sample_designs(n, fixed=fixed)

    # -- travel limits --

    def _position(self) -> torch.Tensor | None:
        """Where the hardware is: the design issued (or recorded) most recently."""
        latest, key = None, None
        items = [(r.issued_at if r.issued_at is not None else r.finished_at, 0, r.candidate)
                 for r in self.history.records]
        items += [(it.issued_at, 1, it.x) for it in self._inflight.values()]
        for i, (t, rank, x) in enumerate(items):
            k = (t if t is not None else float("-inf"), rank, i)
            if key is None or k >= key:
                latest, key = x, k
        return None if latest is None else latest.double().reshape(-1)

    def _drop_retries_out_of_reach(self) -> None:
        """A queued retry re-runs an earlier design; once the hardware has moved
        more than a step from it, it can't, so the run stays failed."""
        position = self._position()
        if position is None:
            return
        keep = []
        for entry in self._retry_queue:
            far = self.spec.out_of_reach(position, entry[0])
            if far:
                self.last_warning = (
                    f"dropped a retry: its design is out of reach ({_far_text(far, self.spec)})."
                )
                logger.warning("%s", self.last_warning)
            else:
                keep.append(entry)
        self._retry_queue = keep

    def _check_walk(self, designs: list[torch.Tensor], pinned: dict[int, float]) -> None:
        """Retries and seeds about to be issued, and ``fixed`` values, must each be a
        step from the one before."""
        position = self._position()
        for x in designs:
            if position is not None:
                far = self.spec.out_of_reach(position, x)
                if far:
                    raise ValueError(
                        f"the next queued design is out of reach of the last one: "
                        f"{_far_text(far, self.spec)}."
                    )
            position = x
        if position is not None and pinned:
            probe = position.clone().double().reshape(-1)
            for j, v in pinned.items():
                probe[j] = v
            far = self.spec.out_of_reach(position, probe)
            if far:
                raise ValueError(
                    f"fixed= puts the design out of reach of the last one: "
                    f"{_far_text(far, self.spec)}."
                )

    def _matches(self, x: torch.Tensor, pinned: dict[int, float]) -> bool:
        x = x.double().reshape(-1)
        return all(bool(torch.isclose(x[j], torch.tensor(v, dtype=x.dtype))) for j, v in pinned.items())

    def _pin(self, x: torch.Tensor, pinned: dict[int, float]) -> torch.Tensor:
        """A queued seed with the fixed inputs set, or a fresh draw with them if that
        would break a domain constraint."""
        if not pinned:
            return x
        x = x.clone()
        for j, v in pinned.items():
            x[..., j] = v
        if self.spec._feasible(x.double().reshape(1, -1))[0]:
            return x
        return self.spec._sample(1, None, pinned)[0].to(x)

    def _stale(self) -> None:
        """The history changed: drop the cached fits."""
        self._model_valid = False
        self._usability_valid = False
        self._scored = None

    def _fit_now(self):
        """Fit the surrogate on the current clean history (uncached), or None."""
        if not len(self.history.clean):
            return None
        # Outside the try: a broken ``targets`` is a setup error, not a bad fit.
        history = self.scored_history()
        try:
            return fit_model(self.model_constructor, history, self.spec)
        except Exception:
            logger.exception("model fit failed.")
            return None

    def _safe_fit(self):
        """Deprecated alias for :meth:`current_model` (kept for one release)."""
        return self.current_model()

    def _safe_recommend(self, model) -> list[int]:
        """Denoised handover via the shared :func:`recommend` rule, with the
        observed front as a fallback if the model-based pass yields nothing or errors.
        """
        try:
            rec = recommend(
                self.scored_history(), self.spec, model, denoise=True,
                usability=self.current_usability(),
            )
        except Exception:
            logger.exception("recommend failed; using the observed front.")
            rec = []
        if not rec:
            rec = observed_front(self.scored_history(), self.spec)
        return rec

    def _after_ingest(self) -> None:
        # Only pay for a per-step fit + recommend + hooks when someone is listening
        # (the UI registers hooks; run_campaign does not and recommends once at end).
        # Inside a batch, ``step`` defers this until the whole batch is recorded.
        if self._hooks_deferred:
            return
        if not self._hooks:
            self._recommended = []
            return
        model = self.current_model()
        self._recommended = self._safe_recommend(model)
        state = self.state()
        for hook in self._hooks:
            try:
                hook(state, model)
            except Exception:
                logger.exception("on_step hook failed.")

    def _new_id(self) -> str:
        cid = f"c{self._next_id:04d}"
        self._next_id += 1
        return cid

    def _issue(
        self,
        x: torch.Tensor,
        retry_of: int | None = None,
        attempt: int = 0,
        *,
        acquisition: float | None = None,
        fit_seconds: float | None = None,
        acquisition_seconds: float | None = None,
    ) -> _InFlight:
        item = _InFlight(
            id=self._new_id(), x=x.reshape(-1).double(), retry_of=retry_of,
            attempt=attempt, acquisition=acquisition, issued_at=time.time(),
            fit_seconds=fit_seconds, acquisition_seconds=acquisition_seconds,
        )
        self._inflight[item.id] = item
        return item

    def _outstanding_item(self, id: str) -> _InFlight:
        try:
            return self._inflight[id]
        except KeyError:
            raise KeyError(
                f"{id!r} is not outstanding (outstanding: {list(self._inflight)}); "
                "it may already have been ingested, failed or canceled."
            ) from None

    def _candidate(self, item: _InFlight) -> Candidate:
        return Candidate(self._to_dict(item.x), id=item.id)

    def _save_queue(self) -> None:
        if self.store is None:
            return
        self.store.save_queue({
            "version": 1,
            "next_id": self._next_id,
            "in_flight": [
                {"id": it.id, "design": it.x.tolist(), "retry_of": it.retry_of,
                 "attempt": it.attempt, "acquisition": it.acquisition,
                 "issued_at": it.issued_at, "fit_seconds": it.fit_seconds,
                 "acquisition_seconds": it.acquisition_seconds}
                for it in self._inflight.values()
            ],
            "retries": [
                {"design": x.tolist(), "retry_of": r, "attempt": a}
                for x, r, a in self._retry_queue
            ],
            "seeds": [x.reshape(-1).tolist() for x in self._seed_queue],
            # A rebuilt campaign is still halted, until resolved.
            "halted": self.halted,
            "halted_on": [{"design": x.tolist(), "retry_of": first} for x, first in self._halted_on],
        })

    def _load_queue(self) -> bool:
        """Restore outstanding work from the store's queue.json. Returns whether
        one was found.
        """
        q = self.store.load_queue() if self.store is not None else None
        # Ids already in the history: if a crash landed between writing a record
        # and the queue, that item is done, not outstanding.
        done = {r.id for r in self.history.records if getattr(r, "id", None)}
        # Pre-queue stores: continue numbering after any ids the history carries.
        self._next_id = 1 + max(
            (int(i[1:]) for i in done if i[:1] == "c" and i[1:].isdigit()), default=-1
        )
        if q is None:
            return False

        def t(v: list) -> torch.Tensor:
            return torch.tensor(v, dtype=torch.double)

        self._next_id = max(self._next_id, int(q["next_id"]))
        for it in q["in_flight"]:
            if it["id"] not in done:
                self._inflight[it["id"]] = _InFlight(
                    id=it["id"], x=t(it["design"]), retry_of=it["retry_of"],
                    attempt=it["attempt"], acquisition=it.get("acquisition"),
                    issued_at=it.get("issued_at"), fit_seconds=it.get("fit_seconds"),
                    acquisition_seconds=it.get("acquisition_seconds"),
                )
        self._retry_queue = [(t(r["design"]), r["retry_of"], r["attempt"]) for r in q["retries"]]
        self._seed_queue = [t(x) for x in q["seeds"]]
        self.halted = q.get("halted")
        self._halted_on = [(t(h["design"]), h["retry_of"]) for h in q.get("halted_on", [])]
        if self.halted:
            logger.warning("Resumed a halted campaign; resolve() to carry on: %s", self.halted)
        if self._inflight or self._retry_queue or self._seed_queue:
            logger.info(
                "Restored %d outstanding, %d retries, %d seeds from the store.",
                len(self._inflight), len(self._retry_queue), len(self._seed_queue),
            )
        return True

    def _to_dict(self, x: torch.Tensor) -> dict:
        return self.spec.as_dict(x)

    @staticmethod
    def _as_list(results: object) -> list:
        if isinstance(results, list):
            return results
        return [results]
