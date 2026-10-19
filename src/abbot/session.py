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
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import torch
from botorch.utils.multi_objective.pareto import is_non_dominated

from abbot.experiment import Experiment, Supervisor, Verdict
from abbot.history import DataStore, History, Record
from abbot.model import ModelConstructor, default_constructor
from abbot.policy import Policy, default_policy
from abbot.problem import ProblemSpec

if TYPE_CHECKING:
    from botorch.models.model import Model

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
    """Space-filling seed designs for a fresh campaign: ``2d + 1`` Sobol points
    (at least ``2m`` for ``m`` objectives, ParetoBO's cold-start floor).
    """
    n = max(2 * len(spec.domain) + 1, 2 * spec.n_objectives)
    return spec.sample_designs(n, seed=seed)


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


def _accepts_pending(policy: Policy) -> bool:
    """Does ``policy.suggest`` take ``pending=``? (Policies written before it was
    added to the protocol don't.)"""
    import inspect

    try:
        params = inspect.signature(policy.suggest).parameters
    except (TypeError, ValueError):
        return False
    return "pending" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


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


def recommend(
    history: History, spec: ProblemSpec, model=None, *, denoise: bool = True
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

    Returns ``[]`` for a pure-BAX spec (no objectives — handover is the goal's job)
    or when there are no clean records.
    """
    idx = spec.objective_indices()
    if not idx:
        return []
    keep = [i for i, r in enumerate(history.records) if not r.fault]
    if not keep:
        return []
    if denoise and model is not None:
        X = torch.stack([history.records[i].candidate for i in keep]).double()
        with torch.no_grad():
            # Model is full-width & canonical; select objective columns (already
            # canonical, so no sign flip — unlike raw observed values).
            Y = model.posterior(X).mean.reshape(len(keep), -1)[:, idx]
    else:
        Y = spec.objective_view(
            torch.stack([history.records[i].outputs.reshape(-1) for i in keep])
        )
    return _front_indices(Y, keep)


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
    the campaign (see :class:`abbot.composite.Rescore`). Faulted records pass through
    unchanged; the stored history is not modified.
    """
    clean = [r for r in history.records if not r.fault]
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
        if r.fault:
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
    """A design handed out by ``suggest`` and not yet ingested, failed or cancelled."""

    id: str
    x: torch.Tensor
    retry_of: int | None = None
    attempt: int = 0


# A hook is called after each ingest with (state, fitted_model_or_None).
StepHook = Callable[[CampaignState, Any], None]

# Called by ``step`` with the named candidate before the experiment runs. Return
# None to proceed, or a reason string to block the run (the campaign halts).
PreRunCheck = Callable[[dict], "str | None"]


class Campaign:
    """A resumable, step-based optimization session over the abbot engine.

    ``policy`` and ``model_constructor`` default to what the spec implies
    (``default_policy`` / ``default_constructor``); pass either to override.
    """

    def __init__(
        self,
        spec: ProblemSpec,
        policy: Policy | None = None,
        model_constructor: ModelConstructor | None = None,
        *,
        store: DataStore | None = None,
        supervisor: Supervisor | None = None,
        max_retries: int = 1,
        on_step: StepHook | list[StepHook] | None = None,
        pre_run: PreRunCheck | None = None,
        fingerprint: dict | None = None,
        allow_resume_mismatch: bool = False,
        targets: Targets | None = None,
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
        self.supervisor = supervisor
        self.max_retries = max_retries
        self.pre_run = pre_run
        # Optional re-derivation of the training targets from the history; the
        # surrogate, the policy and the hand-over all see ``scored_history()``.
        self.targets = targets
        self._scored: History | None = None
        self._hooks: list[StepHook] = []
        if on_step is not None:
            self._hooks.extend(on_step if isinstance(on_step, list) else [on_step])

        # Resume from disk if the store already has steps — but only into the same
        # problem: the spec plus any caller ``fingerprint`` must match what the
        # store was written with (``allow_resume_mismatch`` overrides).
        if store is not None:
            fp = {"spec": spec.model_dump(mode="json")}
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
        # The last non-fatal warning surfaced to a caller (e.g. a transient
        # proposal fallback). A UI reads this after suggest/step and shows a toast.
        self.last_warning: str | None = None
        # Cached fit of the current clean history, invalidated on history mutation.
        self._model = None
        self._model_valid = False
        self._queue_restored = self._load_queue()

    # -- setup --

    def add_hook(self, hook: StepHook) -> None:
        self._hooks.append(hook)

    def seed(self, designs: torch.Tensor) -> None:
        """Queue initial designs to measure before the policy proposes. No-op (with
        a log) if the campaign resumed from a store — its seeds are already in the
        history or the saved queue.
        """
        if len(self.history) or self._queue_restored:
            logger.info("seed() ignored: campaign resumed with %d steps.", len(self.history))
            return
        self._seed_queue.extend(list(designs))
        self._save_queue()

    def load_history(self, history: History) -> None:
        """Append already-measured records (e.g. imported from CSV) to the history.

        Unlike ``seed`` (which queues designs to be *run*), these are prior results
        folded straight in and persisted, so the next ``suggest`` fits on them. Any
        unmeasured seed queue is cleared — imported data supersedes it.
        """
        self._seed_queue.clear()
        self._model_valid = False  # history mutates -> cached fit is stale
        self._scored = None
        for record in history.records:
            step = len(self.history)
            self.history.append(record)
            if self.store is not None:
                self.store.save_step(step, record)
        self._save_queue()

    # -- ask/tell --

    def suggest(self, n: int = 1) -> list[Candidate]:
        """Propose ``n`` more candidates. Retries take priority, then queued seeds,
        then a batch proposal from the policy.

        Each returned :class:`Candidate` is a ``{variable name: value}`` dict with
        an ``.id``. It stays outstanding (``outstanding()``, ``state().pending``)
        until ``ingest``, ``fail`` or ``cancel`` closes it; calling ``suggest``
        again adds to the outstanding set rather than replacing it.
        """
        if self.halted:
            raise RuntimeError(f"campaign is halted: {self.halted}")
        n_retry = min(n, len(self._retry_queue))
        n_seed = min(n - n_retry, len(self._seed_queue))
        rem = n - n_retry - n_seed
        # Propose first: a misconfiguration raises before any queue is touched. The
        # policy sees everything in flight, incl. retries/seeds issued alongside.
        xs = []
        if rem > 0:
            in_flight = [it.x for it in self._inflight.values()]
            in_flight += [x for x, _, _ in self._retry_queue[:n_retry]]
            in_flight += self._seed_queue[:n_seed]
            xs = self._propose(rem, in_flight)
        issued = [self._issue(*self._retry_queue.pop(0)) for _ in range(n_retry)]
        issued += [self._issue(self._seed_queue.pop(0)) for _ in range(n_seed)]
        issued += [self._issue(xs[i]) for i in range(rem)]
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
        All results are validated before any is recorded. Runs the supervisor,
        persists, and fires hooks.
        """
        if id is not None and ids is not None:
            raise TypeError("pass id= or ids=, not both")
        if id is not None:
            ids, results = [id], [results]
            observations = [observations]
            output_vars = [output_vars]
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
        items = [self._outstanding_item(i) for i in ids]
        return self._commit(list(zip(items, results, obs_list, var_list)))

    def fail(self, id: str, error: str | BaseException) -> CampaignState:
        """Close an outstanding candidate whose run failed (the instrument errored,
        the sample was lost). Recorded as a faulted record carrying ``error``, then
        retried up to ``max_retries`` times before the campaign halts — the same
        path ``step`` takes when ``experiment.run`` raises.
        """
        return self._record_run_error(self._outstanding_item(id), error)

    def cancel(self, id: str) -> CampaignState:
        """Withdraw an outstanding candidate without running it. Nothing is
        recorded and it is not retried.
        """
        item = self._outstanding_item(id)
        del self._inflight[item.id]
        self._save_queue()
        logger.info("cancelled %s at %s", item.id, self._to_dict(item.x))
        return self.state()

    def step(
        self, experiment: Experiment, evaluate: Callable[[object], object]
    ) -> CampaignState:
        """One full cycle: suggest -> experiment.run -> evaluate -> ingest.

        The single primitive both ``run_campaign`` and the UI auto-runner call.
        If candidates are already outstanding (e.g. resumed after a crash
        mid-experiment), the oldest is run instead of suggesting a new one.

        Before running, ``pre_run`` (if set) sees the named candidate; a returned
        reason halts the campaign with the candidate left outstanding and NOT run. If
        ``experiment.run`` raises, the shot is recorded as a fault (with the error
        text, no outputs) and retried up to ``max_retries`` times, then the
        campaign halts. Errors from ``evaluate`` propagate (they are analysis bugs,
        and re-running the experiment would not fix them).
        """
        if self.halted:
            raise RuntimeError(f"campaign is halted: {self.halted}")
        if not self._inflight:
            self.suggest(1)
        item = next(iter(self._inflight.values()))
        design = self._to_dict(item.x)
        if self.pre_run is not None:
            reason = self.pre_run(dict(design))
            if reason:
                self.halted = f"pre-run check blocked {design}: {reason}"
                logger.warning("%s", self.halted)
                return self.state()
        try:
            observation = experiment.run(design)
        except Exception as exc:
            logger.exception("experiment.run failed at %s", design)
            return self._record_run_error(item, exc)
        out = evaluate(observation)
        return self.ingest(out, observations=observation, id=item.id)

    def _record_run_error(
        self, item: _InFlight, exc: str | BaseException
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
            id=item.id,
        )
        step = len(self.history)
        self.history.append(record)
        if self.store is not None:
            self.store.save_step(step, record)
        self._inflight.pop(item.id, None)
        self._model_valid = False
        self._scored = None
        where = self._to_dict(x)
        if attempt < self.max_retries:
            self._retry_queue.append(
                (x, step if retry_of is None else retry_of, attempt + 1)
            )
            self.last_warning = f"Experiment failed at {where} ({error}); retrying."
        else:
            self.halted = (
                f"Experiment failed {attempt + 1} time(s) at {where}; last error: {error}"
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
    ) -> CampaignState:
        """Fold an already-evaluated ``(candidate, outputs)`` pair in directly.

        For external evaluators (an Xopt/gest generator, a live batch the caller
        measured itself) that hand back a design rather than an id. ``x`` is one
        design (a ``(d,)`` tensor or a ``{variable name: value}`` dict), ``y`` the
        RAW outputs (``(m,)`` in ``spec.outputs`` order, or a ``{output name:
        value}`` dict); ``aux`` is the optional cached payload (see :mod:`abbot.composite`);
        ``output_var`` the optional per-output noise variance ``(m,)``. If ``x``
        exactly equals an outstanding candidate, that candidate is closed (the
        oldest, if several match); otherwise the design is recorded as new.
        Runs the supervisor, persists, queues a retry if flagged, and fires
        ``on_step`` hooks — identical to ``ingest``.

        Distinct from ``load_history`` (which bulk-imports prior data WITHOUT the
        supervisor/hooks). Returns the resulting :class:`CampaignState`.
        """
        if isinstance(x, Mapping):
            x = [float(x[n]) for n in self.spec.variable_names]
        xt = torch.as_tensor(x, dtype=torch.double).reshape(-1)
        item = next(
            (it for it in self._inflight.values() if torch.equal(it.x, xt)), None
        )
        if item is None:
            item = _InFlight(id=self._new_id(), x=xt)
        result = (y, aux) if aux is not None else y
        return self._commit([(item, result, observation, output_var)])

    def _commit(self, entries: list[tuple[_InFlight, object, object, object]]) -> CampaignState:
        """Record ``(item, result, observation, output_var)`` entries: validate all,
        then append, supervise, persist and close each, then fire hooks once.
        """
        parsed = [split_eval(res, self.spec) for _, res, _, _ in entries]
        for (item, _, obs, var), (y, aux) in zip(entries, parsed):
            ov = None if var is None else torch.as_tensor(var, dtype=torch.double).reshape(-1)
            record = Record(
                candidate=item.x, observation=obs, outputs=y, output_var=ov,
                aux=aux, retry_of=item.retry_of, id=item.id,
            )
            step = len(self.history)
            self.history.append(record)

            verdict = self.supervisor(self.history) if self.supervisor else Verdict()
            record.fault = verdict.retry

            if self.store is not None:
                self.store.save_step(step, record)
            self._inflight.pop(item.id, None)

            if verdict.halt is not None:
                self.halted = verdict.halt
            if verdict.retry and item.attempt < self.max_retries:
                self._retry_queue.append(
                    (item.x, step if item.retry_of is None else item.retry_of, item.attempt + 1)
                )

        self._save_queue()
        self._model_valid = False  # history grew -> cached fit is stale
        self._scored = None
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
        """A ``CampaignResult`` with the end-of-run recommendation."""
        return CampaignResult(history=self.history, recommended=self.recommend())

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
        self._model = self._fit_now()
        self._model_valid = True
        return self._model

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
        )

    # -- internals --

    # Deterministic misconfiguration (wrong policy for the spec, mixed design
    # space, a non-enumerable discrete axis) fires on the first suggest(). These
    # propagate so the caller/UI sees them immediately — never a silent random run.
    _MISCONFIG_ERRORS = (ValueError, NotImplementedError, TypeError)

    def _propose(self, n: int, pending: list[torch.Tensor] | None = None) -> torch.Tensor:
        model = self.current_model()
        if model is None:
            # Cold start: too few clean points to fit yet. A uniform probe is the
            # correct behavior here (not an error) — stay quiet.
            logger.debug("no model yet (too few clean points); drawing a random probe.")
            return self._random_draw(n)
        try:
            return self._policy_suggest(model, n, pending or [])
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
            return self._random_draw(n)

    def _policy_suggest(self, model, n: int, pending: list[torch.Tensor]) -> torch.Tensor:
        if not pending:
            return self.policy.suggest(self.scored_history(), self.spec, model, n=n)
        X_pending = torch.stack([x.reshape(-1).double() for x in pending])
        if _accepts_pending(self.policy):
            return self.policy.suggest(
                self.scored_history(), self.spec, model, n=n, pending=X_pending
            )
        logger.warning(
            "%s.suggest takes no pending= argument, so it can't see the %d design(s) "
            "in flight and may propose duplicates.",
            type(self.policy).__name__, len(pending),
        )
        return self.policy.suggest(self.scored_history(), self.spec, model, n=n)

    def _random_draw(self, n: int) -> torch.Tensor:
        # Space-filling, and respects enumerated values + inequality constraints.
        return self.spec.sample_designs(n)

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
            rec = recommend(self.scored_history(), self.spec, model, denoise=True)
        except Exception:
            logger.exception("recommend failed; using the observed front.")
            rec = []
        if not rec:
            rec = observed_front(self.scored_history(), self.spec)
        return rec

    def _after_ingest(self) -> None:
        # Only pay for a per-step fit + recommend + hooks when someone is listening
        # (the UI registers hooks; run_campaign does not and recommends once at end).
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
        self, x: torch.Tensor, retry_of: int | None = None, attempt: int = 0
    ) -> _InFlight:
        item = _InFlight(id=self._new_id(), x=x.reshape(-1).double(),
                         retry_of=retry_of, attempt=attempt)
        self._inflight[item.id] = item
        return item

    def _outstanding_item(self, id: str) -> _InFlight:
        try:
            return self._inflight[id]
        except KeyError:
            raise KeyError(
                f"{id!r} is not outstanding (outstanding: {list(self._inflight)}); "
                "it may already have been ingested, failed or cancelled."
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
                 "attempt": it.attempt}
                for it in self._inflight.values()
            ],
            "retries": [
                {"design": x.tolist(), "retry_of": r, "attempt": a}
                for x, r, a in self._retry_queue
            ],
            "seeds": [x.reshape(-1).tolist() for x in self._seed_queue],
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
                    attempt=it["attempt"],
                )
        self._retry_queue = [(t(r["design"]), r["retry_of"], r["attempt"]) for r in q["retries"]]
        self._seed_queue = [t(x) for x in q["seeds"]]
        if self._inflight or self._retry_queue or self._seed_queue:
            logger.info(
                "Restored %d outstanding, %d retries, %d seeds from the store.",
                len(self._inflight), len(self._retry_queue), len(self._seed_queue),
            )
        return True

    def _to_dict(self, x: torch.Tensor) -> dict:
        return dict(zip(self.spec.variable_names, x.reshape(-1).tolist()))

    @staticmethod
    def _as_list(results: object) -> list:
        if isinstance(results, list):
            return results
        return [results]
