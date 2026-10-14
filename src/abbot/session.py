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
from collections.abc import Callable
from dataclasses import dataclass, field
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

    Accepts a bare tensor (aux=None) or an ``(outputs, aux)`` tuple. Validates the
    number of OUTPUT columns against the spec so a wrong-width return fails here
    with a clear message instead of crashing downstream.
    """
    if isinstance(result, tuple):
        outputs, aux = result
    else:
        outputs, aux = result, None
    outputs = torch.as_tensor(outputs).reshape(-1).double()
    if outputs.shape[0] != spec.n_outputs:
        raise ValueError(
            f"evaluate() returned {outputs.shape[0]} value(s) but the spec declares "
            f"{spec.n_outputs} output(s); return one column per output in "
            "spec.outputs order."
        )
    return outputs, aux


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
      posterior MEAN (the sbido denoising idea — avoids chasing a lucky noisy
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
    pending: list[dict] | None
    recommended: list[int]
    halted: str | None

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

# A hook is called after each ingest with (state, fitted_model_or_None).
StepHook = Callable[[CampaignState, Any], None]


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
        self._hooks: list[StepHook] = []
        if on_step is not None:
            self._hooks.extend(on_step if isinstance(on_step, list) else [on_step])

        # Resume from disk if the store already has steps.
        self.history = store.load() if store is not None else History()
        if len(self.history):
            logger.info("Resumed %d steps", len(self.history))

        self._seed_queue: list[torch.Tensor] = []
        self._retry_queue: list[tuple[torch.Tensor, int, int]] = []  # (x, retry_of, attempt)
        self._pending: list[tuple[torch.Tensor, int | None, int]] | None = None
        self._recommended: list[int] = []
        self.halted: str | None = None
        # The last non-fatal warning surfaced to a caller (e.g. a transient
        # proposal fallback). A UI reads this after suggest/step and shows a toast.
        self.last_warning: str | None = None
        # Cached fit of the current clean history, invalidated on history mutation.
        self._model = None
        self._model_valid = False

    # -- setup --

    def add_hook(self, hook: StepHook) -> None:
        self._hooks.append(hook)

    def seed(self, designs: torch.Tensor) -> None:
        """Queue initial designs to measure before the policy proposes. No-op (with
        a log) if the campaign resumed with existing history — seeds are already in.
        """
        if len(self.history):
            logger.info("seed() ignored: campaign resumed with %d steps.", len(self.history))
            return
        self._seed_queue.extend(list(designs))

    def load_history(self, history: History) -> None:
        """Append already-measured records (e.g. imported from CSV) to the history.

        Unlike ``seed`` (which queues designs to be *run*), these are prior results
        folded straight in and persisted, so the next ``suggest`` fits on them. Any
        unmeasured seed queue is cleared — imported data supersedes it.
        """
        self._seed_queue.clear()
        self._model_valid = False  # history mutates -> cached fit is stale
        for record in history.records:
            step = len(self.history)
            self.history.append(record)
            if self.store is not None:
                self.store.save_step(step, record)

    # -- ask/tell --

    def suggest(self, n: int = 1) -> list[dict]:
        """Propose ``n`` candidates. Retries take priority, then queued seeds, then
        a batch proposal from the policy. Returns gest-shaped dicts keyed by
        variable name; the pending candidates are held until ``ingest``.
        """
        if self.halted:
            raise RuntimeError(f"campaign is halted: {self.halted}")
        pend: list[tuple[torch.Tensor, int | None, int]] = []
        while len(pend) < n and self._retry_queue:
            pend.append(self._retry_queue.pop(0))
        while len(pend) < n and self._seed_queue:
            pend.append((self._seed_queue.pop(0), None, 0))
        rem = n - len(pend)
        if rem > 0:
            xs = self._propose(rem)
            for i in range(rem):
                pend.append((xs[i], None, 0))
        self._pending = pend
        return [self._to_dict(x) for x, _, _ in pend]

    def ingest(
        self,
        results: object,
        observations: list | None = None,
        output_vars: list | None = None,
    ) -> CampaignState:
        """Fold measured results for the pending suggestion back in.

        ``results`` is one eval-output per pending candidate (a tensor or an
        ``(outputs, aux)`` tuple each); a bare value is accepted when a single
        candidate is pending. ``observations`` optionally stores the raw
        observation per record; ``output_vars`` optionally supplies per-output noise
        VARIANCE ``(m,)`` per candidate (e.g. from replicate measurements) — stored
        on the record and threaded to the surrogate's ``train_Yvar`` when every clean
        record has it. Runs the supervisor, persists, and fires hooks.
        """
        if self._pending is None:
            raise RuntimeError("ingest() called before suggest()")
        results = self._as_list(results)
        if len(results) != len(self._pending):
            raise ValueError(
                f"ingest got {len(results)} result(s) for {len(self._pending)} pending "
                "candidate(s)."
            )
        obs_list = observations if observations is not None else [None] * len(results)
        var_list = output_vars if output_vars is not None else [None] * len(results)

        for (x, retry_of, attempt), res, obs, var in zip(
            self._pending, results, obs_list, var_list
        ):
            y, aux = split_eval(res, self.spec)
            ov = None if var is None else torch.as_tensor(var, dtype=torch.double).reshape(-1)
            record = Record(
                candidate=x, observation=obs, outputs=y, output_var=ov,
                aux=aux, retry_of=retry_of,
            )
            step = len(self.history)
            self.history.append(record)

            verdict = self.supervisor(self.history) if self.supervisor else Verdict()
            record.fault = verdict.retry

            if self.store is not None:
                self.store.save_step(step, record)

            if verdict.halt is not None:
                self.halted = verdict.halt
            if verdict.retry and attempt < self.max_retries:
                self._retry_queue.append(
                    (x, step if retry_of is None else retry_of, attempt + 1)
                )

        self._pending = None
        self._model_valid = False  # history grew -> cached fit is stale
        self._after_ingest()
        return self.state()

    def step(
        self, experiment: Experiment, evaluate: Callable[[object], object]
    ) -> CampaignState:
        """One full cycle: suggest -> experiment.run -> evaluate -> ingest.

        The single primitive both ``run_campaign`` and the UI auto-runner call.
        """
        self.suggest(1)
        x = self._pending[0][0]
        observation = experiment.run(x)
        out = evaluate(observation)
        return self.ingest([out], observations=[observation])

    def record(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        observation: object | None = None,
        aux: object | None = None,
        output_var: object | None = None,
    ) -> CampaignState:
        """Fold an already-evaluated ``(candidate, outputs)`` pair in directly.

        For external evaluators (an Xopt/gest generator, a live batch the caller
        measured itself) that chose the point rather than taking it from
        ``suggest``. ``x`` is one design ``(d,)``, ``y`` the RAW outputs ``(m,)`` in
        ``spec.outputs`` order; ``aux`` is the optional cached payload (SBI theta
        samples); ``output_var`` the optional per-output noise variance ``(m,)``.
        Runs the supervisor, persists, queues a retry if flagged, and fires
        ``on_step`` hooks — identical to ``ingest``, only the candidate is supplied
        by the caller instead of being pending.

        Distinct from ``load_history`` (which bulk-imports prior data WITHOUT the
        supervisor/hooks). Returns the resulting :class:`CampaignState`.
        """
        xt = torch.as_tensor(x, dtype=torch.double).reshape(-1)
        self._pending = [(xt, None, 0)]
        result = (y, aux) if aux is not None else y
        return self.ingest(
            [result], observations=[observation], output_vars=[output_var]
        )

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
        """
        return len(self._seed_queue) + len(self._retry_queue)

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
            pending=[self._to_dict(x) for x, _, _ in self._pending]
            if self._pending
            else None,
            recommended=list(self._recommended),
            halted=self.halted,
        )

    # -- internals --

    # Deterministic misconfiguration (wrong policy for the spec, mixed design
    # space, a non-enumerable discrete axis) fires on the first suggest(). These
    # propagate so the caller/UI sees them immediately — never a silent random run.
    _MISCONFIG_ERRORS = (ValueError, NotImplementedError, TypeError)

    def _propose(self, n: int) -> torch.Tensor:
        model = self.current_model()
        if model is None:
            # Cold start: too few clean points to fit yet. A uniform probe is the
            # correct behavior here (not an error) — stay quiet.
            logger.debug("no model yet (too few clean points); drawing a random probe.")
            return self._random_draw(n)
        try:
            return self.policy.suggest(self.history, self.spec, model, n=n)
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

    def _random_draw(self, n: int) -> torch.Tensor:
        # Space-filling, and respects enumerated values + inequality constraints.
        return self.spec.sample_designs(n)

    def _fit_now(self):
        """Fit the surrogate on the current clean history (uncached), or None."""
        if not len(self.history.clean):
            return None
        try:
            return fit_model(self.model_constructor, self.history, self.spec)
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
            rec = recommend(self.history, self.spec, model, denoise=True)
        except Exception:
            logger.exception("recommend failed; using the observed front.")
            rec = []
        if not rec:
            rec = observed_front(self.history, self.spec)
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

    def _to_dict(self, x: torch.Tensor) -> dict:
        return dict(zip(self.spec.variable_names, x.reshape(-1).tolist()))

    @staticmethod
    def _as_list(results: object) -> list:
        if isinstance(results, list):
            return results
        return [results]
