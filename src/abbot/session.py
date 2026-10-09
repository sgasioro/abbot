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
from typing import Any

import torch
from botorch.utils.multi_objective.pareto import is_non_dominated

from abbot.experiment import Experiment, Supervisor, Verdict
from abbot.history import DataStore, History, Record
from abbot.model import ModelConstructor
from abbot.policy import Policy
from abbot.problem import ProblemSpec

logger = logging.getLogger(__name__)

# A model's capability: what UQ quality its posterior actually provides.
ModelUQ = str  # "point" | "samples" | "calibrated_variance"

_SATISFIES: dict[str, set[str]] = {
    "point": {"point", "samples", "calibrated_variance"},
    "samples": {"samples", "calibrated_variance"},
    "calibrated_variance": {"calibrated_variance"},
}


class PersistentFaultError(RuntimeError):
    """The apparatus needs a human. Every completed step is already on disk."""


def check_uq_pairing(policy: Policy, model_uq: ModelUQ) -> None:
    """Refuse a model<->policy pairing the surrogate can't support.

    e.g. deterministic-NN ("point") + UncertaintySampling ("calibrated_variance")
    is refused with a clear error; deterministic-NN + MeanBAX ("point") is fine.
    """
    required = policy.uq_requirement
    if model_uq not in _SATISFIES[required]:
        raise ValueError(
            f"{type(policy).__name__} requires uq_requirement={required!r}, but the "
            f"model only provides {model_uq!r}. Pair it with a model that provides "
            f"at least {required!r} UQ, or choose a lower-UQ policy."
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


def observed_front(history: History, spec: ProblemSpec) -> list[int]:
    """Fallback handover: non-dominated set of the OBSERVED (noisy) OBJECTIVE
    columns, in canonical space. Scalar case collapses to the single argmax.
    Returns ``[]`` for a pure-BAX spec (no objectives — handover is the goal's job).
    """
    idx = spec.objective_indices()
    if not idx:
        return []
    keep = [i for i, r in enumerate(history.records) if not r.fault]
    if not keep:
        return []
    Y = torch.stack([history.records[i].outputs.reshape(-1) for i in keep]).double()
    Y = Y[:, idx] * spec.objective_signs()
    if Y.shape[-1] == 1:
        return [keep[int(Y.reshape(-1).argmax())]]
    nd = is_non_dominated(Y)
    return [keep[i] for i in range(len(keep)) if bool(nd[i])]


def fit_model(
    model_constructor: ModelConstructor, history: History, spec: ProblemSpec
):
    """Fit the surrogate on the clean history in maximize-canonical space."""
    X = history.clean.designs.double()
    Y = canonical_Y(history, spec)
    return model_constructor.build(X, Y)


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
    """A resumable, step-based optimization session over the abbot engine."""

    def __init__(
        self,
        spec: ProblemSpec,
        policy: Policy,
        model_constructor: ModelConstructor,
        *,
        store: DataStore | None = None,
        supervisor: Supervisor | None = None,
        max_retries: int = 1,
        model_uq: ModelUQ = "calibrated_variance",
        on_step: StepHook | list[StepHook] | None = None,
    ) -> None:
        check_uq_pairing(policy, model_uq)
        self.spec = spec
        self.policy = policy
        self.model_constructor = model_constructor
        self.store = store
        self.supervisor = supervisor
        self.max_retries = max_retries
        self.model_uq = model_uq
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
        self, results: object, observations: list | None = None
    ) -> CampaignState:
        """Fold measured results for the pending suggestion back in.

        ``results`` is one eval-output per pending candidate (a tensor or an
        ``(outputs, aux)`` tuple each); a bare value is accepted when a single
        candidate is pending. ``observations`` optionally stores the raw
        observation per record. Runs the supervisor, persists, and fires hooks.
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

        for (x, retry_of, attempt), res, obs in zip(self._pending, results, obs_list):
            y, aux = split_eval(res, self.spec)
            record = Record(
                candidate=x, observation=obs, outputs=y, aux=aux, retry_of=retry_of
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

    # -- queries --

    def recommend(self) -> list[int]:
        """Current denoised handover index/indices (fits a final model)."""
        if not len(self.history.clean):
            return []
        model = self._safe_fit()
        return self._safe_recommend(model)

    def result(self) -> CampaignResult:
        """A ``CampaignResult`` with the end-of-run recommendation."""
        return CampaignResult(history=self.history, recommended=self.recommend())

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

    def _propose(self, n: int) -> torch.Tensor:
        try:
            model = self._safe_fit()
            if model is None:
                raise RuntimeError("no model (too few clean points)")
            return self.policy.suggest(self.history, self.spec, model, n=n)
        except Exception:
            logger.exception("proposal failed; falling back to uniform random draws.")
            bounds = self.spec.bounds_tensor()
            lo, hi = bounds[0], bounds[1]
            return lo + (hi - lo) * torch.rand(n, bounds.shape[-1], dtype=torch.double)

    def _safe_fit(self):
        if not len(self.history.clean):
            return None
        try:
            return fit_model(self.model_constructor, self.history, self.spec)
        except Exception:
            logger.exception("model fit failed.")
            return None

    def _safe_recommend(self, model) -> list[int]:
        rec: list[int] = []
        if model is not None:
            try:
                rec = self.policy.recommend(self.history, self.spec, model)
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
        model = self._safe_fit()
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
