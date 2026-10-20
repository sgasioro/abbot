"""``run_campaign`` / ``drive`` — thin blocking drivers over a :class:`abbot.session.Campaign`.

The engine logic (fit, propose, canonical-Y, recommendation, resume, retry/halt)
lives in ``abbot.session`` so the web UI, a notebook, and an agent share it with
this loop. ``drive`` just owns the loop and calls ``Campaign.step`` — it is the
"who triggers the step = this blocking loop" caller; ``run_campaign`` builds the
campaign and drives it. An application that builds its own ``Campaign`` (custom
hooks, a wrapped experiment) calls ``drive`` directly.

The engine is natively multi-objective and maximize-canonical: ``evaluate``
returns RAW physical measurements (one column per objective, in ``spec.objectives``
order), and the session applies per-objective signs so the surrogate/acquisition
always see a "bigger is better" space.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch

from abbot.experiment import BatchExperiment, Experiment, Supervisor
from abbot.history import DataStore
from abbot.model import ModelConstructor
from abbot.policy import Policy
from abbot.problem import ProblemSpec

# Re-exported for backward compatibility — these moved to abbot.session.
from abbot.session import (  # noqa: F401
    Campaign,
    CampaignResult,
    CampaignState,
    ModelUQ,
    PersistentFaultError,
    PreRunCheck,
    Targets,
    canonical_Y,
    check_uq_pairing,
    default_initial_designs,
    fit_model,
    observed_front,
    split_eval,
)

logger = logging.getLogger(__name__)


def _identity(observation: object) -> object:
    return observation


def drive(
    campaign: Campaign,
    experiment: Experiment | BatchExperiment,
    evaluate: Callable[[object], object],
    *,
    n_total: int | None = None,
    batch_size: int = 1,
    on_persistent_fault: str = "halt",
    should_continue: Callable[[], bool] | None = None,
) -> Campaign:
    """Step ``campaign`` until its history holds ``n_total`` records — the one
    blocking loop, for a campaign you built yourself (``run_campaign`` and the UI's
    ``RunController`` both run on it).

    The budget is a total, not "this many more": a rerun after a halt or a crash
    finishes the remainder instead of spending the whole budget again, and every
    record counts against it — seeds, retries and failed runs alike. ``None`` runs
    until a halt or the gate stops it.

    ``batch_size`` designs are run per step (``Campaign.step(n=...)``); the last
    batch shrinks so the budget isn't overshot.

    ``should_continue`` is checked before each step; returning False stops the loop
    early (a UI's pause/stop). Read why it stopped off the campaign: ``halted``, or
    ``len(history)`` against the budget.

    On a persistent fault (the supervisor halted, or a run kept failing),
    ``on_persistent_fault="halt"`` raises :class:`PersistentFaultError` (the store
    keeps everything, rerun to resume); ``"continue"`` logs it and spends the rest
    of the budget. A halt that recorded nothing (``pre_run`` blocked the design)
    always raises, since continuing would retry the same blocked design forever.
    """
    if on_persistent_fault not in ("halt", "continue"):
        raise ValueError(
            f"on_persistent_fault must be 'halt' or 'continue', got {on_persistent_fault!r}"
        )
    while n_total is None or len(campaign.history) < n_total:
        if should_continue is not None and not should_continue():
            break
        before = len(campaign.history)
        n = batch_size if n_total is None else min(batch_size, n_total - before)
        campaign.step(experiment, evaluate, n=n)
        if not campaign.halted:
            continue
        if on_persistent_fault == "halt" or len(campaign.history) == before:
            done = len(campaign.history)
            of = f" of {n_total}" if n_total is not None else ""
            raise PersistentFaultError(
                f"{campaign.halted}\n{done}{of} shots completed. Fix the cause and "
                "rerun to resume."
            )
        logger.warning("Persistent fault (%s); continuing as configured.", campaign.halted)
        campaign.halted = None
    return campaign


def run_campaign(
    spec: ProblemSpec,
    experiment: Experiment | BatchExperiment,
    *,
    n_shots: int = 40,
    batch_size: int = 1,
    policy: Policy | None = None,
    model_constructor: ModelConstructor | None = None,
    evaluate: Callable[[object], object] | None = None,
    initial_designs: torch.Tensor | None = None,
    store: DataStore | None = None,
    supervisor: Supervisor | None = None,
    max_retries: int = 1,
    on_persistent_fault: str = "halt",
    pre_run: PreRunCheck | None = None,
    on_step=None,
    fingerprint: dict | None = None,
    allow_resume_mismatch: bool = False,
    targets: Targets | None = None,
) -> CampaignResult:
    """Run the propose -> evaluate -> retrain loop to completion.

    Builds a :class:`abbot.session.Campaign`, seeds it, and runs :func:`drive` to
    ``len(initial_designs) + n_shots`` records in total — retries and failed runs
    included, so rerunning the same call after a halt resumes and finishes the
    remainder. ``batch_size`` designs are run per step (see ``Campaign.step``);
    ``n_shots`` still counts designs, not batches. ``evaluate`` maps a raw observation to RAW physical objective
    value(s) (a tensor, or an ``(outputs, aux)`` tuple for the SBI seam); objective
    direction is applied internally (no caller pre-negation). Returns a
    ``CampaignResult`` whose ``recommended`` holds the handover index/indices.

    Only ``spec`` and ``experiment`` are required. ``policy`` and
    ``model_constructor`` default to what the spec implies, ``evaluate`` to the
    identity (the experiment returns the outputs directly), and ``initial_designs``
    to ``default_initial_designs(spec)``.
    """
    if evaluate is None:
        evaluate = _identity
    if initial_designs is None:
        initial_designs = default_initial_designs(spec)
    campaign = Campaign(
        spec,
        policy,
        model_constructor,
        store=store,
        supervisor=supervisor,
        max_retries=max_retries,
        pre_run=pre_run,
        on_step=on_step,
        fingerprint=fingerprint,
        allow_resume_mismatch=allow_resume_mismatch,
        targets=targets,
    )
    campaign.seed(initial_designs)
    drive(
        campaign,
        experiment,
        evaluate,
        n_total=len(initial_designs) + n_shots,
        batch_size=batch_size,
        on_persistent_fault=on_persistent_fault,
    )
    return campaign.result()
