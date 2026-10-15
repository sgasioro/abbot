"""``run_campaign`` — a thin blocking driver over a :class:`abbot.session.Campaign`.

Merge basis: ``~/sbido/src/sbido/optimizer.py::measure_and_optimize``. The actual
engine logic (fit, propose, canonical-Y, recommendation, resume, retry/halt) now
lives in ``abbot.session`` so the web UI, a notebook, and an agent share it with
this loop. ``run_campaign`` just owns the for-loop and calls ``Campaign.step`` —
it is the "who triggers the step = this blocking loop" caller.

The engine is natively multi-objective and maximize-canonical: ``evaluate``
returns RAW physical measurements (one column per objective, in ``spec.objectives``
order), and the session applies per-objective signs so the surrogate/acquisition
always see a "bigger is better" space.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch

from abbot.experiment import Experiment, Supervisor
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


def run_campaign(
    spec: ProblemSpec,
    experiment: Experiment,
    *,
    n_shots: int = 40,
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
) -> CampaignResult:
    """Run the propose -> evaluate -> retrain loop to completion.

    Builds a :class:`abbot.session.Campaign`, seeds it, and drives ``n_shots`` steps
    plus any seed/retry steps. ``evaluate`` maps a raw observation to RAW physical
    objective value(s) (a tensor, or an ``(outputs, aux)`` tuple for the SBI seam);
    objective direction is applied internally (no caller pre-negation). Returns a
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
    )
    campaign.seed(initial_designs)

    # Total steps = queued seeds (0 if resumed) + the requested shots. Retries add
    # themselves to the queue and are consumed by subsequent suggest() calls, so we
    # drive until the shot budget and all pending retries are exhausted.
    # (+ anything left outstanding by an interrupted run, which step() finishes first)
    n_seeds = campaign.pending_count() + len(campaign.outstanding())
    n_total = n_seeds + n_shots
    for _ in range(n_total):
        if campaign.halted:
            if on_persistent_fault == "halt":
                raise PersistentFaultError(
                    f"{campaign.halted}\n{len(campaign.history)} shots completed. Fix "
                    "the hardware and rerun to resume."
                )
            # "continue": spend the remaining budget anyway.
            logger.warning("Persistent fault (%s); continuing as configured.", campaign.halted)
            campaign.halted = None
        campaign.step(experiment, evaluate)

    # Drain any outstanding retries queued by the supervisor on the final steps.
    while campaign.pending_count() and not campaign.halted:
        campaign.step(experiment, evaluate)

    if campaign.halted and on_persistent_fault == "halt":
        raise PersistentFaultError(
            f"{campaign.halted}\n{len(campaign.history)} shots completed. Fix the "
            "hardware and rerun to resume."
        )

    return campaign.result()
