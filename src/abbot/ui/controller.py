"""RunController — the auto/manual run-state machine (framework-free, testable).

This is the "who triggers the step" layer the core deliberately omits. It drives a
:class:`abbot.session.Campaign` either manually (the caller invokes ``suggest`` /
``submit`` per step) or automatically (``run`` hands the campaign to :func:`abbot.loop.drive`,
the same blocking loop ``run_campaign`` uses, until paused/stopped/halted). It contains NO optimization logic — only sequencing
and the controls (run / pause / step / stop) — so it is unit-testable without a
browser and reused unchanged by the NiceGUI app.

Auto mode requires a programmatic ``Experiment`` (+ an ``evaluate``); without one,
only manual mode is available. ``Supervisor`` halt (surfaced on the campaign) forces
the controller to ``halted`` regardless of mode.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

from abbot.experiment import BatchExperiment, Experiment
from abbot.loop import drive
from abbot.session import Campaign, CampaignState, PersistentFaultError


class RunMode(str, Enum):
    MANUAL = "manual"
    AUTO = "auto"


class RunStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    HALTED = "halted"
    DONE = "done"


class RunController:
    """Sequencing + controls over a Campaign; holds no optimization logic."""

    def __init__(
        self,
        campaign: Campaign,
        *,
        experiment: Experiment | BatchExperiment | None = None,
        evaluate: Callable[[object], object] | None = None,
        max_steps: int | None = None,
    ) -> None:
        self.campaign = campaign
        self.experiment = experiment
        self.evaluate = evaluate
        self.max_steps = max_steps
        self.status = RunStatus.HALTED if campaign.halted else RunStatus.IDLE

    @property
    def can_auto(self) -> bool:
        """Auto mode is only possible with a programmatic experiment + evaluate."""
        return self.experiment is not None and self.evaluate is not None

    # -- manual mode: caller supplies measurements --

    def suggest(self, n: int = 1) -> list[dict]:
        self._guard()
        self.status = RunStatus.RUNNING
        return self.campaign.suggest(n)

    def submit(
        self, results: object, observations: list | None = None, *, id: str | None = None
    ) -> CampaignState:
        state = self.campaign.ingest(results, observations=observations, id=id)
        self._sync_status()
        return state

    def cancel(self, id: str) -> CampaignState:
        """Withdraw an outstanding design: nothing recorded, no retry."""
        state = self.campaign.cancel(id)
        self._sync_status()
        return state

    def fail(self, id: str, error: str) -> CampaignState:
        """Report that an outstanding design couldn't be run (retried as usual)."""
        state = self.campaign.fail(id, error)
        self._sync_status()
        return state

    # -- auto mode: controller drives Campaign.step --

    def step_once(self) -> CampaignState:
        """One auto step. Requires a programmatic experiment + evaluate."""
        self._guard()
        if not self.can_auto:
            raise RuntimeError(
                "auto step requires a programmatic Experiment and evaluate(); none "
                "attached (manual mode only)."
            )
        state = self.campaign.step(self.experiment, self.evaluate)
        self._sync_status()
        return state

    def run(self, should_continue: Callable[[], bool] | None = None) -> CampaignState:
        """Fire steps until paused/stopped/halted or ``max_steps`` reached.

        ``should_continue`` is an optional caller gate checked before each step (a
        UI passes a flag the Pause/Stop buttons flip; a headless test passes a
        counter). Returns the final state.
        """
        self._guard()
        if not self.can_auto:
            raise RuntimeError("run() requires auto mode (experiment + evaluate).")
        self.status = RunStatus.RUNNING
        try:
            # The shared blocking loop; a halt ends it as a status, not an error.
            drive(
                self.campaign,
                self.experiment,
                self.evaluate,
                n_total=self.max_steps,
                should_continue=should_continue,
            )
        except PersistentFaultError:
            pass
        self._sync_status()
        if self.status == RunStatus.RUNNING:  # neither halted nor out of budget
            self.status = RunStatus.PAUSED
        return self.campaign.state()

    # -- timer-driven auto mode: the controller owns run-state --
    #
    # A UI (or any periodic driver) calls start() then tick() on a timer and reads
    # is_running / status. This is the single source of truth for "are we running";
    # callers no longer keep a parallel flag or poke status directly. tick() is the
    # one-step-at-a-time analogue of run()'s loop body.

    @property
    def is_running(self) -> bool:
        return self.status == RunStatus.RUNNING

    def start(self) -> None:
        """Enter the running state (requires auto mode). Raises if halted."""
        self._guard()
        if not self.can_auto:
            raise RuntimeError("auto-run requires an Experiment + evaluate (manual mode only).")
        self.status = RunStatus.RUNNING

    def tick(self) -> bool:
        """Advance one auto step if running. Returns True iff a step was taken.

        No-op (returns False) when not running, halted/done, or the step budget is
        spent. Status transitions (DONE on budget or a supervisor stop, HALTED on
        fault) are owned here,
        so the caller just fires this on a timer and re-renders.
        """
        if self.status != RunStatus.RUNNING or not self.can_auto:
            return False  # manual mode: suggest() marks it running, the lab steps it
        if self.campaign.halted:
            self.status = RunStatus.HALTED
            return False
        if self.campaign.stopped or (
            self.max_steps is not None and len(self.campaign.history) >= self.max_steps
        ):
            self.status = RunStatus.DONE
            return False
        self.campaign.step(self.experiment, self.evaluate)
        self._sync_status()
        return True

    def pause(self) -> None:
        if self.status == RunStatus.RUNNING:
            self.status = RunStatus.PAUSED

    def stop(self) -> None:
        if self.status not in (RunStatus.HALTED,):
            self.status = RunStatus.DONE

    # -- internals --

    def _guard(self) -> None:
        if self.campaign.halted:
            self.status = RunStatus.HALTED
            raise RuntimeError(f"campaign is halted: {self.campaign.halted}")

    def _sync_status(self) -> None:
        if self.campaign.halted:
            self.status = RunStatus.HALTED
        elif self.campaign.stopped or (
            self.max_steps is not None and len(self.campaign.history) >= self.max_steps
        ):
            self.status = RunStatus.DONE
