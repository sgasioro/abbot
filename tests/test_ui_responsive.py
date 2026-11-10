"""The UI stays responsive while a slow experiment runs (steps run off the event loop).

Drives the real page headlessly (NiceGUI user simulation) in mock mode with a
blocking 1.5 s experiment, and measures the worst event-loop stall during auto-run.
Run on the loop, every step would freeze the page for the whole experiment.
"""

import asyncio
import time

import pytest

pytest.importorskip("nicegui")

from nicegui.testing import user_simulation

from abbot.ui import app

pytestmark = pytest.mark.slow


@pytest.mark.anyio
async def test_auto_run_does_not_block_event_loop(monkeypatch, ui_states):
    orig = app.MockAnalyticExperiment.run

    def slow_run(self, candidate):
        time.sleep(1.5)  # a blocking instrument call
        return orig(self, candidate)

    monkeypatch.setattr(app.MockAnalyticExperiment, "run", slow_run)
    async with user_simulation(root=app._build_page) as user:
        await user.open("/")
        state = ui_states[-1]
        state.mock = True
        state.step = state.reached = 4
        with user.client:
            state.refresh()
        await user.should_see(marker="create")
        user.find(marker="create").click()
        await user.should_see("Simulated experiment")
        user.find("Run").click()

        worst = 0.0
        t_end = time.perf_counter() + 4.0
        while time.perf_counter() < t_end:
            t0 = time.perf_counter()
            await asyncio.sleep(0.05)
            worst = max(worst, time.perf_counter() - t0 - 0.05)
        user.find("Pause").click()
        await asyncio.sleep(2.0)  # let the in-flight step finish

        assert len(state.controller.campaign.history) > 0  # steps actually ran
        assert worst < 1.0, f"event loop stalled {worst:.2f}s during a step"
