"""A halted campaign offers its answers in the UI: retry, skip, or bad region."""

import pytest

pytest.importorskip("nicegui")

from nicegui.testing import user_simulation

from abbot.ui import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def _halted(user, monkeypatch):
    """Open the page, create a campaign, and fail one design until it halts."""
    states = []
    original = app.UIState.__init__

    def keep(self, *a, **kw):
        original(self, *a, **kw)
        states.append(self)

    monkeypatch.setattr(app.UIState, "__init__", keep)
    await user.open("/")
    user.find("Create campaign").click()
    await user.should_see("Suggest")
    state = states[-1]
    ctrl = state.controller
    for _ in range(ctrl.campaign.max_retries + 1):
        (cand,) = ctrl.suggest(1)
        ctrl.fail(cand.id, "shutter stuck")
    assert ctrl.campaign.halted
    state._left.refresh()
    await user.should_see("What went wrong at")
    return state


@pytest.mark.anyio
async def test_a_halt_resolved_as_a_bad_region(monkeypatch):
    async with user_simulation(root=app._build_page) as user:
        state = await _halted(user, monkeypatch)
        campaign = state.controller.campaign
        user.find(marker="halt-reason").type("dark: flat profile")
        user.find(marker="halt-bad-region").click()
        await user.should_see("Suggest")
        assert campaign.halted is None
        assert {r.bad_region for r in campaign.history.records} == {"dark: flat profile"}
        assert state.controller.status.value == "idle"


@pytest.mark.anyio
async def test_a_halt_resolved_by_retrying(monkeypatch):
    async with user_simulation(root=app._build_page) as user:
        state = await _halted(user, monkeypatch)
        campaign = state.controller.campaign
        halted_on = campaign.halted_on
        user.find(marker="halt-retry").click()
        await user.should_see("Suggest")
        assert campaign.halted is None
        assert [dict(c) for c in state.controller.suggest(1)] == halted_on
