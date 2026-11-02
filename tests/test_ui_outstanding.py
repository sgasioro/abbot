"""Manual mode lists every outstanding design, each with its own entry and actions."""

import pytest

pytest.importorskip("nicegui")

from nicegui import ui
from nicegui.testing import user_simulation

from abbot.ui import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_several_outstanding_designs_each_closed_on_its_own(monkeypatch):
    states = []
    original = app.UIState.__init__

    def keep(self, *a, **kw):
        original(self, *a, **kw)
        states.append(self)

    monkeypatch.setattr(app.UIState, "__init__", keep)
    async with user_simulation(root=app._build_page) as user:
        await user.open("/")
        user.find("Create campaign").click()
        await user.should_see("Suggest")
        state = states[-1]
        campaign = state.controller.campaign
        state.n_suggest = 3
        user.find("Suggest").click()
        await user.should_see("Outstanding (3)")
        ids = [c.id for c in campaign.outstanding()]

        # Report the second, cancel the third, submit the first.
        assert len(user.find(kind=ui.button, content="Submit").elements) == 3
        user.find(marker=f"{ids[1]}-fail").click()
        await user.should_see(marker=f"{ids[2]}-cancel")
        user.find(marker=f"{ids[2]}-cancel").click()
        await user.should_see("Outstanding (1)")
        assert [c.id for c in campaign.outstanding()] == [ids[0]]
        assert campaign.history.records[-1].fault  # the reported failure
        assert campaign._retry_queue  # it will be retried
        for name in campaign.spec.output_names:
            user.find(marker=f"{ids[0]}-{name}").type("1.5")
        user.find(marker=f"{ids[0]}-submit").click()
        await user.should_see("No designs outstanding.")
        assert campaign.history.records[-1].id == ids[0]
        assert campaign.history.records[-1].outputs.tolist() == [1.5] * len(campaign.spec.output_names)
