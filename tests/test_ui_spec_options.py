"""The setup panel covers the spec: sources with costs, travel limits, the model check."""

import pytest

pytest.importorskip("nicegui")

from nicegui.testing import user_simulation

from abbot.ui import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _keep_states(monkeypatch) -> list:
    states = []
    original = app.UIState.__init__

    def keep(self, *a, **kw):
        original(self, *a, **kw)
        states.append(self)

    monkeypatch.setattr(app.UIState, "__init__", keep)
    return states


@pytest.mark.anyio
async def test_a_source_campaign_records_the_cost_entered(monkeypatch):
    states = _keep_states(monkeypatch)
    async with user_simulation(root=app._build_page) as user:
        await user.open("/")
        user.find("Runs come from more than one source (fidelities, codes)").click()
        await user.should_see("target")
        user.find("Create campaign").click()
        await user.should_see("Suggest")
        campaign = states[-1].controller.campaign
        assert campaign.spec.sources[0].target == "experiment"
        user.find("Suggest").click()
        await user.should_see("Outstanding (1)")
        (cand,) = campaign.outstanding()
        assert cand["source"] in ("simulation", "experiment")
        user.find(marker=f"{cand.id}-y").type("0.5")
        user.find(marker=f"{cand.id}-cost").type("3")
        user.find(marker=f"{cand.id}-submit").click()
        await user.should_see("No designs outstanding.")
        assert campaign.history.records[-1].cost == 3.0


@pytest.mark.anyio
async def test_travel_limits_start_from_the_entered_setting(monkeypatch):
    states = _keep_states(monkeypatch)
    async with user_simulation(root=app._build_page) as user:
        await user.open("/")
        state = states[-1]
        state.variable_rows[0].update(max_step=0.1, start=0.3)
        user.find("Create campaign").click()
        await user.should_see("travel limits")
        campaign = state.controller.campaign
        (first,) = campaign.suggest(1)
        assert first["x1"] == pytest.approx(0.3)


@pytest.mark.anyio
async def test_the_model_check_opens_in_a_dialog(monkeypatch):
    states = _keep_states(monkeypatch)
    async with user_simulation(root=app._build_page) as user:
        await user.open("/")
        user.find("Create campaign").click()
        await user.should_see("Check the model")
        user.find(marker="check-model").click()
        await user.should_see(marker="check-report")
        assert states[-1].controller is not None


def test_variable_options_fold_into_a_summary():
    assert app._options_summary({"log": False, "max_step": None}) == ""
    assert app._options_summary({"log": True, "max_step": 0.1, "start": 0.5}) == (
        "log · step ≤ 0.1 from 0.5"
    )
    assert app._options_summary({"log": False, "max_step": 2.0, "start": None}) == "step ≤ 2"
