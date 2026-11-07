"""The UI pages, driven headlessly: the guided steps, the run view, halts, and the
expert panel."""

import asyncio

import pytest

pytest.importorskip("nicegui")

from nicegui.testing import user_simulation

from abbot.ui import app


async def _start(user, ui_states, configure=None):
    """Open the page, set the rows up, jump to Start and start the campaign."""
    await user.open("/")
    state = ui_states[-1]
    if configure is not None:
        configure(state)
    state.step = state.reached = 4
    with user.client:
        state.refresh()
    await user.should_see(marker="create")
    user.find(marker="create").click()
    await user.should_see("Run this next" if not state.mock else "Simulated experiment")
    return state


async def _wait_outstanding(user, campaign):
    """The outstanding designs, once the UI has suggested one."""
    for _ in range(200):
        if campaign.outstanding():
            return campaign.outstanding()
        await asyncio.sleep(0.05)
    raise AssertionError("no design was suggested")


@pytest.mark.anyio
async def test_the_steps_check_each_answer_before_moving_on(ui_states):
    async with user_simulation(root=app._build_page) as user:
        await user.open("/")
        state = ui_states[-1]
        user.find(marker="next").click()
        await user.should_see("What does each run measure?")
        state.objective_rows = [{"name": "y", "direction": "MAXIMIZE"},
                                {"name": "y", "direction": "OBSERVE"}]
        user.find(marker="next").click()
        await user.should_see("Two measurements have the same name.")
        state.objective_rows = [{"name": "y", "direction": "OBSERVE"}]
        user.find(marker="next").click()
        await user.should_see("What are you after?")
        user.find(marker="goal-target").click()
        await user.should_see("A band")
        user.find(marker="next").click()
        await user.should_see("needs a [lo, hi] on at least one output")
        state.goal_row["bands"] = {"y": {"lo": 0.4, "hi": 0.7}}
        user.find(marker="next").click()
        await user.should_see("Anything else?")
        user.find(marker="next").click()
        await user.should_see("Ready to start")
        await user.should_see("Map every setting where y is between 0.4 and 0.7")
        assert state.campaign_config()["goal"] == {"kind": "band", "bounds": {"y": [0.4, 0.7]}}
        user.find(marker="chip-Knobs").click()
        await user.should_see("What can you set?")


@pytest.mark.anyio
async def test_the_expert_panel_follows_the_steps(ui_states):
    async with user_simulation(root=app._build_page) as user:
        await user.open("/")
        state = ui_states[-1]
        user.find(marker="expert").click()
        await user.should_see("Inputs")
        user.find(marker="knob-0-name").clear().type("temp")
        await user.should_see("temp")
        assert state.campaign_config()["spec"]["domain"][0]["name"] == "temp"
        user.find("spec.json").click()
        await user.should_see(marker="spec-json")


@pytest.mark.anyio
async def test_a_source_campaign_records_the_cost_entered(ui_states):
    async with user_simulation(root=app._build_page) as user:
        state = await _start(user, ui_states, lambda s: setattr(s, "source_on", True))
        campaign = state.controller.campaign
        assert campaign.spec.sources[0].target == "experiment"
        (cand,) = await _wait_outstanding(user, campaign)
        assert cand["source"] in ("simulation", "experiment")
        user.find(marker=f"{cand.id}-y").type("0.5")
        user.find(marker=f"{cand.id}-cost").type("3")
        user.find(marker=f"{cand.id}-submit").click()
        await user.should_see(f"Recorded {cand.id}.")
        assert campaign.history.records[-1].cost == 3.0
        (nxt,) = await _wait_outstanding(user, campaign)  # the next one, suggested at once
        assert nxt.id != cand.id


@pytest.mark.anyio
async def test_travel_limits_start_from_the_entered_setting(ui_states):
    async with user_simulation(root=app._build_page) as user:
        state = await _start(user, ui_states, lambda s: s.variable_rows[0].update(
            max_step=0.1, start=0.3))
        (first,) = await _wait_outstanding(user, state.controller.campaign)
        assert first["x1"] == pytest.approx(0.3)


@pytest.mark.anyio
async def test_several_outstanding_designs_each_closed_on_its_own(ui_states):
    async with user_simulation(root=app._build_page) as user:
        state = await _start(user, ui_states)
        campaign = state.controller.campaign
        await _wait_outstanding(user, campaign)
        state.controller.suggest(2)
        with user.client:
            state.refresh()
        ids = [c.id for c in campaign.outstanding()]
        assert len(ids) == 3
        await user.should_see("Also out:")
        # The oldest shows first: record it; then fail the next, skip the last.
        user.find(marker=f"{ids[0]}-y").type("1.5")
        user.find(marker=f"{ids[0]}-submit").click()
        await user.should_see(marker=f"{ids[1]}-fail-open")
        user.find(marker=f"{ids[1]}-fail-open").click()
        user.find(marker=f"{ids[1]}-fail-reason").type("pump jammed")
        user.find(marker=f"{ids[1]}-fail").click()
        await user.should_see(marker=f"{ids[2]}-cancel")
        user.find(marker=f"{ids[2]}-cancel").click()
        await user.should_see(f"Skipped {ids[2]}.")
        records = campaign.history.records
        assert [r.id for r in records] == ids[:2]
        assert records[0].outputs.tolist() == [1.5]
        assert records[1].fault and records[1].error == "pump jammed"
        (retry,) = await _wait_outstanding(user, campaign)  # the failed design, again
        assert campaign.spec.as_tensor(dict(retry)).tolist() == records[1].candidate.tolist()


async def _halted(user, ui_states):
    state = await _start(user, ui_states)
    ctrl = state.controller
    await _wait_outstanding(user, ctrl.campaign)
    for _ in range(ctrl.campaign.max_retries + 1):
        cand = ctrl.campaign.outstanding()[0] if ctrl.campaign.outstanding() else ctrl.suggest(1)[0]
        ctrl.fail(cand.id, "shutter stuck")
    assert ctrl.campaign.halted
    with user.client:
        state.refresh()
    await user.should_see("What went wrong at")
    return state


@pytest.mark.anyio
async def test_a_halt_resolved_as_a_bad_region(ui_states):
    async with user_simulation(root=app._build_page) as user:
        state = await _halted(user, ui_states)
        campaign = state.controller.campaign
        user.find(marker="halt-reason").type("dark: flat profile")
        user.find(marker="halt-bad-region").click()
        await _wait_outstanding(user, campaign)
        assert campaign.halted is None
        assert {r.bad_region for r in campaign.history.records} == {"dark: flat profile"}


@pytest.mark.anyio
async def test_a_halt_resolved_by_retrying(ui_states):
    async with user_simulation(root=app._build_page) as user:
        state = await _halted(user, ui_states)
        campaign = state.controller.campaign
        halted_on = campaign.halted_on
        user.find(marker="halt-retry").click()
        (retry,) = await _wait_outstanding(user, campaign)
        assert campaign.halted is None
        assert [dict(retry)] == halted_on


@pytest.mark.anyio
async def test_a_value_is_corrected_from_the_runs_table(ui_states):
    async with user_simulation(root=app._build_page) as user:
        state = await _start(user, ui_states)
        campaign = state.controller.campaign
        (cand,) = await _wait_outstanding(user, campaign)
        user.find(marker=f"{cand.id}-y").type("-0.84")
        user.find(marker=f"{cand.id}-submit").click()
        await user.should_see(f"Recorded {cand.id}.")
        user.find(marker="expert").click()
        await user.should_see(marker="runs-table")
        rows = user.find(marker="runs-table").elements.pop().rows
        measured = next(r for r in rows if r["id"] == cand.id)
        user.find(marker="runs-table").trigger("rowClick", [None, measured, 0])
        await user.should_see(marker="correct-save")
        user.find(marker="correct-y").clear().type("0.65")
        user.find(marker="correct-save").click()
        await user.should_see("Corrected run 1")
        assert campaign.history.records[0].outputs.tolist() == [0.65]
        assert campaign.history.records[0].corrections[0]["outputs"] == [-0.84]


@pytest.mark.anyio
async def test_finishing_shows_the_recommendation(ui_states):
    async with user_simulation(root=app._build_page) as user:
        state = await _start(user, ui_states)
        campaign = state.controller.campaign
        (cand,) = await _wait_outstanding(user, campaign)
        user.find(marker=f"{cand.id}-y").type("0.5")
        user.find(marker=f"{cand.id}-submit").click()
        await user.should_see(marker="finish")
        user.find(marker="finish").click()
        user.find(marker="finish-confirm").click()
        await user.should_see("Finished")
        assert state.finished
        user.find(marker="keep-going").click()
        await user.should_see("Run this next")
