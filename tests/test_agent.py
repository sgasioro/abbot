"""Agent access: named campaigns saved in a directory, and the MCP server over them."""

import asyncio

import pytest

from abbot.agent import Workspace

pytestmark = pytest.mark.slow  # GP fits via suggest

SPEC = {
    "domain": [{"kind": "continuous", "name": "temp", "lower": 20, "upper": 80}],
    "outputs": [{"name": "yield", "direction": "MAXIMIZE"}],
}


def _yield(design):
    return 1 - ((design["temp"] - 55) / 30) ** 2


def _run(ws, name, n, f=_yield, output="yield"):
    for _ in range(n):
        (s,) = ws.suggest(name)
        ws.report_result(name, s["id"], {output: f(s["design"])})


def test_a_campaign_resumes_by_name_after_a_restart(tmp_path):
    ws = Workspace(tmp_path)
    ws.create("opt", SPEC)
    _run(ws, "opt", 6)
    (pending,) = ws.suggest("opt")

    again = Workspace(tmp_path)  # a new server process
    assert again.campaigns() == [
        {"name": "opt", "runs": 6, "outstanding": 1, "halted": None, "stopped": None}
    ]
    again.report_result("opt", pending["id"], {"yield": _yield(pending["design"])})
    (best,) = again.recommend("opt")
    assert abs(best["design"]["temp"] - 55) < 8
    assert "temp" in again.history("opt")


def test_a_goal_makes_a_bax_campaign_and_halts_wait_for_an_answer(tmp_path):
    ws = Workspace(tmp_path)
    spec = {"domain": [{"kind": "continuous", "name": "x", "lower": 0, "upper": 1}],
            "outputs": [{"name": "y", "direction": "OBSERVE"}]}
    status = ws.create("bax", spec, goal={"kind": "band", "bounds": {"y": [0.2, 0.5]}})
    assert status["policy"] == "BAXPolicy"
    _run(ws, "bax", 4, f=lambda d: d["x"] ** 2, output="y")
    for _ in range(2):  # max_retries=1: the second failure halts
        (s,) = ws.suggest("bax")
        ws.report_failure("bax", s["id"], "shutter stuck")
    status = ws.status("bax")
    assert status["halted"] and status["resolve_with"] == ["retry", "exclude", "bad_region"]
    with pytest.raises(ValueError, match="halted"):
        ws.suggest("bax")
    assert ws.resolve_halt("bax", "bad_region", "dark")["halted"] is None
    assert Workspace(tmp_path).status("bax")["halted"] is None  # saved


def test_results_from_outside_the_loop_and_clear_errors(tmp_path):
    ws = Workspace(tmp_path)
    ws.create("opt", SPEC)
    status = ws.add_results("opt", [
        {"design": {"temp": t}, "outputs": {"yield": _yield({"temp": t})}, "cost": 2.0}
        for t in (25.0, 50.0, 75.0)
    ])
    assert status["runs"] == 3 and status["cost_spent"] == 6.0
    with pytest.raises(ValueError, match="already exists"):
        ws.create("opt", SPEC)
    with pytest.raises(ValueError, match="letters, digits"):
        ws.create("../escape", SPEC)
    with pytest.raises(ValueError, match="No campaign named 'nope'"):
        ws.status("nope")
    with pytest.raises(ValueError, match="strategy"):
        ws.create("other", SPEC, strategy="Fastest")
    assert not (tmp_path / "other").exists()


def test_a_narrowed_campaign_keeps_the_runs_that_fit(tmp_path):
    ws = Workspace(tmp_path)
    ws.create("wide", SPEC)
    ws.add_results("wide", [
        {"design": {"temp": t}, "outputs": {"yield": _yield({"temp": t})}}
        for t in (25.0, 50.0, 75.0)
    ])
    (s,) = ws.suggest("wide")
    ws.report_failure("wide", s["id"], "interlock")
    narrow = {**SPEC, "domain": [{"kind": "continuous", "name": "temp",
                                  "lower": 20, "upper": 62}]}
    out = ws.create("narrow", narrow, results_from="wide")
    assert out["runs"] == 2 and out["copied"] == 2
    assert out["skipped"] == {"outside the new ranges": 1, "failed or bad-region run": 1}
    with pytest.raises(ValueError, match="No campaign named"):
        ws.create("other", SPEC, results_from="nope")


def test_the_loop_replies_are_short(tmp_path):
    ws = Workspace(tmp_path)
    ws.create("opt", SPEC)
    (s,) = ws.suggest("opt")
    brief = ws.report_result("opt", s["id"], {"yield": 0.5})
    assert "recommended" not in brief and "cost_spent" not in brief
    assert ws.status("opt")["recommended"]


def test_a_value_reported_wrong_is_corrected_in_place(tmp_path):
    ws = Workspace(tmp_path)
    ws.create("opt", SPEC)
    (s,) = ws.suggest("opt")
    ws.report_result("opt", s["id"], {"yield": -0.84})
    reply = ws.correct_result("opt", s["id"], {"yield": 0.65}, reason="typo")
    assert reply["runs"] == 1
    assert reply["corrected"] == {"run": s["id"], "was": {"yield": -0.84},
                                  "now": {"yield": 0.65}}
    assert Workspace(tmp_path).recommend("opt")[0]["outputs"] == {"yield": 0.65}


def test_plots_are_saved_in_the_campaign_folder(tmp_path):
    pytest.importorskip("matplotlib")
    ws = Workspace(tmp_path)
    ws.create("opt", SPEC)
    _run(ws, "opt", 6)
    path = ws.plot("opt")
    assert path == tmp_path / "opt" / "plots" / "model-006runs.png" and path.exists()
    assert ws.plot("opt", "check").name == "check-006runs.png"
    with pytest.raises(ValueError, match="kind must be"):
        ws.plot("opt", "contour")


def test_the_mcp_server_serves_the_workspace(tmp_path):
    pytest.importorskip("mcp")
    from mcp import Client

    from abbot.agent.server import build_server

    async def session():
        async with Client(build_server(tmp_path)) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            assert {"spec_guide", "create_campaign", "suggest", "report_result",
                    "resolve_halt", "recommend", "check_model"} <= names
            r = await client.call_tool("create_campaign", {"name": "opt", "spec": SPEC})
            assert not r.is_error
            r = await client.call_tool("suggest", {"name": "opt", "n": 2})
            designs = r.structured_content["result"]
            assert len(designs) == 2 and "temp" in designs[0]["design"]
            r = await client.call_tool("status", {"name": "nope"})
            assert r.is_error and "No campaign named 'nope'" in r.content[0].text
            r = await client.call_tool("report_result", {
                "name": "opt", "id": designs[0]["id"], "outputs": {"yield": 0.1}})
            r = await client.call_tool("correct_result", {
                "name": "opt", "run": designs[0]["id"], "outputs": {"yield": 0.2}})
            assert not r.is_error and '"now"' in r.content[0].text
            r = await client.call_tool("plot", {"name": "opt"})
            assert not r.is_error
            assert [c.type for c in r.content] == ["text", "image"]
            assert r.content[0].text.endswith("model-001runs.png")
            r = await client.call_tool("spec_guide", {})
            assert "Who runs each experiment" in r.content[0].text
            r = await client.call_tool("close_campaign", {"name": "opt"})
            assert not r.is_error and not (tmp_path / "opt" / ".lock").exists()

            prompts = {p.name for p in (await client.list_prompts()).prompts}
            assert "plan_campaign" in prompts
            r = await client.get_prompt("plan_campaign", {"task": "anneal films"})
            text = r.messages[0].content.text
            assert "The task: anneal films" in text and "spec_guide" in text

    asyncio.run(session())


def test_one_process_at_a_time_holds_a_campaign(tmp_path):
    ws = Workspace(tmp_path)
    ws.create("opt", SPEC)
    folder = tmp_path / "opt"
    assert (folder / ".lock").exists()
    (folder / ".lock").write_text("1")  # pid 1 is always running: another holder
    ws.close("opt")
    assert (folder / ".lock").read_text() == "1"  # not ours to remove
    with pytest.raises(ValueError, match="open in another process"):
        Workspace(tmp_path).open("opt")
    assert Workspace(tmp_path).campaigns()[0]["open_elsewhere"] == 1  # listed, not opened
    (folder / ".lock").write_text("999999999")  # a process that has exited
    assert len(Workspace(tmp_path).open("opt").history) == 0


def test_free_names(tmp_path):
    ws = Workspace(tmp_path)
    assert ws.free_name("My campaign!") == "My-campaign"
    ws.create("My-campaign", SPEC)
    assert ws.free_name("My campaign") == "My-campaign-2"
    assert ws.free_name("  ") == "campaign"
