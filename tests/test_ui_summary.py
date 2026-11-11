"""The UI's plain-language text: the plan before starting, and how it's going."""

import math

import pytest

pytest.importorskip("nicegui")

from abbot.ui import summary
from abbot.ui.state import UIState


def _state(**kw) -> UIState:
    s = UIState()
    s.variable_rows = [
        {"kind": "continuous", "name": "temp", "lower": 20.0, "upper": 80.0,
         "max_step": 2.0, "start": 25.0},
        {"kind": "continuous", "name": "conc", "lower": 0.001, "upper": 1.0, "log": True},
        {"kind": "enumerated", "name": "solvent", "values": "water, ethanol"},
    ]
    s.objective_rows = [{"name": "yield", "direction": "MAXIMIZE"},
                        {"name": "purity", "direction": "OBSERVE", "min_val": 0.9}]
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_the_plan_says_what_abbot_will_do():
    text = "\n".join(summary.plan_lines(_state()))
    assert ("Suggest runs over temp (20–80), conc (0.001–1, log scale) and solvent "
            "(water or ethanol).") in text
    assert "Look for the highest yield, keeping purity at least 0.9." in text
    assert "Also model purity, without optimizing it." in text
    assert "Move temp at most 2 between runs." in text
    assert "Start from your current settings." in text
    assert "You run each design" in text


def test_the_plan_for_a_target_and_a_simulation():
    s = _state(method="bax", mock=True)
    s.goal_row = {"kind": "level", "output": "yield", "threshold": 0.8, "upper": False}
    text = "\n".join(summary.plan_lines(s))
    assert "Map every setting where yield is at least 0.8, keeping purity at least 0.9" in text
    assert "Strategy: balanced (switchbax)." in text
    assert "A simulated experiment runs each design" in text


def test_an_incomplete_setup_says_why():
    s = _state()
    s.variable_rows[0]["upper"] = 10.0
    (line,) = summary.plan_lines(s)
    assert line.startswith("Not ready yet:")


def test_goal_words():
    assert summary.goal_words({"kind": "band", "bounds": {"y": [0.4, None]}}) == (
        "every setting where y is at least 0.4")
    assert summary.goal_words({"kind": "topk", "output": "y", "k": 3, "largest": False}) == (
        "the 3 settings with the lowest y")
    assert summary.goal_words({"kind": "percentile", "percentile": 90}) == "the top 10% of settings"


@pytest.mark.slow
def test_how_it_is_going_for_one_objective():
    s = UIState()
    s.variable_rows = [{"kind": "continuous", "name": "x", "lower": 0.0, "upper": 3.0}]
    s.start_campaign()
    c = s.controller.campaign
    assert summary.progress_lines(c) == [
        "No results yet. Record the first run to see how it's going."]
    for _ in range(8):
        (cand,) = c.suggest(1)
        c.ingest({"y": math.sin(cand["x"])}, id=cand.id)
    lines = summary.progress_lines(c)
    assert lines[0] == "8 runs."
    assert lines[1].startswith("Best so far: **y ")
    assert lines[2].startswith(("Still improving", "No better result"))
    (cand,) = c.suggest(1)
    c.fail(cand.id, "lamp")
    assert summary.progress_lines(c)[0] == "9 runs, 1 failed."


@pytest.mark.slow
def test_how_it_is_going_for_a_target():
    s = UIState()
    s.variable_rows = [{"kind": "continuous", "name": "x", "lower": 0.0, "upper": 4.0}]
    s.objective_rows = [{"name": "y", "direction": "OBSERVE"}]
    s.method = "bax"
    s.goal_row = {"kind": "band", "bands": {"y": {"lo": 0.4, "hi": 0.7}}}
    s.start_campaign()
    c = s.controller.campaign
    for _ in range(6):
        (cand,) = c.suggest(1)
        c.ingest({"y": math.sin(cand["x"])}, id=cand.id)
    lines = summary.progress_lines(c)
    assert "of the settings it considers in the target" in lines[1]
    assert lines[2].endswith("usable runs landed in it.")


def test_the_plan_and_progress_for_an_output_aimed_at_a_value():
    import torch

    from abbot import Campaign

    s = _state()
    s.objective_rows[0].update(direction="TARGET", target=None)
    (line,) = summary.plan_lines(s)
    assert line == "Not ready yet: Give the value to bring yield near."
    s.objective_rows[0]["target"] = 0.5
    assert "Look for the yield nearest 0.5, keeping purity at least 0.9." in summary.plan_lines(s)

    spec = s.build_spec()
    campaign = Campaign(spec, initial_designs=torch.empty(0, 3))
    for temp, v in zip((20.0, 40.0, 60.0, 80.0), (0.1, 0.9, 0.7, 0.55)):
        campaign.record({"temp": temp, "conc": 0.1, "solvent": "water"}, [v, 0.95])
    text = "\n".join(summary.progress_lines(campaign))
    assert "Best so far: **yield 0.55** (target 0.5)" in text
    assert "Still improving: 0.35 closer over the last 3 runs." in text
