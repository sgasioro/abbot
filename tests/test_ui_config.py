"""The setup as campaign.json: written, read back into the steps, and saved in the
directory the agent shares."""

import pytest

pytest.importorskip("nicegui")

from abbot.agent import Workspace
from abbot.ui.state import UIState


def _band_state() -> UIState:
    s = UIState()
    s.variable_rows = [
        {"kind": "continuous", "name": "temp", "lower": 20.0, "upper": 80.0, "max_step": 2.0,
         "start": 25.0},
        {"kind": "enumerated", "name": "solvent", "values": "water, ethanol",
         "start": "ethanol"},
    ]
    s.objective_rows = [{"name": "y", "direction": "OBSERVE"}]
    s.method = "bax"
    s.goal_row = {"kind": "band", "bands": {"y": {"lo": 0.4, "hi": None}}}
    return s


def test_campaign_json_reads_back_into_the_steps():
    config = _band_state().campaign_config()
    assert config["initial_designs"] == [{"temp": 25.0, "solvent": "ethanol"}]
    assert [d["kind"] for d in config["spec"]["domain"]] == ["continuous", "enumerated"]
    s = UIState()
    s.apply_config(config)
    assert s.method == "bax" and s.variable_rows[0]["start"] == 25.0
    assert s.campaign_config() == config


def test_fields_the_steps_cant_show_are_kept():
    config = _band_state().campaign_config()
    config["spec"]["domain"].append({"kind": "integer", "name": "n", "lower": 1, "upper": 4})
    config["spec"]["domain_constraints"] = [{"coeffs": {"temp": 1, "n": 1}, "max_val": 50}]
    config["initial_designs"] = [{"temp": 25, "solvent": "water", "n": 2}]
    config["max_retries"] = 3
    s = UIState()
    s.apply_config(config)
    assert s.variable_rows[-1]["kind"] == "json"
    assert set(s.extra) == {"domain_constraints", "max_retries"}
    assert s.campaign_config() == config
    s.start_campaign()  # in memory: no workspace
    assert s.controller.campaign.max_retries == 3


@pytest.mark.parametrize(("change", "why"), [
    ({"foo": 1}, "Unknown key 'foo'"),
    ({"goal": {"kind": "level", "output": "nope", "threshold": 1}}, "unknown output"),
    ({"model": "huge"}, "model must be one of"),
    ({"initial_designs": [{"temp": 25}, {"temp": 26}]}, r"initial_designs\[0\].*solvent"),
    ({"initial_designs": [{"solvent": "water"}]}, "temp' has a travel limit"),
])
def test_a_bad_campaign_json_changes_nothing(change, why):
    s = _band_state()
    before = s.campaign_config()
    with pytest.raises(Exception, match=why):
        s.apply_config({**before, **change})
    assert s.campaign_config() == before


def test_a_knob_renamed_in_the_json_keeps_the_other_starts():
    config = _band_state().campaign_config()
    config["spec"]["domain"][1]["name"] = "medium"
    s = UIState()
    s.apply_config(config)
    assert s.variable_rows[0]["start"] == 25.0  # temp, by name
    assert s.campaign_config()["initial_designs"] == [{"temp": 25.0, "medium": "water"}]


def test_a_bare_spec_loads_too():
    s = UIState()
    s.apply_config({"domain": [{"kind": "continuous", "name": "a", "lower": 0, "upper": 1}],
                    "outputs": [{"name": "y", "direction": "MINIMIZE"}]})
    assert s.method == "optimize" and s.objective_rows[0]["direction"] == "MINIMIZE"


@pytest.mark.slow
def test_a_campaign_started_in_the_ui_is_saved_where_the_agent_looks(tmp_path):
    s = _band_state()
    s.workspace = Workspace(tmp_path)
    s.name = "Solvent screen"
    assert s.save_path() == str(tmp_path / "Solvent-screen")
    s.start_campaign()
    assert s.saved_as == "Solvent-screen"
    (cand,) = s.controller.suggest(1)
    s.controller.submit(0.5, id=cand.id)
    s.new_campaign()  # lets go of it

    agent = Workspace(tmp_path)  # what the agent's server sees
    assert agent.status("Solvent-screen")["runs"] == 1
    agent.close("Solvent-screen")

    t = UIState()
    t.workspace = Workspace(tmp_path)
    t.open_saved("Solvent-screen")
    assert len(t.controller.campaign.history) == 1
    assert t.campaign_config() == s.campaign_config()

    s.mock = True
    assert s.save_path() is None  # a simulation isn't saved
