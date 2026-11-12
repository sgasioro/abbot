"""Your own experiment in the UI: --experiment module:Name, and abbot.ui.serve()."""

import asyncio
import textwrap

import pytest
import torch

pytest.importorskip("nicegui")

from nicegui.testing import user_simulation

from abbot import Campaign, ContinuousDomain, OutputSpec, ProblemSpec
from abbot.ui import app, yours

SPEC = ProblemSpec(domain=[ContinuousDomain(name="x1", lower=0, upper=1)],
                   outputs=[OutputSpec(name="y")])


class Rig:
    """A stand-in instrument: y = 1 - (x1 - 0.3)²."""

    def __init__(self, spec=None):
        self.spec = spec
        self.seen = []

    def run(self, design):
        self.seen.append(dict(design))
        return {"y": 1 - (design["x1"] - 0.3) ** 2}


@pytest.fixture
def served(monkeypatch):
    """app._served for this test only, and a serve() that doesn't start a server."""
    monkeypatch.setattr(app, "_served", {"yours": None, "controller": None, "name": None})
    monkeypatch.setattr(app, "_launch", lambda *a, **k: None)
    return app._served


# -- loading --


def test_load_by_module_or_file(tmp_path, monkeypatch):
    (tmp_path / "myrig.py").write_text(textwrap.dedent("""
        class Rig:
            def __init__(self, spec):
                self.spec = spec
            def run(self, design):
                return [design["x1"]]

        def plain():
            return Rig(None)

        instance = Rig(None)
        not_one = 3
        def makes_junk():
            return 3
    """))
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    rig = yours.load("myrig:Rig")
    assert rig.name == "Rig" and rig.source == "myrig:Rig"
    assert rig.make(SPEC).spec is SPEC  # a maker taking an argument gets the spec
    assert yours.load("myrig:plain").make(SPEC).spec is None  # one taking none doesn't
    by_file = yours.load(f"{tmp_path / 'myrig.py'}:instance")
    assert by_file.make(SPEC) is by_file.make(SPEC)  # an instance is used as is
    assert by_file.source == "myrig.py:instance"
    with pytest.raises(ValueError, match="module:Name"):
        yours.load("myrig")
    with pytest.raises(AttributeError, match="has no 'Nope'"):
        yours.load("myrig:Nope")
    with pytest.raises(TypeError, match="neither an experiment"):
        yours.load("myrig:not_one")
    with pytest.raises(TypeError, match="made int, which has no run"):
        yours.load("myrig:makes_junk").make(SPEC)
    with pytest.raises(FileNotFoundError):
        yours.load("missing.py:Rig")


def test_a_bad_experiment_flag_stops_with_the_reason(served, capsys):
    with pytest.raises(SystemExit):
        app.main(["--no-show", "--experiment", "no_such_module_here:Rig"])
    assert "--experiment no_such_module_here:Rig" in capsys.readouterr().err


# -- the page --


@pytest.mark.anyio
async def test_your_experiment_runs_each_design_and_is_saved(ui_states, served, tmp_path):
    served["yours"] = yours.YourExperiment(Rig, source="lab:Rig")
    async with user_simulation(root=app._build_page) as user:
        await user.open("/")
        state = ui_states[-1]
        state.step = state.reached = 4
        with user.client:
            state.refresh()
        await user.should_see("Your experiment: Rig")
        await user.should_see("Rig runs each design when you press Run")
        assert state.own and state.runs_itself()
        user.find(marker="create").click()
        await user.should_see(marker="runner")
        ctrl = state.controller
        assert isinstance(ctrl.experiment, Rig) and ctrl.experiment.spec == state.spec
        assert len(ctrl.campaign.history) == 0  # nothing runs before Run
        user.find("Run").click()
        for _ in range(100):
            if len(ctrl.campaign.history) >= 3:
                break
            await asyncio.sleep(0.1)
        user.find("Pause").click()
        await asyncio.sleep(0.3)
        assert len(ctrl.campaign.history) >= 3
        assert len(ctrl.experiment.seen) == len(ctrl.campaign.history)
        assert state.saved_as is not None  # real measurements: saved like any campaign
        assert state.workspace.campaigns()


@pytest.mark.anyio
async def test_you_can_still_choose_to_enter_results(ui_states, served):
    served["yours"] = yours.YourExperiment(Rig())
    async with user_simulation(root=app._build_page) as user:
        await user.open("/")
        state = ui_states[-1]
        state.step = state.reached = 4
        with user.client:
            state.refresh()
        await user.should_see(marker="who-me")
        user.find(marker="who-me").click()
        await user.should_see("You run each design and enter what you measured.")
        user.find(marker="create").click()
        await user.should_see("Run this next")
        assert not state.controller.can_auto


@pytest.mark.anyio
async def test_serve_opens_on_a_campaign_from_code(ui_states, served):
    torch.manual_seed(0)
    campaign = Campaign(SPEC)
    app.serve(campaign, Rig, name="Bench test")
    async with user_simulation(root=app._build_page) as user:
        await user.open("/")
        state = ui_states[-1]
        await user.should_see("Bench test")
        await user.should_see(marker="runner")
        assert state.controller is served["controller"]
        assert state.controller.campaign is campaign
        assert state.saved_as is None  # your code owns it
        state.controller.step_once()
        assert len(campaign.history) == 1
        # Another tab shows the same campaign.
        await user.open("/")
        assert ui_states[-1].controller is state.controller


@pytest.mark.anyio
async def test_serve_without_an_experiment_waits_for_your_results(ui_states, served):
    app.serve(Campaign(SPEC))
    async with user_simulation(root=app._build_page) as user:
        await user.open("/")
        await user.should_see("Run this next")
