"""Your own experiment in the UI: --experiment module:Name, and abbot.ui.serve()."""

import asyncio
import json
import textwrap

import pytest
import torch

pytest.importorskip("nicegui")

from nicegui.testing import user_simulation

from abbot import Campaign, ContinuousDomain, EnumeratedDomain, OutputSpec, ProblemSpec
from abbot.problem import LinearConstraint
from abbot.ui import app, yours
from abbot.ui.builder import spec_json

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


# -- the spec your experiment works with --

DECLARED = ProblemSpec(
    domain=[ContinuousDomain(name="temp", lower=20, upper=80, max_step=5),
            EnumeratedDomain(name="solvent", values=["water", "ethanol", "dmso"])],
    outputs=[OutputSpec(name="yield")],
    domain_constraints=[LinearConstraint({"temp": 1}, max_val=70)],
)


def _setup(**fields) -> ProblemSpec:
    base = {"domain": [ContinuousDomain(name="temp", lower=30, upper=60, max_step=2),
                       EnumeratedDomain(name="solvent", values=["water", "dmso"])],
            "outputs": [OutputSpec(name="yield", direction="MINIMIZE", min_val=0)],
            "domain_constraints": [LinearConstraint({"temp": 1}, max_val=65)]}
    return ProblemSpec(**{**base, **fields})


class Declared:
    """A rig that says which knobs it reads, how far they go, and what it returns."""

    spec = DECLARED

    def run(self, design):
        return {"yield": 1.0}


def test_within_narrows_but_never_widens():
    assert yours.within(DECLARED, _setup()) is None  # narrower, tighter: fine; goal free
    temp, solvent = _setup().domain
    cases = {
        "goes from 20 to 80": [temp.model_copy(update={"lower": 10}), solvent],
        "at most 5 per run": [temp.model_copy(update={"max_step": None}), solvent],
        "not acetone": [temp, EnumeratedDomain(name="solvent", values=["water", "acetone"])],
        "takes choices, not a range": [temp, ContinuousDomain(name="solvent", lower=0,
                                                              upper=1)],
    }
    for words, domain in cases.items():
        assert words in yours.within(DECLARED, _setup(domain=domain), "Rig")
    assert "sets temp and solvent; this setup has x1" in yours.within(
        DECLARED, _setup(domain=[ContinuousDomain(name="x1", lower=0, upper=1)],
                         domain_constraints=None), "Rig")
    assert "measures yield; this setup has y" in yours.within(
        DECLARED, _setup(outputs=[OutputSpec(name="y")]), "Rig")
    for kept in (None, [LinearConstraint({"temp": 1}, max_val=75)]):
        assert "keeps temp ≤ 70" in yours.within(DECLARED, _setup(domain_constraints=kept),
                                                 "Rig")


def test_the_declared_spec_comes_from_the_experiment_or_alongside(tmp_path, monkeypatch):
    rig = yours.YourExperiment(Declared)
    assert rig.spec is DECLARED
    assert isinstance(rig.make(_setup()), Declared)
    with pytest.raises(ValueError, match="goes from 20 to 80"):
        rig.make(_setup(domain=[ContinuousDomain(name="temp", lower=0, upper=60, max_step=1),
                                EnumeratedDomain(name="solvent", values=["water"])]))
    assert yours.YourExperiment(Rig).spec is None  # no declaration: anything goes
    assert yours.YourExperiment(Rig, spec=DECLARED).make(_setup()).spec == _setup()
    # A spec file (bare, or campaign.json), or a module attribute, via --spec.
    (tmp_path / "rig.json").write_text(json.dumps(spec_json(DECLARED)))
    (tmp_path / "campaign.json").write_text(json.dumps({"spec": spec_json(DECLARED)}))
    (tmp_path / "lab.py").write_text(textwrap.dedent("""
        from abbot import ProblemSpec
        class Rig:
            def run(self, design):
                return {"yield": 1.0}
        SPEC = None
    """))
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    import lab

    lab.SPEC = DECLARED
    for ref in ("rig.json", str(tmp_path / "campaign.json"), "lab:SPEC"):
        assert yours.load("lab:Rig", spec=ref).spec == DECLARED
    with pytest.raises(FileNotFoundError, match="--spec"):
        yours.load("lab:Rig", spec="missing.json")


def test_spec_without_experiment_is_refused(served, capsys):
    with pytest.raises(SystemExit):
        app.main(["--no-show", "--spec", "rig.json"])
    assert "--spec goes with --experiment" in capsys.readouterr().err
    with pytest.raises(ValueError, match="pass the experiment too"):
        app.serve(Campaign(SPEC), spec=DECLARED)


def test_returned_names_that_dont_match_say_so():
    check = yours.YourExperiment(Declared).outputs(_setup(outputs=[OutputSpec(name="y")]))
    with pytest.raises(ValueError, match="Declared returned yield; this setup expects y"):
        check({"yield": 1.0})
    assert check(({"y": 1.0}, "raw")) == ({"y": 1.0}, "raw")  # (outputs, aux) passes
    assert check([1.0]) == [1.0]  # a list: checked by count downstream


def test_a_run_asking_for_a_knob_that_isnt_there_says_so():
    class Wrong:
        def run(self, design):
            return {"y": design["temp"]}

    campaign = Campaign(SPEC, max_retries=0)
    campaign.step(Wrong())
    assert "asked for 'temp', which isn't a variable here; the design has x1" in (
        campaign.history.records[-1].error)


def test_serve_refuses_a_campaign_beyond_the_declared_spec(served):
    with pytest.raises(ValueError, match="sets temp and solvent; this setup has x1"):
        app.serve(Campaign(SPEC), Declared)


@pytest.mark.anyio
async def test_the_setup_starts_from_the_declared_spec_and_stays_in_it(ui_states, served):
    served["yours"] = yours.YourExperiment(Declared, source="lab:Rig")
    async with user_simulation(root=app._build_page) as user:
        await user.open("/")
        state = ui_states[-1]
        assert [r["name"] for r in state.variable_rows] == ["temp", "solvent"]
        await user.should_see("Declared: 20 to 80, moves ≤ 5 per run")
        await user.should_see("Declared: choices water, ethanol, dmso")
        await user.should_not_see(marker="add-knob")  # its knobs, by its names
        state.variable_rows[0].update(start=40, upper=90)
        user.find(marker="next").click()
        await user.should_see("temp: Declared goes from 20 to 80; keep the range inside that.")
        assert state.step == 0
        state.variable_rows[0]["upper"] = 60
        user.find(marker="next").click()
        await user.should_see("Declared returns these.")
        await user.should_not_see(marker="add-output")
        # The goal is free: aim yield at a value.
        state.objective_rows[0].update(direction="TARGET", target=0.8)
        state.step = state.reached = 4
        with user.client:
            state.refresh()
        await user.should_see("Your experiment: Declared")
        await user.should_not_see(marker="fit-problem")
        user.find(marker="create").click()
        await user.should_see(marker="runner")
        assert isinstance(state.controller.experiment, Declared)


@pytest.mark.anyio
async def test_an_undeclared_experiment_asks_you_to_check_the_names(ui_states, served):
    served["yours"] = yours.YourExperiment(Rig)
    async with user_simulation(root=app._build_page) as user:
        await user.open("/")
        state = ui_states[-1]
        await user.should_see(marker="add-knob")  # any setup
        state.step = state.reached = 4
        with user.client:
            state.refresh()
        await user.should_see("Rig doesn't say which knobs it reads or what it returns. "
                              "Check it reads x1 and returns y.")


# -- a command as your experiment --


def test_command_flag_makes_a_command_experiment(served, tmp_path):
    app.main(["--no-show", "--command", "./measure.sh {temp} {solvent}", "--timeout", "5"])
    made = served["yours"]
    assert made.name == "measure.sh" and made.source == "./measure.sh {temp} {solvent}"
    assert made.reads == ["solvent", "temp"]
    assert made.problem(SPEC) == ("measure.sh reads solvent and temp, which aren't a knob "
                                  "here; the knobs are x1.")
    experiment = made.make(DECLARED)
    assert experiment.timeout == 5 and experiment.spec is DECLARED


@pytest.mark.parametrize(("argv", "words"), [
    (["--command", "a", "--experiment", "b:c"], "not allowed with argument"),
    (["--timeout", "3"], "--timeout goes with --command"),
    (["--command", "   "], "the command is empty"),
])
def test_command_flag_refusals(served, capsys, argv, words):
    with pytest.raises(SystemExit):
        app.main(["--no-show", *argv])
    assert words in capsys.readouterr().err


@pytest.mark.anyio
async def test_the_start_step_checks_a_commands_names(ui_states, served):
    served["yours"] = yours.command("./measure.sh {x1}")
    async with user_simulation(root=app._build_page) as user:
        await user.open("/")
        state = ui_states[-1]
        state.step = state.reached = 4
        with user.client:
            state.refresh()
        await user.should_see("Your experiment: measure.sh")
        await user.should_see("measure.sh reads x1. Check it prints y.")
        state.variable_rows[0]["name"] = "temp"
        with user.client:
            state.refresh()
        await user.should_see("measure.sh reads x1, which isn't a knob here; the knobs are temp.")
