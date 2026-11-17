"""CommandExperiment: a command run for each design, printing its outputs as JSON."""

import sys
import textwrap

import pytest
import torch

from abbot import Campaign, CommandExperiment, ContinuousDomain, OutputSpec, ProblemSpec
from abbot.problem import EnumeratedDomain

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell scripts")

SPEC = ProblemSpec(
    domain=[ContinuousDomain(name="temp", lower=20, upper=80),
            ContinuousDomain(name="conc", lower=0, upper=0.5)],
    outputs=[OutputSpec(name="yield")],
)


@pytest.fixture
def rig(tmp_path):
    """measure.sh: logs to stderr and stdout, then prints its JSON last."""
    script = tmp_path / "measure.sh"
    script.write_text(textwrap.dedent(f"""\
        #!/bin/sh
        echo "moving to $1 $2" >&2
        echo "settling"
        {sys.executable} -c '
        import json, sys
        t, c = float(sys.argv[1]), float(sys.argv[2])
        print(json.dumps({{"yield": 100 - (t - 55) ** 2 / 10 - (c - 0.2) ** 2 * 100,
                          "cost": 2.5, "stage": t + 0.1}}))
        ' "$1" "$2"
    """))
    script.chmod(0o755)
    return script


def test_runs_the_command_and_reads_its_last_line(rig, tmp_path):
    e = CommandExperiment("./measure.sh {temp} {conc:.3f}", spec=SPEC, cwd=tmp_path)
    design = {"temp": 55.0, "conc": 0.2}
    assert e.args(design) == ["./measure.sh", "55.0", "0.200"]
    outputs, extras = e.run(design)
    assert outputs == {"yield": pytest.approx(100.0)}
    assert extras == {"stage": pytest.approx(55.1)}  # kept with the run, not refused
    assert e.cost(design) == 2.5  # "cost" is the run's cost, asked once
    assert e.cost(design) is None
    assert e.fields == ["conc", "temp"] and e.name == "measure.sh"


def test_a_campaign_steps_through_it(rig, tmp_path):
    torch.manual_seed(0)
    e = CommandExperiment(f"{rig} {{temp}} {{conc}}", spec=SPEC)
    campaign = Campaign(SPEC)
    for _ in range(3):
        campaign.step(e)
    records = campaign.history.records
    assert len(records) == 3 and not any(r.fault for r in records)
    assert all(r.cost == 2.5 for r in records)
    assert all(set(r.aux) == {"stage"} for r in records)


def test_values_are_one_argument_each_and_stdin_has_the_design():
    spec = ProblemSpec(domain=[EnumeratedDomain(name="solvent", values=["a b; rm -rf x", "c"])],
                       outputs=[OutputSpec(name="y")])
    echo = (f"{sys.executable} -c \"import json,sys; d=json.load(sys.stdin); "
            "print(json.dumps({{'y': len(sys.argv[1]) + len(d['solvent'])}}))\" {solvent}")
    assert CommandExperiment(echo, spec=spec).run({"solvent": "a b; rm -rf x"}) == {"y": 26}
    shell = CommandExperiment("echo {solvent} | cat", shell=True)
    assert shell.args({"solvent": "a b; rm"}) == "echo 'a b; rm' | cat"


def test_a_number_or_a_list_is_taken_as_is():
    assert CommandExperiment("echo 3.5").run({}) == 3.5
    assert CommandExperiment("echo [1, 2]").run({}) == [1, 2]


@pytest.mark.parametrize(("command", "words"), [
    ("sh -c 'echo broke >&2; exit 3'", "sh exited with code 3: broke"),
    ("sh -c 'echo >&2 nothing here'", "sh printed nothing: nothing here"),
    ("echo hello", "echo's last line isn't JSON: 'hello'"),
    ("./no-such-program {temp}", "no-such-program: no such program"),
    ("sleep 5", "sleep ran past 0.2 s"),
])
def test_a_failed_run_says_why(command, words):
    with pytest.raises(RuntimeError, match=words.replace("(", r"\(")):
        CommandExperiment(command, timeout=0.2).run({"temp": 1.0})


def test_a_failed_run_is_a_fault_and_retried():
    campaign = Campaign(SPEC, max_retries=1)
    campaign.step(CommandExperiment("sh -c 'exit 1'"))
    assert campaign.history.records[-1].fault
    assert "sh exited with code 1" in campaign.history.records[-1].error
    campaign.step(CommandExperiment("sh -c 'exit 1'"))  # the retry, then a halt
    first, retry = campaign.history.records
    assert retry.retry_of is not None and torch.equal(retry.candidate, first.candidate)
    assert campaign.halted


def test_names_are_checked_against_the_spec():
    with pytest.raises(ValueError, match=r"\{tmp\} isn't a variable; the variables are temp, conc"):
        CommandExperiment("./measure.sh {tmp}", spec=SPEC)
    with pytest.raises(KeyError, match=r"\{tmp\} isn't a variable; the design has temp"):
        CommandExperiment("./measure.sh {tmp}").args({"temp": 1.0})
    # Literal braces are doubled, as in str.format.
    assert CommandExperiment("echo {{x}} {temp}").args({"temp": 1.0}) == ["echo", "{x}", "1.0"]
    assert CommandExperiment("""echo '{{"y": {temp}}}'""").run({"temp": 2.0}) == {"y": 2.0}
