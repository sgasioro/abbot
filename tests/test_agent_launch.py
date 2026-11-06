"""``abbot-mcp setup`` / ``start`` against a stand-in ``claude`` command."""

import json
import sys

import pytest

from abbot.agent import launch

FAKE_CLAUDE = """\
#!{python}
import json, pathlib, sys
log = pathlib.Path({log!r})
registered = log.with_name("registered")
with log.open("a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1:3] == ["mcp", "get"]:
    sys.exit(0 if registered.exists() else 1)
if sys.argv[1:3] == ["mcp", "add"]:
    registered.write_text("yes")
"""


@pytest.fixture
def claude(tmp_path, monkeypatch):
    """A fake ``claude`` on the PATH and a fresh home; returns its call log."""
    bin_dir, home = tmp_path / "bin", tmp_path / "home"
    bin_dir.mkdir()
    home.mkdir()
    log = tmp_path / "calls.jsonl"
    exe = bin_dir / "claude"
    exe.write_text(FAKE_CLAUDE.format(python=sys.executable, log=str(log)))
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.setenv("HOME", str(home))

    def calls():
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text().splitlines()]

    return calls


def test_setup_registers_the_server_and_installs_the_skill(claude, tmp_path):
    assert launch.setup(str(tmp_path / "camps"), yes=True)
    (add,) = [c for c in claude() if c[:2] == ["mcp", "add"]]
    assert add[2:6] == ["--scope", "user", "abbot", "--"]
    assert add[6:] == [sys.executable, "-m", "abbot.agent", "--dir",
                       str((tmp_path / "camps").resolve())]
    assert launch.skill_target().read_text() == launch.skill_source().read_text()


def test_setup_again_changes_nothing(claude, tmp_path, capsys):
    launch.setup(str(tmp_path / "camps"), yes=True)
    before = len(claude())
    assert launch.setup(str(tmp_path / "camps"), yes=True)
    assert "already set up" in capsys.readouterr().out
    assert not [c for c in claude()[before:] if c[:2] == ["mcp", "add"]]


def test_an_edited_skill_is_kept_unless_the_person_agrees(claude, tmp_path,
                                                          monkeypatch):
    launch.setup(str(tmp_path / "camps"), yes=True)
    launch.skill_target().write_text("my notes")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    assert not launch.setup(str(tmp_path / "camps"), ask=lambda q: "n")
    assert launch.skill_target().read_text() == "my notes"
    assert launch.setup(str(tmp_path / "camps"), ask=lambda q: "")
    assert launch.skill_target().read_text() == launch.skill_source().read_text()


def test_setup_without_a_terminal_needs_yes(claude, tmp_path, monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(SystemExit, match="--yes"):
        launch.setup(str(tmp_path / "camps"))
    assert launch.skill_target().exists() is False


def test_start_sets_up_then_opens_the_planning_prompt(claude, tmp_path):
    with pytest.raises(SystemExit) as done:
        launch.main(["start", "anneal films", "--dir", str(tmp_path / "c"), "--yes"])
    assert done.value.code == 0
    assert claude()[-1] == ["/mcp__abbot__plan_campaign anneal films"]
    assert ["mcp", "get", "abbot"] in claude()


def test_without_claude_the_message_says_how_to_get_it(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(SystemExit, match="isn't on the PATH"):
        launch.setup(str(tmp_path / "camps"), yes=True)


def test_the_repository_skill_is_what_ships():
    assert launch.skill_source().read_text().startswith("---\nname: abbot\n")
