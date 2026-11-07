"""``abbot-mcp setup`` and ``abbot-mcp start``: connect Claude Code to abbot.

``setup`` registers the MCP server with Claude Code (user scope, so it works from any
folder) and installs the abbot skill in ``~/.claude/skills/abbot``. ``start`` runs
setup if it's missing, then opens Claude Code in the current folder on the skill
(``/abbot``), which plans the campaign with the person. It doesn't open on the
server's ``plan_campaign`` prompt: Claude Code submits the first message before MCP
servers connect, so that command doesn't exist yet; skills load from disk first. Other MCP clients don't need either: give them the
``abbot-mcp --dir ...`` command.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

SERVER = "abbot"
DEFAULT_DIR = "~/abbot-campaigns"


def skill_source() -> Path:
    """The skill shipped with abbot: in the wheel, or in a source checkout."""
    packaged = Path(__file__).with_name("skill") / "SKILL.md"
    if packaged.exists():
        return packaged
    return Path(__file__).resolve().parents[3] / "skills" / "abbot" / "SKILL.md"


def skill_target() -> Path:
    return Path.home() / ".claude" / "skills" / "abbot" / "SKILL.md"


def _need_mcp() -> None:
    if importlib.util.find_spec("mcp") is None:
        raise SystemExit('The MCP server needs the mcp extra: pip install "abbot[mcp]"')


def _claude() -> str:
    claude = shutil.which("claude")
    if claude is None:
        raise SystemExit(
            "Claude Code's `claude` command isn't on the PATH. Install it "
            "(https://claude.com/claude-code), then run this again."
        )
    return claude


def _registered(claude: str) -> bool:
    found = subprocess.run(
        [claude, "mcp", "get", SERVER], capture_output=True, text=True, check=False
    )
    return found.returncode == 0


def _confirm(question: str, yes: bool, ask: Callable[[str], str]) -> bool:
    if yes:
        return True
    if not sys.stdin.isatty():
        raise SystemExit("Not asking without a terminal; run again with --yes.")
    return ask(f"{question} [Y/n] ").strip().lower() in ("", "y", "yes")


def setup(directory: str = DEFAULT_DIR, yes: bool = False, ask=input) -> bool:
    """Register the server and install the skill, asking first. True when both
    are in place afterwards."""
    _need_mcp()
    claude = _claude()
    folder = Path(directory).expanduser().resolve()
    server = [sys.executable, "-m", "abbot.agent", "--dir", str(folder)]
    register = not _registered(claude)
    source, target = skill_source(), skill_target()
    install = not target.exists() or target.read_bytes() != source.read_bytes()
    replace = install and target.exists()

    if not register and not install:
        print("abbot is already set up for Claude Code.")
        return True
    print("This will:")
    if register:
        print(f"  - register the abbot MCP server with Claude Code (user scope),\n"
              f"    saving campaigns in {folder}")
    else:
        print(f"  - keep the registered '{SERVER}' server "
              f"(`claude mcp get {SERVER}` shows it)")
    if install:
        print(f"  - {'replace' if replace else 'install'} the abbot skill at {target}"
              + (" (it differs from this version's)" if replace else ""))
    if not _confirm("Go ahead?", yes, ask):
        print("Nothing changed.")
        return False
    sys.stdout.flush()
    if register:
        subprocess.run([claude, "mcp", "add", "--scope", "user", SERVER, "--", *server],
                       check=True)
    if install:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        print(f"Installed the skill at {target}")
    return True


def start(task: str = "", directory: str = DEFAULT_DIR, yes: bool = False,
          ask=input) -> int:
    """Open Claude Code here on the abbot skill, setting up first if needed."""
    claude = _claude()
    if (not _registered(claude) or not skill_target().exists()) and not setup(
        directory, yes, ask
    ):
        return 1
    prompt = f"/abbot {task}".strip()
    sys.stdout.flush()
    return subprocess.call([claude, prompt])


def main(argv: list[str] | None = None) -> None:
    """Entry point: ``abbot-mcp [--dir DIR]`` serves over stdio; ``abbot-mcp setup``
    and ``abbot-mcp start`` connect Claude Code."""
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("setup", "start"):
        _launch(argv)
        return
    parser = argparse.ArgumentParser(
        prog="abbot-mcp",
        description="Serve abbot campaigns to an MCP client over stdio. "
        "`abbot-mcp setup` connects Claude Code; `abbot-mcp start` opens it to "
        "plan a campaign.",
    )
    parser.add_argument(
        "--dir", default="abbot-campaigns",
        help="where campaigns are saved (default: ./abbot-campaigns)",
    )
    args = parser.parse_args(argv)
    _need_mcp()
    from abbot.agent.server import build_server

    build_server(args.dir).run("stdio")


def _launch(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="abbot-mcp")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--dir", default=DEFAULT_DIR,
        help=f"where campaigns are saved (default: {DEFAULT_DIR})",
    )
    common.add_argument("--yes", action="store_true", help="don't ask before changes")
    sub.add_parser(
        "setup", parents=[common],
        help="register the server with Claude Code and install the abbot skill",
    )
    p = sub.add_parser(
        "start", parents=[common],
        help="open Claude Code here to plan a campaign (sets up first if needed)",
    )
    p.add_argument("task", nargs="?", default="", help="what you want to do, briefly")
    args = parser.parse_args(argv)
    if args.command == "setup":
        sys.exit(0 if setup(args.dir, args.yes) else 1)
    sys.exit(start(args.task, args.dir, args.yes))
