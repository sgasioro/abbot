"""Try abbot's MCP server with a real agent on a simulated anneal rig.

Each turn starts the rig (``rig.py``), runs one headless ``claude -p`` session in a
fresh lab folder that holds only ``measure.py``, and stops the rig. The agent can
call abbot's tools and ``python3 measure.py``; it can't read files, so it learns
the rig only from the prompt, the client's ``--help`` and its replies. The server
runs from this checkout's ``src``.

    python benchmarks/agent/run.py start plain /tmp/agent-runs/plain-1
    python benchmarks/agent/run.py start vague /tmp/agent-runs/vague-1
    python benchmarks/agent/run.py start vague /tmp/agent-runs/vague-2 --skill
    python benchmarks/agent/run.py reply /tmp/agent-runs/vague-1 "150 to 450 C, ..."
    python benchmarks/agent/show.py /tmp/agent-runs/vague-1 > vague-1.md

Keep the run folder outside this repository, so nothing the agent can reach holds
the rig's answers. ``--skill`` puts the repository's abbot skill
(``skills/abbot``) in the lab folder's ``.claude/skills`` and lets the agent use it.
Each turn writes ``turn-N.jsonl`` (the stream-json transcript) and ``turn-N.err``;
``run.json`` records the prompts, the model and the abbot revision. Needs the ``claude`` CLI, logged in, and ``abbot[mcp]``'s dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

SCENARIOS = {
    "plain": (
        "I'm annealing thin films and want to find the conditions that give the "
        "most conductive film. I can set the anneal temperature from 150 to 450 C, "
        "the anneal time from 10 s to 1 hour, the stage position from 0 to 50 mm "
        "(the sample has a thickness gradient along it), and the substrate: Si, "
        "sapphire or glass. The stage can only move 2 mm between runs, and it's at "
        "10 mm now. Each run gives conductivity and roughness; roughness has to "
        "stay under 3 nm. To run one: python3 measure.py --anneal-temp C --time "
        "SECONDS --position MM --substrate NAME. Use the abbot tools to plan and "
        "run about 15 experiments, then tell me the best conditions. You don't "
        "need to confirm the setup with me; if something needs my decision, stop "
        "and tell me."
    ),
    "vague": (
        "I'm annealing thin films and want help choosing which runs to do. "
        "The rig script is measure.py."
    ),
}

ALLOWED = ["mcp__abbot", "Bash(python3 measure.py:*)"]
DISALLOWED = ["Read", "Grep", "Glob"]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_rig(out: Path, seed: int | None) -> tuple[subprocess.Popen, int]:
    port = _free_port()
    cmd = [sys.executable, str(HERE / "rig.py"), "--port", str(port),
           "--state", str(out / "rig-state.json")]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    rig = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.1).close()
            return rig, port
        except OSError:
            time.sleep(0.05)
    rig.kill()
    raise RuntimeError("the rig didn't start")


def _session_id(transcript: Path) -> str:
    sid = None
    for line in transcript.read_text().splitlines():
        sid = json.loads(line).get("session_id", sid)
    if sid is None:
        raise RuntimeError(f"no session id in {transcript}")
    return sid


def _turn(out: Path, prompt: str, record: dict, resume: str | None = None) -> Path:
    n = len(record["turns"]) + 1
    transcript = out / f"turn-{n}.jsonl"
    cmd = ["claude", "-p", prompt, "--mcp-config", str(out / "mcp.json"),
           "--strict-mcp-config",
           "--allowedTools", *ALLOWED, *(["Skill"] if record.get("skill") else []),
           "--disallowedTools", *DISALLOWED,
           "--output-format", "stream-json", "--verbose"]
    if record.get("model"):
        cmd += ["--model", record["model"]]
    if resume:
        cmd += ["--resume", resume]
    seed = record.get("seed")
    rig, port = _start_rig(out, None if seed is None else seed + n)
    env = {**os.environ, "ANNEAL_RIG_URL": f"http://127.0.0.1:{port}/run"}
    try:
        with transcript.open("w") as stdout, (out / f"turn-{n}.err").open("w") as err:
            code = subprocess.call(cmd, cwd=out / "lab", env=env,
                                   stdout=stdout, stderr=err)
    finally:
        rig.terminate()
        rig.wait()
    record["turns"].append({"prompt": prompt, "exit": code})
    (out / "run.json").write_text(json.dumps(record, indent=1))
    return transcript


def _summary(transcript: Path) -> str:
    text, result = "", ""
    for line in transcript.read_text().splitlines():
        msg = json.loads(line)
        if msg.get("type") == "assistant":
            parts = [c["text"] for c in msg["message"]["content"] if c["type"] == "text"]
            if parts:
                text = "\n".join(parts)
        elif msg.get("type") == "result":
            result = (f"[{msg.get('subtype')}, {msg.get('num_turns')} turns, "
                      f"${msg.get('total_cost_usd', 0):.2f}, "
                      f"{msg.get('duration_ms', 0) / 1000:.0f} s]")
    return f"{text}\n\n{result}".strip()


def start(args) -> None:
    out = args.out.resolve()
    if out.is_relative_to(REPO):
        sys.exit("keep the run folder outside the repository")
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} isn't empty")
    prompt = args.prompt or SCENARIOS[args.scenario]
    lab = out / "lab"
    lab.mkdir(parents=True)
    shutil.copy(HERE / "measure.py", lab / "measure.py")
    if args.skill:
        shutil.copytree(REPO / "skills" / "abbot", lab / ".claude" / "skills" / "abbot")
    server = {"command": sys.executable,
              "args": ["-m", "abbot.agent", "--dir", str(lab / "campaigns")],
              "env": {"PYTHONPATH": str(REPO / "src")}}
    (out / "mcp.json").write_text(json.dumps({"mcpServers": {"abbot": server}}, indent=1))
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                         capture_output=True, text=True, check=False).stdout.strip()
    record = {"scenario": args.scenario, "model": args.model, "seed": args.seed,
              "skill": args.skill, "abbot": rev, "turns": []}
    print(_summary(_turn(out, prompt, record)))


def reply(args) -> None:
    out = args.out.resolve()
    record = json.loads((out / "run.json").read_text())
    last = out / f"turn-{len(record['turns'])}.jsonl"
    print(_summary(_turn(out, args.message, record, resume=_session_id(last))))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("start", help="start a session")
    p.add_argument("scenario", choices=sorted(SCENARIOS))
    p.add_argument("out", type=Path)
    p.add_argument("--prompt", help="use this prompt instead of the scenario's")
    p.add_argument("--model", help="passed to claude --model")
    p.add_argument("--seed", type=int, help="seed the rig's noise")
    p.add_argument("--skill", action="store_true", help="give the agent the abbot skill")
    p.set_defaults(func=start)
    p = sub.add_parser("reply", help="answer the agent and continue the session")
    p.add_argument("out", type=Path)
    p.add_argument("message")
    p.set_defaults(func=reply)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
