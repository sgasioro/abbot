# Agent sessions on a simulated rig

These scripts check how a real agent uses abbot's MCP server (`abbot-mcp`). The agent
gets a plain description of a lab task and a `measure.py` script, and nothing else.

- `rig.py`: a simulated thin-film anneal rig behind a local HTTP endpoint. Its
  docstring says what a person would tell the agent and what they wouldn't: a
  furnace trip inside the stated range, a 2 mm stage travel limit, noise that scales
  with the signal.
- `measure.py`: the thin client the agent runs. It posts one anneal to the rig and
  prints the film's conductivity and roughness.
- `run.py`: runs one headless `claude -p` turn per call. Each call starts the rig,
  runs the turn in a fresh lab folder, then stops the rig. The agent may call
  abbot's tools and `python3 measure.py`. It can't read, grep or glob files, so it
  never sees the simulator.
- `show.py`: renders a session as Markdown.

```
python benchmarks/agent/run.py start vague /tmp/agent-runs/vague-1
python benchmarks/agent/run.py reply /tmp/agent-runs/vague-1 "The range is 150 to 450 C ..."
python benchmarks/agent/show.py /tmp/agent-runs/vague-1 > vague-1.md
```

There are two scenarios:
- `plain` describes the rig fully and asks for about 15 runs without confirmation.
- `vague` only says what the task is and names the script; a good agent asks before
  it builds a spec.

Pass `--prompt` to try any other wording.

Keep run folders outside the repository. You need the `claude` CLI, logged in.
Sessions cost real tokens, so run one at a time.

These are qualitative checks, not a benchmark with scores. Read the transcripts for:
- where the spec's numbers came from (the person, or the examples in `spec_guide`);
- whether the agent stops to ask when a decision belongs to the person;
- how it handles failures and halts.
