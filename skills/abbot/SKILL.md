---
name: abbot
description: Plan and run an experiment campaign with abbot, which chooses the next experiment to run (Bayesian optimization or finding a region of settings). Use when someone wants help choosing which experiments, runs, samples or simulations to do next, tuning a process or an apparatus, or mapping where a measurement meets a specification.
---

# Choosing experiments with abbot

abbot's MCP server (`abbot-mcp`) holds the campaigns and decides what to run next.
This skill gets the server connected, then hands off to its guide.

## 1. Connect the server

If the `abbot` MCP tools (`spec_guide`, `create_campaign`, `suggest`, ...) are
available, skip to step 2.

Otherwise, set them up with the person's agreement. Each command changes their
setup, so say what it does before you run it.
1. Check for abbot: `python -c "import abbot.agent.server"`. If it's missing,
   install it into their environment: `pip install "abbot[mcp]"`.
2. Run `abbot-mcp setup` (or `python -m abbot.agent setup`). It registers the
   server with Claude Code for all their folders, saving campaigns in
   `~/abbot-campaigns` (`--dir` to choose another), and installs this skill. It
   asks before changing anything; pass `--yes` only once they've agreed.
3. Tell them to restart Claude Code (or run `/mcp`) so the tools load. Next time,
   `abbot-mcp start` opens Claude Code straight on the planning questions.

## 2. Plan the campaign with the person

Call `spec_guide`. Before you write a spec, work through its planning questions
with the person: knobs and the ranges the apparatus can reach, what's measured, the
goal, limits and failure modes, budget, earlier results, and who runs each
experiment. Ask about everything that's missing in one message. Don't take numbers
from the guide's examples. Show the plan in plain words, and create the campaign
once they agree. If they asked you to go ahead without confirming, still ask about
anything you'd otherwise have to guess.

The person can also start this interview with the server's `plan_campaign` prompt
(`/mcp__abbot__plan_campaign` in Claude Code).

## 3. Run the loop

Repeat: `suggest`, run the design (or ask the person to run it), then
`report_result` with the raw measured values (never negate them). If a run fails,
use `report_failure`. If a value was reported wrong, fix it with `correct_result`
(don't start a new campaign). When the person wants to see where things stand, use
`plot` and give them the file's path. When the campaign halts, or a failure
suggests the spec is wrong (an interlock inside a range, say), stop and ask the
person. Don't resolve it yourself. To change the spec, create a new campaign with `results_from=<old name>`.
Finish with `recommend`, and say how sure the model is (`check_model`).
