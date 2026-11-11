"""Campaigns an agent drives by name, saved in a directory so they survive restarts.

The MCP server (:mod:`abbot.agent.server`) is a thin layer over :class:`Workspace`;
everything here takes and returns plain JSON-able values (dicts, lists, strings),
so it is usable and testable without an MCP client.

Each campaign lives in ``<root>/<name>/``: ``campaign.json`` holds how to rebuild it
(the spec, the goal, the strategy and model choice, the starting designs), and
``store/`` is the campaign's own :class:`~abbot.history.DataStore`, so every result,
outstanding design and halt is saved as it happens. Reopening a name resumes it.

The agent's server and the UI (``python -m abbot.ui``) share the same directory by
default (``~/abbot-campaigns``). A process that opens a campaign holds it (a
``.lock`` file with its process id) until it closes it or exits, so two processes
never write the same campaign; the other one is told who has it.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import torch

from abbot import __version__
from abbot import io as abbot_io
from abbot.history import DataStore, Record
from abbot.policy import BAXPolicy
from abbot.problem import ProblemSpec
from abbot.session import Campaign
from abbot.ui.builder import (
    BAX_STRATEGY_LABELS,
    MODEL_LABELS,
    build_constructor,
    build_goal,
)

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

# What an agent reads before writing a spec. Kept short; the schema has the rest.
PLANNING_GUIDE = """\
Setting up a campaign with the person. Find out, from what they've said and by
asking:
1. What they can set: each knob, its units, and the range the apparatus can
   actually reach (not only the interesting part). Whether a knob spans orders of
   magnitude. Choices that aren't numbers (a material, a mode).
2. What each run measures, with units. Which outputs to maximize or minimize, which
   only have to stay within limits (and the limits), which are only recorded.
3. The goal: the single best setting, or a region (every setting that meets a
   specification, everything above a threshold, the top few).
4. Practical limits: how far a knob can move between runs and where it is now;
   settings known to fail or be unsafe. Keep unsafe settings out of the ranges
   rather than letting abbot find them.
5. What a failed run looks like, so you can tell a failure from a bad result.
6. The budget: how many runs, and whether runs differ in cost (a cheap simulation
   next to the real experiment).
7. Results they already have; add them with add_results after creating.
8. Who runs each experiment: you, through a script or command, or the person by
   hand (then you suggest, wait, and report what they tell you).
Ask about everything that's missing in one message rather than one question at a
time, and skip what you already know. Don't fill gaps with numbers from the
examples in spec_guide or from probing the apparatus at its extremes; if the person
doesn't know, propose a value, say it's a guess, and ask. Then show the plan in
plain words (knobs and ranges, outputs and limits, goal, budget) and create the
campaign once the person agrees.
"""

SPEC_GUIDE = """\
A campaign is a spec (what you can set and what you measure), optionally a goal
(find a region rather than an optimum), and the measurements reported back.
The numbers in the examples below are illustrations. Take ranges, limits and
directions from the person or their setup, and ask when they're missing.

Spec (JSON):
  {"domain": [
     {"kind": "continuous", "name": "temp", "lower": 20, "upper": 80},
     {"kind": "continuous", "name": "conc", "lower": 0.001, "upper": 1, "log": true},
     {"kind": "continuous", "name": "angle", "lower": 0, "upper": 90, "max_step": 2},
     {"kind": "enumerated", "name": "solvent", "values": ["water", "ethanol"]}],
   "outputs": [
     {"name": "yield", "direction": "MAXIMIZE"},
     {"name": "purity", "direction": "OBSERVE", "min_val": 0.9}],
   "sources": [{"domain": {"kind": "enumerated", "name": "code",
                           "values": ["sim", "experiment"]},
                "target": "experiment"}]}
- direction: MAXIMIZE or MINIMIZE (optimized), or OBSERVE (modeled, not optimized).
  Report raw values; never negate a minimized output.
- min_val / max_val on an output keep it in range (a constraint).
- log: true for quantities spanning orders of magnitude.
- max_step: a travel limit between consecutive runs; pass initial_designs with the
  current setting as the first design.
- sources (optional): runs at different fidelities or from different codes; abbot
  chooses which to run and recommends at the target. Report each run's cost if known.

Goal (optional; without one, abbot optimizes the MAXIMIZE/MINIMIZE outputs). Mark
the outputs the goal reads OBSERVE. Goal kinds:
  {"kind": "band", "bounds": {"yield": [0.4, 0.6]}}   every setting within the bands
  {"kind": "level", "output": "yield", "threshold": 0.8, "upper": false}
                                                       where yield >= 0.8 (upper: <=)
  {"kind": "topk", "output": "yield", "k": 5, "largest": true}
  {"kind": "percentile", "percentile": 90}             top 10% of a single output
strategy (with a goal): SwitchBAX (balanced, default), MeanBAX (fast), InfoBAX
(thorough), US (explore only).
model: "default" (independent GPs) or "correlated" (one GP over correlated outputs).

Loop: suggest -> run the design -> report_result (or report_failure). A value
reported wrong is fixed with correct_result (by the run's id); plot draws the
model, the progress or predicted-vs-measured for the person. A failed design
is retried; after repeated failures the campaign halts until resolve_halt is called
with "retry", "exclude" or "bad_region" (the design itself gives nothing usable).
The person can also follow a campaign in the abbot UI (python -m abbot.ui, which
saves in the same folder): call close_campaign first, since one process at a time
holds a campaign. A campaign's spec is fixed once created. To change it (narrow a range the apparatus
can't reach, add an output), create a new campaign with results_from=<old name>:
the runs that fit the new spec are copied in.
"""


class Workspace:
    """Named campaigns under ``root``, opened on demand and kept open."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._open: dict[str, Campaign] = {}

    # -- campaigns --

    def create(
        self,
        name: str,
        spec: dict,
        goal: dict | None = None,
        strategy: str = "SwitchBAX",
        model: str = "default",
        initial_designs: list[dict] | None = None,
        max_retries: int = 1,
        results_from: str | None = None,
    ) -> dict:
        """Create and save a campaign. Refuses a name that already exists.

        ``results_from`` copies another campaign's results into the new one (say,
        after narrowing a range): every measured run whose design and outputs fit
        the new spec. The reply counts what was copied and skipped.
        """
        if not _NAME.match(name):
            raise ValueError(
                f"Campaign name {name!r}: use letters, digits, '_', '-' or '.', "
                "starting with a letter or digit (at most 64 characters)."
            )
        folder = self.root / name
        if (folder / "campaign.json").exists():
            raise ValueError(f"A campaign named {name!r} already exists.")
        parsed = ProblemSpec.model_validate(spec)
        if strategy not in BAX_STRATEGY_LABELS:
            raise ValueError(f"strategy must be one of {list(BAX_STRATEGY_LABELS)}.")
        if model not in MODEL_LABELS:
            raise ValueError(f"model must be one of {list(MODEL_LABELS)}.")
        config = {
            "spec": json.loads(abbot_io.spec_to_json(parsed)),
            "goal": goal,
            "strategy": strategy,
            "model": model,
            "initial_designs": initial_designs,
            "max_retries": max_retries,
            "abbot_version": __version__,
        }
        source = self._campaign(results_from) if results_from else None
        campaign = self._build(folder, config)  # validates goal, model, designs
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "campaign.json").write_text(json.dumps(config, indent=1))
        _lock(folder)
        self._open[name] = campaign
        extra: dict[str, Any] = {"policy": type(campaign.policy).__name__}
        if source is not None:
            copied, skipped = _copy_results(source, campaign)
            extra.update(copied=copied, skipped=skipped)
        return {**self._brief(name), **extra}

    def campaigns(self) -> list[dict]:
        """Every saved campaign, with its size and whether it needs attention.
        Listing doesn't open them: one another process has open is listed with
        ``"open_elsewhere"`` (its process id)."""
        out = []
        for path in sorted(self.root.glob("*/campaign.json")):
            name = path.parent.name
            try:
                c = self._open.get(name) or self._build(
                    path.parent, json.loads(path.read_text()))
            except Exception as exc:  # noqa: BLE001 - list the rest, name the broken one
                out.append({"name": name, "error": str(exc)})
                continue
            item = {
                "name": name,
                "runs": len(c.history),
                "outstanding": len(c.outstanding()),
                "halted": c.halted,
                "stopped": c.stopped,
            }
            holder = _holder(path.parent)
            if holder is not None:
                item["open_elsewhere"] = holder
            out.append(item)
        return out

    def open(self, name: str) -> Campaign:
        """The campaign named ``name``, resumed from its folder, and held by this
        process until :meth:`close`. Raises if another process holds it."""
        return self._campaign(name)

    def config(self, name: str) -> dict:
        """The campaign's ``campaign.json``: how it was set up."""
        self._campaign(name)
        return json.loads((self.root / name / "campaign.json").read_text())

    def close(self, name: str) -> None:
        """Let go of a campaign, so another process (the UI, the agent) can open it."""
        if self._open.pop(name, None) is not None:
            _unlock(self.root / name)

    def free_name(self, wanted: str) -> str:
        """A campaign name from free text that isn't taken yet: ``"My campaign"``
        becomes ``"My-campaign"``, then ``"My-campaign-2"``."""
        base = re.sub(r"[^A-Za-z0-9_.-]+", "-", wanted.strip()).strip("-._")[:56]
        base = base or "campaign"
        name, n = base, 1
        while (self.root / name / "campaign.json").exists():
            n += 1
            name = f"{base}-{n}"
        return name

    # -- the loop --

    def suggest(self, name: str, n: int = 1, fixed: dict | None = None) -> list[dict]:
        """``n`` designs to run next, each with the id to report its result under."""
        c = self._campaign(name)
        if c.halted:
            raise ValueError(f"Campaign {name!r} is halted: {c.halted} Call resolve_halt.")
        return [{"id": cand.id, "design": dict(cand)} for cand in c.suggest(n, fixed=fixed)]

    def report_result(
        self, name: str, id: str, outputs: dict[str, float], cost: float | None = None
    ) -> dict:
        """The measured outputs (raw values, by name) for an outstanding design."""
        c = self._campaign(name)
        c.ingest(outputs, id=id, costs=cost)
        return self._brief(name)

    def report_failure(self, name: str, id: str, error: str) -> dict:
        """An outstanding design couldn't be run; it is retried as usual."""
        self._campaign(name).fail(id, error)
        return self._brief(name)

    def cancel(self, name: str, id: str) -> dict:
        """Withdraw an outstanding design: nothing recorded, no retry."""
        self._campaign(name).cancel(id)
        return self._brief(name)

    def add_results(self, name: str, results: list[dict]) -> dict:
        """Results measured outside the loop (earlier data, a run the lab chose).

        Each item is ``{"design": {variable: value}, "outputs": {output: value}}``,
        optionally with ``"cost"``.
        """
        c = self._campaign(name)
        for item in results:
            c.record(item["design"], item["outputs"], cost=item.get("cost"))
        return self._brief(name)

    def correct_result(
        self,
        name: str,
        run: str | int,
        outputs: dict[str, float],
        reason: str | None = None,
    ) -> dict:
        """Correct a reported run's measured values (a value typed wrong). ``run``
        is its id (as ``suggest`` gave it) or its index in the history; ``outputs``
        names only the outputs to change. The old values are kept on the run."""
        c = self._campaign(name)
        step = c._find_run(run)
        before = _record_dict(c.history.records[step], c.spec)["outputs"]
        c.correct(step, outputs, reason=reason)
        out = self._brief(name)
        out["corrected"] = {
            "run": run, "was": before,
            "now": _record_dict(c.history.records[step], c.spec)["outputs"],
        }
        if out["outstanding"]:
            out["note"] = (
                "The outstanding designs were chosen before this correction; cancel "
                "them and suggest again to use it."
            )
        return out

    def resolve_halt(self, name: str, answer: str, reason: str | None = None) -> dict:
        """Answer a halt: ``"retry"``, ``"exclude"`` or ``"bad_region"``."""
        self._campaign(name).resolve(answer, reason=reason)
        return self._brief(name)

    # -- reading --

    def status(self, name: str) -> dict:
        """Where the campaign stands: runs, outstanding designs, halt, recommendation."""
        c = self._campaign(name)
        spec = c.spec
        out = self._brief(name)
        out["policy"] = type(c.policy).__name__
        if any(r.outputs is not None for r in c.history.records):
            out["recommended"] = self.recommend(name)
        out["variables"] = spec.variable_names
        out["outputs"] = spec.output_names
        return out

    def recommend(self, name: str) -> list[dict]:
        """The recommended design(s): one, or the Pareto set for several objectives."""
        c = self._campaign(name)
        result = c.result()
        recs = []
        for k, i in enumerate(result.recommended):
            item = _record_dict(c.history.records[i], c.spec)
            for key in ("run_at_target", "p_feasible", "p_usable"):
                values = getattr(result, key)
                if values:
                    item[key] = values[k]
            recs.append(item)
        return recs

    def check(self, name: str) -> str:
        """``campaign.check()``: does the model fit, and what looks off."""
        return str(self._campaign(name).check())

    def plot(
        self,
        name: str,
        kind: str = "model",
        output: str | list[str] | None = None,
        variables: list[str] | None = None,
        at: dict | None = None,
    ) -> Path:
        """Draw the campaign and save it as a PNG in its folder (``plots/``);
        returns the file's path. ``kind`` is ``"model"`` (what the model predicts,
        with the runs and the goal), ``"progress"`` or ``"check"`` (each run's
        leave-one-out error); see :mod:`abbot.plots` for ``output``,
        ``variables`` and ``at``."""
        from abbot import plots

        c = self._campaign(name)
        if kind == "model":
            fig = plots.model(c, output=output, variables=variables, at=at)
        elif kind == "progress":
            fig = plots.progress(c)
        elif kind == "check":
            fig = plots.check(c)
        else:
            raise ValueError(f"kind must be 'model', 'progress' or 'check', not {kind!r}")
        parts = [kind, f"{len(c.history):03d}runs"]
        for v in ([output] if isinstance(output, str) else output or []) + list(variables or []):
            parts.append(re.sub(r"[^A-Za-z0-9_.-]", "_", str(v)))
        if at:
            parts.append("at")
        folder = self.root / name / "plots"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / ("-".join(parts) + ".png")
        fig.savefig(path, dpi=110)
        import matplotlib.pyplot as plt

        plt.close(fig)
        return path

    def history(self, name: str) -> str:
        """The spec and every result as the self-describing CSV the UI also reads."""
        c = self._campaign(name)
        return abbot_io.spec_history_to_csv(c.spec, c.history)

    # -- internals --

    def _brief(self, name: str) -> dict:
        """The short state the loop's calls return: no model fit, nothing repeated
        that the caller already knows. ``status`` adds the recommendation."""
        c = self._campaign(name)
        out: dict[str, Any] = {
            "name": name,
            "runs": len(c.history),
            "failed_runs": sum(bool(r.fault) for r in c.history.records),
            "outstanding": [{"id": cand.id, "design": dict(cand)} for cand in c.outstanding()],
            "halted": c.halted,
        }
        if c.halted:
            out["halted_on"] = c.halted_on
            out["resolve_with"] = ["retry", "exclude", "bad_region"]
        if c.stopped:
            out["stopped"] = c.stopped
        if any(r.cost is not None for r in c.history.records):
            out["cost_spent"] = c.history.cost_spent
        return out

    def _campaign(self, name: str) -> Campaign:
        if name in self._open:
            return self._open[name]
        path = self.root / name / "campaign.json"
        if not _NAME.match(name) or not path.exists():
            names = [p.parent.name for p in sorted(self.root.glob("*/campaign.json"))]
            raise ValueError(f"No campaign named {name!r}. Campaigns: {names}.")
        _lock(path.parent)
        try:
            campaign = self._build(path.parent, json.loads(path.read_text()))
        except BaseException:
            _unlock(path.parent)
            raise
        self._open[name] = campaign
        return campaign

    @staticmethod
    def _build(folder: Path, config: dict) -> Campaign:
        spec = ProblemSpec.model_validate(config["spec"])
        policy = None
        if config.get("goal"):
            goal = build_goal(_goal_row(config["goal"]), spec)
            policy = BAXPolicy(goal, strategy=config.get("strategy", "SwitchBAX"))
        return Campaign(
            spec,
            policy,
            build_constructor(spec, config.get("model", "default")),
            store=DataStore(folder / "store"),
            initial_designs=config.get("initial_designs"),
            max_retries=config.get("max_retries", 1),
        )


def _holder(folder: Path) -> int | None:
    """The id of another running process holding the campaign, or None."""
    try:
        pid = int((folder / ".lock").read_text().strip())
    except (OSError, ValueError):
        return None
    if pid == os.getpid():
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None  # it exited without letting go
    except PermissionError:
        return pid  # alive, someone else's
    except OSError:
        return None
    return pid


def _lock(folder: Path) -> None:
    holder = _holder(folder)
    if holder is not None:
        raise ValueError(
            f"Campaign {folder.name!r} is open in another process (id {holder}): the "
            "abbot agent's server or another abbot UI. Close it there first (ask the "
            "agent to close_campaign, end that session, or stop that UI), then try "
            "again."
        )
    folder.mkdir(parents=True, exist_ok=True)
    (folder / ".lock").write_text(str(os.getpid()))


def _unlock(folder: Path) -> None:
    if _holder(folder) is None:
        (folder / ".lock").unlink(missing_ok=True)


def _copy_results(source: Campaign, target: Campaign) -> tuple[int, dict[str, int]]:
    """Record ``source``'s measured runs in ``target`` where they fit its spec.

    Failed runs and bad-region runs aren't copied: they belong to the old setup.
    Returns the number copied and the skipped ones counted by reason.
    """
    spec = target.spec
    copied, skipped = 0, {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for r in source.history.records:
        if r.outputs is None or r.fault or r.bad_region:
            skip("failed or bad-region run")
            continue
        design = source.spec.as_dict(r.candidate)
        outputs = dict(zip(source.spec.output_names,
                           torch.as_tensor(r.outputs).reshape(-1).tolist()))
        if set(spec.variable_names) - set(design) or set(spec.output_names) - set(outputs):
            skip("variables or outputs differ")
            continue
        design = {k: design[k] for k in spec.variable_names}
        if not _fits(spec, design):
            skip("outside the new ranges")
            continue
        target.record(design, {k: outputs[k] for k in spec.output_names}, cost=r.cost)
        copied += 1
    return copied, skipped


def _fits(spec: ProblemSpec, design: dict) -> bool:
    """Whether a design is inside the spec's ranges and choices."""
    for d in spec.inputs:
        value = design[d.name]
        lower, upper = getattr(d, "lower", None), getattr(d, "upper", None)
        if lower is not None and not lower <= value <= upper:
            return False
        values = getattr(d, "values", None)
        if values is not None and value not in values:
            return False
    return True


def _goal_row(goal: dict) -> dict:
    """A goal from JSON (lists for band bounds) as ``build_goal`` takes it."""
    row = dict(goal)
    if "bounds" in row:
        row["bounds"] = {k: tuple(v) for k, v in row["bounds"].items()}
    return row


def _record_dict(record: Record, spec: ProblemSpec) -> dict:
    out: dict[str, Any] = {"design": spec.as_dict(record.candidate)}
    if record.id is not None:
        out["run"] = record.id
    if record.outputs is not None:
        values = torch.as_tensor(record.outputs).reshape(-1).tolist()
        out["outputs"] = dict(zip(spec.output_names, values))
    if record.cost is not None:
        out["cost"] = record.cost
    return out
