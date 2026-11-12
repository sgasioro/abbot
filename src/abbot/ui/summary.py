"""The UI's plain-language text: what abbot will do, and how the campaign is going.

Pure functions of the setup state or the campaign, returning Markdown lines, so
the wording is testable without a page.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from abbot import analysis
from abbot.policy import BAXPolicy
from abbot.problem import EnumeratedDomain, IntegerDomain, ProblemSpec, StructuredDomain
from abbot.ui.builder import BAX_STRATEGY_LABELS, build_start

DIRECTION_WORDS = {"MAXIMIZE": "highest", "MINIMIZE": "lowest"}


def fmt(value) -> str:
    """A value for people: a label as is, a number to 4 significant figures."""
    if isinstance(value, str):
        return value
    if isinstance(value, float) and math.isinf(value):
        return "∞" if value > 0 else "−∞"
    return f"{value:.4g}"


def _knob(d) -> str:
    if isinstance(d, StructuredDomain):
        return f"{d.name} ({d.dim} fractions summing to 1)"
    if isinstance(d, EnumeratedDomain):
        values = [fmt(v) for v in d.values]
        return f"{d.name} ({_or(values)})"
    text = f"{d.name} ({fmt(d.lower)}–{fmt(d.upper)}"
    if getattr(d, "log", False):
        text += ", log scale"
    if isinstance(d, IntegerDomain):
        text += ", whole numbers"
    return text + ")"


def _or(items: list[str]) -> str:
    if len(items) <= 2:
        return " or ".join(items)
    return ", ".join(items[:-1]) + " or " + items[-1]


def _and(items: list[str]) -> str:
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _limits(spec: ProblemSpec) -> list[str]:
    out = []
    for o in spec.outputs:
        if o.min_val is not None and o.max_val is not None:
            out.append(f"{o.name} between {fmt(o.min_val)} and {fmt(o.max_val)}")
        elif o.min_val is not None:
            out.append(f"{o.name} at least {fmt(o.min_val)}")
        elif o.max_val is not None:
            out.append(f"{o.name} at most {fmt(o.max_val)}")
    return out


def goal_words(goal: dict) -> str:
    """A target goal (the JSON form) in words: "every setting where y is between
    0.4 and 0.7"."""
    kind = goal.get("kind")
    if kind == "band":
        parts = []
        for name, (lo, hi) in goal.get("bounds", {}).items():
            if lo is not None and hi is not None:
                parts.append(f"{name} is between {fmt(lo)} and {fmt(hi)}")
            elif lo is not None:
                parts.append(f"{name} is at least {fmt(lo)}")
            elif hi is not None:
                parts.append(f"{name} is at most {fmt(hi)}")
        return "every setting where " + _and(parts) if parts else "a target band"
    if kind == "level":
        side = "at most" if goal.get("upper") else "at least"
        threshold = goal.get("threshold")
        value = fmt(threshold) if threshold is not None else "?"
        return f"every setting where {goal.get('output')} is {side} {value}"
    if kind == "topk":
        word = "highest" if goal.get("largest", True) else "lowest"
        return f"the {goal.get('k', 1)} settings with the {word} {goal.get('output')}"
    pct = 100 - float(goal.get("percentile", 80.0))
    return f"the top {fmt(pct)}% of settings"


def plan_lines(state) -> list[str]:
    """What abbot will do with this setup, in a few sentences (Markdown)."""
    spec, error = state.preview()
    if spec is None:
        return [f"Not ready yet: {error}"]
    lines = [f"Suggest runs over {_and([_knob(d) for d in spec.domain])}."]
    limits = _limits(spec)
    keeping = f", keeping {_and(limits)}" if limits else ""
    if state.method == "bax":
        try:
            goal = goal_words(state.goal_json())
        except Exception:  # noqa: BLE001
            goal = "the target"
        strategy = BAX_STRATEGY_LABELS.get(state.bax_strategy, state.bax_strategy)
        lines.append(f"Map {goal}{keeping}. Strategy: {strategy.lower()}.")
    else:
        objectives = [o for o in spec.outputs if o.direction != "OBSERVE"]
        watched = [o.name for o in spec.outputs if o.direction == "OBSERVE"]
        if not objectives:
            lines.append("Nothing to optimize yet: pick Higher, Lower or Near for an output.")
        elif len(objectives) == 1:
            o = objectives[0]
            if o.direction == "TARGET":
                lines.append(f"Look for the {o.name} nearest {fmt(o.target)}{keeping}.")
            else:
                lines.append(f"Look for the {DIRECTION_WORDS[o.direction]} {o.name}{keeping}.")
        else:
            wants = [f"{o.name} near {fmt(o.target)}" if o.direction == "TARGET"
                     else f"{o.name} {'higher' if o.direction == 'MAXIMIZE' else 'lower'}"
                     for o in objectives]
            lines.append(f"Map the trade-off between {_and(wants)}{keeping}.")
        if watched:
            lines.append(f"Also model {_and(watched)}, without optimizing "
                         f"{'it' if len(watched) == 1 else 'them'}.")
    for d in spec.domain:
        if getattr(d, "max_step", None) is not None:
            lines.append(f"Move {d.name} at most {fmt(d.max_step)} between runs.")
    n_constraints = len(spec.domain_constraints or []) + len(spec.output_constraints or [])
    if n_constraints:
        lines.append(f"Keep to {n_constraints} linear constraint"
                     f"{'s' if n_constraints > 1 else ''}.")
    if spec.sources and len(spec.sources) == 1:
        s = spec.sources[0]
        what = (_or([str(v) for v in s.domain.values])
                if isinstance(s.domain, EnumeratedDomain)
                else f"{s.name} from {fmt(s.domain.lower)} to {fmt(s.domain.upper)}")
        lines.append(f"Choose where each run happens ({what}), weighing what it costs, "
                     f"and recommend at {s.target}.")
    elif spec.sources:
        targets = _and([f"{s.name} {s.target}" for s in spec.sources])
        lines.append(f"Choose {_and([s.name for s in spec.sources])} for each run, "
                     f"weighing what it costs, and recommend at {targets}.")
    designs = state.extra.get("initial_designs")
    try:
        start = build_start(spec, state.variable_rows)
    except Exception:  # noqa: BLE001 - reported on the Knobs step
        start = None
    if designs:
        n = len(designs)
        lines.append(f"Start with the {n} design{'s' if n > 1 else ''} in campaign.json, "
                     "then follow the model." if n > 1 or start is None
                     else "Start from your current settings.")
    elif start is not None:
        lines.append("Start from your current settings.")
    else:
        lines.append("Start with a few runs spread over the space, then follow the model.")
    lines.append("A simulated experiment runs each design, so you can watch how it works."
                 if state.mock else "You run each design and enter what you measured.")
    where = state.save_path()
    if where:
        home = str(Path.home())
        if where.startswith(home + "/"):
            where = "~" + where[len(home):]
        lines.append(f"Save it in `{where}`, where the abbot agent can open it too.")
    elif state.mock:
        lines.append("A simulation isn't saved.")
    return lines


# -- how it's going ------------------------------------------------------------------


def progress_lines(campaign) -> list[str]:
    """Where the campaign stands, in plain words (Markdown)."""
    spec = campaign.spec
    history = campaign.history
    n = len(history)
    if not n:
        return ["No results yet. Record the first run to see how it's going."]
    failed = sum(bool(r.fault) for r in history.records)
    head = f"{n} run{'s' if n != 1 else ''}"
    if failed:
        head += f", {failed} failed"
    if spec.sources and any(r.cost is not None for r in history.records):
        head += f", cost so far {fmt(history.cost_spent)}"
    lines = [head + "."]
    if not len(history.clean):
        lines.append("No usable result yet.")
        return lines
    if isinstance(campaign.policy, BAXPolicy):
        lines += _region_lines(campaign)
    elif spec.n_objectives == 1:
        lines += _best_lines(campaign)
    elif spec.n_objectives > 1:
        lines += _front_lines(campaign)
    if campaign.stopped:
        lines.append(f"Stopped: {campaign.stopped}")
    return lines


def _setting(spec, record) -> str:
    return ", ".join(f"{k} {fmt(v)}" for k, v in spec.as_dict(record.candidate).items())


def _best_lines(campaign) -> list[str]:
    spec = campaign.spec
    k = spec.objective_indices()[0]
    name = spec.outputs[k].name
    result = campaign.result()
    lines = []
    if result.recommended:
        r = campaign.history.records[result.recommended[0]]
        others = [f"{o.name} {fmt(float(r.outputs.reshape(-1)[j]))}"
                  for j, o in enumerate(spec.outputs) if j != k]
        extra = f" ({', '.join(others)})" if others else ""
        aim = f" (target {fmt(spec.outputs[k].target)})" if spec.outputs[k].target is not None else ""
        line = (f"Best so far: **{name} {fmt(float(r.outputs.reshape(-1)[k]))}**{aim} at "
                f"{_setting(spec, r)}{extra}.")
        if result.p_feasible:
            line += f" **{result.p_feasible[0]:.0%} likely** to meet the limits."
        lines.append(line)
    best = analysis.convergence(campaign.history, spec)
    known = [b for b in best if not math.isnan(b)]
    if len(known) >= 4:
        output = spec.outputs[k]
        gain = output.score(best[-1]) - output.score(best[-4])
        if gain > 0:
            change = (f"{fmt(gain)} closer" if output.direction == "TARGET"
                      else f"{'+' if output.direction == 'MAXIMIZE' else '−'}{fmt(gain)}")
            lines.append(f"Still improving: {change} over the last 3 runs.")
        else:
            since = next(i for i in range(len(best) - 1, -1, -1)
                         if i == 0 or best[i] != best[i - 1])
            quiet = len(best) - 1 - since
            lines.append(f"No better result in the last {quiet} runs.")
    elif not known:
        lines.append("No run meets the limits yet.")
    return lines


def _front_lines(campaign) -> list[str]:
    spec = campaign.spec
    rec = campaign.recommend()
    names = _and([o.name for o in spec.objectives])
    plural = "s" if len(rec) != 1 else ""
    lines = [f"{len(rec)} setting{plural} on the trade-off between {names}:"]
    for i in rec[:4]:
        r = campaign.history.records[i]
        outs = ", ".join(f"{o.name} {fmt(float(r.outputs.reshape(-1)[j]))}"
                         for j, o in enumerate(spec.outputs))
        lines.append(f"- {outs} at {_setting(spec, r)}")
    if len(rec) > 4:
        lines.append(f"- and {len(rec) - 4} more (see Runs)")
    return lines


def _region_lines(campaign) -> list[str]:
    spec = campaign.spec
    policy = campaign.policy
    lines = []
    model = campaign.current_model()
    if model is not None:
        region = analysis.bax_target_region(policy, model, spec)
        share = region["identified"] / max(region["grid_size"], 1)
        lines.append(f"The model puts **{share:.0%}** of the settings it considers in "
                     "the target.")
    clean = campaign.history.clean
    Y = torch.stack([r.outputs.double().reshape(-1) for r in clean.records])
    X = torch.stack([r.candidate.double().reshape(-1) for r in clean.records])
    try:
        hits = len(policy._identify(spec, Y, X))
    except Exception:  # noqa: BLE001 - a goal that ranks needs more than the runs
        hits = None
    if hits is not None and policy.goal.__class__.__name__ in (
        "BandIntersection", "LevelSet", "Wishlist"
    ):
        lines.append(f"{hits} of your {len(clean)} usable runs landed in it.")
    return lines


def status_words(controller) -> tuple[str, str]:
    """``(text, tone)`` for the header chip: what the campaign is waiting for."""
    campaign = controller.campaign
    if campaign.halted:
        return "needs you: it halted", "red"
    if campaign.stopped or controller.status.value == "done":
        return "finished", "blue"
    if controller.can_auto:
        return {"running": ("running", "green"), "paused": ("paused", "amber")}.get(
            controller.status.value, ("ready to run", "grey"))
    if campaign.outstanding():
        return "waiting for your result", "green"
    return "ready", "grey"


__all__ = ["fmt", "goal_words", "plan_lines", "progress_lines", "status_words"]
