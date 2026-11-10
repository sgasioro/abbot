"""Plotly figure builders for the UI — pure ``(history, spec, model, …) -> dict``.

Kept out of ``app.py`` (which only wires widgets) so the adaptive-to-dimensionality
dispatch — ``n_objectives`` → convergence / hypervolume / BAX, and domain-dim →
1-D band / 2-D contour / slice — is testable without a browser. Every builder is a
pure function returning a plotly figure dict (the shape ``ui.plotly`` consumes);
none touches NiceGUI or ``UIState``.

The visual constants and the dimensionality logic live here; ``app.py`` chooses
which builder to call and owns the one refresh-vs-update rule (see
``UIState.needs_structure_rebuild``).
"""

from __future__ import annotations

from dataclasses import dataclass

from abbot import analysis
from abbot.history import History
from abbot.policy import BAXPolicy
from abbot.problem import ProblemSpec

PLOT_LAYOUT = {
    "height": 300,
    "margin": {"l": 56, "r": 16, "t": 44, "b": 44},
    "template": "plotly_white",
    "font": {"size": 12},
    "showlegend": False,
}
ACCENT = "#1d4ed8"
BAND = "rgba(29,78,216,0.15)"
GOAL_FILL = "rgba(22,163,74,0.14)"
GOAL_LINE = "#16a34a"
REGION_FILL = "rgba(22,163,74,0.10)"
NEXT = "#ea580c"


@dataclass
class VizSelections:
    """Which output / which variables the model + parity views slice on."""

    output: int = 0
    var_x: int = 0
    var_y: int = 1


def blank_fig(title: str = "building…") -> dict:
    return {"data": [], "layout": {**PLOT_LAYOUT, "title": {"text": title}}}


def progress_fig(history: History, spec: ProblemSpec, policy=None, model=None) -> dict:
    """Convergence (1 objective) / hypervolume (≥2) / BAX target region (0).

    The ``n_objectives`` dispatch. For pure-BAX (0 objectives) there is no single
    quantity to converge, so it shows the discovered target-region size — which
    needs the ``policy`` and fitted ``model``.
    """
    if spec.n_objectives == 0:
        return bax_progress_fig(history, spec, policy, model)
    if spec.n_objectives == 1:
        y = analysis.convergence(history, spec)
        title, ylab = f"Best {spec.objectives[0].name} so far", spec.objectives[0].name
    else:
        y = analysis.hypervolume(history, spec)
        title, ylab = "Dominated hypervolume", "hypervolume"
    return {
        "data": [{
            "y": y, "x": list(range(len(y))), "type": "scatter", "mode": "lines+markers",
            "line": {"color": ACCENT, "width": 2}, "marker": {"color": ACCENT, "size": 6},
        }],
        "layout": {**PLOT_LAYOUT, "title": {"text": title}, "xaxis": {"title": {"text": "step"}},
                   "yaxis": {"title": {"text": ylab}}},
    }


def bax_progress_fig(history: History, spec: ProblemSpec, policy, model) -> dict:
    """Target candidates the goal identifies on the posterior mean vs. experiments
    run. A current-state snapshot (we don't re-fit the past), so it needs the fitted
    ``model`` + the ``BAXPolicy``. ``identified`` is 0 before a model exists.
    """
    identified = 0
    if model is not None and isinstance(policy, BAXPolicy):
        identified = analysis.bax_target_region(policy, model, spec)["identified"]
    collected = len(history.clean)
    return {
        "data": [{
            "x": ["target candidates", "experiments run"],
            "y": [identified, collected],
            "type": "bar", "marker": {"color": [ACCENT, "#9ca3af"]},
        }],
        "layout": {**PLOT_LAYOUT, "title": {"text": "Target region discovered"},
                   "yaxis": {"title": {"text": "count"}}},
    }


def model_fig(
    history: History, spec: ProblemSpec, model, sel: VizSelections, policy=None,
    pending=None,
) -> dict:
    """Adaptive model view: 1-D band, 2-D contour, or 3-D+ slice through best point.

    The domain-dimensionality dispatch. Requires a fitted ``model``. With the
    campaign's ``policy`` it marks the goal (a band or threshold on the output, and
    where the model puts the goal region); ``pending`` designs show as the next ones.
    """
    import torch

    from abbot.plots import _bounded, _pointwise, _stretches

    goal = policy._goal_for(spec) if hasattr(policy, "_goal_for") else None
    pending = [p.double().reshape(-1) for p in (pending or [])]
    d = len(spec.domain)
    obj = sel.output
    if d == 1:
        s = analysis.slice_1d(history, spec, model, var=0, output=obj, n=160)
        upper = [m + 2 * sd for m, sd in zip(s["mean"], s["std"])]
        lower = [m - 2 * sd for m, sd in zip(s["mean"], s["std"])]
        clean = [r for r in history.records if r.usable and r.outputs is not None]
        px = [r.candidate.reshape(-1)[0].item() for r in clean]
        py = [r.outputs.reshape(-1)[obj].item() for r in clean]
        ys = upper + lower + py
        lo_y, hi_y = min(ys), max(ys)
        pad = 0.08 * (hi_y - lo_y or 1.0)
        lo_y, hi_y = lo_y - pad, hi_y + pad
        shapes = []
        band = _goal_band(goal, obj)
        if band is not None and _bounded(band):
            b_lo = lo_y if band[0] is None or band[0] == float("-inf") else max(band[0], lo_y)
            b_hi = hi_y if band[1] is None or band[1] == float("inf") else min(band[1], hi_y)
            shapes.append({"type": "rect", "xref": "paper", "x0": 0, "x1": 1, "yref": "y",
                           "y0": b_lo, "y1": b_hi, "fillcolor": GOAL_FILL, "line": {"width": 0},
                           "layer": "below"})
        level = _goal_level(goal, obj)
        if level is not None:
            shapes.append(_hline(level, GOAL_LINE, "solid"))
        o = spec.outputs[obj]
        for v in (o.min_val, o.max_val):
            if v is not None:
                shapes.append(_hline(v, "#6b7280", "dash"))
        if goal is not None and len(spec.domain) == 1:
            xs = torch.tensor(s["x"], dtype=torch.double).reshape(-1, 1)
            grid = spec.at_target(torch.cat(
                [xs, torch.zeros(len(xs), len(spec.inputs) - 1, dtype=torch.double)], dim=1)
            ) if spec.sources else xs
            mean, _ = analysis._posterior_raw(model, grid, spec)
            try:
                region = policy._identify(spec, mean, grid)
            except Exception:  # noqa: BLE001 - a goal that can't run here: no shading
                region = []
            if _pointwise(goal) or d == 1:
                for x0, x1 in _stretches(s["x"], list(region)):
                    shapes.append({"type": "rect", "xref": "x", "x0": x0, "x1": x1,
                                   "yref": "paper", "y0": 0, "y1": 1, "fillcolor": REGION_FILL,
                                   "line": {"width": 0}, "layer": "below"})
        for p in pending:
            shapes.append({"type": "line", "xref": "x", "x0": p[0].item(), "x1": p[0].item(),
                           "yref": "paper", "y0": 0, "y1": 1,
                           "line": {"color": NEXT, "dash": "dash", "width": 2}})
        latest = [r for r in history.records if r.outputs is not None and r.usable][-1:]
        data = [
            {"x": s["x"] + s["x"][::-1], "y": upper + lower[::-1], "fill": "toself",
             "fillcolor": BAND, "line": {"color": "rgba(0,0,0,0)"}, "type": "scatter",
             "hoverinfo": "skip", "name": "model ± 2σ"},
            {"x": s["x"], "y": s["mean"], "type": "scatter", "mode": "lines",
             "line": {"color": ACCENT, "width": 2}, "name": "model"},
            {"x": px, "y": py, "type": "scatter", "mode": "markers",
             "marker": {"color": "#111", "size": 7}, "name": "runs"},
        ]
        if latest:
            r = latest[0]
            data.append({"x": [r.candidate.reshape(-1)[0].item()],
                         "y": [r.outputs.reshape(-1)[obj].item()], "type": "scatter",
                         "mode": "markers", "name": "latest",
                         "marker": {"size": 15, "color": "rgba(0,0,0,0)",
                                    "line": {"color": "#111", "width": 2}}})
        log = bool(getattr(spec.domain[0], "log", False))
        return {
            "data": data,
            "layout": {**PLOT_LAYOUT, "showlegend": False, "shapes": shapes,
                       "title": {"text": f"{s['objective_name']} over {s['var_name']}"},
                       "xaxis": {"title": {"text": s["var_name"]},
                                 "type": "log" if log else "linear"},
                       "yaxis": {"title": {"text": s["objective_name"]},
                                 "range": [lo_y, hi_y]}},
        }
    # 2-D (or a 2-var slice of higher-D) contour.
    vx, vy = sel.var_x, sel.var_y
    s = analysis.slice_2d(history, spec, model, var_x=vx, var_y=vy, output=obj)
    clean = [r for r in history.records if r.usable]
    px = [r.candidate.reshape(-1)[vx].item() for r in clean]
    py = [r.candidate.reshape(-1)[vy].item() for r in clean]
    title = f"Model of {s['objective_name']}"
    if d > 2:
        title += " (slice through the best point)"
    data = [
        {"x": s["x"], "y": s["y"], "z": s["z"], "type": "contour",
         "colorscale": "Viridis", "colorbar": {"title": {"text": s["objective_name"]}}},
        {"x": px, "y": py, "type": "scatter", "mode": "markers", "name": "runs",
         "marker": {"color": "white", "size": 7, "line": {"color": "black", "width": 1}}},
    ]
    if goal is not None and (d == 2 or _pointwise(goal)):
        mask = _region_mask(spec, model, policy, history, vx, vy, s)
        if mask is not None:
            data.append({"x": s["x"], "y": s["y"], "z": mask, "type": "contour",
                         "showscale": False, "hoverinfo": "skip", "name": "goal region",
                         "contours": {"coloring": "none", "start": 0.5, "end": 0.5, "size": 1},
                         "line": {"color": "white", "width": 3, "dash": "dash"}})
    if pending:
        data.append({"x": [p[vx].item() for p in pending], "y": [p[vy].item() for p in pending],
                     "type": "scatter", "mode": "markers", "name": "next",
                     "marker": {"symbol": "star", "size": 16, "color": NEXT,
                                "line": {"color": "white", "width": 1}}})
    logx = bool(getattr(spec.domain[vx], "log", False))
    logy = bool(getattr(spec.domain[vy], "log", False))
    return {
        "data": data,
        "layout": {**PLOT_LAYOUT, "title": {"text": title},
                   "xaxis": {"title": {"text": s["var_x_name"]}, "type": "log" if logx else "linear"},
                   "yaxis": {"title": {"text": s["var_y_name"]}, "type": "log" if logy else "linear"}},
    }


def _goal_band(goal, k: int):
    from abbot.goal import BandIntersection

    if isinstance(goal, BandIntersection) and k < len(goal.bounds):
        return goal.bounds[k]
    return None


def _goal_level(goal, k: int):
    from abbot.goal import LevelSet

    if isinstance(goal, LevelSet) and goal.output == k:
        return goal.threshold
    return None


def _hline(y: float, color: str, dash: str) -> dict:
    return {"type": "line", "xref": "paper", "x0": 0, "x1": 1, "yref": "y", "y0": y, "y1": y,
            "line": {"color": color, "dash": dash, "width": 1.5}}


def _region_mask(spec, model, policy, history, vx, vy, s) -> list | None:
    """1 where the model puts the goal region on the contour's grid, else 0."""
    import torch

    xs = torch.tensor(s["x"], dtype=torch.double)
    ys = torch.tensor(s["y"], dtype=torch.double)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    anchor = analysis._anchor_point(history, spec)
    grid = anchor.repeat(gx.numel(), 1)
    grid[:, vx], grid[:, vy] = gx.reshape(-1), gy.reshape(-1)
    grid = spec.at_target(grid)
    mean, _ = analysis._posterior_raw(model, grid, spec)
    try:
        region = policy._identify(spec, mean, grid)
    except Exception:  # noqa: BLE001
        return None
    mask = torch.zeros(gx.numel(), dtype=torch.double)
    mask[list(region)] = 1.0
    return mask.reshape(gx.shape).tolist()


def parity_fig(history: History, spec: ProblemSpec, model, sel: VizSelections) -> dict | None:
    """Predicted vs. observed for the selected output, or ``None`` if unavailable."""
    p = analysis.parity(history, spec, model)
    name = spec.outputs[sel.output].name
    d = p.get(name)
    if not d:
        return None
    obs, pred, std = d["observed"], d["predicted"], d["std"]
    lo = min(obs + pred)
    hi = max(obs + pred)
    return {
        "data": [
            {"x": [lo, hi], "y": [lo, hi], "type": "scatter", "mode": "lines",
             "line": {"color": "#999", "dash": "dash"}, "hoverinfo": "skip"},
            {"x": obs, "y": pred, "error_y": {"type": "data", "array": std, "visible": True},
             "type": "scatter", "mode": "markers", "marker": {"color": ACCENT, "size": 7}},
        ],
        "layout": {**PLOT_LAYOUT, "title": {"text": f"Parity — {name}"},
                   "xaxis": {"title": {"text": "observed"}}, "yaxis": {"title": {"text": "predicted"}}},
    }
