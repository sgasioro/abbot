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
ACCENT = "#4f46e5"
BAND = "rgba(79,70,229,0.18)"


@dataclass
class VizSelections:
    """Which output / which variables the model + parity views slice on."""

    output: int = 0
    var_x: int = 0
    var_y: int = 1


def blank_fig(title: str = "building…") -> dict:
    return {"data": [], "layout": {**PLOT_LAYOUT, "title": title}}


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
        "layout": {**PLOT_LAYOUT, "title": title, "xaxis": {"title": "step"},
                   "yaxis": {"title": ylab}},
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
        "layout": {**PLOT_LAYOUT, "title": "Target region discovered",
                   "yaxis": {"title": "count"}},
    }


def model_fig(history: History, spec: ProblemSpec, model, sel: VizSelections) -> dict:
    """Adaptive model view: 1-D band, 2-D contour, or 3-D+ slice through best point.

    The domain-dimensionality dispatch. Requires a fitted ``model``.
    """
    d = len(spec.domain)
    obj = sel.output
    if d == 1:
        s = analysis.slice_1d(history, spec, model, var=0, output=obj)
        upper = [m + sd for m, sd in zip(s["mean"], s["std"])]
        lower = [m - sd for m, sd in zip(s["mean"], s["std"])]
        px = [r.candidate.reshape(-1)[0].item() for r in history.clean.records]
        py = [r.outputs.reshape(-1)[obj].item() for r in history.clean.records]
        return {
            "data": [
                {"x": s["x"] + s["x"][::-1], "y": upper + lower[::-1], "fill": "toself",
                 "fillcolor": BAND, "line": {"color": "rgba(0,0,0,0)"}, "type": "scatter",
                 "hoverinfo": "skip"},
                {"x": s["x"], "y": s["mean"], "type": "scatter", "mode": "lines",
                 "line": {"color": ACCENT, "width": 2}, "name": "posterior mean"},
                {"x": px, "y": py, "type": "scatter", "mode": "markers",
                 "marker": {"color": "#111", "size": 7}, "name": "observed"},
            ],
            "layout": {**PLOT_LAYOUT, "title": f"Posterior over {s['var_name']}",
                       "xaxis": {"title": s["var_name"]},
                       "yaxis": {"title": s["objective_name"]}},
        }
    # 2-D (or a 2-var slice of higher-D) contour.
    vx, vy = sel.var_x, sel.var_y
    s = analysis.slice_2d(history, spec, model, var_x=vx, var_y=vy, output=obj)
    px = [r.candidate.reshape(-1)[vx].item() for r in history.clean.records]
    py = [r.candidate.reshape(-1)[vy].item() for r in history.clean.records]
    title = f"Posterior mean of {s['objective_name']}"
    if d > 2:
        title += " (slice through best point)"
    return {
        "data": [
            {"x": s["x"], "y": s["y"], "z": s["z"], "type": "contour",
             "colorscale": "Viridis", "colorbar": {"title": s["objective_name"]}},
            {"x": px, "y": py, "type": "scatter", "mode": "markers",
             "marker": {"color": "white", "size": 7, "line": {"color": "black", "width": 1}}},
        ],
        "layout": {**PLOT_LAYOUT, "title": title,
                   "xaxis": {"title": s["var_x_name"]}, "yaxis": {"title": s["var_y_name"]}},
    }


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
        "layout": {**PLOT_LAYOUT, "title": f"Parity — {name}",
                   "xaxis": {"title": "observed"}, "yaxis": {"title": "predicted"}},
    }
