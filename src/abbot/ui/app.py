"""NiceGUI web client for abbot — a thin, single-page dashboard over Campaign.

Left column: define the problem (type-polymorphic variable editor), mock toggle,
and the run controls (manual suggest/measure, or mock auto-run run/pause/step/stop).
Right column: live model visualization (adaptive to dimensionality) + convergence,
the recommended design(s), and the history table. One scrolling dashboard, no
click-through screens.

All optimization logic is delegated to the core (``Campaign``, ``RunController``,
``builder``, ``analysis``); this module only builds widgets and wires events.
Per-client state lives on a ``UIState`` created inside the page function — no
module-global state. Run with ``python -m abbot.ui``.
"""

from __future__ import annotations

import logging

import torch
from nicegui import ui

from abbot import analysis
from abbot.experiment import SyntheticExperiment
from abbot.model import default_constructor
from abbot.policy import ParetoBO, SimpleBO
from abbot.session import Campaign
from abbot.ui.builder import build_spec, choose_optimizer
from abbot.ui.controller import RunController, RunStatus

logger = logging.getLogger(__name__)

_PLOT_LAYOUT = {
    "height": 300,
    "margin": {"l": 56, "r": 16, "t": 44, "b": 44},
    "template": "plotly_white",
    "font": {"size": 12},
    "showlegend": False,
}
_ACCENT = "#4f46e5"
_BAND = "rgba(79,70,229,0.18)"


class UIState:
    """Per-client session state: editor rows, the live controller, viz selections."""

    def __init__(self) -> None:
        self.variable_rows: list[dict] = [
            {"kind": "continuous", "name": "x1", "lower": 0.0, "upper": 1.0}
        ]
        self.objective_rows: list[dict] = [{"name": "y", "direction": "MAXIMIZE"}]
        self.controller: RunController | None = None
        self.spec = None
        self.pending: list[dict] | None = None
        self.mock: bool = False
        # Model-viz selections (which objective / which vars to slice).
        self.viz_obj = 0
        self.viz_vx = 0
        self.viz_vy = 1
        # Auto-run state — created ONCE per page (see _build_page), never inside a
        # refreshable, so a panel refresh can't orphan the timer or reset the flag.
        self.auto_running = False
        # Panel refreshers + viz widget handles, set during _build_page / _viz_panel.
        # Per-step updates mutate these in place (no teardown -> no jitter).
        self._left = None
        self._right = None
        self._plot_model = None
        self._plot_conv = None
        self._plot_parity = None
        self._rec_box = None
        self._table = None

    def start_campaign(self) -> None:
        spec = build_spec(self.variable_rows, self.objective_rows)  # may raise
        optimizer = choose_optimizer(spec)  # may raise on mixed spaces
        policy = SimpleBO(optimizer) if spec.n_objectives == 1 else ParetoBO(optimizer)
        campaign = Campaign(
            spec, policy, default_constructor(spec, bounds=spec.bounds_tensor())
        )
        bounds = spec.bounds_tensor()
        lo, hi = bounds[0], bounds[1]
        seeds = lo + (hi - lo) * torch.rand(
            max(3, spec.n_objectives + 1), bounds.shape[-1], dtype=torch.double
        )
        campaign.seed(seeds)
        self.spec = spec
        self.pending = None
        self.viz_obj = 0
        self.viz_vx = 0
        self.viz_vy = 1 if len(spec.domain) > 1 else 0
        self.auto_running = False
        if self.mock:
            exp = SyntheticExperiment(spec, seed=0, noise=0.0)
            self.controller = RunController(campaign, experiment=exp, evaluate=lambda o: o)
        else:
            self.controller = RunController(campaign)

    def fitted_model(self):
        """The current fitted surrogate (canonical space) or None."""
        if self.controller is None:
            return None
        return self.controller.campaign._safe_fit()


# --------------------------------------------------------------------------- #
# Editors
# --------------------------------------------------------------------------- #


def _variable_editor(state: UIState) -> None:
    with ui.row().classes("items-center gap-2"):
        ui.icon("tune").classes("text-indigo-600")
        ui.label("Design variables").classes("text-base font-semibold")
    container = ui.column().classes("gap-2 w-full")

    def redraw() -> None:
        container.clear()
        with container:
            for i, row in enumerate(state.variable_rows):
                with ui.row().classes("items-center gap-1 w-full"):
                    ui.select(
                        {"continuous": "Continuous", "enumerated": "Enumerated"},
                        value=row["kind"],
                        on_change=lambda e, r=row: (_set(r, "kind", e.value), redraw()),
                    ).props("dense outlined").classes("w-32")
                    ui.input("name", value=row.get("name", "")).bind_value(row, "name").props(
                        "dense outlined"
                    ).classes("w-24")
                    if row["kind"] == "continuous":
                        ui.number("lo", value=row.get("lower", 0.0)).bind_value(row, "lower").props(
                            "dense outlined"
                        ).classes("w-20")
                        ui.number("hi", value=row.get("upper", 1.0)).bind_value(row, "upper").props(
                            "dense outlined"
                        ).classes("w-20")
                    else:
                        ui.input("values", value=row.get("values", "0, 1, 2")).bind_value(
                            row, "values"
                        ).props("dense outlined").classes("w-48")
                    ui.button(
                        icon="delete",
                        on_click=lambda _, idx=i: (_pop(state.variable_rows, idx), redraw()),
                    ).props("flat dense round color=grey")

    def add() -> None:
        n = len(state.variable_rows) + 1
        state.variable_rows.append({"kind": "continuous", "name": f"x{n}", "lower": 0.0, "upper": 1.0})
        redraw()

    redraw()
    ui.button("Add variable", icon="add", on_click=add).props("flat dense color=indigo")


def _objective_editor(state: UIState) -> None:
    with ui.row().classes("items-center gap-2"):
        ui.icon("flag").classes("text-indigo-600")
        ui.label("Objectives").classes("text-base font-semibold")
    container = ui.column().classes("gap-2 w-full")

    def redraw() -> None:
        container.clear()
        with container:
            for i, row in enumerate(state.objective_rows):
                with ui.row().classes("items-center gap-1 w-full"):
                    ui.input("name", value=row.get("name", "")).bind_value(row, "name").props(
                        "dense outlined"
                    ).classes("w-32")
                    ui.select(
                        {"MAXIMIZE": "Maximize", "MINIMIZE": "Minimize", "OBSERVE": "Observe (BAX)"},
                        value=row["direction"],
                    ).bind_value(row, "direction").props("dense outlined").classes("w-36")
                    ui.button(
                        icon="delete",
                        on_click=lambda _, idx=i: (_pop(state.objective_rows, idx), redraw()),
                    ).props("flat dense round color=grey")

    def add() -> None:
        n = len(state.objective_rows) + 1
        state.objective_rows.append({"name": f"y{n}", "direction": "MAXIMIZE"})
        redraw()

    redraw()
    ui.button("Add objective", icon="add", on_click=add).props("flat dense color=indigo")


def _set(row: dict, key: str, value) -> None:
    row[key] = value


def _pop(rows: list, idx: int) -> None:
    if 0 <= idx < len(rows) and len(rows) > 1:
        rows.pop(idx)


# --------------------------------------------------------------------------- #
# Plot builders (pure dict -> ui.plotly)
# --------------------------------------------------------------------------- #


def _convergence_fig(state: UIState):
    hist = state.controller.campaign.history
    spec = state.spec
    if spec.n_objectives == 1:
        y = analysis.convergence(hist, spec)
        title, ylab = f"Best {spec.objectives[0].name} so far", spec.objectives[0].name
    else:
        y = analysis.hypervolume(hist, spec)
        title, ylab = "Dominated hypervolume", "hypervolume"
    return {
        "data": [{
            "y": y, "x": list(range(len(y))), "type": "scatter", "mode": "lines+markers",
            "line": {"color": _ACCENT, "width": 2}, "marker": {"color": _ACCENT, "size": 6},
        }],
        "layout": {**_PLOT_LAYOUT, "title": title, "xaxis": {"title": "step"}, "yaxis": {"title": ylab}},
    }


def _model_fig(state: UIState, model):
    """Adaptive model view: 1-D band, 2-D contour, or 3-D+ slice through best point."""
    spec = state.spec
    hist = state.controller.campaign.history
    d = len(spec.domain)
    obj = state.viz_obj
    if d == 1:
        s = analysis.slice_1d(hist, spec, model, var=0, output=obj)
        upper = [m + sd for m, sd in zip(s["mean"], s["std"])]
        lower = [m - sd for m, sd in zip(s["mean"], s["std"])]
        px = [r.candidate.reshape(-1)[0].item() for r in hist.clean.records]
        py = [r.outputs.reshape(-1)[obj].item() for r in hist.clean.records]
        return {
            "data": [
                {"x": s["x"] + s["x"][::-1], "y": upper + lower[::-1], "fill": "toself",
                 "fillcolor": _BAND, "line": {"color": "rgba(0,0,0,0)"}, "type": "scatter",
                 "hoverinfo": "skip"},
                {"x": s["x"], "y": s["mean"], "type": "scatter", "mode": "lines",
                 "line": {"color": _ACCENT, "width": 2}, "name": "posterior mean"},
                {"x": px, "y": py, "type": "scatter", "mode": "markers",
                 "marker": {"color": "#111", "size": 7}, "name": "observed"},
            ],
            "layout": {**_PLOT_LAYOUT, "title": f"Posterior over {s['var_name']}",
                       "xaxis": {"title": s["var_name"]}, "yaxis": {"title": s["objective_name"]}},
        }
    # 2-D (or a 2-var slice of higher-D) contour.
    vx, vy = state.viz_vx, state.viz_vy
    s = analysis.slice_2d(hist, spec, model, var_x=vx, var_y=vy, output=obj)
    px = [r.candidate.reshape(-1)[vx].item() for r in hist.clean.records]
    py = [r.candidate.reshape(-1)[vy].item() for r in hist.clean.records]
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
        "layout": {**_PLOT_LAYOUT, "title": title,
                   "xaxis": {"title": s["var_x_name"]}, "yaxis": {"title": s["var_y_name"]}},
    }


def _parity_fig(state: UIState, model):
    spec = state.spec
    p = analysis.parity(state.controller.campaign.history, spec, model)
    name = spec.outputs[state.viz_obj].name
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
             "type": "scatter", "mode": "markers", "marker": {"color": _ACCENT, "size": 7}},
        ],
        "layout": {**_PLOT_LAYOUT, "title": f"Parity — {name}",
                   "xaxis": {"title": "observed"}, "yaxis": {"title": "predicted"}},
    }


# --------------------------------------------------------------------------- #
# Page (single dashboard)
# --------------------------------------------------------------------------- #


def _build_page() -> None:
    state = UIState()

    with ui.header().classes("items-center bg-indigo-700"), ui.row().classes("items-center gap-2"):
        ui.icon("science").classes("text-white text-2xl")
        ui.label("abbot").classes("text-white text-xl font-bold")
        ui.label("a Bayesian guide for lab experiments").classes("text-indigo-200 text-sm")

    def _create() -> None:
        try:
            state.start_campaign()
        except Exception as exc:  # noqa: BLE001 - surface build/validation errors as a toast
            ui.notify(str(exc), type="negative", multi_line=True, close_button=True)
            return
        mode = "mock / auto-run" if state.mock else "manual"
        ui.notify(f"Campaign ready ({mode}).", type="positive")
        state._left.refresh()
        state._right.refresh()

    with ui.row().classes("w-full no-wrap gap-4 p-2"):
        # ---------- left: setup + controls ----------
        with ui.column().classes("gap-3").style("min-width: 24rem; max-width: 28rem"):

            @ui.refreshable
            def left() -> None:
                if state.controller is None:
                    with ui.card().classes("w-full"):
                        _variable_editor(state)
                    with ui.card().classes("w-full"):
                        _objective_editor(state)
                    with ui.card().classes("w-full"):
                        ui.switch(
                            "Mock mode (synthetic experiment → auto-run)"
                        ).bind_value(state, "mock").props("color=indigo")
                    ui.button("Create campaign", icon="rocket_launch", on_click=_create).props(
                        "color=indigo"
                    )
                    _import_spec_control(state)
                else:
                    _control_panel(state)

            left()
            state._left = left

        # ---------- right: live viz ----------
        with ui.column().classes("grow gap-3"):

            @ui.refreshable
            def right() -> None:
                _viz_panel(state)

            right()
            state._right = right  # so viz selectors + the timer can re-render it

    # One auto-run timer, created once; the tick reads stable state and refreshes
    # via state._right, so refreshing any panel never orphans it (this was the
    # "Run didn't keep stepping" bug — the timer used to be rebuilt on refresh).
    def _auto_tick() -> None:
        if not state.auto_running or state.controller is None:
            return
        ctrl = state.controller
        if ctrl.status in (RunStatus.HALTED, RunStatus.DONE):
            state.auto_running = False
            state._left.refresh()
            state._right.refresh()
            return
        first = len(ctrl.campaign.history) == 0 or state._plot_conv is None
        try:
            ctrl.step_once()
        except Exception:
            logger.exception("auto step failed")
            state.auto_running = False
            return
        if first:
            # The campaign started with queued-but-unrun seeds, so the viz panel was
            # built in its empty state (no plot widgets). The first step produces
            # history -> rebuild the structure once, then update in place after.
            state._right.refresh()
        else:
            _update_viz(state)  # in-place update; no teardown -> smooth

    ui.timer(0.5, _auto_tick)  # always active; gated by state.auto_running


def _control_panel(state: UIState) -> None:
    ctrl = state.controller
    with ui.card().classes("w-full"):
        with ui.row().classes("items-center justify-between w-full"):
            with ui.row().classes("items-center gap-2"):
                _status_chip(ctrl)
                ui.label(f"{len(ctrl.campaign.history)} experiment(s)").classes(
                    "text-sm text-gray-600"
                )
            ui.button(icon="restart_alt", on_click=lambda: _reset(state)).props(
                "flat dense round color=grey"
            ).tooltip("New campaign")

        spec = state.spec
        ui.label(
            f"{len(spec.domain)} variable(s), {spec.n_objectives} objective(s)"
        ).classes("text-xs text-gray-500")

        if ctrl.can_auto:
            _auto_controls(state)
        else:
            _manual_controls(state)

        ui.separator()
        _data_io_controls(state)


def _data_io_controls(state: UIState) -> None:
    """Export spec/history, and import history CSV into the running campaign."""
    from abbot import io as abbot_io

    spec = state.spec
    with ui.row().classes("gap-1 flex-wrap items-center"):

        def export_spec() -> None:
            ui.download.content(abbot_io.spec_to_json(spec).encode(), "abbot_spec.json")

        def export_history() -> None:
            text = abbot_io.history_to_csv(state.controller.campaign.history, spec)
            ui.download.content(text.encode(), "abbot_history.csv")

        ui.button("Spec JSON", icon="download", on_click=export_spec).props("flat dense color=indigo")
        ui.button("History CSV", icon="download", on_click=export_history).props("flat dense color=indigo")

        def import_history(e) -> None:
            text = e.content.read().decode()
            try:
                hist = abbot_io.history_from_csv(text, spec)
            except Exception as exc:  # noqa: BLE001
                ui.notify(str(exc), type="negative", multi_line=True, close_button=True)
                return
            state.controller.campaign.load_history(hist)
            ui.notify(f"Imported {len(hist)} measurement(s).", type="positive")
            state._left.refresh()
            state._right.refresh()

        ui.upload(on_upload=import_history, auto_upload=True, label="Import history CSV").props(
            "flat dense accept=.csv"
        ).classes("max-w-xs")


def _import_spec_control(state: UIState) -> None:
    """Upload a ProblemSpec JSON to populate the editor rows."""
    from abbot import io as abbot_io
    from abbot.problem import ContinuousDomain, EnumeratedDomain

    def on_upload(e) -> None:
        try:
            spec = abbot_io.spec_from_json(e.content.read().decode())
        except Exception as exc:  # noqa: BLE001
            ui.notify(str(exc), type="negative", multi_line=True, close_button=True)
            return
        rows = []
        for d in spec.domain:
            if isinstance(d, ContinuousDomain):
                rows.append({"kind": "continuous", "name": d.name, "lower": d.lower, "upper": d.upper})
            elif isinstance(d, EnumeratedDomain):
                rows.append({"kind": "enumerated", "name": d.name,
                             "values": ", ".join(str(v) for v in d.values)})
        state.variable_rows = rows or state.variable_rows
        state.objective_rows = [{"name": o.name, "direction": o.direction} for o in spec.outputs]
        ui.notify("Spec loaded into the editor.", type="positive")
        state._left.refresh()

    ui.upload(on_upload=on_upload, auto_upload=True, label="Import spec JSON").props(
        "flat dense accept=.json"
    ).classes("max-w-xs")


def _reset(state: UIState) -> None:
    state.auto_running = False
    state.controller = None
    state.spec = None
    state.pending = None
    state._left.refresh()
    state._right.refresh()


def _manual_controls(state: UIState) -> None:
    ctrl = state.controller
    ui.markdown("**Manual** — suggest, run in the lab, enter the value(s).").classes(
        "text-sm text-gray-600"
    )

    def do_suggest() -> None:
        try:
            state.pending = ctrl.suggest(1)
        except Exception as exc:  # noqa: BLE001
            ui.notify(str(exc), type="negative")
            return
        state._left.refresh()

    ui.button("Suggest next experiment", icon="lightbulb", on_click=do_suggest).props("color=indigo")

    if state.pending:
        cand = state.pending[0]
        with ui.card().classes("bg-indigo-50 w-full"):
            ui.label("Run at:").classes("font-semibold text-indigo-900 text-sm")
            with ui.row().classes("gap-3 flex-wrap"):
                for name, val in cand.items():
                    with ui.column().classes("gap-0"):
                        ui.label(name).classes("text-xs text-gray-500")
                        ui.label(f"{val:.4g}").classes("font-mono")
            inputs = {}
            with ui.row().classes("gap-2 flex-wrap"):
                for o in state.spec.outputs:  # every modeled output, incl. OBSERVE
                    inputs[o.name] = ui.number(f"{o.name}").props("outlined dense").classes("w-28")

            def do_submit() -> None:
                vals = [inputs[o.name].value for o in state.spec.outputs]
                if any(v is None for v in vals):
                    ui.notify("Enter all measured values first.", type="warning")
                    return
                ctrl.submit(torch.tensor(vals, dtype=torch.double))
                state.pending = None
                ui.notify("Recorded.", type="positive")
                state._left.refresh()
                # First measurement builds the viz structure; later ones update in place.
                if len(ctrl.campaign.history) <= 1 or state._plot_conv is None:
                    state._right.refresh()
                else:
                    _update_viz(state)

            ui.button("Submit", icon="check", on_click=do_submit).props("color=indigo")


def _auto_controls(state: UIState) -> None:
    ctrl = state.controller
    ui.markdown("**Mock / auto-run** — the synthetic experiment drives the loop.").classes(
        "text-sm text-gray-600"
    )

    def do_run() -> None:
        state.auto_running = True
        ctrl.status = RunStatus.RUNNING
        state._left.refresh()

    def do_pause() -> None:
        state.auto_running = False
        ctrl.pause()
        state._left.refresh()

    def do_step() -> None:
        n_before = len(ctrl.campaign.history)
        try:
            ctrl.step_once()
        except Exception as exc:  # noqa: BLE001
            ui.notify(str(exc), type="negative")
            return
        state._left.refresh()
        if n_before == 0 or state._plot_conv is None:
            state._right.refresh()  # build structure on the first point
        else:
            _update_viz(state)

    def do_stop() -> None:
        state.auto_running = False
        ctrl.stop()
        state._left.refresh()
        state._right.refresh()

    with ui.row().classes("gap-1"):
        ui.button("Run", icon="play_arrow", on_click=do_run).props("color=indigo")
        ui.button("Pause", icon="pause", on_click=do_pause).props("flat color=indigo")
        ui.button("Step", icon="skip_next", on_click=do_step).props("flat color=indigo")
        ui.button("Stop", icon="stop", on_click=do_stop).props("flat color=red")


# --------------------------------------------------------------------------- #
# Right panel: visualization
# --------------------------------------------------------------------------- #


def _viz_panel(state: UIState) -> None:
    """Build the viz STRUCTURE once (plots/table widgets), then update in place.

    Rebuilt only on rare events (create / reset / selector change); per-step data
    goes through ``_update_viz`` which calls ``update_figure`` on the existing
    plot widgets, so the panels update smoothly instead of being torn down and
    recreated each tick (the stutter/jitter the user saw).
    """
    if state.controller is None or len(state.controller.campaign.history) == 0:
        with ui.card().classes("w-full"), ui.row().classes("items-center gap-2 text-gray-500"):
            ui.icon("insights").classes("text-gray-400")
            ui.label("Create a campaign and take a few steps to see model + progress.")
        return

    spec = state.spec
    model = state.fitted_model()

    if model is not None and (spec.n_outputs > 1 or len(spec.domain) > 2):
        with ui.row().classes("items-center gap-2"), ui.card().classes("w-full"):
            if spec.n_outputs > 1:
                ui.select(
                    {j: o.name for j, o in enumerate(spec.outputs)},
                    value=state.viz_obj, label="output",
                    on_change=lambda e: _reviz(state, "viz_obj", e.value),
                ).props("dense outlined")
            if len(spec.domain) > 2:
                names = {i: n for i, n in enumerate(spec.variable_names)}
                ui.select(names, value=state.viz_vx, label="x-axis var",
                          on_change=lambda e: _reviz(state, "viz_vx", e.value)).props("dense outlined")
                ui.select(names, value=state.viz_vy, label="y-axis var",
                          on_change=lambda e: _reviz(state, "viz_vy", e.value)).props("dense outlined")

    with ui.row().classes("w-full no-wrap gap-3"):
        with ui.card().classes("grow"):
            state._plot_model = ui.plotly(
                _model_fig(state, model) if model is not None else _blank_fig("model")
            ).classes("w-full")
        with ui.card().classes("grow"):
            state._plot_conv = ui.plotly(_convergence_fig(state)).classes("w-full")

    with ui.card().classes("w-full"):
        state._plot_parity = ui.plotly(
            _parity_fig(state, model) or _blank_fig("parity") if model is not None
            else _blank_fig("parity")
        ).classes("w-full")

    state._rec_box = ui.column().classes("w-full gap-0")
    with ui.expansion("History", icon="table_rows").classes("w-full"):
        state._table = ui.table(
            columns=_history_columns(spec), rows=_history_rows(state), row_key="step"
        ).classes("w-full")

    _update_recommendation(state)


def _update_viz(state: UIState) -> None:
    """Push new data into the existing plot/table widgets (no teardown)."""
    if getattr(state, "_plot_conv", None) is None or state.controller is None:
        return
    if len(state.controller.campaign.history) == 0:
        return
    model = state.fitted_model()
    state._plot_conv.update_figure(_convergence_fig(state))
    if model is not None:
        state._plot_model.update_figure(_model_fig(state, model))
        pf = _parity_fig(state, model)
        if pf is not None:
            state._plot_parity.update_figure(pf)
    if state._table is not None:
        state._table.rows = _history_rows(state)
        state._table.update()
    _update_recommendation(state)


def _blank_fig(_name: str) -> dict:
    return {"data": [], "layout": {**_PLOT_LAYOUT, "title": "building…"}}


def _reviz(state: UIState, attr: str, value) -> None:
    """A viz selector changed: update the selection and re-render the right panel."""
    setattr(state, attr, value)
    right = getattr(state, "_right", None)
    if right is not None:
        right.refresh()


def _status_chip(ctrl: RunController) -> None:
    colors = {
        RunStatus.IDLE: ("grey", "idle"), RunStatus.RUNNING: ("green", "running"),
        RunStatus.PAUSED: ("amber", "paused"), RunStatus.DONE: ("blue", "done"),
        RunStatus.HALTED: ("red", "halted"),
    }
    color, text = colors.get(ctrl.status, ("grey", str(ctrl.status)))
    ui.badge(text).props(f"color={color}")
    if ctrl.campaign.halted:
        ui.label(f"⚠ {ctrl.campaign.halted}").classes("text-red-600 text-sm")


def _update_recommendation(state: UIState) -> None:
    """Repopulate the small recommendation box in place (cheap; no plot teardown)."""
    box = getattr(state, "_rec_box", None)
    if box is None:
        return
    box.clear()
    rec = state.controller.campaign.recommend()
    if not rec:
        return
    hist = state.controller.campaign.history
    spec = state.spec
    with box, ui.card().classes("bg-green-50 w-full"):
        ui.label("Recommended" + (" (Pareto set)" if len(rec) > 1 else " design")).classes(
            "font-semibold text-green-900"
        )
        for i in rec:
            r = hist.records[i]
            setting = ", ".join(
                f"{n} = {v:.4g}" for n, v in zip(spec.variable_names, r.candidate.reshape(-1).tolist())
            )
            outs = ", ".join(
                f"{o.name} = {float(r.outputs[j]):.4g}" for j, o in enumerate(spec.outputs)
            )
            ui.label(f"{setting}  →  {outs}").classes("font-mono text-sm")


def _history_columns(spec) -> list[dict]:
    cols = [{"name": "step", "label": "step", "field": "step", "align": "left"}]
    cols += [{"name": n, "label": n, "field": n} for n in spec.variable_names]
    cols += [{"name": o.name, "label": o.name, "field": o.name} for o in spec.outputs]
    return cols


def _history_rows(state: UIState) -> list[dict]:
    hist = state.controller.campaign.history
    spec = state.spec
    return [
        {
            "step": i,
            **{n: round(float(v), 4) for n, v in zip(spec.variable_names, r.candidate.reshape(-1).tolist())},
            **{o.name: round(float(r.outputs[j]), 4) for j, o in enumerate(spec.outputs)},
        }
        for i, r in enumerate(hist.records)
        if r.outputs is not None
    ]


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> None:
    """Entry point: build the page and launch the server.

    Flags (also honored via env vars for headless/cluster runs):
      --host (ABBOT_UI_HOST, default 127.0.0.1)
      --port (ABBOT_UI_PORT, default 8080) — auto-picks a free port if taken
      --strict-port — fail instead of auto-picking
      --show / --no-show (ABBOT_UI_SHOW=1/0) — open a browser tab on start.
    """
    import argparse
    import os

    def _env_bool(name: str, default: bool) -> bool:
        val = os.environ.get(name)
        return default if val is None else val.strip().lower() in {"1", "true", "yes", "on"}

    parser = argparse.ArgumentParser(prog="abbot.ui", description="Launch the abbot lab frontend.")
    parser.add_argument("--host", default=os.environ.get("ABBOT_UI_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ABBOT_UI_PORT", "8080")))
    show_default = _env_bool("ABBOT_UI_SHOW", True)
    show_group = parser.add_mutually_exclusive_group()
    show_group.add_argument("--show", dest="show", action="store_true", default=show_default,
                            help="open a browser tab on start (default locally)")
    show_group.add_argument("--no-show", dest="show", action="store_false",
                            help="do not open a browser (use on a headless/cluster host)")
    parser.add_argument("--strict-port", action="store_true",
                        help="fail if the requested port is taken instead of auto-picking a free one")
    args = parser.parse_args(argv)

    port = args.port
    if not args.strict_port:
        port = _free_port(args.host, args.port)
        if port != args.port:
            logger.warning("port %d is in use; using %d instead.", args.port, port)
            print(f"abbot.ui: port {args.port} busy -> using {port}")

    @ui.page("/")
    def index() -> None:
        _build_page()

    ui.run(title="abbot", host=args.host, port=port, show=args.show, reload=False)


def _free_port(host: str, preferred: int) -> int:
    """Return ``preferred`` if it can be bound, else an OS-assigned free port."""
    import socket

    for candidate in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, candidate))
                return sock.getsockname()[1]
            except OSError:
                continue
    return preferred


if __name__ in {"__main__", "__mp_main__"}:
    main()
