"""NiceGUI web client for abbot — a thin UI over :class:`abbot.session.Campaign`.

Three tabs: Define -> Run -> Progress. All optimization logic is delegated to the
core (``Campaign``, ``RunController``, ``builder``, ``analysis``); this module only
builds widgets and wires events. Per-client state lives on a ``UIState`` created
inside the page function (NiceGUI gives each connection its own) — no module-global
``ProgramState`` trap like BAXstics had.

Two run modes share one ``RunController``:
  - Manual: the user enters each measurement.
  - Auto (mock): a ``SyntheticExperiment`` is attached to the campaign, which
    enables run/pause/step/stop — the campaign drives itself with no typed values.

Run with ``python -m abbot.ui``.
"""

from __future__ import annotations

import logging

import torch
from nicegui import ui

from abbot.analysis import convergence, hypervolume
from abbot.experiment import SyntheticExperiment
from abbot.model import default_constructor
from abbot.policy import ParetoBO, SimpleBO
from abbot.session import Campaign
from abbot.ui.builder import build_spec, choose_optimizer
from abbot.ui.controller import RunController, RunStatus

logger = logging.getLogger(__name__)

# A calm, legible Plotly layout shared by both progress plots.
_PLOT_LAYOUT = {
    "height": 340,
    "margin": {"l": 60, "r": 20, "t": 48, "b": 48},
    "template": "plotly_white",
    "font": {"size": 13},
    "showlegend": False,
}
_ACCENT = "#4f46e5"


class UIState:
    """Per-client session state: editor rows + the live controller."""

    def __init__(self) -> None:
        self.variable_rows: list[dict] = [
            {"kind": "continuous", "name": "x1", "lower": 0.0, "upper": 1.0}
        ]
        self.objective_rows: list[dict] = [{"name": "y", "direction": "MAXIMIZE"}]
        self.controller: RunController | None = None
        self.spec = None
        self.pending: list[dict] | None = None
        self.mock: bool = False

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
        # Mock mode attaches a synthetic experiment -> auto-run becomes available.
        if self.mock:
            exp = SyntheticExperiment(spec, seed=0, noise=0.0)
            self.controller = RunController(campaign, experiment=exp, evaluate=lambda o: o)
        else:
            self.controller = RunController(campaign)


# --------------------------------------------------------------------------- #
# Editors
# --------------------------------------------------------------------------- #


def _variable_editor(state: UIState) -> None:
    with ui.row().classes("items-center gap-2"):
        ui.icon("tune").classes("text-indigo-600")
        ui.label("Design variables").classes("text-lg font-semibold")
    container = ui.column().classes("gap-2 w-full")

    def redraw() -> None:
        container.clear()
        with container:
            for i, row in enumerate(state.variable_rows):
                with ui.row().classes("items-center gap-2 w-full"):
                    ui.select(
                        {"continuous": "Continuous", "enumerated": "Enumerated"},
                        value=row["kind"],
                        on_change=lambda e, r=row: (_set(r, "kind", e.value), redraw()),
                    ).props("dense outlined").classes("w-36")
                    ui.input("name", value=row.get("name", "")).bind_value(row, "name").props(
                        "dense outlined"
                    ).classes("w-28")
                    if row["kind"] == "continuous":
                        ui.number("lower", value=row.get("lower", 0.0)).bind_value(row, "lower").props(
                            "dense outlined"
                        ).classes("w-28")
                        ui.number("upper", value=row.get("upper", 1.0)).bind_value(row, "upper").props(
                            "dense outlined"
                        ).classes("w-28")
                    else:
                        ui.input(
                            "values (comma/space-separated)", value=row.get("values", "0, 1, 2")
                        ).bind_value(row, "values").props("dense outlined").classes("w-64")
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
        ui.label("Objectives").classes("text-lg font-semibold")
    container = ui.column().classes("gap-2 w-full")

    def redraw() -> None:
        container.clear()
        with container:
            for i, row in enumerate(state.objective_rows):
                with ui.row().classes("items-center gap-2 w-full"):
                    ui.input("name", value=row.get("name", "")).bind_value(row, "name").props(
                        "dense outlined"
                    ).classes("w-40")
                    ui.select(
                        {"MAXIMIZE": "Maximize", "MINIMIZE": "Minimize"},
                        value=row["direction"],
                    ).bind_value(row, "direction").props("dense outlined").classes("w-40")
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
# Plot builders
# --------------------------------------------------------------------------- #


def _progress_figure(state: UIState):
    hist = state.controller.campaign.history
    spec = state.spec
    if spec.n_objectives == 1:
        series = convergence(hist, spec)
        return {
            "data": [
                {
                    "y": series,
                    "x": list(range(len(series))),
                    "type": "scatter",
                    "mode": "lines+markers",
                    "line": {"color": _ACCENT, "width": 2},
                    "marker": {"color": _ACCENT, "size": 7},
                }
            ],
            "layout": {
                **_PLOT_LAYOUT,
                "title": f"Best {spec.objectives[0].name} so far",
                "xaxis": {"title": "step"},
                "yaxis": {"title": spec.objectives[0].name},
            },
        }
    series = hypervolume(hist, spec)
    return {
        "data": [
            {
                "y": series,
                "x": list(range(len(series))),
                "type": "scatter",
                "mode": "lines+markers",
                "line": {"color": _ACCENT, "width": 2},
                "marker": {"color": _ACCENT, "size": 7},
            }
        ],
        "layout": {
            **_PLOT_LAYOUT,
            "title": "Dominated hypervolume",
            "xaxis": {"title": "step"},
            "yaxis": {"title": "hypervolume"},
        },
    }


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #


def _build_page() -> None:
    state = UIState()

    with ui.header().classes("items-center bg-indigo-700"), ui.row().classes(
        "items-center gap-2"
    ):
        ui.icon("science").classes("text-white text-2xl")
        ui.label("abbot").classes("text-white text-xl font-bold")
        ui.label("a Bayesian guide for lab experiments").classes("text-indigo-200 text-sm")

    with ui.tabs().classes("w-full") as tabs:
        tab_define = ui.tab("1 · Define", icon="edit")
        tab_run = ui.tab("2 · Run", icon="play_arrow")
        tab_progress = ui.tab("3 · Progress", icon="insights")

    with ui.tab_panels(tabs, value=tab_define).classes("w-full"):
        # ---- Define ----
        with ui.tab_panel(tab_define), ui.column().classes("gap-4 max-w-3xl"):
            with ui.card().classes("w-full"):
                _variable_editor(state)
            with ui.card().classes("w-full"):
                _objective_editor(state)
            with ui.card().classes("w-full"):
                mock_toggle = ui.switch(
                    "Mock mode — attach a synthetic experiment (enables auto-run, no "
                    "typing of measurements)"
                ).bind_value(state, "mock")
                mock_toggle.props("color=indigo")

            def do_start() -> None:
                try:
                    state.start_campaign()
                except Exception as exc:  # noqa: BLE001 - surface build/validation errors as a toast
                    ui.notify(str(exc), type="negative", multi_line=True, close_button=True)
                    return
                mode = "mock / auto-run" if state.mock else "manual"
                ui.notify(
                    f"Campaign ready ({len(state.spec.domain)} variable(s), "
                    f"{state.spec.n_objectives} objective(s), {mode}). Go to Run.",
                    type="positive",
                )
                tabs.set_value(tab_run)
                run_panel.refresh()
                progress_panel.refresh()

            ui.button("Create campaign", icon="rocket_launch", on_click=do_start).props(
                "color=indigo size=lg"
            )

        # ---- Run ----
        with ui.tab_panel(tab_run), ui.column().classes("gap-4 w-full").style("max-width: 60rem"):

            @ui.refreshable
            def run_panel() -> None:
                if state.controller is None:
                    _empty("Create a campaign on the Define tab first.")
                    return
                ctrl = state.controller

                with ui.row().classes("items-center gap-3"):
                    _status_chip(ctrl)
                    ui.label(f"{len(ctrl.campaign.history)} experiment(s) recorded").classes(
                        "text-sm text-gray-600"
                    )

                if ctrl.can_auto:
                    _auto_controls(state)
                else:
                    _manual_controls(state)

                ui.separator()
                ui.label("Live progress").classes("text-sm font-semibold text-gray-700")
                mini_progress()

            @ui.refreshable
            def mini_progress() -> None:
                if state.controller is None or len(state.controller.campaign.history) == 0:
                    _empty("No measurements yet.")
                    return
                ui.plotly(_progress_figure(state)).classes("w-full")
                _recommendation(state)

            run_panel()

        # ---- Progress ----
        with ui.tab_panel(tab_progress), ui.column().classes("gap-4 w-full").style("max-width: 60rem"):

            @ui.refreshable
            def progress_panel() -> None:
                if state.controller is None or len(state.controller.campaign.history) == 0:
                    _empty("No measurements yet — create and run a campaign.")
                    return
                ui.plotly(_progress_figure(state)).classes("w-full")
                _recommendation(state)
                _history_table(state)

            progress_panel()

    # Expose refreshers on state so control handlers can trigger redraws.
    state._run_panel = run_panel
    state._mini_progress = mini_progress
    state._progress_panel = progress_panel


# --------------------------------------------------------------------------- #
# Run-tab controls
# --------------------------------------------------------------------------- #


def _manual_controls(state: UIState) -> None:
    ctrl = state.controller
    ui.markdown(
        "**Manual mode** — suggest an experiment, run it in the lab, enter the "
        "measured value(s). _(Enable Mock mode on Define for hands-free auto-run.)_"
    ).classes("text-sm text-gray-600")

    def do_suggest() -> None:
        try:
            state.pending = ctrl.suggest(1)
        except Exception as exc:  # noqa: BLE001
            ui.notify(str(exc), type="negative")
            return
        suggestion_card.refresh()

    ui.button("Suggest next experiment", icon="lightbulb", on_click=do_suggest).props(
        "color=indigo"
    )

    @ui.refreshable
    def suggestion_card() -> None:
        if not state.pending:
            return
        cand = state.pending[0]
        with ui.card().classes("bg-indigo-50 w-full"):
            ui.label("Run the experiment at:").classes("font-semibold text-indigo-900")
            with ui.row().classes("gap-4 flex-wrap"):
                for name, val in cand.items():
                    with ui.column().classes("gap-0"):
                        ui.label(name).classes("text-xs text-gray-500")
                        ui.label(f"{val:.4g}").classes("text-lg font-mono")
            ui.separator()
            inputs = {}
            with ui.row().classes("gap-3 flex-wrap"):
                for obj in state.spec.objectives:
                    inputs[obj.name] = ui.number(f"measured {obj.name}").props("outlined dense")

            def do_submit() -> None:
                vals = [inputs[o.name].value for o in state.spec.objectives]
                if any(v is None for v in vals):
                    ui.notify("Enter all measured values first.", type="warning")
                    return
                state.controller.submit(torch.tensor(vals, dtype=torch.double))
                state.pending = None
                ui.notify("Measurement recorded.", type="positive")
                suggestion_card.refresh()
                state._run_panel.refresh()
                state._mini_progress.refresh()
                state._progress_panel.refresh()

            ui.button("Submit measurement", icon="check", on_click=do_submit).props("color=indigo")

    suggestion_card()


def _auto_controls(state: UIState) -> None:
    ctrl = state.controller
    ui.markdown(
        "**Mock / auto-run** — a synthetic experiment is attached; the campaign "
        "drives itself. Use the controls below."
    ).classes("text-sm text-gray-600")

    # A per-run flag the Pause/Stop buttons flip; the timer checks it each tick.
    flags = {"running": False}

    def _refresh_all() -> None:
        state._run_panel.refresh()
        state._mini_progress.refresh()
        state._progress_panel.refresh()

    def tick() -> None:
        if not flags["running"]:
            return
        if ctrl.status in (RunStatus.HALTED, RunStatus.DONE):
            flags["running"] = False
            _refresh_all()
            return
        ctrl.step_once()
        state._mini_progress.refresh()

    timer = ui.timer(0.6, tick, active=False)

    def do_run() -> None:
        flags["running"] = True
        ctrl.status = RunStatus.RUNNING
        timer.activate()
        state._run_panel.refresh()

    def do_pause() -> None:
        flags["running"] = False
        ctrl.pause()
        timer.deactivate()
        _refresh_all()

    def do_step() -> None:
        try:
            ctrl.step_once()
        except Exception as exc:  # noqa: BLE001
            ui.notify(str(exc), type="negative")
            return
        _refresh_all()

    def do_stop() -> None:
        flags["running"] = False
        ctrl.stop()
        timer.deactivate()
        _refresh_all()

    with ui.row().classes("gap-2"):
        ui.button("Run", icon="play_arrow", on_click=do_run).props("color=indigo")
        ui.button("Pause", icon="pause", on_click=do_pause).props("flat color=indigo")
        ui.button("Step", icon="skip_next", on_click=do_step).props("flat color=indigo")
        ui.button("Stop", icon="stop", on_click=do_stop).props("flat color=red")


# --------------------------------------------------------------------------- #
# Shared widgets
# --------------------------------------------------------------------------- #


def _status_chip(ctrl: RunController) -> None:
    colors = {
        RunStatus.IDLE: ("grey", "idle"),
        RunStatus.RUNNING: ("green", "running"),
        RunStatus.PAUSED: ("amber", "paused"),
        RunStatus.DONE: ("blue", "done"),
        RunStatus.HALTED: ("red", "halted"),
    }
    color, text = colors.get(ctrl.status, ("grey", str(ctrl.status)))
    ui.badge(text).props(f"color={color}").classes("text-sm")
    if ctrl.campaign.halted:
        ui.label(f"⚠ {ctrl.campaign.halted}").classes("text-red-600 text-sm")


def _recommendation(state: UIState) -> None:
    rec = state.controller.campaign.recommend()
    if not rec:
        return
    hist = state.controller.campaign.history
    spec = state.spec
    with ui.card().classes("bg-green-50 w-full"):
        label = "Recommended design" + ("s (Pareto set)" if len(rec) > 1 else "")
        ui.label(label).classes("font-semibold text-green-900")
        for i in rec:
            r = hist.records[i]
            setting = ", ".join(
                f"{n} = {v:.4g}"
                for n, v in zip(spec.variable_names, r.candidate.reshape(-1).tolist())
            )
            outs = ", ".join(
                f"{o.name} = {float(r.outputs[j]):.4g}" for j, o in enumerate(spec.objectives)
            )
            ui.label(f"{setting}  →  {outs}").classes("font-mono text-sm")


def _history_table(state: UIState) -> None:
    hist = state.controller.campaign.history
    spec = state.spec
    rows = [
        {
            "step": i,
            **{n: round(float(v), 4) for n, v in zip(spec.variable_names, r.candidate.reshape(-1).tolist())},
            **{o.name: round(float(r.outputs[j]), 4) for j, o in enumerate(spec.objectives)},
        }
        for i, r in enumerate(hist.records)
        if r.outputs is not None
    ]
    cols = [{"name": "step", "label": "step", "field": "step", "align": "left"}]
    cols += [{"name": n, "label": n, "field": n} for n in spec.variable_names]
    cols += [{"name": o.name, "label": o.name, "field": o.name} for o in spec.objectives]
    ui.table(columns=cols, rows=rows, row_key="step").classes("w-full")


def _empty(msg: str) -> None:
    with ui.row().classes("items-center gap-2 text-gray-500"):
        ui.icon("info").classes("text-gray-400")
        ui.label(msg)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> None:
    """Entry point: build the page and launch the server.

    Flags (also honored via env vars so a headless/cluster run can set them):
      --host (ABBOT_UI_HOST, default 127.0.0.1)
      --port (ABBOT_UI_PORT, default 8080) — auto-picks a free port if taken
      --strict-port — fail instead of auto-picking
      --show / --no-show (ABBOT_UI_SHOW=1/0) — open a browser tab on start.
    On a cluster, use --no-show and tunnel the port; locally, --show pops a tab.
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

    for candidate in (preferred, 0):  # 0 -> let the OS pick any open port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, candidate))
                return sock.getsockname()[1]
            except OSError:
                continue
    return preferred  # both failed; let ui.run surface the error


if __name__ in {"__main__", "__mp_main__"}:
    main()
