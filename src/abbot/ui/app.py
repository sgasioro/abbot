"""NiceGUI web client for abbot — a thin UI over :class:`abbot.session.Campaign`.

Three tabs: Define problem -> Run loop (manual/auto) -> Progress. All optimization
logic is delegated to the core (``Campaign``, ``RunController``, ``builder``,
``analysis``); this module only builds widgets and wires their events. Per-client
state lives on a ``UIState`` instance created inside the page function (NiceGUI
gives each browser connection its own), so there is no module-global ``ProgramState``
trap like BAXstics had.

Run it with ``python -m abbot.ui``.
"""

from __future__ import annotations

import logging

import torch
from nicegui import ui

from abbot.analysis import convergence, hypervolume
from abbot.model import default_constructor
from abbot.policy import ParetoBO, SimpleBO
from abbot.session import Campaign
from abbot.ui.builder import build_spec, choose_optimizer
from abbot.ui.controller import RunController

logger = logging.getLogger(__name__)


class UIState:
    """Per-client session state. Holds the editor rows and the live controller."""

    def __init__(self) -> None:
        self.variable_rows: list[dict] = [
            {"kind": "continuous", "name": "x1", "lower": 0.0, "upper": 1.0}
        ]
        self.objective_rows: list[dict] = [{"name": "y", "direction": "MAXIMIZE"}]
        self.controller: RunController | None = None
        self.spec = None
        self.pending: list[dict] | None = None

    # -- building the campaign from the editor --

    def start_campaign(self) -> None:
        spec = build_spec(self.variable_rows, self.objective_rows)  # may raise -> caught by caller
        optimizer = choose_optimizer(spec)  # may raise on mixed spaces
        policy = SimpleBO(optimizer) if spec.n_objectives == 1 else ParetoBO(optimizer)
        campaign = Campaign(
            spec, policy, default_constructor(spec, bounds=spec.bounds_tensor())
        )
        # Seed with a few space-filling points so the first fit has data.
        bounds = spec.bounds_tensor()
        lo, hi = bounds[0], bounds[1]
        seeds = lo + (hi - lo) * torch.rand(max(3, spec.n_objectives + 1), bounds.shape[-1], dtype=torch.double)
        campaign.seed(seeds)
        self.spec = spec
        self.controller = RunController(campaign)  # manual: no programmatic experiment


def _variable_editor(state: UIState) -> None:
    ui.label("Design variables").classes("text-lg font-bold")
    container = ui.column()

    def redraw() -> None:
        container.clear()
        with container:
            for i, row in enumerate(state.variable_rows):
                with ui.row().classes("items-center"):
                    ui.select(
                        {"continuous": "Continuous", "enumerated": "Enumerated"},
                        value=row["kind"],
                        on_change=lambda e, r=row: (_set(r, "kind", e.value), redraw()),
                    ).props("dense")
                    ui.input("name", value=row.get("name", "")).bind_value(row, "name").props("dense")
                    if row["kind"] == "continuous":
                        ui.number("lower", value=row.get("lower", 0.0)).bind_value(row, "lower").props("dense")
                        ui.number("upper", value=row.get("upper", 1.0)).bind_value(row, "upper").props("dense")
                    else:
                        ui.input(
                            "values (comma/space-separated)", value=row.get("values", "0, 1, 2")
                        ).bind_value(row, "values").props("dense").classes("w-64")
                    ui.button(icon="delete", on_click=lambda _, idx=i: (_pop(state.variable_rows, idx), redraw())).props("flat dense")

    def add() -> None:
        state.variable_rows.append({"kind": "continuous", "name": f"x{len(state.variable_rows) + 1}", "lower": 0.0, "upper": 1.0})
        redraw()

    redraw()
    ui.button("+ variable", on_click=add).props("flat")


def _objective_editor(state: UIState) -> None:
    ui.label("Objectives").classes("text-lg font-bold")
    container = ui.column()

    def redraw() -> None:
        container.clear()
        with container:
            for i, row in enumerate(state.objective_rows):
                with ui.row().classes("items-center"):
                    ui.input("name", value=row.get("name", "")).bind_value(row, "name").props("dense")
                    ui.select(
                        {"MAXIMIZE": "Maximize", "MINIMIZE": "Minimize"},
                        value=row["direction"],
                    ).bind_value(row, "direction").props("dense")
                    ui.button(icon="delete", on_click=lambda _, idx=i: (_pop(state.objective_rows, idx), redraw())).props("flat dense")

    def add() -> None:
        state.objective_rows.append({"name": f"y{len(state.objective_rows) + 1}", "direction": "MAXIMIZE"})
        redraw()

    redraw()
    ui.button("+ objective", on_click=add).props("flat")


def _set(row: dict, key: str, value) -> None:
    row[key] = value


def _pop(rows: list, idx: int) -> None:
    if 0 <= idx < len(rows) and len(rows) > 1:
        rows.pop(idx)


def _build_page() -> None:
    state = UIState()

    ui.markdown("## abbot — a Bayesian guide for lab experiments")
    with ui.tabs() as tabs:
        tab_define = ui.tab("1. Define")
        tab_run = ui.tab("2. Run")
        tab_progress = ui.tab("3. Progress")

    with ui.tab_panels(tabs, value=tab_define).classes("w-full"):
        # --- Define ---
        with ui.tab_panel(tab_define):
            _variable_editor(state)
            ui.separator()
            _objective_editor(state)
            ui.separator()

            def do_start() -> None:
                try:
                    state.start_campaign()
                except Exception as exc:  # noqa: BLE001 - surface any build/validation error as a toast
                    ui.notify(str(exc), type="negative", multi_line=True, close_button=True)
                    return
                ui.notify(
                    f"Campaign ready: {len(state.spec.domain)} variable(s), "
                    f"{state.spec.n_objectives} objective(s). Go to the Run tab.",
                    type="positive",
                )
                tabs.set_value(tab_run)
                run_panel.refresh()

            ui.button("Create campaign", on_click=do_start).props("color=primary")

        # --- Run (manual) ---
        with ui.tab_panel(tab_run):
            run_container = ui.column().classes("w-full")

            @ui.refreshable
            def run_panel() -> None:
                run_container.clear()
                with run_container:
                    if state.controller is None:
                        ui.label("Create a campaign on the Define tab first.")
                        return
                    ui.label(
                        "Manual mode: suggest an experiment, run it in the lab, enter "
                        "the measured value(s)."
                    )
                    ui.markdown(
                        "_Auto mode is available in the library when a programmatic "
                        "`Experiment` is attached; this UI session is manual._"
                    ).classes("text-sm text-gray-500")

                    def do_suggest() -> None:
                        try:
                            state.pending = state.controller.suggest(1)
                        except Exception as exc:  # noqa: BLE001 - surface proposal errors as a toast
                            ui.notify(str(exc), type="negative")
                            return
                        suggestion_panel.refresh()

                    ui.button("Suggest next experiment", on_click=do_suggest).props("color=primary")

                    @ui.refreshable
                    def suggestion_panel() -> None:
                        if not state.pending:
                            return
                        cand = state.pending[0]
                        with ui.card():
                            ui.label("Run the experiment at:").classes("font-bold")
                            for name, val in cand.items():
                                ui.label(f"  {name} = {val:.4g}")
                            inputs = {}
                            for obj in state.spec.objectives:
                                inputs[obj.name] = ui.number(f"measured {obj.name}")

                            def do_submit() -> None:
                                vals = [inputs[o.name].value for o in state.spec.objectives]
                                if any(v is None for v in vals):
                                    ui.notify("Enter all measured values first.", type="warning")
                                    return
                                state.controller.submit(torch.tensor(vals, dtype=torch.double))
                                state.pending = None
                                ui.notify("Measurement recorded.", type="positive")
                                suggestion_panel.refresh()
                                progress_panel.refresh()

                            ui.button("Submit measurement", on_click=do_submit).props("color=primary")

                    suggestion_panel()

            run_panel()

        # --- Progress ---
        with ui.tab_panel(tab_progress):
            progress_container = ui.column().classes("w-full")

            @ui.refreshable
            def progress_panel() -> None:
                progress_container.clear()
                with progress_container:
                    if state.controller is None or len(state.controller.campaign.history) == 0:
                        ui.label("No measurements yet.")
                        return
                    hist = state.controller.campaign.history
                    spec = state.spec
                    try:
                        if spec.n_objectives == 1:
                            series = convergence(hist, spec)
                            fig = {
                                "data": [{"y": series, "type": "scatter", "mode": "lines+markers", "name": "best"}],
                                "layout": {"title": "Best objective vs. step", "height": 320},
                            }
                        else:
                            series = hypervolume(hist, spec)
                            fig = {
                                "data": [{"y": series, "type": "scatter", "mode": "lines+markers", "name": "hypervolume"}],
                                "layout": {"title": "Dominated hypervolume vs. step", "height": 320},
                            }
                        ui.plotly(fig).classes("w-full")
                    except Exception as exc:  # noqa: BLE001 - a plot failure must not break the page
                        ui.label(f"(plot unavailable: {exc})")

                    rec = state.controller.campaign.recommend()
                    if rec:
                        ui.label("Recommended design(s):").classes("font-bold")
                        for i in rec:
                            r = hist.records[i]
                            setting = ", ".join(
                                f"{n}={v:.4g}" for n, v in zip(spec.variable_names, r.candidate.reshape(-1).tolist())
                            )
                            ui.label(f"  {setting}")

                    rows = [
                        {
                            "step": i,
                            **{n: round(float(v), 4) for n, v in zip(spec.variable_names, r.candidate.reshape(-1).tolist())},
                            **{o.name: round(float(r.outputs[j]), 4) for j, o in enumerate(spec.objectives)},
                        }
                        for i, r in enumerate(hist.records)
                        if r.outputs is not None
                    ]
                    cols = [{"name": "step", "label": "step", "field": "step"}]
                    cols += [{"name": n, "label": n, "field": n} for n in spec.variable_names]
                    cols += [{"name": o.name, "label": o.name, "field": o.name} for o in spec.objectives]
                    ui.table(columns=cols, rows=rows).classes("w-full")

            progress_panel()


def main(argv: list[str] | None = None) -> None:
    """Entry point: build the page and launch the server.

    Flags (also honored via env vars so a headless/cluster run can set them):
      --host (ABBOT_UI_HOST, default 127.0.0.1)
      --port (ABBOT_UI_PORT, default 8080)
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
