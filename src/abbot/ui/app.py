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
from nicegui import run, ui

from abbot.experiment import MockAnalyticExperiment
from abbot.policy import BAXPolicy, default_policy
from abbot.session import Campaign, default_initial_designs
from abbot.ui import figures
from abbot.ui.builder import (
    BAX_STRATEGY_LABELS,
    MODEL_LABELS,
    build_constructor,
    build_goal,
    build_spec,
    build_start,
    spec_to_rows,
)
from abbot.ui.controller import RunController, RunStatus

logger = logging.getLogger(__name__)


class UIState:
    """Per-client session state: editor rows, the live controller, viz selections."""

    def __init__(self) -> None:
        self.variable_rows: list[dict] = [
            {"kind": "continuous", "name": "x1", "lower": 0.0, "upper": 1.0}
        ]
        self.objective_rows: list[dict] = [{"name": "y", "direction": "MAXIMIZE"}]
        # Method: "optimize" (BO/MO) or "bax" (find a target region). BAX adds a
        # goal (type + params) and a strategy; the output directions become OBSERVE.
        self.method: str = "optimize"
        self.goal_row: dict = {"kind": "band", "percentile": 80.0}
        self.bax_strategy: str = "SwitchBAX"
        # An optional source variable (fidelity levels, two codes, sim vs. experiment).
        self.source_on: bool = False
        self.source_row: dict = {
            "kind": "enumerated", "name": "source", "values": "simulation, experiment",
            "target": "experiment", "lower": 0.0, "upper": 1.0, "log": False,
            "accurate": "high",
        }
        self.model: str = "default"  # a MODEL_LABELS key
        self.controller: RunController | None = None
        self.spec = None
        self.n_suggest: int = 1  # how many designs "Suggest" asks for
        self.mock: bool = False
        # Model-viz selections (which objective / which vars to slice).
        self.viz_obj = 0
        self.viz_vx = 0
        self.viz_vy = 1
        # Run-state is owned by the controller (ctrl.is_running / status); the UI no
        # longer keeps a parallel flag. The timer in _build_page drives ctrl.tick().
        # Steps run on a worker thread (a real experiment may take minutes); this
        # guard stops the timer or the Step button from starting a second one.
        self._stepping = False
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
        # In BAX mode the outputs are modeled-but-not-optimized (the goal consumes
        # them), so present them to the engine as OBSERVE regardless of the UI arrow.
        rows = self.objective_rows
        if self.method == "bax":
            rows = [{**r, "direction": "OBSERVE"} for r in self.objective_rows]
        source = self.source_row if self.source_on else None
        spec = build_spec(self.variable_rows, rows, source)  # may raise

        if self.method == "bax":
            goal = build_goal(self._normalized_goal_row(), spec)  # may raise
            policy = BAXPolicy(goal, strategy=self.bax_strategy)
        else:
            policy = default_policy(spec)
        # The GP constructor declares its own UQ capability (calibrated variance),
        # which satisfies every builtin policy; Campaign checks the pairing.
        # With travel limits the campaign starts from the current settings, not
        # from space-filling seeds.
        start = build_start(spec, self.variable_rows)  # may raise
        campaign = Campaign(
            spec, policy, build_constructor(spec, self.model), initial_designs=start
        )
        if start is None:
            campaign.seed(default_initial_designs(spec))
        self.spec = spec
        self.viz_obj = 0
        self.viz_vx = 0
        self.viz_vy = 1 if len(spec.domain) > 1 else 0
        if self.mock:
            exp = MockAnalyticExperiment(spec, seed=0, noise=0.0)
            self.controller = RunController(campaign, experiment=exp, evaluate=lambda o: o)
        else:
            self.controller = RunController(campaign)

    def _normalized_goal_row(self) -> dict:
        """Turn the UI goal row into the dict ``build_goal`` expects.

        For a band goal the editor stores per-output ``{name: {"lo": v, "hi": v}}``
        under ``bands``; collapse that into ``bounds = {name: (lo, hi)}`` dropping
        rows where both ends are blank (unconstrained).
        """
        row = dict(self.goal_row)
        if row.get("kind", "band") == "band":
            bounds: dict[str, tuple] = {}
            for name, band in (row.get("bands") or {}).items():
                lo, hi = band.get("lo"), band.get("hi")
                if lo is None and hi is None:
                    continue
                bounds[name] = (lo, hi)
            row["bounds"] = bounds
        return row

    def fitted_model(self):
        """The current fitted surrogate (canonical space) or None."""
        if self.controller is None:
            return None
        return self.controller.campaign.current_model()

    def selections(self) -> figures.VizSelections:
        """The current viz selections as the struct the figure builders consume."""
        return figures.VizSelections(
            output=self.viz_obj, var_x=self.viz_vx, var_y=self.viz_vy
        )

    def needs_structure_rebuild(self) -> bool:
        """Whether the right panel must be rebuilt (vs. updated in place).

        One rule for every step handler (was three divergent thresholds): the viz
        structure must be (re)built whenever the plot widgets don't exist yet — the
        panel was last drawn in its empty state. Once built, every later step
        updates in place. ``_plot_conv`` is the sentinel set in ``_viz_panel``.
        """
        return self._plot_conv is None

    def history(self):
        return self.controller.campaign.history

    def policy(self):
        return self.controller.campaign.policy


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
                with ui.row().classes("items-center gap-1 w-full no-wrap"):
                    ui.select(
                        {"continuous": "Continuous", "enumerated": "Enumerated"},
                        value=row["kind"],
                        on_change=lambda e, r=row: (_set(r, "kind", e.value), redraw()),
                    ).props("dense outlined").classes("w-32")
                    ui.input("name", value=row.get("name", "")).bind_value(row, "name").props(
                        "dense outlined"
                    ).classes("w-20")
                    if row["kind"] == "continuous":
                        ui.number("lo", value=row.get("lower", 0.0)).bind_value(row, "lower").props(
                            "dense outlined"
                        ).classes("w-16")
                        ui.number("hi", value=row.get("upper", 1.0)).bind_value(row, "upper").props(
                            "dense outlined"
                        ).classes("w-16")
                    else:
                        ui.input("values", value=row.get("values", "0, 1, 2")).bind_value(
                            row, "values"
                        ).props("dense outlined").classes("w-40")
                    ui.button(
                        icon="delete",
                        on_click=lambda _, idx=i: (_pop(state.variable_rows, idx), redraw()),
                    ).props("flat dense round color=grey")
                if row["kind"] == "continuous":
                    _continuous_options(row)

    def add() -> None:
        n = len(state.variable_rows) + 1
        state.variable_rows.append({"kind": "continuous", "name": f"x{n}", "lower": 0.0, "upper": 1.0})
        redraw()

    redraw()
    ui.button("Add variable", icon="add", on_click=add).props("flat dense color=indigo")


def _continuous_options(row: dict) -> None:
    """Log scale and a travel limit for one continuous variable (a second line)."""
    row.setdefault("log", False)
    row.setdefault("max_step", None)
    row.setdefault("start", None)
    with ui.row().classes("items-center gap-1 w-full pl-4"):
        ui.checkbox("log scale").bind_value(row, "log").props("dense").tooltip(
            "For a quantity spanning orders of magnitude (a concentration, a rate)."
        )
        ui.number("max step").bind_value(row, "max_step").props(
            "dense outlined"
        ).classes("w-24").tooltip(
            "Travel limit: how far it may move between runs. Blank for none."
        )
        ui.number("start").bind_value(row, "start").props(
            "dense outlined"
        ).classes("w-20").tooltip(
            "The current setting; needed with a travel limit."
        )


def _objective_editor(state: UIState) -> None:
    bax = state.method == "bax"
    with ui.row().classes("items-center gap-2"):
        ui.icon("flag").classes("text-indigo-600")
        ui.label("Measured outputs" if bax else "Objectives").classes(
            "text-base font-semibold"
        )
    if bax:
        ui.label(
            "In target-region mode every output is modeled; the goal below picks "
            "which region to find."
        ).classes("text-xs text-gray-500")
    container = ui.column().classes("gap-2 w-full")

    def redraw() -> None:
        container.clear()
        with container:
            for i, row in enumerate(state.objective_rows):
                with ui.row().classes("items-center gap-1 w-full"):
                    ui.input("name", value=row.get("name", "")).bind_value(row, "name").props(
                        "dense outlined"
                    ).classes("w-32")
                    if not bax:
                        ui.select(
                            {"MAXIMIZE": "Maximize", "MINIMIZE": "Minimize",
                             "OBSERVE": "Observe"},
                            value=row["direction"],
                        ).bind_value(row, "direction").props("dense outlined").classes("w-36")
                    ui.button(
                        icon="delete",
                        on_click=lambda _, idx=i: (_pop(state.objective_rows, idx), redraw()),
                    ).props("flat dense round color=grey")
                row.setdefault("min_val", None)
                row.setdefault("max_val", None)
                with ui.row().classes("items-center gap-1 w-full pl-4"):
                    ui.label("keep within").classes("text-xs text-gray-500")
                    ui.number("min").bind_value(row, "min_val").props(
                        "dense outlined"
                    ).classes("w-20")
                    ui.number("max").bind_value(row, "max_val").props(
                        "dense outlined"
                    ).classes("w-20")

    def add() -> None:
        n = len(state.objective_rows) + 1
        state.objective_rows.append({"name": f"y{n}", "direction": "MAXIMIZE"})
        redraw()

    redraw()
    ui.button("Add output" if bax else "Add objective", icon="add", on_click=add).props(
        "flat dense color=indigo"
    )


def _source_editor(state: UIState) -> None:
    """An optional source: runs that come at different fidelities or from different
    codes, with a target (the one you trust) where abbot recommends."""
    with ui.row().classes("items-center gap-2"):
        ui.icon("layers").classes("text-indigo-600")
        ui.label("Sources").classes("text-base font-semibold")
    row = state.source_row
    container = None

    def redraw() -> None:
        container.clear()
        if not state.source_on:
            return
        with container:
            ui.label(
                "abbot picks which source to run, trading cost against information, "
                "and recommends at the target. Report each run's cost if you know it."
            ).classes("text-xs text-gray-500")
            with ui.row().classes("items-center gap-1 w-full"):
                ui.select(
                    {"enumerated": "Labeled", "continuous": "A range (fidelity)"},
                    value=row["kind"],
                    on_change=lambda e: (_set(row, "kind", e.value), redraw()),
                ).props("dense outlined").classes("w-40")
                ui.input("name").bind_value(row, "name").props("dense outlined").classes("w-28")
            with ui.row().classes("items-center gap-1 w-full"):
                if row["kind"] == "enumerated":
                    ui.input("values").bind_value(row, "values").props(
                        "dense outlined"
                    ).classes("w-48")
                    ui.input("target").bind_value(row, "target").props(
                        "dense outlined"
                    ).classes("w-28")
                else:
                    ui.number("lo").bind_value(row, "lower").props("dense outlined").classes("w-20")
                    ui.number("hi").bind_value(row, "upper").props("dense outlined").classes("w-20")
                    ui.input("target").bind_value(row, "target").props(
                        "dense outlined"
                    ).classes("w-20")
            if row["kind"] == "continuous":
                with ui.row().classes("items-center gap-1 w-full"):
                    ui.checkbox("log scale").bind_value(row, "log").props("dense")
                    ui.select(
                        {"high": "higher is more accurate", "low": "lower is more accurate"},
                        value=row.get("accurate", "high"),
                    ).bind_value(row, "accurate").props("dense outlined").classes("w-52")

    ui.switch(
        "Runs come from more than one source (fidelities, codes)",
        value=state.source_on,
        on_change=lambda e: (_set(state, "source_on", e.value), redraw()),
    ).props("color=indigo")
    container = ui.column().classes("gap-2 w-full")
    redraw()


def _model_selector(state: UIState) -> None:
    """The surrogate: independent GPs per output, or one GP over correlated outputs."""
    with ui.row().classes("items-center gap-2"):
        ui.icon("insights").classes("text-indigo-600")
        ui.label("Model").classes("text-base font-semibold")
    ui.select(MODEL_LABELS, value=state.model).bind_value(state, "model").props(
        "dense outlined"
    ).classes("w-full")
    ui.label(
        "Correlated outputs share what one output teaches about another; use it when "
        "outputs move together."
    ).classes("text-xs text-gray-500")


def _method_selector(state: UIState, on_change) -> None:
    """Choose what the campaign does: optimize, or find a target region (BAX)."""
    with ui.row().classes("items-center gap-2"):
        ui.icon("alt_route").classes("text-indigo-600")
        ui.label("What do you want to do?").classes("text-base font-semibold")

    def pick(value: str) -> None:
        state.method = value
        on_change()  # refresh the setup panel so dependent editors appear/vanish

    ui.toggle(
        {"optimize": "Optimize", "bax": "Find a target region (BAX)"},
        value=state.method,
        on_change=lambda e: pick(e.value),
    ).props("no-caps").classes("w-full")
    if state.method == "optimize":
        ui.label("Drive the objective(s) up/down (Bayesian optimization).").classes(
            "text-xs text-gray-500"
        )
    else:
        ui.label(
            "Map where the outputs land in a region you care about "
            "(Bayesian Algorithm Execution)."
        ).classes("text-xs text-gray-500")


def _goal_editor(state: UIState) -> None:
    """Define the BAX goal: a goal-type dropdown with params that follow the choice."""
    with ui.row().classes("items-center gap-2"):
        ui.icon("my_location").classes("text-indigo-600")
        ui.label("Target region").classes("text-base font-semibold")

    names = [r.get("name", "") for r in state.objective_rows]
    goal = state.goal_row
    params = ui.column().classes("gap-2 w-full")

    def redraw_params() -> None:
        params.clear()
        kind = goal.get("kind", "band")
        with params:
            if kind == "band":
                ui.label("Keep each output within a [low, high] band (blank = any):").classes(
                    "text-xs text-gray-500"
                )
                bands = goal.setdefault("bands", {})
                for name in names:
                    band = bands.setdefault(name, {"lo": None, "hi": None})
                    with ui.row().classes("items-center gap-1 w-full"):
                        ui.label(name).classes("w-24 text-sm font-mono")
                        ui.number("low", value=band.get("lo")).bind_value(
                            band, "lo"
                        ).props("dense outlined").classes("w-24")
                        ui.number("high", value=band.get("hi")).bind_value(
                            band, "hi"
                        ).props("dense outlined").classes("w-24")
            elif kind == "percentile":
                ui.label("Find the top (100 − p)% of a single output.").classes(
                    "text-xs text-gray-500"
                )
                ui.number("percentile p", value=goal.get("percentile", 80.0)).bind_value(
                    goal, "percentile"
                ).props("dense outlined").classes("w-40")
            elif kind == "level":
                ui.label("Find where an output crosses a threshold (a level set).").classes(
                    "text-xs text-gray-500"
                )
                with ui.row().classes("items-center gap-1 w-full"):
                    ui.select(
                        {n: n for n in names}, value=goal.get("output", names[0]), label="output"
                    ).bind_value(goal, "output").props("dense outlined").classes("w-32")
                    ui.number("threshold", value=goal.get("threshold", 0.0)).bind_value(
                        goal, "threshold"
                    ).props("dense outlined").classes("w-32")
                ui.select(
                    {False: "above threshold (≥)", True: "below threshold (≤)"},
                    value=goal.get("upper", False),
                ).bind_value(goal, "upper").props("dense outlined").classes("w-48")
            elif kind == "topk":
                ui.label("Find the k best candidates on one output.").classes(
                    "text-xs text-gray-500"
                )
                with ui.row().classes("items-center gap-1 w-full"):
                    ui.number("k", value=goal.get("k", 1)).bind_value(goal, "k").props(
                        "dense outlined"
                    ).classes("w-20")
                    ui.select(
                        {n: n for n in names}, value=goal.get("output", names[0]), label="output"
                    ).bind_value(goal, "output").props("dense outlined").classes("w-32")
                ui.select(
                    {True: "largest", False: "smallest"}, value=goal.get("largest", True),
                ).bind_value(goal, "largest").props("dense outlined").classes("w-36")

    def pick_kind(value: str) -> None:
        goal["kind"] = value
        redraw_params()

    ui.select(
        {
            "band": "Target band",
            "percentile": "Top-percentile",
            "level": "Level set",
            "topk": "Top-K",
        },
        value=goal.get("kind", "band"),
        label="Goal",
        on_change=lambda e: pick_kind(e.value),
    ).props("dense outlined").classes("w-full")
    redraw_params()

    ui.select(
        BAX_STRATEGY_LABELS, value=state.bax_strategy, label="Strategy"
    ).bind_value(state, "bax_strategy").props("dense outlined").classes("w-full")


def _set(row, key: str, value) -> None:
    if isinstance(row, dict):
        row[key] = value
    else:
        setattr(row, key, value)


def _pop(rows: list, idx: int) -> None:
    if 0 <= idx < len(rows) and len(rows) > 1:
        rows.pop(idx)


# --------------------------------------------------------------------------- #
# Page (single dashboard)
# --------------------------------------------------------------------------- #


def _build_page() -> None:
    state = UIState()

    with ui.header().classes("items-center bg-indigo-700"), ui.row().classes("items-center gap-2"):
        ui.icon("science").classes("text-white text-2xl")
        ui.label("abbot").classes("text-white text-xl font-bold")
        ui.label("goal-driven Bayesian experimental design").classes("text-indigo-200 text-sm")

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
                        _method_selector(state, on_change=lambda: state._left.refresh())
                    with ui.card().classes("w-full"):
                        _variable_editor(state)
                    with ui.card().classes("w-full"):
                        _objective_editor(state)
                    if state.method == "bax":
                        with ui.card().classes("w-full"):
                            _goal_editor(state)
                    with ui.card().classes("w-full"):
                        _source_editor(state)
                    with ui.card().classes("w-full"):
                        _model_selector(state)
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
    async def _auto_tick() -> None:
        # The controller owns run-state; the timer just asks it to advance and
        # re-renders. The step itself runs off the event loop so a slow experiment
        # doesn't freeze the page (or its Pause/Stop buttons).
        ctrl = state.controller
        if ctrl is None or not ctrl.is_running or state._stepping:
            return
        was_running = ctrl.is_running
        first = state.needs_structure_rebuild()
        state._stepping = True
        try:
            stepped = await run.io_bound(ctrl.tick)
        except Exception as exc:  # misconfig now raises here — show it, don't hide it
            logger.exception("auto step failed")
            ctrl.stop()
            ui.notify(str(exc), type="negative", multi_line=True, close_button=True)
            state._left.refresh()
            return
        finally:
            state._stepping = False
        if state.controller is not ctrl:
            return  # the campaign was reset while this step ran
        if not stepped:
            # tick() stopped us (budget spent / halted) — refresh the controls once.
            if was_running:
                state._left.refresh()
                state._right.refresh()
            return
        _surface_warning(state)
        if first:
            # The campaign started with queued-but-unrun seeds, so the viz panel was
            # built in its empty state (no plot widgets). The first step produces
            # history -> rebuild the structure once, then update in place after.
            state._right.refresh()
        else:
            _update_viz(state)  # in-place update; no teardown -> smooth

    ui.timer(0.5, _auto_tick)  # always active; gated by ctrl.is_running


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
        _halt_actions(state)

        spec = state.spec
        if isinstance(ctrl.campaign.policy, BAXPolicy):
            summary = f"{len(spec.domain)} variable(s), target-region (BAX) over " \
                      f"{spec.n_outputs} output(s)"
        else:
            summary = f"{len(spec.domain)} variable(s), {spec.n_objectives} objective(s)"
        extras = []
        if spec.sources:
            extras.append(f"source {spec.sources[0].name!r} (target {spec.sources[0].target})")
        if any(o.min_val is not None or o.max_val is not None for o in spec.outputs):
            extras.append("output ranges")
        if spec.has_travel_limits:
            extras.append("travel limits")
        if extras:
            summary += "; " + ", ".join(extras)
        ui.label(summary).classes("text-xs text-gray-500")
        ui.button("Check the model", icon="fact_check", on_click=lambda: _show_check(state)).props(
            "flat dense color=indigo"
        ).mark("check-model")

        if ctrl.can_auto:
            _auto_controls(state)
        else:
            _manual_controls(state)

        ui.separator()
        _data_io_controls(state)


def _show_check(state: UIState) -> None:
    """``campaign.check()`` in a dialog: fit quality, noise, lengthscales, warnings."""
    try:
        report = str(state.controller.campaign.check())
    except Exception as exc:  # noqa: BLE001 - too few runs, no model yet, ...
        report = f"Can't check yet: {exc}"
    with ui.dialog() as dialog, ui.card().classes("w-full"):
        ui.label("Model check").classes("text-base font-semibold")
        ui.label(report).classes("font-mono text-sm whitespace-pre-wrap").mark("check-report")
        ui.button("Close", on_click=dialog.close).props("flat color=indigo")
    dialog.open()


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

    def on_upload(e) -> None:
        try:
            spec = abbot_io.spec_from_json(e.content.read().decode())
            variables, outputs, source = spec_to_rows(spec)
        except Exception as exc:  # noqa: BLE001
            ui.notify(str(exc), type="negative", multi_line=True, close_button=True)
            return
        state.variable_rows = variables or state.variable_rows
        state.objective_rows = outputs
        state.source_on = source is not None
        if source is not None:
            state.source_row = source
        ui.notify("Spec loaded into the editor.", type="positive")
        state._left.refresh()

    ui.upload(on_upload=on_upload, auto_upload=True, label="Import spec JSON").props(
        "flat dense accept=.json"
    ).classes("max-w-xs")


def _reset(state: UIState) -> None:
    state.controller = None  # drops the controller -> ctrl.is_running gate goes False
    state.spec = None
    state._left.refresh()
    state._right.refresh()


def _manual_controls(state: UIState) -> None:
    ctrl = state.controller
    ui.markdown(
        "**Manual** — suggest designs, run them in the lab, enter the value(s). "
        "Several can be out at once (vials, jobs); each stays listed until you "
        "submit, report or cancel it."
    ).classes("text-sm text-gray-600")

    def do_suggest() -> None:
        try:
            ctrl.suggest(max(1, int(state.n_suggest or 1)))
        except Exception as exc:  # noqa: BLE001
            ui.notify(str(exc), type="negative", multi_line=True, close_button=True)
            return
        _surface_warning(state)
        state._left.refresh()

    with ui.row().classes("items-center gap-2"):
        ui.number("How many", min=1, max=20, step=1, precision=0).bind_value(
            state, "n_suggest"
        ).props("outlined dense").classes("w-24")
        ui.button("Suggest", icon="lightbulb", on_click=do_suggest).props("color=indigo")

    outstanding = ctrl.campaign.outstanding()
    if not outstanding:
        ui.label("No designs outstanding.").classes("text-sm text-gray-500")
        return
    ui.label(f"Outstanding ({len(outstanding)}), oldest first:").classes(
        "font-semibold text-sm text-indigo-900"
    )
    for cand in outstanding:
        _outstanding_card(state, cand)


def _outstanding_card(state: UIState, cand) -> None:
    """One outstanding design: where to run, the measured values, and what to do."""
    ctrl = state.controller
    with ui.card().classes("bg-indigo-50 w-full"):
        ui.label(f"Run at ({cand.id}):").classes("font-semibold text-indigo-900 text-sm")
        with ui.row().classes("gap-3 flex-wrap"):
            for name, val in cand.items():
                with ui.column().classes("gap-0"):
                    ui.label(name).classes("text-xs text-gray-500")
                    ui.label(_fmt(val)).classes("font-mono")
        inputs = {}
        with ui.row().classes("gap-2 flex-wrap"):
            for o in state.spec.outputs:  # every modeled output, incl. OBSERVE
                inputs[o.name] = ui.number(f"{o.name}").props("outlined dense").classes(
                    "w-28"
                ).mark(f"{cand.id}-{o.name}")
            cost = None
            if state.spec.sources:
                cost = ui.number("cost (optional)").props("outlined dense").classes(
                    "w-32"
                ).mark(f"{cand.id}-cost")

        def done(message: str, kind: str = "positive") -> None:
            ui.notify(message, type=kind)
            state._left.refresh()
            # First measurement builds the viz structure; later ones update in place.
            if state.needs_structure_rebuild():
                state._right.refresh()
            else:
                _update_viz(state)

        def do_submit() -> None:
            vals = [inputs[o.name].value for o in state.spec.outputs]
            if any(v is None for v in vals):
                ui.notify("Enter all measured values first.", type="warning")
                return
            spent = cost.value if cost is not None else None
            ctrl.submit(torch.tensor(vals, dtype=torch.double), id=cand.id, cost=spent)
            done("Recorded.")

        def do_fail() -> None:
            ctrl.fail(cand.id, "reported in the UI: couldn't run")
            done("Recorded as failed; it will be retried if retries remain.", "warning")

        def do_cancel() -> None:
            ctrl.cancel(cand.id)
            ui.notify("Withdrawn.", type="info")
            state._left.refresh()

        with ui.row().classes("gap-1"):
            ui.button("Submit", icon="check", on_click=do_submit).props("color=indigo").mark(
                f"{cand.id}-submit"
            )
            ui.button("Couldn't run", icon="report", on_click=do_fail).props(
                "flat color=orange"
            ).mark(f"{cand.id}-fail")
            ui.button("Cancel", icon="close", on_click=do_cancel).props("flat color=grey").mark(
                f"{cand.id}-cancel"
            )


def _auto_controls(state: UIState) -> None:
    ctrl = state.controller
    ui.markdown("**Mock / auto-run** — the synthetic experiment drives the loop.").classes(
        "text-sm text-gray-600"
    )

    def do_run() -> None:
        # The controller owns run-state; the timer picks it up via ctrl.tick().
        try:
            ctrl.start()
        except Exception as exc:  # noqa: BLE001
            ui.notify(str(exc), type="negative", multi_line=True, close_button=True)
            return
        state._left.refresh()

    def do_pause() -> None:
        ctrl.pause()
        state._left.refresh()

    async def do_step() -> None:
        if state._stepping:
            ui.notify("A step is already running.", type="info")
            return
        rebuild = state.needs_structure_rebuild()
        state._stepping = True
        try:
            await run.io_bound(ctrl.step_once)
        except Exception as exc:  # noqa: BLE001
            ui.notify(str(exc), type="negative", multi_line=True, close_button=True)
            return
        finally:
            state._stepping = False
        if state.controller is not ctrl:
            return  # the campaign was reset while this step ran
        _surface_warning(state)
        state._left.refresh()
        if rebuild:
            state._right.refresh()  # build structure on the first point
        else:
            _update_viz(state)

    def do_stop() -> None:
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
        state._plot_conv = None  # structure not built -> needs_structure_rebuild()
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
                ).props("dense outlined").classes("w-32")
            if len(spec.domain) > 2:
                # Design variables only: a source is shown at its target.
                names = {i: d.name for i, d in enumerate(spec.domain)}
                ui.select(names, value=state.viz_vx, label="x-axis var",
                          on_change=lambda e: _reviz(state, "viz_vx", e.value)).props(
                              "dense outlined").classes("w-32")
                ui.select(names, value=state.viz_vy, label="y-axis var",
                          on_change=lambda e: _reviz(state, "viz_vy", e.value)).props(
                              "dense outlined").classes("w-32")

    with ui.row().classes("w-full no-wrap gap-3"):
        with ui.card().classes("grow"):
            state._plot_model = ui.plotly(
                _safe_fig(_model_fig_for, state, model) if model is not None
                else figures.blank_fig()
            ).classes("w-full")
        with ui.card().classes("grow"):
            state._plot_conv = ui.plotly(_safe_fig(_progress_fig_for, state)).classes("w-full")

    with ui.card().classes("w-full"):
        parity = _safe_fig(_parity_fig_for, state, model) if model is not None else None
        state._plot_parity = ui.plotly(parity or figures.blank_fig("parity")).classes("w-full")

    state._rec_box = ui.column().classes("w-full gap-0")
    with ui.expansion("History", icon="table_rows").classes("w-full"):
        state._table = ui.table(
            columns=_history_columns(spec), rows=_history_rows(state), row_key="step"
        ).classes("w-full")

    _update_recommendation(state)


def _surface_warning(state: UIState) -> None:
    """Show (and clear) a transient-fallback warning the campaign recorded, if any."""
    ctrl = state.controller
    if ctrl is None:
        return
    msg = getattr(ctrl.campaign, "last_warning", None)
    if msg:
        ui.notify(msg, type="warning", multi_line=True, close_button=True)
        ctrl.campaign.last_warning = None


def _update_viz(state: UIState) -> None:
    """Push new data into the existing plot/table widgets (no teardown)."""
    if getattr(state, "_plot_conv", None) is None or state.controller is None:
        return
    if len(state.controller.campaign.history) == 0:
        return
    model = state.fitted_model()
    state._plot_conv.update_figure(_safe_fig(_progress_fig_for, state))
    if model is not None:
        state._plot_model.update_figure(_safe_fig(_model_fig_for, state, model))
        pf = _safe_fig(_parity_fig_for, state, model)
        if pf is not None:
            state._plot_parity.update_figure(pf)
    if state._table is not None:
        state._table.rows = _history_rows(state)
        state._table.update()
    _update_recommendation(state)


# -- thin state -> figures.* adapters (the pure builders live in figures.py) ----


def _progress_fig_for(state: UIState) -> dict:
    return figures.progress_fig(
        state.history(), state.spec, state.policy(), state.fitted_model()
    )


def _model_fig_for(state: UIState, model) -> dict:
    return figures.model_fig(state.history(), state.spec, model, state.selections())


def _parity_fig_for(state: UIState, model) -> dict | None:
    return figures.parity_fig(state.history(), state.spec, model, state.selections())


def _safe_fig(builder, *args):
    """Call a figure builder; on failure show a toast + a placeholder (never a blank
    crash). Replaces the old silent ``except: identified=0`` that hid BAX bugs.
    """
    try:
        return builder(*args)
    except Exception as exc:  # a viz error must not kill the dashboard
        logger.exception("figure build failed")
        ui.notify(f"Could not draw a plot: {exc}", type="warning", close_button=True)
        return figures.blank_fig("unavailable")


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
        # The buttons below replace the API hint.
        why = "\n".join(
            line for line in ctrl.campaign.halted.split("\n")
            if not line.startswith("Decide with resolve()")
        )
        ui.label(f"⚠ {why}").classes("text-red-600 text-sm whitespace-pre-line")


def _halt_actions(state: UIState) -> None:
    """A halted campaign's answers: retry (fixed), exclude the design, or mark it a
    bad region; a halt not about a design only resumes."""
    ctrl = state.controller
    if not ctrl.campaign.halted:
        return

    def answer(choice: str, reason: str | None = None) -> None:
        try:
            ctrl.resolve(choice, reason=reason or None)
        except Exception as exc:  # noqa: BLE001
            ui.notify(str(exc), type="negative", multi_line=True, close_button=True)
            return
        state._left.refresh()

    with ui.card().classes("w-full bg-red-50"):
        if not ctrl.campaign.halted_on:
            ui.label("Fix the cause, then resume.").classes("text-sm")
            ui.button("Resume", icon="play_arrow", on_click=lambda: answer("retry")).props(
                "color=indigo"
            ).mark("halt-retry")
            return
        designs = "; ".join(
            ", ".join(f"{k} = {_fmt(v)}" for k, v in d.items()) for d in ctrl.campaign.halted_on
        )
        ui.label(f"What went wrong at {designs}?").classes("text-sm font-semibold")
        with ui.row().classes("items-center gap-2 flex-wrap"):
            ui.button(
                "Apparatus fixed: retry", icon="replay", on_click=lambda: answer("retry")
            ).props("color=indigo").mark("halt-retry")
            ui.button(
                "Skip this design", icon="block", on_click=lambda: answer("exclude")
            ).props("outline").mark("halt-exclude")
        with ui.row().classes("items-center gap-2 flex-wrap"):
            reason = ui.input("Why (optional)", placeholder="dark: flat profile").props(
                "outlined dense"
            ).mark("halt-reason")
            ui.button(
                "The design is the cause: bad region", icon="report",
                on_click=lambda: answer("bad_region", reason.value),
            ).props("outline color=red").mark("halt-bad-region")


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
                f"{n} = {_fmt(v)}" for n, v in spec.as_dict(r.candidate).items()
            )
            outs = ", ".join(
                f"{o.name} = {float(r.outputs[j]):.4g}" for j, o in enumerate(spec.outputs)
            )
            ui.label(f"{setting}  →  {outs}").classes("font-mono text-sm")


def _fmt(value) -> str:
    """A design value for display: a label as is, a number to 4 significant figures."""
    return value if isinstance(value, str) else f"{value:.4g}"


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
            **{n: v if isinstance(v, str) else round(v, 4) for n, v in spec.as_dict(r.candidate).items()},
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
