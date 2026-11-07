"""The expert panel: every field of the campaign, in a dense console style.

While setting up: the spec as tables, ``campaign.json`` (the format ``abbot-mcp``
saves), and the policy and model abbot will build, all following the steps as they
are edited. While running: every run (with suggesting several at once, holding
variables fixed, CSV in and out, and correcting a value), the plots, the model
check, and the spec.
"""

from __future__ import annotations

import json
import logging

from nicegui import ui

from abbot.problem import EnumeratedDomain, IntegerDomain
from abbot.ui import figures, running, summary, theme
from abbot.ui.builder import MODEL_LABELS, build_constructor, build_start
from abbot.ui.state import UIState

logger = logging.getLogger(__name__)


def render(state: UIState) -> None:
    with ui.column().classes("w-full gap-0"):
        if state.controller is None:
            _setup(state)
        else:
            _running(state)


def _tabs(names: list[str], note: str = ""):
    with ui.row().classes("w-full items-center justify-between px-4 pt-3 pb-2 no-wrap"):
        tabs = ui.tabs().props("dense no-caps active-bg-color=white indicator-color=transparent "
                               "active-color=grey-10").classes("text-gray-600")
        with tabs:
            made = [ui.tab(n).classes("rounded-lg") for n in names]
        if note:
            ui.label(note).classes("text-xs ab-sub max-md:hidden")
    ui.separator()
    return tabs, made


# -- setting up ---------------------------------------------------------------


def _setup(state: UIState) -> None:
    tabs, (t_spec, t_json, t_policy) = _tabs(["Spec", "spec.json", "Policy & model"],
                                             "follows the steps as you edit them")
    spec, error = state.preview()
    with ui.tab_panels(tabs, value=t_spec).classes("w-full bg-transparent"):
        with ui.tab_panel(t_spec).classes("p-4"):
            if spec is None:
                ui.label(f"Not a complete spec yet: {error}").classes("ab-error")
            else:
                _spec_tables(state, spec, state.variable_rows)
        with ui.tab_panel(t_json).classes("p-4"):
            _config_json(state)
        with ui.tab_panel(t_policy).classes("p-4"):
            if spec is None:
                ui.label(f"Not a complete spec yet: {error}").classes("ab-error")
            else:
                _policy_and_model(state, spec)


def _spec_tables(state: UIState, spec, variable_rows: list[dict] | None) -> None:
    starts = {r.get("name"): r.get("start") for r in variable_rows or []}
    with ui.column().classes("w-full gap-4"):
        theme.section("Inputs")
        rows = []
        for d in spec.domain:
            if isinstance(d, EnumeratedDomain):
                rows.append({"name": d.name, "kind": "choices",
                             "low": ", ".join(str(v) for v in d.values), "high": "", "log": "",
                             "max step": _dash(d.max_step), "start": _dash(starts.get(d.name))})
            else:
                rows.append({"name": d.name,
                             "kind": "whole" if isinstance(d, IntegerDomain) else "range",
                             "low": summary.fmt(d.lower), "high": summary.fmt(d.upper),
                             "log": "yes" if d.log else "–", "max step": _dash(d.max_step),
                             "start": _dash(starts.get(d.name))})
        theme.table(["name", "kind", "low", "high", "log", "max step", "start"], rows)
        theme.section("Outputs")
        theme.table(["name", "direction", "min", "max"], [
            {"name": o.name, "direction": o.direction, "min": _dash(o.min_val),
             "max": _dash(o.max_val)} for o in spec.outputs])
        if spec.sources:
            theme.section("Sources")
            rows = []
            for s in spec.sources:
                values = (", ".join(str(v) for v in s.domain.values)
                          if isinstance(s.domain, EnumeratedDomain)
                          else f"{summary.fmt(s.domain.lower)} – {summary.fmt(s.domain.upper)}")
                rows.append({"name": s.name, "values": values, "target": str(s.target),
                             "cost": "reported per run"})
            theme.table(["name", "values", "target", "cost"], rows)
        goal = state.goal_json() if state.controller is None else _running_goal(state)
        if goal is not None:
            theme.section("Goal")
            ui.label(summary.goal_words(goal)).classes("ab-mono text-sm")


def _running_goal(state: UIState) -> dict | None:
    try:
        return state.goal_json()
    except Exception:  # noqa: BLE001 - rows changed since the start; skip
        return None


def _dash(value) -> str:
    return "–" if value is None else summary.fmt(value)


def _config_json(state: UIState) -> None:
    try:
        config = state.campaign_config()
    except Exception as exc:  # noqa: BLE001
        ui.label(f"Not a complete spec yet: {exc}").classes("ab-error")
        return
    text = json.dumps(config, indent=1)
    ui.code(text, language="json").classes("w-full text-xs").mark("spec-json")
    with ui.row().classes("items-center gap-4 pt-2 text-sm"):
        theme.link("Download campaign.json", lambda: ui.download.content(
            text.encode(), "campaign.json"))
        ui.label("The same format abbot-mcp saves.").classes("ab-sub text-xs")


def _policy_and_model(state: UIState, spec) -> None:
    from abbot.model import default_constructor
    from abbot.session import default_initial_designs

    with ui.row().classes("w-full gap-3 no-wrap max-md:flex-wrap"):
        with ui.element("div").classes("ab-card p-4 grow"):
            theme.section("Policy")
            try:
                policy = state.build_policy(spec)
                name = type(policy).__name__
                detail = (f"goal-driven, strategy {state.bax_strategy}"
                          if state.method == "bax"
                          else "one objective" if spec.n_objectives == 1
                          else f"{spec.n_objectives} objectives, Pareto front")
                if spec.sources:
                    detail += ", chooses sources by cost"
            except Exception as exc:  # noqa: BLE001
                name, detail = "–", str(exc)
            ui.label(name).classes("ab-mono text-sm")
            ui.label(detail).classes("ab-mono text-xs ab-sub")
        with ui.element("div").classes("ab-card p-4 grow"):
            theme.section("Model · run")
            try:
                constructor = build_constructor(spec, state.model) or default_constructor(spec)
                model = type(constructor).__name__
            except Exception as exc:  # noqa: BLE001
                model = f"– ({exc})"
            ui.label(model).classes("ab-mono text-sm")
            try:
                start = build_start(spec, state.variable_rows)
            except Exception:  # noqa: BLE001
                start = None
            seeds = ("starts at the current setting" if start is not None
                     else f"{len(default_initial_designs(spec))} spread-out starting runs")
            ui.label(f"{seeds} · max_retries 1").classes("ab-mono text-xs ab-sub")
            ui.label(MODEL_LABELS.get(state.model, state.model)).classes("text-xs ab-sub")


# -- running ----------------------------------------------------------------


def _running(state: UIState) -> None:
    campaign = state.controller.campaign
    n = len(campaign.history)
    failed = sum(bool(r.fault) for r in campaign.history.records)
    note = f"{n} runs · {failed} failed"
    if state.spec.sources and any(r.cost is not None for r in campaign.history.records):
        note += f" · cost {summary.fmt(campaign.history.cost_spent)}"
    tabs, (t_runs, t_plots, t_model, t_spec) = _tabs(["Runs", "Plots", "Model", "Spec"], note)
    with ui.tab_panels(tabs, value=t_runs).classes("w-full bg-transparent"):
        with ui.tab_panel(t_runs).classes("p-4"):
            _runs(state)
        with ui.tab_panel(t_plots).classes("p-4"):
            _plots(state)
        with ui.tab_panel(t_model).classes("p-4"):
            _check(state)
        with ui.tab_panel(t_spec).classes("p-4"):
            _spec_tables(state, state.spec, state.variable_rows)
            ui.element("div").classes("h-4")
            _config_json(state)


def _runs(state: UIState) -> None:
    ctrl = state.controller
    spec = state.spec
    if not ctrl.can_auto:
        _suggest_row(state)
    table = theme.table(_run_columns(spec), _run_rows(state), key="id",
                        pagination=0).mark("runs-table")
    table.on("rowClick", lambda e: _correct_dialog(state, e.args[1]))
    ui.label("Click a measured run to correct a value.").classes("text-xs ab-sub pt-1")

    def live() -> None:
        table.rows = _run_rows(state)
        table.update()

    state._live = live
    with ui.row().classes("items-center gap-4 pt-3 text-sm"):
        _csv_io(state)


def _suggest_row(state: UIState) -> None:
    ctrl = state.controller
    spec = state.spec
    with ui.row().classes("items-center gap-2 pb-3 flex-wrap"):
        ui.number("how many", min=1, max=20, precision=0).bind_value(state, "n_suggest").props(
            "outlined dense").classes("w-24").mark("n-suggest")
        names = {"": "hold nothing", **{d.name: f"hold {d.name}" for d in spec.domain}}
        hold_var = ui.select(names, value=next(iter(state.hold), "")).props(
            "outlined dense").classes("w-40")
        hold_value = ui.input("at", value=str(next(iter(state.hold.values()), ""))).props(
            "outlined dense").classes("w-28")

        async def suggest() -> None:
            fixed = {}
            if hold_var.value:
                d = next(x for x in spec.domain if x.name == hold_var.value)
                raw = hold_value.value
                fixed[d.name] = raw if (isinstance(d, EnumeratedDomain)
                                        and d.is_categorical) else float(raw)
            state.hold = fixed
            try:
                ctrl.suggest(max(1, int(state.n_suggest or 1)), fixed=fixed)
            except Exception as exc:  # noqa: BLE001
                ui.notify(str(exc), type="negative", multi_line=True, close_button=True)
                return
            state.refresh()

        ui.button("Suggest", on_click=suggest).props("outline no-caps").mark("expert-suggest")


def _run_columns(spec) -> list[str]:
    cols = ["#", "id", *spec.variable_names, *spec.output_names]
    if spec.sources:
        cols.append("cost")
    return [*cols, "status"]


def _run_rows(state: UIState) -> list[dict]:
    campaign = state.controller.campaign
    spec = state.spec
    rows = []
    for cand in reversed(campaign.outstanding()):
        row = {"#": "▸", "id": cand.id, **{k: summary.fmt(v) for k, v in cand.items()},
               "status": "out"}
        rows.append(row)
    for i, r in reversed(list(enumerate(campaign.history.records))):
        row = {"#": str(i + 1), "id": r.id or "",
               **{k: summary.fmt(v) for k, v in spec.as_dict(r.candidate).items()}}
        if r.outputs is not None:
            row.update({o.name: summary.fmt(float(r.outputs.reshape(-1)[j]))
                        for j, o in enumerate(spec.outputs)})
        if spec.sources:
            row["cost"] = _dash(r.cost)
        row["status"] = ("failed" if r.fault else "bad region" if r.bad_region
                         else "corrected" if r.corrections else "")
        if r.excluded:
            row["status"] += " · skipped"
        row["_step"] = i
        rows.append(row)
    return rows


def _correct_dialog(state: UIState, row: dict) -> None:
    step = row.get("_step")
    campaign = state.controller.campaign
    if step is None:
        ui.notify("That design hasn't been measured yet.", type="info")
        return
    r = campaign.history.records[step]
    if r.outputs is None or r.fault:
        ui.notify("A failed run has no values to correct.", type="info")
        return
    spec = state.spec
    with ui.dialog() as dialog, ui.element("div").classes("ab-card p-6 bg-white"), \
            ui.column().classes("gap-3"):
        ui.label(f"Correct run {step + 1} ({r.id})").classes("text-lg font-semibold")
        fields = {o.name: ui.number(o.name, value=float(r.outputs.reshape(-1)[j])).props(
            "outlined dense").classes("w-40").mark(f"correct-{o.name}")
            for j, o in enumerate(spec.outputs)}
        reason = ui.input("why (optional)").props("outlined dense").classes("w-72")
        if r.corrections:
            ui.label(f"Corrected {len(r.corrections)} time(s) before.").classes("ab-sub text-xs")

        def save() -> None:
            values = {k: f.value for k, f in fields.items()}
            if any(v is None for v in values.values()):
                ui.notify("Enter every value.", type="warning")
                return
            try:
                campaign.correct(step, values, reason=reason.value or None)
            except Exception as exc:  # noqa: BLE001
                ui.notify(str(exc), type="negative", multi_line=True, close_button=True)
                return
            dialog.close()
            ui.notify(f"Corrected run {step + 1}; the model refits.", type="positive")
            state.refresh()

        with ui.row().classes("gap-3 pt-2"):
            theme.primary("Save", save, marker="correct-save")
            ui.button("Cancel", on_click=dialog.close).props("flat no-caps")
    dialog.open()


def _csv_io(state: UIState) -> None:
    from abbot import io as abbot_io

    campaign = state.controller.campaign
    spec = state.spec

    def export() -> None:
        ui.download.content(abbot_io.history_to_csv(campaign.history, spec).encode(),
                            "abbot_history.csv")

    def imported(e) -> None:
        try:
            hist = abbot_io.history_from_csv(e.content.read().decode(), spec)
        except Exception as exc:  # noqa: BLE001
            ui.notify(str(exc), type="negative", multi_line=True, close_button=True)
            return
        campaign.load_history(hist)
        ui.notify(f"Imported {len(hist)} run(s).", type="positive")
        state.refresh()

    upload = ui.upload(on_upload=imported, auto_upload=True).props("accept=.csv").classes(
        "hidden")
    theme.link("import csv", lambda: upload.run_method("pickFiles"))
    theme.link("export csv", export)


def _plots(state: UIState) -> None:
    model = state.fitted_model()
    campaign = state.controller.campaign
    if not len(campaign.history):
        ui.label("Plots appear after the first result.").classes("ab-sub")
        return
    with ui.column().classes("w-full gap-3"):
        if model is not None:
            with ui.element("div").classes("ab-card p-2 w-full"):
                ui.plotly(running.model_figure(state, model)).classes("w-full")
        with ui.element("div").classes("ab-card p-2 w-full"):
            ui.plotly(_safe(figures.progress_fig, campaign.history, state.spec,
                            campaign.policy, model)).classes("w-full")
        if model is not None:
            parity = _safe(figures.parity_fig, campaign.history, state.spec, model,
                           state.selections())
            if parity is not None:
                with ui.element("div").classes("ab-card p-2 w-full"):
                    ui.plotly(parity).classes("w-full")


def _safe(builder, *args):
    try:
        return builder(*args)
    except Exception as exc:  # a plot must not break the page
        logger.exception("figure failed")
        return figures.blank_fig(f"unavailable: {exc}")


def _check(state: UIState) -> None:
    try:
        report = str(state.controller.campaign.check())
    except Exception as exc:  # noqa: BLE001 - too few runs, no model yet
        report = f"Can't check yet: {exc}"
    ui.label(report).classes("ab-mono text-sm whitespace-pre-wrap").mark("check-report")
