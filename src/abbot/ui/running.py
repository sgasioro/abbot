"""The running campaign's main column: what to run next, and how it's going.

With a person running the experiments, one card holds the next design, the boxes
for what was measured, and Record; the next design is suggested as soon as the
last one is recorded. With a simulated experiment, Run / Pause / Step / Stop drive
it. A halt replaces both with its question. Below: how it's going in plain words,
the model's picture, and a way to finish with the current best.
"""

from __future__ import annotations

import logging

import torch
from nicegui import run, ui

from abbot.ui import figures, summary, theme
from abbot.ui.controller import RunStatus
from abbot.ui.state import UIState

logger = logging.getLogger(__name__)


def render(state: UIState) -> None:
    ctrl = state.controller
    campaign = ctrl.campaign
    with ui.column().classes("w-full gap-4"):
        if campaign.halted:
            _halt_card(state)
        elif state.finished:
            _finished_card(state)
        elif ctrl.can_auto:
            _auto_card(state)
        else:
            _next_card(state)
        with ui.element("div").classes("ab-card w-full p-6"), ui.column().classes("gap-2 w-full"):
            ui.label("How it's going").classes("text-lg font-semibold")
            state._progress = ui.markdown(_progress_text(state)).classes(
                "text-[0.95rem] w-full").mark("progress")
        with ui.element("div").classes("ab-card w-full p-4"):
            _model_plot(state)
        if not state.finished and len(campaign.history.clean):
            theme.link("Use the current best and finish", lambda: _finish(state),
                       marker="finish")


async def started(state: UIState) -> None:
    """The campaign was just created: show it, and suggest the first design."""
    state.refresh()
    await suggest_next(state)


async def suggest_next(state: UIState) -> None:
    """With a person running the experiments and nothing out, suggest the next
    design (off the event loop: it may fit a model)."""
    ctrl = state.controller
    if ctrl is None or ctrl.can_auto or state.finished or state.suggesting:
        return
    campaign = ctrl.campaign
    if campaign.halted or campaign.stopped or campaign.outstanding():
        return
    state.suggesting = True
    state.refresh()
    try:
        await run.io_bound(ctrl.suggest, 1)
    except Exception as exc:  # shown, not raised into the page
        logger.exception("suggest failed")
        ui.notify(str(exc), type="negative", multi_line=True, close_button=True)
    finally:
        state.suggesting = False
    if state.controller is ctrl:
        _surface_warning(state)
        state.refresh()


# -- what to run next ----------------------------------------------------------


def _next_card(state: UIState) -> None:
    campaign = state.controller.campaign
    outstanding = campaign.outstanding()
    with ui.element("div").classes("ab-card w-full p-6"):
        if not outstanding:
            with ui.row().classes("items-center gap-3"):
                if state.suggesting:
                    ui.spinner(size="md")
                    ui.label("Working out the next design…").classes("ab-sub")
                else:
                    theme.primary("Suggest the next design", lambda: suggest_next(state),
                                  marker="suggest-next")
            return
        ids = [c.id for c in outstanding]
        cand = outstanding[ids.index(state.focus)] if state.focus in ids else outstanding[0]
        _run_this(state, cand)
        others = [c.id for c in outstanding if c.id != cand.id]
        if others:
            with ui.row().classes("items-center gap-2 pt-3 text-sm"):
                ui.label("Also out:").classes("ab-sub")
                for other in others:
                    theme.link(other, lambda o=other: _focus(state, o))


def _focus(state: UIState, id: str) -> None:
    state.focus = id
    state.refresh()


def _run_this(state: UIState, cand) -> None:
    spec = state.spec
    ctrl = state.controller
    n = len(ctrl.campaign.history) + 1
    with ui.row().classes("items-baseline gap-3"):
        ui.label("Run this next").classes("text-xl font-semibold")
        ui.label(f"{cand.id} · run {n}").classes("ab-sub text-sm")
    with ui.row().classes("gap-8 pt-3 flex-wrap").mark("next-design"):
        for name, value in cand.items():
            with ui.column().classes("gap-0"):
                ui.label(name).classes("ab-value-label")
                ui.label(summary.fmt(value)).classes(
                    "ab-value" + (" label" if isinstance(value, str) else ""))
    inputs = {}
    with ui.row().classes("items-end gap-3 pt-4 flex-wrap"):
        for o in spec.outputs:
            inputs[o.name] = ui.number(o.name).props("outlined").classes("w-36").mark(
                f"{cand.id}-{o.name}")
        cost = None
        if spec.sources:
            cost = ui.number("cost", placeholder="optional").props("outlined").classes(
                "w-32").mark(f"{cand.id}-cost")

        async def record() -> None:
            values = [inputs[o.name].value for o in spec.outputs]
            if any(v is None for v in values):
                ui.notify("Enter every measured value first.", type="warning")
                return
            spent = cost.value if cost is not None else None
            try:
                ctrl.submit(torch.tensor(values, dtype=torch.double), id=cand.id, cost=spent)
            except Exception as exc:  # noqa: BLE001
                ui.notify(str(exc), type="negative", multi_line=True, close_button=True)
                return
            ui.notify(f"Recorded {cand.id}.", type="positive")
            state.focus = None
            await _after_change(state)

        theme.primary("Record", record, marker=f"{cand.id}-submit").classes("h-14")
    failure = ui.row().classes("items-center gap-3 pt-2")
    failure.set_visibility(False)
    with ui.row().classes("items-center gap-6 pt-3 text-sm"):
        theme.link("It didn't run", lambda: failure.set_visibility(True), warn=True,
                   marker=f"{cand.id}-fail-open")
        theme.link("Skip this one", lambda: _cancel(state, cand.id),
                   marker=f"{cand.id}-cancel").classes("ab-sub")

    async def failed() -> None:
        ctrl.fail(cand.id, reason.value or "reported in the UI: it didn't run")
        ui.notify("Recorded as failed; abbot retries it if retries remain.", type="warning")
        state.focus = None
        await _after_change(state)

    with failure:
        reason = ui.input("What happened? (optional)").props("outlined dense").classes(
            "w-72").mark(f"{cand.id}-fail-reason")
        ui.button("Record as failed", on_click=failed).props(
            "outline no-caps color=warning").mark(f"{cand.id}-fail")


async def _cancel(state: UIState, id: str) -> None:
    state.controller.cancel(id)
    ui.notify(f"Skipped {id}.", type="info")
    await _after_change(state)


async def _after_change(state: UIState) -> None:
    """A run was recorded, failed or skipped: redraw, then suggest the next one."""
    state.refresh()
    await suggest_next(state)


# -- a simulated experiment --------------------------------------------------------


def _auto_card(state: UIState) -> None:
    ctrl = state.controller
    with ui.element("div").classes("ab-card w-full p-6"), ui.column().classes("gap-3"):
        with ui.row().classes("items-baseline gap-3"):
            ui.label("Simulated experiment").classes("text-xl font-semibold")
            ui.label("a made-up experiment runs each design").classes("ab-sub text-sm")

        def show_running() -> None:
            run_button.set_visibility(not ctrl.is_running)
            pause_button.set_visibility(ctrl.is_running)
            update_live(state)  # the status chip

        def go() -> None:
            try:
                ctrl.start()
            except Exception as exc:  # noqa: BLE001
                ui.notify(str(exc), type="negative", multi_line=True, close_button=True)
                return
            show_running()

        def pause() -> None:
            ctrl.pause()
            show_running()

        async def step() -> None:
            if state._stepping:
                ui.notify("A step is already running.", type="info")
                return

            def work() -> dict:
                ctrl.step_once()
                return live_data(state)

            state._stepping = True
            try:
                data = await run.io_bound(work)
            except Exception as exc:  # noqa: BLE001
                ui.notify(str(exc), type="negative", multi_line=True, close_button=True)
                return
            finally:
                state._stepping = False
            if state.controller is ctrl:
                _surface_warning(state)
                update_live(state, data)

        def stop() -> None:
            ctrl.stop()
            state.refresh()

        # Run and Pause swap in place, so starting or pausing redraws nothing.
        with ui.row().classes("gap-2"):
            run_button = theme.primary("Run", go)
            pause_button = theme.primary("Pause", pause)
            run_button.set_visibility(not ctrl.is_running)
            pause_button.set_visibility(ctrl.is_running)
            ui.button("Step", on_click=step).props("outline no-caps")
            ui.button("Stop", on_click=stop).props("flat no-caps color=grey-8")


async def tick(state: UIState) -> None:
    """The auto-run timer: one step if running, then update in place.

    The step, the model fit and everything drawn from them (the summary, the
    figures, the expert panel's rows) are computed on a worker thread; the event
    loop only swaps the new content in, so the page stays clickable."""
    ctrl = state.controller
    if ctrl is None or not ctrl.is_running or state._stepping:
        return

    def work() -> dict | None:
        if not ctrl.tick():
            return None
        return live_data(state)

    state._stepping = True
    try:
        data = await run.io_bound(work)
    except Exception as exc:  # misconfiguration surfaces here; show it
        logger.exception("auto step failed")
        ctrl.stop()
        ui.notify(str(exc), type="negative", multi_line=True, close_button=True)
        state.refresh()
        return
    finally:
        state._stepping = False
    if state.controller is not ctrl:
        return  # the campaign was reset while this step ran
    if data is None:
        state.refresh()  # it stopped (budget, halt): redraw the controls once
        return
    _surface_warning(state)
    update_live(state, data)


# -- a halt ------------------------------------------------------------------------


def _halt_card(state: UIState) -> None:
    ctrl = state.controller
    campaign = ctrl.campaign

    async def answer(choice: str, reason: str | None = None) -> None:
        try:
            ctrl.resolve(choice, reason=reason or None)
        except Exception as exc:  # noqa: BLE001
            ui.notify(str(exc), type="negative", multi_line=True, close_button=True)
            return
        await _after_change(state)

    if campaign.halted_on:  # the question below names the design; show the error
        errors = [r.error for r in campaign.history.records if r.fault and r.error]
        why = f"It failed every time it ran. Last error: {errors[-1]}" if errors else ""
    else:
        why = "\n".join(line for line in (campaign.halted or "").split("\n")
                        if line and not line.startswith("Decide with resolve()"))
    with ui.element("div").classes("ab-card w-full p-6 border-red-300"), \
            ui.column().classes("gap-3 w-full"):
        ui.label("abbot stopped to ask you").classes("text-xl font-semibold")
        ui.label(why).classes("ab-sub whitespace-pre-line")
        if not campaign.halted_on:
            ui.label("Fix the cause, then carry on.").classes("text-sm")
            theme.primary("Carry on", lambda: answer("retry"), marker="halt-retry")
            return
        designs = "; ".join(", ".join(f"{k} {summary.fmt(v)}" for k, v in d.items())
                            for d in campaign.halted_on)
        ui.label(f"What went wrong at {designs}?").classes("font-medium")
        with ui.column().classes("gap-2 w-full"):
            with ui.row().classes("items-center gap-3"):
                theme.primary("The apparatus is fixed: try it again", lambda: answer("retry"),
                              marker="halt-retry")
                ui.button("Skip this design", on_click=lambda: answer("exclude")).props(
                    "outline no-caps").mark("halt-exclude")
            with ui.row().classes("items-center gap-3"):
                reason = ui.input("Why (optional)", placeholder="dark: flat profile").props(
                    "outlined dense").classes("w-64").mark("halt-reason")
                ui.button("The design itself gives nothing: avoid the region",
                          on_click=lambda: answer("bad_region", reason.value)).props(
                    "outline no-caps color=negative").mark("halt-bad-region")


# -- how it's going ----------------------------------------------------------------


def _progress_text(state: UIState) -> str:
    try:
        lines = summary.progress_lines(state.controller.campaign)
    except Exception as exc:  # a summary must not break the page
        logger.exception("progress summary failed")
        lines = [f"Couldn't summarize yet: {exc}"]
    return "\n\n".join(lines)


def _model_plot(state: UIState) -> None:
    spec = state.spec
    model = state.fitted_model()
    if model is None:
        state._plot_model = None
        ui.label("The model's picture appears after the first result.").classes("ab-sub")
        return
    if spec.n_outputs > 1 or len(spec.domain) > 2:
        with ui.row().classes("items-center gap-2"):
            if spec.n_outputs > 1:
                ui.select({j: o.name for j, o in enumerate(spec.outputs)}, value=state.viz_obj,
                          label="output", on_change=lambda e: _reviz(state, "viz_obj", e.value)
                          ).props("dense outlined").classes("w-36")
            if len(spec.domain) > 2:
                names = {i: d.name for i, d in enumerate(spec.domain)}
                ui.select(names, value=state.viz_vx, label="across",
                          on_change=lambda e: _reviz(state, "viz_vx", e.value)).props(
                    "dense outlined").classes("w-32")
                ui.select(names, value=state.viz_vy, label="and",
                          on_change=lambda e: _reviz(state, "viz_vy", e.value)).props(
                    "dense outlined").classes("w-32")
    state._plot_model = ui.plotly(model_figure(state, model)).classes("w-full").mark(
        "model-plot")


def model_figure(state: UIState, model) -> dict:
    campaign = state.controller.campaign
    try:
        return figures.model_fig(
            campaign.history, state.spec, model, state.selections(), policy=campaign.policy,
            pending=[item.x for item in campaign._inflight.values()],
        )
    except Exception as exc:  # a plot must not break the page
        logger.exception("model figure failed")
        return figures.blank_fig(f"Couldn't draw the model: {exc}")


def _reviz(state: UIState, attr: str, value) -> None:
    setattr(state, attr, value)
    state.refresh()


def live_data(state: UIState) -> dict:
    """Everything the live parts of the page show, computed without touching the
    page (safe on a worker thread): the summary, the model figure, the status, and
    the expert panel's parts."""
    model = state.fitted_model()
    return {
        "progress": _progress_text(state),
        "model": model_figure(state, model) if model is not None else None,
        "status": summary.status_words(state.controller),
        "expert": state._live_data() if state._live_data is not None else None,
    }


def update_live(state: UIState, data: dict | None = None) -> None:
    """Show new results by changing content in place: no element is rebuilt, so
    nothing flickers or moves while an auto-run steps."""
    if data is None:
        data = live_data(state)
    progress = state._progress
    if progress is not None and progress.content != data["progress"]:
        progress.set_content(data["progress"])
    if data["model"] is not None:
        if state._plot_model is None:
            state.refresh()  # the first result: build the plot once
            return
        state._plot_model.update_figure(data["model"])
    if state._status is not None:
        text, tone = data["status"]
        if state._status.text != text:
            state._status.set_text(text)
            state._status.classes(replace=f"ab-status {tone}")
    if state._live is not None and data["expert"] is not None:
        state._live(data["expert"])


def _surface_warning(state: UIState) -> None:
    campaign = state.controller.campaign
    msg = getattr(campaign, "last_warning", None)
    if msg:
        ui.notify(msg, type="warning", multi_line=True, close_button=True)
        campaign.last_warning = None


# -- finishing -------------------------------------------------------------------


def _finish(state: UIState) -> None:
    campaign = state.controller.campaign
    rec = campaign.recommend()
    with ui.dialog() as dialog, ui.element("div").classes("ab-card p-6 bg-white"), \
            ui.column().classes("gap-3"):
        ui.label("Finish with the current best?").classes("text-lg font-semibold")
        for line in _recommendation_lines(state, rec):
            ui.label(line).classes("ab-mono text-sm")
        ui.label("You can download the runs afterwards, or keep going.").classes("ab-sub text-sm")

        def done() -> None:
            state.controller.stop()
            state.finished = True
            dialog.close()
            state.refresh()

        with ui.row().classes("gap-3 pt-2"):
            theme.primary("Finish", done, marker="finish-confirm")
            ui.button("Not yet", on_click=dialog.close).props("flat no-caps")
    dialog.open()


def _recommendation_lines(state: UIState, rec: list[int]) -> list[str]:
    spec = state.spec
    records = state.controller.campaign.history.records
    lines = []
    for i in rec[:6]:
        r = records[i]
        setting = ", ".join(f"{k} {summary.fmt(v)}" for k, v in spec.as_dict(r.candidate).items())
        outs = ", ".join(f"{o.name} {summary.fmt(float(r.outputs.reshape(-1)[j]))}"
                         for j, o in enumerate(spec.outputs))
        lines.append(f"{setting}  →  {outs}")
    return lines or ["No recommendation yet."]


def _finished_card(state: UIState) -> None:
    from abbot import io as abbot_io

    campaign = state.controller.campaign
    rec = campaign.recommend()
    with ui.element("div").classes("ab-card w-full p-6"), ui.column().classes("gap-3 w-full"):
        ui.label("Finished").classes("text-xl font-semibold")
        ui.label("Recommended:" if len(rec) <= 1 else "Recommended (the trade-off):").classes(
            "ab-sub")
        for line in _recommendation_lines(state, rec):
            ui.label(line).classes("ab-mono")

        async def keep_going() -> None:
            state.finished = False
            state.controller.status = RunStatus.IDLE
            await _after_change(state)

        with ui.row().classes("items-center gap-6 pt-2 text-sm"):
            theme.link("Download the runs (CSV)", lambda: ui.download.content(
                abbot_io.history_to_csv(campaign.history, state.spec).encode(),
                "abbot_history.csv"))
            theme.link("Keep going", keep_going, marker="keep-going")
