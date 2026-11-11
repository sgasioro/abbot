"""The guided setup: five short steps, each one question.

Knobs (what can be set), Measurements (what each run gives), Goal (the best setting,
or every setting that hits a target), Extras (sources, the model, a name), and
Start (who runs the experiments, and what abbot will do). The rows these edit are
the same ones the expert panel shows. What only the expert panel's JSON can set
(whole-number knobs, constraints, several sources) shows here, tagged as such. The
first step also offers the other ways in: a saved campaign, a file, or writing
campaign.json in the expert panel.
"""

from __future__ import annotations

from nicegui import ui

from abbot.ui import summary, theme
from abbot.ui.builder import BAX_STRATEGY_LABELS, domain_words, parse_values
from abbot.ui.state import STEPS, UIState, _message


def render(state: UIState) -> None:
    """The current step, with Back / Next."""
    steps = [_knobs, _measurements, _goal, _extras, _start]
    with ui.column().classes("w-full gap-5 grow"):
        steps[state.step](state)
    error = ui.label("").classes("ab-error").mark("step-error")
    error.set_visibility(False)

    def go(to: int) -> None:
        if to > state.step:
            problem = state.check_step(state.step)
            if problem:
                error.set_text(problem)
                error.set_visibility(True)
                return
        state.step = to
        state.reached = max(state.reached, to)
        state.refresh()

    with ui.row().classes("w-full items-center justify-between pt-6"):
        if state.step > 0:
            theme.link("Back", lambda: go(state.step - 1), marker="back")
        else:
            ui.label("")
        if state.step < len(STEPS) - 1:
            theme.primary(f"Next: {STEPS[state.step + 1].lower()}",
                          lambda: go(state.step + 1), marker="next")
        else:
            theme.primary("Start the campaign", lambda: _create(state), marker="create")


def chips(state: UIState) -> None:
    """The steps along the header: done ones ticked, the current one highlighted."""
    with ui.row().classes("items-center gap-1 no-wrap max-sm:gap-0"):
        for i, name in enumerate(STEPS):
            done = i < state.step or (i <= state.reached and i != state.step)
            cls = "ab-chip" + (" current" if i == state.step else " done" if done else "")
            if i != state.step:
                cls += " ab-not-current"  # a phone shows only the current step
            reachable = i != state.step and i <= state.reached
            if reachable:
                cls += " reachable"
            chip = ui.element("div").classes(cls).mark(f"chip-{name}")
            if reachable:
                chip.on("click", lambda _, i=i: _jump(state, i))
            with chip:
                ui.label("✓" if done else str(i + 1)).classes("ab-dot")
                ui.label(name).classes("max-md:hidden" if i != state.step else "")


def _jump(state: UIState, i: int) -> None:
    state.step = i
    state.refresh()


async def _create(state: UIState) -> None:
    from abbot.ui import running

    try:
        state.start_campaign()
    except Exception as exc:  # noqa: BLE001 - a setup problem, shown to the person
        ui.notify(str(exc), type="negative", multi_line=True, close_button=True)
        return
    await running.started(state)


def _changed(state: UIState) -> None:
    """A field changed: refresh what shows the setup live (the expert panel)."""
    state.refresh_expert()


# -- 1. knobs ----------------------------------------------------------------


def _knobs(state: UIState) -> None:
    theme.title("What can you set?",
                "Each knob abbot may change between runs, and the range it can reach. "
                "Use Choices for settings that aren't numbers, like a material.")
    _other_starts(state)
    box = ui.column().classes("w-full gap-3")

    def redraw() -> None:
        box.clear()
        with box:
            for i, row in enumerate(state.variable_rows):
                _knob_row(state, row, i, redraw)

    def add() -> None:
        n = len(state.variable_rows) + 1
        state.variable_rows.append(
            {"kind": "continuous", "name": f"x{n}", "lower": 0.0, "upper": 1.0}
        )
        redraw()
        _changed(state)

    redraw()
    extra = _extra_words(state)
    if extra:
        with ui.row().classes("items-center gap-2"):
            ui.label(extra).classes("ab-sub text-sm").mark("extra-note")
            _expert_only(state)
    theme.link("+ Add a knob", add, marker="add-knob")


def _other_starts(state: UIState) -> None:
    """Ways in besides answering the steps: carry on a saved campaign, load a file,
    or write campaign.json by hand."""
    with ui.row().classes("items-center gap-x-5 gap-y-1 flex-wrap text-sm px-4 py-2 "
                          "rounded-lg").style("background: var(--ab-panel)").mark(
            "other-starts"):
        ui.label("Or:").classes("ab-sub")
        if state.workspace is not None:
            theme.link("Open a saved campaign", lambda: _open_saved(state), marker="open-saved")
        _import_spec(state)
        theme.link("Write campaign.json", lambda: open_json(state), marker="write-json")


def open_json(state: UIState) -> None:
    """Open the expert panel on the campaign.json editor."""
    state.expert = True
    state.expert_tab = "campaign.json"
    state.refresh()


def _extra_words(state: UIState) -> str:
    """The constraints and boxes only the JSON sets, in a few words."""
    n = len(state.extra.get("domain_constraints") or []) + len(
        state.extra.get("output_constraints") or [])
    parts = []
    if n:
        parts.append(f"{n} linear constraint{'s' if n > 1 else ''}")
    if state.extra.get("boxes"):
        k = len(state.extra["boxes"])
        parts.append(f"{k} box{'es' if k > 1 else ''}")
    return " and ".join(parts)


def _expert_only(state: UIState) -> None:
    """The mark on what's set in the expert panel's JSON, not in the steps; it
    opens the editor."""
    tag = ui.label("set in campaign.json").classes("ab-status grey text-xs cursor-pointer")
    tag.tooltip("Edit it in the expert panel")
    tag.on("click", lambda _: open_json(state))


def _knob_row(state: UIState, row: dict, i: int, redraw) -> None:
    if row["kind"] == "json":
        _json_knob(state, row, i, redraw)
        return
    with ui.element("div").classes("ab-card w-full p-4"), ui.column().classes("gap-2 w-full"):
        with ui.row().classes("items-center gap-3 w-full no-wrap"):
            ui.input("name", on_change=lambda _: _changed(state)).bind_value(row, "name").props(
                "outlined dense"
            ).classes("w-40").mark(f"knob-{i}-name")

            def kind(value: str, row=row) -> None:
                row["kind"] = value
                redraw()
                _changed(state)

            theme.segmented({"continuous": "Range", "enumerated": "Choices"}, row["kind"], kind)
            ui.space()
            if len(state.variable_rows) > 1:
                ui.button(icon="close", on_click=lambda: _remove(state.variable_rows, i, redraw,
                                                                  state)).props(
                    "flat dense round color=grey-6"
                ).tooltip("Remove this knob")
        if row["kind"] == "continuous":
            with ui.row().classes("items-center gap-3 w-full"):
                ui.number("from", on_change=lambda _: _changed(state)).bind_value(
                    row, "lower").props("outlined dense").classes("w-32").mark(f"knob-{i}-lower")
                ui.number("to", on_change=lambda _: _changed(state)).bind_value(
                    row, "upper").props("outlined dense").classes("w-32").mark(f"knob-{i}-upper")
            _knob_options(state, row)
        else:
            if not isinstance(row.get("values"), str):
                row["values"] = ", ".join(str(v) for v in row.get("values") or ["a", "b"])
            ui.input("choices, separated by commas",
                     on_change=lambda _: _changed(state)).bind_value(row, "values").props(
                "outlined dense"
            ).classes("w-full").mark(f"knob-{i}-values")


def _json_knob(state: UIState, row: dict, i: int, redraw) -> None:
    """A knob the steps can't edit (whole numbers, a simplex): shown, and removable."""
    with ui.element("div").classes("ab-card w-full p-4"), \
            ui.row().classes("items-center gap-3 w-full no-wrap"):
        ui.label(row.get("name") or "?").classes("font-medium w-40 truncate")
        ui.label(domain_words(row.get("raw") or {})).classes("ab-sub text-sm").mark(
            f"knob-{i}-json")
        _expert_only(state)
        ui.space()
        if len(state.variable_rows) > 1:
            ui.button(icon="close", on_click=lambda: _remove(state.variable_rows, i, redraw,
                                                              state)).props(
                "flat dense round color=grey-6").tooltip("Remove this knob")


def _knob_options(state: UIState, row: dict) -> None:
    """Log scale and a travel limit, folded behind one line."""
    row.setdefault("log", False)
    row.setdefault("max_step", None)
    row.setdefault("start", None)
    row.setdefault("_open", False)
    box = ui.column().classes("gap-2 w-full")

    def redraw() -> None:
        box.clear()
        with box:
            text = _options_summary(row)
            label = "hide options" if row["_open"] else (
                f"options: {text}" if text else "options: log scale, travel limit")
            theme.link(label, lambda: (row.update(_open=not row["_open"]), redraw())).classes(
                "text-sm")
            if not row["_open"]:
                return
            with ui.row().classes("items-center gap-3"):
                ui.checkbox("log scale", on_change=lambda _: _changed(state)).bind_value(
                    row, "log").tooltip("For a quantity spanning orders of magnitude.")
                ui.number("most it can move per run",
                          on_change=lambda e: (start.set_visibility(e.value is not None),
                                               _changed(state))).bind_value(
                    row, "max_step").props("outlined dense").classes("w-56")
                start = ui.number("where it is now", on_change=lambda _: _changed(state)
                                  ).bind_value(row, "start").props("outlined dense").classes("w-40")
                start.set_visibility(row.get("max_step") is not None)

    redraw()


def _options_summary(row: dict) -> str:
    """A knob's options in a few words: "log · moves ≤ 2 from 25"."""
    parts = []
    if row.get("log"):
        parts.append("log scale")
    if row.get("max_step") is not None:
        step = f"moves ≤ {summary.fmt(row['max_step'])}"
        if row.get("start") is not None:
            step += f" from {summary.fmt(row['start'])}"
        parts.append(step)
    return " · ".join(parts)


def _remove(rows: list, i: int, redraw, state: UIState) -> None:
    if 0 <= i < len(rows) and len(rows) > 1:
        rows.pop(i)
    redraw()
    _changed(state)


def _import_spec(state: UIState) -> None:
    """Fill the steps from a saved spec or campaign.json."""
    import json

    def on_upload(e) -> None:
        try:
            state.apply_config(json.loads(e.content.read().decode()))
        except Exception as exc:  # noqa: BLE001
            ui.notify(_message(exc), type="negative", multi_line=True, close_button=True)
            return
        state.step = state.reached = len(STEPS) - 1  # complete: straight to Start
        ui.notify("Loaded. Check what abbot will do, or go back to any step.",
                  type="positive")
        state.refresh()

    upload = ui.upload(on_upload=on_upload, auto_upload=True).props(
        "accept=.json flat bordered").classes("hidden")
    theme.link("Load a spec or campaign.json", lambda: upload.run_method("pickFiles"))


def _open_saved(state: UIState) -> None:
    """Pick a campaign saved in the workspace (by this UI or the agent) to carry on."""
    ws = state.workspace
    saved = ws.campaigns()
    with ui.dialog() as dialog, ui.element("div").classes("ab-card p-6 bg-white"), \
            ui.column().classes("gap-3 min-w-[22rem]"):
        ui.label("Open a saved campaign").classes("text-lg font-semibold")
        ui.label(f"In {ws.root}, shared with the abbot agent.").classes("ab-sub text-sm")
        if not saved:
            ui.label("Nothing saved yet.").classes("ab-sub")

        async def go(name: str) -> None:
            from abbot.ui import running

            try:
                state.open_saved(name)
            except Exception as exc:  # noqa: BLE001
                ui.notify(_message(exc), type="negative", multi_line=True, close_button=True)
                return
            dialog.close()
            await running.started(state)

        for item in saved:
            with ui.row().classes("items-center gap-3 w-full no-wrap"):
                if "error" in item:
                    ui.label(item["name"]).classes("ab-mono w-48 truncate")
                    ui.label(f"can't open: {item['error']}").classes("ab-error text-xs")
                    continue
                theme.link(item["name"], lambda n=item["name"]: go(n),
                           marker=f"saved-{item['name']}").classes("ab-mono w-48 truncate")
                ui.label(_saved_words(item)).classes("ab-sub text-sm")
        with ui.row().classes("gap-3 pt-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat no-caps")
    dialog.open()


def _saved_words(item: dict) -> str:
    words = [f"{item['runs']} run{'s' if item['runs'] != 1 else ''}"]
    if item["outstanding"]:
        words.append(f"{item['outstanding']} out")
    if item.get("halted"):
        words.append("halted")
    if item.get("open_elsewhere"):
        words.append("open elsewhere")
    return " · ".join(words)


# -- 2. measurements ------------------------------------------------------


def _measurements(state: UIState) -> None:
    theme.title("What does each run measure?",
                "Name every value you'll record. The next step says which to push up "
                "or down.")
    box = ui.column().classes("w-full gap-2")

    def redraw() -> None:
        box.clear()
        with box:
            for i, row in enumerate(state.objective_rows):
                with ui.row().classes("items-center gap-2 w-full no-wrap"):
                    ui.input("name", on_change=lambda _: _changed(state)).bind_value(
                        row, "name").props("outlined dense").classes("w-64").mark(
                        f"output-{i}-name")
                    if len(state.objective_rows) > 1:
                        ui.button(icon="close", on_click=lambda _, i=i: _remove(
                            state.objective_rows, i, redraw, state)).props(
                            "flat dense round color=grey-6").tooltip("Remove")

    def add() -> None:
        n = len(state.objective_rows) + 1
        state.objective_rows.append({"name": f"y{n}", "direction": "OBSERVE"})
        redraw()
        _changed(state)

    redraw()
    theme.link("+ Add a measurement", add, marker="add-output")


# -- 3. goal -------------------------------------------------------------------


def _goal(state: UIState) -> None:
    theme.title("What are you after?", "Pick one. You can change it before starting.")

    def pick(method: str) -> None:
        state.method = method
        state.refresh()

    with ui.column().classes("w-full gap-3"):
        theme.option("The best setting", _best_blurb(state), state.method == "optimize",
                     lambda: pick("optimize"), marker="goal-best")
        theme.option("Every setting that hits a target",
                     "A band, a threshold or the top few: a map of where it works.",
                     state.method == "bax", lambda: pick("bax"), marker="goal-target")
    with ui.element("div").classes("ab-card w-full p-5"):
        if state.method == "optimize":
            _directions(state)
        else:
            _target(state)


def _best_blurb(state: UIState) -> str:
    wants = [r for r in state.objective_rows if r.get("direction") in ("MAXIMIZE", "MINIMIZE")]
    if not wants:
        return "Push a measurement up or down, keeping others in range."
    words = [f"{'highest' if r['direction'] == 'MAXIMIZE' else 'lowest'} {r['name']}"
             for r in wants]
    text = ("The " + " and ".join(words)) if len(words) <= 2 else "A trade-off between them"
    return text + "."


def _directions(state: UIState) -> None:
    with ui.column().classes("w-full gap-3"):
        for i, row in enumerate(state.objective_rows):
            row.setdefault("min_val", None)
            row.setdefault("max_val", None)
            with ui.row().classes("items-center gap-3 w-full no-wrap max-sm:flex-wrap"):
                ui.label(row.get("name") or "?").classes("w-24 font-medium truncate")

                def direction(value: str, row=row) -> None:
                    row["direction"] = value
                    state.refresh()

                theme.segmented({"MAXIMIZE": "Higher", "MINIMIZE": "Lower",
                                 "OBSERVE": "Just watch"}, row.get("direction", "OBSERVE"),
                                direction, marker=f"direction-{i}")
                ui.number("at least", on_change=lambda _: _changed(state)).bind_value(
                    row, "min_val").props("outlined dense").classes("w-24").mark(
                    f"output-{i}-min")
                ui.number("at most", on_change=lambda _: _changed(state)).bind_value(
                    row, "max_val").props("outlined dense").classes("w-24").mark(
                    f"output-{i}-max")
    ui.label("At least / at most keep a measurement within limits; leave them blank "
             "for none.").classes("text-xs ab-sub pt-2")


GOAL_KINDS = {"band": "A band", "level": "A threshold", "topk": "The top few",
              "percentile": "A top percent"}


def _target(state: UIState) -> None:
    goal = state.goal_row
    names = [r.get("name") for r in state.objective_rows]

    def kind(value: str) -> None:
        goal["kind"] = value
        state.refresh()

    kinds = dict(GOAL_KINDS)
    if len(names) > 1:
        kinds.pop("percentile")
        if goal.get("kind") == "percentile":
            goal["kind"] = "band"
    theme.segmented(kinds, goal.get("kind", "band"), kind, marker="goal-kind")
    k = goal.get("kind", "band")
    with ui.column().classes("w-full gap-3 pt-3"):
        if k == "band":
            ui.label("Every setting where each output lands in its range. Leave both "
                     "ends blank for an output that doesn't matter.").classes("ab-sub text-sm")
            bands = goal.setdefault("bands", {})
            for name in names:
                band = bands.setdefault(name, {"lo": None, "hi": None})
                with ui.row().classes("items-center gap-3"):
                    ui.label(name).classes("w-28 font-medium")
                    ui.number("from", on_change=lambda _: _changed(state)).bind_value(
                        band, "lo").props("outlined dense").classes("w-32").mark(f"band-{name}-lo")
                    ui.number("to", on_change=lambda _: _changed(state)).bind_value(
                        band, "hi").props("outlined dense").classes("w-32").mark(f"band-{name}-hi")
        elif k == "level":
            goal.setdefault("output", names[0])
            goal.setdefault("upper", False)
            with ui.row().classes("items-center gap-3"):
                ui.label("Every setting where").classes("ab-sub")
                ui.select({n: n for n in names}, on_change=lambda _: _changed(state)).bind_value(
                    goal, "output").props("outlined dense").classes("w-36")
                ui.select({False: "is at least", True: "is at most"},
                          on_change=lambda _: _changed(state)).bind_value(goal, "upper").props(
                    "outlined dense").classes("w-36")
                ui.number("value", on_change=lambda _: _changed(state)).bind_value(
                    goal, "threshold").props("outlined dense").classes("w-32").mark("threshold")
        elif k == "topk":
            goal.setdefault("output", names[0])
            goal.setdefault("k", 5)
            goal.setdefault("largest", True)
            with ui.row().classes("items-center gap-3"):
                ui.label("The").classes("ab-sub")
                ui.number(on_change=lambda _: _changed(state), min=1, precision=0).bind_value(
                    goal, "k").props("outlined dense").classes("w-20")
                ui.label("settings with the").classes("ab-sub")
                ui.select({True: "highest", False: "lowest"},
                          on_change=lambda _: _changed(state)).bind_value(goal, "largest").props(
                    "outlined dense").classes("w-32")
                ui.select({n: n for n in names}, on_change=lambda _: _changed(state)).bind_value(
                    goal, "output").props("outlined dense").classes("w-36")
        else:
            goal.setdefault("percentile", 80.0)
            with ui.row().classes("items-center gap-3"):
                ui.label("The top").classes("ab-sub")
                pct = ui.number(value=100 - float(goal["percentile"]), min=1, max=99,
                                suffix="%").props("outlined dense").classes("w-28")
                pct.on_value_change(lambda e: (goal.update(
                    percentile=100 - float(e.value or 20)), _changed(state)))
                ui.label(f"of settings by {names[0]}").classes("ab-sub")


# -- 4. extras -------------------------------------------------------------


def _extras(state: UIState) -> None:
    theme.title("Anything else?", "All optional.")
    with ui.element("div").classes("ab-card w-full p-5"), ui.column().classes("gap-2 w-full"):
        ui.label("Name").classes("font-medium")
        ui.input(on_change=lambda _: state.refresh_header()).bind_value(state, "name").props(
            "outlined dense").classes("w-80").mark("campaign-name")
    with ui.element("div").classes("ab-card w-full p-5"), ui.column().classes("gap-2 w-full"):
        _sources(state)
    if len(state.objective_rows) > 1:
        with ui.element("div").classes("ab-card w-full p-5"), ui.column().classes("gap-2"):
            ui.label("Do the measurements move together?").classes("font-medium")
            theme.segmented({"default": "Model each on its own", "correlated": "Model them together"},
                            state.model, lambda v: (setattr(state, "model", v), _changed(state)),
                            marker="model")
            ui.label("Together lets one measurement inform another; use it when they "
                     "rise and fall together.").classes("ab-sub text-sm")
    if state.method == "bax":
        with ui.element("div").classes("ab-card w-full p-5"), ui.column().classes("gap-2"):
            ui.label("How should abbot search for the target?").classes("font-medium")
            labels = {k: v.split(" (")[0] for k, v in BAX_STRATEGY_LABELS.items()}
            theme.segmented(labels, state.bax_strategy,
                            lambda v: (setattr(state, "bax_strategy", v), _changed(state)),
                            marker="strategy")
            ui.label("Balanced suits most campaigns.").classes("ab-sub text-sm")


def _sources(state: UIState) -> None:
    if state.extra.get("sources"):
        names = [src.get("domain", {}).get("name", "?") for src in state.extra["sources"]]
        ui.label("Sources").classes("font-medium")
        with ui.row().classes("items-center gap-2"):
            ui.label(", ".join(names)).classes("ab-mono text-sm").mark("json-sources")
            _expert_only(state)
        ui.label("abbot chooses where each run happens, weighing cost against what it "
                 "learns. Enter each run's cost when you record it, if you know it.").classes(
            "ab-sub text-sm")
        return
    row = state.source_row
    box = ui.column().classes("w-full gap-2")

    def redraw() -> None:
        box.clear()
        if not state.source_on:
            return
        with box:
            with ui.row().classes("items-center gap-3"):
                ui.input("name", on_change=lambda _: _changed(state)).bind_value(
                    row, "name").props("outlined dense").classes("w-40")
                theme.segmented({"enumerated": "Named sources", "continuous": "A fidelity range"},
                                row["kind"],
                                lambda v: (row.update(kind=v), redraw(), _changed(state)))
            if row["kind"] == "enumerated":
                _labeled_source(state, row)
            else:
                _ranged_source(state, row)
            ui.label("abbot chooses where each run happens, weighing cost against what it "
                     "learns, and recommends at the one you trust. Enter each run's cost "
                     "when you record it, if you know it.").classes("ab-sub text-sm")

    ui.switch("Runs come from more than one source (a simulation and the experiment, "
              "several fidelities)", value=state.source_on,
              on_change=lambda e: (setattr(state, "source_on", e.value), redraw(),
                                   _changed(state))).mark("sources")
    redraw()


def _labeled_source(state: UIState, row: dict) -> None:
    def labels() -> list[str]:
        return [str(v) for v in parse_values(row.get("values") or "")]

    target = None

    def refresh_target() -> None:
        if target is None:
            return
        options = labels()
        if row.get("target") not in options:
            row["target"] = options[-1] if options else None
        target.set_options(options, value=row["target"])
        _changed(state)

    with ui.row().classes("items-center gap-3 w-full"):
        ui.input("sources, separated by commas", on_change=lambda _: refresh_target()).bind_value(
            row, "values").props("outlined dense").classes("w-72")
        target = ui.select(labels(), value=row.get("target"), label="the one to trust").props(
            "outlined dense").classes("w-44").on_value_change(
            lambda e: (row.update(target=e.value), _changed(state)))
    refresh_target()


def _ranged_source(state: UIState, row: dict) -> None:
    with ui.row().classes("items-center gap-3 w-full"):
        ui.number("from", on_change=lambda _: _changed(state)).bind_value(row, "lower").props(
            "outlined dense").classes("w-28")
        ui.number("to", on_change=lambda _: _changed(state)).bind_value(row, "upper").props(
            "outlined dense").classes("w-28")
        ui.input("recommend at", on_change=lambda _: _changed(state)).bind_value(
            row, "target").props("outlined dense").classes("w-32")
    with ui.row().classes("items-center gap-3"):
        theme.segmented({"high": "Higher is more accurate", "low": "Lower is more accurate"},
                        row.get("accurate", "high"),
                        lambda v: (row.update(accurate=v), _changed(state)))
        ui.checkbox("log scale", on_change=lambda _: _changed(state)).bind_value(row, "log")


# -- 5. start ------------------------------------------------------------------


def _start(state: UIState) -> None:
    theme.title("Ready to start", "Who runs the experiments?")

    def who(mock: bool) -> None:
        state.mock = mock
        state.refresh()

    with ui.column().classes("w-full gap-3"):
        theme.option("I do", "abbot suggests a design; you run it and enter what you "
                     "measured.", not state.mock, lambda: who(False), marker="who-me")
        theme.option("A simulation", "Try abbot on a made-up experiment and watch it "
                     "work.", state.mock, lambda: who(True), marker="who-sim")
    with ui.element("div").classes("ab-card w-full p-5"), ui.column().classes("gap-1"):
        theme.section("What abbot will do")
        ui.markdown("\n".join(f"- {line}" for line in summary.plan_lines(state))).classes(
            "text-[0.95rem]").mark("plan")
