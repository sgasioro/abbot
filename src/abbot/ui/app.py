"""NiceGUI web client for abbot: a guided setup, a run view, and an expert panel.

The page has a header (the setup steps, or the campaign and what it's waiting for),
the main column (:mod:`abbot.ui.setup` while setting up, :mod:`abbot.ui.running`
once started), and an optional expert panel (:mod:`abbot.ui.expert`) beside it.

All optimization logic is in the core (``Campaign``, ``RunController``, ``builder``,
``analysis``); these modules build widgets and wire events. Per-client state lives
on a ``UIState`` made inside the page function. Campaigns are saved in a directory
shared with the agent (``~/abbot-campaigns`` unless ``--dir`` or ``ABBOT_UI_DIR``
says otherwise), through one :class:`~abbot.agent.workspace.Workspace` per process.
Run with ``python -m abbot.ui`` (``--experiment module:Name`` adds your own
experiment), or show a campaign built in code with :func:`serve`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from nicegui import ui

from abbot.experiment import MockAnalyticExperiment  # noqa: F401 - tests patch it here
from abbot.ui import expert, running, setup, summary, theme
from abbot.ui.state import STEPS, UIState

logger = logging.getLogger(__name__)

__all__ = ["UIState", "main", "serve"]

_workspaces: dict = {}
# What every page this process serves starts with: your experiment (``yours``), and
# a campaign some code runs (``controller``, from serve()).
_served: dict = {"yours": None, "controller": None, "name": None}


def workspace():
    """The workspace for ``ABBOT_UI_DIR`` (default ``~/abbot-campaigns``), shared by
    every page this process serves, so two tabs on one campaign share it."""
    from abbot.agent.launch import DEFAULT_DIR
    from abbot.agent.workspace import Workspace

    root = Path(os.environ.get("ABBOT_UI_DIR") or DEFAULT_DIR).expanduser().resolve()
    if root not in _workspaces:
        _workspaces[root] = Workspace(root)
    return _workspaces[root]


def _build_page() -> None:
    state = UIState()
    state.workspace = workspace()
    state.yours = _served["yours"]
    state.own = state.yours is not None
    if _served["controller"] is not None:
        _show_served(state, _served["controller"], _served["name"])
    else:
        state.from_declared()  # the knobs and measurements your experiment declares
    theme.apply()

    @ui.refreshable
    def header() -> None:
        with ui.row().classes(
            "w-full items-center justify-between no-wrap px-6 py-3 max-sm:px-3"
        ):
            state._status = None
            with ui.row().classes("items-center gap-6 no-wrap max-sm:gap-2 min-w-0"):
                ui.label("abbot").classes("ab-brand")
                if state.controller is None:
                    setup.chips(state)
                else:
                    ui.label(state.name).classes("text-base max-sm:hidden").mark("campaign-title")
                    text, tone = summary.status_words(state.controller)
                    state._status = ui.label(text).classes(f"ab-status {tone}").mark("status")
            with ui.row().classes("items-center gap-4 no-wrap max-sm:gap-2 shrink-0"):
                if state.controller is not None:
                    theme.link("New campaign", lambda: _new_campaign(state), marker="new"
                               ).classes("max-sm:hidden")
                look = "outline color=secondary" if state.expert else "unelevated color=secondary"
                with ui.button(on_click=lambda: _toggle_expert(state)).props(
                    f"no-caps {look}").classes("rounded-lg ab-mono font-semibold").mark(
                    "expert").tooltip("Hide the expert panel" if state.expert
                                      else "Every field, the runs and the model"):
                    ui.label("{ }")
                    ui.label("Expert").classes("pl-2 max-sm:hidden")

    @ui.refreshable
    def expert_panel() -> None:
        expert.render(state)

    @ui.refreshable
    def column() -> None:
        if state.controller is None:
            setup.render(state)
        else:
            running.render(state)

    @ui.refreshable
    def body() -> None:
        state._live = state._live_data = None
        state._expert_parts, state._editor_config = [], None  # set if the panel shows
        with ui.row().classes("w-full no-wrap items-stretch gap-0 max-md:flex-col"):
            width = "max-w-3xl mx-auto" if not state.expert else "max-w-2xl"
            with ui.column().classes(f"grow w-full {width} px-8 py-8 max-sm:px-4 gap-4"):
                column()
            if state.expert:
                with ui.element("div").classes(
                    "ab-expert w-[48%] shrink-0 max-md:w-full min-h-[calc(100vh-4rem)]"
                ):
                    expert_panel()

    with ui.header().classes("ab-header p-0"):
        header()
    body()
    state._header = header
    state._main = body
    state._column = column
    state._expert = expert_panel

    # One auto-run timer, made once; it reads stable state, so redrawing a panel
    # never orphans it.
    ui.timer(0.5, lambda: running.tick(state))
    if state.controller is not None:  # serve()'s campaign: the next design, ready
        ui.timer(0.1, lambda: running.suggest_next(state), once=True)


def _show_served(state: UIState, controller, name: str | None) -> None:
    """Open the page on serve()'s campaign. The steps hold its setup where they can
    show it, for New campaign → Start from this setup."""
    from abbot.ui.builder import spec_json

    try:
        state.apply_config({"spec": spec_json(controller.campaign.spec)})
    except Exception:  # a setup the steps can't hold: the campaign still runs
        logger.info("serve: the steps can't show this campaign's setup", exc_info=True)
    state.name = name or "Campaign"
    state.step = state.reached = len(STEPS) - 1
    state.attach(controller)


def _toggle_expert(state: UIState) -> None:
    state.expert = not state.expert
    state.refresh()


def _new_campaign(state: UIState) -> None:
    with ui.dialog() as dialog, ui.element("div").classes("ab-card p-6 bg-white"), \
            ui.column().classes("gap-3"):
        ui.label("Start a new campaign?").classes("text-lg font-semibold")
        if state.saved_as is not None:
            ui.label(f"This one stays saved as {state.saved_as}. Open it again from the "
                     "first step, or from the abbot agent.").classes("ab-sub max-w-md")
        else:
            ui.label("This one isn't saved: its runs are lost unless you download them "
                     "first (expert panel, Runs, export csv).").classes("ab-sub max-w-md")

        def go(keep_setup: bool) -> None:
            dialog.close()
            state.new_campaign(keep_setup)
            state.refresh()

        with ui.row().classes("gap-3 pt-2 flex-wrap"):
            theme.primary("Start from this setup", lambda: go(True), marker="new-keep")
            ui.button("Start from scratch", on_click=lambda: go(False)).props(
                "outline no-caps").mark("new-scratch")
            ui.button("Cancel", on_click=dialog.close).props("flat no-caps")
    dialog.open()


def main(argv: list[str] | None = None) -> None:
    """Entry point: build the page and launch the server.

    Flags (also honored via env vars for headless/cluster runs):
      --host (ABBOT_UI_HOST, default 127.0.0.1)
      --port (ABBOT_UI_PORT, default 8080) — auto-picks a free port if taken
      --strict-port — fail instead of auto-picking
      --show / --no-show (ABBOT_UI_SHOW=1/0) — open a browser tab on start
      --dir (ABBOT_UI_DIR, default ~/abbot-campaigns) — where campaigns are saved;
        the abbot agent (``abbot-mcp``) saves there too.
      --experiment module:Name (or file.py:Name) — your experiment, offered on the
        Start step to run each design (see :mod:`abbot.ui.yours`).
      --command "./measure.sh {temp} {conc}" — a command as your experiment: run
        for each design, printing its outputs as JSON on its last line (see
        :class:`abbot.experiment.CommandExperiment`); --timeout SECONDS for each run.
      --spec spec.json (or module:NAME) — the spec it works with, if it has no
        ``spec`` attribute: the setup starts from it and stays within it.
    """
    import argparse

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
    parser.add_argument("--dir", default=os.environ.get("ABBOT_UI_DIR"),
                        help="where campaigns are saved, shared with the abbot agent "
                        "(default: ~/abbot-campaigns)")
    yours_group = parser.add_mutually_exclusive_group()
    yours_group.add_argument("--experiment", metavar="MODULE:NAME",
                             help="your experiment (an object with run(design), or a class "
                             "or function making one; module:Name or file.py:Name), offered "
                             "to run each design")
    yours_group.add_argument("--command", metavar='"PROGRAM {knob} ..."',
                             help="a command to run for each design, with {knob} for each "
                             "value; it prints the outputs as JSON on its last line")
    parser.add_argument("--timeout", type=float, metavar="SECONDS",
                        help="with --command: fail a run that takes longer than this")
    parser.add_argument("--spec", metavar="FILE.json|MODULE:NAME",
                        help="the spec your experiment works with (a spec or campaign.json, "
                        "or module:NAME), if it has no spec attribute: the setup starts "
                        "from it and stays within its knobs, ranges and measurements")
    args = parser.parse_args(argv)
    if args.spec and not (args.experiment or args.command):
        parser.error("--spec goes with --experiment or --command (it's the spec your "
                     "experiment works with)")
    if args.timeout is not None and not args.command:
        parser.error("--timeout goes with --command")
    if args.dir:
        os.environ["ABBOT_UI_DIR"] = args.dir
    if args.experiment or args.command:
        from abbot.ui import yours

        try:
            if args.command:
                _served["yours"] = yours.command(args.command, spec=args.spec,
                                                 timeout=args.timeout)
            else:
                _served["yours"] = yours.load(args.experiment, spec=args.spec)
        except Exception as exc:  # noqa: BLE001 - a bad reference: say why and stop
            flags = (f"--command {args.command!r}" if args.command
                     else f"--experiment {args.experiment}")
            flags += f" --spec {args.spec}" if args.spec else ""
            parser.error(f"{flags}: {exc}")
    _launch(args.host, args.port, args.show, args.strict_port)


def serve(
    campaign=None,
    experiment=None,
    *,
    spec=None,
    evaluate=None,
    name: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    show: bool = True,
) -> None:
    """Open the UI from Python, and serve it until interrupted.

    ``campaign`` (a :class:`~abbot.session.Campaign` built in code, with any policy or
    model) opens straight on its run view, shared by every tab, and isn't saved to
    the campaigns folder: your code owns it. ``experiment`` runs each design (an
    object with ``run(design)`` or ``run_batch(designs)``, or a class or function
    that makes one from the spec), with ``evaluate`` reducing what it returns to
    the outputs; it is also offered on the Start step of a new campaign. With only
    an experiment, the page starts at the first step. Without one, you enter each
    result. ``name`` titles the campaign.

    ``spec`` is the spec the experiment works with (by default its ``spec``
    attribute, if any): a new campaign's setup starts from it and stays within its
    knobs, ranges and measurements, and ``campaign``'s spec must stay within it too.

    Call it from a script (it blocks, as ``ui.run`` does)::

        from abbot.ui import serve
        serve(Campaign(spec), Rig(port="/dev/ttyUSB0"))
    """
    from abbot.ui.controller import RunController
    from abbot.ui.yours import YourExperiment

    if spec is not None and experiment is None:
        raise ValueError("serve(spec=...) is the spec your experiment works with; pass "
                         "the experiment too.")
    yours = None if experiment is None else YourExperiment(experiment, spec=spec,
                                                           evaluate=evaluate)
    _served["yours"] = yours
    _served["name"] = name
    _served["controller"] = None
    if campaign is not None:
        made = None if yours is None else yours.make(campaign.spec)
        _served["controller"] = RunController(
            campaign, experiment=made,
            evaluate=None if made is None else yours.outputs(campaign.spec))
    _launch(host, port, show, strict=False)


def _launch(host: str, port: int, show: bool, strict: bool) -> None:
    if not strict:
        free = _free_port(host, port)
        if free != port:
            logger.warning("port %d is in use; using %d instead.", port, free)
            print(f"abbot.ui: port {port} busy -> using {free}")
            port = free

    @ui.page("/")
    def index() -> None:
        _build_page()

    ui.run(title="abbot", host=host, port=port, show=show, reload=False)


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
