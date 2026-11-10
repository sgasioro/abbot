"""Per-client UI state: the setup rows, where the person is in the steps, and the
live controller once the campaign starts.

Framework-free apart from holding NiceGUI refreshables, so the setup logic (the
spec preview, the campaign config, starting a campaign) is testable without a page.

The setup round-trips through ``campaign.json``, the format the agent's
:class:`~abbot.agent.workspace.Workspace` saves: :meth:`UIState.campaign_config`
writes it and :meth:`UIState.apply_config` reads it back into the steps. What the
steps can't show (whole-number knobs, linear constraints, several sources, a list
of starting designs) is kept in :attr:`UIState.extra` and edited as JSON.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from abbot.experiment import MockAnalyticExperiment
from abbot.policy import BAXPolicy, default_policy
from abbot.problem import ProblemSpec
from abbot.session import Campaign, default_initial_designs
from abbot.ui import figures
from abbot.ui.builder import (
    BAX_STRATEGY_LABELS,
    MODEL_LABELS,
    build_constructor,
    build_goal,
    build_spec,
    build_start,
    default_start,
    spec_json,
    spec_to_rows,
)
from abbot.ui.controller import RunController

if TYPE_CHECKING:
    from abbot.agent.workspace import Workspace

STEPS = ["Knobs", "Measurements", "Goal", "Extras", "Start"]

# The keys of campaign.json. ``abbot_version`` is written by the workspace and
# ignored on reading.
CONFIG_KEYS = ("spec", "goal", "strategy", "model", "initial_designs", "max_retries",
               "abbot_version")


class UIState:
    """One browser tab's session: editor rows, the step, the controller."""

    def __init__(self) -> None:
        self.name: str = "My campaign"
        self.variable_rows: list[dict] = [
            {"kind": "continuous", "name": "x1", "lower": 0.0, "upper": 1.0}
        ]
        self.objective_rows: list[dict] = [{"name": "y", "direction": "MAXIMIZE"}]
        # "optimize": the best setting (BO / multi-objective). "bax": every setting
        # that hits a target (a goal + strategy; the outputs become OBSERVE).
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
        self.mock: bool = False  # who runs the experiments: the person, or a simulation
        # What the steps don't edit, as JSON: spec fields (EXTRA_SPEC_FIELDS),
        # ``initial_designs`` other than the current setting, ``max_retries``.
        self.extra: dict = {}
        # Where campaigns are saved (shared with the agent); None keeps them in memory.
        self.workspace: Workspace | None = None
        self.saved_as: str | None = None  # the running campaign's name there
        self.step: int = 0  # index into STEPS while setting up
        self.reached: int = 0  # the furthest step reached (earlier ones are done)
        self.expert: bool = False  # the expert panel is open
        self.expert_tab: str | None = None  # its tab last chosen (kept across redraws)
        self.controller: RunController | None = None
        self.spec: ProblemSpec | None = None
        self.n_suggest: int = 1  # how many designs the expert panel's Suggest asks for
        self.hold: dict = {}  # {variable: value} the expert panel's Suggest holds fixed
        self.focus: str | None = None  # the outstanding id "Run this next" shows
        self.finished: bool = False  # the person took the current best and finished
        self.suggesting: bool = False  # a suggestion is being computed
        # Model-plot selections (which output / which variables to slice).
        self.viz_obj = 0
        self.viz_vx = 0
        self.viz_vy = 1
        # A step runs on a worker thread; this stops a second one starting.
        self._stepping = False
        # Refreshables and live widgets, set while the page is built.
        self._header = None
        self._main = None
        self._column = None  # the main column alone (not the expert panel)
        self._expert = None
        self._expert_parts: list = []  # the expert panel's parts a JSON edit redraws
        self._editor_config = None  # the campaign.json the expert editor holds
        self._live = None  # shows new data in the expert panel's live parts
        self._live_data = None  # computes that data (no page access; any thread)
        self._status = None  # the header's status chip
        self._progress = None
        self._plot_model = None

    # -- setup --

    def output_rows(self) -> list[dict]:
        """The output rows as the engine sees them: OBSERVE throughout for a target
        goal (the goal reads them; nothing is optimized)."""
        if self.method == "bax":
            return [{**r, "direction": "OBSERVE"} for r in self.objective_rows]
        return self.objective_rows

    def build_spec(self) -> ProblemSpec:
        source = self.source_row if self.source_on else None
        return build_spec(self.variable_rows, self.output_rows(), source, self.extra)

    def preview(self) -> tuple[ProblemSpec | None, str | None]:
        """The spec the rows describe so far, or the reason they don't yet."""
        try:
            return self.build_spec(), None
        except Exception as exc:  # noqa: BLE001 - shown to the person as is
            return None, _message(exc)

    def goal_json(self) -> dict | None:
        """The goal as JSON (the MCP server's format), or None when optimizing."""
        if self.method != "bax":
            return None
        row = self.normalized_goal_row()
        kind = row.get("kind", "band")
        if kind == "band":
            return {"kind": "band",
                    "bounds": {k: list(v) for k, v in (row.get("bounds") or {}).items()}}
        if kind == "level":
            return {"kind": "level", "output": row.get("output"),
                    "threshold": row.get("threshold"), "upper": bool(row.get("upper", False))}
        if kind == "topk":
            return {"kind": "topk", "output": row.get("output"), "k": int(row.get("k", 1)),
                    "largest": bool(row.get("largest", True))}
        return {"kind": "percentile", "percentile": row.get("percentile", 80.0)}

    def campaign_config(self) -> dict:
        """The campaign as ``campaign.json`` (the format ``abbot-mcp`` saves):
        spec, goal, strategy, model, and the starting designs. Raises if the rows
        don't form a spec yet."""
        spec = self.build_spec()
        config: dict = {"spec": spec_json(spec)}
        goal = self.goal_json()
        if goal is not None:
            config["goal"] = goal
            config["strategy"] = self.bax_strategy
        config["model"] = self.model
        start = self.extra.get("initial_designs") or build_start(spec, self.variable_rows)
        if start is not None:
            config["initial_designs"] = start
        if self.extra.get("max_retries") is not None:
            config["max_retries"] = self.extra["max_retries"]
        return config

    def apply_config(self, config: dict) -> None:
        """Fill the steps from a ``campaign.json`` (or a bare spec). Checks it all
        first and raises with the reason, leaving the setup as it was, if it isn't
        a complete campaign."""
        if not isinstance(config, dict):
            raise TypeError("Expected a JSON object with a \"spec\".")
        if "domain" in config and "spec" not in config:
            config = {"spec": config}  # a bare spec
        unknown = sorted(set(config) - set(CONFIG_KEYS))
        if unknown:
            raise ValueError(f"Unknown key {unknown[0]!r}; campaign.json has "
                             f"{', '.join(CONFIG_KEYS[:-1])}.")
        if "spec" not in config:
            raise ValueError("campaign.json needs a \"spec\".")
        spec = ProblemSpec.model_validate(config["spec"])
        variables, outputs, source_row, extra = spec_to_rows(spec)
        fields: dict = {"variable_rows": variables, "objective_rows": outputs,
                        "source_on": source_row is not None}
        if source_row is not None:
            fields["source_row"] = source_row
        goal = config.get("goal")
        fields["method"] = "bax" if goal else "optimize"
        if goal:
            fields["goal_row"] = _goal_row(goal)
            # As written, before the steps' tidying (which drops unknown outputs).
            build_goal({**goal, "bounds": {k: tuple(v) for k, v in
                                           (goal.get("bounds") or {}).items()}}, spec)
        strategy = config.get("strategy") or "SwitchBAX"
        if strategy not in BAX_STRATEGY_LABELS:
            raise ValueError(f"strategy must be one of {list(BAX_STRATEGY_LABELS)}.")
        fields["bax_strategy"] = strategy
        model = config.get("model") or "default"
        if model not in MODEL_LABELS:
            raise ValueError(f"model must be one of {list(MODEL_LABELS)}.")
        fields["model"] = model
        designs = config.get("initial_designs")
        if designs:
            if not isinstance(designs, list) or not all(isinstance(d, dict) for d in designs):
                raise ValueError("initial_designs is a list of {variable: value} objects.")
            if spec.has_travel_limits and len(designs) == 1:
                # The current setting, shown on the knobs; by name, so a renamed knob
                # just needs its start again if it has a travel limit.
                for row, d in zip(variables, spec.domain):
                    value = designs[0].get(row["name"])
                    if getattr(d, "max_step", None) is None and value == default_start(d):
                        value = None  # the default: not something the person set
                    row["start"] = value
            else:
                for i, d in enumerate(designs):
                    try:
                        spec.as_tensor(d)
                    except Exception as exc:  # noqa: BLE001 - say which design
                        raise ValueError(f"initial_designs[{i}]: {_message(exc)}") from None
                extra["initial_designs"] = designs
        retries = config.get("max_retries")
        if retries is not None and retries != 1:
            extra["max_retries"] = int(retries)
        fields["extra"] = extra
        trial = copy.copy(self)  # the setup as it would be, checked before it's taken
        for key, value in fields.items():
            setattr(trial, key, value)
        trial.campaign_config()
        trial.build_policy(trial.build_spec())
        if trial.method != "bax" and not trial.build_spec().n_objectives:
            raise ValueError("Nothing to optimize: give an output a direction "
                             "(MAXIMIZE or MINIMIZE), or add a goal.")
        build_constructor(trial.build_spec(), model)
        for key, value in fields.items():
            setattr(self, key, value)

    def save_path(self) -> str | None:
        """Where starting the campaign will save it, or None (a simulation, or no
        workspace)."""
        if self.workspace is None or self.mock:
            return None
        return str(self.workspace.root / self.workspace.free_name(self.name))

    def normalized_goal_row(self) -> dict:
        """The goal editor's row as ``build_goal`` takes it: a band goal's
        per-output ``{"lo", "hi"}`` entries become ``bounds = {name: (lo, hi)}``,
        leaving out outputs with neither end set."""
        row = dict(self.goal_row)
        names = [r.get("name") for r in self.objective_rows]
        if row.get("kind", "band") == "band":
            bounds: dict[str, tuple] = {}
            for name, band in (row.get("bands") or {}).items():
                lo, hi = band.get("lo"), band.get("hi")
                if name not in names or (lo is None and hi is None):
                    continue
                bounds[name] = (lo, hi)
            row["bounds"] = bounds
        elif row.get("output") not in names and names:
            row["output"] = names[0]
        return row

    # Kept for callers of the old name.
    _normalized_goal_row = normalized_goal_row

    def check_step(self, step: int) -> str | None:
        """Why the person can't leave ``step`` yet, or None. Each step checks what
        it sets: the knobs, the measurements' names, the goal, the extras."""
        try:
            if step == 0:
                if not self.variable_rows:
                    return "Add at least one knob."
                spec = build_spec(self.variable_rows, [{"name": "_", "direction": "MAXIMIZE"}])
                build_start(spec, self.variable_rows)
            elif step == 1:
                names = [(r.get("name") or "").strip() for r in self.objective_rows]
                if not names:
                    return "Add at least one measurement."
                if "" in names:
                    return "Give every measurement a name."
                if len(set(names)) != len(names):
                    return "Two measurements have the same name."
                knobs = {r.get("name") for r in self.variable_rows}
                if knobs & set(names):
                    return f"{min(knobs & set(names))!r} is both a knob and a measurement."
            elif step == 2:
                spec = self.build_spec()
                if self.method == "bax":
                    build_goal(self.normalized_goal_row(), spec)
                elif not spec.n_objectives:
                    return "Pick Higher or Lower for at least one measurement."
            elif step == 3:
                build_constructor(self.build_spec(), self.model)
        except Exception as exc:  # noqa: BLE001 - shown to the person as is
            return _message(exc)
        return None

    def build_policy(self, spec: ProblemSpec):
        if self.method == "bax":
            return BAXPolicy(build_goal(self.normalized_goal_row(), spec),
                             strategy=self.bax_strategy)
        return default_policy(spec)

    def start_campaign(self) -> None:
        """Build the campaign from the rows and attach a controller. Raises with a
        message for the person if the setup isn't complete.

        With a workspace, the campaign is saved there (as the agent saves one), under
        a name made from :attr:`name`; a simulation stays in memory."""
        if self.workspace is not None and not self.mock:
            config = self.campaign_config()
            name = self.workspace.free_name(self.name)
            self.workspace.create(name, **config)
            self.saved_as = name
            self._attach(self.workspace.open(name))
            return
        spec = self.build_spec()
        policy = self.build_policy(spec)
        # With travel limits the campaign starts from the current settings, not
        # from space-filling seeds.
        start = self.extra.get("initial_designs") or build_start(spec, self.variable_rows)
        campaign = Campaign(
            spec, policy, build_constructor(spec, self.model), initial_designs=start,
            max_retries=self.extra.get("max_retries", 1),
        )
        if start is None:
            campaign.seed(default_initial_designs(spec))
        self._attach(campaign)

    def open_saved(self, name: str) -> None:
        """Resume a campaign saved in the workspace (by the UI or the agent): its
        setup fills the steps, and its runs carry on."""
        config = self.workspace.config(name)
        try:
            self.apply_config(config)
            campaign = self.workspace.open(name)
        except Exception:
            self.workspace.close(name)
            raise
        self.name = self.saved_as = name
        self.mock = False
        self.step = self.reached = len(STEPS) - 1
        self._attach(campaign)

    def _attach(self, campaign: Campaign) -> None:
        spec = campaign.spec
        self.spec = spec
        self.viz_obj = 0
        self.viz_vx = 0
        self.viz_vy = 1 if len(spec.domain) > 1 else 0
        self.focus = None
        self.finished = False
        if self.mock:
            exp = MockAnalyticExperiment(spec, seed=0, noise=0.0)
            self.controller = RunController(campaign, experiment=exp, evaluate=lambda o: o)
        else:
            self.controller = RunController(campaign)

    SETUP_FIELDS = ("name", "variable_rows", "objective_rows", "method", "goal_row",
                    "bax_strategy", "source_on", "source_row", "model", "mock", "extra")

    def new_campaign(self, keep_setup: bool = True) -> None:
        """Drop the running campaign and go back to the first step, nothing ticked.
        ``keep_setup`` keeps the answers as a starting point; otherwise they go
        back to the defaults."""
        if self.controller is not None:
            self.controller.stop()
        if self.saved_as is not None and self.workspace is not None:
            self.workspace.close(self.saved_as)  # the agent may open it now
        self.saved_as = None
        self.controller = None
        self.spec = None
        self.finished = False
        self.focus = None
        self.hold = {}
        self.n_suggest = 1
        self.step = self.reached = 0
        self._progress = self._plot_model = self._live = self._live_data = None
        if not keep_setup:
            fresh = UIState.__new__(UIState)
            UIState.__init__(fresh)
            for field in self.SETUP_FIELDS:
                setattr(self, field, getattr(fresh, field))

    # -- redrawing --

    def refresh(self) -> None:
        """Redraw the header and the body (the main column and the expert panel)."""
        for part in (self._header, self._main):
            if part is not None:
                part.refresh()

    def refresh_setup(self) -> None:
        """Redraw the header, the main column and the expert panel's tables, but
        not the JSON editor (the person is typing in it)."""
        for part in (self._header, self._column, *self._expert_parts):
            if part is not None:
                part.refresh()

    def refresh_expert(self) -> None:
        """Redraw the expert panel after a step's field changed. While setting up,
        the JSON editor is rebuilt only if the setup no longer matches it, so the
        steps redrawn after a JSON edit don't replace what the person is typing."""
        if self.controller is None and self._editor_config is not None:
            try:
                same = self.campaign_config() == self._editor_config
            except Exception:  # noqa: BLE001 - incomplete: the editor shows why
                same = False
            if same:
                for part in self._expert_parts:
                    part.refresh()
                return
        if self._expert is not None and self.expert:
            self._expert.refresh()

    def refresh_header(self) -> None:
        if self._header is not None:
            self._header.refresh()

    # -- running --

    @property
    def campaign(self) -> Campaign | None:
        return None if self.controller is None else self.controller.campaign

    def fitted_model(self):
        """The current fitted surrogate (canonical space) or None."""
        return None if self.controller is None else self.controller.campaign.current_model()

    def selections(self) -> figures.VizSelections:
        return figures.VizSelections(output=self.viz_obj, var_x=self.viz_vx, var_y=self.viz_vy)

    def history(self):
        return self.controller.campaign.history

    def policy(self):
        return self.controller.campaign.policy


def _goal_row(goal: dict) -> dict:
    """A goal from campaign.json as the goal step's row (bands per output)."""
    if not isinstance(goal, dict) or goal.get("kind") not in (
            "band", "level", "topk", "percentile"):
        raise ValueError("goal.kind must be band, level, topk or percentile.")
    row = dict(goal)
    if row["kind"] == "band":
        bounds = row.pop("bounds", None) or {}
        row["bands"] = {name: {"lo": lo, "hi": hi} for name, (lo, hi) in bounds.items()}
    return row


def _message(exc: Exception) -> str:
    """An exception as one readable message (pydantic's first error, plainly)."""
    from pydantic import ValidationError

    if isinstance(exc, ValidationError) and exc.errors():
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first.get("loc", ()))
        text = first.get("msg", str(exc)).removeprefix("Value error, ")
        return f"{where}: {text}" if where else text
    return str(exc)
