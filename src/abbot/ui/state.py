"""Per-client UI state: the setup rows, where the person is in the steps, and the
live controller once the campaign starts.

Framework-free apart from holding NiceGUI refreshables, so the setup logic (the
spec preview, the campaign config, starting a campaign) is testable without a page.
"""

from __future__ import annotations

from abbot.experiment import MockAnalyticExperiment
from abbot.policy import BAXPolicy, default_policy
from abbot.problem import ProblemSpec
from abbot.session import Campaign, default_initial_designs
from abbot.ui import figures
from abbot.ui.builder import build_constructor, build_goal, build_spec, build_start
from abbot.ui.controller import RunController

STEPS = ["Knobs", "Measurements", "Goal", "Extras", "Start"]


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
        self.step: int = 0  # index into STEPS while setting up
        self.reached: int = 0  # the furthest step reached (earlier ones are done)
        self.expert: bool = False  # the expert panel is open
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
        self._expert = None
        self._live = None  # updates the expert panel's live parts in place
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
        return build_spec(self.variable_rows, self.output_rows(), source)

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
        config: dict = {"spec": spec.model_dump(mode="json", exclude_defaults=True)}
        goal = self.goal_json()
        if goal is not None:
            config["goal"] = goal
            config["strategy"] = self.bax_strategy
        config["model"] = self.model
        start = build_start(spec, self.variable_rows)
        if start is not None:
            config["initial_designs"] = start
        return config

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
        message for the person if the setup isn't complete."""
        spec = self.build_spec()
        policy = self.build_policy(spec)
        # With travel limits the campaign starts from the current settings, not
        # from space-filling seeds.
        start = build_start(spec, self.variable_rows)
        campaign = Campaign(
            spec, policy, build_constructor(spec, self.model), initial_designs=start
        )
        if start is None:
            campaign.seed(default_initial_designs(spec))
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
                    "bax_strategy", "source_on", "source_row", "model", "mock")

    def new_campaign(self, keep_setup: bool = True) -> None:
        """Drop the running campaign and go back to the first step, nothing ticked.
        ``keep_setup`` keeps the answers as a starting point; otherwise they go
        back to the defaults."""
        if self.controller is not None:
            self.controller.stop()
        self.controller = None
        self.spec = None
        self.finished = False
        self.focus = None
        self.hold = {}
        self.n_suggest = 1
        self.step = self.reached = 0
        self._progress = self._plot_model = self._live = None
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

    def refresh_expert(self) -> None:
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


def _message(exc: Exception) -> str:
    """An exception as one readable message (pydantic's first error, plainly)."""
    from pydantic import ValidationError

    if isinstance(exc, ValidationError) and exc.errors():
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first.get("loc", ()))
        text = first.get("msg", str(exc)).removeprefix("Value error, ")
        return f"{where}: {text}" if where else text
    return str(exc)
