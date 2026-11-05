"""Framework-free helpers the UI uses to turn form rows into engine objects.

Kept out of the NiceGUI module so they are unit-testable without a browser: the
UI collects plain dict rows from its variable/objective editors and calls these.
"""

from __future__ import annotations

from abbot.goal import BandIntersection, Goal, LevelSet, PercentileSet, TopK
from abbot.model import ModelConstructor, default_constructor
from abbot.problem import (
    ContinuousDomain,
    EnumeratedDomain,
    OutputSpec,
    ProblemSpec,
    Source,
)

# BAX strategy labels: plain-language first, the real name kept in parentheses so
# an expert still recognizes it. Keys are the engine strategy identifiers.
BAX_STRATEGY_LABELS: dict[str, str] = {
    "SwitchBAX": "Balanced (SwitchBAX)",
    "MeanBAX": "Fast / greedy (MeanBAX)",
    "InfoBAX": "Thorough (InfoBAX)",
    "US": "Explore only (US)",
}

# Surrogate choices the builder offers. Keys go to ``build_constructor``.
MODEL_LABELS: dict[str, str] = {
    "default": "Independent GPs (default)",
    "correlated": "Correlated outputs (one GP)",
}


def build_spec(
    variable_rows: list[dict], output_rows: list[dict], source_row: dict | None = None
) -> ProblemSpec:
    """Build a ``ProblemSpec`` from UI rows.

    ``variable_rows``: each ``{"kind": "continuous", "name", "lower", "upper"}`` or
    ``{"kind": "enumerated", "name", "values": [...]}``. Enumerated values are
    numbers, or labels (categories) if any isn't a number; as text they are
    separated by commas, or by spaces when there are no commas. A continuous row
    may add ``log`` (model on a log scale) and ``max_step`` (a travel limit; blank
    for none).
    ``output_rows``: each ``{"name", "direction": "MINIMIZE"|"MAXIMIZE"|"OBSERVE"}``,
    optionally with ``min_val`` / ``max_val`` (keep the output in range; blank for
    no bound).
    ``source_row``: an optional source variable, ``{"kind", "name", "target"}`` plus
    ``values`` (enumerated) or ``lower`` / ``upper`` / ``log`` / ``accurate``
    (continuous). ``None`` or a blank name means no source.
    Pydantic validation raises with field-level messages the UI surfaces inline.
    """
    domain = [_domain(row) for row in variable_rows]
    outputs = [
        OutputSpec(
            name=o["name"],
            direction=o.get("direction", "MAXIMIZE"),
            min_val=_number(o.get("min_val")),
            max_val=_number(o.get("max_val")),
        )
        for o in output_rows
    ]
    sources = None
    if source_row and (source_row.get("name") or "").strip():
        sources = [_source(source_row)]
    return ProblemSpec(domain=domain, outputs=outputs, sources=sources)


def build_constructor(spec: ProblemSpec, model: str = "default") -> ModelConstructor | None:
    """The surrogate for a ``MODEL_LABELS`` key; ``None`` lets ``Campaign`` pick."""
    if model == "default":
        return None
    if model == "correlated":
        if spec.n_outputs < 2:
            raise ValueError("Correlated outputs need at least two outputs.")
        if spec.sources:
            raise ValueError(
                "Correlated outputs can't be combined with sources yet; use the "
                "default model."
            )
        return default_constructor(spec, correlated=True)
    raise ValueError(f"unknown model choice {model!r}")


def build_start(spec: ProblemSpec, variable_rows: list[dict]) -> list[dict] | None:
    """Where the hardware starts, as ``initial_designs``, when travel limits are set.

    A travel limit is measured from the current setting, so each limited variable
    needs its ``start``. Variables without a limit start mid-range (enumerated: the
    first value). Without travel limits this returns ``None`` (space-filling seeds).
    """
    if not spec.has_travel_limits:
        return None
    start: dict = {}
    for row, d in zip(variable_rows, spec.domain):
        value = _number(row.get("start"))
        if value is None and getattr(d, "max_step", None) is not None:
            raise ValueError(
                f"Variable {d.name!r} has a travel limit: enter where it starts "
                "(its current setting)."
            )
        if value is None:
            if isinstance(d, ContinuousDomain):
                value = (d.lower + d.upper) / 2
            else:
                value = d.values[0]
        start[d.name] = value
    for s in spec.sources or []:
        start[s.name] = s.target
    return [start]


def spec_to_rows(spec: ProblemSpec) -> tuple[list[dict], list[dict], dict | None]:
    """The editor rows for a ``spec``: the inverse of :func:`build_spec`.

    Raises ``ValueError`` naming what the editor can't show (integer or structured
    variables, more than one source, linear constraints), so an import refuses
    rather than silently dropping part of the problem.
    """
    unsupported = [
        d.name for d in spec.domain if not isinstance(d, (ContinuousDomain, EnumeratedDomain))
    ]
    sources = spec.sources or []
    if len(sources) > 1:
        unsupported.append(f"{len(sources)} sources")
    if sources and not isinstance(sources[0].domain, (ContinuousDomain, EnumeratedDomain)):
        unsupported.append(sources[0].name)
    if spec.domain_constraints or spec.output_constraints:
        unsupported.append("linear constraints")
    if unsupported:
        raise ValueError(
            f"The editor can't show {unsupported} yet; run this spec through the "
            "Python API."
        )
    variable_rows = []
    for d in spec.domain:
        if isinstance(d, ContinuousDomain):
            variable_rows.append({
                "kind": "continuous", "name": d.name, "lower": d.lower, "upper": d.upper,
                "log": d.log, "max_step": d.max_step, "start": None,
            })
        else:
            variable_rows.append({
                "kind": "enumerated", "name": d.name,
                "values": ", ".join(str(v) for v in d.values),
            })
    output_rows = [
        {"name": o.name, "direction": o.direction, "min_val": o.min_val, "max_val": o.max_val}
        for o in spec.outputs
    ]
    source_row = None
    if sources:
        src = sources[0]
        d = src.domain
        source_row = {"name": d.name, "target": str(src.target), "log": src.is_log,
                      "accurate": src.accurate, "lower": 0.0, "upper": 1.0,
                      "values": "simulation, experiment"}
        if isinstance(d, EnumeratedDomain):
            source_row.update(kind="enumerated", values=", ".join(str(v) for v in d.values))
        else:
            source_row.update(kind="continuous", lower=d.lower, upper=d.upper)
    return variable_rows, output_rows, source_row


def _number(value) -> float | None:
    """A form field as a float, or ``None`` when left blank."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return float(value)


def parse_values(values) -> list:
    """Enumerated values from a text field or a list: numbers, else labels."""
    if isinstance(values, str):
        sep = "," if "," in values else None
        tokens = [t.strip() for t in values.split(sep) if t.strip()]
    else:
        tokens = list(values or [])
    try:
        return [float(tok) for tok in tokens]
    except (TypeError, ValueError):
        return [str(tok) for tok in tokens]  # labels: categories


def _domain(row: dict, what: str = "Variable"):
    kind = row.get("kind", "continuous")
    name = row.get("name") or "(unnamed)"
    if kind == "continuous":
        lower, upper = row.get("lower"), row.get("upper")
        if lower is None or upper is None:
            raise ValueError(
                f"{what} {name!r}: enter both a low and a high bound "
                "(the lo/hi fields can't be blank)."
            )
        return ContinuousDomain(
            name=row["name"],
            lower=float(lower),
            upper=float(upper),
            log=bool(row.get("log", False)),
            max_step=_number(row.get("max_step")),
        )
    if kind == "enumerated":
        values = parse_values(row.get("values"))
        if not values:
            raise ValueError(f"{what} {name!r}: list at least one allowed value.")
        return EnumeratedDomain(name=row["name"], values=values)
    raise ValueError(f"unknown variable kind {kind!r}")


def _source(row: dict) -> Source:
    name = row["name"].strip()
    target = row.get("target")
    if target is None or (isinstance(target, str) and not target.strip()):
        raise ValueError(f"Source {name!r}: enter the target (the source you trust).")
    if row.get("kind", "enumerated") == "enumerated":
        domain = _domain({**row, "name": name}, "Source")
        target = str(target).strip()
        if domain.values and isinstance(domain.values[0], float):
            target = float(target)
        return Source(domain=domain, target=target)
    domain = _domain({"kind": "continuous", "name": name, "lower": row.get("lower"),
                      "upper": row.get("upper")}, "Source")
    return Source(
        domain=domain,
        target=float(target),
        log=bool(row.get("log", False)),
        accurate=row.get("accurate", "high"),
    )


def build_goal(goal_row: dict, spec: ProblemSpec) -> Goal:
    """Build a BAX ``Goal`` from a UI goal row against a spec's outputs.

    ``goal_row`` keys by ``kind``:
      - ``"band"``   : ``bounds`` = ``{output_name: (lo, hi)}`` for the outputs the
        target region constrains (an AND box across those outputs — ``BandIntersection``
        padded to full width with unconstrained outputs).
      - ``"percentile"`` : ``percentile`` (top ``100-p``%% of a single output).
      - ``"level"``  : ``threshold``, ``output`` (name), ``upper`` (sub-level if True).
      - ``"topk"``   : ``k``, ``output`` (name), ``largest``.

    ``output`` names are resolved to column indices in ``spec.outputs`` order so the
    goal lines up with the model's output columns (incl. OBSERVE).
    """
    kind = goal_row.get("kind", "band")
    names = spec.output_names

    def _col(name) -> int:
        if name in names:
            return names.index(name)
        raise ValueError(f"goal references unknown output {name!r}; outputs are {names}.")

    if kind == "percentile":
        if spec.n_outputs != 1:
            raise ValueError(
                "Top-percentile goal is single-output; this spec has "
                f"{spec.n_outputs} outputs — use a target band or level set."
            )
        return PercentileSet(float(goal_row.get("percentile", 80.0)))

    if kind == "level":
        threshold = goal_row.get("threshold")
        if threshold is None:
            raise ValueError("Level-set goal: enter a threshold value.")
        return LevelSet(
            threshold=float(threshold),
            output=_col(goal_row.get("output", names[0])),
            upper=bool(goal_row.get("upper", False)),
        )

    if kind == "topk":
        return TopK(
            k=int(goal_row.get("k", 1)),
            output=_col(goal_row.get("output", names[0])),
            largest=bool(goal_row.get("largest", True)),
        )

    if kind == "band":
        # A per-output [lo, hi] box, padded to full width (unconstrained outputs span
        # (-inf, inf)). At least one output must carry a finite band.
        bounds_by_name = goal_row.get("bounds") or {}
        if not bounds_by_name:
            raise ValueError("Target-band goal needs a [lo, hi] on at least one output.")
        full: list[tuple[float, float]] = []
        for o in spec.outputs:
            if o.name in bounds_by_name:
                lo, hi = bounds_by_name[o.name]
                lo = float("-inf") if lo is None else float(lo)
                hi = float("inf") if hi is None else float(hi)
                if hi <= lo:
                    raise ValueError(
                        f"output {o.name!r}: band upper ({hi}) must exceed lower ({lo})."
                    )
                full.append((lo, hi))
            else:
                full.append((float("-inf"), float("inf")))
        return BandIntersection(full)

    raise ValueError(f"unknown goal kind {kind!r}")
