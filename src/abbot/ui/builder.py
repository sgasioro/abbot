"""Framework-free helpers the UI uses to turn form rows into engine objects.

Kept out of the NiceGUI module so they are unit-testable without a browser: the
UI collects plain dict rows from its variable/objective editors and calls these.
"""

from __future__ import annotations

from abbot.goal import BandIntersection, Goal, LevelSet, PercentileSet, TopK
from abbot.problem import (
    ContinuousDomain,
    EnumeratedDomain,
    OutputSpec,
    ProblemSpec,
)

# BAX strategy labels: plain-language first, the real name kept in parentheses so
# an expert still recognizes it. Keys are the engine strategy identifiers.
BAX_STRATEGY_LABELS: dict[str, str] = {
    "SwitchBAX": "Balanced (SwitchBAX)",
    "MeanBAX": "Fast / greedy (MeanBAX)",
    "InfoBAX": "Thorough (InfoBAX)",
    "US": "Explore only (US)",
}


def build_spec(variable_rows: list[dict], output_rows: list[dict]) -> ProblemSpec:
    """Build a ``ProblemSpec`` from UI rows.

    ``variable_rows``: each ``{"kind": "continuous", "name", "lower", "upper"}`` or
    ``{"kind": "enumerated", "name", "values": [...]}``. Enumerated values are
    numbers, or labels (categories) if any isn't a number; as text they are
    separated by commas, or by spaces when there are no commas.
    ``output_rows``: each ``{"name", "direction": "MINIMIZE"|"MAXIMIZE"|"OBSERVE"}``.
    Pydantic validation raises with field-level messages the UI surfaces inline.
    """
    domain = []
    for row in variable_rows:
        kind = row.get("kind", "continuous")
        name = row.get("name") or "(unnamed)"
        if kind == "continuous":
            lower, upper = row.get("lower"), row.get("upper")
            if lower is None or upper is None:
                raise ValueError(
                    f"Variable {name!r}: enter both a low and a high bound "
                    "(the lo/hi fields can't be blank)."
                )
            domain.append(ContinuousDomain(name=row["name"], lower=float(lower), upper=float(upper)))
        elif kind == "enumerated":
            values = row.get("values")
            if isinstance(values, str):
                sep = "," if "," in values else None
                tokens = [t.strip() for t in values.split(sep) if t.strip()]
            else:
                tokens = list(values or [])
            if not tokens:
                raise ValueError(f"Variable {name!r}: list at least one allowed value.")
            try:
                parsed = [float(tok) for tok in tokens]
            except (TypeError, ValueError):
                parsed = [str(tok) for tok in tokens]  # labels: categories
            domain.append(EnumeratedDomain(name=row["name"], values=parsed))
        else:
            raise ValueError(f"unknown variable kind {kind!r}")
    outputs = [
        OutputSpec(name=o["name"], direction=o.get("direction", "MAXIMIZE"))
        for o in output_rows
    ]
    return ProblemSpec(domain=domain, outputs=outputs)


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
