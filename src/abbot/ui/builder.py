"""Framework-free helpers the UI uses to turn form rows into engine objects.

Kept out of the NiceGUI module so they are unit-testable without a browser: the
UI collects plain dict rows from its variable/objective editors and calls these.
"""

from __future__ import annotations

from abbot.acq_opt import Continuous, DiscreteArgmax
from abbot.problem import (
    ContinuousDomain,
    EnumeratedDomain,
    ObjectiveSpec,
    ProblemSpec,
)


def build_spec(variable_rows: list[dict], objective_rows: list[dict]) -> ProblemSpec:
    """Build a ``ProblemSpec`` from UI rows.

    ``variable_rows``: each ``{"kind": "continuous", "name", "lower", "upper"}`` or
    ``{"kind": "enumerated", "name", "values": [...]}``.
    ``objective_rows``: each ``{"name", "direction": "MINIMIZE"|"MAXIMIZE"}``.
    Pydantic validation raises with field-level messages the UI surfaces inline.
    """
    domain = []
    for row in variable_rows:
        kind = row.get("kind", "continuous")
        if kind == "continuous":
            domain.append(
                ContinuousDomain(
                    name=row["name"], lower=float(row["lower"]), upper=float(row["upper"])
                )
            )
        elif kind == "enumerated":
            values = row["values"]
            if isinstance(values, str):
                values = [float(v) for v in values.replace(",", " ").split()]
            domain.append(EnumeratedDomain(name=row["name"], values=[float(v) for v in values]))
        else:
            raise ValueError(f"unknown variable kind {kind!r}")
    objectives = [
        ObjectiveSpec(name=o["name"], direction=o.get("direction", "MAXIMIZE"))
        for o in objective_rows
    ]
    return ProblemSpec(domain=domain, objectives=objectives)


def choose_optimizer(spec: ProblemSpec):
    """Pick the CandidateOptimizer the spec supports; surface the mixed-space limit.

    - all-continuous  -> ``Continuous`` (``optimize_acqf``)
    - all-enumerated  -> ``DiscreteArgmax`` (``optimize_acqf_discrete``)
    - mixed           -> ``ValueError`` (optimize_acqf_mixed is deferred). The UI
      reports this rather than silently relaxing the enumerated axis.
    """
    kinds = {type(d).__name__ for d in spec.domain}
    all_cont = kinds == {"ContinuousDomain"}
    all_enum = kinds == {"EnumeratedDomain"}
    if all_cont:
        return Continuous()
    if all_enum:
        return DiscreteArgmax()
    raise ValueError(
        "Mixed continuous + enumerated design spaces are not yet supported for "
        "proposal (optimize_acqf_mixed is deferred). Use all-continuous or "
        "all-enumerated variables for now."
    )
