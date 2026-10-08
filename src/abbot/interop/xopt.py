"""Xopt / gest-api interop (F3).

Targets the gest-api **Generator Standard**, not Xopt internals. gest-api is
pydantic-only; and crucially ``xopt.VOCS is gest_api.vocs.VOCS`` (same class), so
an abbot ``ProblemSpec`` mapped to a gest ``VOCS`` is exactly what ``Xopt``
consumes — one adapter, no second bridge. The core never imports Xopt; only this
module (lazy) touches gest-api, and the ``[xopt]`` extra is only needed to run
under Xopt's orchestrator.

Surface:
  - ``spec_to_vocs`` / ``vocs_to_spec`` — field map between abbot ``ProblemSpec``
    and gest ``VOCS`` (ContinuousVariable/DiscreteVariable, Minimize/Maximize
    objectives, Less/GreaterThan constraints).
  - ``CorePolicyGenerator`` — a gest ``Generator`` wrapping an abbot ``Campaign``,
    so a problem defined in abbot runs under ``Xopt(generator=...)`` on Xopt's
    (async/parallel) evaluator WITHOUT the core depending on Xopt. This is the
    "never the spine" proof: a user can always walk their ``ProblemSpec`` back to
    Xopt/Badger and the HPC-scale evaluation stack.

Known limitation (by design): composite-SBI's cached ``aux`` posterior payload has
no DataFrame representation, so only the reduced-scalar objective form runs as a
plain Xopt generator — the composite-SBI case is exactly the one that doesn't fit
Xopt's tabular interchange, which is why it lives in abbot.
"""

from __future__ import annotations

from typing import Any

import torch

from abbot.model import default_constructor
from abbot.policy import ParetoBO, SimpleBO
from abbot.problem import (
    ContinuousDomain,
    EnumeratedDomain,
    ObjectiveSpec,
    ProblemSpec,
)
from abbot.session import Campaign

# --- ProblemSpec <-> gest VOCS ------------------------------------------------


def spec_to_vocs(spec: ProblemSpec):
    """abbot ``ProblemSpec`` -> gest ``VOCS`` (== ``xopt.VOCS``)."""
    from gest_api.vocs import (
        VOCS,
        ContinuousVariable,
        DiscreteVariable,
        MaximizeObjective,
        MinimizeObjective,
    )

    variables: dict[str, Any] = {}
    for d in spec.domain:
        if isinstance(d, ContinuousDomain):
            variables[d.name] = ContinuousVariable(domain=[d.lower, d.upper])
        elif isinstance(d, EnumeratedDomain):
            variables[d.name] = DiscreteVariable(values=list(d.values))
        else:
            raise NotImplementedError(
                f"spec_to_vocs does not support {type(d).__name__} ({d.name!r})."
            )
    objectives = {
        o.name: (MinimizeObjective() if o.direction == "MINIMIZE" else MaximizeObjective())
        for o in spec.objectives
    }
    return VOCS(variables=variables, objectives=objectives)


def vocs_to_spec(vocs, boxes=None) -> ProblemSpec:
    """gest ``VOCS`` -> abbot ``ProblemSpec`` (continuous + discrete variables).

    ``boxes`` (optional) carries abbot target bands, which VOCS has no field for.
    """
    from gest_api.vocs import ContinuousVariable, DiscreteVariable, MinimizeObjective

    domain = []
    for name, var in vocs.variables.items():
        if isinstance(var, ContinuousVariable):
            lo, hi = var.domain
            domain.append(ContinuousDomain(name=name, lower=float(lo), upper=float(hi)))
        elif isinstance(var, DiscreteVariable):
            domain.append(EnumeratedDomain(name=name, values=[float(v) for v in var.values]))
        else:
            raise NotImplementedError(
                f"vocs_to_spec does not support variable type {type(var).__name__} ({name!r})."
            )
    objectives = [
        ObjectiveSpec(
            name=name,
            direction="MINIMIZE" if isinstance(obj, MinimizeObjective) else "MAXIMIZE",
        )
        for name, obj in vocs.objectives.items()
    ]
    return ProblemSpec(domain=domain, objectives=objectives, boxes=boxes)


# --- CorePolicyGenerator: run an abbot Policy under Xopt ----------------------


def _default_policy(spec: ProblemSpec):
    return SimpleBO() if spec.n_objectives == 1 else ParetoBO()


class CorePolicyGenerator:
    """A gest-api ``Generator`` backed by an abbot ``Campaign``.

    Pass an instance to ``Xopt(generator=..., evaluator=...)``: Xopt's loop calls
    ``suggest(n)`` for candidates and feeds results back via ``add_data`` (a
    DataFrame) or ``ingest`` (list[dict]) — both routed into the Campaign. The core
    stays Xopt-free; this adapter is the only Xopt-facing piece.

    Subclasses gest's ``Generator`` dynamically (import is lazy) so abbot's core
    never depends on gest-api at import time.
    """

    def __new__(cls, *args, **kwargs):
        # Build a concrete subclass of gest Generator the first time, so the ABC's
        # abstract methods are satisfied by our implementations below.
        from gest_api import Generator as GestGenerator

        if GestGenerator not in cls.__bases__:
            impl = type(cls.__name__, (cls, GestGenerator), {})
            return super().__new__(impl)
        return super().__new__(cls)

    def __init__(
        self,
        spec: ProblemSpec,
        policy=None,
        model_constructor=None,
        *,
        vocs=None,
    ) -> None:
        self.spec = spec
        policy = policy or _default_policy(spec)
        model_constructor = model_constructor or default_constructor(
            spec, bounds=spec.bounds_tensor()
        )
        self._campaign = Campaign(spec, policy, model_constructor)
        # gest Generator.__init__ wants a VOCS.
        self.vocs = vocs if vocs is not None else spec_to_vocs(spec)
        try:
            super().__init__(vocs=self.vocs)
        except TypeError:
            pass  # some gest versions take no positional vocs on the mixed class

    def _validate_vocs(self, vocs):  # required abstract hook; our spec is the source of truth
        return vocs

    # -- gest Generator contract --

    def suggest(self, num_points: int | None = 1) -> list[dict]:
        return self._campaign.suggest(num_points or 1)

    def ingest(self, results: list[dict]) -> None:
        """Fold Xopt/gest results (list[dict] with var + objective keys) into the
        Campaign. Reconstructs each candidate from variable columns and the output
        vector from objective columns (spec order), then calls Campaign.ingest.
        """
        if not results:
            return
        var_names = self.spec.variable_names
        obj_names = [o.name for o in self.spec.objectives]
        for row in results:
            x = torch.tensor([float(row[v]) for v in var_names], dtype=torch.double)
            y = torch.tensor([float(row[o]) for o in obj_names], dtype=torch.double)
            # Align the Campaign's pending slot with this candidate, then ingest.
            self._campaign._pending = [(x, None, 0)]
            self._campaign.ingest([y])

    def add_data(self, new_data) -> None:
        """Xopt's result-feedback path: a pandas DataFrame of evaluated points.

        Converts rows to the gest list[dict] shape and defers to ``ingest``.
        """
        self.ingest(new_data.to_dict("records"))

    def finalize(self) -> None:
        return None

    @property
    def campaign(self) -> Campaign:
        return self._campaign
