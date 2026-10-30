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
    objectives, output bounds as Less/GreaterThan/Bounds constraints).
  - ``CorePolicyGenerator`` — a gest ``Generator`` wrapping an abbot ``Campaign``,
    so a problem defined in abbot runs under ``Xopt(generator=...)`` on Xopt's
    (async/parallel) evaluator WITHOUT the core depending on Xopt. This is the
    "never the spine" proof: a user can always walk their ``ProblemSpec`` back to
    Xopt/Badger and the HPC-scale evaluation stack.
  - ``GestGeneratorPolicy`` — the other direction: any gest ``Generator`` (Xopt's
    Nelder-Mead, RCDS, extremum seeking, genetic algorithms, its BO generators
    with TuRBO) as an abbot policy, under abbot's loop, supervisors, store and UI.

Known limitation (by design): composite-SBI's cached ``aux`` posterior payload has
no DataFrame representation, so only the reduced-scalar objective form runs as a
plain Xopt generator — the composite-SBI case is exactly the one that doesn't fit
Xopt's tabular interchange, which is why it lives in abbot.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from abbot.problem import (
    ContinuousDomain,
    EnumeratedDomain,
    OutputSpec,
    ProblemSpec,
)
from abbot.session import Campaign

logger = logging.getLogger(__name__)

# --- ProblemSpec <-> gest VOCS ------------------------------------------------


def spec_to_vocs(spec: ProblemSpec):
    """abbot ``ProblemSpec`` -> gest ``VOCS`` (== ``xopt.VOCS``).

    An output's ``min_val`` / ``max_val`` become a VOCS constraint on it (an
    OBSERVE output with bounds is a constraint only, not also an observable).
    VOCS has no linear constraints, so a spec with ``domain_constraints`` or
    ``output_constraints`` is refused rather than silently loosened.
    """
    from gest_api.vocs import (
        VOCS,
        BoundsConstraint,
        ContinuousVariable,
        DiscreteVariable,
        GreaterThanConstraint,
        LessThanConstraint,
        MaximizeObjective,
        MinimizeObjective,
    )

    if spec.sources:
        raise NotImplementedError(
            f"spec_to_vocs: VOCS has no source variables {spec.source_names} (no "
            "target); use the JSON spec for this campaign."
        )
    if spec.domain_constraints or spec.output_constraints:
        raise NotImplementedError(
            "spec_to_vocs: VOCS has no linear constraints (domain_constraints, "
            "output_constraints); use the JSON spec for this campaign."
        )
    variables: dict[str, Any] = {}
    for d in spec.domain:
        if isinstance(d, ContinuousDomain):
            if d.log:
                logger.warning(
                    "spec_to_vocs: VOCS has no log scale; %r is exported on a linear one.",
                    d.name,
                )
            variables[d.name] = ContinuousVariable(domain=[d.lower, d.upper])
        elif isinstance(d, EnumeratedDomain):
            variables[d.name] = DiscreteVariable(values=list(d.values))
        else:
            raise NotImplementedError(
                f"spec_to_vocs does not support {type(d).__name__} ({d.name!r})."
            )
    # Directional outputs -> gest objectives; OBSERVE outputs -> gest observables.
    objectives = {
        o.name: (MinimizeObjective() if o.direction == "MINIMIZE" else MaximizeObjective())
        for o in spec.outputs
        if o.direction != "OBSERVE"
    }
    constraints = {}
    for o in spec.outputs:
        if o.min_val is not None and o.max_val is not None:
            constraints[o.name] = BoundsConstraint(range=[o.min_val, o.max_val])
        elif o.min_val is not None:
            constraints[o.name] = GreaterThanConstraint(value=o.min_val)
        elif o.max_val is not None:
            constraints[o.name] = LessThanConstraint(value=o.max_val)
    observables = [
        o.name for o in spec.outputs if o.direction == "OBSERVE" and not o.is_constrained
    ]
    return VOCS(
        variables=variables, objectives=objectives, constraints=constraints,
        observables=observables,
    )


def vocs_to_spec(vocs, boxes=None) -> ProblemSpec:
    """gest ``VOCS`` -> abbot ``ProblemSpec`` (continuous + discrete variables).

    ``boxes`` (optional) carries abbot target bands, which VOCS has no field for.
    A VOCS constraint becomes ``min_val`` / ``max_val`` on the output: on the
    objective of the same name if there is one, else on a new OBSERVE output.
    Note: VOCS stores objectives, constraints and observables separately, so the
    merged ``outputs`` are canonicalized in that order — a VOCS round-trip may
    reorder columns relative to an interleaved original.
    """
    from gest_api.vocs import (
        BoundsConstraint,
        ContinuousVariable,
        DiscreteVariable,
        GreaterThanConstraint,
        LessThanConstraint,
        MinimizeObjective,
    )

    domain = []
    for name, var in vocs.variables.items():
        if isinstance(var, ContinuousVariable):
            lo, hi = var.domain
            domain.append(ContinuousDomain(name=name, lower=float(lo), upper=float(hi)))
        elif isinstance(var, DiscreteVariable):
            # VOCS keeps a set, so sort for a stable order (labels' codes depend on it).
            labels = any(isinstance(v, str) for v in var.values)
            values = sorted(str(v) for v in var.values) if labels else sorted(float(v) for v in var.values)
            domain.append(EnumeratedDomain(name=name, values=values))
        else:
            raise NotImplementedError(
                f"vocs_to_spec does not support variable type {type(var).__name__} ({name!r})."
            )
    outputs = [
        OutputSpec(
            name=name,
            direction="MINIMIZE" if isinstance(obj, MinimizeObjective) else "MAXIMIZE",
        )
        for name, obj in vocs.objectives.items()
    ]
    position = {o.name: i for i, o in enumerate(outputs)}
    for name, con in (vocs.constraints or {}).items():
        if isinstance(con, BoundsConstraint):
            bounds = {"min_val": float(con.range[0]), "max_val": float(con.range[1])}
        elif isinstance(con, GreaterThanConstraint):
            bounds = {"min_val": float(con.value)}
        elif isinstance(con, LessThanConstraint):
            bounds = {"max_val": float(con.value)}
        else:
            raise NotImplementedError(
                f"vocs_to_spec does not support constraint type {type(con).__name__} ({name!r})."
            )
        if name in position:
            o = outputs[position[name]]
            outputs[position[name]] = OutputSpec(name=name, direction=o.direction, **bounds)
        else:
            outputs.append(OutputSpec(name=name, direction="OBSERVE", **bounds))
    known = {o.name for o in outputs}
    outputs += [
        OutputSpec(name=name, direction="OBSERVE")
        for name in (vocs.observables or {}) if name not in known
    ]
    return ProblemSpec(domain=domain, outputs=outputs, boxes=boxes)


# --- CorePolicyGenerator: run an abbot Policy under Xopt ----------------------


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
        out_names = self.spec.output_names  # ALL outputs (incl. OBSERVE), spec order
        for row in results:
            x = self.spec.as_tensor(row)  # labels -> codes
            y = torch.tensor([float(row[o]) for o in out_names], dtype=torch.double)
            # Xopt chose this candidate itself (no abbot suggest()), so record it.
            self._campaign.record(x, y)

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


# --- gest Generator -> abbot Policy -------------------------------------------


class GestGeneratorPolicy:
    """Any gest-api ``Generator`` as an abbot policy: Xopt's Nelder-Mead, RCDS,
    extremum seeking, genetic algorithms, or its Bayesian generators (TuRBO via
    their ``turbo_controller``), under abbot's loop, supervisors, store and UI.

    ``generator`` is a generator class, built on first use with
    ``vocs=GestGeneratorPolicy.vocs_for(spec)`` and ``**kwargs``, or a generator
    already built on that VOCS::

        from xopt.generators.sequential.neldermead import NelderMeadGenerator
        Campaign(spec, GestGeneratorPolicy(NelderMeadGenerator, adaptive=True))

    - **No abbot model.** The generator brings its own (or none), so the campaign
      skips its fit when proposing (``needs_model = False``). Recommendations and
      the UI still fit one on the results.
    - **Data.** Before each proposal the generator is given every clean record it
      hasn't seen, as ``{name: value}`` rows with every output by name. A resumed
      campaign hands a new generator the whole history.
    - **Sequential generators** (Nelder-Mead, RCDS, extremum seeking) propose one
      design at a time and expect that design's result next. They start from the
      last run they're given, so the runs they haven't seen are passed best last:
      they start from the best run so far (the default initial designs work as a
      start). If something else arrives while one waits on its design (a
      ``record`` of an outside run), it is reset and restarts from the best of the
      runs that arrived, and ``last_warning`` says so. Asking one
      for several designs, or for another while its last is still out, is
      refused.
    - **Labeled variables** reach the generator as their codes (Xopt's generators
      expect numbers), so a generator with a model treats them as ordered. Designs
      outside the generator carry the labels as usual.
    - **Not supported:** ``fixed=`` (a generator can't hold an input), source
      variables, and what VOCS can't hold (see :func:`spec_to_vocs`). Outstanding
      designs aren't passed on: the generator tracks what it proposed.
    """

    needs_model = False
    uq_requirement = "point"

    def __init__(self, generator, **kwargs) -> None:
        if isinstance(generator, type):
            self._factory, self.generator = generator, None
        else:
            if kwargs:
                raise TypeError("pass keyword arguments with a generator class, not an instance")
            self._factory, self.generator = None, generator
        self.kwargs = kwargs
        self._seen = 0  # history records already handed to the generator
        self.last_acquisition_value = None
        self.last_warning: str | None = None

    @staticmethod
    def vocs_for(spec: ProblemSpec):
        """The VOCS a generator is built on: :func:`spec_to_vocs`, with each
        labeled variable's values as codes."""
        from gest_api.vocs import DiscreteVariable

        vocs = spec_to_vocs(spec)
        for j in spec.categorical_indices():
            d = spec.domain[j]
            vocs.variables[d.name] = DiscreteVariable(values=d.codes)
        return vocs

    def _build(self, spec: ProblemSpec):
        if spec.sources:
            raise ValueError(
                f"GestGeneratorPolicy doesn't support source variables {spec.source_names}"
            )
        vocs = self.vocs_for(spec)
        if self.generator is None:
            self.generator = self._factory(vocs=vocs, **self.kwargs)
        else:
            names = set(self.generator.vocs.variable_names) | set(self.generator.vocs.output_names)
            want = set(vocs.variable_names) | set(vocs.output_names)
            if names != want:
                raise ValueError(
                    f"the generator's VOCS names {sorted(names)} don't match the spec's "
                    f"{sorted(want)}; build it with vocs=GestGeneratorPolicy.vocs_for(spec)"
                )
        return self.generator

    @property
    def _sequential(self) -> bool:
        return not getattr(self.generator, "supports_batch_generation", True)

    def _feed(self, history, spec: ProblemSpec) -> None:
        records = history.records[self._seen:]
        self._seen = len(history.records)
        rows = [  # codes, not labels: see vocs_for
            {
                **dict(zip(spec.variable_names, r.candidate.reshape(-1).tolist())),
                **dict(zip(spec.output_names, r.outputs.reshape(-1).tolist())),
            }
            for r in records
            if not r.fault and r.outputs is not None
        ]
        if not rows:
            return
        if self._sequential and not getattr(self.generator, "is_active", False):
            rows = self._best_last(rows, spec)
        try:
            self.generator.ingest(rows)
        except Exception as exc:
            if not hasattr(self.generator, "reset") or type(exc).__name__ != "SeqGeneratorError":
                raise
            # An outside result arrived while the generator waited on its own design.
            self.generator.reset()
            self.generator.ingest(self._best_last(rows, spec))
            self.last_warning = (
                f"{type(self.generator).__name__} was given results it didn't propose, so "
                "it was reset and restarts from the best of them."
            )

    @staticmethod
    def _best_last(rows: list[dict], spec: ProblemSpec) -> list[dict]:
        """``rows`` ordered so the best objective value is last (a sequential
        generator starts from the last row). Unchanged without one objective."""
        if spec.n_objectives != 1:
            return rows
        (j,) = spec.objective_indices()
        name, sign = spec.output_names[j], float(spec.output_signs()[j])
        return sorted(rows, key=lambda row: sign * row[name])

    def suggest(self, history, spec: ProblemSpec, model=None, n: int = 1, *, pending=None):
        self.last_warning = None
        generator = self._build(spec)
        name = type(generator).__name__
        if self._sequential:
            if n > 1:
                raise ValueError(f"{name} proposes one design at a time; ask for n=1")
            if pending is not None and len(pending):
                raise ValueError(
                    f"{name} proposes one design at a time and waits for its result; "
                    f"{len(pending)} design(s) are still outstanding"
                )
        self._feed(history, spec)
        try:
            points = generator.suggest(n)
        except Exception as exc:
            if type(exc).__name__ in ("SeqGeneratorError", "VOCSError"):
                raise ValueError(f"{name}: {exc}") from exc
            raise
        if len(points) != n:
            raise ValueError(f"{name} returned {len(points)} designs for n={n}")
        names = spec.variable_names
        return torch.tensor([[float(p[k]) for k in names] for p in points], dtype=torch.double)

    def ingest(self, results) -> None:
        return None  # results reach the generator through the history, in suggest
