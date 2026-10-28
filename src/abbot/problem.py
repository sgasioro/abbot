"""Serializable problem specification.

The ``ProblemSpec`` is the typed, JSON-serializable contract shared by the loop,
the (future) web frontend, and any agent driver. It is pydantic so that
``model_dump_json`` + JSON-schema export give browser forms for free, and so the
``vocs <-> spec`` interop adapter is a field map against gest-api's typed VOCS
rather than a format bridge.

Design spaces are **type-polymorphic from day one** (Continuous | Integer |
Enumerated | Structured) — the exact axis Badger is rigid on, and hence a differentiator.
Each Domain maps onto a gest-api variable type for interop (see interop/xopt.py).

NB: this is the frontend/interop-facing spec. The in-memory numeric bounds that
botorch's ``optimize_acqf`` consumes are derived from it (see ``bounds_tensor``);
heavy / live objects (a trained model, an HTTP reactor Experiment) are referenced
by registry key + config, never embedded here.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    import torch

# --- Domains -----------------------------------------------------------------
# A Domain describes ONE design variable. The surrogate always sees a (..., d)
# float tensor; truly-discrete *design spaces* are handled at the candidate-
# proposal layer (acq_opt), not the surrogate. Enumerated/categorical variables
# are featurized to floats for the model but enumerated for proposal.


def _check_log(kind: str, name: str, log: bool, lower: float) -> None:
    if log and lower <= 0:
        raise ValueError(f"{kind} {name!r}: log=True needs a positive lower bound, got {lower}")


class ContinuousDomain(BaseModel):
    """A real-valued variable from ``lower`` to ``upper``.

    ``log=True`` for quantities that span orders of magnitude (concentrations,
    rates, sizes): the model sees the variable on a log scale, and initial designs
    and searches spread over its decades rather than crowding the top one.
    """

    kind: Literal["continuous"] = "continuous"
    name: str
    lower: float
    upper: float
    log: bool = False

    @model_validator(mode="after")
    def _check_bounds(self) -> ContinuousDomain:
        if self.upper <= self.lower:
            raise ValueError(
                f"ContinuousDomain {self.name!r}: upper ({self.upper}) must exceed "
                f"lower ({self.lower})"
            )
        _check_log("ContinuousDomain", self.name, self.log, self.lower)
        return self


class IntegerDomain(BaseModel):
    """Whole numbers from ``lower`` to ``upper`` inclusive (atom counts, replicates,
    grid sizes). The surrogate treats it as continuous; proposals are rounded.
    ``log=True`` as for :class:`ContinuousDomain`.
    """

    kind: Literal["integer"] = "integer"
    name: str
    lower: int
    upper: int
    log: bool = False

    @model_validator(mode="after")
    def _check_bounds(self) -> IntegerDomain:
        if self.upper <= self.lower:
            raise ValueError(
                f"IntegerDomain {self.name!r}: upper ({self.upper}) must exceed "
                f"lower ({self.lower})"
            )
        _check_log("IntegerDomain", self.name, self.log, self.lower)
        return self


class EnumeratedDomain(BaseModel):
    """A finite set of allowed values: numbers (an ordered grid of settings) or
    labels (unordered categories: solvents, catalysts, codes).

    Numbers are modeled as points on a line, so nearby values are expected to
    behave alike. Labels are categories: the model learns how alike each pair is
    rather than assuming an order. Designs carry the labels (``{"solvent":
    "water"}``); inside the loop each label is a code, its position in ``values``
    (see :meth:`encode`). Proposals enumerate the values either way.
    """

    kind: Literal["enumerated"] = "enumerated"
    name: str
    values: list[float] | list[str]

    @model_validator(mode="after")
    def _check_values(self) -> EnumeratedDomain:
        if len(self.values) < 1:
            raise ValueError(f"EnumeratedDomain {self.name!r} needs at least one value")
        if len(set(self.values)) != len(self.values):
            raise ValueError(f"EnumeratedDomain {self.name!r} has repeated values: {self.values}")
        return self

    @property
    def is_categorical(self) -> bool:
        """Whether the values are labels (unordered categories)."""
        return isinstance(self.values[0], str)

    @property
    def codes(self) -> list[float]:
        """The values as the loop's tensors hold them: the numbers themselves, or
        each label's position in ``values``."""
        if self.is_categorical:
            return [float(i) for i in range(len(self.values))]
        return [float(v) for v in self.values]

    def encode(self, value: float | str) -> float:
        """A value (a label or a number) as its code. Raises if not allowed."""
        if self.is_categorical:
            if isinstance(value, str) and value in self.values:
                return float(self.values.index(value))
        elif not isinstance(value, str) and float(value) in self.values:
            return float(value)
        raise ValueError(f"{self.name}={value!r} is not one of {self.values}")

    def decode(self, code: float) -> float | str:
        """A code as the value a design carries: its label, or the number."""
        if self.is_categorical:
            return self.values[round(code)]
        return float(code)


class StructuredDomain(BaseModel):
    """Placeholder for constrained composite spaces (e.g. mixture / sum-to-one).

    Day-one scope is continuous + enumerated (covers all three source repos as
    they run today). True combinatorial search is an extension point, not solved
    speculatively (see plan open-question #6).
    """

    kind: Literal["structured"] = "structured"
    name: str
    dim: int
    constraint: Literal["simplex"] = "simplex"


Domain = Annotated[
    ContinuousDomain | IntegerDomain | EnumeratedDomain | StructuredDomain,
    Field(discriminator="kind"),
]

SourceDomain = Annotated[
    ContinuousDomain | IntegerDomain | EnumeratedDomain,
    Field(discriminator="kind"),
]


class Source(BaseModel):
    """A source variable: an input the experiment needs that isn't itself a design
    choice to recommend. Fidelity knobs (system size, mesh, convergence settings)
    and information sources (two codes, simulation vs experiment) are both sources.

    The experiment receives it like any variable, and a source-aware policy chooses
    it, trading cost against information. Recommendations, goals and policies that
    don't know about sources all work *at the target*: the source you trust
    (the fidelity you're happy with, the real experiment).

    ``ordered`` says whether nearby values are related (fidelity levels) or not
    (different codes); it defaults to True for continuous and integer domains and
    False for enumerated ones. An unordered source must be enumerated, with
    labels (``["dft", "experiment"]``) or whole numbers. ``log=True`` asks
    the model to see the source on a log scale (size-like knobs). ``accurate`` says
    which end of an ordered source is the more accurate one: ``"high"`` for knobs
    like system size or iteration count, ``"low"`` for knobs like a tolerance or a
    mesh spacing. ``target`` is a recommendation-time setting: it may change between
    runs without invalidating a resumed store.
    """

    domain: SourceDomain
    target: float | str
    ordered: bool | None = None
    log: bool = False
    accurate: Literal["high", "low"] = "high"

    @property
    def name(self) -> str:
        return self.domain.name

    @property
    def is_log(self) -> bool:
        """``log=True`` on the source or on its domain."""
        return self.log or bool(getattr(self.domain, "log", False))

    @property
    def is_ordered(self) -> bool:
        if self.ordered is not None:
            return self.ordered
        return not isinstance(self.domain, EnumeratedDomain)

    @property
    def target_code(self) -> float:
        """The target as the loop's tensors hold it (see ``EnumeratedDomain.codes``)."""
        d = self.domain
        return d.encode(self.target) if isinstance(d, EnumeratedDomain) else float(self.target)

    @model_validator(mode="after")
    def _check(self) -> Source:
        d, t = self.domain, self.target
        if isinstance(d, EnumeratedDomain):
            try:
                d.encode(t)
            except ValueError:
                raise ValueError(f"Source {d.name!r}: target {t!r} is not one of {d.values}") from None
            if d.is_categorical and self.ordered:
                raise ValueError(
                    f"Source {d.name!r}: labels are unordered; use numbers for ordered levels"
                )
        elif isinstance(t, str):
            # pydantic reports a ValueError as a validation error; a TypeError escapes.
            raise ValueError(f"Source {d.name!r}: target must be a number, got {t!r}")  # noqa: TRY004
        elif not d.lower <= t <= d.upper:
            raise ValueError(
                f"Source {d.name!r}: target {t} is outside [{d.lower}, {d.upper}]"
            )
        if isinstance(d, IntegerDomain) and t != int(t):
            raise ValueError(f"Source {d.name!r}: target {t} must be a whole number")
        if self.is_log:
            low = min(d.codes) if isinstance(d, EnumeratedDomain) else d.lower
            if low <= 0:
                raise ValueError(f"Source {d.name!r}: log=True needs positive values")
        if not self.is_ordered:
            if not isinstance(d, EnumeratedDomain):
                raise ValueError(
                    f"Source {d.name!r}: an unordered source must be enumerated "
                    "(one value per code or source)"
                )
            if any(v != int(v) for v in d.codes):
                raise ValueError(
                    f"Source {d.name!r}: label unordered sources with names or whole "
                    f"numbers, got {d.values}"
                )
            if self.accurate != "high" or self.is_log:
                raise ValueError(
                    f"Source {d.name!r}: accurate= and log= apply to ordered sources only"
                )
        return self


# --- Objectives / constraints ------------------------------------------------


Direction = Literal["MINIMIZE", "MAXIMIZE", "OBSERVE"]


class OutputSpec(BaseModel):
    """A named MODELED output of an experiment (one surrogate column).

    ``direction``:
      - ``MAXIMIZE`` / ``MINIMIZE`` — an optimized *objective* (BO/MO).
      - ``OBSERVE``  — modeled but NOT optimized: a feature a BAX ``Goal``
        consumes, or an output that only has to stay within bounds. OBSERVE outputs
        are fit by the surrogate like any other column; they are simply excluded
        from the optimized-objective subset (``ProblemSpec.objectives``).

    ``min_val`` / ``max_val`` make the output a constraint, in its raw units: a
    design is only acceptable if the output lands within them (``purity`` at least
    0.9). Policies steer towards designs likely to satisfy every constraint, and the
    recommendation only picks designs likely to. For a bound on a combination of
    outputs, use ``ProblemSpec.output_constraints``.
    """

    name: str
    direction: Direction = "MAXIMIZE"
    min_val: float | None = None
    max_val: float | None = None

    @model_validator(mode="after")
    def _check_bounds(self) -> OutputSpec:
        _check_range(f"OutputSpec {self.name!r}", self.min_val, self.max_val)
        return self

    @property
    def is_constrained(self) -> bool:
        return self.min_val is not None or self.max_val is not None


# Back-compat: an "objective" is just a non-OBSERVE output. Existing
# ``ObjectiveSpec(name=..., direction="MAXIMIZE"|"MINIMIZE")`` call sites keep working.
ObjectiveSpec = OutputSpec


class Box(BaseModel):
    """A target band on the outputs: a design is in the box when every banded
    output lies within its ``(low, high)``.

    Name the outputs::

        Box({"yield": (0.4, 0.6), "purity": (0.9, 1.0)})

    Outputs left out are unconstrained. The older positional form,
    ``Box(lower=[...], upper=[...])`` with one entry per output in ``outputs``
    order, still works. A list of boxes drives a goal (see ``goals_from_spec``):
    a design in any box is in the target set.
    """

    bands: dict[str, tuple[float, float]] | None = None
    lower: list[float] | None = None
    upper: list[float] | None = None

    def __init__(self, bands: dict[str, tuple[float, float]] | None = None, /, **data) -> None:
        if bands is not None:
            data["bands"] = bands
        super().__init__(**data)

    @model_validator(mode="after")
    def _check(self) -> Box:
        if self.bands is not None:
            if self.lower is not None or self.upper is not None:
                raise ValueError("Box takes named bands or lower/upper, not both")
            if not self.bands:
                raise ValueError("Box needs at least one band")
            for name, (lo, hi) in self.bands.items():
                if hi <= lo:
                    raise ValueError(f"Box band {name!r}: high ({hi}) must exceed low ({lo})")
            return self
        if self.lower is None or self.upper is None:
            raise ValueError("Box needs named bands, e.g. Box({'yield': (0.4, 0.6)})")
        if len(self.lower) != len(self.upper):
            raise ValueError("Box lower/upper must have equal length")
        for j, (lo, hi) in enumerate(zip(self.lower, self.upper)):
            if hi <= lo:
                raise ValueError(f"Box dim {j}: upper ({hi}) must exceed lower ({lo})")
        return self

    def band_dict(self, output_names: list[str]) -> dict[str, tuple[float, float]]:
        """The bands by output name (the positional form mapped onto
        ``output_names``)."""
        if self.bands is not None:
            return {k: (float(lo), float(hi)) for k, (lo, hi) in self.bands.items()}
        if len(self.lower) != len(output_names):
            raise ValueError(
                f"Box has {len(self.lower)} bands but the spec has {len(output_names)} "
                f"outputs {output_names}; name them instead, e.g. "
                f"Box({{{output_names[0]!r}: (low, high)}})"
            )
        return {
            name: (float(lo), float(hi))
            for name, lo, hi in zip(output_names, self.lower, self.upper)
        }


def _check_range(what: str, lo: float | None, hi: float | None) -> None:
    if lo is not None and hi is not None and hi <= lo:
        raise ValueError(f"{what}: max_val ({hi}) must exceed min_val ({lo})")


class LinearConstraint(BaseModel):
    """A bound on a weighted sum of named quantities:
    ``min_val <= sum(coeff * value) <= max_val``, either bound optional.

    In ``ProblemSpec.domain_constraints`` the names are design variables (or
    sources) and the constraint is hard: no design outside it is ever proposed. In
    ``ProblemSpec.output_constraints`` the names are outputs, whose values are only
    known after a run, so the constraint is learned with the outputs.

    The coefficients may be passed positionally::

        LinearConstraint({"temp": 1, "conc": 2}, max_val=100)
    """

    coeffs: dict[str, float]
    min_val: float | None = None
    max_val: float | None = None

    def __init__(self, coeffs: dict[str, float] | None = None, /, **data) -> None:
        if coeffs is not None:
            data["coeffs"] = coeffs
        super().__init__(**data)

    @model_validator(mode="after")
    def _check(self) -> LinearConstraint:
        if not self.coeffs:
            raise ValueError("LinearConstraint needs at least one name in coeffs")
        if self.min_val is None and self.max_val is None:
            raise ValueError(
                f"LinearConstraint over {list(self.coeffs)} needs min_val, max_val or both"
            )
        _check_range(f"LinearConstraint over {list(self.coeffs)}", self.min_val, self.max_val)
        return self


class InequalitySpec(BaseModel):
    """botorch sparse inequality: sum(coeffs_i * x[indices_i]) >= rhs.

    Superseded by :class:`LinearConstraint`, which names the variables; still
    accepted in ``ProblemSpec.domain_constraints``.
    """

    indices: list[int]
    coeffs: list[float]
    rhs: float

    @model_validator(mode="after")
    def _check(self) -> InequalitySpec:
        if len(self.indices) != len(self.coeffs):
            raise ValueError("InequalitySpec indices/coeffs must have equal length")
        return self


def _check_names(c, allowed, kind, other, where, other_hint) -> None:
    """Every name in constraint ``c`` must be one of ``allowed``; a name from the
    other list gets a pointed error saying where it belongs."""
    for name in c.coeffs:
        if name in allowed:
            continue
        if name in other:
            raise ValueError(f"{where}: {name!r} is {other_hint}")
        raise ValueError(f"{where}: {name!r} is not a known {kind} (known: {allowed})")


# --- The spec ----------------------------------------------------------------


def _encode(d, value) -> float:
    """One input's value as a code: an enumerated value via its domain, else the
    number."""
    if isinstance(d, EnumeratedDomain):
        return d.encode(value)
    if isinstance(value, str):
        raise ValueError(f"{d.name}={value!r}: expected a number")  # noqa: TRY004 - like a bad label
    return float(value)


class ProblemSpec(BaseModel):
    """The full, round-trippable problem definition.

    Round-trips to/from JSON (``model_dump_json`` / ``model_validate_json``) with
    no live-object references — the property that simultaneously serves the
    frontend, run resume, and agent drivers.

    Constraints come in two lists, named after what they constrain:

    - ``domain_constraints``: linear bounds on the design variables (and sources).
      Hard: every proposal and seed satisfies them. ``constraints=`` is accepted as
      the old name.
    - ``output_constraints``: linear bounds on combinations of outputs, learned from
      the runs like the outputs themselves. A bound on one output is simpler as
      ``OutputSpec(min_val=..., max_val=...)``.
    """

    domain: list[Domain]
    outputs: list[OutputSpec]
    boxes: list[Box] | None = None
    domain_constraints: list[LinearConstraint | InequalitySpec] | None = None
    output_constraints: list[LinearConstraint] | None = None
    sources: list[Source] | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_objectives_alias(cls, data):
        """Accept a legacy ``objectives=[...]`` kwarg as an alias for ``outputs``.

        Keeps every existing BO call site (and old JSON written with
        ``"objectives"``) working: a legacy objectives list is all-directional and
        becomes the ``outputs`` list verbatim.
        """
        if isinstance(data, dict) and "objectives" in data:
            if data.get("outputs") is not None:
                raise ValueError("pass either `outputs=` or `objectives=`, not both")
            data = {**data, "outputs": data["objectives"]}
            data.pop("objectives", None)
        if isinstance(data, dict) and "constraints" in data:
            if data.get("domain_constraints") is not None:
                raise ValueError(
                    "pass either `domain_constraints=` or `constraints=` (its old "
                    "name), not both"
                )
            data = {**data, "domain_constraints": data["constraints"]}
            data.pop("constraints", None)
        return data

    @model_validator(mode="after")
    def _check(self) -> ProblemSpec:
        if not self.domain:
            raise ValueError("ProblemSpec needs at least one design variable")
        if not self.outputs:
            raise ValueError("ProblemSpec needs at least one output")
        names = [d.name for d in self.inputs]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate variable names across domain and sources: {names}")
        onames = [o.name for o in self.outputs]
        if len(onames) != len(set(onames)):
            raise ValueError(f"Duplicate output names: {onames}")
        labeled = {
            d.name for d in self.inputs if isinstance(d, EnumeratedDomain) and d.is_categorical
        }
        for c in self.domain_constraints or []:
            if isinstance(c, InequalitySpec):
                bad = [i for i in c.indices if not 0 <= i < len(names)]
                if bad:
                    raise ValueError(
                        f"InequalitySpec indices {bad} are out of range for the "
                        f"{len(names)} variables {names}"
                    )
                used = [names[i] for i in c.indices]
            else:
                _check_names(c, names, "variable", onames, "domain_constraints",
                             "an output: put it in output_constraints, or set min_val/max_val "
                             "on its OutputSpec")
                used = list(c.coeffs)
            if labeled & set(used):
                raise ValueError(
                    f"domain_constraints: {sorted(labeled & set(used))} take labels, which "
                    "a linear constraint can't weigh"
                )
        for c in self.output_constraints or []:
            _check_names(c, onames, "output", names, "output_constraints",
                         "a design variable: put it in domain_constraints")
        for box in self.boxes or []:
            for name in box.bands or {}:
                if name not in onames:
                    raise ValueError(
                        f"Box band {name!r} is not an output (outputs: {onames})"
                    )
        return self

    @property
    def constraints(self) -> list | None:
        """Old name for ``domain_constraints`` (read-only)."""
        return self.domain_constraints

    # --- outputs vs objectives -------------------------------------------------
    # The surrogate models every OUTPUT (one column each, in ``outputs`` order) —
    # that is the evaluate()/History column contract. The OBJECTIVES are the subset
    # with a non-OBSERVE direction (the ones BO/MO optimize); they may be a
    # non-contiguous view, selected by ``objective_indices()``.

    # --- inputs: design variables then sources --------------------------------
    # A design vector (``Record.candidate``, ``suggest``'s dicts, the model's input,
    # ``bounds_tensor``) holds every INPUT: the design variables in ``domain`` order,
    # then the sources. ``design_names`` is the design-only view, for what gets
    # recommended. Without sources the two coincide.

    @property
    def inputs(self) -> list:
        """Every input's domain: ``domain`` then each source's domain."""
        return [*self.domain, *(s.domain for s in self.sources or [])]

    @property
    def variable_names(self) -> list[str]:
        """Names of every input (design variables, then sources)."""
        return [d.name for d in self.inputs]

    @property
    def design_names(self) -> list[str]:
        return [d.name for d in self.domain]

    @property
    def source_names(self) -> list[str]:
        return [s.name for s in self.sources or []]

    def source_indices(self) -> list[int]:
        """Positions of the sources in a design vector."""
        n = len(self.domain)
        return list(range(n, n + len(self.sources or [])))

    def log_indices(self) -> list[int]:
        """Positions of the inputs on a log scale (``log=True`` design variables and
        sources)."""
        out = [j for j, d in enumerate(self.domain) if getattr(d, "log", False)]
        out += [i for i, s in zip(self.source_indices(), self.sources or []) if s.is_log]
        return out

    def categorical_indices(self) -> list[int]:
        """Positions of the design variables whose values are labels (categories).
        A labeled source is a source, not a category: the source-aware model
        handles it."""
        return [
            j for j, d in enumerate(self.domain)
            if isinstance(d, EnumeratedDomain) and d.is_categorical
        ]

    def as_tensor(self, design) -> torch.Tensor:
        """A design ``{name: value}`` (labels for categories) as the ``(d,)`` tensor
        the loop works with, in ``variable_names`` order. Raises on a missing
        variable or a value its domain doesn't allow (an enumerated value)."""
        import torch

        missing = [n for n in self.variable_names if n not in design]
        if missing:
            raise ValueError(f"design is missing {missing} (variables: {self.variable_names})")
        return torch.tensor(
            [_encode(d, design[d.name]) for d in self.inputs], dtype=torch.double
        )

    def as_dict(self, x) -> dict:
        """A ``(d,)`` design tensor as ``{name: value}``, with labels for
        categories: the form ``Campaign.suggest`` hands out."""
        values = x.reshape(-1).tolist()
        return {
            d.name: d.decode(v) if isinstance(d, EnumeratedDomain) else v
            for d, v in zip(self.inputs, values, strict=True)
        }

    def resolve_fixed(self, fixed: dict[str, float | str] | None) -> dict[int, float]:
        """``{name: value}`` for inputs held fixed in a proposal, checked against
        their domains, as ``{position: value}`` (a label as its code)."""
        if not fixed:
            return {}
        position = {name: j for j, name in enumerate(self.variable_names)}
        out = {}
        for name, value in fixed.items():
            if name not in position:
                where = " (an output; only inputs can be fixed)" if name in self.output_names else ""
                raise ValueError(
                    f"fixed: {name!r} is not a variable{where} (variables: "
                    f"{self.variable_names})"
                )
            j = position[name]
            d = self.inputs[j]
            if isinstance(d, EnumeratedDomain):
                try:
                    out[j] = d.encode(value)
                except ValueError as e:
                    raise ValueError(f"fixed: {e}") from None
                continue
            value = float(value)
            if not d.lower <= value <= d.upper:
                raise ValueError(
                    f"fixed: {name}={value} is outside [{d.lower}, {d.upper}]"
                )
            elif isinstance(d, IntegerDomain) and value != int(value):
                raise ValueError(f"fixed: {name}={value} must be a whole number")
            out[j] = value
        return out

    def target_features(self) -> dict[int, float]:
        """``{position: target}`` for every source; empty without sources."""
        return {i: s.target_code for i, s in zip(self.source_indices(), self.sources or [])}

    def at_target(self, X):
        """``X (..., d)`` with every source column set to its target (a copy)."""
        X = X.clone()
        for i, t in self.target_features().items():
            X[..., i] = t
        return X

    def fingerprint(self) -> dict:
        """The spec as JSON for a store's resume check, minus source targets (a
        target is a recommendation-time setting and may change between runs)."""
        data = self.model_dump(mode="json")
        for s in data.get("sources") or []:
            s.pop("target", None)
        # Under its old key, so stores written before the rename still resume.
        data["constraints"] = data.pop("domain_constraints")
        return data

    @property
    def output_names(self) -> list[str]:
        return [o.name for o in self.outputs]

    @property
    def n_outputs(self) -> int:
        return len(self.outputs)

    @property
    def objectives(self) -> list[OutputSpec]:
        """The optimized subset of outputs (direction != OBSERVE)."""
        return [o for o in self.outputs if o.direction != "OBSERVE"]

    @property
    def n_objectives(self) -> int:
        return len(self.objectives)

    def objective_indices(self) -> list[int]:
        """Column positions (in ``outputs`` order) of the optimized objectives."""
        return [i for i, o in enumerate(self.outputs) if o.direction != "OBSERVE"]

    def output_signs(self, dtype=None, device=None):
        """Per-OUTPUT sign vector for maximize-canonical fitting: +1 for MAXIMIZE
        and OBSERVE, -1 for MINIMIZE. Column order matches ``self.outputs``.

        The engine is maximize-canonical: multiplying raw physical outputs by this
        vector gives the "bigger is better" targets the surrogate sees, so an
        ``evaluate`` callback returns raw measurements and never pre-negates.
        OBSERVE columns pass through unflipped (the goal reads them raw).
        """
        import torch

        signs = [-1.0 if o.direction == "MINIMIZE" else 1.0 for o in self.outputs]
        return torch.tensor(signs, dtype=dtype or torch.double, device=device)

    def objective_signs(self, dtype=None, device=None):
        """Signs over the OBJECTIVE subset only (compat / BO-best helpers)."""
        import torch

        signs = [-1.0 if o.direction == "MINIMIZE" else 1.0 for o in self.objectives]
        return torch.tensor(signs, dtype=dtype or torch.double, device=device)

    def objective_view(self, Y):
        """Select the objective columns of raw outputs and flip to canonical space.

        ``Y`` is ``(n, n_outputs)`` of RAW physical values (``spec.outputs`` order);
        the result is ``(n, n_objectives)`` in maximize-canonical space — the
        objective subset, signs applied. This concentrates the load-bearing
        ``Y[:, objective_indices()] * objective_signs()`` pairing (the index set and
        the sign set must agree) in one place instead of at every BO/analysis site.
        """
        Y = Y.double()
        return Y[:, self.objective_indices()] * self.objective_signs(
            dtype=Y.dtype, device=Y.device
        )

    def to_raw(self, canonical):
        """Flip a canonical-space quantity back to RAW physical units (per output).

        The inverse of the maximize-canonical fit: multiply by ``output_signs()``
        (sign is its own inverse). ``canonical`` is ``(..., n_outputs)`` in
        ``spec.outputs`` order; OBSERVE columns pass through unchanged. Used to
        report posterior means/observations in physical units for display.
        """
        canonical = canonical.double()
        return canonical * self.output_signs(dtype=canonical.dtype, device=canonical.device)

    def bounds_tensor(self, dtype=None, device=None):
        """Return a (2, d) lower/upper bounds tensor for ``optimize_acqf``.

        Continuous → (lower, upper); Enumerated → (min, max) of its codes (the
        proposer still enumerates — these bounds are only a containing box).
        StructuredDomain is not yet supported here (deferred).
        """
        import torch

        lowers: list[float] = []
        uppers: list[float] = []
        for d in self.inputs:
            if isinstance(d, (ContinuousDomain, IntegerDomain)):
                lowers.append(d.lower)
                uppers.append(d.upper)
            elif isinstance(d, EnumeratedDomain):
                lowers.append(min(d.codes))
                uppers.append(max(d.codes))
            else:
                raise NotImplementedError(
                    f"bounds_tensor does not yet support {type(d).__name__} "
                    f"(variable {d.name!r}); structured spaces are deferred."
                )
        return torch.tensor([lowers, uppers], dtype=dtype or torch.double, device=device)

    def inequality_constraints(self, dtype=None, device=None):
        """The domain constraints in botorch ``inequality_constraints`` form
        (``sum(coeffs * x[indices]) >= rhs``, one entry per bound), or None."""
        if not self.domain_constraints:
            return None
        import torch

        def entry(indices, coeffs, rhs):
            return (
                torch.tensor(indices, dtype=torch.long, device=device),
                torch.tensor(coeffs, dtype=dtype or torch.double, device=device),
                float(rhs),
            )

        position = {name: j for j, name in enumerate(self.variable_names)}
        out = []
        for c in self.domain_constraints:
            if isinstance(c, InequalitySpec):
                out.append(entry(c.indices, c.coeffs, c.rhs))
                continue
            indices = [position[name] for name in c.coeffs]
            coeffs = list(c.coeffs.values())
            if c.min_val is not None:
                out.append(entry(indices, coeffs, c.min_val))
            if c.max_val is not None:
                out.append(entry(indices, [-a for a in coeffs], -c.max_val))
        return out

    @property
    def has_output_constraints(self) -> bool:
        """Whether any output is bounded (``OutputSpec`` bounds or
        ``output_constraints``)."""
        return bool(self.output_constraints) or any(o.is_constrained for o in self.outputs)

    def output_constraint_rows(self, dtype=None, device=None):
        """Every output constraint as rows ``lower <= A @ y <= upper`` over RAW
        outputs: ``A (k, n_outputs)``, ``lower (k,)``, ``upper (k,)``, with a missing
        bound as -inf / inf. ``k`` is 0 without output constraints."""
        import torch

        kw = {"dtype": dtype or torch.double, "device": device}
        rows, lower, upper = [], [], []
        position = {name: j for j, name in enumerate(self.output_names)}

        def add(coeffs: dict[str, float], lo, hi):
            a = torch.zeros(self.n_outputs, **kw)
            for name, coef in coeffs.items():
                a[position[name]] = coef
            rows.append(a)
            lower.append(-math.inf if lo is None else lo)
            upper.append(math.inf if hi is None else hi)

        for o in self.outputs:
            if o.is_constrained:
                add({o.name: 1.0}, o.min_val, o.max_val)
        for c in self.output_constraints or []:
            add(c.coeffs, c.min_val, c.max_val)
        A = torch.stack(rows) if rows else torch.zeros(0, self.n_outputs, **kw)
        return A, torch.tensor(lower, **kw), torch.tensor(upper, **kw)

    def sample_designs(
        self, n: int, *, seed: int | None = None, fixed: dict[str, float] | None = None
    ):
        """Draw ``n`` designs as an (n, d) tensor.

        Unconstrained, they are space-filling (scrambled Sobol): continuous axes
        span their bounds, enumerated axes land on their allowed values. With
        inequality constraints on an all-continuous space the draw is uniform over
        the feasible polytope by hit-and-run (botorch's ``get_polytope_samples``),
        which cannot fail on a thin feasible region but is random rather than
        low-discrepancy. With constraints and enumerated axes, Sobol draws are
        filtered by rejection.

        ``seed=None`` draws the seed from torch's global RNG, so
        ``torch.manual_seed`` reproduces it. ``fixed`` (``{name: value}``) holds
        inputs at given values.
        """
        return self._sample(n, seed, self.resolve_fixed(fixed))

    def _sample(self, n: int, seed: int | None, pinned: dict[int, float]):
        """``sample_designs`` with the fixed inputs as ``{position: code}``."""
        import torch
        from torch.quasirandom import SobolEngine

        if seed is None:
            seed = int(torch.randint(2**31 - 1, (1,)))
        if self.domain_constraints and all(isinstance(d, ContinuousDomain) for d in self.inputs):
            from botorch.utils.sampling import get_polytope_samples

            equality = [
                (torch.tensor([j]), torch.tensor([1.0], dtype=torch.double), v)
                for j, v in pinned.items()
            ]
            return get_polytope_samples(
                n,
                self.bounds_tensor(),
                inequality_constraints=self.inequality_constraints(),
                equality_constraints=equality or None,
                seed=seed,
            )
        engine = SobolEngine(dimension=len(self.inputs), scramble=True, seed=seed)

        def draw(k):
            x = self._from_unit(engine.draw(k).double())
            for j, v in pinned.items():
                x[:, j] = v
            return x

        if not self.domain_constraints:
            return draw(n)

        kept: list = []
        n_kept = 0
        for _ in range(100):
            x = draw(max(n, 64))
            x = x[self._feasible(x)]
            kept.append(x)
            n_kept += x.shape[0]
            if n_kept >= n:
                return torch.cat(kept)[:n]
        raise ValueError(
            f"Could only find {n_kept} of {n} designs satisfying the inequality "
            "constraints by rejection sampling; the feasible region may be tiny. "
            "Pass initial designs explicitly."
        )

    def _from_unit(self, u, inputs: list | None = None):
        """Map (n, k) points in [0, 1]^k onto the design space: every input, or the
        first k (``inputs``, e.g. the design variables alone)."""
        import torch

        inputs = self.inputs if inputs is None else inputs
        log_cols = set(self.log_indices())
        cols = []
        for j, d in enumerate(inputs):
            if isinstance(d, (ContinuousDomain, IntegerDomain)) and j in log_cols:
                # log-uniform: a size-like source spends its draws across scales
                lo, hi = math.log(d.lower), math.log(d.upper)
                col = torch.exp(lo + u[:, j] * (hi - lo))
                cols.append(col.round().clamp(d.lower, d.upper) if isinstance(d, IntegerDomain) else col)
            elif isinstance(d, IntegerDomain):
                span = d.upper - d.lower + 1
                cols.append((d.lower + (u[:, j] * span).floor()).clamp(max=d.upper))
            elif isinstance(d, ContinuousDomain):
                cols.append(d.lower + u[:, j] * (d.upper - d.lower))
            elif isinstance(d, EnumeratedDomain):
                values = torch.tensor(d.codes, dtype=torch.double)
                idx = (u[:, j] * len(values)).long().clamp(max=len(values) - 1)
                cols.append(values[idx])
            else:
                raise NotImplementedError(
                    f"sample_designs does not yet support {type(d).__name__} "
                    f"(variable {d.name!r}); structured spaces are deferred."
                )
        return torch.stack(cols, dim=-1)

    def _feasible(self, x):
        """Boolean mask of rows of ``x`` satisfying every domain constraint."""
        import torch

        ok = torch.ones(x.shape[0], dtype=torch.bool)
        for indices, coeffs, rhs in self.inequality_constraints(dtype=x.dtype) or []:
            ok &= (x[:, indices] * coeffs).sum(dim=-1) >= rhs
        return ok
