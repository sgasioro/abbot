"""Serializable problem specification.

The ``ProblemSpec`` is the typed, JSON-serializable contract shared by the loop,
the (future) web frontend, and any agent driver. It is pydantic so that
``model_dump_json`` + JSON-schema export give browser forms for free, and so the
``vocs <-> spec`` interop adapter is a field map against gest-api's typed VOCS
rather than a format bridge.

Design spaces are **type-polymorphic from day one** (Continuous | Enumerated |
Structured) — the exact axis Badger is rigid on, and hence a differentiator.
Each Domain maps onto a gest-api variable type for interop (see interop/xopt.py).

NB: this is the frontend/interop-facing spec. The in-memory numeric bounds that
botorch's ``optimize_acqf`` consumes are derived from it (see ``bounds_tensor``);
heavy / live objects (a trained model, an HTTP reactor Experiment) are referenced
by registry key + config, never embedded here.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

# --- Domains -----------------------------------------------------------------
# A Domain describes ONE design variable. The surrogate always sees a (..., d)
# float tensor; truly-discrete *design spaces* are handled at the candidate-
# proposal layer (acq_opt), not the surrogate. Enumerated/categorical variables
# are featurized to floats for the model but enumerated for proposal.


class ContinuousDomain(BaseModel):
    kind: Literal["continuous"] = "continuous"
    name: str
    lower: float
    upper: float

    @model_validator(mode="after")
    def _check_bounds(self) -> ContinuousDomain:
        if self.upper <= self.lower:
            raise ValueError(
                f"ContinuousDomain {self.name!r}: upper ({self.upper}) must exceed "
                f"lower ({self.lower})"
            )
        return self


class EnumeratedDomain(BaseModel):
    """A finite, enumerable set of allowed values (categorical or discrete grid).

    Proposal enumerates ``values``; the surrogate sees the featurized form.
    """

    kind: Literal["enumerated"] = "enumerated"
    name: str
    values: list[float]

    @model_validator(mode="after")
    def _check_values(self) -> EnumeratedDomain:
        if len(self.values) < 1:
            raise ValueError(f"EnumeratedDomain {self.name!r} needs at least one value")
        return self


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
    ContinuousDomain | EnumeratedDomain | StructuredDomain,
    Field(discriminator="kind"),
]


# --- Objectives / constraints ------------------------------------------------


class ObjectiveSpec(BaseModel):
    """A named objective with a direction (Xopt VOCS parity)."""

    name: str
    direction: Literal["MINIMIZE", "MAXIMIZE"] = "MAXIMIZE"


class Box(BaseModel):
    """An sbido-style target band in output/theta space.

    A list of boxes drives a BoxMembershipGoal (see goal.py). ``lower``/``upper``
    are per-dimension vectors of equal length.
    """

    lower: list[float]
    upper: list[float]

    @model_validator(mode="after")
    def _check(self) -> Box:
        if len(self.lower) != len(self.upper):
            raise ValueError("Box lower/upper must have equal length")
        for j, (lo, hi) in enumerate(zip(self.lower, self.upper)):
            if hi <= lo:
                raise ValueError(f"Box dim {j}: upper ({hi}) must exceed lower ({lo})")
        return self


class InequalitySpec(BaseModel):
    """botorch sparse inequality: sum(coeffs_i * x[indices_i]) >= rhs."""

    indices: list[int]
    coeffs: list[float]
    rhs: float

    @model_validator(mode="after")
    def _check(self) -> InequalitySpec:
        if len(self.indices) != len(self.coeffs):
            raise ValueError("InequalitySpec indices/coeffs must have equal length")
        return self


# --- The spec ----------------------------------------------------------------


class ProblemSpec(BaseModel):
    """The full, round-trippable problem definition.

    Round-trips to/from JSON (``model_dump_json`` / ``model_validate_json``) with
    no live-object references — the property that simultaneously serves the
    frontend, run resume, and agent drivers.
    """

    domain: list[Domain]
    objectives: list[ObjectiveSpec]
    boxes: list[Box] | None = None
    constraints: list[InequalitySpec] | None = None

    @model_validator(mode="after")
    def _check(self) -> ProblemSpec:
        if not self.domain:
            raise ValueError("ProblemSpec needs at least one design variable")
        if not self.objectives:
            raise ValueError("ProblemSpec needs at least one objective")
        names = [d.name for d in self.domain]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate variable names in domain: {names}")
        return self

    @property
    def variable_names(self) -> list[str]:
        return [d.name for d in self.domain]

    def bounds_tensor(self, dtype=None, device=None):
        """Return a (2, d) lower/upper bounds tensor for ``optimize_acqf``.

        Continuous → (lower, upper); Enumerated → (min, max) of its values (the
        proposer still enumerates — these bounds are only a containing box).
        StructuredDomain is not yet supported here (deferred).
        """
        import torch

        lowers: list[float] = []
        uppers: list[float] = []
        for d in self.domain:
            if isinstance(d, ContinuousDomain):
                lowers.append(d.lower)
                uppers.append(d.upper)
            elif isinstance(d, EnumeratedDomain):
                lowers.append(min(d.values))
                uppers.append(max(d.values))
            else:
                raise NotImplementedError(
                    f"bounds_tensor does not yet support {type(d).__name__} "
                    f"(variable {d.name!r}); structured spaces are deferred."
                )
        return torch.tensor([lowers, uppers], dtype=dtype or torch.double, device=device)

    def inequality_constraints(self, dtype=None, device=None):
        """Return botorch ``inequality_constraints`` form, or None."""
        if not self.constraints:
            return None
        import torch

        return [
            (
                torch.tensor(c.indices, dtype=torch.long, device=device),
                torch.tensor(c.coeffs, dtype=dtype or torch.double, device=device),
                float(c.rhs),
            )
            for c in self.constraints
        ]
