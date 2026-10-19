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


Direction = Literal["MINIMIZE", "MAXIMIZE", "OBSERVE"]


class OutputSpec(BaseModel):
    """A named MODELED output of an experiment (one surrogate column).

    ``direction``:
      - ``MAXIMIZE`` / ``MINIMIZE`` — an optimized *objective* (BO/MO).
      - ``OBSERVE``  — modeled but NOT optimized: a feature a BAX ``Goal``
        consumes, or (future) a constraint output. OBSERVE outputs are fit by the
        surrogate like any other column; they are simply excluded from the
        optimized-objective subset (``ProblemSpec.objectives``).
    """

    name: str
    direction: Direction = "MAXIMIZE"


# Back-compat: an "objective" is just a non-OBSERVE output. Existing
# ``ObjectiveSpec(name=..., direction="MAXIMIZE"|"MINIMIZE")`` call sites keep working.
ObjectiveSpec = OutputSpec


class Box(BaseModel):
    """A target band (per-dimension lower/upper) in output or theta space.

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
    outputs: list[OutputSpec]
    boxes: list[Box] | None = None
    constraints: list[InequalitySpec] | None = None

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
        return data

    @model_validator(mode="after")
    def _check(self) -> ProblemSpec:
        if not self.domain:
            raise ValueError("ProblemSpec needs at least one design variable")
        if not self.outputs:
            raise ValueError("ProblemSpec needs at least one output")
        names = [d.name for d in self.domain]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate variable names in domain: {names}")
        onames = [o.name for o in self.outputs]
        if len(onames) != len(set(onames)):
            raise ValueError(f"Duplicate output names: {onames}")
        return self

    # --- outputs vs objectives -------------------------------------------------
    # The surrogate models every OUTPUT (one column each, in ``outputs`` order) —
    # that is the evaluate()/History column contract. The OBJECTIVES are the subset
    # with a non-OBSERVE direction (the ones BO/MO optimize); they may be a
    # non-contiguous view, selected by ``objective_indices()``.

    @property
    def variable_names(self) -> list[str]:
        return [d.name for d in self.domain]

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

    def sample_designs(self, n: int, *, seed: int | None = None):
        """Draw ``n`` designs as an (n, d) tensor.

        Unconstrained, they are space-filling (scrambled Sobol): continuous axes
        span their bounds, enumerated axes land on their allowed values. With
        inequality constraints on an all-continuous space the draw is uniform over
        the feasible polytope by hit-and-run (botorch's ``get_polytope_samples``),
        which cannot fail on a thin feasible region but is random rather than
        low-discrepancy; with enumerated axes, Sobol draws are filtered by rejection. ``seed=None``
        draws the seed from torch's global RNG, so ``torch.manual_seed``
        reproduces it.
        """
        import torch
        from torch.quasirandom import SobolEngine

        if seed is None:
            seed = int(torch.randint(2**31 - 1, (1,)))
        if self.constraints and all(isinstance(d, ContinuousDomain) for d in self.domain):
            from botorch.utils.sampling import get_polytope_samples

            return get_polytope_samples(
                n,
                self.bounds_tensor(),
                inequality_constraints=self.inequality_constraints(),
                seed=seed,
            )
        engine = SobolEngine(dimension=len(self.domain), scramble=True, seed=seed)
        if not self.constraints:
            return self._from_unit(engine.draw(n).double())

        kept: list = []
        n_kept = 0
        for _ in range(100):
            x = self._from_unit(engine.draw(max(n, 64)).double())
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

    def _from_unit(self, u):
        """Map (n, d) points in [0, 1]^d onto the design space."""
        import torch

        cols = []
        for j, d in enumerate(self.domain):
            if isinstance(d, ContinuousDomain):
                cols.append(d.lower + u[:, j] * (d.upper - d.lower))
            elif isinstance(d, EnumeratedDomain):
                values = torch.tensor(d.values, dtype=torch.double)
                idx = (u[:, j] * len(values)).long().clamp(max=len(values) - 1)
                cols.append(values[idx])
            else:
                raise NotImplementedError(
                    f"sample_designs does not yet support {type(d).__name__} "
                    f"(variable {d.name!r}); structured spaces are deferred."
                )
        return torch.stack(cols, dim=-1)

    def _feasible(self, x):
        """Boolean mask of rows of ``x`` satisfying every inequality constraint."""
        import torch

        ok = torch.ones(x.shape[0], dtype=torch.bool)
        for c in self.constraints or []:
            coeffs = torch.tensor(c.coeffs, dtype=x.dtype)
            ok &= (x[:, c.indices] * coeffs).sum(dim=-1) >= c.rhs
        return ok
