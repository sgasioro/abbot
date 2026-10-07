"""Xopt / gest-api interop (F3, stub).

Target the gest-api STANDARD, not Xopt internals. gest-api is pydantic-only
(no torch/botorch/pandas), so ``vocs <-> spec`` is a field map against stable
typed objects, and ``abbot[xopt]`` is only needed for the orchestrator/Badger
path, never the core.

Planned surface:
  - ``vocs_to_spec`` / ``spec_to_vocs`` — map abbot Domains onto gest's
    ``ContinuousVariable.domain`` / ``DiscreteVariable.values`` /
    ``ContextualVariable``, objectives onto ``MinimizeObjective`` /
    ``MaximizeObjective`` / ``ExploreObjective``, constraints onto
    ``LessThanConstraint`` / ``GreaterThanConstraint`` / ``BoundsConstraint``.
  - ``CorePolicyGenerator`` — implement the gest ``Generator`` surface
    (``suggest`` / ``ingest`` / ``finalize``) wrapping an abbot ``Policy``, so a
    problem runs under Xopt/Badger WITHOUT the core depending on Xopt.

Known limitation: sbido's composite ``aux`` payload has no DataFrame
representation, so only its reduced-scalar form runs as a plain Xopt generator.

Not implemented in M1 — gest-api types are confirmed available
(``gest_api.vocs``: Continuous/Discrete/Contextual variables, Minimize/Maximize/
Explore objectives; ``gest_api.Generator``: suggest/ingest/finalize). This module
exists to pin the interop shape; see plan F3.
"""

from __future__ import annotations

from abbot.problem import ProblemSpec


def vocs_to_spec(vocs) -> ProblemSpec:  # pragma: no cover - F3
    raise NotImplementedError("gest-api VOCS <-> ProblemSpec adapter lands in F3")


def spec_to_vocs(spec: ProblemSpec):  # pragma: no cover - F3
    raise NotImplementedError("gest-api VOCS <-> ProblemSpec adapter lands in F3")
