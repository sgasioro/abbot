"""Interop with sbido (lazy import; ``abbot[sbi]`` for the heavy path).

sbido's classes satisfy abbot's ``Experiment`` / ``PosteriorSampler`` protocols
by shape, and abbot core never imports sbido. sbido is becoming a thin SBI layer
on abbot (its BAX designers already import it), so this adapter is slated to
move to sbido's side; until then it is the one place that absorbs sbido drift.

Two bridges:
  - ``spec_from_sbido`` — sbido ``ProblemSpec`` (xi_bounds + target boxes +
    inequality constraints as tensors) -> abbot ``ProblemSpec`` (Domains + Boxes +
    InequalitySpecs). sbido's design vars are unnamed/continuous, so they become
    ``x0..xd`` ContinuousDomains; its one scalar objective is MAXIMIZE (both
    indicator in [0,1] and distance <=0 are maximized toward their best).
  - ``conforms`` — a runtime check that a sbido object satisfies an abbot protocol
    (used by the conformance test to catch drift at the boundary, not mid-run).

sbido's ``Experiment.run`` takes a design TENSOR ``xi`` (abbot's ``run`` takes a
``{name: value}`` dict), so wrap it: ``abbot.TensorExperiment(sbido_experiment)``.
It returns a TypedDict ``Observation`` ({"obs", "meta"}); its
``PosteriorSampler.sample(observation, n)`` already matches abbot's shape, so the
``evaluate`` seam (see ``abbot.sbi.sbi_evaluate``) consumes both unchanged.
"""

from __future__ import annotations

from typing import Any

from abbot.problem import (
    Box,
    ContinuousDomain,
    InequalitySpec,
    OutputSpec,
    ProblemSpec,
)


def spec_from_sbido(sbido_spec: Any, objective_name: str = "F") -> ProblemSpec:
    """Convert a sbido ``ProblemSpec`` to an abbot ``ProblemSpec``.

    sbido spec fields (``~/sbido/src/sbido/interfaces.py``):
      - ``xi_bounds``: (2, d) lower/upper design bounds.
      - ``boxes``: list of ``(a, b)`` target-band tensor tuples over theta.
      - ``inequality_constraints``: optional botorch sparse form
        ``[(indices, coeffs, rhs), ...]`` with ``sum(coef*xi[idx]) >= rhs``.
    """
    lower = sbido_spec.xi_bounds[0].tolist()
    upper = sbido_spec.xi_bounds[1].tolist()
    domain = [
        ContinuousDomain(name=f"x{i}", lower=float(lo), upper=float(hi))
        for i, (lo, hi) in enumerate(zip(lower, upper))
    ]
    boxes = [
        Box(lower=a.tolist(), upper=b.tolist()) for a, b in sbido_spec.boxes
    ]
    constraints = None
    if getattr(sbido_spec, "inequality_constraints", None):
        constraints = [
            InequalitySpec(
                indices=[int(i) for i in idx.tolist()],
                coeffs=[float(c) for c in coef.tolist()],
                rhs=float(rhs),
            )
            for idx, coef, rhs in sbido_spec.inequality_constraints
        ]
    return ProblemSpec(
        domain=domain,
        outputs=[OutputSpec(name=objective_name, direction="MAXIMIZE")],
        boxes=boxes,
        constraints=constraints,
    )


def conforms(obj: Any, protocol: type) -> bool:
    """True if ``obj`` structurally satisfies a ``runtime_checkable`` abbot protocol.

    Thin wrapper over ``isinstance(obj, protocol)`` with a clearer name; used by the
    conformance test so a sbido API change surfaces here, not deep in a run.
    """
    return isinstance(obj, protocol)


def bax_policy_for(
    spec: Any,
    strategy: str = "SwitchBAX",
    *,
    percentile: float = 80.0,
    scalar: bool = True,
    n_candidates: int = 1000,
    n_posterior_samples: int = 20,
):
    """Build an abbot ``BAXPolicy`` matching a sbido BAX designer's intent.

    sbido has two BAX modes (``~/sbido/src/sbido/methods/bax.py``):
      - ``BaxDesigner``: GP over the scalar ``or_score`` + ``PercentileSet1D`` ->
        here ``scalar=True`` -> ``BAXPolicy(PercentileSet, ...)`` on a 1-output model.
      - ``ThetaBaxDesigner``: GP over per-dim theta means + ``Wishlist`` -> here
        ``scalar=False`` -> ``BAXPolicy(goals_from_spec(spec), ...)`` on a
        multi-output model (the "outputs" are the tracked theta-dim means, NOT the
        BoxMembershipReducer scalar).
    """
    from abbot.goal import PercentileSet, goals_from_spec
    from abbot.policy import BAXPolicy

    goal = PercentileSet(percentile) if scalar else goals_from_spec(spec)
    return BAXPolicy(
        goal,
        strategy=strategy,
        n_candidates=n_candidates,
        n_posterior_samples=n_posterior_samples,
    )
