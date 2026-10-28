"""Live-analysis built-ins, computed from a campaign's history + fitted model.

These are pure functions of ``(History, spec[, model])`` so they are unit-testable
without a UI and reusable by a notebook or an agent. The UI registers them through
``Campaign``'s ``on_step`` hook and renders the results; callers can register their
own analyses the same way. Kept deliberately small — the general-purpose "any
analysis" surface is the hook itself; these are just the common basics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from abbot.history import History
from abbot.problem import ProblemSpec
from abbot.session import canonical_Y, observed_front

if TYPE_CHECKING:
    from abbot.policy import BAXPolicy


def _counts(r, spec: ProblemSpec) -> bool:
    """A record counts towards the best so far: clean, measured, and satisfying
    every output constraint."""
    if r.fault or r.outputs is None:
        return False
    if not spec.has_output_constraints:
        return True
    from abbot.feasibility import feasible

    return bool(feasible(spec, r.outputs.double().reshape(1, -1))[0])


def convergence(history: History, spec: ProblemSpec) -> list[float]:
    """Running best objective value vs. step, in RAW units (single-objective only).

    Element ``i`` is the best clean observed objective among the first ``i+1``
    records, respecting direction (max of raw for MAXIMIZE, min for MINIMIZE).
    Faulted records carry forward the previous best, and so do records that break
    an output constraint (NaN until one satisfies them).
    """
    if spec.n_objectives != 1:
        raise ValueError("convergence is single-objective; use hypervolume for MO.")
    col = spec.objective_indices()[0]  # objective may not be column 0 (OBSERVE outputs)
    maximize = spec.objectives[0].direction == "MAXIMIZE"
    best = None
    out: list[float] = []
    for r in history.records:
        if _counts(r, spec):
            v = float(r.outputs.reshape(-1)[col])
            if best is None:
                best = v
            else:
                best = max(best, v) if maximize else min(best, v)
        out.append(best if best is not None else float("nan"))
    return out


def hypervolume(history: History, spec: ProblemSpec) -> list[float]:
    """Dominated hypervolume of the observed canonical outputs vs. step (MO).

    Uses botorch's exact ``Hypervolume`` against an inferred reference point (from
    the final canonical set, so the series is comparable across steps). Returns a
    per-step list; steps before any clean point are 0.0. With output
    constraints, only records that satisfy them count.
    """
    if spec.n_objectives < 2:
        raise ValueError("hypervolume needs >=2 objectives; use convergence for 1.")
    from botorch.utils.multi_objective.hypervolume import (
        Hypervolume,
        infer_reference_point,
    )
    from botorch.utils.multi_objective.pareto import is_non_dominated

    clean = [r for r in history.records if _counts(r, spec)]
    if not clean:
        return [0.0] * len(history.records)
    # Objective columns in canonical space, one row per clean record (same rule
    # everywhere via spec.objective_view).
    all_Y = spec.objective_view(torch.stack([r.outputs.reshape(-1) for r in clean]))
    ref = infer_reference_point(all_Y[is_non_dominated(all_Y)])
    hv = Hypervolume(ref_point=ref)

    out: list[float] = []
    acc: list[torch.Tensor] = []
    for r in history.records:
        if _counts(r, spec):
            acc.append(spec.objective_view(r.outputs.reshape(1, -1))[0])
        if acc:
            Y = torch.stack(acc)
            front = Y[is_non_dominated(Y)]
            try:
                out.append(float(hv.compute(front)))
            except Exception:  # noqa: BLE001 - a degenerate front -> HV 0, keep the series going
                out.append(0.0)
        else:
            out.append(0.0)
    return out


def recommended_set(history: History, spec: ProblemSpec, model=None) -> list[int]:
    """Current recommended handover indices.

    If a fitted ``model`` is given, uses the denoised posterior-mean recommendation
    (via the policy path is the caller's job); here we expose the model-free
    observed non-dominated set, which is also the loop's fallback.
    """
    return observed_front(history, spec)


# --- model visualization ------------------------------------------------------
# The surrogate is fit in maximize-canonical space (see session.canonical_Y), so
# these un-apply the objective signs to report posterior values in RAW units.


def _posterior_raw(model, X: torch.Tensor, spec: ProblemSpec):
    """Posterior (mean, std) at X over ALL outputs, converted back to RAW units.

    Returns tensors shaped (n, n_outputs). The sign flip maps canonical -> raw for
    the mean (OBSERVE columns pass through); std is sign-invariant.
    """
    with torch.no_grad():
        post = model.posterior(X.double())
        mean = spec.to_raw(post.mean.reshape(X.shape[0], -1))  # canonical -> raw
        std = post.variance.clamp_min(1e-12).sqrt().reshape(X.shape[0], -1)
    return mean, std


def parity(history: History, spec: ProblemSpec, model) -> dict:
    """Predicted (posterior mean ± std) vs. observed, per OUTPUT — dimension-agnostic.

    Returns ``{output_name: {"observed": [...], "predicted": [...], "std": [...]}}``
    for every modeled output (incl. OBSERVE). The go-to "is the model any good?"
    view when the design space is too high-D to plot.
    """
    clean = history.clean
    if not len(clean):
        return {}
    X = clean.designs.double()
    Y = clean.outputs.double()  # raw observed, full width
    mean, std = _posterior_raw(model, X, spec)
    out = {}
    for j, o in enumerate(spec.outputs):
        out[o.name] = {
            "observed": Y[:, j].tolist(),
            "predicted": mean[:, j].tolist(),
            "std": std[:, j].tolist(),
        }
    return out


def _anchor_point(history: History, spec: ProblemSpec) -> torch.Tensor:
    """The design to hold the non-varied dimensions at: the recommended/best point."""
    rec = observed_front(history, spec)
    idx = rec[0] if rec else 0
    return history.records[idx].candidate.reshape(-1).double()


def slice_1d(
    history: History,
    spec: ProblemSpec,
    model,
    var: int = 0,
    output: int = 0,
    n: int = 80,
) -> dict:
    """1-D posterior slice: vary one variable over its range, others fixed at the
    recommended point. ``output`` indexes ``spec.outputs``. Returns grid ``x``, raw
    ``mean``, and ``std`` for a band.
    """
    bounds = spec.bounds_tensor()
    lo, hi = float(bounds[0, var]), float(bounds[1, var])
    anchor = _anchor_point(history, spec)
    xs = torch.linspace(lo, hi, n, dtype=torch.double)
    grid = anchor.repeat(n, 1)
    grid[:, var] = xs
    mean, std = _posterior_raw(model, grid, spec)
    return {
        "x": xs.tolist(),
        "mean": mean[:, output].tolist(),
        "std": std[:, output].tolist(),
        "var_name": spec.variable_names[var],
        "objective_name": spec.outputs[output].name,
    }


def slice_2d(
    history: History,
    spec: ProblemSpec,
    model,
    var_x: int = 0,
    var_y: int = 1,
    output: int = 0,
    n: int = 40,
) -> dict:
    """2-D posterior-mean slice: vary two variables over a grid, others fixed at the
    recommended point. ``output`` indexes ``spec.outputs``. Returns ``x``, ``y``
    axes and a ``z`` matrix (raw mean).
    """
    bounds = spec.bounds_tensor()
    xs = torch.linspace(float(bounds[0, var_x]), float(bounds[1, var_x]), n, dtype=torch.double)
    ys = torch.linspace(float(bounds[0, var_y]), float(bounds[1, var_y]), n, dtype=torch.double)
    anchor = _anchor_point(history, spec)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    grid = anchor.repeat(n * n, 1)
    grid[:, var_x] = gx.reshape(-1)
    grid[:, var_y] = gy.reshape(-1)
    mean, _ = _posterior_raw(model, grid, spec)
    z = mean[:, output].reshape(n, n)
    return {
        "x": xs.tolist(),
        "y": ys.tolist(),
        "z": z.tolist(),
        "var_x_name": spec.variable_names[var_x],
        "var_y_name": spec.variable_names[var_y],
        "objective_name": spec.outputs[output].name,
    }


def bax_target_region(
    policy: BAXPolicy, model, spec: ProblemSpec
) -> dict[str, int]:
    """Snapshot of the BAX discovered-target region on the current posterior mean.

    Returns ``{"identified": int, "grid_size": int}``: how many candidate-grid
    points the policy's goal selects from the posterior-MEAN prediction (the size
    of the region found so far — should grow then stabilize), and the grid size.
    When the policy applies output constraints, ``identified`` counts what the goal
    selects among points predicted to satisfy them, and ``"before_constraints"``
    what it selects from every point, so the narrowing is visible.
    Pure function of ``(policy, model, spec)``; encapsulates grid + raw-posterior +
    ``goal.identify_subspace`` so no BAX internals leak to the caller.
    """
    grid = policy.candidate_grid(spec)
    mean, _ = _posterior_raw(model, grid, spec)
    out = {
        "identified": len(policy._identify(spec, mean, grid)),
        "grid_size": int(grid.shape[0]),
    }
    if policy._constrains(spec):
        out["before_constraints"] = len(policy.goal.identify_subspace(mean, grid))
    return out


__all__ = [
    "bax_target_region",
    "canonical_Y",
    "convergence",
    "hypervolume",
    "parity",
    "recommended_set",
    "slice_1d",
    "slice_2d",
]
