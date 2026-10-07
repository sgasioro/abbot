"""Live-analysis built-ins, computed from a campaign's history + fitted model.

These are pure functions of ``(History, spec[, model])`` so they are unit-testable
without a UI and reusable by a notebook or an agent. The UI registers them through
``Campaign``'s ``on_step`` hook and renders the results; callers can register their
own analyses the same way. Kept deliberately small — the general-purpose "any
analysis" surface is the hook itself; these are just the common basics.
"""

from __future__ import annotations

import torch

from abbot.history import History
from abbot.problem import ProblemSpec
from abbot.session import canonical_Y, observed_front


def convergence(history: History, spec: ProblemSpec) -> list[float]:
    """Running best objective value vs. step, in RAW units (single-objective only).

    Element ``i`` is the best clean observed objective among the first ``i+1``
    records, respecting direction (max of raw for MAXIMIZE, min for MINIMIZE).
    Faulted records carry forward the previous best.
    """
    if spec.n_objectives != 1:
        raise ValueError("convergence is single-objective; use hypervolume for MO.")
    maximize = spec.objectives[0].direction == "MAXIMIZE"
    best = None
    out: list[float] = []
    for r in history.records:
        if not r.fault and r.outputs is not None:
            v = float(r.outputs.reshape(-1)[0])
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
    per-step list; steps before any clean point are 0.0.
    """
    if spec.n_objectives < 2:
        raise ValueError("hypervolume needs >=2 objectives; use convergence for 1.")
    from botorch.utils.multi_objective.hypervolume import (
        Hypervolume,
        infer_reference_point,
    )
    from botorch.utils.multi_objective.pareto import is_non_dominated

    signs = spec.objective_signs()
    clean = [r for r in history.records if not r.fault and r.outputs is not None]
    if not clean:
        return [0.0] * len(history.records)
    all_Y = torch.stack([r.outputs.reshape(-1) for r in clean]).double() * signs
    ref = infer_reference_point(all_Y[is_non_dominated(all_Y)])
    hv = Hypervolume(ref_point=ref)

    out: list[float] = []
    acc: list[torch.Tensor] = []
    for r in history.records:
        if not r.fault and r.outputs is not None:
            acc.append(r.outputs.reshape(-1).double() * signs)
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


__all__ = ["canonical_Y", "convergence", "hypervolume", "recommended_set"]
