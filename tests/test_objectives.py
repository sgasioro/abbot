"""Objective direction (maximize-canonical, no caller negation) and native
multi-objective (ParetoBO) behavior.
"""

import logging

import pytest
import torch

from abbot.experiment import ManualExperiment
from abbot.loop import run_campaign
from abbot.model import (
    ModelListConstructor,
    SingleTaskGPConstructor,
    default_constructor,
)
from abbot.policy import ParetoBO, SimpleBO
from abbot.problem import ContinuousDomain, ObjectiveSpec, ProblemSpec

pytestmark = pytest.mark.slow  # run_campaign / ParetoBO GP fits

# --- objective_signs ----------------------------------------------------------


def test_objective_signs():
    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        objectives=[
            ObjectiveSpec(name="yield", direction="MAXIMIZE"),
            ObjectiveSpec(name="cost", direction="MINIMIZE"),
        ],
    )
    assert torch.equal(spec.objective_signs(), torch.tensor([1.0, -1.0], dtype=torch.double))
    assert spec.n_objectives == 2


# --- MINIMIZE without caller negation -----------------------------------------


def test_minimize_recovers_minimum_with_raw_outputs():
    """A MINIMIZE objective: evaluate() returns RAW cost (positive, minimized at
    x=0.3). The engine must drive it DOWN without the caller negating anything.
    """
    torch.manual_seed(0)
    minimizer = 0.3

    def responder(candidate: torch.Tensor) -> torch.Tensor:
        # raw physical cost: a positive parabola with its floor at `minimizer`
        return (candidate.reshape(-1) - minimizer) ** 2 + 0.5

    spec = ProblemSpec(
        domain=[ContinuousDomain(name="temp", lower=0.0, upper=1.0)],
        objectives=[ObjectiveSpec(name="cost", direction="MINIMIZE")],
    )
    result = run_campaign(
        experiment=ManualExperiment(responder=responder),
        spec=spec,
        policy=SimpleBO(),
        model_constructor=SingleTaskGPConstructor(bounds=spec.bounds_tensor()),
        evaluate=lambda obs: obs,
        initial_designs=torch.tensor([[0.1], [0.6], [0.9]], dtype=torch.double),
        n_shots=12,
    )
    best = result.best
    assert abs(float(best.candidate) - minimizer) < 0.12
    # Reported outputs are RAW units (positive cost), not the internal negation.
    assert float(best.outputs) > 0.0


# --- multi-objective ParetoBO -------------------------------------------------


def _biobjective_spec() -> ProblemSpec:
    # Two competing objectives over x in [0, 1]: maximize f1=x, maximize f2=1-x.
    # The entire interval is Pareto-optimal (a pure trade-off).
    return ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        objectives=[
            ObjectiveSpec(name="f1", direction="MAXIMIZE"),
            ObjectiveSpec(name="f2", direction="MAXIMIZE"),
        ],
    )


def test_pareto_bo_returns_non_dominated_set():
    torch.manual_seed(0)

    def responder(candidate: torch.Tensor) -> torch.Tensor:
        x = candidate.reshape(-1)
        return torch.cat([x, 1.0 - x])  # (f1, f2)

    spec = _biobjective_spec()
    result = run_campaign(
        experiment=ManualExperiment(responder=responder),
        spec=spec,
        policy=ParetoBO(),
        model_constructor=default_constructor(spec, bounds=spec.bounds_tensor()),
        evaluate=lambda obs: obs,
        initial_designs=torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.double),
        n_shots=10,
    )
    rec = result.recommended
    assert len(rec) >= 2, "a trade-off problem should hand back a multi-point front"
    # Recommended points are mutually non-dominated in RAW objective space.
    Y = torch.stack([result.history.records[i].outputs for i in rec])
    for i in range(len(Y)):
        for j in range(len(Y)):
            if i == j:
                continue
            # no point dominates another (>= on all, > on some)
            dominates = bool((Y[j] >= Y[i]).all() and (Y[j] > Y[i]).any())
            assert not dominates, "recommended set must be mutually non-dominated"
    # .best must refuse to collapse a true front.
    with pytest.raises(ValueError, match="single recommended design"):
        _ = result.best


def test_pareto_bo_mixed_directions():
    """One MAXIMIZE + one MINIMIZE: sign handling is per-objective. f1=x (max),
    cost=x (min) are perfectly opposed -> again a pure trade-off.
    """
    torch.manual_seed(0)

    def responder(candidate: torch.Tensor) -> torch.Tensor:
        x = candidate.reshape(-1)
        return torch.cat([x, x])  # (f1 to maximize, cost to minimize)

    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        objectives=[
            ObjectiveSpec(name="f1", direction="MAXIMIZE"),
            ObjectiveSpec(name="cost", direction="MINIMIZE"),
        ],
    )
    result = run_campaign(
        experiment=ManualExperiment(responder=responder),
        spec=spec,
        policy=ParetoBO(),
        model_constructor=ModelListConstructor(bounds=spec.bounds_tensor()),
        evaluate=lambda obs: obs,
        initial_designs=torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.double),
        n_shots=10,
    )
    assert len(result.recommended) >= 2


# --- guardrails ---------------------------------------------------------------


def test_simple_bo_rejects_multi_objective():
    spec = _biobjective_spec()
    model = default_constructor(spec, bounds=spec.bounds_tensor())
    # Build with two seed points so suggest reaches the guard.
    from abbot.history import History, Record

    hist = History()
    for xv in (0.2, 0.8):
        x = torch.tensor([xv], dtype=torch.double)
        hist.append(Record(candidate=x, observation=None, outputs=torch.tensor([xv, 1 - xv])))
    fitted = model.build(hist.designs, hist.outputs)
    with pytest.raises(ValueError, match="single-objective"):
        SimpleBO().suggest(hist, spec, fitted, n=1)


def test_wrong_output_width_fails_clearly():
    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        objectives=[ObjectiveSpec(name="f", direction="MAXIMIZE")],
    )
    # evaluate returns 2 columns but spec declares 1.
    with pytest.raises(ValueError, match="objective"):
        run_campaign(
            experiment=ManualExperiment(responder=lambda c: torch.tensor([1.0, 2.0])),
            spec=spec,
            policy=SimpleBO(),
            model_constructor=SingleTaskGPConstructor(bounds=spec.bounds_tensor()),
            evaluate=lambda obs: obs,
            initial_designs=torch.tensor([[0.5]], dtype=torch.double),
            n_shots=0,
        )


def test_pareto_bo_cold_start_falls_back_without_nehvi(caplog):
    """Below the cold-start floor, ParetoBO must NOT invoke NEHVI; it falls back to
    uncertainty sampling and logs that it did (no silent undirected proposal).
    """
    torch.manual_seed(0)
    spec = _biobjective_spec()
    model = default_constructor(spec, bounds=spec.bounds_tensor())

    from abbot.history import History, Record

    # Only ONE clean point -> below floor = max(2, 2*m) = 4.
    hist = History()
    x = torch.tensor([0.3], dtype=torch.double)
    hist.append(Record(candidate=x, observation=None, outputs=torch.tensor([0.3, 0.7])))
    fitted = model.build(hist.designs, hist.outputs * spec.objective_signs())

    policy = ParetoBO()
    with caplog.at_level(logging.INFO, logger="abbot.policy"):
        out = policy.suggest(hist, spec, fitted, n=1)
    assert out.shape[-1] == 1  # a candidate was still produced
    assert any("clean points" in r.message for r in caplog.records), (
        "cold-start fallback should be logged"
    )
