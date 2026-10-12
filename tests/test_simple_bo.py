"""SimpleBO on a SingleTaskGP recovers a known optimum (M1 verify), plus the
loop driver and the uq-pairing guardrail.
"""

import pytest
import torch

from abbot.experiment import Experiment
from abbot.history import DataStore
from abbot.loop import check_uq_pairing, run_campaign
from abbot.model import SingleTaskGPConstructor
from abbot.policy import SimpleBO, UncertaintySampling
from abbot.problem import ContinuousDomain, ObjectiveSpec, ProblemSpec

pytestmark = pytest.mark.slow  # GP fits + optimize_acqf

# A smooth 1-D objective maximized at x=0.3, in [0, 1].
_OPT = 0.3


class _NegQuadratic(Experiment):
    def run(self, candidate: torch.Tensor) -> torch.Tensor:
        return -(candidate - _OPT) ** 2


def _evaluate(obs: torch.Tensor) -> torch.Tensor:
    return obs.reshape(-1)


def _spec() -> ProblemSpec:
    return ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        objectives=[ObjectiveSpec(name="f", direction="MAXIMIZE")],
    )


def test_simple_bo_recovers_optimum(tmp_path):
    torch.manual_seed(0)
    spec = _spec()
    bounds = spec.bounds_tensor()
    seeds = torch.tensor([[0.05], [0.5], [0.95]], dtype=torch.double)

    result = run_campaign(
        experiment=_NegQuadratic(),
        spec=spec,
        policy=SimpleBO(),
        model_constructor=SingleTaskGPConstructor(bounds=bounds),
        evaluate=_evaluate,
        initial_designs=seeds,
        n_shots=12,
        store=DataStore(tmp_path),
    )
    assert len(result.history) == len(seeds) + 12
    best_x = result.best.candidate
    assert abs(float(best_x) - _OPT) < 0.1, f"best_x={float(best_x)} not near {_OPT}"


def test_resume_from_store(tmp_path):
    torch.manual_seed(0)
    spec = _spec()
    bounds = spec.bounds_tensor()
    seeds = torch.tensor([[0.05], [0.5], [0.95]], dtype=torch.double)
    kw = {
        "experiment": _NegQuadratic(),
        "spec": spec,
        "policy": SimpleBO(),
        "model_constructor": SingleTaskGPConstructor(bounds=bounds),
        "evaluate": _evaluate,
        "initial_designs": seeds,
        "store": DataStore(tmp_path),
    }
    run_campaign(n_shots=4, **kw)
    result = run_campaign(n_shots=3, **kw)
    # Resume picks up the 7 prior records and adds 3 more.
    assert len(result.history) == 7 + 3


def test_uq_pairing_guardrail():
    # A point-only model paired with a variance-hungry policy is refused.
    import pytest

    with pytest.raises(ValueError, match="calibrated uncertainty"):
        check_uq_pairing(UncertaintySampling(), model_uq="point")
    # And allowed when the model provides enough.
    check_uq_pairing(UncertaintySampling(), model_uq="calibrated_variance")
    check_uq_pairing(SimpleBO(), model_uq="calibrated_variance")
