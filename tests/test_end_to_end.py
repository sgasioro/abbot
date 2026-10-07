"""End-to-end integration: the M1 engine driving the M3 lab-BO flows and the M2
SBI aux seam, through run_campaign.

Goal of this file is to surface integration breakage across the real seams a
shippable lab tool exercises:
  - continuous lab-BO via ManualExperiment (human-in-the-loop, injected responder)
  - enumerated-discrete design space via SimpleBO(DiscreteArgmax)
  - the composite-SBI (outputs, aux) evaluate seam: aux persisted and resumed
"""

import torch

from abbot.acq_opt import DiscreteArgmax
from abbot.experiment import ManualExperiment
from abbot.history import DataStore
from abbot.loop import run_campaign
from abbot.model import SingleTaskGPConstructor
from abbot.policy import SimpleBO
from abbot.problem import ContinuousDomain, EnumeratedDomain, ObjectiveSpec, ProblemSpec

# --- M3: continuous lab-BO through a human-in-the-loop Experiment -------------


def test_lab_bo_continuous_via_manual_experiment(tmp_path):
    """A non-expert 'enters measurements': the responder stands in for a human.

    Objective peaks at x=0.7 in [0, 1]; the responder returns the measured value.
    """
    torch.manual_seed(0)
    opt = 0.7

    def responder(candidate: torch.Tensor) -> torch.Tensor:
        return -(candidate.reshape(-1) - opt) ** 2

    spec = ProblemSpec(
        domain=[ContinuousDomain(name="temp", lower=0.0, upper=1.0)],
        objectives=[ObjectiveSpec(name="yield", direction="MAXIMIZE")],
    )
    experiment = ManualExperiment(responder=responder, variable_names=["temp"])
    seeds = torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.double)

    result = run_campaign(
        experiment=experiment,
        spec=spec,
        policy=SimpleBO(),
        model_constructor=SingleTaskGPConstructor(bounds=spec.bounds_tensor()),
        evaluate=lambda obs: obs,
        initial_designs=seeds,
        n_shots=12,
        store=DataStore(tmp_path),
    )
    assert len(result.history) == 3 + 12
    assert abs(float(result.best.candidate) - opt) < 0.1


# --- M3: enumerated-discrete design space -------------------------------------


def test_lab_bo_enumerated_discrete():
    """Discrete catalyst choice: proposal must enumerate, not run optimize_acqf.

    Best value is at 4.0 among {0,1,2,3,4,5}.
    """
    torch.manual_seed(0)
    best_choice = 4.0

    def responder(candidate: torch.Tensor) -> torch.Tensor:
        return -(candidate.reshape(-1) - best_choice) ** 2

    spec = ProblemSpec(
        domain=[EnumeratedDomain(name="catalyst", values=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0])],
        objectives=[ObjectiveSpec(name="yield", direction="MAXIMIZE")],
    )
    experiment = ManualExperiment(responder=responder)
    seeds = torch.tensor([[0.0], [5.0]], dtype=torch.double)

    result = run_campaign(
        experiment=experiment,
        spec=spec,
        policy=SimpleBO(optimizer=DiscreteArgmax()),
        model_constructor=SingleTaskGPConstructor(bounds=spec.bounds_tensor()),
        evaluate=lambda obs: obs,
        initial_designs=seeds,
        n_shots=8,
    )
    # Every proposed point must be one of the allowed values.
    best_x = result.best.candidate
    assert float(best_x) in {0.0, 1.0, 2.0, 3.0, 4.0, 5.0}
    assert abs(float(best_x) - best_choice) <= 1.0


# --- M2 seam: composite (outputs, aux) is persisted and resumed ---------------


def test_sbi_aux_payload_persisted_and_resumed(tmp_path):
    """The evaluate seam can return (outputs, aux); aux round-trips through the
    DataStore. Proves the loop carries the SBI shape without any SBI code.
    """
    torch.manual_seed(0)

    def evaluate_with_aux(obs: torch.Tensor):
        f = obs.reshape(-1)
        fake_theta = torch.randn(16, 2, dtype=torch.double)  # cached posterior samples
        return f, fake_theta

    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        objectives=[ObjectiveSpec(name="F", direction="MAXIMIZE")],
    )

    def responder(candidate: torch.Tensor) -> torch.Tensor:
        return -(candidate.reshape(-1) - 0.4) ** 2

    kw = {
        "experiment": ManualExperiment(responder=responder),
        "spec": spec,
        "policy": SimpleBO(),
        "model_constructor": SingleTaskGPConstructor(bounds=spec.bounds_tensor()),
        "evaluate": evaluate_with_aux,
        "initial_designs": torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.double),
        "store": DataStore(tmp_path),
    }
    run_campaign(n_shots=3, **kw)

    # Reload from disk: every record must carry its aux payload.
    reloaded = DataStore(tmp_path).load()
    assert len(reloaded) == 6
    for r in reloaded.records:
        assert r.aux is not None
        assert r.aux.shape == (16, 2)

    # And a resumed run sees them.
    result = run_campaign(n_shots=2, **kw)
    assert len(result.history) == 6 + 2
    assert result.history.records[0].aux is not None


# --- Known gap, surfaced loudly: mixed continuous + enumerated ----------------


def test_mixed_domain_continuous_optimizer_fails_loudly():
    """A mixed space must NOT silently relax the enumerated axis to a float range.

    This is the documented deferred case (optimize_acqf_mixed). We want a clear
    error now, not a wrong answer.
    """
    import pytest

    from abbot.acq_opt import Continuous

    spec = ProblemSpec(
        domain=[
            ContinuousDomain(name="temp", lower=0.0, upper=1.0),
            EnumeratedDomain(name="catalyst", values=[0.0, 1.0, 2.0]),
        ],
        objectives=[ObjectiveSpec(name="yield", direction="MAXIMIZE")],
    )

    # The loop swallows proposal errors and falls back to random, so assert the
    # optimizer itself refuses the mixed spec directly.
    with pytest.raises(NotImplementedError, match="all-continuous"):
        Continuous().optimize(acq=None, spec=spec, n=1)
