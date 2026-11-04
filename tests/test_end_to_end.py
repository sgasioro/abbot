"""End-to-end integration: the engine driving the lab-BO flows and the
SBI aux seam, through run_campaign.

Goal of this file is to surface integration breakage across the real seams a
shippable lab tool exercises:
  - continuous lab-BO via ManualExperiment (human-in-the-loop, injected responder)
  - enumerated-discrete design space via SimpleBO(DiscreteArgmax)
  - the composite-SBI (outputs, aux) evaluate seam: aux persisted and resumed
"""

import pytest
import torch

from abbot.acq_opt import DiscreteArgmax
from abbot.experiment import ManualExperiment
from abbot.history import DataStore
from abbot.loop import run_campaign
from abbot.model import SingleTaskGPConstructor
from abbot.policy import SimpleBO
from abbot.problem import ContinuousDomain, EnumeratedDomain, ObjectiveSpec, ProblemSpec
from abbot.session import Campaign

pytestmark = pytest.mark.slow  # run_campaign GP fits

# --- continuous lab-BO through a human-in-the-loop Experiment -------------


def test_lab_bo_continuous_via_manual_experiment(tmp_path):
    """A non-expert 'enters measurements': the responder stands in for a human.

    Objective peaks at x=0.7 in [0, 1]; the responder returns the measured value.
    """
    torch.manual_seed(0)
    opt = 0.7

    def responder(design: dict) -> float:
        return -((design["temp"] - opt) ** 2)

    spec = ProblemSpec(
        domain=[ContinuousDomain(name="temp", lower=0.0, upper=1.0)],
        objectives=[ObjectiveSpec(name="yield", direction="MAXIMIZE")],
    )
    experiment = ManualExperiment(responder=responder)
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


# --- enumerated-discrete design space -----------------------------------------


def test_lab_bo_enumerated_discrete():
    """Discrete catalyst choice: proposal must enumerate, not run optimize_acqf.

    Best value is at 4.0 among {0,1,2,3,4,5}.
    """
    torch.manual_seed(0)
    best_choice = 4.0

    def responder(design: dict) -> float:
        return -((design["catalyst"] - best_choice) ** 2)

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


# --- SBI seam: composite (outputs, aux) is persisted and resumed ---------------


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

    def responder(design: dict) -> float:
        return -((design["x"] - 0.4) ** 2)

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

    # Rerunning the same call finds the budget spent; a bigger one resumes onto it.
    assert len(run_campaign(n_shots=3, **kw).history) == 6
    result = run_campaign(n_shots=5, **kw)
    assert len(result.history) == 3 + 5
    assert result.history.records[0].aux is not None


# --- Mixed continuous + enumerated -------------------------------------------


def test_mixed_domain_proposes_allowed_values():
    """A mixed space enumerates the enumerated axis (``optimize_acqf_mixed``): the
    proposal lands on an allowed value, never on a relaxed float in between."""
    from botorch.acquisition import PosteriorMean
    from botorch.models import SingleTaskGP

    from abbot.acq_opt import Continuous

    spec = ProblemSpec(
        domain=[
            ContinuousDomain(name="temp", lower=0.0, upper=1.0),
            EnumeratedDomain(name="catalyst", values=[0.0, 1.0, 2.0]),
        ],
        objectives=[ObjectiveSpec(name="yield", direction="MAXIMIZE")],
    )
    X = spec.sample_designs(12, seed=0)
    Y = -(X[:, :1] - 0.7) ** 2 - (X[:, 1:] - 1.0) ** 2
    acq = PosteriorMean(SingleTaskGP(X, Y))
    (x,) = Continuous(num_restarts=2, raw_samples=32).optimize(acq, spec, 1)
    assert x[1].item() in {0.0, 1.0, 2.0}

    # A batch goes through the default policy (sequential, pending-aware).
    torch.manual_seed(0)
    c = Campaign(spec)
    for row, y in zip(X, Y):
        c.record(row, y)
    picks = c.suggest(2)
    assert not c.last_warning  # proposed, not a random fallback
    assert all(p["catalyst"] in {0.0, 1.0, 2.0} for p in picks)
