"""Experiment.run takes a {name: value} design dict; outputs may come back by name."""

import warnings

import pytest
import torch

from abbot import (
    Campaign,
    ContinuousDomain,
    EnumeratedDomain,
    ManualExperiment,
    OutputSpec,
    ProblemSpec,
    TensorExperiment,
)
from abbot.experiment import as_design_experiment


def _spec() -> ProblemSpec:
    return ProblemSpec(
        domain=[
            ContinuousDomain(name="temp", lower=20.0, upper=80.0),
            EnumeratedDomain(name="catalyst", values=[0.0, 1.0, 2.0]),
        ],
        outputs=[
            OutputSpec(name="yield", direction="MAXIMIZE"),
            OutputSpec(name="purity", direction="OBSERVE"),
        ],
    )


def _seeded() -> Campaign:
    from abbot import SimpleBO
    from abbot.acq_opt import DiscreteArgmax

    # Mixed space has no default policy; seeds alone never reach the optimizer.
    c = Campaign(_spec(), SimpleBO(DiscreteArgmax()))
    c.seed(torch.tensor([[30.0, 2.0], [50.0, 0.0]], dtype=torch.double))
    return c


class _Recorder:
    def __init__(self, result):
        self.result = result
        self.designs: list = []

    def run(self, design):
        self.designs.append(design)
        return self.result


def _identity(obs):
    return obs


def test_run_receives_named_design_in_spec_order():
    c = _seeded()
    exp = _Recorder([1.0, 2.0])
    c.step(exp, _identity)
    (design,) = exp.designs
    assert design == {"temp": 30.0, "catalyst": 2.0}
    assert list(design) == c.spec.variable_names


def test_outputs_by_name_any_order():
    c = _seeded()
    c.step(_Recorder({"purity": 0.9, "yield": 42.0}), _identity)
    assert c.history.records[0].outputs.tolist() == [42.0, 0.9]


def test_outputs_dict_with_wrong_keys_is_a_clear_error():
    c = _seeded()
    with pytest.raises(ValueError, match="missing \\['purity'\\]"):
        c.step(_Recorder({"yield": 42.0}), _identity)


def test_ingest_and_record_accept_dicts():
    c = _seeded()
    c.suggest(1)
    c.ingest({"yield": 1.0, "purity": 0.5})
    c.record({"catalyst": 1.0, "temp": 60.0}, {"purity": 0.7, "yield": 3.0})
    rec = c.history.records[1]
    assert rec.candidate.tolist() == [60.0, 1.0]
    assert rec.outputs.tolist() == [3.0, 0.7]


def test_manual_experiment_responder_gets_the_design():
    seen = []
    exp = ManualExperiment(responder=lambda d: seen.append(d) or {"yield": 1.0, "purity": 0.0})
    c = _seeded()
    c.step(exp, _identity)
    assert seen == [{"temp": 30.0, "catalyst": 2.0}]


class _LegacyTensorExperiment:
    def __init__(self):
        self.seen: list[torch.Tensor] = []

    def run(self, candidate: torch.Tensor) -> torch.Tensor:
        self.seen.append(candidate)
        return torch.tensor([candidate[0] / 10.0, 0.0])


def test_tensor_annotated_experiment_is_adapted_with_one_warning():
    c = _seeded()
    exp = _LegacyTensorExperiment()
    with pytest.warns(FutureWarning, match="TensorExperiment"):
        c.step(exp, _identity)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # warned once per class, not every step
        c.step(exp, _identity)
    assert [x.tolist() for x in exp.seen] == [[30.0, 2.0], [50.0, 0.0]]
    assert c.history.records[0].outputs.tolist() == [3.0, 0.0]


def test_explicit_tensor_experiment_does_not_warn():
    exp = TensorExperiment(_LegacyTensorExperiment())
    assert as_design_experiment(exp) is exp
    c = _seeded()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c.step(exp, _identity)


def test_unannotated_tensor_code_fails_with_a_hint():
    class _Old:
        def run(self, candidate):
            return candidate.reshape(-1)

    c = _seeded()
    c.step(_Old(), _identity)
    assert "TensorExperiment" in c.history.records[0].error
