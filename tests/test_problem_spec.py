"""ProblemSpec round-trips to JSON (M1 verify)."""

import torch

from abbot.problem import (
    Box,
    ContinuousDomain,
    EnumeratedDomain,
    InequalitySpec,
    ObjectiveSpec,
    ProblemSpec,
)


def _spec() -> ProblemSpec:
    return ProblemSpec(
        domain=[
            ContinuousDomain(name="temp", lower=20.0, upper=80.0),
            EnumeratedDomain(name="catalyst", values=[0.0, 1.0, 2.0]),
        ],
        objectives=[ObjectiveSpec(name="yield", direction="MAXIMIZE")],
        boxes=[Box(lower=[0.0, 0.0], upper=[1.0, 2.0])],
        constraints=[InequalitySpec(indices=[0], coeffs=[1.0], rhs=25.0)],
    )


def test_json_round_trip():
    spec = _spec()
    restored = ProblemSpec.model_validate_json(spec.model_dump_json())
    assert restored == spec
    assert restored.variable_names == ["temp", "catalyst"]


def test_json_schema_exports():
    # Frontend-ready: a JSON schema can be produced for browser forms.
    schema = ProblemSpec.model_json_schema()
    assert "properties" in schema


def test_discriminated_domain_types_survive():
    spec = _spec()
    restored = ProblemSpec.model_validate_json(spec.model_dump_json())
    assert isinstance(restored.domain[0], ContinuousDomain)
    assert isinstance(restored.domain[1], EnumeratedDomain)


def test_bounds_tensor():
    spec = _spec()
    b = spec.bounds_tensor()
    assert b.shape == (2, 2)
    assert torch.allclose(b[0], torch.tensor([20.0, 0.0], dtype=torch.double))
    assert torch.allclose(b[1], torch.tensor([80.0, 2.0], dtype=torch.double))


def test_inequality_constraints_form():
    spec = _spec()
    cons = spec.inequality_constraints()
    assert cons is not None
    idx, _coef, rhs = cons[0]
    assert idx.dtype == torch.long
    assert rhs == 25.0
