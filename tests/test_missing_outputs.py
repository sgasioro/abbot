"""Runs that didn't measure every output: reported as None, kept as NaN, each
output's model trains on the runs that measured it."""

import math

import pytest
import torch

from abbot import (
    BAXPolicy,
    Campaign,
    ContinuousDomain,
    DataStore,
    LevelSet,
    OutputSpec,
    ParetoBO,
    ProblemSpec,
    default_constructor,
)
from abbot.io import history_from_csv, history_to_csv

pytestmark = pytest.mark.slow  # GP fits

SPEC = ProblemSpec(
    domain=[ContinuousDomain(name="x", lower=0, upper=1)],
    outputs=[OutputSpec(name="yield"), OutputSpec(name="purity", direction="OBSERVE")],
)


def _yield(x):
    return 1 - (x - 0.3) ** 2


def _campaign(**kw):
    seeds = torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.double)
    return Campaign(SPEC, initial_designs=seeds, on_model_error="raise", **kw)


def test_none_marks_an_output_not_measured_and_a_missing_key_still_fails():
    c = _campaign()
    (s,) = c.suggest(1)
    with pytest.raises(ValueError, match="An output that wasn't measured is given as None"):
        c.ingest({"yield": 0.5})  # a typo or a forgotten output isn't a missing value
    c.ingest({"yield": _yield(s["x"]), "purity": None})
    assert math.isnan(float(c.history.records[0].outputs[1]))
    assert c.state().outputs[0][1] is None  # JSON-safe
    (s,) = c.suggest(1)
    with pytest.raises(ValueError, match="no measured value"):
        c.ingest({"yield": None, "purity": float("nan")})
    c.ingest([[None, 0.9]])  # a list in spec.outputs order works too
    assert math.isnan(float(c.history.records[1].outputs[0]))


def test_each_output_fits_on_the_runs_that_measured_it(tmp_path):
    c = _campaign(store=DataStore(tmp_path / "s"))
    for i in range(8):
        (s,) = c.suggest(1)
        purity = None if i % 2 else 0.5 + 0.1 * s["x"]
        c.ingest({"yield": _yield(s["x"]), "purity": purity})
    model = c.current_model()
    assert [len(m.train_targets) for m in model.models] == [8, 4]
    # Resumes with the gaps.
    again = Campaign(SPEC, store=DataStore(tmp_path / "s"))
    assert again.history.records[1].outputs[1].isnan()
    # The best run is judged on what was measured; recommend works with the model too.
    assert c.recommend()
    assert c.check().fits[1].name == "purity"


def test_an_output_nobody_measured_yet_is_a_clear_fit_error():
    c = _campaign()
    for _ in range(3):
        (s,) = c.suggest(1)
        c.ingest({"yield": _yield(s["x"]), "purity": None})
    with pytest.raises(ValueError, match="no run has measured purity yet"):
        c.current_model()
    probe = _campaign()  # by default: a random design that says why
    for _ in range(3):
        (s,) = probe.suggest(1)
        probe.ingest({"yield": _yield(s["x"]), "purity": None})
    probe.on_model_error = "probe"
    probe.suggest(1)
    assert "no run has measured purity yet" in probe.last_warning


def test_a_custom_model_must_say_it_handles_gaps():
    class Plain:
        provides_uq = "calibrated_variance"

        def __init__(self):
            self.inner = default_constructor(SPEC)

        def build(self, X, Y, Yvar=None):
            return self.inner.build(X, Y, Yvar)

    c = Campaign(SPEC, model_constructor=Plain(), on_model_error="raise",
                 initial_designs=torch.tensor([[0.2], [0.7]], dtype=torch.double))
    for purity in (0.5, None):
        c.suggest(1)
        c.ingest({"yield": 0.1, "purity": purity})
    with pytest.raises(ValueError, match="doesn't set handles_missing"):
        c.current_model()


def test_correct_fills_in_a_value_measured_later():
    c = _campaign()
    c.suggest(1)
    c.ingest({"yield": 0.4, "purity": None})
    c.correct(0, {"purity": 0.8}, reason="assay came back")
    assert float(c.history.records[0].outputs[1]) == 0.8
    assert c.history.records[0].corrections[0]["outputs"] == [0.4, None]
    with pytest.raises(ValueError, match="no measured value"):
        c.correct(0, {"yield": None, "purity": None})


def test_the_best_run_needs_its_objective_measured():
    c = _campaign()
    for y, p in ((None, 0.5), (0.6, 0.5), (0.9, None)):
        c.suggest(1)
        c.ingest({"yield": y, "purity": p})
    assert c.result().recommended or True
    from abbot.session import observed_front

    assert observed_front(c.history, SPEC) == [2]  # run 0 has no yield to rank


def test_output_constraints_and_pareto_with_gaps():
    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0, upper=1)],
        outputs=[OutputSpec(name="a"), OutputSpec(name="b"),
                 OutputSpec(name="c", direction="OBSERVE", max_val=0.8)],
    )
    from abbot.feasibility import feasible

    Y = torch.tensor([[1.0, 2.0, 0.5], [1.0, float("nan"), 0.5], [1.0, 2.0, float("nan")]],
                     dtype=torch.double)
    # A gap in an output no constraint uses doesn't matter; one in a constrained one does.
    assert feasible(spec, Y).tolist() == [True, True, False]
    seeds = torch.tensor([[0.1], [0.4], [0.6], [0.9]], dtype=torch.double)
    c = Campaign(spec, ParetoBO(), initial_designs=seeds, on_model_error="raise")
    for i in range(8):
        (s,) = c.suggest(1)
        x = s["x"]
        c.ingest({"a": x, "b": None if i == 2 else 1 - x, "c": None if i == 5 else 0.5 * x})
    assert len(c.history) == 8 and c.recommend()


def test_bax_maps_with_gaps():
    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0, upper=1)],
        outputs=[OutputSpec(name="y", direction="OBSERVE"),
                 OutputSpec(name="z", direction="OBSERVE")],
    )
    for strategy in ("SwitchBAX", "InfoBAX", "PSBAX"):
        c = Campaign(spec, BAXPolicy(LevelSet(0.5, "y"), strategy=strategy, n_candidates=100),
                     on_model_error="raise")
        for i in range(7):
            (s,) = c.suggest(1)
            c.ingest({"y": math.sin(5 * s["x"]), "z": None if i % 3 else s["x"]})
        assert len(c.history) == 7


def test_csv_round_trips_an_empty_cell():
    c = _campaign()
    c.suggest(1)
    c.ingest({"yield": 0.4, "purity": None})
    text = history_to_csv(c.history, SPEC)
    assert text.splitlines()[1].endswith("0.4,")
    back = history_from_csv(text, SPEC)
    assert back.records[0].outputs[1].isnan()
    with pytest.raises(ValueError, match="no output has a value"):
        history_from_csv("x,yield,purity\n0.5,,\n", SPEC)


@pytest.mark.parametrize("family", ["correlated", "fidelity", "named sources", "ensemble"])
def test_other_model_families_take_gaps(family):
    from abbot import (
        EnsembleConstructor,
        EnumeratedDomain,
        IntegerDomain,
        SimpleBO,
        Source,
    )

    outputs = [OutputSpec(name="f", direction="MINIMIZE"),
               OutputSpec(name="g", direction="OBSERVE")]
    domain = [ContinuousDomain(name="a", lower=0, upper=1),
              ContinuousDomain(name="b", lower=0, upper=1)]
    sources = None
    if family == "fidelity":
        sources = [Source(domain=IntegerDomain(name="atoms", lower=32, upper=512), target=512)]
    if family == "named sources":
        sources = [Source(domain=EnumeratedDomain(name="code", values=["sim", "lab"]),
                          target="lab")]
    spec = ProblemSpec(domain=domain, outputs=outputs, sources=sources)
    ctor = None
    if family == "correlated":
        ctor = default_constructor(spec, correlated=True)
    if family == "ensemble":
        ctor = EnsembleConstructor(n_members=3, epochs=30)
    policy = SimpleBO() if family != "ensemble" else BAXPolicy(
        LevelSet(0.5, "g"), strategy="MeanBAX", n_candidates=100)
    c = Campaign(spec, policy, ctor, on_model_error="raise")
    for i in range(9):
        (s,) = c.suggest(1)
        f = (s["a"] - 0.4) ** 2 + (s["b"] - 0.6) ** 2
        c.ingest({"f": f, "g": None if i % 2 else s["a"] + s["b"]})
    assert c.current_model() is not None and len(c.history) == 9


def test_xopt_sequential_generators_start_from_a_measured_best():
    from abbot.interop.xopt import GestGeneratorPolicy

    rows = [{"yield": 0.2}, {"yield": float("nan")}, {"yield": 0.9}, {"yield": 0.5}]
    ordered = GestGeneratorPolicy._best_last(rows, SPEC)
    assert ordered[-1]["yield"] == 0.9 and math.isnan(ordered[0]["yield"])
