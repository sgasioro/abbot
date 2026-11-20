"""``Campaign(device="cuda")``: models fit and policies propose on the GPU, while the
history, the store, candidates and state stay on the CPU.

The tests marked ``cuda`` are skipped without a GPU; the two that check the device
argument itself run everywhere. GPU numbers aren't compared to CPU numbers (seeds
and reductions differ across devices); the tests check behavior: where tensors
live, shapes, dtypes, and that the search still finds what it should. Campaigns use ``on_model_error="raise"`` so a device
mix-up fails the test instead of turning into a random probe.
"""

import dataclasses
import itertools
import math

import pytest
import torch

from abbot import (
    BAXPolicy,
    Campaign,
    ContinuousDomain,
    DataStore,
    EnsembleConstructor,
    EnumeratedDomain,
    IntegerDomain,
    LevelSet,
    OutputSpec,
    ParetoBO,
    ProblemSpec,
    SimpleBO,
    Source,
    TopK,
    UncertaintySampling,
    default_constructor,
    run_campaign,
)
from abbot.acq_opt import DiscreteArgmax
from abbot.model import model_device
from abbot.supervision import per_shot

# -- helpers ----------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_swallowed_errors(caplog):
    """Some failures are logged and survived (a fantasy that can't be conditioned, a
    recommendation that falls back, a usability fit); on the GPU they would hide a
    device mix-up, so any of them fails the test."""
    import logging

    caplog.set_level(logging.INFO, logger="abbot")
    yield
    bad = [r for r in caplog.records
           if r.levelno >= logging.ERROR or "can't condition" in r.getMessage()]
    assert not bad, [f"{r.name}: {r.getMessage()} {r.exc_text or ''}" for r in bad]


def _on_gpu(model) -> bool:
    return model_device(model).type == "cuda"


def _cpu_tensors(obj, where: str) -> None:
    """Every tensor in ``obj`` (nested in lists, tuples, dicts) is a CPU tensor."""
    if isinstance(obj, torch.Tensor):
        assert obj.device.type == "cpu", f"{where} is on {obj.device}"
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _cpu_tensors(v, f"{where}[{k!r}]")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _cpu_tensors(v, f"{where}[{i}]")


def _assert_on_cpu(c: Campaign) -> None:
    """Nothing but the model and the policy holds device tensors: the records (also
    as saved), what's in flight and queued, the candidates and the state."""
    for i, r in enumerate(c.history.records):
        assert r.candidate.dtype == torch.double, f"record {i} candidate is {r.candidate.dtype}"
        if r.outputs is not None:
            assert r.outputs.dtype == torch.double
        for f in dataclasses.fields(r):
            _cpu_tensors(getattr(r, f.name), f"record {i}.{f.name}")
        assert r.acquisition is None or isinstance(r.acquisition, float)
    for item in c._inflight.values():
        assert item.x.device.type == "cpu" and item.x.dtype == torch.double
        assert item.acquisition is None or isinstance(item.acquisition, float)
    _cpu_tensors(c._seed_queue, "seed queue")
    _cpu_tensors(c._retry_queue, "retry queue")
    _cpu_tensors(c._halted_on, "halted on")
    for cand in c.outstanding():
        assert all(not isinstance(v, torch.Tensor) for v in cand.values()), cand
    c.state().to_json()  # plain JSON all the way down
    if c.store is not None:
        for i, r in enumerate(c.store.load().records):
            for f in dataclasses.fields(r):
                _cpu_tensors(getattr(r, f.name), f"saved record {i}.{f.name}")


def _bowl_spec(**kw):
    domain = kw.pop("domain", [
        ContinuousDomain(name="a", lower=-2.0, upper=2.0),
        ContinuousDomain(name="b", lower=-2.0, upper=2.0),
    ])
    outputs = kw.pop("outputs", [OutputSpec(name="f", direction="MINIMIZE")])
    return ProblemSpec(domain=domain, outputs=outputs, **kw)


class _Bowl:
    """``f = (a - 0.5)² + (b + 0.5)²``, best at (0.5, -0.5); extra outputs as asked."""

    def __init__(self, extra=None) -> None:
        self.extra = extra

    def run(self, design):
        a, b = float(design["a"]), float(design["b"])
        f = (a - 0.5) ** 2 + (b + 0.5) ** 2
        return [f, *(self.extra(a, b) if self.extra else [])]


def _gpu_campaign(spec, policy=None, model_constructor=None, **kw):
    kw.setdefault("on_model_error", "raise")
    c = Campaign(spec, policy, model_constructor, device="cuda", **kw)
    assert c.device.type == "cuda"
    return c


def _steps(c, experiment, k, n=1):
    for _ in range(k):
        c.step(experiment, n=n)
        _assert_on_cpu(c)


# -- the device setting -----------------------------------------------------------


@pytest.mark.cuda
def test_device_is_stored_and_the_default_stays_cpu():
    spec = _bowl_spec()
    assert Campaign(spec).device == torch.device("cpu")
    assert Campaign(spec, device=torch.device("cuda")).device.type == "cuda"
    assert Campaign(spec, device="cuda:0").device == torch.device("cuda:0")
    with pytest.raises(ValueError, match="CUDA device"):
        Campaign(spec, device=f"cuda:{torch.cuda.device_count()}")


@pytest.mark.parametrize("device", ["mps", "bogus"])
def test_unsupported_devices_are_refused(device):
    with pytest.raises(ValueError, match="cpu|cuda"):
        Campaign(_bowl_spec(), device=device)


def test_cuda_without_cuda_says_how_to_fix_it(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="Install a CUDA build of torch"):
        Campaign(_bowl_spec(), device="cuda")
    with pytest.raises(ValueError, match="Install a CUDA build of torch"):
        run_campaign(_bowl_spec(), _Bowl(), n_shots=1, device="cuda")
    assert Campaign(_bowl_spec()).device.type == "cpu"  # the default needs no GPU


# -- single and multi-objective BO -------------------------------------------------


@pytest.mark.cuda
def test_simple_bo_fits_and_proposes_on_the_gpu(tmp_path):
    torch.manual_seed(0)
    c = _gpu_campaign(_bowl_spec(), store=DataStore(tmp_path / "run"))
    _steps(c, _Bowl(), 12)
    model = c.current_model()
    assert _on_gpu(model)
    assert model.train_inputs[0].dtype == torch.double
    assert c.policy.last_acquisition_value is not None
    assert all(r.acquisition is not None for r in c.history.records[-5:])
    best = c.result().best.candidate
    assert best.device.type == "cpu"
    assert torch.allclose(best, torch.tensor([0.5, -0.5], dtype=torch.double), atol=0.35)


@pytest.mark.cuda
def test_batches_pending_and_fixed_inputs_on_the_gpu():
    torch.manual_seed(1)
    c = _gpu_campaign(_bowl_spec())
    _steps(c, _Bowl(), 6)
    first = c.suggest(2)  # outstanding, so the next proposal sees them as pending
    more = c.suggest(2, fixed={"a": 1.0})
    assert len(c.outstanding()) == 4 and all(m["a"] == 1.0 for m in more)
    _assert_on_cpu(c)
    c.ingest([_Bowl().run(d) for d in first + more])
    c.step(_Bowl(), n=3)
    _assert_on_cpu(c)


@pytest.mark.cuda
def test_run_campaign_takes_a_device():
    torch.manual_seed(2)
    result = run_campaign(_bowl_spec(), _Bowl(), n_shots=4, device="cuda")
    assert len(result.history) == 4 + 5
    assert result.designs.device.type == "cpu"


@pytest.mark.cuda
def test_pareto_bo_with_two_objectives_on_the_gpu():
    torch.manual_seed(3)
    spec = _bowl_spec(outputs=[
        OutputSpec(name="f", direction="MINIMIZE"),
        OutputSpec(name="g", direction="MINIMIZE"),
    ])
    c = _gpu_campaign(spec)
    _steps(c, _Bowl(lambda a, b: [(a + 0.5) ** 2 + (b - 0.5) ** 2]), 4, n=2)
    assert isinstance(c.policy, ParetoBO) and _on_gpu(c.current_model())
    front = c.recommend()
    assert front and all(0 <= i < len(c.history) for i in front)
    pinned = _gpu_campaign(spec, ParetoBO(ref_point=[-20.0, -20.0]), initial_designs=torch.empty(0, 2))
    pinned.load_history(c.history)
    pinned.step(_Bowl(lambda a, b: [(a + 0.5) ** 2 + (b - 0.5) ** 2]))
    assert pinned.history.records[-1].acquisition is not None  # proposed, not a probe
    _assert_on_cpu(pinned)


@pytest.mark.cuda
def test_uncertainty_sampling_and_cost_on_the_gpu():
    from abbot import CallableCost

    torch.manual_seed(4)
    c = _gpu_campaign(_bowl_spec(), UncertaintySampling())
    _steps(c, _Bowl(), 3, n=2)
    priced = _gpu_campaign(
        _bowl_spec(), SimpleBO(cost_model=CallableCost(lambda X: 1.0 + X[..., 0].abs()))
    )
    _steps(priced, _Bowl(), 3)


# -- BAX --------------------------------------------------------------------------


def _bax_spec():
    return _bowl_spec(outputs=[OutputSpec(name="f", direction="OBSERVE")])


@pytest.mark.cuda
@pytest.mark.parametrize("strategy", ["US", "MeanBAX", "SwitchBAX", "InfoBAX", "PSBAX"])
def test_bax_strategies_on_the_gpu(strategy):
    torch.manual_seed(5)
    policy = BAXPolicy(TopK(k=5, output="f", largest=False), strategy,
                       n_candidates=512, n_posterior_samples=8)
    c = _gpu_campaign(_bax_spec(), policy)
    _steps(c, _Bowl(), 3)
    _steps(c, _Bowl(), 2, n=3)  # greedy batches, conditioned on earlier picks
    first = c.suggest(2)
    picks = c.suggest(2)  # with two outstanding
    assert not {tuple(p.values()) for p in picks} & {tuple(p.values()) for p in first}
    _assert_on_cpu(c)
    assert policy.candidate_grid(c.spec).device.type == "cpu"  # the public grid stays put
    assert policy._on_device[1].device.type == "cuda"
    assert policy.last_acquisition_value.shape == (2,)
    # Every pick is a grid point not measured before.
    grid = policy.candidate_grid(c.spec)
    for r in c.history.records[5:]:
        assert torch.isclose(grid, r.candidate).all(-1).any()


@pytest.mark.cuda
def test_bax_with_fixed_inputs_noise_and_a_bad_region():
    torch.manual_seed(6)
    seen = []

    def noise_at(X):
        seen.append(X.device)
        return torch.full((len(X), 1), 1e-4, dtype=torch.double, device=X.device)

    policy = BAXPolicy(LevelSet(1.0, output="f"), "InfoBAX", n_candidates=256,
                       n_posterior_samples=6, noise_at=noise_at)
    c = _gpu_campaign(_bax_spec(), policy, supervisor=per_shot(lambda o: o[0] > 6.0, bad_region="far"))
    _steps(c, _Bowl(), 8)  # 5 seeds, then proposals
    assert seen and all(d.type == "cuda" for d in seen)
    assert c.current_usability() is not None and _on_gpu(c.current_usability().model)
    picks = c.suggest(1, fixed={"a": float(policy.candidate_grid(c.spec)[0, 0])})
    assert len(picks) == 1
    _assert_on_cpu(c)


@pytest.mark.cuda
def test_ensemble_model_with_meanbax_on_the_gpu():
    torch.manual_seed(7)
    policy = BAXPolicy(TopK(k=3, output="f", largest=False), "MeanBAX", n_candidates=128)
    c = _gpu_campaign(_bax_spec(), policy, EnsembleConstructor(n_members=1, epochs=50))
    _steps(c, _Bowl(), 3)
    assert _on_gpu(c.current_model())


# -- sources ----------------------------------------------------------------------


def _source_spec():
    return ProblemSpec(
        domain=[
            ContinuousDomain(name="a", lower=-2.0, upper=2.0),
            ContinuousDomain(name="b", lower=-2.0, upper=2.0),
        ],
        sources=[Source(domain=IntegerDomain(name="atoms", lower=32, upper=512),
                        target=512, log=True)],
        outputs=[OutputSpec(name="f", direction="MINIMIZE")],
    )


class _Cheap(_Bowl):
    """Fewer atoms: a biased bowl that costs less (seconds via ``cost``)."""

    def run(self, design):
        bias = 0.3 * (1 - float(design["atoms"]) / 512)
        return [_Bowl().run(design)[0] + bias]

    def cost(self, design):
        return float(design["atoms"]) / 512


@pytest.mark.cuda
def test_infobax_chooses_sources_on_the_gpu():
    torch.manual_seed(8)
    spec = _source_spec()
    policy = BAXPolicy(TopK(k=3, output="f", largest=False), "InfoBAX",
                       n_candidates=128, n_posterior_samples=6, source_levels=3)
    c = _gpu_campaign(spec, policy)
    assert type(c.model_constructor).__name__ == "MultiSourceGPConstructor"
    _steps(c, _Cheap(), 4)
    _steps(c, _Cheap(), 1, n=3)
    assert _on_gpu(c.current_model())
    assert len(policy.proposal_set(spec)) == 3 * len(policy.candidate_grid(spec))
    result = c.result()
    assert result.target_std and all(math.isfinite(s) for row in result.target_std for s in row)


@pytest.mark.cuda
def test_simple_bo_chooses_sources_on_the_gpu():
    torch.manual_seed(9)
    c = _gpu_campaign(_source_spec(), SimpleBO(n_candidates=256))
    _steps(c, _Cheap(), 3, n=2)
    assert _on_gpu(c.current_model())


@pytest.mark.cuda
def test_an_unordered_source_on_the_gpu():
    torch.manual_seed(10)
    spec = ProblemSpec(
        domain=[ContinuousDomain(name="a", lower=-2.0, upper=2.0),
                ContinuousDomain(name="b", lower=-2.0, upper=2.0)],
        sources=[Source(domain=EnumeratedDomain(name="code", values=["sim", "lab"]), target="lab")],
        outputs=[OutputSpec(name="f", direction="MINIMIZE")],
    )
    policy = BAXPolicy(TopK(k=3, output="f", largest=False), "InfoBAX",
                       n_candidates=128, n_posterior_samples=6)
    c = _gpu_campaign(spec, policy)
    _steps(c, _Bowl(), 4)
    assert _on_gpu(c.current_model())


# -- models -----------------------------------------------------------------------


@pytest.mark.cuda
def test_correlated_outputs_on_the_gpu():
    torch.manual_seed(11)
    spec = _bowl_spec(outputs=[OutputSpec(name="f", direction="MINIMIZE"),
                               OutputSpec(name="g", direction="OBSERVE")])
    c = _gpu_campaign(spec, model_constructor=default_constructor(spec, correlated=True))
    _steps(c, _Bowl(lambda a, b: [a + b]), 4)
    model = c.current_model()
    assert type(model).__name__ == "_OutputsMultiTaskGP" and _on_gpu(model)
    assert model.run_index.device.type == "cuda"
    report = c.check()
    assert {f.name for f in report.fits} == {"f", "g"}
    bax = _gpu_campaign(spec, BAXPolicy(TopK(k=3, output="f", largest=False), "InfoBAX",
                                        n_candidates=128, n_posterior_samples=6),
                        default_constructor(spec, correlated=True))
    _steps(bax, _Bowl(lambda a, b: [a + b]), 4)


@pytest.mark.cuda
def test_prior_mean_and_categories_on_the_gpu():
    torch.manual_seed(12)
    spec = _bowl_spec(domain=[
        ContinuousDomain(name="a", lower=-2.0, upper=2.0),
        ContinuousDomain(name="b", lower=-2.0, upper=2.0),
        EnumeratedDomain(name="glass", values=["crown", "flint"]),
    ])
    prior = default_constructor(spec, prior_mean={"f": lambda x: x["a"] ** 2 + x["b"] ** 2})
    c = _gpu_campaign(spec, model_constructor=prior)
    _steps(c, _Bowl(), 3)
    assert _on_gpu(c.current_model())


# -- input spaces -----------------------------------------------------------------


@pytest.mark.cuda
def test_enumerated_and_continuous_inputs_on_the_gpu():
    torch.manual_seed(13)
    spec = _bowl_spec(domain=[
        ContinuousDomain(name="a", lower=-2.0, upper=2.0),
        EnumeratedDomain(name="b", values=[-1.0, -0.5, 0.0, 0.5]),
        IntegerDomain(name="n", lower=1, upper=4),
    ])
    c = _gpu_campaign(spec)
    _steps(c, _Bowl(), 3, n=2)
    for r in c.history.records:
        assert float(r.candidate[1]) in (-1.0, -0.5, 0.0, 0.5)
        assert float(r.candidate[2]).is_integer()


@pytest.mark.cuda
def test_all_enumerated_inputs_on_the_gpu():
    torch.manual_seed(14)
    values = [-1.5, -0.5, 0.5, 1.5]
    spec = _bowl_spec(domain=[EnumeratedDomain(name="a", values=values),
                              EnumeratedDomain(name="b", values=values)])
    c = _gpu_campaign(spec)
    assert isinstance(c.policy.optimizer, DiscreteArgmax)
    _steps(c, _Bowl(), 3, n=2)
    choices = torch.tensor([[0.5, -0.5], [1.5, 0.5], [-0.5, -1.5]], dtype=torch.double)
    given = _gpu_campaign(spec, SimpleBO(DiscreteArgmax(choices)), initial_designs=torch.empty(0, 2))
    given.load_history(c.history)
    given.step(_Bowl())
    assert torch.isclose(choices, given.history.records[-1].candidate).all(-1).any()


@pytest.mark.cuda
def test_domain_constraints_on_the_gpu():
    from abbot import LinearConstraint

    torch.manual_seed(15)
    spec = _bowl_spec(domain_constraints=[LinearConstraint({"a": 1, "b": 1}, max_val=0.5)])
    c = _gpu_campaign(spec)
    _steps(c, _Bowl(), 3)
    assert all(float(r.candidate.sum()) <= 0.5 + 1e-6 for r in c.history.records)
    bax = _gpu_campaign(spec, BAXPolicy(TopK(k=3, output="f", largest=False), "MeanBAX",
                                        n_candidates=256))
    _steps(bax, _Bowl(), 3)


@pytest.mark.cuda
@pytest.mark.parametrize("policy", ["SimpleBO", "BAX"])
def test_travel_limits_on_the_gpu(policy):
    torch.manual_seed(16)
    spec = _bowl_spec(domain=[
        ContinuousDomain(name="a", lower=-2.0, upper=2.0, max_step=0.4),
        ContinuousDomain(name="b", lower=-2.0, upper=2.0, max_step=0.4),
    ])
    p = SimpleBO() if policy == "SimpleBO" else BAXPolicy(
        TopK(k=3, output="f", largest=False), "MeanBAX", n_candidates=1024)
    c = _gpu_campaign(spec, p, initial_designs=[{"a": -1.8, "b": 1.8}])
    _steps(c, _Bowl(), 5)
    _steps(c, _Bowl(), 1, n=2)
    designs = [r.candidate for r in c.history.records]
    for x, y in itertools.pairwise(designs):
        assert not spec.out_of_reach(x, y)


# -- output constraints and bad regions --------------------------------------------


@pytest.mark.cuda
def test_output_constraints_on_the_gpu():
    torch.manual_seed(17)
    spec = _bowl_spec(outputs=[OutputSpec(name="f", direction="MINIMIZE"),
                               OutputSpec(name="g", direction="OBSERVE", max_val=0.0)])
    c = _gpu_campaign(spec)
    _steps(c, _Bowl(lambda a, b: [a]), 6)  # g = a <= 0: best feasible at (0, -0.5)
    result = c.result()
    assert result.p_feasible and result.feasible[0]
    assert float(result.best.candidate[0]) <= 0.1
    pareto = _gpu_campaign(_bowl_spec(outputs=[
        OutputSpec(name="f", direction="MINIMIZE"), OutputSpec(name="h", direction="MINIMIZE"),
        OutputSpec(name="g", direction="OBSERVE", max_val=0.0),
    ]))
    _steps(pareto, _Bowl(lambda a, b: [(a + 0.5) ** 2, a]), 3)
    bax = _gpu_campaign(spec, BAXPolicy(TopK(k=3, output="f", largest=False), "InfoBAX",
                                        n_candidates=256, n_posterior_samples=6))
    _steps(bax, _Bowl(lambda a, b: [a]), 3)


@pytest.mark.cuda
@pytest.mark.parametrize("policy", ["SimpleBO", "UncertaintySampling", "MeanBAX"])
def test_bad_regions_on_the_gpu(policy):
    torch.manual_seed(18)
    p = {"SimpleBO": SimpleBO(), "UncertaintySampling": UncertaintySampling(),
         "MeanBAX": BAXPolicy(TopK(k=3, output="f", largest=False), "MeanBAX", n_candidates=256)}[policy]
    seeds = torch.tensor([[-1.5, 0.0], [0.0, 1.0], [0.5, -0.5], [1.2, -1.0], [1.8, 1.5]],
                         dtype=torch.double)
    c = _gpu_campaign(_bowl_spec(), p, initial_designs=seeds,
                      supervisor=per_shot(lambda o: o[0] > 3.0, bad_region="dark"))
    _steps(c, _Bowl(), 7)
    usability = c.current_usability()
    assert usability is not None and _on_gpu(usability.model)
    p_cpu = usability(torch.zeros(3, 2, dtype=torch.double))
    assert p_cpu.device.type == "cpu"  # asked on the CPU, answered on the CPU
    result = c.result()
    assert result.p_usable and result.p_usable[0] >= 0.5
    assert "gave no usable result" in str(c.check())


# -- reading the model from outside --------------------------------------------------


@pytest.mark.cuda
def test_analysis_plots_and_figures_read_a_gpu_model():
    from abbot import analysis

    torch.manual_seed(19)
    policy = BAXPolicy(TopK(k=5, output="f", largest=False), "MeanBAX", n_candidates=256)
    c = _gpu_campaign(_bax_spec(), policy)
    _steps(c, _Bowl(), 7)
    c.suggest(1)
    model = c.current_model()
    assert _on_gpu(model)
    s = analysis.slice_1d(c.history, c.spec, model, var=1, n=20)
    assert len(s["mean"]) == 20 and all(math.isfinite(v) for v in s["std"])
    assert len(analysis.slice_2d(c.history, c.spec, model, n=8)["z"]) == 8
    assert set(analysis.parity(c.history, c.spec, model)) == {"f"}
    region = analysis.bax_target_region(policy, model, c.spec)
    assert region["grid_size"] == len(policy.candidate_grid(c.spec))
    report = c.check()
    assert report.fits and report.fits[0].z

    bo = _gpu_campaign(_bowl_spec())
    _steps(bo, _Bowl(), 3)
    assert bo.recommend() and isinstance(bo.result().best.candidate, torch.Tensor)

    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from abbot import plots

    for fig in (plots.model(c), plots.model(c, variables=["a"]), plots.progress(c),
                plots.check(c), plots.model(bo)):
        matplotlib.pyplot.close(fig)

    from abbot.ui.figures import VizSelections, model_fig, parity_fig, progress_fig

    pending = [item.x for item in c._inflight.values()]
    assert model_fig(c.history, c.spec, model, VizSelections(), policy, pending)["data"]
    assert parity_fig(c.history, c.spec, model, VizSelections())
    assert progress_fig(c.history, c.spec, policy, model)


# -- resume -----------------------------------------------------------------------


@pytest.mark.cuda
def test_a_gpu_store_resumes_on_the_cpu_and_back(tmp_path):
    torch.manual_seed(20)
    spec = _bowl_spec()
    store = DataStore(tmp_path / "run")
    gpu = _gpu_campaign(spec, store=store)
    _steps(gpu, _Bowl(), 7)
    gpu.suggest(1)  # left in flight when the GPU job ends
    n = len(gpu.history)

    cpu = Campaign(spec, store=DataStore(tmp_path / "run"), on_model_error="raise")
    assert len(cpu.history) == n and len(cpu.outstanding()) == 1
    cpu.step(_Bowl())  # runs the outstanding design
    cpu.step(_Bowl())  # and proposes on the CPU
    assert len(cpu.history) == n + 2 and model_device(cpu.current_model()).type == "cpu"
    _assert_on_cpu(cpu)

    again = _gpu_campaign(spec, store=DataStore(tmp_path / "run"))
    again.step(_Bowl())
    assert len(again.history) == n + 3
    _assert_on_cpu(again)


@pytest.mark.cuda
def test_xopt_generators_stay_on_the_cpu():
    """``GestGeneratorPolicy`` brings its own model (``needs_model=False``): on a GPU
    campaign it still proposes from CPU tensors, while abbot's model (for the
    recommendation) is fitted on the GPU."""
    pytest.importorskip("xopt")
    from xopt.generators.sequential.neldermead import NelderMeadGenerator

    from abbot.interop.xopt import GestGeneratorPolicy

    torch.manual_seed(22)
    c = _gpu_campaign(_bowl_spec(), GestGeneratorPolicy(NelderMeadGenerator))
    _steps(c, _Bowl(), 12)
    assert _on_gpu(c.current_model()) and c.result().best is not None


@pytest.mark.cuda
def test_a_custom_model_left_on_the_cpu_still_works():
    """A constructor that ignores the device: the policy follows the model, so the
    campaign runs on the CPU rather than failing."""
    from abbot import SingleTaskGPConstructor

    class OnCPU:
        provides_uq = "calibrated_variance"

        def build(self, X, Y, Yvar=None):
            return SingleTaskGPConstructor().build(X.cpu(), Y.cpu())

    torch.manual_seed(21)
    c = _gpu_campaign(_bowl_spec(), model_constructor=OnCPU())
    _steps(c, _Bowl(), 3)
    assert model_device(c.current_model()).type == "cpu"


# -- the UI and the agent -----------------------------------------------------------


@pytest.mark.cuda
def test_the_agent_and_the_ui_fit_on_the_gpu(tmp_path):
    from abbot.agent import Workspace
    from abbot.ui.state import UIState

    spec = {"domain": [{"kind": "continuous", "name": "x", "lower": 0, "upper": 1}],
            "outputs": [{"name": "y", "direction": "MAXIMIZE"}]}
    ws = Workspace(tmp_path, device="cuda")
    ws.create("opt", spec)
    for _ in range(6):
        (s,) = ws.suggest("opt")
        ws.report_result("opt", s["id"], {"y": -(s["design"]["x"] - 0.3) ** 2})
    c = ws.open("opt")
    assert c.device.type == "cuda" and _on_gpu(c.current_model())
    _assert_on_cpu(c)

    state = UIState()
    state.device = "cuda"
    state.mock = True  # the built-in simulation, in memory
    state.start_campaign()
    campaign = state.controller.campaign
    assert campaign.device.type == "cuda"


@pytest.mark.cuda
@pytest.mark.parametrize("kind", ["ensemble", "dropout", "deep kernel"])
def test_network_models_on_the_gpu(kind):
    from abbot.nn import (
        DeepKernelConstructor,
        MCDropoutConstructor,
        NetEnsembleConstructor,
    )

    def make_net(d, m):
        return torch.nn.Sequential(
            torch.nn.Linear(d, 32), torch.nn.ReLU(), torch.nn.Dropout(0.1), torch.nn.Linear(32, m)
        )

    spec = _bax_spec()
    bounds = spec.bounds_tensor()
    ctor = {
        "ensemble": lambda: NetEnsembleConstructor(make_net, n_members=3, epochs=100, bounds=bounds),
        "dropout": lambda: MCDropoutConstructor(make_net, n_samples=4, epochs=100, bounds=bounds),
        "deep kernel": lambda: DeepKernelConstructor(make_net, epochs=50, bounds=bounds),
    }[kind]()
    torch.manual_seed(3)
    policy = BAXPolicy(TopK(k=3, output="f", largest=False), "InfoBAX",
                       n_candidates=128, n_posterior_samples=4)
    c = _gpu_campaign(spec, policy, ctor)
    _steps(c, _Bowl(), 3)
    _steps(c, _Bowl(), 1, n=2)
    assert _on_gpu(c.current_model())
