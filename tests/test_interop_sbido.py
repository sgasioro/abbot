"""Integration with REAL sbido code (skipped if sbido isn't importable).

Exercises the actual sbido ``LinearGaussianExperiment`` + ``AnalyticalGaussianPosterior``
driving abbot's ``Campaign`` through the composite-SBI seam, plus the spec adapter and
the protocol-conformance checks. This is the drift sentinel: if sbido changes a method
signature, this fails at the boundary, not deep in a run.

Add sbido to the path to run it, e.g. PYTHONPATH=~/sbido/src pytest tests/test_interop_sbido.py
(the full mat_finder/SWAXS benchmark reproduction, which needs the `sbi` package + a
trained posterior.pkl, is a separate follow-on).
"""

import pytest
import torch

sbido_lg = pytest.importorskip("sbido.testing.linear_gaussian")

from abbot.experiment import Experiment, PosteriorSampler
from abbot.interop.sbido import conforms, spec_from_sbido
from abbot.model import SingleTaskGPConstructor
from abbot.policy import SimpleBO
from abbot.reduce import BoxMembershipReducer
from abbot.sbi import sbi_evaluate
from abbot.session import Campaign

pytestmark = pytest.mark.slow  # real-sbido integration + GP fits


def test_sbido_classes_conform_to_abbot_protocols():
    exp = sbido_lg.LinearGaussianExperiment(sigma_obs=0.1)
    post = sbido_lg.AnalyticalGaussianPosterior(sigma_prior=5.0, sigma_obs=0.1)
    assert conforms(exp, Experiment), "sbido Experiment must satisfy abbot.Experiment"
    assert conforms(post, PosteriorSampler), "sbido posterior must satisfy abbot.PosteriorSampler"


def test_spec_from_sbido_bridges_bounds_and_boxes():
    from sbido.interfaces import ProblemSpec as SbidoSpec

    sb = SbidoSpec(
        xi_bounds=torch.tensor([[0.0, 0.0], [1.0, 2.0]]),
        boxes=[(torch.tensor([0.2, 0.2]), torch.tensor([0.8, 1.5]))],
    )
    spec = spec_from_sbido(sb)
    assert spec.variable_names == ["x0", "x1"]
    assert spec.n_objectives == 1
    b = spec.bounds_tensor()
    assert torch.allclose(b[1], torch.tensor([1.0, 2.0], dtype=torch.double))
    # float32 -> list introduces tiny precision noise; compare approximately.
    assert torch.allclose(
        torch.tensor(spec.boxes[0].lower), torch.tensor([0.2, 0.2]), atol=1e-6
    )


def test_real_sbido_sbi_loop_on_abbot_core():
    """A full composite-SBI campaign on genuine sbido components + abbot engine."""
    torch.manual_seed(0)
    from abbot.problem import Box, ContinuousDomain, ObjectiveSpec, ProblemSpec

    # 1-D design/theta: LinearGaussianExperiment maps theta = xi (scale 1). Target
    # band [0.4, 0.6] on theta -> best design ~0.5.
    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x0", lower=0.0, upper=1.0)],
        objectives=[ObjectiveSpec(name="F", direction="MAXIMIZE")],
        boxes=[Box(lower=[0.4], upper=[0.6])],
    )
    exp = sbido_lg.LinearGaussianExperiment(sigma_obs=0.05, theta_scale=1.0)
    post = sbido_lg.AnalyticalGaussianPosterior(sigma_prior=5.0, sigma_obs=0.05)
    ev = sbi_evaluate(post, BoxMembershipReducer("indicator", beta=50.0), spec, n_samples=512)

    c = Campaign(spec, SimpleBO(), SingleTaskGPConstructor(bounds=spec.bounds_tensor()))
    c.seed(torch.tensor([[0.05], [0.5], [0.95]], dtype=torch.double))
    for _ in range(12):
        c.step(exp, ev)

    assert all(r.aux is not None for r in c.history.records)  # theta cached
    best_x = float(c.history.records[c.recommend()[0]].candidate)
    assert abs(best_x - 0.5) < 0.2


def test_bax_on_sbido_analytic_benchmark_concentrates_on_target():
    """Behavioral parity: BAXPolicy + Wishlist over sbido's analytic benchmark map
    (exact theta_true, no learned posterior) concentrates sampling in the oracle
    target region — the multibax-replacement acceptance check.
    """
    torch.manual_seed(0)
    bm = pytest.importorskip("sbido.testing.benchmark_maps")

    from abbot.interop.sbido import bax_policy_for
    from abbot.model import ModelListConstructor
    from abbot.problem import Box, ContinuousDomain, ObjectiveSpec, ProblemSpec
    from abbot.session import Campaign

    level = "L0"  # monotone, most-feasible level -> a learnable target in few shots
    # Model two tracked theta dims; target band = their oracle "diff" sub-target.
    dims = ["r_diff", "sigma_diff"]
    from sbido.mat_finder import PARAM_NAMES

    dim_idx = [PARAM_NAMES.index(n) for n in dims]
    band = {"r_diff": (20.0, 30.0), "sigma_diff": (-1e9, 0.1)}

    spec = ProblemSpec(
        domain=[ContinuousDomain(name=n, lower=float(lo), upper=float(hi))
                for n, lo, hi in zip(bm.XI_NAMES, bm.XI_BOUNDS[0], bm.XI_BOUNDS[1])],
        objectives=[ObjectiveSpec(name=d, direction="MAXIMIZE") for d in dims],
        boxes=[Box(lower=[band[d][0] for d in dims], upper=[band[d][1] for d in dims])],
    )

    class _AnalyticExperiment:
        def run(self, candidate):
            theta = bm.theta_true(candidate.reshape(1, -1), level=level)[0]
            return theta[dim_idx]  # the two tracked dims, exact

    policy = bax_policy_for(spec, strategy="MeanBAX", scalar=False, n_candidates=400)
    c = Campaign(spec, policy, ModelListConstructor(bounds=spec.bounds_tensor()))
    # Seed from the policy's own grid so fits and proposals share the candidate set.
    grid = policy.candidate_grid(spec)
    c.seed(grid[torch.randperm(grid.shape[0])[:6]])
    exp = _AnalyticExperiment()
    for _ in range(20):
        c.step(exp, lambda obs: obs)

    # Fraction of post-seed shots whose true theta lands in the target band should
    # exceed the base rate of the band over the design space (concentration).
    def in_band(cand):
        th = bm.theta_true(cand.reshape(1, -1), level=level)[0][dim_idx]
        return bool((th[0] >= 20.0) and (th[0] <= 30.0) and (th[1] <= 0.1))

    shots = [r.candidate for r in c.history.records[6:]]  # exclude seeds
    hit_rate = sum(in_band(x) for x in shots) / len(shots)
    base_rate = sum(in_band(grid[i]) for i in range(grid.shape[0])) / grid.shape[0]
    assert hit_rate > base_rate, f"BAX hit rate {hit_rate:.2f} <= base rate {base_rate:.2f}"
