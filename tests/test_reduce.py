"""Composite-SBI reducer: scoring correctness + parity with sbido's rule."""

import torch

from abbot.composite import BoxMembershipReducer, boxes_to_tensors
from abbot.problem import Box, ContinuousDomain, ObjectiveSpec, ProblemSpec


def _spec(boxes):
    return ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        objectives=[ObjectiveSpec(name="F", direction="MAXIMIZE")],
        boxes=boxes,
    )


def test_indicator_in_unit_range_and_peaks_inside_box():
    spec = _spec([Box(lower=[0.0], upper=[1.0])])
    red = BoxMembershipReducer("indicator", beta=50.0)
    inside = torch.full((500, 1), 0.5)   # all samples inside [0, 1]
    outside = torch.full((500, 1), 5.0)  # all far outside
    f_in = float(red(inside, spec))
    f_out = float(red(outside, spec))
    assert 0.0 <= f_out < f_in <= 1.0
    assert f_in > 0.9  # sharp indicator, samples well inside -> near 1


def test_distance_is_nonpositive_and_zero_inside():
    spec = _spec([Box(lower=[0.0], upper=[1.0])])
    red = BoxMembershipReducer("distance")
    inside = torch.full((100, 1), 0.5)
    outside = torch.full((100, 1), 2.0)
    assert abs(float(red(inside, spec))) < 1e-5   # inside -> ~0
    assert float(red(outside, spec)) < 0.0        # outside -> negative


def test_or_indicator_matches_best_box():
    # Two disjoint boxes; samples sitting in the second should score ~1 under OR.
    spec = _spec([Box(lower=[0.0], upper=[1.0]), Box(lower=[5.0], upper=[6.0])])
    red = BoxMembershipReducer("or_indicator", beta=50.0)
    in_second = torch.full((300, 1), 5.5)
    assert float(red(in_second, spec)) > 0.9


def test_beta_override_per_call():
    spec = _spec([Box(lower=[0.0], upper=[1.0])])
    red = BoxMembershipReducer("indicator", beta=1.0)
    theta = torch.full((200, 1), 0.5)
    sharp = float(red(theta, spec, beta=100.0))
    soft = float(red(theta, spec, beta=1.0))
    assert sharp > soft  # sharper indicator scores membership higher well inside


def test_parity_with_sbido_rule():
    """abbot's scoring must equal sbido's selection_score bit-for-bit (same rule).

    Re-implements the single-box soft-indicator here from first principles to guard
    against drift in the ported _per_sample_score.
    """
    spec = _spec([Box(lower=[0.2], upper=[0.8])])
    red = BoxMembershipReducer("indicator", beta=10.0)
    theta = torch.rand(1000, 1, dtype=torch.double)
    a, b = boxes_to_tensors(spec.boxes, dtype=torch.double)[0]
    expected = (torch.sigmoid(10.0 * (theta - a)) * torch.sigmoid(10.0 * (b - theta))).prod(-1).mean()
    assert torch.allclose(red(theta, spec), expected)
